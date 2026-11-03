import numpy as np
import pytest
from synth import SR, progression, to_stereo, voicing

from opencapo.analysis.audio import Audio
from opencapo.analysis.chords.bassline import (
    BassLineDetector,
    bass_pitch_classes,
    bass_source,
    pool_per_beat,
)
from opencapo.analysis.chords.chroma import (
    Chromagram,
    ChromaTemplateExpert,
    chromagram,
    mean_chroma,
    templates,
)
from opencapo.analysis.chords.fusion import aggregate_to_beats
from opencapo.analysis.chords.key import (
    estimate_key,
    key_from_index,
    key_index,
    scores_from_chords,
    scores_from_chroma,
)
from opencapo.analysis.chords.vocab import NO_CHORD, index_label, label_index
from opencapo.analysis.separate import Stems
from opencapo.model import Chord, ChordQuality, Mode

C_MAJ = voicing([0, 4, 7])
F_MAJ = voicing([5, 9, 0])
G_DOM7 = voicing([7, 11, 2, 5])
A_MIN = voicing([9, 0, 4])


# --- E3: chroma templates -------------------------------------------------


def test_templates_are_unit_vectors_over_chord_tones() -> None:
    t = templates()
    assert t.shape == (NO_CHORD, 12)
    np.testing.assert_allclose(np.linalg.norm(t, axis=1), 1.0)
    c_maj = t[label_index(0, ChordQuality.MAJ)]
    assert set(np.flatnonzero(c_maj)) == {0, 4, 7}


def test_chroma_expert_recognises_sine_chords() -> None:
    audio = progression([(C_MAJ, 1.0), (F_MAJ, 1.0), (G_DOM7, 1.0), (A_MIN, 1.0)])
    post = ChromaTemplateExpert().estimate(audio, None)
    beats = aggregate_to_beats(post, [0.0, 1.0, 2.0, 3.0], end_time=4.0)
    labels = [index_label(int(i)) for i in beats.argmax(axis=1)]
    assert labels == [
        (0, ChordQuality.MAJ),
        (5, ChordQuality.MAJ),
        (7, ChordQuality.DOM7),
        (9, ChordQuality.MIN),
    ]
    np.testing.assert_allclose(post.probs.sum(axis=1), 1.0)


def test_chroma_expert_marks_silence_as_no_chord() -> None:
    audio = progression([(C_MAJ, 1.0), ([], 1.0)])
    post = ChromaTemplateExpert().estimate(audio, None)
    beats = aggregate_to_beats(post, [0.0, 1.0], end_time=2.0)
    assert beats[0].argmax() != NO_CHORD
    assert beats[1].argmax() == NO_CHORD


def test_chroma_expert_uses_stems_without_drums() -> None:
    audio = to_stereo(progression([(C_MAJ, 1.0)]))
    silence = np.zeros_like(audio.samples)
    stems = Stems(sample_rate=SR, vocals=silence, drums=audio.samples, bass=silence, other=silence)
    # Harmony lives in the drums stem only, which the expert ignores: silence -> N.
    post = ChromaTemplateExpert().estimate(audio, stems)
    assert post.probs.mean(axis=0).argmax() == NO_CHORD


def test_mean_chroma_is_energy_weighted_and_unit_norm() -> None:
    gram = Chromagram(
        times=np.array([0.0, 0.1]),
        chroma=np.array([[1.0] + [0.0] * 11, [0.0, 1.0] + [0.0] * 10]),
        energy=np.array([1.0, 0.0]),
    )
    np.testing.assert_allclose(mean_chroma(gram), [1.0] + [0.0] * 11)
    assert mean_chroma(chromagram(progression([(C_MAJ, 0.5)]))).shape == (12,)


# --- E4: bass line --------------------------------------------------------


def test_bass_pitch_classes_follow_a_sine_bass_line() -> None:
    bass = progression([(voicing([0], 2), 1.0), (voicing([5], 2), 1.0)], harmonics=3)
    times, pcs, strength = bass_pitch_classes(bass)
    first = pcs[(times >= 0.2) & (times < 0.8)]
    second = pcs[(times >= 1.2) & (times < 1.8)]
    assert (first == 0).mean() > 0.9
    assert (second == 5).mean() > 0.9
    assert strength[(times >= 0.2) & (times < 0.8)].min() > 0.3


def test_bass_detector_pools_per_beat() -> None:
    bass = progression([(voicing([2], 2), 1.0), (voicing([6], 2), 1.0)], harmonics=3)
    estimate = BassLineDetector().detect(bass, [0.0, 0.5, 1.0, 1.5], end_time=2.0)
    assert estimate.pitch_class.tolist() == [2, 2, 6, 6]
    assert estimate.strength.min() > 0.5


def test_bass_silence_gives_no_estimate() -> None:
    silent = Audio(samples=np.zeros((1, SR), np.float32), sample_rate=SR)
    estimate = BassLineDetector().detect(silent, [0.0, 0.5], end_time=1.0)
    assert estimate.pitch_class.tolist() == [-1, -1]
    assert (
        pool_per_beat(np.zeros(0), np.zeros(0, np.int64), np.zeros(0), [], 1.0).pitch_class.size
        == 0
    )


# --- E5: key --------------------------------------------------------------


def test_key_index_round_trip() -> None:
    for i in range(24):
        assert key_index(*key_from_index(i)) == i


def test_chroma_key_scores_find_c_major_from_audio() -> None:
    audio = progression([(C_MAJ, 1.0), (F_MAJ, 1.0), (G_DOM7, 1.0), (C_MAJ, 1.0)])
    scores = scores_from_chroma(mean_chroma(chromagram(audio)))
    assert int(np.argmax(scores)) == key_index(0, Mode.MAJOR)


def chord(start: float, end: float, root: int, quality: ChordQuality) -> Chord:
    return Chord(start=start, end=end, root=root, quality=quality, confidence=1.0)


def test_chord_key_scores_prefer_the_explaining_key() -> None:
    # I - IV - V7 - I in C.
    chords = [
        chord(0, 1, 0, ChordQuality.MAJ),
        chord(1, 2, 5, ChordQuality.MAJ),
        chord(2, 3, 7, ChordQuality.DOM7),
        chord(3, 4, 0, ChordQuality.MAJ),
    ]
    scores = scores_from_chords(chords)
    assert int(np.argmax(scores)) == key_index(0, Mode.MAJOR)
    # i - iv - V - i in A minor (E major uses the raised seventh).
    minor = [
        chord(0, 1, 9, ChordQuality.MIN),
        chord(1, 2, 2, ChordQuality.MIN),
        chord(2, 3, 4, ChordQuality.MAJ),
        chord(3, 4, 9, ChordQuality.MIN),
    ]
    assert int(np.argmax(scores_from_chords(minor))) == key_index(9, Mode.MINOR)
    assert scores_from_chords([]).sum() == 0.0


def test_estimate_key_combines_and_cross_checks() -> None:
    chroma_scores = np.zeros(24)
    chroma_scores[key_index(0, Mode.MAJOR)] = 1.0
    chord_scores = np.zeros(24)
    chord_scores[key_index(0, Mode.MAJOR)] = 1.0
    agreed = estimate_key(chroma_scores, chord_scores)
    assert agreed is not None and (agreed.tonic, agreed.mode) == (0, Mode.MAJOR)
    chord_scores_g = np.zeros(24)
    chord_scores_g[key_index(7, Mode.MAJOR)] = 1.0
    chord_scores_g[key_index(0, Mode.MAJOR)] = 0.9
    disagreed = estimate_key(chroma_scores, chord_scores_g)
    assert disagreed is not None
    assert disagreed.confidence < agreed.confidence
    assert estimate_key(np.zeros(24), np.zeros(24)) is None


def test_relative_major_minor_settled_by_tonic_chord_duration() -> None:
    tie = np.zeros(24)
    tie[key_index(0, Mode.MAJOR)] = 1.0
    tie[key_index(9, Mode.MINOR)] = 1.0
    mostly_am = [chord(0, 3, 9, ChordQuality.MIN), chord(3, 4, 0, ChordQuality.MAJ)]
    key = estimate_key(tie, np.zeros(24), mostly_am)
    assert key is not None and (key.tonic, key.mode) == (9, Mode.MINOR)
    mostly_c = [chord(0, 3, 0, ChordQuality.MAJ), chord(3, 4, 9, ChordQuality.MIN)]
    key = estimate_key(tie, np.zeros(24), mostly_c)
    assert key is not None and (key.tonic, key.mode) == (0, Mode.MAJOR)
    assert key.confidence == pytest.approx(min(1.0, 6.0 * 0.5))  # 3 s vs 1 s -> lopsidedness 0.5


def test_settled_relative_tie_never_yields_negative_confidence() -> None:
    # The minor key scores a hair higher, but the chords favour the major: the
    # tie-break puts the major first, so the raw score margin would be negative.
    scores = np.zeros(24)
    scores[key_index(9, Mode.MINOR)] = 1.0
    scores[key_index(0, Mode.MAJOR)] = 1.0 - 1e-4
    chords = [chord(0, 3, 0, ChordQuality.MAJ), chord(3, 4, 9, ChordQuality.MIN)]
    key = estimate_key(scores, np.zeros(24), chords)
    assert key is not None and (key.tonic, key.mode) == (0, Mode.MAJOR)
    assert 0.0 <= key.confidence <= 1.0
    key = estimate_key(scores, np.zeros(24), [])  # nothing to settle with: stays non-negative
    assert key is not None and key.confidence >= 0.0


def test_chord_scores_vote_window_by_window() -> None:
    # 40 s in C major then 20 s in F# major: pooled, the union of both scales
    # fits neither key well; windowed, C wins two thirds of the windows.
    c_major = [
        chord(t, t + 2, [0, 5, 7][i % 3], ChordQuality.MAJ) for i, t in enumerate(range(0, 40, 2))
    ]
    fs_major = [
        chord(t, t + 2, [6, 11, 1][i % 3], ChordQuality.MAJ) for i, t in enumerate(range(40, 60, 2))
    ]
    windowed = scores_from_chords([*c_major, *fs_major])
    assert int(np.argmax(windowed)) == key_index(0, Mode.MAJOR)
    assert windowed.sum() == pytest.approx(1.0)  # a mean of per-window distributions
    pooled = scores_from_chords([*c_major, *fs_major], window=0)
    assert pooled.sum() == pytest.approx(1.0)
    assert scores_from_chords(c_major, window=0).argmax() == key_index(0, Mode.MAJOR)


def test_opening_and_closing_chords_settle_a_relative_tie() -> None:
    # C, F, Am, G7 over and over: the same chords read as C major or A minor
    # and Am gets as much time as C, so only the phrase boundaries decide.
    def cycle(first: ChordQuality, first_root: int) -> list[Chord]:
        plan = [
            (first_root, first),
            (5, ChordQuality.MAJ),
            (9, ChordQuality.MIN),
            (7, ChordQuality.DOM7),
        ]
        return [chord(i, i + 1, r, q) for i, (r, q) in enumerate(plan * 4)]

    opens_on_c = cycle(ChordQuality.MAJ, 0)  # C ... G7: tonic opens, dominant closes
    key = estimate_key(np.zeros(24), scores_from_chords(opens_on_c), opens_on_c)
    assert key is not None and (key.tonic, key.mode) == (0, Mode.MAJOR)
    opens_on_am = [
        chord(0, 1, 9, ChordQuality.MIN),
        *opens_on_c[1:-1],
        chord(15, 16, 4, ChordQuality.MAJ),
    ]
    key = estimate_key(np.zeros(24), scores_from_chords(opens_on_am), opens_on_am)
    assert key is not None and (key.tonic, key.mode) == (9, Mode.MINOR)  # opens on Am, closes on E


@pytest.mark.parametrize("tonic", [0, 4, 7])
def test_end_to_end_key_from_synthetic_audio(tonic: int) -> None:
    i = voicing([(tonic + d) % 12 for d in (0, 4, 7)])
    iv = voicing([(tonic + d) % 12 for d in (5, 9, 0)])
    v = voicing([(tonic + d) % 12 for d in (7, 11, 2)])
    audio = progression([(i, 1.0), (iv, 1.0), (v, 1.0), (i, 1.0)])
    chroma_scores = scores_from_chroma(mean_chroma(chromagram(audio)))
    key = estimate_key(chroma_scores, np.zeros(24))
    assert key is not None and (key.tonic, key.mode) == (tonic, Mode.MAJOR)


def test_bass_source_falls_back_to_the_mix_when_the_stem_is_silent() -> None:
    mix = to_stereo(progression([(voicing([0, 4, 7]), 1.0)]))
    quiet = np.zeros_like(mix.samples)
    quiet[:, :100] = 0.001
    stems = Stems(sample_rate=SR, vocals=quiet, drums=quiet, bass=quiet, other=mix.samples)
    assert bass_source(mix, stems) is mix
    loud = Stems(sample_rate=SR, vocals=quiet, drums=quiet, bass=mix.samples, other=quiet)
    assert bass_source(mix, loud) is not mix  # the stem itself
    assert bass_source(mix, None) is mix
