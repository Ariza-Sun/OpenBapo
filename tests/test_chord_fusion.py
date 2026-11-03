import numpy as np
import pytest

from opencapo.analysis.chords.fusion import (
    BassEstimate,
    ChordPosteriors,
    FusionParams,
    aggregate_to_beats,
    agreement,
    beats_per_bar,
    decode,
    fill_gaps,
    fuse,
    key_prior,
    viterbi,
)
from opencapo.analysis.chords.vocab import N_LABELS, NO_CHORD, label_index
from opencapo.model import Analysis, Beat, ChordQuality, Key, Mode

C = label_index(0, ChordQuality.MAJ)
F = label_index(5, ChordQuality.MAJ)
G = label_index(7, ChordQuality.MAJ)
AM = label_index(9, ChordQuality.MIN)


def one_hot(label: int, p: float = 0.9) -> np.ndarray:
    row = np.full(N_LABELS, (1 - p) / (N_LABELS - 1))
    row[label] = p
    return row


def beat_grid(n: int, per_bar: int = 4, period: float = 0.5) -> list[Beat]:
    return [
        Beat(
            time=i * period,
            is_downbeat=i % per_bar == 0,
            bar_index=i // per_bar,
            beat_in_bar=i % per_bar + 1,
        )
        for i in range(n)
    ]


def probs_for(labels: list[int], p: float = 0.9) -> np.ndarray:
    return np.array([one_hot(label, p) for label in labels])


def test_posteriors_shape_is_checked() -> None:
    with pytest.raises(ValueError):
        ChordPosteriors(times=np.zeros(3), probs=np.zeros((3, 5)))
    with pytest.raises(ValueError):
        ChordPosteriors(times=np.zeros(2), probs=np.zeros((3, N_LABELS)))


def test_aggregate_averages_frames_inside_each_beat() -> None:
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.6, 0.7])
    probs = probs_for([C, C, C, G, F, F])
    post = ChordPosteriors(times=times, probs=probs)
    beats = aggregate_to_beats(post, [0.0, 0.5], end_time=1.0)
    assert beats.shape == (2, N_LABELS)
    np.testing.assert_allclose(beats[0], probs[:4].mean(axis=0))
    np.testing.assert_allclose(beats[1], probs[4:].mean(axis=0))


def test_aggregate_uses_nearest_frame_for_an_empty_beat() -> None:
    post = ChordPosteriors(times=np.array([0.0, 2.0]), probs=probs_for([C, G]))
    beats = aggregate_to_beats(post, [0.0, 0.5, 1.9], end_time=2.5)
    assert beats[1].argmax() == C  # 0.5 is closer to the frame at 0.0
    assert beats[2].argmax() == G


def test_aggregate_with_no_beats() -> None:
    post = ChordPosteriors(times=np.array([0.0]), probs=probs_for([C]))
    assert aggregate_to_beats(post, [], end_time=1.0).shape == (0, N_LABELS)


def test_fuse_is_a_weighted_product() -> None:
    a = probs_for([C, C])
    b = probs_for([G, C])
    fused = fuse({"a": a, "b": b}, FusionParams())
    np.testing.assert_allclose(fused.sum(axis=1), 1.0)
    assert fused[1].argmax() == C
    assert fused[0, C] == pytest.approx(fused[0, G])  # equal weights: a tie
    heavy_a = fuse({"a": a, "b": b}, FusionParams(weights={"a": 2.0}))
    assert heavy_a[0].argmax() == C
    with pytest.raises(ValueError):
        fuse({}, FusionParams())


def test_key_prior_marks_diatonic_roots() -> None:
    prior = key_prior(Key(tonic=0, mode=Mode.MAJOR, confidence=1.0), bonus=0.5)
    assert prior[C] == 0.5 and prior[G] == 0.5 and prior[AM] == 0.5
    assert prior[label_index(1, ChordQuality.MAJ)] == 0.0  # Db is not in C major
    assert prior[NO_CHORD] == 0.0
    minor = key_prior(Key(tonic=9, mode=Mode.MINOR, confidence=1.0), bonus=1.0)
    assert minor[label_index(8, ChordQuality.MAJ)] == 1.0  # raised 7th (G#) allowed in A minor
    assert key_prior(None, 0.5).sum() == 0.0


def test_viterbi_removes_a_mid_bar_blip_but_keeps_downbeat_changes() -> None:
    labels = [C, C, G, C, F, F, F, F]  # beat 2 is a blip; beat 4 is a real downbeat change
    log_scores = np.log(probs_for(labels, p=0.6))
    downbeats = [i % 4 == 0 for i in range(8)]
    path = viterbi(
        log_scores, downbeats, FusionParams(change_cost_downbeat=0.1, change_cost_beat=3.0)
    )
    assert path.tolist() == [C, C, C, C, F, F, F, F]
    cheap = viterbi(
        log_scores, downbeats, FusionParams(change_cost_downbeat=0.0, change_cost_beat=0.0)
    )
    assert cheap.tolist() == labels
    assert viterbi(np.zeros((0, N_LABELS)), [], FusionParams()).size == 0


def test_agreement_counts_model_experts_by_root() -> None:
    fused = np.array([C, G, NO_CHORD])
    m1 = probs_for([C, G, NO_CHORD])
    m2 = probs_for([label_index(0, ChordQuality.MAJ7), F, NO_CHORD])  # same root as C, wrong on G
    np.testing.assert_allclose(agreement(fused, [m1, m2]), [1.0, 0.5, 1.0])
    np.testing.assert_allclose(agreement(fused, []), [1.0, 1.0, 1.0])


def test_decode_builds_chords_on_beats_with_gaps_for_no_chord() -> None:
    beats = beat_grid(8)
    labels = [C, C, C, C, NO_CHORD, NO_CHORD, G, G]
    result = decode(
        beats,
        {"m": probs_for(labels), "r": probs_for(labels, p=0.5)},
        ["m"],
        duration=4.25,
        params=FusionParams(
            change_cost_beat=0.0, change_cost_downbeat=0.0, fill_gaps_under_bar=False
        ),
    )
    chords = result.chords
    assert [(c.start, c.end) for c in chords] == [(0.0, 2.0), (3.0, 4.25)]
    assert [(c.root, c.quality) for c in chords] == [(0, ChordQuality.MAJ), (7, ChordQuality.MAJ)]
    beat_times = {b.time for b in beats}
    assert all(c.start in beat_times for c in chords)
    assert all(c.end in beat_times or c.end == 4.25 for c in chords)
    assert all(0.0 < c.confidence <= 1.0 for c in chords)
    Analysis(beats=tuple(beats), chords=chords)  # satisfies the data model


def test_short_gaps_are_held_like_a_lead_sheet() -> None:
    beats = beat_grid(12)  # 3 bars of 4
    labels = [NO_CHORD, C, C, NO_CHORD, NO_CHORD, NO_CHORD, G, G, G, G, NO_CHORD, NO_CHORD]
    filled = fill_gaps(np.array(labels), beats_per_bar(beats))
    assert filled.tolist() == [NO_CHORD, C, C, C, C, C, G, G, G, G, NO_CHORD, NO_CHORD]
    long_gap = [C, *([NO_CHORD] * 4), G]  # a whole bar of silence stays N
    assert fill_gaps(np.array(long_gap), 4).tolist() == long_gap
    assert beats_per_bar(beat_grid(9, per_bar=3)) == 3 and beats_per_bar(beat_grid(2)) == 4
    result = decode(
        beats,
        {"m": probs_for(labels)},
        ["m"],
        duration=6.0,
        params=FusionParams(change_cost_beat=0.0, change_cost_downbeat=0.0),
    )
    assert [(c.start, c.end, c.root) for c in result.chords] == [(0.5, 3.0, 0), (3.0, 5.0, 7)]


def test_decode_confidence_is_posterior_times_agreement() -> None:
    beats = beat_grid(2)
    m1 = probs_for([C, C], p=0.8)
    m2 = probs_for([G, C], p=0.8)
    result = decode(beats, {"m1": m1, "m2": m2}, ["m1", "m2"], duration=1.0)
    (chord,) = result.chords
    # Beat 0 is a tie broken by Viterbi; the surviving label has 50% agreement there.
    assert chord.confidence == pytest.approx(float(result.confidence.min()))
    assert result.confidence[1] > result.confidence[0]


def test_decode_inversion_from_bass_line() -> None:
    beats = beat_grid(4)
    d_major = label_index(2, ChordQuality.MAJ)
    probs = probs_for([d_major] * 4)
    f_sharp = BassEstimate(pitch_class=np.array([6, 6, 6, 6]), strength=np.ones(4))
    (chord,) = decode(beats, {"m": probs}, ["m"], duration=2.0, bass=f_sharp).chords
    assert chord.bass == 6  # D/F#
    root_bass = BassEstimate(pitch_class=np.array([2, 2, 2, 2]), strength=np.ones(4))
    (chord,) = decode(beats, {"m": probs}, ["m"], duration=2.0, bass=root_bass).chords
    assert chord.bass is None
    weak = BassEstimate(pitch_class=np.array([6, 6, 6, 6]), strength=np.full(4, 0.1))
    (chord,) = decode(beats, {"m": probs}, ["m"], duration=2.0, bass=weak).chords
    assert chord.bass is None  # below min_bass_strength: ignored


def test_decode_foreign_bass_lowers_confidence() -> None:
    beats = beat_grid(4)
    probs = probs_for([C] * 4)
    plain = decode(beats, {"m": probs}, ["m"], duration=2.0).chords[0]
    foreign = BassEstimate(pitch_class=np.array([1, 1, 1, 1]), strength=np.ones(4))
    penalised = decode(beats, {"m": probs}, ["m"], duration=2.0, bass=foreign).chords[0]
    assert penalised.bass is None
    assert penalised.confidence == pytest.approx(plain.confidence * 0.7)


def test_decode_key_prior_breaks_ties_towards_the_key() -> None:
    beats = beat_grid(4)
    c_sharp = label_index(1, ChordQuality.MAJ)
    probs = probs_for([C] * 4)
    probs[:, c_sharp] = probs[:, C]  # exact tie between C and C#
    in_c = decode(
        beats, {"m": probs}, ["m"], duration=2.0, key=Key(tonic=0, mode=Mode.MAJOR, confidence=1.0)
    )
    assert in_c.chords[0].root == 0
    # C is the leading tone of C# major, so use F# major, where C# is in the scale and C is not.
    in_fs = decode(
        beats, {"m": probs}, ["m"], duration=2.0, key=Key(tonic=6, mode=Mode.MAJOR, confidence=1.0)
    )
    assert in_fs.chords[0].root == 1


def test_decode_with_no_beats() -> None:
    result = decode([], {"m": np.zeros((0, N_LABELS))}, ["m"], duration=1.0)
    assert result.chords == ()
