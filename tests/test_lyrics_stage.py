import numpy as np
import pytest

from openbapo.analysis.lyrics.align import (
    CtcModel,
    TimedToken,
    ctc_targets,
    emissions,
    song_language,
)
from openbapo.analysis.lyrics.stage import fill_gaps, to_syllables
from openbapo.lyrics import parse_lyrics, tokenize


def timed(text: str, spans: list[tuple[float, float] | None], line: int = 0) -> list[TimedToken]:
    return [
        TimedToken(line, tok, *(s if s else (None, None)))
        for tok, s in zip(tokenize(text), spans, strict=True)
    ]


def test_fill_gaps_interpolates_between_neighbours() -> None:
    tokens = timed("我爱你们", [(1.0, 1.5), None, None, (3.0, 3.5)])
    assert fill_gaps(tokens, 10.0) == [(1.0, 1.5), (1.5, 2.25), (2.25, 3.0), (3.0, 3.5)]


def test_fill_gaps_at_the_edges() -> None:
    tokens = timed("我爱你", [None, (2.0, 2.5), None])
    spans = fill_gaps(tokens, 10.0)
    assert spans[0] == (0.0, 2.0)
    assert spans[2][0] == 2.5 and spans[2][1] > 2.5


def test_syllables_split_a_word_by_letters() -> None:
    syllables, warnings = to_syllables(timed("beautiful day", [(1.0, 1.9), (2.0, 2.4)]), 10.0)
    assert [s.text for s in syllables] == ["beau-", "ti-", "ful", "day"]
    assert syllables[0].start == 1.0 and syllables[2].end == 1.9
    assert syllables[1].start == pytest.approx(1.4)
    assert not warnings


def test_syllables_never_go_backwards() -> None:
    syllables, warnings = to_syllables(timed("我爱你", [(2.0, 2.5), (1.0, 1.2), None]), 10.0)
    starts = [s.start for s in syllables]
    assert starts == sorted(starts)
    assert warnings and "1 of 3" in warnings[0]


def test_song_language() -> None:
    assert song_language(parse_lyrics("我说 baby\n别走")) == "zh"
    assert song_language(parse_lyrics("hold me 我\nnow and forever")) == "en"


def letters_model() -> CtcModel:
    labels = {"-": 0, "a": 1, "b": 2}

    def forward(chunk: np.ndarray) -> np.ndarray:
        raise AssertionError("not used")

    return CtcModel(forward, labels, 0, 0.02, lambda t: list(t.spoken))


def test_ctc_targets_leave_unknown_tokens_empty() -> None:
    tokens = tokenize("ab ba abc")
    assert ctc_targets(letters_model(), tokens) == [[1, 2], [2, 1], []]


def test_emissions_retain_singing_missed_by_vad() -> None:
    model = letters_model()

    def forward(chunk: np.ndarray) -> np.ndarray:
        frames = len(chunk) // 320
        assert len(chunk) == 16000 * 4  # infer the whole audio, including VAD misses
        out = np.full((frames, 3), -10.0, dtype=np.float32)
        out[:, 0] = 0.0
        out[20:25, 0], out[20:25, 1] = -10.0, 0.0
        out[150:160, 0], out[150:160, 2] = -10.0, 0.0
        return out

    model.forward = forward
    log_probs = emissions(model, np.zeros(16000 * 4, dtype=np.float32), [(0.4, 0.5)])
    assert log_probs.shape[0] == 201
    assert (log_probs[150:160, 2] > log_probs[150:160, 0]).all()
    assert (log_probs[100:110, 0] > log_probs[100:110, 2]).all()
    assert np.exp(log_probs).sum(axis=1) == pytest.approx(np.ones(201))
    pytest.importorskip("torchaudio")
    from openbapo.analysis.lyrics.align import forced_align

    spans = forced_align(log_probs, [[1], [2]], model.blank, model.frame_s)
    assert spans[0][0] == pytest.approx(0.4)
    assert spans[1][0] == pytest.approx(3.0)


def test_activity_prior_is_finite_and_keeps_relative_acoustic_evidence() -> None:
    from openbapo.analysis.lyrics.align import activity_prior

    probs = np.log(np.array([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]], dtype=np.float32))
    adjusted = activity_prior(probs, 0, 0.02, [], 2.0)
    assert np.isfinite(adjusted).all()
    assert np.exp(adjusted).sum(axis=1) == pytest.approx(np.ones(2))
    assert adjusted[0, 1] > adjusted[0, 0]
    assert adjusted[1, 0] > adjusted[1, 1]
    assert activity_prior(probs, 0, 0.02, None, 2.0) is probs
    for invalid in (-1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            activity_prior(probs, 0, 0.02, [], invalid)


def test_energy_recovers_missed_phrases_but_needs_speech_evidence() -> None:
    from openbapo.analysis.lyrics.activity import extend_activity, frame_rms

    audio = np.zeros(16000 * 6, dtype=np.float32)
    audio[16000:32000] = 0.2
    audio[64000:80000] = 0.08  # sustained singing the speech detector missed
    energy = frame_rms(audio)
    params = dict(relative_db=-25.0, floor_db=-50.0, min_seed_s=1.0)
    assert extend_activity(energy, [], 6.0, **params) == []
    seeds = [(1.0, 2.0)]
    found = extend_activity(energy, seeds, 6.0, **params)
    assert any(start <= 4.0 and end >= 5.0 for start, end in found)
    assert not any(start <= 3.0 <= end for start, end in found)
    assert all(0 <= start < end <= 6 for start, end in found)
    assert extend_activity(energy * 1e-4, seeds, 6.0, **params) == seeds
    assert frame_rms(np.array([], dtype=np.float32)).size == 0


def test_alignment_detector_recovers_singing_but_default_recognition_stays_strict(monkeypatch):
    from openbapo.analysis.lyrics.activity import ActivityParams, SileroActivity

    audio = np.zeros(16000 * 10, dtype=np.float32)
    audio[16000:32000] = 0.2
    audio[64000:144000] = 0.08
    detector = SileroActivity(ActivityParams(energy_relative_db=-15.0))
    monkeypatch.setattr(detector, "_speech_regions", lambda _: [(1.0, 2.0)])
    assert not detector.has_vocals([(1.0, 2.0)])
    assert detector.has_vocals(detector.regions(audio))
    monkeypatch.setattr(detector, "_speech_regions", lambda _: [])
    assert detector.regions(audio) == []  # energy alone cannot establish a singer
    baseline = SileroActivity()
    monkeypatch.setattr(baseline, "_speech_regions", lambda _: [(1.0, 2.0)])
    assert baseline.regions(audio) == [(1.0, 2.0)]


def test_lyrics_stage_obeys_no_voice_gate_and_supplied_text_skips_recognition() -> None:
    from openbapo.analysis.lyrics.stage import NO_VOCALS, LyricsStage

    class Activity:
        def __init__(self):
            self.found = []

        def regions(self, audio):
            return self.found

        def has_vocals(self, regions):
            return bool(regions)

    class Aligner:
        def align(self, audio, lines, regions):
            assert len(lines) == 1 and lines[0].text == "我爱你"
            assert regions == [(1.0, 2.0)]
            return timed("我爱你", [(1.0, 1.2), (1.2, 1.5), (1.5, 2.0)])

    activity = Activity()
    stage = LyricsStage(activity, None, Aligner())
    audio = np.zeros(16000 * 3, dtype=np.float32)
    assert stage.run(audio, "我爱你").warnings == (NO_VOCALS,)
    activity.found = [(1.0, 2.0)]
    result = stage.run(audio, "我爱你")
    assert [s.text for s in result.syllables] == list("我爱你")
    assert result.recognition is None


@pytest.mark.parametrize("speech_found", [False, True])
def test_energy_recovery_is_used_only_with_supplied_lyrics(speech_found) -> None:
    from openbapo.analysis.lyrics.recognize import Recognition, RecognizedLine
    from openbapo.analysis.lyrics.stage import NO_VOCALS, LyricsStage

    strict = [(1.0, 2.0)] if speech_found else []
    expanded = [(1.0, 7.0)]
    aligned_regions = []
    recognized_regions = []

    class Activity:
        def __init__(self, found):
            self.found = found

        def regions(self, audio):
            return self.found

        def has_vocals(self, regions):
            return bool(regions)

    class Recognizer:
        def recognize(self, audio, regions):
            recognized_regions.append(regions)
            return Recognition("zh", (RecognizedLine(1.0, 2.0, "我", ()),))

    class Aligner:
        def align(self, audio, lines, regions):
            aligned_regions.append(regions)
            return timed("我", [(1.0, 2.0)])

    stage = LyricsStage(
        Activity(strict), Recognizer(), Aligner(), alignment_activity=Activity(expanded)
    )
    audio = np.zeros(16000 * 8, dtype=np.float32)
    supplied = stage.run(audio, "我")
    assert supplied.syllables and supplied.recognition is None
    assert aligned_regions == [expanded] and not recognized_regions
    automatic = stage.run(audio)
    if speech_found:
        assert automatic.recognition is not None
        assert recognized_regions == [strict]
        assert aligned_regions == [expanded, strict]
    else:
        assert automatic.warnings == (NO_VOCALS,)
        assert not recognized_regions and aligned_regions == [expanded]


def test_forced_align_places_tokens_where_the_labels_are() -> None:
    pytest.importorskip("torchaudio")
    from openbapo.analysis.lyrics.align import forced_align

    log_probs = np.full((100, 3), -10.0, dtype=np.float32)
    log_probs[:, 0] = 0.0
    log_probs[20:25, 0], log_probs[20:25, 1] = -10.0, 0.0  # "a" at 0.40-0.50 s
    log_probs[60:70, 0], log_probs[60:70, 2] = -10.0, 0.0  # "b" at 1.20-1.40 s
    out = forced_align(log_probs, [[1], [], [2]], blank=0, frame_s=0.02)
    assert out[1] is None
    assert (
        out[0] is not None and out[0][0] == pytest.approx(0.40) and out[0][1] == pytest.approx(0.50)
    )
    assert out[2] is not None and out[2][0] == pytest.approx(1.20)


def test_ctc_targets_put_the_separator_between_words() -> None:
    model = letters_model()
    model.labels = {**model.labels, "|": 3}
    model.separator = "|"
    assert ctc_targets(model, tokenize("ab ba abc ab")) == [[1, 2, 3], [2, 1, 3], [], [1, 2]]
