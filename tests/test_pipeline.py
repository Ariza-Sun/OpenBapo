import tempfile
from pathlib import Path

import pytest
from fakes import FakeSeparator, FakeTracker

from opencapo.analysis.deps import missing_analysis_packages, require_analysis_packages
from opencapo.analysis.pipeline import StageTimings, analyze_file
from opencapo.errors import InputError
from opencapo.model import Analysis


def stem_dirs() -> list[Path]:
    return list(Path(tempfile.gettempdir()).glob("opencapo-stems-*"))


def test_analyze_file_fills_beats_tempo_meter_and_duration(tone: Path) -> None:
    result = analyze_file(tone, separator=FakeSeparator(), tracker=FakeTracker(bpm=120))
    analysis = result.analysis
    assert analysis.title == "Example Song"
    assert analysis.duration == pytest.approx(3.0)
    assert analysis.tempo is not None and analysis.tempo.bpm == pytest.approx(120.0)
    assert analysis.time_signature is not None
    assert (analysis.time_signature.numerator, analysis.time_signature.denominator) == (4, 4)
    assert len(analysis.beats) == 6
    assert [b.beat_in_bar for b in analysis.beats] == [1, 2, 3, 4, 1, 2]
    # The JSON leaving the pipeline is valid and round-trips.
    assert Analysis.from_json(analysis.to_json()) == analysis


def test_timings_cover_every_stage(tone: Path) -> None:
    result = analyze_file(tone, separator=FakeSeparator(), tracker=FakeTracker())
    timings = result.timings.as_dict()
    assert list(timings) == ["decode", "separate", "beats", "chords", "key", "total"]
    assert all(value >= 0.0 for value in timings.values())
    assert timings["total"] >= timings["decode"] + timings["separate"] + timings["beats"]


def test_stems_do_not_outlive_the_pipeline(tone: Path) -> None:
    before = stem_dirs()
    analyze_file(tone, separator=FakeSeparator(), tracker=FakeTracker())
    assert stem_dirs() == before


def test_stems_do_not_outlive_a_failure(tone: Path) -> None:
    before = stem_dirs()
    with pytest.raises(RuntimeError, match="crashed"):
        analyze_file(tone, separator=FakeSeparator(fail=True), tracker=FakeTracker())
    assert stem_dirs() == before


def test_stems_are_written_while_the_pipeline_runs(
    tone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []
    tracker = FakeTracker()
    original = tracker.track

    def track_and_look(audio):  # type: ignore[no-untyped-def]
        seen.extend(sorted(p.name for p in d.iterdir()) for d in stem_dirs())
        return original(audio)

    monkeypatch.setattr(tracker, "track", track_and_look)
    analyze_file(tone, separator=FakeSeparator(), tracker=tracker)
    assert ["bass.wav", "drums.wav", "other.wav", "vocals.wav"] in seen


def test_beat_input_can_be_the_rhythm_stems(tone: Path) -> None:
    tracker = FakeTracker()
    analyze_file(tone, separator=FakeSeparator(), tracker=tracker, beat_input="drums_bass")
    # The fake separator leaves drums and bass silent, so the tracker saw silence.
    assert not tracker.inputs[0].samples.any()
    tracker = FakeTracker()
    analyze_file(tone, separator=FakeSeparator(), tracker=tracker)
    assert tracker.inputs[0].samples.any()


def test_too_long_is_rejected_before_any_model_runs(
    tone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("opencapo.analysis.audio.MAX_DURATION_SECONDS", 2.0)
    separator = FakeSeparator()
    with pytest.raises(InputError, match="limit is 0:02"):
        analyze_file(tone, separator=separator, tracker=FakeTracker())
    assert separator.calls == 0


def test_stage_timer_accumulates() -> None:
    timings = StageTimings()
    with timings.stage("decode"):
        pass
    with timings.stage("decode"):
        pass
    with timings.stage("warmup"):
        pass
    assert timings.decode >= 0.0
    assert list(timings.as_dict()) == [
        "decode",
        "separate",
        "beats",
        "chords",
        "key",
        "warmup",
        "total",
    ]


def test_missing_packages_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("opencapo.analysis.deps.ANALYSIS_PACKAGES", ("numpy", "no_such_pkg"))
    assert missing_analysis_packages() == ["no_such_pkg"]
    with pytest.raises(InputError, match=r"no_such_pkg.*opencapo\[analysis\]"):
        require_analysis_packages()
