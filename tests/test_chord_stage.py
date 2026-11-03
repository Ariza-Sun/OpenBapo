from pathlib import Path

import numpy as np
import pytest
from fakes import FakeChordExpert, FakeSeparator, FakeTracker
from synth import SR, progression, to_stereo, voicing

from opencapo.analysis.chords.chroma import ChromaTemplateExpert
from opencapo.analysis.chords.stage import ChordStage
from opencapo.analysis.chords.vocab import label_index
from opencapo.analysis.pipeline import analyze_file
from opencapo.analysis.separate import Stems
from opencapo.cli import build_chord_stage, main
from opencapo.errors import InputError
from opencapo.model import Analysis, Beat, ChordQuality, Mode

C, F, G7, AM = (
    label_index(0, ChordQuality.MAJ),
    label_index(5, ChordQuality.MAJ),
    label_index(7, ChordQuality.DOM7),
    label_index(9, ChordQuality.MIN),
)


def beats(n: int, period: float = 0.5) -> list[Beat]:
    return [
        Beat(time=i * period, is_downbeat=i % 4 == 0, bar_index=i // 4, beat_in_bar=i % 4 + 1)
        for i in range(n)
    ]


def synthetic_stems(seconds: float) -> tuple[Stems, float]:
    """C F G7 Am as sines in `other`, root notes in `bass`, silence elsewhere."""
    bar = seconds / 4
    chords = progression(
        [
            (voicing([0, 4, 7]), bar),
            (voicing([5, 9, 0]), bar),
            (voicing([7, 11, 2, 5]), bar),
            (voicing([9, 0, 4]), bar),
        ],
        harmonics=2,
    )
    bass = progression(
        [
            (voicing([0], 2), bar),
            (voicing([5], 2), bar),
            (voicing([7], 2), bar),
            (voicing([9], 2), bar),
        ],
        harmonics=3,
    )
    other, bass_st = to_stereo(chords).samples, to_stereo(bass).samples
    silence = np.zeros_like(other)
    return Stems(sample_rate=SR, vocals=silence, drums=silence, bass=bass_st, other=other), seconds


def test_stage_fuses_experts_and_estimates_the_key() -> None:
    stems, seconds = synthetic_stems(8.0)
    mix = stems.mix("bass", "other")
    schedule = [(C, 2.0), (F, 2.0), (G7, 2.0), (AM, 2.0)]
    stage = ChordStage([FakeChordExpert(schedule, name="m1"), ChromaTemplateExpert()])
    result = stage.run(mix, stems, beats(16), seconds)
    assert [(c.root, c.quality) for c in result.chords] == [
        (0, ChordQuality.MAJ),
        (5, ChordQuality.MAJ),
        (7, ChordQuality.DOM7),
        (9, ChordQuality.MIN),
    ]
    assert [c.start for c in result.chords] == [0.0, 2.0, 4.0, 6.0]
    assert result.chords[-1].end == seconds
    assert all(c.bass is None for c in result.chords)  # the bass plays roots
    assert result.key is not None and (result.key.tonic, result.key.mode) == (0, Mode.MAJOR)
    assert set(result.beat_probs) == {"m1", "chroma"}
    assert {"chroma", "m1", "bass", "key", "fusion"} <= set(result.timings)
    Analysis(beats=tuple(beats(16)), chords=result.chords, key=result.key)


def test_stage_reports_an_inversion_from_the_bass_stem() -> None:
    stems, seconds = synthetic_stems(4.0)
    # Chord experts hear C major throughout; the bass plays C, F, G, A.
    stage = ChordStage([FakeChordExpert([(C, 4.0)], name="m1")])
    result = stage.run(stems.mix("bass", "other"), stems, beats(8), seconds)
    # One C major chord spanning everything; the bass vote is split, so the
    # winner decides: E/G would be chord tones, F and A are not.
    assert len(result.chords) == 1
    chord = result.chords[0]
    assert chord.root == 0 and chord.bass in (None, 7)


def test_stage_without_stems_reads_the_bass_from_the_mix() -> None:
    audio = progression([(voicing([0, 4, 7]), 2.0)])
    stage = ChordStage([FakeChordExpert([(C, 2.0)])])
    result = stage.run(audio, None, beats(4), 2.0)
    assert result.bass is not None and set(result.bass.pitch_class.tolist()) == {0}
    assert len(result.chords) == 1


def test_stage_needs_an_expert() -> None:
    with pytest.raises(ValueError):
        ChordStage([])


def test_pipeline_includes_chords_and_key(tone: Path) -> None:
    stage = ChordStage([FakeChordExpert([(C, 3.0)])])
    result = analyze_file(tone, separator=FakeSeparator(), tracker=FakeTracker(), chord_stage=stage)
    analysis = result.analysis
    assert analysis.chords and analysis.chords[0].root == 0
    assert analysis.key is not None
    timings = result.timings.as_dict()
    assert list(timings)[:5] == ["decode", "separate", "beats", "chords", "key"]
    assert "chords.fusion" in timings and timings["chords"] >= 0.0


def test_pipeline_without_chord_stage_leaves_them_empty(tone: Path) -> None:
    result = analyze_file(tone, separator=FakeSeparator(), tracker=FakeTracker())
    assert result.analysis.chords == () and result.analysis.key is None


def test_cli_chord_expert_selection(tone: Path, tmp_path: Path) -> None:
    stage = ChordStage([FakeChordExpert([(C, 3.0)])])
    out = tmp_path / "a.json"
    assert (
        main(
            ["analyze", str(tone), "--out", str(out)],
            separator=FakeSeparator(),
            tracker=FakeTracker(),
            chord_stage=stage,
        )
        == 0
    )
    assert Analysis.from_json(out.read_text(encoding="utf-8")).chords
    assert (
        main(
            ["analyze", str(tone), "--out", str(out), "--chord-experts", "none"],
            separator=FakeSeparator(),
            tracker=FakeTracker(),
        )
        == 0
    )
    assert Analysis.from_json(out.read_text(encoding="utf-8")).chords == ()
    with pytest.raises(InputError, match="unknown chord experts"):
        build_chord_stage("btc,nope", "cpu")
    assert build_chord_stage("none", "cpu") is None
    assert len(build_chord_stage("chroma", "cpu").experts) == 1  # type: ignore[union-attr]
