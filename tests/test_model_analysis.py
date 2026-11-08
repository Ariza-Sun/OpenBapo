import math
from typing import Any

import pytest
from pydantic import ValidationError

from openbapo.model import (
    Analysis,
    Beat,
    Chord,
    ChordQuality,
    Key,
    LyricSyllable,
    Mode,
    Note,
    Tempo,
    TimeSignature,
)


def beat(time: float, bar: int = 0, n: int = 1) -> Beat:
    return Beat(time=time, is_downbeat=n == 1, bar_index=bar, beat_in_bar=n)


def chord(start: float, end: float, **kw: Any) -> Chord:
    fields: dict[str, Any] = {"root": 7, "quality": ChordQuality.MAJ, "confidence": 0.9}
    return Chord(start=start, end=end, **(fields | kw))


def note(start: float, end: float, tick: int, ticks: int = 12) -> Note:
    return Note(start=start, end=end, pitch=67, start_tick=tick, duration_ticks=ticks)


def syllable(start: float, line: int = 0, ref: int | None = None) -> LyricSyllable:
    return LyricSyllable(start=start, end=start + 0.2, text="la", line_index=line, note_ref=ref)


class TestBeat:
    def test_valid(self) -> None:
        assert beat(0.5, bar=2, n=3).beat_in_bar == 3

    @pytest.mark.parametrize(
        "kw",
        [
            {"time": -0.1},
            {"bar_index": -1},
            {"beat_in_bar": 0},
            {"time": math.nan},
            {"time": math.inf},
        ],
    )
    def test_out_of_range(self, kw: dict[str, Any]) -> None:
        fields = {"time": 0.0, "is_downbeat": True, "bar_index": 0, "beat_in_bar": 1}
        with pytest.raises(ValidationError):
            Beat(**(fields | kw))

    @pytest.mark.parametrize(("downbeat", "n"), [(True, 2), (False, 1)])
    def test_downbeat_must_match_first_beat(self, downbeat: bool, n: int) -> None:
        with pytest.raises(ValidationError, match="is_downbeat"):
            Beat(time=0.0, is_downbeat=downbeat, bar_index=0, beat_in_bar=n)


class TestTempoAndKey:
    @pytest.mark.parametrize("bpm", [0.0, -90.0])
    def test_bpm_must_be_positive(self, bpm: float) -> None:
        with pytest.raises(ValidationError):
            Tempo(bpm=bpm)

    @pytest.mark.parametrize("kw", [{"tonic": 12}, {"tonic": -1}, {"confidence": 1.01}])
    def test_key_out_of_range(self, kw: dict[str, Any]) -> None:
        fields: dict[str, Any] = {"tonic": 0, "mode": Mode.MAJOR, "confidence": 0.5}
        with pytest.raises(ValidationError):
            Key(**(fields | kw))


class TestChord:
    def test_inversion(self) -> None:
        d_over_fsharp = chord(1.0, 2.0, root=2, bass=6)
        assert d_over_fsharp.bass == 6

    def test_has_fourteen_qualities(self) -> None:
        assert len(ChordQuality) == 14

    @pytest.mark.parametrize(("start", "end"), [(2.0, 2.0), (2.0, 1.0)])
    def test_start_before_end(self, start: float, end: float) -> None:
        with pytest.raises(ValidationError, match="before end"):
            chord(start, end)

    def test_bass_equal_to_root_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="root position"):
            chord(0.0, 1.0, root=2, bass=2)

    @pytest.mark.parametrize("confidence", [-0.1, 1.1, math.nan])
    def test_confidence_range(self, confidence: float) -> None:
        with pytest.raises(ValidationError):
            chord(0.0, 1.0, confidence=confidence)


class TestNote:
    def test_end_tick(self) -> None:
        assert note(0.0, 0.5, tick=6, ticks=18).end_tick == 24

    @pytest.mark.parametrize(
        "kw",
        [
            {"pitch": 128},
            {"pitch": -1},
            {"start_tick": -1},
            {"duration_ticks": 0},
            {"start": 1.0, "end": 1.0},
        ],
    )
    def test_invalid(self, kw: dict[str, Any]) -> None:
        fields = {"start": 0.0, "end": 0.5, "pitch": 60, "start_tick": 0, "duration_ticks": 12}
        with pytest.raises(ValidationError):
            Note(**(fields | kw))


class TestLyricSyllable:
    def test_zero_length_is_allowed(self) -> None:
        assert LyricSyllable(start=1.0, end=1.0, text="啊", line_index=0).note_ref is None

    @pytest.mark.parametrize(
        "kw", [{"text": ""}, {"line_index": -1}, {"start": 2.0}, {"note_ref": -1}]
    )
    def test_invalid(self, kw: dict[str, Any]) -> None:
        fields = {"start": 0.0, "end": 1.0, "text": "Des", "line_index": 0}
        with pytest.raises(ValidationError):
            LyricSyllable(**(fields | kw))


class TestAnalysis:
    def test_empty_is_valid(self) -> None:
        empty = Analysis()
        assert empty.schema_version == 2
        assert empty.beats == () and empty.tempo is None and empty.key is None

    def test_has_no_stem_field(self) -> None:
        assert not any("stem" in name for name in Analysis.model_fields)

    def test_unknown_field_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Analysis(stems=())  # type: ignore[call-arg]

    def test_is_immutable(self) -> None:
        with pytest.raises(ValidationError):
            Analysis().title = "x"  # type: ignore[misc]

    def test_no_silent_coercion(self) -> None:
        with pytest.raises(ValidationError):
            Analysis.from_json('{"schema_version": 1, "title": 5}')
        with pytest.raises(ValidationError):
            Tempo.from_json('{"bpm": "120"}')

    def test_beats_strictly_increasing(self) -> None:
        with pytest.raises(ValidationError, match="strictly increasing"):
            Analysis(beats=(beat(1.0), beat(1.0, bar=1)))

    def test_beat_bars_do_not_decrease(self) -> None:
        with pytest.raises(ValidationError, match="bar_index"):
            Analysis(beats=(beat(0.0, bar=1), beat(0.5, bar=0)))

    def test_adjacent_chords_are_allowed(self) -> None:
        assert len(Analysis(chords=(chord(0.0, 1.0), chord(1.0, 2.0))).chords) == 2

    def test_gaps_between_chords_are_allowed(self) -> None:
        assert len(Analysis(chords=(chord(0.0, 1.0), chord(3.0, 4.0))).chords) == 2

    @pytest.mark.parametrize(
        "chords",
        [(chord(0.0, 1.5), chord(1.0, 2.0)), (chord(2.0, 3.0), chord(0.0, 1.0))],
    )
    def test_chords_overlapping_or_unsorted(self, chords: tuple[Chord, ...]) -> None:
        with pytest.raises(ValidationError, match="overlap"):
            Analysis(chords=chords)

    @pytest.mark.parametrize(
        "notes",
        [
            (note(0.0, 0.6, tick=0), note(0.5, 1.0, tick=12)),  # overlap in time
            (note(0.0, 0.5, tick=0, ticks=18), note(0.5, 1.0, tick=12)),  # on the grid
        ],
    )
    def test_notes_are_monophonic(self, notes: tuple[Note, ...]) -> None:
        with pytest.raises(ValidationError, match="overlap"):
            Analysis(notes=notes)

    def test_lyrics_sorted(self) -> None:
        with pytest.raises(ValidationError, match="sorted"):
            Analysis(lyrics=(syllable(1.0), syllable(0.5)))

    def test_lyric_lines_do_not_decrease(self) -> None:
        with pytest.raises(ValidationError, match="line_index"):
            Analysis(lyrics=(syllable(0.0, line=1), syllable(1.0, line=0)))

    def test_note_ref_in_range(self) -> None:
        notes = (note(0.0, 0.5, tick=0),)
        assert Analysis(notes=notes, lyrics=(syllable(0.0, ref=0),)).lyrics[0].note_ref == 0
        with pytest.raises(ValidationError, match="out of range"):
            Analysis(notes=notes, lyrics=(syllable(0.0, ref=1),))

    @pytest.mark.parametrize("duration", [0.0, -1.0])
    def test_duration_positive(self, duration: float) -> None:
        with pytest.raises(ValidationError):
            Analysis(duration=duration)


class TestTimeSignature:
    @pytest.mark.parametrize(("num", "den"), [(4, 4), (3, 4), (6, 8), (1, 2), (7, 16)])
    def test_valid(self, num: int, den: int) -> None:
        assert TimeSignature(numerator=num, denominator=den).numerator == num

    @pytest.mark.parametrize(("num", "den"), [(0, 4), (4, 3), (4, 0), (4, 32)])
    def test_invalid(self, num: int, den: int) -> None:
        with pytest.raises(ValidationError):
            TimeSignature(numerator=num, denominator=den)

    def test_no_silent_coercion(self) -> None:
        with pytest.raises(ValidationError):
            TimeSignature.from_json('{"numerator": 4, "denominator": "4"}')


class TestSchemaVersion:
    @pytest.mark.parametrize("version", [1, 3])
    def test_other_versions_are_rejected(self, version: int) -> None:
        with pytest.raises(ValidationError):
            Analysis.from_json(f'{{"schema_version": {version}}}')

    def test_missing_version_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="schema_version is missing"):
            Analysis.from_json("{}")

    def test_version_is_written(self) -> None:
        assert '"schema_version": 2' in Analysis().to_json()
