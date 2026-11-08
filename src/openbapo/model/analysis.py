"""Analysis results: a faithful transcription of what is in the recording."""

from itertools import pairwise
from typing import Annotated, Any, Final, Literal, Self

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationInfo,
    model_validator,
)

from openbapo.model._base import Model
from openbapo.model.music import ChordQuality, Confidence, Mode, PitchClass, Seconds

SCHEMA_VERSION: Final = 2

# Quantization grid for notes: each beat of the beat grid is split into this many
# ticks. 12 divides evenly by 2, 3, 4 and 6 (sixteenths and triplets).
TICKS_PER_BEAT = 12


class Beat(Model):
    time: Seconds
    is_downbeat: StrictBool
    bar_index: Annotated[StrictInt, Field(ge=0)]
    beat_in_bar: Annotated[StrictInt, Field(ge=1)]  # 1-based, as counted aloud

    @model_validator(mode="after")
    def _downbeat_is_first_beat(self) -> Self:
        if self.is_downbeat != (self.beat_in_bar == 1):
            raise ValueError("is_downbeat must be true exactly when beat_in_bar == 1")
        return self


class Tempo(Model):
    bpm: Annotated[StrictFloat, Field(gt=0.0)]


class TimeSignature(Model):
    """Meter as written in the header: beats per bar over the beat unit.

    One Beat object is one `denominator` note, so ticks and bpm share that unit.
    """

    numerator: Annotated[StrictInt, Field(ge=1)]
    denominator: Literal[2, 4, 8, 16]


class Key(Model):
    tonic: PitchClass
    mode: Mode
    confidence: Confidence


class Chord(Model):
    start: Seconds
    end: Seconds
    root: PitchClass
    quality: ChordQuality
    bass: PitchClass | None = None  # None = root position; else the inversion bass
    confidence: Confidence

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.start >= self.end:
            raise ValueError("chord start must be before end")
        if self.bass == self.root:
            raise ValueError("bass equal to root is root position; use bass=None")
        return self


class Note(Model):
    """A sung note: observed timing and pitch, plus its position on the beat grid.

    Only the sounding duration is stored. Dots, ties and tuplets are notation and
    are derived at render time.
    """

    start: Seconds
    end: Seconds
    pitch: Annotated[StrictInt, Field(ge=0, le=127)]  # MIDI
    start_tick: Annotated[StrictInt, Field(ge=0)]  # counted from the first beat
    duration_ticks: Annotated[StrictInt, Field(ge=1)]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.start >= self.end:
            raise ValueError("note start must be before end")
        return self

    @property
    def end_tick(self) -> int:
        return self.start_tick + self.duration_ticks


class LyricSyllable(Model):
    start: Seconds
    end: Seconds
    text: Annotated[StrictStr, Field(min_length=1)]
    line_index: Annotated[StrictInt, Field(ge=0)]
    note_ref: Annotated[StrictInt, Field(ge=0)] | None = None  # index into Analysis.notes

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.start > self.end:
            raise ValueError("syllable start must not be after end")
        return self


class Analysis(Model):
    """Everything the pipeline extracts from one recording.

    An empty Analysis is valid: every list may be empty and tempo / key unset.
    Stems are deliberately absent; they never leave the pipeline.
    """

    schema_version: Literal[2] = SCHEMA_VERSION
    title: StrictStr | None = None
    artist: StrictStr | None = None
    duration: Annotated[StrictFloat, Field(gt=0.0)] | None = None
    beats: tuple[Beat, ...] = ()
    tempo: Tempo | None = None
    time_signature: TimeSignature | None = None
    key: Key | None = None
    chords: tuple[Chord, ...] = ()
    notes: tuple[Note, ...] = ()
    lyrics: tuple[LyricSyllable, ...] = ()

    @classmethod
    def from_json(cls, data: str | bytes) -> Self:
        return cls.model_validate_json(data, context={"from_json": True})

    @model_validator(mode="before")
    @classmethod
    def _require_version_in_json(cls, data: Any, info: ValidationInfo) -> Any:
        # The default version is for building new results in code; stored JSON
        # must state its version rather than have one guessed for it.
        from_json = bool(info.context and info.context.get("from_json"))
        if from_json and isinstance(data, dict) and "schema_version" not in data:
            raise ValueError("schema_version is missing")
        return data

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        for prev_beat, beat in pairwise(self.beats):
            if beat.time <= prev_beat.time:
                raise ValueError("beats must be strictly increasing in time")
            if beat.bar_index < prev_beat.bar_index:
                raise ValueError("beat bar_index must not decrease")
        for prev_chord, chord in pairwise(self.chords):
            if chord.start < prev_chord.end:
                raise ValueError("chords must be sorted and must not overlap")
        # The melody is monophonic: notes must not overlap in time or on the grid.
        for prev_note, note in pairwise(self.notes):
            if note.start < prev_note.end or note.start_tick < prev_note.end_tick:
                raise ValueError("notes must be sorted and must not overlap")
        for prev_syl, syl in pairwise(self.lyrics):
            if syl.start < prev_syl.start:
                raise ValueError("lyric syllables must be sorted by start")
            if syl.line_index < prev_syl.line_index:
                raise ValueError("lyric line_index must not decrease")
        for syllable in self.lyrics:
            if syllable.note_ref is not None and syllable.note_ref >= len(self.notes):
                raise ValueError(f"note_ref {syllable.note_ref} is out of range")
        return self
