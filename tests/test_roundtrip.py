"""JSON round trips must be lossless: from_json(to_json(x)) == x."""

import json
from pathlib import Path
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from openbapo.model import (
    Analysis,
    Arrangement,
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

GOLDEN = Path(__file__).parent / "data" / "analysis_full.json"

seconds = st.floats(min_value=0.0, max_value=1e5, allow_nan=False, allow_infinity=False)
confidence = st.floats(min_value=0.0, max_value=1.0)
pitch_class = st.integers(min_value=0, max_value=11)
optional_text = st.none() | st.text()


def test_golden_file_round_trips() -> None:
    raw = GOLDEN.read_text(encoding="utf-8")
    analysis = Analysis.from_json(raw)
    # Every field is present in the file, so dumping must reproduce it exactly.
    assert json.loads(analysis.to_json()) == json.loads(raw)
    assert Analysis.from_json(analysis.to_json()) == analysis


def test_golden_file_covers_every_object() -> None:
    analysis = Analysis.from_json(GOLDEN.read_text(encoding="utf-8"))
    assert analysis.tempo and analysis.time_signature and analysis.key and analysis.duration
    assert analysis.beats and analysis.chords and analysis.notes and analysis.lyrics
    assert any(c.bass is not None for c in analysis.chords)
    assert any(s.note_ref is None for s in analysis.lyrics)


@st.composite
def beats(draw: st.DrawFn) -> tuple[Beat, ...]:
    times = sorted(draw(st.lists(seconds, unique=True, max_size=20)))
    per_bar = draw(st.integers(min_value=1, max_value=7))
    return tuple(
        Beat(
            time=t,
            is_downbeat=i % per_bar == 0,
            bar_index=i // per_bar,
            beat_in_bar=i % per_bar + 1,
        )
        for i, t in enumerate(times)
    )


def spans(values: list[Any]) -> list[tuple[Any, Any]]:
    """Pair up sorted distinct values into non-overlapping (start, end) spans."""
    values = sorted(values)
    return list(zip(values[::2], values[1::2], strict=False))


@st.composite
def chords(draw: st.DrawFn) -> tuple[Chord, ...]:
    result = []
    for start, end in spans(draw(st.lists(seconds, unique=True, max_size=20))):
        root = draw(pitch_class)
        bass = draw(st.none() | pitch_class.filter(lambda b, r=root: b != r))
        result.append(
            Chord(
                start=start,
                end=end,
                root=root,
                quality=draw(st.sampled_from(ChordQuality)),
                bass=bass,
                confidence=draw(confidence),
            )
        )
    return tuple(result)


@st.composite
def notes(draw: st.DrawFn) -> tuple[Note, ...]:
    times = spans(draw(st.lists(seconds, unique=True, max_size=20)))
    ticks = spans(draw(st.lists(st.integers(0, 10**6), unique=True, min_size=2 * len(times))))
    return tuple(
        Note(
            start=start,
            end=end,
            pitch=draw(st.integers(min_value=0, max_value=127)),
            start_tick=tick,
            duration_ticks=tick_end - tick,
        )
        for (start, end), (tick, tick_end) in zip(times, ticks, strict=False)
    )


@st.composite
def lyrics(draw: st.DrawFn, n_notes: int) -> tuple[LyricSyllable, ...]:
    starts = sorted(draw(st.lists(seconds, max_size=20)))
    line_steps = draw(st.lists(st.integers(0, 2), min_size=len(starts), max_size=len(starts)))
    note_ref = st.none() | st.integers(0, n_notes - 1) if n_notes else st.none()
    result, line = [], 0
    for start, step in zip(starts, line_steps, strict=True):
        line += step
        length = draw(st.floats(min_value=0.0, max_value=10.0))
        result.append(
            LyricSyllable(
                start=start,
                end=start + length,
                text=draw(st.text(min_size=1)),
                line_index=line,
                note_ref=draw(note_ref),
            )
        )
    return tuple(result)


@st.composite
def analyses(draw: st.DrawFn) -> Analysis:
    melody = draw(notes())
    return Analysis(
        title=draw(optional_text),
        artist=draw(optional_text),
        duration=draw(st.none() | seconds.filter(lambda d: d > 0)),
        beats=draw(beats()),
        tempo=draw(st.none() | st.builds(Tempo, bpm=st.floats(min_value=1.0, max_value=400.0))),
        time_signature=draw(
            st.none()
            | st.builds(
                TimeSignature,
                numerator=st.integers(1, 12),
                denominator=st.sampled_from([2, 4, 8, 16]),
            )
        ),
        key=draw(
            st.none()
            | st.builds(Key, tonic=pitch_class, mode=st.sampled_from(Mode), confidence=confidence)
        ),
        chords=draw(chords()),
        notes=melody,
        lyrics=draw(lyrics(len(melody))),
    )


arrangements = st.builds(
    Arrangement,
    transpose=st.integers(-6, 6),
    fingering_key=st.none() | pitch_class,
    strum_pattern=st.text(min_size=1),
    simplify_chords=st.booleans(),
    lyrics_text=optional_text,
)


@given(analyses())
def test_analysis_round_trip(analysis: Analysis) -> None:
    assert Analysis.from_json(analysis.to_json()) == analysis


@given(arrangements)
def test_arrangement_round_trip(arrangement: Arrangement) -> None:
    assert Arrangement.from_json(arrangement.to_json()) == arrangement
