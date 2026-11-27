"""Shared, metered TAB notation for SVG and MusicXML.

Pattern beats use the time signature's denominator. Written durations use
quarter notes, independently of that unit. A sustained offbeat attack is
split at the next beat group and tied, never turned into a second attack.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from openbapo.arrange import Bar, Shape, StrumEvent, StrumPattern

Meter = tuple[int, int]
DIVISIONS = 96  # MusicXML divisions per quarter note, not per denominator beat
NOTE_TYPES = ("whole", "half", "quarter", "eighth", "16th", "32nd", "64th")


@dataclass(frozen=True)
class NoteValue:
    kind: str
    dots: int
    quarters: Fraction
    beams: int


VALUES = tuple(
    sorted(
        (
            NoteValue(kind, dots, Fraction(4, 2**i) * factor, max(0, i - 2))
            for i, kind in enumerate(NOTE_TYPES)
            for dots, factor in ((0, Fraction(1)), (1, Fraction(3, 2)))
        ),
        key=lambda v: v.quarters,
        reverse=True,
    )
)


@dataclass(frozen=True)
class TabSymbol:
    beat: Fraction
    length: Fraction
    value: NoteValue
    shape: str | None
    strings: tuple[int, ...]  # low E = 0, high E = 5
    event: StrumEvent | None
    tie_stop: bool = False
    tie_start: bool = False

    @property
    def rest(self) -> bool:
        return not self.strings


def quarter_length(beats: Fraction | float | int, denominator: int) -> Fraction:
    return Fraction(str(beats)) * Fraction(4, denominator)


def xml_duration(beats: Fraction | float | int, denominator: int) -> int:
    value = quarter_length(beats, denominator) * DIVISIONS
    if value.denominator != 1:
        raise ValueError(f"duration {beats} is outside the MusicXML timing grid")
    return int(value)


def beat_group(meter: Meter) -> int:
    num, den = meter
    return 3 if den == 8 and num > 3 and num % 3 == 0 else 1


def spell_span(
    start: Fraction, length: Fraction, meter: Meter
) -> list[tuple[Fraction, Fraction, NoteValue]]:
    """Spell a span, preserving beat boundaries after syncopated attacks."""
    out = []
    group = beat_group(meter)
    end = start + length
    while start < end:
        limit = end - start
        if start % group:
            limit = min(limit, (start // group + 1) * group - start)
        value = next((v for v in VALUES if v.quarters <= quarter_length(limit, meter[1])), None)
        if value is None:
            raise ValueError(f"cannot notate duration {limit} in {meter[0]}/{meter[1]}")
        part = value.quarters * Fraction(meter[1], 4)
        out.append((start, part, value))
        start += part
    return out


def strings_for(event: StrumEvent | None, shape: Shape) -> tuple[int, ...]:
    sounded = tuple(i for i, fret in enumerate(shape.frets) if fret is not None)
    if event is None or event.stroke is not None:
        return sounded[-4:] if event is not None and event.stroke == "up" else sounded
    picked = set()
    for ref in event.strings:
        if ref == "bass":
            if sounded:
                picked.add(sounded[0])
        elif isinstance(ref, int) and shape.frets[6 - ref] is not None:
            picked.add(6 - ref)
    return tuple(sorted(picked))


def tab_symbols(
    bar: Bar, pattern: StrumPattern | None, shapes: dict[str, Shape], meter: Meter
) -> list[TabSymbol]:
    """Fill the bar with timed attacks, tied continuations and explicit rests."""
    end = Fraction(bar.beats)
    if not bar.chords:
        # A bar with no chord (a pickup, a stretch the analysis heard no
        # harmony in) is silent: one rest per beat, as the sheets draw a
        # chord-less pickup bar (four quarter rests in 4/4), not one rest
        # per slot of the strum pattern.
        return [
            TabSymbol(beat, length, value, None, (), None)
            for b in range(bar.beats)
            for beat, length, value in spell_span(Fraction(b), Fraction(1), meter)
        ]
    if pattern is None:
        points: list[tuple[Fraction, StrumEvent | None]] = [
            (Fraction(c.beat), None) for c in bar.chords
        ]
    else:
        points = [(Fraction(str(e.beat)), e) for e in pattern.events if e.beat < end]
    if not points or points[0][0] > 0:
        points.insert(0, (Fraction(0), None))
    out = []
    for i, (start, event) in enumerate(points):
        stop = points[i + 1][0] if i + 1 < len(points) else end
        active = [c for c in bar.chords if c.beat <= start]
        shape_name = active[-1].shape if active else None
        # The inserted leading point is silence, not an invented down-strum.
        if pattern is not None and event is None:
            shape_name = None
        strings = strings_for(event, shapes[shape_name]) if shape_name else ()
        parts = spell_span(start, stop - start, meter)
        for k, (beat, length, value) in enumerate(parts):
            out.append(
                TabSymbol(
                    beat,
                    length,
                    value,
                    shape_name,
                    strings,
                    event,
                    tie_stop=bool(strings) and k > 0,
                    tie_start=bool(strings) and k < len(parts) - 1,
                )
            )
    return out


def beam_groups(symbols: list[TabSymbol], meter: Meter) -> list[list[int]]:
    """Consecutive short notes in one metrical group; singleton groups need flags."""
    groups: list[list[int]] = []
    for i, symbol in enumerate(symbols):
        if symbol.rest or not symbol.value.beams:
            continue
        if (
            groups
            and groups[-1][-1] == i - 1
            and symbols[i - 1].beat // beat_group(meter) == symbol.beat // beat_group(meter)
        ):
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def beam_marks(symbols: list[TabSymbol], meter: Meter) -> dict[int, tuple[str, ...]]:
    """MusicXML beam states, also used to draw exactly the same SVG connections."""
    out: dict[int, tuple[str, ...]] = {}
    for group in beam_groups(symbols, meter):
        if len(group) < 2:
            continue
        for k, i in enumerate(group):
            marks = []
            for level in range(1, symbols[i].value.beams + 1):
                before = k > 0 and symbols[group[k - 1]].value.beams >= level
                after = k + 1 < len(group) and symbols[group[k + 1]].value.beams >= level
                marks.append(
                    "continue"
                    if before and after
                    else "end"
                    if before
                    else "begin"
                    if after
                    else "forward hook"
                    if k + 1 < len(group)
                    else "backward hook"
                )
            out[i] = tuple(marks)
    return out
