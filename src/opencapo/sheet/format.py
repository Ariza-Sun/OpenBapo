"""The text form of a guitar lead sheet: chords per bar plus the sung melody in 简谱.

A sheet mirrors what a printed Chinese guitar sheet shows: a header (title,
``1=`` key, capo, tempo, metre) and one line per bar with the chord shapes
on the left and the numbered melody on the right::

    title: 送别
    key: C         # the sheet's 1=, i.e. the sung key
    capo: 0        # chord shapes are written in key - capo
    bpm: 72
    time: 4/4
    ---
    C      | 5 3 5 1'
    F C    | 6 1' 5 -
    C(3) G | 5 [1 2] 3 -

Chords divide the bar evenly unless given beat counts in parentheses.
Melody tokens divide the bar evenly too; ``[ ]`` groups tokens into one
share (so ``[6 6]`` is two eighths), ``'`` and ``,`` raise or lower an
octave, ``0`` is a rest and ``-`` extends the previous note. A bar may
leave either side empty. Lines starting with ``#`` are comments.

The sounding pitches follow the sheet's convention: the melody degrees are
in the ``key`` (what is sung), the chord shapes sound ``capo`` semitones
higher than written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from opencapo.model import ChordQuality, Mode

PITCH = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
MAJOR_DEGREES = (0, 2, 4, 5, 7, 9, 11)  # 简谱 1-7 in a major key
QUALITIES: dict[str, ChordQuality] = {
    "": ChordQuality.MAJ,
    "maj": ChordQuality.MAJ,
    "M": ChordQuality.MAJ,
    "m": ChordQuality.MIN,
    "min": ChordQuality.MIN,
    "7": ChordQuality.DOM7,
    "maj7": ChordQuality.MAJ7,
    "M7": ChordQuality.MAJ7,
    "m7": ChordQuality.MIN7,
    "min7": ChordQuality.MIN7,
    "6": ChordQuality.MAJ6,
    "m6": ChordQuality.MIN6,
    "sus2": ChordQuality.SUS2,
    "sus4": ChordQuality.SUS4,
    "sus": ChordQuality.SUS4,
    "5": ChordQuality.POWER,
    "dim": ChordQuality.DIM,
    "aug": ChordQuality.AUG,
    "+": ChordQuality.AUG,
    "dim7": ChordQuality.DIM7,
    "m7b5": ChordQuality.HDIM7,
    "hdim7": ChordQuality.HDIM7,
}


class SheetError(ValueError):
    pass


@dataclass(frozen=True)
class ChordSpan:
    root: int  # written (shape) pitch class
    quality: ChordQuality
    bass: int | None  # written bass pitch class for slash chords
    beats: Fraction


@dataclass(frozen=True)
class NoteSpan:
    degree: int | None  # 1-7, None for a rest
    octave: int  # 0 = the sheet's plain digits, +1 per ', -1 per ,
    beats: Fraction


@dataclass(frozen=True)
class Bar:
    chords: tuple[ChordSpan, ...]
    notes: tuple[NoteSpan, ...]


@dataclass(frozen=True)
class Sheet:
    title: str
    key: int  # the sheet's 1=, the sung key's tonic
    mode: Mode
    capo: int
    bpm: float
    beats_per_bar: int
    bars: tuple[Bar, ...]
    source: str = field(default="", compare=False)

    @property
    def fingering_key(self) -> int:
        """The key the chord shapes are written in."""
        return (self.key - self.capo) % 12

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.bpm

    @property
    def duration(self) -> float:
        return len(self.bars) * self.beats_per_bar * self.seconds_per_beat


def pitch_class(name: str) -> int:
    match = re.fullmatch(r"([A-Ga-g])([#b]*)", name.strip())
    if not match:
        raise SheetError(f"not a note name: {name!r}")
    pc = PITCH[match.group(1).upper()]
    for accidental in match.group(2):
        pc += 1 if accidental == "#" else -1
    return pc % 12


def parse_chord_name(text: str) -> tuple[int, ChordQuality, int | None]:
    """``Am7``, ``F#m``, ``Bb``, ``C/E`` -> (root, quality, bass)."""
    match = re.fullmatch(r"([A-G][#b]?)([^/]*)(?:/([A-G][#b]?))?", text.strip())
    if not match:
        raise SheetError(f"not a chord name: {text!r}")
    root, suffix, bass = match.groups()
    if suffix not in QUALITIES:
        raise SheetError(f"unknown chord quality {suffix!r} in {text!r}")
    return pitch_class(root), QUALITIES[suffix], pitch_class(bass) if bass else None


def parse_chords(text: str, beats_per_bar: int) -> tuple[ChordSpan, ...]:
    """``C F`` splits the bar evenly; ``C(3) G(1)`` gives beat counts."""
    tokens = text.split()
    if not tokens:
        return ()
    explicit = [re.fullmatch(r"(.+)\((\d+(?:\.\d+)?)\)", t) for t in tokens]
    if any(explicit) and not all(explicit):
        raise SheetError(f"mix of counted and uncounted chords in {text!r}")
    spans = []
    for token, match in zip(tokens, explicit, strict=True):
        if match:
            name, count = match.group(1), Fraction(match.group(2))
        else:
            name, count = token, Fraction(beats_per_bar, len(tokens))
        root, quality, bass = parse_chord_name(name)
        spans.append(ChordSpan(root, quality, bass, count))
    total = sum((s.beats for s in spans), Fraction(0))
    if total != beats_per_bar:
        raise SheetError(f"chords cover {total} beats, bar has {beats_per_bar}: {text!r}")
    return tuple(spans)


def _group_tokens(text: str) -> list:  # type: ignore[type-arg]
    """Split melody text into tokens, with ``[...]`` as nested lists."""
    stack: list[list] = [[]]  # type: ignore[type-arg]
    for token in re.findall(r"\[|\]|[^\s\[\]]+", text):
        if token == "[":
            stack.append([])
        elif token == "]":
            if len(stack) == 1:
                raise SheetError(f"unbalanced ']' in {text!r}")
            group = stack.pop()
            stack[-1].append(group)
        else:
            stack[-1].append(token)
    if len(stack) != 1:
        raise SheetError(f"unbalanced '[' in {text!r}")
    return stack[0]


def _flatten(items: list, share: Fraction) -> list[tuple[str, Fraction]]:  # type: ignore[type-arg]
    out: list[tuple[str, Fraction]] = []
    if not items:
        return out
    each = share / len(items)
    for item in items:
        if isinstance(item, list):
            out.extend(_flatten(item, each))
        else:
            out.append((item, each))
    return out


def parse_melody(text: str, beats_per_bar: int) -> tuple[NoteSpan, ...]:
    tokens = _flatten(_group_tokens(text), Fraction(beats_per_bar))
    spans: list[NoteSpan] = []
    for token, beats in tokens:
        if token == "-":
            if not spans:
                raise SheetError(f"'-' with nothing to extend in {text!r}")
            last = spans[-1]
            spans[-1] = NoteSpan(last.degree, last.octave, last.beats + beats)
            continue
        match = re.fullmatch(r"([0-7])('*)(,*)", token)
        if not match:
            raise SheetError(f"not a 简谱 token: {token!r} in {text!r}")
        digit, up, down = match.groups()
        degree = None if digit == "0" else int(digit)
        spans.append(NoteSpan(degree, len(up) - len(down), beats))
    return tuple(spans)


def parse_sheet(text: str, *, source: str = "") -> Sheet:
    header: dict[str, str] = {}
    body: list[str] = []
    in_body = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.strip() == "---":
            in_body = True
            continue
        if in_body:
            body.append(line)
        elif ":" in line:
            name, value = line.split(":", 1)
            header[name.strip().lower()] = value.strip()
        else:
            raise SheetError(f"header line without ':' : {line!r}")
    if not in_body:
        raise SheetError("missing '---' between header and bars")
    try:
        key = pitch_class(header.get("key", "C"))
        mode = Mode(header.get("mode", "major"))
        capo = int(header.get("capo", "0"))
        bpm = float(header.get("bpm", "90"))
        numerator = header.get("time", "4/4").split("/")[0]
        beats_per_bar = int(numerator)
    except ValueError as e:
        raise SheetError(f"bad header value: {e}") from e
    if not 0 <= capo <= 11 or bpm <= 0 or beats_per_bar <= 0:
        raise SheetError("capo must be 0-11, bpm and time positive")
    bars = []
    for line in body:
        chords_text, _, melody_text = line.partition("|")
        bars.append(
            Bar(parse_chords(chords_text, beats_per_bar), parse_melody(melody_text, beats_per_bar))
        )
    if not bars:
        raise SheetError("no bars")
    return Sheet(
        title=header.get("title", Path(source).stem if source else "untitled"),
        key=key,
        mode=mode,
        capo=capo,
        bpm=bpm,
        beats_per_bar=beats_per_bar,
        bars=tuple(bars),
        source=source,
    )


def load_sheet(path: Path) -> Sheet:
    return parse_sheet(path.read_text(encoding="utf-8"), source=str(path))


def degree_midi(sheet: Sheet, note: NoteSpan, base_octave: int = 4) -> int | None:
    """MIDI pitch of a melody note: the sung key's tonic in octave 4 is ``1``."""
    if note.degree is None:
        return None
    tonic = 12 * (base_octave + 1) + sheet.key
    return tonic + MAJOR_DEGREES[note.degree - 1] + 12 * note.octave
