"""The text form of a guitar lead sheet: chords per bar plus the sung melody in 简谱.

A sheet mirrors what a printed Chinese guitar sheet shows: a header (title,
``1=`` key, capo, tempo, metre) and one line per bar with the chord shapes
on the left and the numbered melody on the right::

    title: 小星星
    key: D         # the sheet's 1=, i.e. the sung key
    capo: 2        # chord shapes are written in key - capo
    bpm: 100
    time: 4/4
    ---
    C      | 1 1 5 5
    F C    | 6 6 5 -
    C(3) G | 4 [4 3] 3 -

Chords divide the bar evenly unless given beat counts in parentheses.
Melody tokens divide the bar evenly too; ``[ ]`` groups tokens into one
share (so ``[6 6]`` is two eighths), a trailing ``.`` dots a note (it takes
three quarters of its share plus the next token's, as ``[2. 5]`` for a
dotted eighth and a sixteenth), ``'`` and ``,`` raise or lower an octave,
``0`` is a rest and ``-`` extends the previous note, also across a bar
line when it opens a bar; ``#5`` and ``b7`` are accidentals. A bar may
leave either side empty. A ``#`` at the start of a line, or after a space
and not followed by a note name or degree, opens a comment (so ``D/#F``
and ``#5`` are not comments).

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
# A comment opens at a "#" starting the line or following a space, unless
# the "#" spells a sharp (D/#F in a chord, #5 in a melody).
COMMENT = re.compile(r"(?:^|(?<=\s))#(?![A-G1-7])")
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
    # Chinese sheets write a minor chord with a raised fifth as "m/5+" or
    # "m+5" (E G C, a passing chord); at the triad level it is scored as
    # minor, and the shapes are strummed as the plain minor chord.
    "m/5+": ChordQuality.MIN,
    "m+5": ChordQuality.MIN,
    "m#5": ChordQuality.MIN,
    # Extensions beyond the 14 qualities fold onto the nearest one: the
    # ninth, eleventh and thirteenth keep the seventh's quality, an added
    # ninth or second keeps the triad's, a suspended dominant is scored
    # as the suspension (its seventh is not in the vocabulary).
    "9": ChordQuality.DOM7,
    "11": ChordQuality.DOM7,
    "13": ChordQuality.DOM7,
    "maj9": ChordQuality.MAJ7,
    "M9": ChordQuality.MAJ7,
    "m9": ChordQuality.MIN7,
    "min9": ChordQuality.MIN7,
    "m11": ChordQuality.MIN7,
    "add9": ChordQuality.MAJ,
    "add2": ChordQuality.MAJ,
    "add4": ChordQuality.MAJ,
    "madd9": ChordQuality.MIN,
    "madd2": ChordQuality.MIN,
    "madd4": ChordQuality.MIN,
    "6/9": ChordQuality.MAJ6,
    "69": ChordQuality.MAJ6,
    "7sus4": ChordQuality.SUS4,
    "7sus2": ChordQuality.SUS2,
    "7sus": ChordQuality.SUS4,
    "9sus4": ChordQuality.SUS4,
    "sus9": ChordQuality.SUS4,
    "mmaj7": ChordQuality.MIN7,
    "mM7": ChordQuality.MIN7,
    "m(maj7)": ChordQuality.MIN7,
    "7b9": ChordQuality.DOM7,
    "7#9": ChordQuality.DOM7,
    "7#5": ChordQuality.DOM7,
    "7b5": ChordQuality.DOM7,
    "maj7#11": ChordQuality.MAJ7,
    "aug7": ChordQuality.DOM7,
    "add11": ChordQuality.MAJ,
    "add13": ChordQuality.MAJ,
    "7add11": ChordQuality.DOM7,
    "7add13": ChordQuality.DOM7,
    "m7sus2": ChordQuality.SUS2,
    "m7sus4": ChordQuality.SUS4,
    "msus2": ChordQuality.SUS2,
    "msus4": ChordQuality.SUS4,
    "maj7sus2": ChordQuality.SUS2,
    "maj9sus4": ChordQuality.SUS4,
    "+7": ChordQuality.DOM7,
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
    tied: bool = False  # continues the previous bar's last note (a '-' opening the bar)
    accidental: int = 0  # +1 for #, -1 for b, written before the digit


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


NOTE_NAME = r"(?:[#b]?[A-G][#b]*)"  # Chinese sheets write the accidental first: #F, bB


def pitch_class(name: str) -> int:
    match = re.fullmatch(r"([#b]?)([A-Ga-g])([#b]*)", name.strip())
    if not match:
        raise SheetError(f"not a note name: {name!r}")
    pc = PITCH[match.group(2).upper()]
    for accidental in match.group(1) + match.group(3):
        pc += 1 if accidental == "#" else -1
    return pc % 12


def parse_chord_name(text: str) -> tuple[int, ChordQuality, int | None]:
    """``Am7``, ``F#m``, ``#Fm``, ``Bb``, ``C/E``, ``D/#F`` -> (root, quality, bass)."""
    match = re.fullmatch(rf"({NOTE_NAME})(m/5\+|6/9|[^/]*)(?:/({NOTE_NAME}))?", text.strip())
    if not match:
        raise SheetError(f"not a chord name: {text!r}")
    root, suffix, bass = match.groups()
    if suffix not in QUALITIES:
        raise SheetError(f"unknown chord quality {suffix!r} in {text!r}")
    return pitch_class(root), QUALITIES[suffix], pitch_class(bass) if bass else None


def parse_chords(text: str, beats_per_bar: int) -> tuple[ChordSpan, ...]:
    """``C F`` splits the bar evenly; ``C(3) G(1)`` gives beat counts.

    ``Bm(2) F`` mixes them: the uncounted chords share what the counted
    ones leave (readers write it that way for a chord change on a beat).
    """
    tokens = text.split()
    if not tokens:
        return ()
    explicit = [re.fullmatch(r"(.+)\((\d+(?:\.\d+)?)\)", t) for t in tokens]
    if len(tokens) == 1 and explicit[0]:
        # A lone chord with a beat count fills the bar whatever the count says
        # (readers write "D(2)" for a chord struck on beat 2, or the count of
        # a partial bar); there is nothing else the remaining beats could be.
        tokens, explicit = [explicit[0].group(1)], [None]
    counted = sum((Fraction(m.group(2)) for m in explicit if m), Fraction(0))
    uncounted = sum(1 for m in explicit if not m)
    if uncounted and counted >= beats_per_bar:
        raise SheetError(f"counted chords fill the bar, no beats left for the rest: {text!r}")
    spans = []
    for token, match in zip(tokens, explicit, strict=True):
        if match:
            name, count = match.group(1), Fraction(match.group(2))
        else:
            name, count = token, (beats_per_bar - counted) / uncounted
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
    """Share the beats among the items; a dotted token takes 1.5, the one after it 0.5."""
    out: list[tuple[str, Fraction]] = []
    if not items:
        return out
    weights = []
    for i, item in enumerate(items):
        dotted = isinstance(item, str) and item.endswith(".")
        after_dot = i > 0 and isinstance(items[i - 1], str) and items[i - 1].endswith(".")
        weights.append(Fraction(3, 2) if dotted else Fraction(1, 2) if after_dot else Fraction(1))
    total = sum(weights, Fraction(0))
    for item, weight in zip(items, weights, strict=True):
        part = share * weight / total
        if isinstance(item, list):
            out.extend(_flatten(item, part))
        else:
            out.append((item.rstrip("."), part))
    return out


def parse_melody(text: str, beats_per_bar: int) -> tuple[NoteSpan, ...]:
    tokens = _flatten(_group_tokens(text), Fraction(beats_per_bar))
    spans: list[NoteSpan] = []
    for token, beats in tokens:
        if token == "-":
            if not spans:
                # Opening a bar with '-' ties the previous bar's last note over.
                spans.append(NoteSpan(None, 0, beats, tied=True))
                continue
            last = spans[-1]
            spans[-1] = NoteSpan(
                last.degree, last.octave, last.beats + beats, last.tied, last.accidental
            )
            continue
        match = re.fullmatch(r"([#b]?)([0-7Xx])('*)(,*)", token)
        if not match or (match.group(1) and match.group(2) in "0Xx"):
            raise SheetError(f"not a 简谱 token: {token!r} in {text!r}")
        sign, digit, up, down = match.groups()
        degree = None if digit in "0Xx" else int(digit)  # X: spoken / unpitched, a rest here
        accidental = {"#": 1, "b": -1}.get(sign, 0)
        spans.append(NoteSpan(degree, len(up) - len(down), beats, accidental=accidental))
    return tuple(spans)


def parse_sheet(text: str, *, source: str = "") -> Sheet:
    header: dict[str, str] = {}
    body: list[str] = []
    in_body = False
    for raw in text.splitlines():
        line = COMMENT.split(raw, 1)[0].rstrip()
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
    bars: list[Bar] = []
    for line in body:
        chords_text, _, melody_text = line.partition("|")
        notes = parse_melody(melody_text, beats_per_bar)
        if notes and notes[0].tied:
            previous = bars[-1].notes[-1] if bars and bars[-1].notes else None
            if previous is None:
                raise SheetError(f"'-' opens a bar with nothing to tie to: {line!r}")
            notes = (
                NoteSpan(
                    previous.degree,
                    previous.octave,
                    notes[0].beats,
                    True,
                    previous.accidental,
                ),
                *notes[1:],
            )
        bars.append(Bar(parse_chords(chords_text, beats_per_bar), notes))
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
    return tonic + MAJOR_DEGREES[note.degree - 1] + 12 * note.octave + note.accidental
