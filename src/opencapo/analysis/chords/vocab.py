"""The shared chord label space every expert is mapped into.

Labels follow Harte notation (`C:maj7/3`, `N`), which is what the public
annotations use and what mir_eval scores. Internally a chord is a
(root pitch class, ChordQuality) pair plus an optional bass pitch class.
The 14 qualities x 12 roots plus "no chord" give the 169-way posterior that
the fusion layer works on; anything outside the vocabulary is folded onto its
nearest member and flagged as inexact so callers can lower their confidence.
"""

from dataclasses import dataclass
from typing import Final, Literal

from opencapo.model import ChordQuality

QUALITIES: Final[tuple[ChordQuality, ...]] = tuple(ChordQuality)
N_QUALITIES: Final = len(QUALITIES)
NO_CHORD: Final = 12 * N_QUALITIES  # index of "N" in the posterior
N_LABELS: Final = NO_CHORD + 1

# Chord tones as semitones above the root.
INTERVALS: Final[dict[ChordQuality, frozenset[int]]] = {
    ChordQuality.MAJ: frozenset({0, 4, 7}),
    ChordQuality.MIN: frozenset({0, 3, 7}),
    ChordQuality.DOM7: frozenset({0, 4, 7, 10}),
    ChordQuality.MAJ7: frozenset({0, 4, 7, 11}),
    ChordQuality.MIN7: frozenset({0, 3, 7, 10}),
    ChordQuality.MAJ6: frozenset({0, 4, 7, 9}),
    ChordQuality.MIN6: frozenset({0, 3, 7, 9}),
    ChordQuality.SUS2: frozenset({0, 2, 7}),
    ChordQuality.SUS4: frozenset({0, 5, 7}),
    ChordQuality.POWER: frozenset({0, 7}),
    ChordQuality.DIM: frozenset({0, 3, 6}),
    ChordQuality.AUG: frozenset({0, 4, 8}),
    ChordQuality.DIM7: frozenset({0, 3, 6, 9}),
    ChordQuality.HDIM7: frozenset({0, 3, 6, 10}),
}

# Harte shorthands outside our vocabulary, folded onto the nearest member.
# Extensions beyond the seventh are dropped; minmaj7 loses its major seventh.
FOLDED_QUALITIES: Final[dict[str, ChordQuality]] = {
    "minmaj7": ChordQuality.MIN,
    "maj9": ChordQuality.MAJ7,
    "min9": ChordQuality.MIN7,
    "9": ChordQuality.DOM7,
    "b9": ChordQuality.DOM7,
    "#9": ChordQuality.DOM7,
    "11": ChordQuality.DOM7,
    "min11": ChordQuality.MIN7,
    "#11": ChordQuality.DOM7,
    "13": ChordQuality.DOM7,
    "b13": ChordQuality.DOM7,
    "maj13": ChordQuality.MAJ7,
    "min13": ChordQuality.MIN7,
    "1": ChordQuality.POWER,
}

_NOTE_BASE: Final = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
PITCH_NAMES: Final = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_DEGREE_BASE: Final = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11, 9: 2, 11: 5, 13: 9}
_INTERVAL_DEGREE: Final = ("1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7")


class LabelError(ValueError):
    """A chord label that is not valid Harte notation."""


@dataclass(frozen=True)
class ParsedChord:
    root: int
    quality: ChordQuality
    bass: int | None  # pitch class of the bass if not the root
    exact: bool  # False when the label was folded onto a nearby quality

    @property
    def index(self) -> int:
        return label_index(self.root, self.quality)


def pitch_class(name: str) -> int:
    """`C`, `F#`, `Bb`, `Cb`, `B##` -> 0..11."""
    if not name or name[0] not in _NOTE_BASE:
        raise LabelError(f"bad note name {name!r}")
    value = _NOTE_BASE[name[0]]
    for accidental in name[1:]:
        if accidental == "#":
            value += 1
        elif accidental == "b":
            value -= 1
        else:
            raise LabelError(f"bad note name {name!r}")
    return value % 12


def degree_semitones(degree: str) -> int:
    """`3`, `b3`, `#5`, `b7`, `9` -> semitones above the root (mod 12)."""
    text = degree
    offset = 0
    while text and text[0] in "#b":
        offset += 1 if text[0] == "#" else -1
        text = text[1:]
    if not text.isdigit() or int(text) not in _DEGREE_BASE:
        raise LabelError(f"bad scale degree {degree!r}")
    return (_DEGREE_BASE[int(text)] + offset) % 12


def label_index(root: int, quality: ChordQuality) -> int:
    return root * N_QUALITIES + QUALITIES.index(quality)


def index_label(index: int) -> tuple[int, ChordQuality] | None:
    """Inverse of label_index; None for the no-chord index."""
    if index == NO_CHORD:
        return None
    if not 0 <= index < NO_CHORD:
        raise IndexError(index)
    return divmod(index, N_QUALITIES)[0], QUALITIES[index % N_QUALITIES]


def chord_tones(root: int, quality: ChordQuality) -> frozenset[int]:
    return frozenset((root + i) % 12 for i in INTERVALS[quality])


def parse_label(label: str) -> ParsedChord | None:
    """Parse a Harte label into our vocabulary. `N` and `X` give None.

    Handles `C`, `C:maj`, `C:min7/b7`, `C:(1,3,5)`, `C:maj(9)`, `C:7(*3)`,
    `C/3`. Explicit interval lists are matched to the nearest template;
    extensions in parentheses are ignored (and make the result inexact).
    """
    text = label.strip()
    if text in ("N", "X", ""):
        return None
    bass_text = None
    if "/" in text:
        text, bass_text = text.split("/", 1)
    if ":" in text:
        root_text, quality_text = text.split(":", 1)
    else:
        root_text, quality_text = text, "maj"
    root = pitch_class(root_text)
    exact = True

    shorthand, _, extensions = quality_text.partition("(")
    if extensions:
        extensions = extensions.rstrip(")")
    if shorthand == "" and extensions:
        # An explicit interval list such as (1,3,5) or (1,b3,5,b7).
        tones = frozenset(
            degree_semitones(d.strip()) for d in extensions.split(",") if d.strip() and "*" not in d
        )
        quality = _nearest_quality(tones)
        exact = INTERVALS[quality] == tones
    else:
        if shorthand == "":
            shorthand = "maj"
        try:
            quality = ChordQuality(shorthand)
        except ValueError:
            if shorthand not in FOLDED_QUALITIES:
                raise LabelError(f"unknown chord quality {shorthand!r} in {label!r}") from None
            quality, exact = FOLDED_QUALITIES[shorthand], False
        if extensions:
            exact = False

    bass = None
    if bass_text:
        bass = (root + degree_semitones(bass_text.strip())) % 12
        if bass == root:
            bass = None
    return ParsedChord(root=root, quality=quality, bass=bass, exact=exact)


def _nearest_quality(tones: frozenset[int]) -> ChordQuality:
    """The template sharing the most tones with `tones` (ties go to simpler chords)."""
    best = max(
        QUALITIES,
        key=lambda q: (
            len(INTERVALS[q] & tones) - len(INTERVALS[q] ^ tones),
            -len(INTERVALS[q]),
        ),
    )
    return best


def format_label(
    root: int | None, quality: ChordQuality | None = None, bass: int | None = None
) -> str:
    """Harte text for mir_eval and annotation files; `N` for no chord."""
    if root is None or quality is None:
        return "N"
    text = f"{PITCH_NAMES[root]}:{quality.value}"
    if bass is not None and bass != root:
        text += "/" + _INTERVAL_DEGREE[(bass - root) % 12]
    return text


BassRelation = Literal["root", "chord_tone", "foreign"]


def bass_relation(root: int, quality: ChordQuality, bass_pc: int) -> BassRelation:
    """How a detected bass pitch class relates to a chord.

    "root" is root position, "chord_tone" marks an inversion (the bass is the
    third, fifth or seventh), and "foreign" means the bass is not a chord tone.
    """
    if bass_pc == root:
        return "root"
    if bass_pc in chord_tones(root, quality):
        return "chord_tone"
    return "foreign"
