"""Keys for the sheet: spelling, the sung key, candidate fingering keys and the capo.

A sheet is sung in one key and played with the chord shapes of another:
``shift = sung - fingering`` (mod 12) semitones. The player picks the
fingering key from a short list of easy keys; a shift of up to 9 is a capo
at that fret, 10 or 11 is played by tuning down a whole or half step
(capo 10 / 11 would sound an octave higher, and nobody plays there).
"""

from __future__ import annotations

from dataclasses import dataclass

from openbapo.model import Mode

SHARP_NAMES = ("C", "#C", "D", "#D", "E", "F", "#F", "G", "#G", "A", "#A", "B")
FLAT_NAMES = ("C", "bD", "D", "bE", "E", "F", "bG", "G", "bA", "A", "bB", "B")
# C major / A minor: the spellings closest to the key (secondary dominants
# take sharps, borrowed chords from the parallel minor take flats).
NATURAL_NAMES = ("C", "#C", "D", "bE", "E", "F", "#F", "G", "bA", "A", "bB", "B")
SHARP_KEYS = {7, 2, 9, 4, 11, 6}  # G D A E B F# (majors; minors use the relative major)

# Candidate fingering keys, in the order ties are broken (decision 1).
FINGERING_CANDIDATES = {Mode.MAJOR: (0, 7, 2, 9, 4, 5), Mode.MINOR: (9, 4, 2)}
RECOMMENDED_CAPO_MAX = 7  # higher capos are listed last and marked not recommended
TUNE_DOWN_MAX = 2  # a shift of 12 - 1 or 12 - 2 is played by tuning down that far
# The default: C or G shapes (Am / Em for minor songs), whichever needs the
# lower capo. Published play-along sheets are almost all in C or G shapes;
# picking the lowest capo among every candidate matches them far less often.
# The other candidates stay selectable.
DEFAULT_FINGERINGS = {Mode.MAJOR: (0, 7), Mode.MINOR: (9, 4)}


@dataclass(frozen=True)
class FingeringOption:
    tonic: int
    name: str  # "G" or "Em"
    capo: int  # fret, 0 when tuning down instead
    recommended: bool
    tune_down: int = 0  # semitones the strings are tuned down (0, 1 or 2)


def relative_major(tonic: int, mode: Mode) -> int:
    return tonic if mode == Mode.MAJOR else (tonic + 3) % 12


def pitch_name(pitch_class: int, key_tonic: int, key_mode: Mode) -> str:
    """Spell a pitch class the way the given key spells it (``#F`` / ``bB``, accidental first)."""
    major = relative_major(key_tonic, key_mode)
    if major == 0:
        names = NATURAL_NAMES
    elif major in SHARP_KEYS:
        names = SHARP_NAMES
    else:
        names = FLAT_NAMES
    return names[pitch_class % 12]


def key_name(tonic: int, mode: Mode) -> str:
    """``G`` for G major, ``Em`` for E minor, spelled in that key's own signature."""
    name = pitch_name(tonic, tonic, mode)
    return name if mode == Mode.MAJOR else name + "m"


def sung_key(original_tonic: int, transpose: int) -> int:
    return (original_tonic + transpose) % 12


def capo_for(sung_tonic: int, fingering_tonic: int) -> int:
    """Frets to capo so shapes in the fingering key sound in the sung key (0-11)."""
    return (sung_tonic - fingering_tonic) % 12


def capo_or_tuning(shift: int) -> tuple[int, int]:
    """(capo fret, semitones tuned down) that play a shift of 0-11 semitones."""
    if shift > 12 - TUNE_DOWN_MAX - 1:
        return 0, 12 - shift
    return shift, 0


def fingering_options(sung_tonic: int, mode: Mode) -> list[FingeringOption]:
    """Candidate fingering keys for a song sung in this key, lowest capo first.

    Ties keep the candidate table's order; capos above RECOMMENDED_CAPO_MAX,
    then options that tune down, go last and are not recommended.
    """
    options = []
    for f in FINGERING_CANDIDATES[mode]:
        capo, down = capo_or_tuning(capo_for(sung_tonic, f))
        options.append(
            FingeringOption(
                tonic=f,
                name=key_name(f, mode),
                capo=capo,
                recommended=not down and capo <= RECOMMENDED_CAPO_MAX,
                tune_down=down,
            )
        )
    # Capos up to 7, then higher capos, then retuning.
    return sorted(options, key=lambda o: (not o.recommended, o.tune_down, o.capo))


def default_fingering(sung_tonic: int, mode: Mode) -> FingeringOption:
    """The easiest shapes to play: C or G (Am or Em), the lower capo of the two."""
    preferred = DEFAULT_FINGERINGS[mode]
    return min(
        (o for o in fingering_options(sung_tonic, mode) if o.tonic in preferred),
        key=lambda o: (o.tune_down, o.capo, preferred.index(o.tonic)),
    )


def parse_key_name(text: str) -> tuple[int, Mode]:
    """``G``, ``Em``, ``#F``, ``F#``, ``bB`` -> (tonic, mode)."""
    import re

    match = re.fullmatch(r"\s*([#b]?)([A-Ga-g])([#b]?)(m|min)?\s*", text)
    if not match:
        raise ValueError(f"not a key name: {text!r}")
    before, letter, after, minor = match.groups()
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter.upper()]
    shift = sum(1 if a == "#" else -1 for a in before + after if a)
    return (base + shift) % 12, Mode.MINOR if minor else Mode.MAJOR
