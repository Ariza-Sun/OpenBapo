"""Keys for the sheet: spelling, the sung key, candidate fingering keys and the capo.

A sheet is sung in one key and played with the chord shapes of another:
``capo = sung - fingering`` (mod 12). The player picks the fingering key
from a short list of easy keys; the capo follows (requirements 5.3).
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


@dataclass(frozen=True)
class FingeringOption:
    tonic: int
    name: str  # "G" or "Em"
    capo: int
    recommended: bool


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


def fingering_options(sung_tonic: int, mode: Mode) -> list[FingeringOption]:
    """Candidate fingering keys for a song sung in this key, lowest capo first.

    Ties keep the candidate table's order; capos above RECOMMENDED_CAPO_MAX
    go last and are flagged. No option has a negative capo.
    """
    options = [
        FingeringOption(
            tonic=f,
            name=key_name(f, mode),
            capo=capo_for(sung_tonic, f),
            recommended=capo_for(sung_tonic, f) <= RECOMMENDED_CAPO_MAX,
        )
        for f in FINGERING_CANDIDATES[mode]
    ]
    return sorted(options, key=lambda o: (not o.recommended, o.capo))


def default_fingering(sung_tonic: int, mode: Mode) -> FingeringOption:
    return fingering_options(sung_tonic, mode)[0]
