"""Music primitives shared by analysis results and arrangement options."""

from enum import StrEnum
from typing import Annotated

from pydantic import Field, StrictFloat, StrictInt

# Pitch class as an integer, C = 0 … B = 11. Spelling (F# vs Gb) depends on the
# key and is decided at arrangement / render time, never stored.
PitchClass = Annotated[StrictInt, Field(ge=0, le=11)]

# Model confidence in [0, 1].
Confidence = Annotated[StrictFloat, Field(ge=0.0, le=1.0)]

# Seconds from the start of the source audio.
Seconds = Annotated[StrictFloat, Field(ge=0.0)]


class Mode(StrEnum):
    MAJOR = "major"
    MINOR = "minor"


class ChordQuality(StrEnum):
    """Chord qualities, spelled as in Harte notation so mir_eval can score them."""

    MAJ = "maj"
    MIN = "min"
    DOM7 = "7"
    MAJ7 = "maj7"
    MIN7 = "min7"
    MAJ6 = "maj6"
    MIN6 = "min6"
    SUS2 = "sus2"
    SUS4 = "sus4"
    POWER = "5"
    DIM = "dim"
    AUG = "aug"
    DIM7 = "dim7"
    HDIM7 = "hdim7"
