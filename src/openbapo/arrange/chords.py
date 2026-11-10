"""Chord names as written on the sheet, and the simplified names a beginner wants."""

from __future__ import annotations

from openbapo.arrange.keys import pitch_name
from openbapo.model import ChordQuality, Mode

SUFFIX = {
    ChordQuality.MAJ: "",
    ChordQuality.MIN: "m",
    ChordQuality.DOM7: "7",
    ChordQuality.MAJ7: "maj7",
    ChordQuality.MIN7: "m7",
    ChordQuality.MAJ6: "6",
    ChordQuality.MIN6: "m6",
    ChordQuality.SUS2: "sus2",
    ChordQuality.SUS4: "sus4",
    ChordQuality.POWER: "5",
    ChordQuality.DIM: "dim",
    ChordQuality.AUG: "aug",
    ChordQuality.DIM7: "dim7",
    ChordQuality.HDIM7: "m7-5",
}

# Decision 3: what the simplify switch shows instead of each quality.
SIMPLIFIED = {
    ChordQuality.MAJ7: ChordQuality.MAJ,
    ChordQuality.DOM7: ChordQuality.MAJ,
    ChordQuality.MAJ6: ChordQuality.MAJ,
    ChordQuality.SUS2: ChordQuality.MAJ,
    ChordQuality.SUS4: ChordQuality.MAJ,
    ChordQuality.POWER: ChordQuality.MAJ,
    ChordQuality.MIN7: ChordQuality.MIN,
    ChordQuality.MIN6: ChordQuality.MIN,
    ChordQuality.DIM7: ChordQuality.DIM,
    ChordQuality.HDIM7: ChordQuality.DIM,
}


def simplify(quality: ChordQuality, bass: int | None) -> tuple[ChordQuality, int | None]:
    """The simplified quality, with the slash bass dropped."""
    return SIMPLIFIED.get(quality, quality), None


def written_root(sounding_root: int, capo: int) -> int:
    """The pitch class a shape is written as when it sounds `sounding_root` under the capo."""
    return (sounding_root - capo) % 12


def chord_name(
    root: int, quality: ChordQuality, bass: int | None, key_tonic: int, key_mode: Mode
) -> str:
    """``#Fm7``, ``bB``, ``D/#F``: the root (and bass) spelled in the given key."""
    name = pitch_name(root, key_tonic, key_mode) + SUFFIX[quality]
    if bass is not None and bass != root:
        name += "/" + pitch_name(bass, key_tonic, key_mode)
    return name
