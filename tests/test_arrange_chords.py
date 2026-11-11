"""Chord names on the sheet and the simplify switch."""

import pytest

from openbapo.arrange.chords import SUFFIX, chord_name, simplify, written_root
from openbapo.model import ChordQuality, Mode


def test_every_quality_has_a_suffix() -> None:
    assert set(SUFFIX) == set(ChordQuality)


@pytest.mark.parametrize(
    ("root", "quality", "bass", "expected"),
    [
        (0, ChordQuality.MAJ, None, "C"),
        (9, ChordQuality.MIN, None, "Am"),
        (7, ChordQuality.DOM7, None, "G7"),
        (0, ChordQuality.MAJ7, None, "Cmaj7"),
        (2, ChordQuality.MIN7, None, "Dm7"),
        (11, ChordQuality.HDIM7, None, "Bm7-5"),
        (2, ChordQuality.MAJ, 6, "D/#F"),  # in G major
        (0, ChordQuality.MAJ, 0, "C"),  # bass equal to root is no slash
    ],
)
def test_chord_names_in_g_major(
    root: int, quality: ChordQuality, bass: int | None, expected: str
) -> None:
    assert chord_name(root, quality, bass, 7, Mode.MAJOR) == expected


def test_spelling_follows_the_fingering_key() -> None:
    assert chord_name(10, ChordQuality.MAJ, None, 5, Mode.MAJOR) == "bB"
    assert chord_name(10, ChordQuality.MAJ, None, 7, Mode.MAJOR) == "#A"


def test_simplify_mapping() -> None:
    assert simplify(ChordQuality.MAJ7, None) == (ChordQuality.MAJ, None)
    assert simplify(ChordQuality.SUS4, 4) == (ChordQuality.MAJ, None)  # slash bass dropped
    assert simplify(ChordQuality.MIN7, None) == (ChordQuality.MIN, None)
    assert simplify(ChordQuality.HDIM7, None) == (ChordQuality.DIM, None)
    assert simplify(ChordQuality.AUG, None) == (ChordQuality.AUG, None)
    assert simplify(ChordQuality.MAJ, 7) == (ChordQuality.MAJ, None)


def test_written_root_follows_the_capo() -> None:
    assert written_root(7, 0) == 7
    assert written_root(7, 3) == 4  # G sounding under capo 3 is written E
    assert written_root(1, 3) == 10
