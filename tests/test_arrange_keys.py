"""Key spelling, the sung key, candidate fingering keys and the capo."""

import pytest

from openbapo.arrange.keys import (
    FINGERING_CANDIDATES,
    RECOMMENDED_CAPO_MAX,
    capo_for,
    default_fingering,
    fingering_options,
    key_name,
    pitch_name,
    sung_key,
)
from openbapo.model import Mode


@pytest.mark.parametrize("sung", range(12))
@pytest.mark.parametrize("fingering", range(12))
def test_capo_covers_every_key_pair(sung: int, fingering: int) -> None:
    capo = capo_for(sung, fingering)
    assert 0 <= capo <= 11
    assert (fingering + capo) % 12 == sung  # shapes in the fingering key sound in the sung key


def test_no_negative_capo_and_recommended_first() -> None:
    for mode in (Mode.MAJOR, Mode.MINOR):
        for sung in range(12):
            options = fingering_options(sung, mode)
            assert {o.tonic for o in options} == set(FINGERING_CANDIDATES[mode])
            assert all(o.capo >= 0 for o in options)
            flags = [o.recommended for o in options]
            assert flags == sorted(flags, reverse=True)  # recommended ones come first
            capos = [o.capo for o in options]
            assert capos == sorted(capos)
            assert all(o.recommended == (o.capo <= RECOMMENDED_CAPO_MAX) for o in options)


def test_default_fingering_is_the_lowest_capo_with_table_order_on_ties() -> None:
    assert default_fingering(7, Mode.MAJOR).name == "G"  # G major: G shapes, no capo
    assert default_fingering(8, Mode.MAJOR).name == "G"  # Ab: G + 1 beats others
    assert default_fingering(1, Mode.MAJOR).name == "C"  # Db: C + 1 ties with nothing lower
    assert default_fingering(6, Mode.MAJOR).name == "F"  # Gb: F + 1 beats E + 2
    assert default_fingering(6, Mode.MAJOR).capo == 1
    assert default_fingering(4, Mode.MINOR).name == "Em"
    assert default_fingering(0, Mode.MINOR).name == "Am"  # C minor: Am + 3
    assert default_fingering(0, Mode.MINOR).capo == 3


def test_sung_key_wraps() -> None:
    assert sung_key(7, -6) == 1 and sung_key(7, 6) == 1 and sung_key(0, 0) == 0


@pytest.mark.parametrize(
    ("pc", "tonic", "mode", "expected"),
    [
        (6, 7, Mode.MAJOR, "#F"),  # G major: sharps
        (10, 5, Mode.MAJOR, "bB"),  # F major: flats
        (6, 0, Mode.MAJOR, "#F"),  # C major: closest spelling
        (10, 0, Mode.MAJOR, "bB"),
        (3, 0, Mode.MAJOR, "bE"),
        (1, 0, Mode.MAJOR, "#C"),
        (8, 9, Mode.MINOR, "bA"),  # A minor spells like C major
        (1, 4, Mode.MINOR, "#C"),  # E minor spells like G major
        (3, 2, Mode.MINOR, "bE"),  # D minor spells like F major
    ],
)
def test_pitch_spelling_follows_the_key(pc: int, tonic: int, mode: Mode, expected: str) -> None:
    assert pitch_name(pc, tonic, mode) == expected


def test_key_names() -> None:
    assert key_name(7, Mode.MAJOR) == "G" and key_name(4, Mode.MINOR) == "Em"
    assert key_name(6, Mode.MAJOR) == "#F" and key_name(10, Mode.MAJOR) == "bB"
    assert key_name(10, Mode.MINOR) == "bBm"  # Bb minor spells like Db major: flats
