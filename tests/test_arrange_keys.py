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


def test_capos_up_to_nine_then_tuning_down_recommended_first() -> None:
    for mode in (Mode.MAJOR, Mode.MINOR):
        for sung in range(12):
            options = fingering_options(sung, mode)
            assert {o.tonic for o in options} == set(FINGERING_CANDIDATES[mode])
            for o in options:
                shift = (sung - o.tonic) % 12
                # A shift of 10 or 11 is played by tuning down, never capo 10 / 11.
                assert (o.capo, o.tune_down) == ((0, 12 - shift) if shift >= 10 else (shift, 0))
            flags = [o.recommended for o in options]
            assert flags == sorted(flags, reverse=True)  # recommended ones come first
            order = [(o.tune_down, o.capo) for o in options if not o.recommended]
            assert order == sorted(order)  # high capos, then retuning
            assert all(
                o.recommended == (not o.tune_down and o.capo <= RECOMMENDED_CAPO_MAX)
                for o in options
            )
            assert not default_fingering(sung, mode).tune_down  # never by default


def test_default_fingering_is_c_or_g_shapes_with_the_lower_capo() -> None:
    def default(tonic: int, mode: Mode) -> tuple[str, int]:
        o = default_fingering(tonic, mode)
        return o.name, o.capo

    assert default(7, Mode.MAJOR) == ("G", 0)
    assert default(9, Mode.MAJOR) == ("G", 2)  # A: G + 2, not A shapes (Bm, #Cm, #Fm)
    assert default(4, Mode.MAJOR) == ("C", 4)  # E: C + 4, not E shapes
    assert default(5, Mode.MAJOR) == ("C", 5)  # F: C + 5, not the barre F
    assert default(6, Mode.MAJOR) == ("C", 6)  # Gb: C + 6 beats G + 11
    assert default(11, Mode.MAJOR) == ("G", 4)
    assert default(4, Mode.MINOR) == ("Em", 0)
    assert default(0, Mode.MINOR) == ("Am", 3)  # C minor
    assert default(8, Mode.MINOR) == ("Em", 4)  # G# minor
    # The other shapes stay on the list.
    assert {o.name for o in fingering_options(9, Mode.MAJOR)} == {"C", "G", "D", "A", "E", "F"}


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
