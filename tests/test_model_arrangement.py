from typing import Any

import pytest
from pydantic import ValidationError

from openbapo.model import STRUM_AUTO, Arrangement


def test_defaults() -> None:
    reset = Arrangement()
    assert reset.transpose == 0
    assert reset.fingering_key is None
    assert reset.strum_pattern == STRUM_AUTO
    assert reset.simplify_chords is True
    assert reset.lyrics_text is None


@pytest.mark.parametrize("transpose", [-6, 0, 6])
def test_transpose_range(transpose: int) -> None:
    assert Arrangement(transpose=transpose).transpose == transpose


@pytest.mark.parametrize(
    "kw",
    [
        {"transpose": 7},
        {"transpose": -7},
        {"fingering_key": 12},
        {"fingering_key": -1},
        {"strum_pattern": ""},
    ],
)
def test_out_of_range(kw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Arrangement(**kw)


def test_only_user_adjustable_fields() -> None:
    # Capo override and per-chord edits are out of scope for phase 1.
    assert set(Arrangement.model_fields) == {
        "transpose",
        "fingering_key",
        "strum_pattern",
        "simplify_chords",
        "lyrics_text",
    }
    with pytest.raises(ValidationError):
        Arrangement(capo=3)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "raw",
    [
        '{"transpose": true}',
        '{"transpose": "2"}',
        '{"simplify_chords": 1}',
        '{"fingering_key": 2.0}',
    ],
)
def test_no_silent_coercion(raw: str) -> None:
    with pytest.raises(ValidationError):
        Arrangement.from_json(raw)
