"""The strum-pattern library and the default pattern."""

import pytest
from pydantic import ValidationError

from openbapo.arrange import strums
from openbapo.model import STRUM_AUTO, STRUM_NONE


def test_library_has_the_required_breadth() -> None:
    lib = strums.library()
    assert len(lib) >= 10
    assert {p.meter for p in lib} >= {(4, 4), (3, 4), (6, 8)}
    assert {p.kind for p in lib} == {"strum", "arpeggio"}
    for p in lib:
        assert p.mnemonic and p.name and p.events
        assert all(0 <= e.beat < p.meter[0] for e in p.events)


def test_pattern_validation_rejects_bad_events() -> None:
    with pytest.raises(ValidationError, match="past the bar"):
        strums.StrumPattern(
            id="x", name="x", mnemonic="x", meter=(3, 4), kind="strum", bpm=(60.0, 90.0),
            events=(strums.StrumEvent(beat=3.0, stroke="down"),),
        )  # fmt: skip
    with pytest.raises(ValidationError, match="either a stroke"):
        strums.StrumEvent(beat=0.0)
    with pytest.raises(ValidationError, match="either a stroke"):
        strums.StrumEvent(beat=0.0, stroke="down", strings=(1,))


def test_recommendation_by_meter_and_tempo() -> None:
    assert strums.recommend((4, 4), 60).id == "quarter-down"
    assert strums.recommend((4, 4), 100).id == "folk"
    assert strums.recommend((4, 4), 200).id == "alternate"  # nearest range when none contains it
    assert strums.recommend((3, 4), 120).id == "waltz"
    assert strums.recommend((6, 8), 180).id == "six-eight"
    assert strums.recommend((4, 4), None).id == "quarter-down"
    assert strums.recommend((5, 4), 100) is None


def test_resolve_honours_the_arrangement_choice() -> None:
    assert strums.resolve(STRUM_NONE, (4, 4), 100) == (None, None)
    auto, warning = strums.resolve(STRUM_AUTO, (4, 4), 100)
    assert auto is not None and auto.id == "folk" and warning is None
    chosen, warning = strums.resolve("mute", (4, 4), 100)
    assert chosen is not None and chosen.id == "mute" and warning is None
    fallback, warning = strums.resolve("nope", (4, 4), 100)
    assert fallback is not None and fallback.id == "folk" and warning and "不存在" in warning
    wrong_meter, warning = strums.resolve("waltz", (4, 4), 100)
    assert wrong_meter is not None and wrong_meter.id == "folk" and warning and "3/4" in warning
