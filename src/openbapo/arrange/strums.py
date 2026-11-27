"""The strum-pattern library and the default pattern for a song."""

from __future__ import annotations

import json
from functools import cache
from importlib import resources
from typing import Annotated, Literal, Self

from pydantic import Field, StrictFloat, StrictInt, StrictStr, model_validator

from openbapo.model import STRUM_AUTO, STRUM_NONE
from openbapo.model._base import Model

Stroke = Literal["down", "up", "mute"]
StringRef = Literal[
    "bass", 1, 2, 3, 4, 5, 6
]  # 1 = the high E string; bass = the chord's root string


class StrumEvent(Model):
    beat: Annotated[StrictFloat, Field(ge=0.0)]  # within the bar, 0 = the downbeat, in beats
    stroke: Stroke | None = None  # strumming
    strings: tuple[StringRef, ...] = ()  # picking (an arpeggio); exclusive with stroke

    @model_validator(mode="after")
    def _one_kind(self) -> Self:
        if (self.stroke is None) == (not self.strings):
            raise ValueError("an event is either a stroke or a set of plucked strings")
        return self


class StrumPattern(Model):
    id: StrictStr
    name: StrictStr
    mnemonic: StrictStr  # what a player says while playing it: "下 下上 上下上"
    meter: tuple[StrictInt, StrictInt]  # (numerator, denominator)
    kind: Literal["strum", "arpeggio"]
    bpm: tuple[StrictFloat, StrictFloat]  # recommended tempo range (beats per minute)
    tags: tuple[StrictStr, ...] = ()
    events: tuple[StrumEvent, ...]

    @model_validator(mode="after")
    def _events_fit_the_bar(self) -> Self:
        beats = [e.beat for e in self.events]
        if beats != sorted(beats) or len(set(beats)) != len(beats):
            raise ValueError(f"{self.id}: events must be in strictly increasing order")
        if beats and beats[-1] >= self.meter[0]:
            raise ValueError(f"{self.id}: an event lies past the bar")
        if self.bpm[0] > self.bpm[1]:
            raise ValueError(f"{self.id}: bpm range is reversed")
        return self


@cache
def library() -> tuple[StrumPattern, ...]:
    text = resources.files("openbapo.arrange").joinpath("data/strums.json").read_text("utf-8")
    patterns = tuple(StrumPattern.model_validate(item) for item in json.loads(text))
    ids = [p.id for p in patterns]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate strum pattern id")
    return patterns


def patterns_for(meter: tuple[int, int]) -> tuple[StrumPattern, ...]:
    return tuple(p for p in library() if p.meter == meter)


def recommend(meter: tuple[int, int], bpm: float | None) -> StrumPattern | None:
    """The default pattern: same meter, tempo inside its range (library order on ties)."""
    candidates = patterns_for(meter)
    if not candidates:
        return None
    if bpm is None:
        return candidates[0]
    inside = [p for p in candidates if p.bpm[0] <= bpm <= p.bpm[1]]
    if inside:
        return inside[0]
    return min(candidates, key=lambda p: abs((p.bpm[0] + p.bpm[1]) / 2 - bpm))


def resolve(
    choice: str, meter: tuple[int, int], bpm: float | None
) -> tuple[StrumPattern | None, str | None]:
    """The pattern an Arrangement asks for, with a warning when it fell back to the default."""
    if choice == STRUM_NONE:
        return None, None
    if choice == STRUM_AUTO:
        return recommend(meter, bpm), None
    chosen = next((p for p in library() if p.id == choice), None)
    if chosen is not None and chosen.meter == meter:
        return chosen, None
    fallback = recommend(meter, bpm)
    reason = "不存在" if chosen is None else f"是 {chosen.meter[0]}/{chosen.meter[1]} 的"
    return fallback, f"节奏型 {choice!r} {reason}，改用默认推荐"
