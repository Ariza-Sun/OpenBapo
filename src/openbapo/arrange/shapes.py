"""The chord-shape library and the default choice of shape for a chord."""

from __future__ import annotations

import json
from functools import cache
from importlib import resources
from typing import Annotated

from pydantic import Field, StrictInt, StrictStr

from openbapo.model import ChordQuality
from openbapo.model._base import Model

TUNING = (40, 45, 50, 55, 59, 64)  # E A D G B E, MIDI


class Shape(Model):
    """One way to finger a chord, as a chord diagram shows it."""

    root: Annotated[StrictInt, Field(ge=0, le=11)]
    quality: ChordQuality
    frets: tuple[Annotated[StrictInt, Field(ge=0, le=24)] | None, ...]  # low string first; None = X
    base_fret: Annotated[StrictInt, Field(ge=1)]  # the diagram's first fret
    barre: Annotated[StrictInt, Field(ge=0, le=5)] | None  # string the barre starts on, low = 0
    difficulty: Annotated[StrictInt, Field(ge=1, le=4)]
    source: StrictStr  # "open" or the movable template it came from

    @property
    def pitch_classes(self) -> set[int]:
        return {(o + f) % 12 for o, f in zip(TUNING, self.frets, strict=True) if f is not None}


@cache
def library() -> tuple[Shape, ...]:
    text = resources.files("openbapo.arrange").joinpath("data/shapes.json").read_text("utf-8")
    return tuple(Shape.model_validate(item) for item in json.loads(text))


def shapes_for(root: int, quality: ChordQuality) -> tuple[Shape, ...]:
    return tuple(s for s in library() if s.root == root and s.quality == quality)


def missing() -> list[tuple[int, ChordQuality]]:
    """(root, quality) pairs the library has no shape for; empty when it is complete."""
    have = {(s.root, s.quality) for s in library()}
    return [(r, q) for r in range(12) for q in ChordQuality if (r, q) not in have]


def choose_shape(root: int, quality: ChordQuality, previous_base_fret: int | None = None) -> Shape:
    """The default shape: easiest first, then nearest the previous chord's position, then lowest."""
    candidates = shapes_for(root, quality)
    if not candidates:
        raise KeyError(f"no shape for root {root} {quality}")
    anchor = previous_base_fret if previous_base_fret is not None else 1
    return min(candidates, key=lambda s: (s.difficulty, abs(s.base_fret - anchor), s.base_fret))
