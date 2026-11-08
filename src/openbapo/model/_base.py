from typing import Self

from pydantic import BaseModel, ConfigDict


class Model(BaseModel):
    """Immutable, strictly-shaped base for every serialized object."""

    # Scalar fields use pydantic's Strict* types so nothing is silently coerced
    # ("3" -> 3, true -> 1). Strictness is per field, not model-wide: a model-wide
    # strict mode would also reject JSON arrays for tuples once a model has a
    # "before" validator, because those validate JSON in Python mode.
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str | bytes) -> Self:
        return cls.model_validate_json(data)
