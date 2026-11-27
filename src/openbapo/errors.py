"""Errors shared by the CLI and the analysis pipeline."""

import traceback
from pathlib import Path


class InputError(Exception):
    """A problem with the user's input, reported as a one-line message."""


def where(exc: BaseException, frames: int = 4) -> str:
    """For logs: the exception's type and its innermost frames, e.g. "ValueError at
    pipeline.py:131 < main.py:280". Never the message, which can quote user input."""
    stack = traceback.extract_tb(exc.__traceback__)[-frames:]
    places = " < ".join(f"{Path(f.filename).name}:{f.lineno}" for f in reversed(stack))
    return f"{type(exc).__name__} at {places}" if places else type(exc).__name__
