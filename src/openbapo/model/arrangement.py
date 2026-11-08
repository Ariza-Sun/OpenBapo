"""Arrangement options: the only things a user can adjust on a sheet.

They are parameters applied to an Analysis, never edits to it. Resetting a
sheet means going back to Arrangement().
"""

from typing import Annotated

from pydantic import Field, StrictBool, StrictInt, StrictStr

from openbapo.model._base import Model
from openbapo.model.music import PitchClass

STRUM_AUTO = "auto"  # recommend a pattern from tempo and meter
STRUM_NONE = "none"  # show chord changes only

TRANSPOSE_RANGE = 6


class Arrangement(Model):
    # Singing key relative to the original key, in semitones.
    transpose: Annotated[StrictInt, Field(ge=-TRANSPOSE_RANGE, le=TRANSPOSE_RANGE)] = 0
    # Tonic of the fingering key; its mode follows the song. None = pick the
    # fingering key with the lowest capo.
    fingering_key: PitchClass | None = None
    strum_pattern: Annotated[StrictStr, Field(min_length=1)] = STRUM_AUTO
    simplify_chords: StrictBool = True
    # Pasted lyrics; None = use the recognized lyrics.
    lyrics_text: StrictStr | None = None
