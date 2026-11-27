"""Song titles from file names, when the user gave none."""

import re
from pathlib import PurePath


def title_from_filename(name: str) -> str:
    """'Tom_Orlando_-_The_One__feat._Tina_G_.mp3' -> 'Tom Orlando - The One feat. Tina G'."""
    stem = PurePath(name).stem if PurePath(name).suffix else name
    return re.sub(r"\s+", " ", stem.replace("_", " ")).strip() or stem
