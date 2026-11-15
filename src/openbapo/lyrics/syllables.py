"""English syllables for display: pyphen's hyphenation points.

The aligner times whole words; a word's syllables split its time span. Only
their starts reach the sheet, so typesetting hyphenation is precise enough
(it keeps short words such as "awake" whole). pyphen is used under its MPL
1.1 option.
"""

from functools import cache
from itertools import pairwise

import pyphen


@cache
def _dictionary() -> pyphen.Pyphen:
    return pyphen.Pyphen(lang="en_US")


def split_word(word: str) -> tuple[str, ...]:
    """Syllables of `word`; every one but the last ends with "-"."""
    positions = _dictionary().positions(word) if word.isalpha() else []
    cuts = [0, *positions, len(word)]
    parts = [word[a:b] for a, b in pairwise(cuts)]
    return (*(p + "-" for p in parts[:-1]), parts[-1])
