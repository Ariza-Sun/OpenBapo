"""Lyric text measurement, shared by layout (rows that fit) and drawing.

Widths are estimates from the sheet font's advance widths (svg.FONT); CJK
glyphs are 1 em.
"""

from itertools import pairwise

LYRIC_SIZE = 11.0
LYRIC_MIN_SIZE = 7.0
# Advance widths of Noto Serif CJK SC in ems, measured from the font; CJK
# glyphs are 1 em. Unknown characters count as 0.6 em.
_LATIN_EM = dict(
    zip(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'-.,!?()",
        (
            0.557,
            0.638,
            0.538,
            0.629,
            0.547,
            0.388,
            0.566,
            0.663,
            0.332,
            0.311,
            0.607,
            0.334,
            0.974,
            0.661,
            0.596,
            0.638,
            0.607,
            0.462,
            0.473,
            0.367,
            0.653,
            0.548,
            0.838,
            0.564,
            0.552,
            0.498,
            0.718,
            0.67,
            0.691,
            0.77,
            0.652,
            0.629,
            0.741,
            0.853,
            0.405,
            0.403,
            0.731,
            0.625,
            0.975,
            0.796,
            0.767,
            0.641,
            0.767,
            0.715,
            0.567,
            0.659,
            0.797,
            0.714,
            1.053,
            0.7,
            0.687,
            0.605,
            0.557,
            0.471,
            0.558,
            0.557,
            0.555,
            0.558,
            0.558,
            0.549,
            0.558,
            0.562,
            0.196,
            0.346,
            0.327,
            0.327,
            0.308,
            0.432,
            0.364,
            0.364,
        ),
        strict=True,
    )
)


def text_width(s: str, size: float) -> float:
    return size * sum(1.0 if ord(c) > 0x2E7F else _LATIN_EM.get(c, 0.6) for c in s)


def _latin(s: str) -> bool:
    return any(c.isascii() and c.isalnum() for c in s)


def lyric_gap(previous: str, current: str, size: float) -> float:
    """Space between neighbouring syllables: tight inside a word, a word space between words."""
    if previous.endswith("-"):
        return 0.15 * size
    if _latin(previous) or _latin(current):
        return 0.35 * size
    return 0.15 * size


def line_width(texts: list[str], size: float) -> float:
    """Natural width of a row of syllables set side by side at `size`."""
    return sum(text_width(t, size) for t in texts) + sum(
        lyric_gap(a, b, size) for a, b in pairwise(texts)
    )


def lyric_positions(
    wanted: list[float], texts: list[str], left: float, right: float
) -> tuple[list[float], float]:
    """Centres and font size for one row of syllables between `left` and `right`.

    Each syllable goes under its own time when there is room. A crowded run is
    pushed right, then back left from the row's end, so neighbours never
    overlap; a row that does not fit at LYRIC_SIZE is set smaller as a whole.
    """
    room = right - left

    def natural(size: float) -> float:
        return line_width(texts, size)

    size = LYRIC_SIZE
    if natural(size) > room:
        size = max(LYRIC_MIN_SIZE, size * room / natural(size))
        if natural(size) > room:  # dense even at the minimum: fit it, never overlap
            size *= room / natural(size)
    half = [text_width(t, size) / 2 for t in texts]
    gaps = [0.0] + [lyric_gap(a, b, size) for a, b in pairwise(texts)]
    xs: list[float] = []
    for i, x in enumerate(wanted):
        floor = left + half[i] if i == 0 else xs[-1] + half[i - 1] + gaps[i] + half[i]
        xs.append(max(x, floor))
    for i in range(len(xs) - 1, -1, -1):
        ceiling = (
            right - half[i] if i == len(xs) - 1 else xs[i + 1] - half[i + 1] - gaps[i + 1] - half[i]
        )
        xs[i] = min(xs[i], ceiling)
    return xs, size
