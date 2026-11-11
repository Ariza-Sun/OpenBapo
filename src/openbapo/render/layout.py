"""Page geometry: where every bar, beat and row of the sheet goes (A4, points).

The reference layout is docs/reference/desperado-sheet.webp: systems of
four bars, each system a strip of chord diagrams over a tab staff over
the numbered-notation (jianpu) line over the lyrics.
"""

from __future__ import annotations

from dataclasses import dataclass

from openbapo.arrange import ArrangedSheet, Bar

PAGE_W, PAGE_H = 595.0, 842.0  # A4 portrait in points
MARGIN = 40.0
HEADER_H = 96.0  # first page only
FOOTER_H = 24.0
BARS_PER_SYSTEM = 4
CLEF_W = 26.0  # room for "TAB" and the time signature at the start of a system
BEAT_PAD = 10.0  # inset of the first beat from the bar line

# Rows inside a system, as y offsets from the system's top.
NAME_Y = 9.0  # chord-name baseline
DIAGRAM_Y = 20.0  # diagram top (the x / o marks sit just above it)
DIAGRAM_W, DIAGRAM_H = 30.0, 35.0
STAFF_Y = 68.0  # top tab line
STRING_GAP = 7.0
STAFF_H = STRING_GAP * 5
STEM_Y = STAFF_Y + STAFF_H + 12.0  # where stems end and beams run
JIANPU_Y = 134.0  # digit baseline
LYRIC_Y = 152.0  # lyric baseline
SYSTEM_H = 164.0

CONTENT_W = PAGE_W - 2 * MARGIN


@dataclass(frozen=True)
class BarBox:
    bar: Bar
    left: float
    width: float
    top: float  # the system's top

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def beat_w(self) -> float:
        return (self.width - 2 * BEAT_PAD) / max(1, self.bar.beats)

    def beat_x(self, beat: float) -> float:
        """x of a position in beats from the downbeat (fractions allowed)."""
        return self.left + BEAT_PAD + beat * self.beat_w

    def time_x(self, t: float) -> float:
        span = self.bar.end - self.bar.start
        if span <= 0:
            return self.beat_x(0.0)
        return self.beat_x((t - self.bar.start) / span * self.bar.beats)


@dataclass(frozen=True)
class System:
    top: float
    bars: tuple[BarBox, ...]


@dataclass(frozen=True)
class Page:
    number: int  # 1-based
    systems: tuple[System, ...]
    header: bool


def systems_per_page(first: bool) -> int:
    free = PAGE_H - 2 * MARGIN - FOOTER_H - (HEADER_H if first else 0)
    return max(1, int(free // SYSTEM_H))


def paginate(sheet: ArrangedSheet) -> list[Page]:
    """Bars -> systems of BARS_PER_SYSTEM -> pages; an empty sheet is one page with no system."""
    bar_w = (CONTENT_W - CLEF_W) / BARS_PER_SYSTEM
    rows = [sheet.bars[i : i + BARS_PER_SYSTEM] for i in range(0, len(sheet.bars), BARS_PER_SYSTEM)]
    pages: list[Page] = []
    index = 0
    while index < len(rows) or not pages:
        first = not pages
        top = MARGIN + (HEADER_H if first else 0)
        count = systems_per_page(first)
        systems = []
        for row in rows[index : index + count]:
            left = MARGIN + CLEF_W
            boxes = tuple(
                BarBox(bar=bar, left=left + n * bar_w, width=bar_w, top=top)
                for n, bar in enumerate(row)
            )
            systems.append(System(top=top, bars=boxes))
            top += SYSTEM_H
        pages.append(Page(number=len(pages) + 1, systems=tuple(systems), header=first))
        index += count
    return pages
