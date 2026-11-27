"""Page geometry: where every bar, beat and row of the sheet goes (A4, points).

The layout follows Chinese play-along sheets: systems of
four bars, each system a strip of chord diagrams over a tab staff over
the lyrics. Bars in a system may differ in width, as on printed sheets:
each is at least as wide as its own lyrics and chord diagrams need. The
numbered-notation (jianpu) melody row of such sheets is not drawn.
"""

from __future__ import annotations

from dataclasses import dataclass

from openbapo.arrange import ArrangedSheet, Bar
from openbapo.render.text import LYRIC_SIZE, line_width

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
DIAGRAM_W, DIAGRAM_H = 23.0, 26.0
DIAGRAM_GAP = 8.0  # between diagrams; room for a base-fret number on the left
DIAGRAM_MIN_SCALE = 0.6
STAFF_Y = 60.0  # top tab line
STRING_GAP = 5.5
STAFF_H = STRING_GAP * 5
STEM_Y = STAFF_Y + STAFF_H + 12.0  # where stems end and beams run
LYRIC_Y = STEM_Y + 16.0  # lyric baseline, right under the rhythm stems
SYSTEM_H = 126.5

CONTENT_W = PAGE_W - 2 * MARGIN
ROW_W = CONTENT_W - CLEF_W  # the bars of a system share this
LYRIC_PAD = 2.0  # between a bar's lyrics and its bar lines
# No bar narrower than this share of a bar in a full, even system, so the
# strum pattern and the diagrams stay readable.
MIN_BAR_SHARE = 0.8


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


def bar_need(sheet: ArrangedSheet, bar: Bar) -> float:
    """The narrowest the bar may be: its lyrics side by side at full size, and
    its chord diagrams at their smallest."""
    texts = [s.text for s in sheet.lyrics if bar.start <= s.start < bar.end]
    lyrics = line_width(texts, LYRIC_SIZE) + 2 * LYRIC_PAD if texts else 0.0
    diagrams = len(bar.chords) * (DIAGRAM_W + DIAGRAM_GAP) * DIAGRAM_MIN_SCALE + 2.0
    return max(lyrics, diagrams)


def share_width(needs: list[float], total: float, floor: float) -> list[float] | None:
    """Widths summing to `total`: a bar that needs more than an even share gets
    what it needs, the others share the rest evenly. None if they do not fit."""
    widths = [max(need, floor) for need in needs]
    if sum(widths) > total + 1e-6:
        return None
    fixed: set[int] = set()
    while True:
        free = [i for i in range(len(widths)) if i not in fixed]
        share = (total - sum(widths[i] for i in fixed)) / len(free)
        wider = [i for i in free if widths[i] > share]
        if not wider:
            for i in free:
                widths[i] = share
            return widths
        fixed.update(wider)


def rows(sheet: ArrangedSheet) -> list[tuple[tuple[Bar, ...], tuple[float, ...]]]:
    """Bars grouped into systems, with each bar's width.

    BARS_PER_SYSTEM bars when they fit (see `share_width`; a bar never narrower
    than MIN_BAR_SHARE of an even share), else 3, 2, then 1, as printed sheets
    hold fewer bars where the lyrics are dense. Lyrics
    are then never set smaller nor pushed out from under their own bar; only
    a single bar too dense even alone is set smaller. The song's last bars keep
    the even width when they fit in it.
    """
    floor = MIN_BAR_SHARE * ROW_W / BARS_PER_SYSTEM
    needs = [bar_need(sheet, bar) for bar in sheet.bars]
    out: list[tuple[tuple[Bar, ...], tuple[float, ...]]] = []
    i = 0
    while i < len(sheet.bars):
        for n in range(BARS_PER_SYSTEM, 0, -1):
            row = sheet.bars[i : i + n]
            need = needs[i : i + len(row)]
            widths = None
            if len(row) < n:  # the song ends here
                widths = share_width(need, len(row) * ROW_W / BARS_PER_SYSTEM, floor)
            widths = widths or share_width(need, ROW_W, floor)
            if widths or n == 1:
                break
        out.append((row, tuple(widths or (ROW_W,))))
        i += len(row)
    return out


def paginate(sheet: ArrangedSheet) -> list[Page]:
    """Bars -> systems (see `rows`) -> pages; an empty sheet is one page with no system."""
    grouped = rows(sheet)
    pages: list[Page] = []
    index = 0
    while index < len(grouped) or not pages:
        first = not pages
        top = MARGIN + (HEADER_H if first else 0)
        count = systems_per_page(first)
        systems = []
        for row, widths in grouped[index : index + count]:
            left = MARGIN + CLEF_W
            boxes = []
            for bar, width in zip(row, widths, strict=True):
                boxes.append(BarBox(bar=bar, left=left, width=width, top=top))
                left += width
            systems.append(System(top=top, bars=tuple(boxes)))
            top += SYSTEM_H
        pages.append(Page(number=len(pages) + 1, systems=tuple(systems), header=first))
        index += count
    return pages
