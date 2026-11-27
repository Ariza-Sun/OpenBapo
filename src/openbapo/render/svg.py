"""Drawing the sheet as SVG pages."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction
from xml.sax.saxutils import escape

from openbapo.arrange import ArrangedSheet, Shape, StrumEvent, StrumPattern
from openbapo.render import layout as L
from openbapo.render.layout import BarBox, Page, System
from openbapo.render.rhythm import TabSymbol, beam_marks, strings_for, tab_symbols
from openbapo.render.text import lyric_positions
from openbapo.render.text import text_width as _text_width

FONT = "Noto Serif CJK SC, Noto Serif SC, Source Han Serif SC, Songti SC, SimSun, serif"
INK = "#111"
GREY = "#777"
LOW = "#888"
CHORD_SIZE = 10.0
TUNING = (40, 45, 50, 55, 59, 64)


def _text(
    x: float,
    y: float,
    s: str,
    size: float,
    *,
    anchor: str = "middle",
    weight: str = "normal",
    fill: str = INK,
    family: str | None = None,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-family="{family or FONT}" '
        f'text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{escape(s)}</text>'
    )


def _line(x1: float, y1: float, x2: float, y2: float, width: float = 0.8, color: str = INK) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}" stroke-linecap="round"/>'
    )


def chord_diagram(x: float, y: float, shape: Shape) -> list[str]:
    """A 6-string, 5-fret grid with the shape's dots, x / o marks, barre and base fret."""
    out = []
    gap_x = L.DIAGRAM_W / 5
    gap_y = L.DIAGRAM_H / 5
    nut = y
    for i in range(6):
        out.append(_line(x + i * gap_x, nut, x + i * gap_x, nut + L.DIAGRAM_H, 0.6))
    for j in range(6):
        width = 2.0 if j == 0 and shape.base_fret == 1 else 0.6
        out.append(_line(x, nut + j * gap_y, x + L.DIAGRAM_W, nut + j * gap_y, width))
    if shape.base_fret > 1:
        out.append(_text(x - 3, nut + gap_y * 0.5 + 2.5, str(shape.base_fret), 7, anchor="end"))
    for i, fret in enumerate(shape.frets):
        sx = x + i * gap_x
        if fret is None:
            out.append(_text(sx, nut - 2, "×", 6))
        elif fret == 0:
            out.append(
                f'<circle cx="{sx:.1f}" cy="{nut - 3.5:.1f}" r="1.7" '
                f'fill="none" stroke="{INK}" stroke-width="0.6"/>'
            )
    # A barre is shown the way Chinese sheets show it: a dot on every string it
    # holds down (the frets already list them), no bar drawn across.
    for i, fret in enumerate(shape.frets):
        if fret:
            row = fret - shape.base_fret
            cy = nut + (row + 0.5) * gap_y
            out.append(f'<circle cx="{x + i * gap_x:.1f}" cy="{cy:.1f}" r="2.3" fill="{INK}"/>')
    return out


def _string_y(box: BarBox, index: int) -> float:
    """y of a string line; index 0 (low E) is the bottom line, as tab is written."""
    return box.top + L.STAFF_Y + (5 - index) * L.STRING_GAP


def _stem(out: list[str], box: BarBox, x: float, lowest_index: int) -> None:
    out.append(_line(x, _string_y(box, lowest_index) + 2, x, box.top + L.STEM_Y, 0.8))


def _cross(out: list[str], x: float, y: float) -> None:
    # Clear the staff beneath the mark, like engraved cross noteheads.
    out.append('<g class="tab-notehead">')
    out.append(f'<rect x="{x - 2.3:.1f}" y="{y - 2.3:.1f}" width="4.6" height="4.6" fill="white"/>')
    out.append(_line(x - 1.7, y - 1.7, x + 1.7, y + 1.7, 0.75))
    out.append(_line(x - 1.7, y + 1.7, x + 1.7, y - 1.7, 0.75))
    out.append("</g>")


def _stroke_mark(out: list[str], box: BarBox, x: float, symbol: TabSymbol) -> None:
    """An open arrow within the TAB, as in the local play-along corpus.

    The arrow replaces string crosses; its separate rhythm stem sits below
    the staff. Toward the treble strings (up the page) means a down-strum.
    """
    stroke = symbol.event.stroke if symbol.event is not None else "down"
    high = _string_y(box, max(symbol.strings))
    low = _string_y(box, min(symbol.strings))
    tip, tail = (low, high) if stroke == "up" else (high, low)
    base = tip - 3.4 if stroke == "up" else tip + 3.4
    out.append(
        f'<path class="stroke-direction" data-stroke="{stroke}" '
        f'd="M{x:.1f},{tail:.1f} L{x:.1f},{tip:.1f} '
        f'M{x - 1.9:.1f},{base:.1f} L{x:.1f},{tip:.1f} L{x + 1.9:.1f},{base:.1f}" '
        f'fill="none" stroke="{INK}" stroke-width="0.8" stroke-linecap="round" '
        f'stroke-linejoin="round"/>'
    )
    if stroke == "mute":
        # Corpus sheets place this dot above the arrow, not among the
        # duration dots below the TAB.
        out.append(
            f'<circle class="stroke-mute" cx="{x:.1f}" '
            f'cy="{box.top + L.STAFF_Y - 5:.1f}" r="1.1" fill="{INK}"/>'
        )


def strum_grid(pattern: StrumPattern | None) -> Fraction:
    """The hand's down-up subdivision: sixteenths if the pattern uses them, else eighths."""
    if pattern is not None and any(Fraction(str(e.beat)) * 2 % 1 for e in pattern.events):
        return Fraction(1, 4)
    return Fraction(1, 2)


def ghost_stroke(symbol: TabSymbol, shapes: dict[str, Shape], grid: Fraction) -> TabSymbol:
    """The arrow drawn where a tied sound continues: the stroke the hand passes, not played.

    Play-along sheets keep the strumming hand's motion visible: on the beat
    it moves down, between beats up. The tie arc says the strings are not hit.
    """
    stroke = "down" if (symbol.beat / grid) % 2 == 0 else "up"
    event = StrumEvent(beat=float(symbol.beat), stroke=stroke)
    strings = strings_for(event, shapes[symbol.shape]) if symbol.shape else symbol.strings
    return replace(symbol, event=event, strings=strings or symbol.strings)


def _sustain_mark(out: list[str], box: BarBox, x: float) -> None:
    """A dash extends the previous sound; it is neither a rest nor a stroke."""
    y = box.top + L.STAFF_Y + 2.5 * L.STRING_GAP
    out.append('<g class="rhythm-sustain">')
    out.append(_line(x - 2.5, y, x + 2.5, y, 1.0))
    out.append("</g>")


def _rest_mark(out: list[str], box: BarBox, x: float, symbol: TabSymbol) -> None:
    y = box.top + L.STAFF_Y + 2.5 * L.STRING_GAP
    if symbol.value.kind in ("whole", "half"):
        line_y = box.top + L.STAFF_Y + 2 * L.STRING_GAP
        top = line_y if symbol.value.kind == "whole" else line_y - 2.5
        out.append(f'<rect x="{x - 3:.1f}" y="{top:.1f}" width="6" height="2.5" fill="{INK}"/>')
    elif symbol.value.kind == "quarter":
        out.append(
            f'<path d="M{x - 1:.1f},{y - 6:.1f} l3,4 l-3,4 l2,3 '
            f'q-5,-1 -2,-4 l1,-1 l-2,-3 l3,-3 Z" fill="{INK}"/>'
        )
    else:
        out.append(_line(x + 2, y - 4, x - 1, y + 6, 0.9))
        for level in range(symbol.value.beams):
            yy = y - 3 + level * 3
            out.append(f'<circle cx="{x - 1:.1f}" cy="{yy:.1f}" r="1.5" fill="{INK}"/>')
            out.append(_line(x - 1, yy + 1, x + 2, yy - 1, 0.8))


def _flag(out: list[str], x: float, y: float, level: int) -> None:
    yy = y - level * 3.2
    out.append(
        f'<path class="rhythm-flag" d="M{x:.1f},{yy:.1f} '
        f'C{x + 6:.1f},{yy - 3:.1f} {x + 6:.1f},{yy - 7:.1f} {x + 3:.1f},{yy - 10:.1f}" '
        f'fill="none" stroke="{INK}" stroke-width="1.1"/>'
    )


def tab_bar(
    out: list[str],
    box: BarBox,
    pattern: StrumPattern | None,
    shapes: dict[str, Shape],
    meter: tuple[int, int],
) -> None:
    symbols = tab_symbols(box.bar, pattern, shapes, meter)
    marks = beam_marks(symbols, meter)
    grid = strum_grid(pattern)
    for i, symbol in enumerate(symbols):
        x = box.beat_x(float(symbol.beat))
        y = box.top + L.STEM_Y
        strummed = symbol.event is None or symbol.event.stroke is not None
        out.append(
            f'<g class="tab-symbol" data-beat="{float(symbol.beat):g}" '
            f'data-duration="{float(symbol.length):g}" data-kind="{symbol.value.kind}" '
            f'data-dots="{symbol.value.dots}" data-rest="{str(symbol.rest).lower()}" '
            f'data-attack="{str(not symbol.rest and not symbol.tie_stop).lower()}">'
        )
        if symbol.rest:
            _rest_mark(out, box, x, symbol)
        else:
            if strummed:
                if symbol.tie_stop:
                    out.append('<g class="stroke-ghost">')
                    _stroke_mark(out, box, x, ghost_stroke(symbol, shapes, grid))
                    out.append("</g>")
                else:
                    _stroke_mark(out, box, x, symbol)
                out.append(_line(x, box.top + L.STAFF_Y + L.STAFF_H + 3, x, y, 0.8))
            else:
                for string in symbol.strings:
                    _cross(out, x, _string_y(box, string))
                _stem(out, box, x, min(symbol.strings))
            long_value = symbol.value.kind in ("half", "whole")
            if long_value:
                # Chinese TAB uses a quarter-length attack followed by
                # quarter-length dashes, e.g. arrow — — — for a whole note.
                for quarter in range(1, int(symbol.value.quarters)):
                    beat = float(symbol.beat) + quarter * meter[1] / 4
                    _sustain_mark(out, box, box.beat_x(beat))
            for level, mark in enumerate(marks.get(i, ())):
                yy = y - level * 3.2
                if mark in ("begin", "continue"):
                    end_x = box.beat_x(float(symbols[i + 1].beat))
                elif mark.endswith("hook"):
                    end_x = x + (3.5 if mark == "forward hook" else -3.5)
                else:
                    continue
                out.append(_line(x, yy, end_x, yy, 1.8))
            if i not in marks:
                for level in range(symbol.value.beams):
                    _flag(out, x, y, level)
            if symbol.tie_start:
                end_x = box.beat_x(float(symbols[i + 1].beat))
                tie_y = box.top + L.STAFF_Y - 3
                out.append(
                    f'<path class="rhythm-tie" d="M{x + 1:.1f},{tie_y:.1f} '
                    f'Q{(x + end_x) / 2:.1f},{tie_y - 6:.1f} {end_x - 1:.1f},{tie_y:.1f}" '
                    f'fill="none" stroke="{INK}" stroke-width="0.75"/>'
                )
        # Long sounding values were expanded into quarter-length dashes.
        dots = (
            0 if not symbol.rest and symbol.value.kind in ("half", "whole") else symbol.value.dots
        )
        for dot in range(dots):
            dot_y = box.top + L.STAFF_Y + (2.5 * L.STRING_GAP if symbol.rest else L.STAFF_H + 5)
            out.append(
                f'<circle class="rhythm-dot" cx="{x + 4 + dot * 2.5:.1f}" '
                f'cy="{dot_y:.1f}" r="1" fill="{INK}"/>'
            )
        out.append("</g>")


DIAGRAM_GAP = L.DIAGRAM_GAP
DIAGRAM_MIN_SCALE = L.DIAGRAM_MIN_SCALE


def diagram_layout(box: BarBox, beats: list[float]) -> tuple[list[float], float]:
    """Centres and scale for the bar's diagrams, so that none overlaps the next.

    Each diagram sits over its beat when there is room; crowded ones shrink
    (down to DIAGRAM_MIN_SCALE) and are pushed right, then pulled back inside
    the bar if they would cross its right edge.
    """
    if not beats:
        return [], 1.0
    wanted = [box.beat_x(b) for b in beats]
    room = box.right - box.left - 2.0
    need = len(beats) * (L.DIAGRAM_W + DIAGRAM_GAP)
    scale = max(DIAGRAM_MIN_SCALE, min(1.0, room / need))
    step = (L.DIAGRAM_W + DIAGRAM_GAP) * scale
    half = L.DIAGRAM_W * scale / 2
    xs: list[float] = []
    for x in wanted:
        xs.append(max(x, xs[-1] + step) if xs else max(x, box.left + 1.0 + half))
    overflow = xs[-1] + half - (box.right - 1.0)
    if overflow > 0:
        xs = [x - overflow for x in xs]
        floor = box.left + 1.0 + half
        if xs[0] < floor:
            xs = [x + floor - xs[0] for x in xs]
    return xs, scale


def chord_strip(out: list[str], box: BarBox, shapes: dict[str, Shape]) -> None:
    """Every chord the bar holds, name and diagram, as 小LV吉他 sheets write them.

    The chord is repeated at each bar line and at the start of each line, and
    a re-struck chord inside the bar is written again.
    """
    chords = list(box.bar.chords)
    xs, scale = diagram_layout(box, [float(c.beat) for c in chords])
    for c, x in zip(chords, xs, strict=True):
        name = c.name + ("?" if c.low_confidence else "")
        color = LOW if c.low_confidence else INK
        out.append(
            _text(x, box.top + L.NAME_Y, name, round(CHORD_SIZE * max(scale, 0.8), 1), fill=color)
        )
        top = box.top + L.DIAGRAM_Y
        out.append(
            f'<g class="chord-diagram" transform="translate({x:.1f},{top:.1f}) scale({scale:.3f})">'
        )
        out.extend(chord_diagram(-L.DIAGRAM_W / 2, 0.0, shapes[c.shape]))
        out.append("</g>")


def lyric_row(out: list[str], system: System, sheet: ArrangedSheet) -> None:
    """Each bar's syllables under their own bar; the layout made it wide enough."""
    y = system.top + L.LYRIC_Y
    for box in system.bars:
        placed = [
            (box.time_x(s.start), s.text)
            for s in sheet.lyrics
            if box.bar.start <= s.start < box.bar.end
        ]
        if not placed:
            continue
        xs, size = lyric_positions(
            [x for x, _ in placed],
            [t for _, t in placed],
            box.left + L.LYRIC_PAD,
            box.right - L.LYRIC_PAD,
        )
        for (_, text), x in zip(placed, xs, strict=True):
            out.append(_text(x, y, text, round(size, 1)))


def system_lines(out: list[str], system: System, sheet: ArrangedSheet) -> None:
    """Staff lines, bar lines, the TAB label, the time signature and the bar number."""
    first, last = system.bars[0], system.bars[-1]
    left, right = first.left - L.CLEF_W, last.right
    for i in range(6):
        y = system.top + L.STAFF_Y + i * L.STRING_GAP
        out.append(_line(left, y, right, y, 0.5))
    for ch, dy in zip("TAB", (1.0, 2.5, 4.0), strict=True):
        out.append(_text(left + 6, system.top + L.STAFF_Y + dy * L.STRING_GAP + 2, ch, 7))
    num, den = sheet.header.time_signature
    out.append(
        _text(left + 18, system.top + L.STAFF_Y + 2 * L.STRING_GAP, str(num), 10, weight="bold")
    )
    out.append(
        _text(left + 18, system.top + L.STAFF_Y + 4 * L.STRING_GAP + 1, str(den), 10, weight="bold")
    )
    staff_top, staff_bottom = system.top + L.STAFF_Y, system.top + L.STAFF_Y + L.STAFF_H
    out.append(_line(left, staff_top, left, staff_bottom, 0.8))
    for box in system.bars:
        out.append(_line(box.right, staff_top, box.right, staff_bottom, 0.8))
        if box.bar.beats < num:
            out.append(
                _text(
                    box.right - 3,
                    staff_top - 3,
                    "Pickup" if box.bar.index == 0 else f"{box.bar.beats}/{den}",
                    6,
                    anchor="end",
                    fill=GREY,
                )
            )
    out.append(
        _text(
            first.left + 1,
            system.top + L.STAFF_Y - 4,
            str(first.bar.index + 1),
            6,
            anchor="start",
            fill=GREY,
        )
    )


def header(out: list[str], sheet: ArrangedSheet) -> None:
    h = sheet.header
    top = L.MARGIN
    out.append(_text(L.PAGE_W / 2, top + 26, h.title or "", 21, weight="bold"))
    if h.bpm:
        out.append(
            _text(
                L.MARGIN,
                top + 20,
                f"{ {2: '𝅗𝅥', 4: '♩', 8: '♪', 16: '𝅘𝅥𝅯'}[h.time_signature[1]] }= {round(h.bpm)}",
                12,
                anchor="start",
            )
        )
    num, den = h.time_signature
    out.append(_text(L.MARGIN, top + 44, f"{num}/{den}", 12, anchor="start"))
    y = top + 20
    if h.artist:
        out.append(_text(L.PAGE_W - L.MARGIN, y, f"{h.artist} 演唱", 11, anchor="end"))
        y += 16
    # Key, capo and shapes, one per line at the top right.
    for k, line in enumerate(h.text.split(" · ")):
        out.append(_text(L.PAGE_W - L.MARGIN, y + 14 * k, line, 10.5, anchor="end"))


def render_page(page: Page, sheet: ArrangedSheet, total: int) -> str:
    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{L.PAGE_W:.0f}pt" height="{L.PAGE_H:.0f}pt" '
        f'viewBox="0 0 {L.PAGE_W:.0f} {L.PAGE_H:.0f}">',
        f'<rect width="{L.PAGE_W:.0f}" height="{L.PAGE_H:.0f}" fill="white"/>',
    ]
    if page.header:
        header(out, sheet)
    for system in page.systems:
        system_lines(out, system, sheet)
        lyric_row(out, system, sheet)
        for box in system.bars:
            chord_strip(out, box, sheet.shapes)
            tab_bar(out, box, sheet.strum, sheet.shapes, sheet.header.time_signature)
    out.append(
        _text(L.PAGE_W / 2, L.PAGE_H - L.MARGIN / 2, f"{page.number} / {total}", 8, fill=GREY)
    )
    out.append("</svg>")
    return "\n".join(out)


THUMB_BEAT_W = 25.0  # about the sheet's own beat width, so thumbnails read the same


def _strum_bar(pattern: StrumPattern) -> tuple[list[str], float, float]:
    """One bar of `pattern` on a TAB staff over a C chord: elements, width, height."""
    from openbapo.arrange import Bar, BarChord
    from openbapo.arrange.shapes import choose_shape
    from openbapo.model import ChordQuality

    shape = choose_shape(0, ChordQuality.MAJ)
    beats = pattern.meter[0]
    chord = BarChord(
        beat=0, name="C", original_name="C", shape="C", confidence=1.0, low_confidence=False
    )
    bar = Bar(index=0, start=0.0, end=1.0, beats=beats, chords=(chord,))
    width = 2 * L.BEAT_PAD + beats * THUMB_BEAT_W
    box = BarBox(bar=bar, left=6.0, width=width, top=-L.STAFF_Y + 14.0)
    out: list[str] = []
    for i in range(6):
        y = box.top + L.STAFF_Y + i * L.STRING_GAP
        out.append(_line(box.left, y, box.right, y, 0.5))
    staff_top, staff_bottom = box.top + L.STAFF_Y, box.top + L.STAFF_Y + L.STAFF_H
    for x in (box.left, box.right):
        out.append(_line(x, staff_top, x, staff_bottom, 0.8))
    tab_bar(out, box, pattern, {"C": shape}, pattern.meter)
    return out, box.right + 6.0, box.top + L.STEM_Y + 6.0


def _svg(body: list[str], w: float, h: float) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}pt" height="{h:.0f}pt" '
        f'viewBox="0 0 {w:.1f} {h:.1f}"><rect width="{w:.1f}" height="{h:.1f}" fill="white"/>'
        + "\n".join(body)
        + "</svg>"
    )


def render_strum_thumbnail(pattern: StrumPattern) -> str:
    """One bar of `pattern` on a TAB staff, drawn exactly as on the sheet, over a C chord.

    For choosing a strum by its picture rather than its mnemonic.
    """
    out, w, h = _strum_bar(pattern)
    return _svg(out, w, h)


OVERVIEW_COLUMNS = 3


def render_strum_overview(
    cards: Sequence[tuple[str, StrumPattern]], highlight: int | None = None
) -> str:
    """Numbered cards (a short title after the number, one bar of the pattern),
    `highlight` framed.

    For pickers that hold only text, such as a Discord select menu: the menu
    lists the same numbers and the user picks by the picture.
    """
    bars = [_strum_bar(pattern) for _, pattern in cards]
    card_w = max((w for _, w, _ in bars), default=0.0) + 12.0
    card_h = max((h for _, _, h in bars), default=0.0) + 22.0
    cols = max(1, min(OVERVIEW_COLUMNS, len(cards)))
    rows = -(-len(cards) // cols)
    margin, gap = 8.0, 6.0
    out: list[str] = []
    for k, ((title, _), (body, _, _)) in enumerate(zip(cards, bars, strict=True)):
        x = margin + (k % cols) * (card_w + gap)
        y = margin + (k // cols) * (card_h + gap)
        framed = k == highlight
        out.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{card_w:.1f}" height="{card_h:.1f}" rx="4" '
            f'fill="none" stroke="{INK if framed else "#ccc"}" '
            f'stroke-width="{1.6 if framed else 0.6}"/>'
        )
        label = f"{k + 1}  {title}"
        size = min(11.0, (card_w - 14) / max(_text_width(label, 1.0), 1.0))
        out.append(_text(x + 8, y + 15, label, round(size, 1), anchor="start", weight="bold"))
        out.append(f'<g transform="translate({x + 6:.1f},{y + 16:.1f})">')
        out.extend(body)
        out.append("</g>")
    w = 2 * margin + cols * card_w + (cols - 1) * gap
    h = 2 * margin + rows * card_h + (rows - 1) * gap
    return _svg(out, w, h)


def render_svg(sheet: ArrangedSheet) -> list[str]:
    """One SVG document per page."""
    pages = L.paginate(sheet)
    return [render_page(p, sheet, len(pages)) for p in pages]
