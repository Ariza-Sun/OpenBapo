"""Drawing the sheet as SVG pages."""

from __future__ import annotations

from xml.sax.saxutils import escape

from openbapo.arrange import ArrangedSheet, Shape, StrumPattern
from openbapo.model import TICKS_PER_BEAT, Mode
from openbapo.render import layout as L
from openbapo.render.layout import BarBox, Page, System
from openbapo.render.rhythm import TabSymbol, beam_marks, tab_symbols

FONT = "Noto Sans CJK SC, Noto Sans SC, PingFang SC, Microsoft YaHei, Helvetica, Arial, sans-serif"
INK = "#111"
GREY = "#777"
LOW = "#888"
TUNING = (40, 45, 50, 55, 59, 64)
DEGREES = {0: "1", 2: "2", 4: "3", 5: "4", 7: "5", 9: "6", 11: "7"}


def _text(
    x: float,
    y: float,
    s: str,
    size: float,
    *,
    anchor: str = "middle",
    weight: str = "normal",
    fill: str = INK,
    family: str = FONT,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-family="{family}" '
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
        out.append(_text(x - 3, nut + gap_y * 0.5 + 2.5, str(shape.base_fret), 6, anchor="end"))
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
                    _sustain_mark(out, box, x)
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


def chord_strip(out: list[str], box: BarBox, shapes: dict[str, Shape]) -> None:
    for c in box.bar.chords:
        x = box.beat_x(float(c.beat))
        name = c.name + ("?" if c.low_confidence else "")
        color = LOW if c.low_confidence else INK
        out.append(_text(x, box.top + L.NAME_Y, name, 8, fill=color))
        out.extend(chord_diagram(x - L.DIAGRAM_W / 2, box.top + L.DIAGRAM_Y, shapes[c.shape]))


def jianpu_bar(
    out: list[str], box: BarBox, sheet: ArrangedSheet, first_tick: int, tonic: int
) -> None:
    """Digits of the sung notes that fall in this bar (one digit per note, no rhythm marks yet)."""
    y = box.top + L.JIANPU_Y
    last_tick = first_tick + box.bar.beats * TICKS_PER_BEAT
    for note in sheet.notes:
        if not first_tick <= note.start_tick < last_tick:
            continue
        beat = (note.start_tick - first_tick) / TICKS_PER_BEAT
        x = box.beat_x(beat)
        interval = (note.pitch - tonic) % 12
        degree = DEGREES.get(interval)
        if degree is None:
            degree = "#" + DEGREES[interval - 1]
        out.append(_text(x, y, degree, 9))
        octave = (note.pitch - tonic - interval) // 12 - 5
        for k in range(abs(octave)):
            dy = -(9 + 2.5 * k) if octave > 0 else 2.5 + 2.5 * k
            out.append(f'<circle cx="{x:.1f}" cy="{y + dy:.1f}" r="0.8" fill="{INK}"/>')


LYRIC_SIZE = 7.5
LYRIC_MIN_SIZE = 5.0


def _text_width(s: str, size: float) -> float:
    """Rough advance width: CJK glyphs are square, Latin ones about half as wide."""
    return sum(size if ord(c) > 0x2E7F else 0.55 * size for c in s)


def lyric_bar(out: list[str], box: BarBox, sheet: ArrangedSheet) -> None:
    """Each syllable centred under its start; crowded bars shrink the text rather than overlap."""
    syllables = [s for s in sheet.lyrics if box.bar.start <= s.start < box.bar.end]
    if not syllables:
        return
    wanted = [box.time_x(s.start) for s in syllables]
    room = box.right - box.left - 2.0
    natural = sum(_text_width(s.text, LYRIC_SIZE) + 1.0 for s in syllables)
    size = max(LYRIC_MIN_SIZE, LYRIC_SIZE * min(1.0, room / natural))
    widths = [_text_width(s.text, size) for s in syllables]
    xs: list[float] = []
    edge = box.left + 1.0
    for x, w in zip(wanted, widths, strict=True):
        x = max(x, edge + w / 2)
        xs.append(x)
        edge = x + w / 2 + 1.0
    # Pushed past the bar line: slide the run back left as far as the bar allows.
    overflow = edge - 1.0 - (box.right - 1.0)
    if overflow > 0:
        xs = [x - overflow for x in xs]
        floor = box.left + 1.0 + widths[0] / 2
        if xs[0] < floor:
            xs = [x + floor - xs[0] for x in xs]
    y = box.top + L.LYRIC_Y
    for s, x in zip(syllables, xs, strict=True):
        out.append(_text(x, y, s.text, round(size, 1)))


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
    jianpu_top, jianpu_bottom = system.top + L.JIANPU_Y - 9, system.top + L.JIANPU_Y + 3
    out.append(_line(left, staff_top, left, staff_bottom, 0.8))
    for box in system.bars:
        out.append(_line(box.right, staff_top, box.right, staff_bottom, 0.8))
        if sheet.notes:
            out.append(_line(box.right, jianpu_top, box.right, jianpu_bottom, 0.6))
        if box.bar.beats < num:
            out.append(
                _text(
                    box.right - 3,
                    staff_bottom + 22,
                    f"不完全小节 · {box.bar.beats} 拍",
                    6,
                    anchor="end",
                    fill=GREY,
                )
            )
    if sheet.notes:
        out.append(_line(first.left, jianpu_top, first.left, jianpu_bottom, 0.6))
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
    out.append(_text(L.PAGE_W / 2, top + 26, h.title or "", 18, weight="bold"))
    if h.bpm:
        out.append(
            _text(
                L.MARGIN,
                top + 20,
                f"{ {2: '𝅗𝅥', 4: '♩', 8: '♪', 16: '𝅘𝅥𝅯'}[h.time_signature[1]] }= {round(h.bpm)}",
                11,
                anchor="start",
            )
        )
    num, den = h.time_signature
    out.append(_text(L.MARGIN, top + 44, f"{h.jianpu_key}  {num}/{den}", 11, anchor="start"))
    if h.artist:
        out.append(_text(L.PAGE_W - L.MARGIN, top + 20, f"{h.artist} 演唱", 10, anchor="end"))
    out.append(_text(L.PAGE_W - L.MARGIN, top + 44, h.text, 9, anchor="end"))
    if sheet.strum is not None:
        s = sheet.strum
        out.append(
            _text(L.PAGE_W - L.MARGIN, top + 62, f"节奏型：{s.name}  {s.mnemonic}", 9, anchor="end")
        )
    if sheet.low_confidence_count:
        out.append(
            _text(
                L.MARGIN,
                top + 62,
                f"有 {sheet.low_confidence_count} 个和弦可能不准，已用 ? 标出",
                8,
                anchor="start",
                fill=GREY,
            )
        )


def render_page(page: Page, sheet: ArrangedSheet, total: int) -> str:
    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{L.PAGE_W:.0f}pt" height="{L.PAGE_H:.0f}pt" '
        f'viewBox="0 0 {L.PAGE_W:.0f} {L.PAGE_H:.0f}">',
        f'<rect width="{L.PAGE_W:.0f}" height="{L.PAGE_H:.0f}" fill="white"/>',
    ]
    if page.header:
        header(out, sheet)
    tonic = _sung_tonic(sheet)
    tick = _first_tick(sheet, page)
    for system in page.systems:
        system_lines(out, system, sheet)
        for box in system.bars:
            chord_strip(out, box, sheet.shapes)
            tab_bar(out, box, sheet.strum, sheet.shapes, sheet.header.time_signature)
            jianpu_bar(out, box, sheet, tick, tonic)
            lyric_bar(out, box, sheet)
            tick += box.bar.beats * TICKS_PER_BEAT
    out.append(
        _text(L.PAGE_W / 2, L.PAGE_H - L.MARGIN / 2, f"{page.number} / {total}", 8, fill=GREY)
    )
    out.append("</svg>")
    return "\n".join(out)


def _sung_tonic(sheet: ArrangedSheet) -> int:
    """MIDI-independent tonic pitch class of the sung key's major (jianpu 1)."""
    from openbapo.arrange.keys import parse_key_name, relative_major

    tonic, mode = parse_key_name(sheet.header.sung_key)
    return relative_major(tonic, mode) if mode == Mode.MINOR else tonic


def _first_tick(sheet: ArrangedSheet, page: Page) -> int:
    """Tick count at the start of the page's first bar (bars before it, in order)."""
    if not page.systems:
        return 0
    first_index = page.systems[0].bars[0].bar.index
    return sum(b.beats * TICKS_PER_BEAT for b in sheet.bars if b.index < first_index)


def render_svg(sheet: ArrangedSheet) -> list[str]:
    """One SVG document per page."""
    pages = L.paginate(sheet)
    return [render_page(p, sheet, len(pages)) for p in pages]
