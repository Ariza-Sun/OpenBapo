"""Drawing the sheet as SVG pages."""

from __future__ import annotations

import itertools
from xml.sax.saxutils import escape

from openbapo.arrange import ArrangedSheet, Shape, StrumEvent, StrumPattern
from openbapo.model import TICKS_PER_BEAT, Mode
from openbapo.render import layout as L
from openbapo.render.layout import BarBox, Page, System

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
    if shape.barre is not None:
        barred = [f for f in shape.frets[shape.barre :] if f]
        if barred:
            row = min(barred) - shape.base_fret
            by = nut + (row + 0.5) * gap_y
            out.append(_line(x + shape.barre * gap_x, by, x + 5 * gap_x, by, 3.2))
    for i, fret in enumerate(shape.frets):
        if fret:
            row = fret - shape.base_fret
            cy = nut + (row + 0.5) * gap_y
            out.append(f'<circle cx="{x + i * gap_x:.1f}" cy="{cy:.1f}" r="2.3" fill="{INK}"/>')
    return out


def _sounded(shape: Shape) -> list[int]:
    return [i for i, f in enumerate(shape.frets) if f is not None]


def _string_index(ref: object, shape: Shape) -> int | None:
    """A strum event's string reference -> index into the shape (0 = low E)."""
    if ref == "bass":
        sounded = _sounded(shape)
        return sounded[0] if sounded else None
    if isinstance(ref, int):
        index = 6 - ref
        return index if shape.frets[index] is not None else None
    return None


def _string_y(box: BarBox, index: int) -> float:
    """y of a string line; index 0 (low E) is the bottom line, as tab is written."""
    return box.top + L.STAFF_Y + (5 - index) * L.STRING_GAP


def _stem(out: list[str], box: BarBox, x: float, lowest_index: int) -> None:
    out.append(_line(x, _string_y(box, lowest_index) + 2, x, box.top + L.STEM_Y, 0.8))


def _event_marks(out: list[str], box: BarBox, x: float, event: StrumEvent, shape: Shape) -> None:
    if event.stroke is not None:
        strings = _sounded(shape)
        if not strings:
            return
        for i in strings:
            out.append(_text(x, _string_y(box, i) + 2.3, "×", 7))
        _stem(out, box, x, strings[0])
        label = {"down": "↓", "up": "↑", "mute": "切"}[event.stroke]
        out.append(_text(x, box.top + L.STAFF_Y - 3, label, 6.5 if event.stroke != "mute" else 5))
        return
    picked = [i for i in (_string_index(r, shape) for r in event.strings) if i is not None]
    if not picked:
        return
    for i in picked:
        fret = shape.frets[i]
        y = _string_y(box, i)
        out.append(
            f'<rect x="{x - 3.2:.1f}" y="{y - 3.4:.1f}" width="6.4" height="6.8" fill="white"/>'
        )
        out.append(_text(x, y + 2.4, str(fret), 6.5))
    _stem(out, box, x, min(picked))


def _beams(out: list[str], box: BarBox, beats: list[float]) -> None:
    """Join consecutive events half a beat apart with a beam at the stem ends."""
    y = box.top + L.STEM_Y
    for a, b in itertools.pairwise(beats):
        if abs((b - a) - 0.5) < 1e-6 and int(a) == int(b - 1e-9):
            out.append(_line(box.beat_x(a), y, box.beat_x(b), y, 2.2))


def tab_bar(
    out: list[str], box: BarBox, pattern: StrumPattern | None, shapes: dict[str, Shape]
) -> None:
    """One bar of the tab line: the strum pattern under each chord, or chord changes only."""
    chords = box.bar.chords
    if not chords:
        return
    if pattern is None:
        for c in chords:
            shape = shapes[c.shape]
            strings = _sounded(shape)
            if strings:
                x = box.beat_x(float(c.beat))
                for i in strings:
                    out.append(_text(x, _string_y(box, i) + 2.3, "×", 7))
                _stem(out, box, x, strings[0])
        return
    # Each event sounds the chord in force at its beat.
    starts = [float(c.beat) for c in chords]
    beats_drawn: list[float] = []
    for event in pattern.events:
        if event.beat >= box.bar.beats:
            continue
        current = chords[max(i for i, s in enumerate(starts) if s <= event.beat or i == 0)]
        _event_marks(out, box, box.beat_x(event.beat), event, shapes[current.shape])
        beats_drawn.append(event.beat)
    _beams(out, box, beats_drawn)


def chord_strip(out: list[str], box: BarBox, shapes: dict[str, Shape]) -> None:
    for c in box.bar.chords:
        x = box.beat_x(float(c.beat))
        name = c.name + ("?" if c.low_confidence else "")
        color = LOW if c.low_confidence else INK
        out.append(_text(x + L.DIAGRAM_W / 2, box.top + L.NAME_Y, name, 8, fill=color))
        out.extend(chord_diagram(x, box.top + L.DIAGRAM_Y, shapes[c.shape]))


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


def lyric_bar(out: list[str], box: BarBox, sheet: ArrangedSheet) -> None:
    y = box.top + L.LYRIC_Y
    for syl in sheet.lyrics:
        if box.bar.start <= syl.start < box.bar.end:
            out.append(_text(box.time_x(syl.start), y, syl.text, 7.5))


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
        out.append(_line(box.right, jianpu_top, box.right, jianpu_bottom, 0.6))
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
        out.append(_text(L.MARGIN, top + 20, f"♩= {round(h.bpm)}", 11, anchor="start"))
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
            tab_bar(out, box, sheet.strum, sheet.shapes)
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
