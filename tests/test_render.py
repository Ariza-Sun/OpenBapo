"""Rendering: pagination, the SVG pages, PNG / PDF export and MusicXML."""

import itertools
import re
from pathlib import Path

import pytest
from test_arrange_sheet import SONG, beats, chord

from openbapo.arrange import ArrangedSheet, arrange
from openbapo.model import Analysis, Arrangement, ChordQuality, Key, Mode, Tempo, TimeSignature
from openbapo.render import BARS_PER_SYSTEM, paginate, render_musicxml, render_svg
from openbapo.render import layout as L

ROOTS = (0, 5, 7, 9)
QUALITIES = (ChordQuality.MAJ, ChordQuality.MAJ, ChordQuality.DOM7, ChordQuality.MIN)


def long_song(bars: int) -> Analysis:
    chords = tuple(
        chord(b * 2.0, b * 2.0 + 2.0, ROOTS[b % 4], QUALITIES[b % 4]) for b in range(bars)
    )
    return Analysis(
        title="long",
        duration=bars * 2.0,
        beats=beats(bars),
        tempo=Tempo(bpm=120.0),
        time_signature=TimeSignature(numerator=4, denominator=4),
        key=Key(tonic=0, mode=Mode.MAJOR, confidence=0.9),
        chords=chords,
    )


def test_pagination_fills_systems_and_pages() -> None:
    sheet = arrange(long_song(37))
    pages = paginate(sheet)
    systems = [s for p in pages for s in p.systems]
    assert len(systems) == 10  # 37 bars -> 9 full systems + 1 of one bar
    assert [len(s.bars) for s in systems][-1] == 37 % BARS_PER_SYSTEM
    assert pages[0].header and not pages[1].header
    assert len(pages[0].systems) == L.systems_per_page(True)
    assert all(len(p.systems) <= L.systems_per_page(False) for p in pages[1:])
    box = systems[0].bars[1]
    xs = [box.beat_x(float(k)) for k in range(4)]
    assert xs == sorted(xs) and box.left < xs[0] < xs[-1] < box.right
    assert box.time_x(box.bar.start) == pytest.approx(box.beat_x(0.0))
    assert box.beat_x(2.0) == pytest.approx(box.time_x((box.bar.start + box.bar.end) / 2))
    empty = paginate(arrange(Analysis()))
    assert len(empty) == 1 and empty[0].systems == ()


def test_svg_pages_show_the_sheet() -> None:
    sheet = arrange(SONG, Arrangement(strum_pattern="folk"))
    pages = render_svg(sheet)
    assert len(pages) == 1
    svg = pages[0]
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert ">t<" in svg and ">Key: C<" in svg and ">No capo<" in svg and ">4/4<" in svg
    assert "♩= 120" in svg and "1=C" not in svg
    assert "民谣扫弦" not in svg  # the strum is drawn on the staff, not named in the header
    for name in ("C", "G", "F"):
        assert f">{name}<" in svg
    assert re.search(r'fill="#888">Am\?<', svg)  # Am has confidence 0.3 -> grey with a ?
    assert ">Am<" not in svg
    assert "可能不准" not in svg  # the grey "?" on the chord is the only mark
    assert svg.count("<circle") > 10  # diagram dots
    # Tied continuations add written symbols but never an extra stroke.
    # Six strokes per bar plus the tied-over (not played) one on beat three.
    assert svg.count('class="stroke-direction"') == 5 * 7
    assert svg.count('class="stroke-ghost"') == 5
    assert svg.count('class="rhythm-tie"') == 5
    assert "1 / 1" in svg


def test_rendering_follows_the_arrangement() -> None:
    base = render_svg(arrange(SONG))[0]
    g_shapes = render_svg(arrange(SONG, Arrangement(fingering_key=7)))[0]
    assert ">G<" in g_shapes and ">Em?<" in g_shapes and ">Am" not in g_shapes
    assert ">Capo 5<" in g_shapes and ">G shapes<" in g_shapes
    assert base.count('data-attack="true"') == g_shapes.count('data-attack="true"')
    quarter = render_svg(arrange(SONG, Arrangement(strum_pattern="quarter-down")))[0]
    assert quarter.count('data-attack="true"') == 5 * 4
    none = render_svg(arrange(SONG, Arrangement(strum_pattern="none")))[0]
    assert none.count('data-attack="true"') == 6  # one mark per chord change
    assert "民谣扫弦" not in none
    mute = render_svg(arrange(SONG, Arrangement(strum_pattern="mute")))[0]
    assert mute.count('class="stroke-mute"') == 5 * 2
    arpeggio = render_svg(arrange(SONG, Arrangement(strum_pattern="arp-53231323")))[0]
    assert 'class="stroke-direction"' not in arpeggio
    assert arpeggio.count('class="tab-notehead"') == 5 * 8
    full = render_svg(arrange(SONG, Arrangement(simplify_chords=False)))[0]
    assert ">G/B<" in full and ">Fmaj7<" in full and ">G7<" in full


def test_svg_escapes_text() -> None:
    song = SONG.model_copy(update={"title": "A <b> & c"})
    assert "A &lt;b&gt; &amp; c" in render_svg(arrange(song))[0]


def test_png_and_pdf_export(tmp_path: Path) -> None:
    pytest.importorskip("cairosvg")
    from openbapo.render import render_pdf, render_png

    sheet = arrange(long_song(40))
    png = render_png(sheet, dpi=72)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 1000
    pdf = render_pdf(sheet)
    assert pdf.startswith(b"%PDF")
    from pypdf import PdfReader

    (tmp_path / "x.pdf").write_bytes(pdf)
    reader = PdfReader(tmp_path / "x.pdf")
    assert len(reader.pages) == len(render_svg(sheet)) == 2


def test_musicxml_reads_back_with_music21() -> None:
    music21 = pytest.importorskip("music21")
    sheet = arrange(SONG, Arrangement(transpose=2, fingering_key=0))  # D sung with C shapes, capo 2
    xml = render_musicxml(sheet)
    assert xml.startswith('<?xml version="1.0"') and "score-partwise" in xml
    score = music21.converter.parse(xml, format="musicxml")
    assert [p.partName for p in score.parts] == ["Chords", "Guitar"]
    assert len(score.parts[0].getElementsByClass("Measure")) == 5
    symbols = list(score.parts[0].recurse().getElementsByClass("ChordSymbol"))
    # Sounding, capo applied; the A held into the next bar is written again there.
    assert [s.figure for s in symbols] == ["D", "A", "Bm", "G", "A", "A"]
    tab = list(score.parts[1].recurse().notes)
    assert tab and all(len(n.pitches) == len(n.articulations) // 2 for n in tab[:3])
    strings = [a for a in tab[0].articulations if type(a).__name__ == "StringIndication"]
    frets = [a for a in tab[0].articulations if type(a).__name__ == "FretIndication"]
    assert [s.number for s in strings] == [5, 4, 3, 2, 1] and [f.number for f in frets] == [
        3,
        2,
        0,
        1,
        0,
    ]
    assert sorted(p.midi for p in tab[0].pitches) == [48, 52, 55, 60, 64]  # C shape as fingered


def test_musicxml_without_strum_and_empty_sheet() -> None:
    xml = render_musicxml(arrange(SONG, Arrangement(strum_pattern="none")))
    assert xml.count("<measure") == 10 and "<harmony>" in xml
    bare = render_musicxml(arrange(Analysis(title="x")))
    assert "<part id=" in bare and "<measure" not in bare


def test_render_cli(tmp_path: Path) -> None:
    pytest.importorskip("cairosvg")
    from openbapo.cli import main

    sheet_path = tmp_path / "sheet.json"
    sheet_path.write_text(arrange(SONG).to_json(), encoding="utf-8")
    svg_dir, png, pdf, xml = (
        tmp_path / "svg",
        tmp_path / "a.png",
        tmp_path / "a.pdf",
        tmp_path / "a.musicxml",
    )
    args = [
        "render",
        str(sheet_path),
        "--svg",
        str(svg_dir),
        "--png",
        str(png),
        "--pdf",
        str(pdf),
        "--musicxml",
        str(xml),
        "--dpi",
        "72",
    ]
    assert main(args) == 0
    assert (svg_dir / "page-1.svg").exists() and png.stat().st_size > 1000
    assert pdf.read_bytes().startswith(b"%PDF") and "<harmony>" in xml.read_text(encoding="utf-8")
    assert main(["render", str(sheet_path)]) == 1  # nothing to write
    assert main(["render", str(tmp_path / "missing.json"), "--png", str(png)]) == 1
    (tmp_path / "bad.json").write_text("{}", encoding="utf-8")
    assert main(["render", str(tmp_path / "bad.json"), "--png", str(png)]) == 1


def test_sheet_json_round_trip_keeps_rendering_identical() -> None:
    sheet = arrange(SONG)
    again = ArrangedSheet.from_json(sheet.to_json())
    assert render_svg(again) == render_svg(sheet)


def sung(song: Analysis, lines: list[tuple[float, float, str]]) -> Analysis:
    """`song` with lyrics: each line's syllables spread evenly over (start, end)."""
    from openbapo.lyrics import parse_lyrics
    from openbapo.model import LyricSyllable

    syllables = []
    for index, ((start, end, _text), line) in enumerate(
        zip(lines, parse_lyrics("\n".join(t for *_, t in lines)), strict=True)
    ):
        parts = [p for token in line.tokens for p in token.syllables]
        step = (end - start) / len(parts)
        syllables += [
            LyricSyllable(
                start=start + k * step, end=start + (k + 1) * step, text=p, line_index=index
            )
            for k, p in enumerate(parts)
        ]
    return song.model_copy(update={"lyrics": tuple(syllables)})


def lyric_texts(svg: str) -> list[tuple[float, float, str]]:
    """(x, font size, text) of every text on the lyric baseline."""
    y = f"{L.HEADER_H + L.MARGIN + L.LYRIC_Y:.1f}"
    return [
        (float(x), float(size), text)
        for x, yy, size, text in re.findall(
            r'<text x="([\d.]+)" y="([\d.]+)" font-size="([\d.]+)"[^>]*>([^<]*)</text>', svg
        )
        if yy == y
    ]


def all_lyric_rows(svg: str, systems: int) -> list[list[tuple[float, float, str]]]:
    """(x, font size, text) on each system's lyric baseline, row by row."""
    found = re.findall(
        r'<text x="([\d.]+)" y="([\d.]+)" font-size="([\d.]+)"[^>]*>([^<]*)</text>', svg
    )
    rows = []
    for k in range(systems):
        y = f"{L.HEADER_H + L.MARGIN + k * L.SYSTEM_H + L.LYRIC_Y:.1f}"
        rows.append([(float(x), float(size), text) for x, yy, size, text in found if yy == y])
    return rows


def test_lyrics_sit_under_their_bars_without_overlap() -> None:
    song = sung(
        SONG,
        [
            (0.0, 2.0, "我爱你"),
            (2.0, 2.5, "一闪一闪亮晶晶满天都是小星星挂在天上放光明好像许多小眼睛"),
            (4.0, 6.0, "beautiful day"),
        ],
    )
    sheet = arrange(song)
    page = paginate(sheet)[0]
    rows = all_lyric_rows(render_svg(sheet)[0], len(page.systems))
    texts = [t for row in rows for t in row]
    assert [t for *_, t in texts][:3] == ["我", "爱", "你"]
    assert [t for *_, t in texts][-4:] == ["beau-", "ti-", "ful", "day"]
    assert len(texts[3:-4]) == 28
    # The 28-syllable bar is set wide enough for its lyrics; its neighbour shares the row.
    first = page.systems[0].bars
    assert len(first) == 2 and first[1].width > 2 * first[0].width
    from openbapo.render.text import lyric_gap, text_width

    for row in rows:
        for (x1, s1, a), (x2, s2, b) in itertools.pairwise(row):
            assert (
                x1 + text_width(a, s1) / 2 + lyric_gap(a, b, s1)
                <= x2 - text_width(b, s2) / 2 + 0.11  # SVG coordinates are written to 0.1 pt
            )


def test_a_row_too_dense_for_the_system_is_set_smaller_but_never_overlaps() -> None:
    from openbapo.render.text import LYRIC_SIZE, lyric_positions
    from openbapo.render.text import lyric_gap as _lyric_gap
    from openbapo.render.text import text_width as _text_width

    words = ("friends on deck when I'm cash- in my checks laid up on the rent " * 3).split()
    xs, size = lyric_positions([100.0] * len(words), words, 60.0, 300.0)
    assert size < LYRIC_SIZE
    for (x1, a), (x2, b) in itertools.pairwise(zip(xs, words, strict=True)):
        assert (
            x1 + _text_width(a, size) / 2 + _lyric_gap(a, b, size)
            <= x2 - _text_width(b, size) / 2 + 0.01
        )
    assert xs[0] - _text_width(words[0], size) / 2 >= 60.0 - 0.01
    assert xs[-1] + _text_width(words[-1], size) / 2 <= 300.0 + 0.01


def test_musicxml_has_one_text_per_lyric_line() -> None:
    music21 = pytest.importorskip("music21")
    song = sung(SONG, [(0.0, 2.0, "我爱你"), (5.0, 7.0, "beautiful day")])
    score = music21.converter.parse(render_musicxml(arrange(song)), format="musicxml")
    words = [
        (m.number, e.getOffsetBySite(m), e.content)
        for m in score.parts[0].getElementsByClass("Measure")
        for e in m.getElementsByClass("TextExpression")
    ]
    assert words == [(1, 0.0, "我爱你"), (3, 2.0, "beautiful day")]


def test_every_chord_is_written_and_crowded_diagrams_do_not_overlap() -> None:
    from openbapo.render.svg import diagram_layout

    sheet = arrange(SONG)
    svg = render_svg(sheet)[0]
    assert svg.count('class="chord-diagram"') == sum(len(b.chords) for b in sheet.bars)
    box = paginate(sheet)[0].systems[0].bars[0]
    xs, scale = diagram_layout(box, [0.0, 1.0, 2.0, 3.0])
    assert scale < 1.0
    width = L.DIAGRAM_W * scale
    assert all(b - a >= width for a, b in itertools.pairwise(xs))
    assert xs[0] - width / 2 >= box.left and xs[-1] + width / 2 <= box.right


def test_every_strum_pattern_has_a_thumbnail() -> None:
    from openbapo.arrange.strums import library
    from openbapo.render import render_strum_thumbnail

    for pattern in library():
        svg = render_strum_thumbnail(pattern)
        assert svg.startswith("<svg") and svg.endswith("</svg>")
        drawn = svg.count('class="stroke-direction"') + svg.count('class="tab-notehead"')
        assert drawn >= len(pattern.events), pattern.id


def test_dense_lyrics_get_fewer_wider_bars_not_smaller_text() -> None:
    from openbapo.render.text import LYRIC_SIZE

    # One long line per 2 s bar: four of them do not fit a row at full size.
    line = "Something's changing inside you and don't you know"
    song = sung(long_song(8), [(b * 2.0, b * 2.0 + 1.9, line) for b in range(8)])
    sheet = arrange(song)
    systems = [s for p in paginate(sheet) for s in p.systems]
    assert all(len(s.bars) < BARS_PER_SYSTEM for s in systems)
    assert systems[0].bars[-1].right == pytest.approx(L.MARGIN + L.CONTENT_W)
    assert {size for _, size, _ in lyric_texts(render_svg(sheet)[0])} == {LYRIC_SIZE}
    # Sparse lyrics keep the usual four bars a row.
    sparse = sung(long_song(8), [(b * 2.0, b * 2.0 + 1.0, "la") for b in range(8)])
    assert [len(s.bars) for p in paginate(arrange(sparse)) for s in p.systems] == [4, 4]


def test_every_rendered_syllable_sits_under_its_own_bar() -> None:
    import random

    from openbapo.render.text import LYRIC_SIZE

    rng = random.Random(7)
    words = [
        "talk",
        "to",
        "me",
        "softly",
        "there's",
        "something",
        "in",
        "your",
        "eyes",
        "我",
        "知道",
        "你",
        "才是",
        "这世界上",
    ]
    lines = []
    for b in range(24):  # 2 s bars, from empty to very crowded
        n = rng.choice([0, 1, 3, 6, 10, 16])
        if n:
            start = b * 2.0 + rng.uniform(0.0, 1.0)
            lines.append((start, start + rng.uniform(0.3, 0.9), " ".join(rng.choices(words, k=n))))
    sheet = arrange(sung(long_song(24), lines))
    pages = paginate(sheet)
    for number, (page, svg) in enumerate(zip(pages, render_svg(sheet), strict=True)):
        top = L.MARGIN + (L.HEADER_H if number == 0 else 0)
        found = re.findall(
            r'<text x="([\d.]+)" y="([\d.]+)" font-size="([\d.]+)"[^>]*>([^<]*)</text>', svg
        )
        for k, system in enumerate(page.systems):
            y = f"{top + k * L.SYSTEM_H + L.LYRIC_Y:.1f}"
            drawn = [(float(x), float(size)) for x, yy, size, _ in found if yy == y]
            owners = [
                box
                for box in system.bars
                for s in sheet.lyrics
                if box.bar.start <= s.start < box.bar.end
            ]
            assert len(drawn) == len(owners)
            for (x, size), box in zip(drawn, owners, strict=True):
                assert box.left - 0.05 <= x <= box.right + 0.05
                assert size == LYRIC_SIZE or len(system.bars) == 1


def test_bars_in_a_row_share_its_width_by_need() -> None:
    from openbapo.render.text import LYRIC_SIZE

    # Bar 2 of each four is crowded; the rest are sparse.
    lines = [
        (b * 2.0, b * 2.0 + 1.5, "you know I still love you baby" if b % 4 == 1 else "la")
        for b in range(8)
    ]
    sheet = arrange(sung(long_song(9), lines))
    systems = [s for p in paginate(sheet) for s in p.systems]
    floor = L.MIN_BAR_SHARE * L.ROW_W / L.BARS_PER_SYSTEM
    for system in systems[:-1]:
        widths = [box.width for box in system.bars]
        assert sum(widths) == pytest.approx(L.ROW_W)
        assert min(widths) >= floor - 1e-6
        lefts = [box.left for box in system.bars]
        assert all(
            b.right == pytest.approx(c) for b, c in zip(system.bars, lefts[1:], strict=False)
        )
    # Four bars still fit a row: the crowded one is wider, the others narrower.
    first = systems[0].bars
    assert len(first) == 4 and first[1].width > first[0].width == pytest.approx(first[2].width)
    sizes = {s for row in all_lyric_rows(render_svg(sheet)[0], len(systems)) for _, s, _ in row}
    assert sizes == {LYRIC_SIZE}
    # The song's last, short row keeps the even width.
    assert systems[-1].bars[-1].width == pytest.approx(L.ROW_W / L.BARS_PER_SYSTEM)


def test_short_bars_are_labelled_pickup_or_with_their_meter() -> None:
    from test_arrange_sheet import beats as make_beats

    full = make_beats(6)
    # Bar 0 keeps its last 2 beats (a pickup); bar 3 loses its last beat.
    kept = [b for b in full if not (b.bar_index == 0 and b.beat_in_bar <= 2)]
    kept = [b for b in kept if not (b.bar_index == 3 and b.beat_in_bar == 4)]
    song = long_song(6).model_copy(update={"beats": tuple(kept)})
    svg = render_svg(arrange(song))[0]
    assert ">Pickup<" in svg and ">3/4<" in svg and "不完全小节" not in svg
