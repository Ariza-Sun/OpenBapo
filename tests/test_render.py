"""Rendering: pagination, the SVG pages, PNG / PDF export and MusicXML."""

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
    sheet = arrange(SONG)
    pages = render_svg(sheet)
    assert len(pages) == 1
    svg = pages[0]
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert ">t<" in svg and "原调 C · 演唱调 C · C 调指法 无变调夹" in svg and "1=C  4/4" in svg
    assert "♩= 120" in svg and "节奏型：民谣扫弦  下 下上 上下上" in svg
    for name in ("C", "G", "F"):
        assert f">{name}<" in svg
    assert re.search(r'fill="#888">Am\?<', svg)  # Am has confidence 0.3 -> grey with a ?
    assert ">Am<" not in svg
    assert "有 1 个和弦可能不准" in svg
    assert svg.count("<circle") > 10  # diagram dots
    # the folk pattern: 6 strokes per bar, each an arrow through the strings
    assert svg.count("<path") == 5 * 6
    assert "1 / 1" in svg


def test_rendering_follows_the_arrangement() -> None:
    base = render_svg(arrange(SONG))[0]
    g_shapes = render_svg(arrange(SONG, Arrangement(fingering_key=7)))[0]
    assert ">G<" in g_shapes and ">Em?<" in g_shapes and ">Am" not in g_shapes
    assert "G 调指法 夹 5 品" in g_shapes
    assert base.count("<path") == g_shapes.count("<path")  # same strokes in another key
    quarter = render_svg(arrange(SONG, Arrangement(strum_pattern="quarter-down")))[0]
    assert quarter.count("<path") == 5 * 4
    none = render_svg(arrange(SONG, Arrangement(strum_pattern="none")))[0]
    assert none.count("<path") == 6  # one strum per chord change
    assert "节奏型：" not in none
    mute = render_svg(arrange(SONG, Arrangement(strum_pattern="mute")))[0]
    assert mute.count('r="1.1"') == 5 * 2  # the two muted strokes per bar carry a dot
    arpeggio = render_svg(arrange(SONG, Arrangement(strum_pattern="arp-53231323")))[0]
    assert "<path" not in arpeggio and arpeggio.count(">×<") >= 5 * 8  # one x per plucked string
    full = render_svg(arrange(SONG, Arrangement(simplify_chords=False)))[0]
    assert ">G/B<" in full and ">Fmaj7<" in full and ">G7<" in full


def test_svg_escapes_text() -> None:
    song = SONG.model_copy(update={"title": "A <b> & c"})
    assert "A &lt;b&gt; &amp; c" in render_svg(arrange(song))[0]


def test_png_and_pdf_export(tmp_path: Path) -> None:
    pytest.importorskip("cairosvg")
    from openbapo.render import render_pdf, render_png

    sheet = arrange(long_song(20))
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
    assert [s.figure for s in symbols] == ["D", "A", "Bm", "G", "A", "A"]  # sounding, capo applied
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
