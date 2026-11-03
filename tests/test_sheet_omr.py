"""Layout analysis on a drawn page and the reading checks / assembly."""

import json
from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image, ImageDraw

from opencapo.sheet import layout, omr


def drawn_page(width: int = 1200, height: int = 1600) -> Image.Image:
    """A page with one system: a chord grid, a 6-line staff, a notation row with 3 bars."""
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    staff_top = 600
    for i in range(6):  # staff lines nearly as wide as the page
        y = staff_top + i * 14
        draw.line([(60, y), (width - 60, y)], fill=0, width=2)
    # chord grid above the staff, over the second bar, with a "name" block above it
    draw.rectangle([520, 560, 580, 590], outline=0, width=2)
    for x in range(526, 580, 6):
        draw.line([(x, 560), (x, 590)], fill=0, width=2)
    draw.rectangle([540, 540, 560, 552], fill=0)
    # notation row: bar lines (tall) and digit-like blocks (shorter), lyrics below
    row_top, row_bottom = 760, 810
    for x in (100, 500, 900, 1100):
        draw.line([(x, row_top), (x, row_bottom)], fill=0, width=3)
    for x in (150, 220, 290, 360, 560, 640, 950, 1020):
        draw.rectangle([x, row_top + 8, x + 18, row_bottom - 8], fill=0)
    draw.rectangle([150, 830, 400, 850], fill=0)  # a lyric line
    return image


def test_layout_finds_system_bars_and_chord(tmp_path: Path) -> None:
    image = drawn_page()
    systems = layout.analyze_page(image, 0, tmp_path, "t")
    assert len(systems) == 1
    bars = systems[0].bars
    for bar, (left, right) in zip(bars, [(100, 500), (500, 900), (900, 1100)], strict=True):
        assert abs(bar.left - left) <= 2 and abs(bar.right - right) <= 2
    assert [len(b.chords) for b in bars] == [0, 1, 0]
    assert 0.0 < float(bars[1].chords[0]["x"]) < 0.2  # type: ignore[arg-type]
    assert all(Path(b.crop).exists() and Path(b.stacked).exists() for b in bars)
    stacked = Image.open(bars[1].stacked)
    assert stacked.height > Image.open(bars[1].crop).height


def test_worksheet_from_an_image(tmp_path: Path) -> None:
    page = tmp_path / "page.png"
    drawn_page().save(page)
    sheet = layout.worksheet(page, tmp_path / "work")
    assert (tmp_path / "work" / "worksheet.json").exists()
    pages = sheet["pages"]
    assert isinstance(pages, list) and len(pages) == 1
    assert len(pages[0]["systems"][0]["bars"]) == 3


def test_reading_parsing_and_checks() -> None:
    r = omr.parse_reading(
        3, 'Sure: ```json\n{"chords": "G", "melody": "[0 2] [2. 5,] 2 [5, 6,]"}\n```'
    )
    assert (r.index, r.chords, r.melody) == (3, "G", "[0 2] [2. 5,] 2 [5, 6,]")
    with pytest.raises(omr.SheetError):
        omr.parse_reading(1, "no json here")
    readings = [
        omr.BarReading(0, "C", "1 2 3 4"),
        omr.BarReading(1, "", "- 0 0 0"),  # tie into the bar: fine
        omr.BarReading(2, "H", "1 2 3"),  # bad chord, three beat items in a 4/4 bar
        omr.BarReading(3, "C", "8 - - -"),  # bad digit
        omr.BarReading(4, "G", ""),  # chord only: fine
        omr.BarReading(5, "C", "- 1 1 1"),  # tie after a bar without melody
    ]
    problems = omr.check_readings(readings, 4)
    assert [(p.index, p.problem.split(":")[0]) for p in problems] == [
        (2, "chords"),
        (2, "3 beat items written, bar has 4"),
        (3, "melody"),
        (5, "tie into the bar but the previous bar has no melody"),
    ]


def test_repeated_passages_and_assembly(tmp_path: Path) -> None:
    verse = [("C", "1 1 5 5"), ("F", "6 6 5 -"), ("C", "4 4 3 3"), ("G", "2 2 1 -")]
    readings = [omr.BarReading(i, c, m) for i, (c, m) in enumerate(verse + verse)]
    readings[6] = omr.BarReading(6, "C", "4 4 3, 3")  # the repeat differs in one bar
    flagged = omr.repeated_passage_mismatches(readings)
    assert [p.index for p in flagged] == [6]
    header = {"title": "t", "key": "D", "mode": "major", "capo": "2", "bpm": "100", "time": "4/4"}
    result = omr.assemble(header, readings, 4)
    assert result.text.startswith("title: t\nkey: D\n") and "C          | 1 1 5 5" in result.text
    assert [p.index for p in result.problems] == [6]
    omr.write_review(tmp_path / "review.md", result.problems, readings)
    assert "小节 6" in (tmp_path / "review.md").read_text(encoding="utf-8")
    # chords carry over bars that have none
    carried = omr.assemble(
        header, [omr.BarReading(0, "Am", "6 6 6 6"), omr.BarReading(1, "", "1 - - -")], 4
    )
    assert "Am         | 1 - - -" in carried.text and not carried.problems
    assert json.dumps(header)  # header fields are plain strings
