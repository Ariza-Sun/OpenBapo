"""Layout analysis on a drawn page and the reading checks / assembly."""

import json
from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image, ImageDraw

from opencapo.sheet import layout, omr


def drawn_page(width: int = 1200, height: int = 1600, systems: int = 1) -> Image.Image:
    """A page with one or two systems: a chord grid, a 6-line staff, a notation row with 3 bars."""
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    for n in range(systems):
        staff_top = 600 + n * 500
        for i in range(6):  # staff lines nearly as wide as the page
            y = staff_top + i * 14
            draw.line([(60, y), (width - 60, y)], fill=0, width=2)
        # chord grid above the staff, over the second bar, with a "name" block above it
        draw.rectangle([520, staff_top - 40, 580, staff_top - 10], outline=0, width=2)
        for x in range(526, 580, 6):
            draw.line([(x, staff_top - 40), (x, staff_top - 10)], fill=0, width=2)
        draw.rectangle([540, staff_top - 60, 560, staff_top - 48], fill=0)
        # notation row: bar lines (tall) and digit-like blocks (shorter), lyrics below
        row_top, row_bottom = staff_top + 160, staff_top + 210
        for x in (100, 500, 900, 1100):
            draw.line([(x, row_top), (x, row_bottom)], fill=0, width=3)
        for x in (150, 220, 290, 360, 560, 640, 950, 1020):
            draw.rectangle([x, row_top + 8, x + 18, row_bottom - 8], fill=0)
        draw.rectangle([150, row_bottom + 20, 400, row_bottom + 40], fill=0)  # a lyric line
    return image


def test_layout_finds_systems_and_writes_row_images(tmp_path: Path) -> None:
    image = drawn_page(systems=2)
    systems = layout.analyze_page(image, 0, tmp_path, "t", start=3)
    assert [s.index for s in systems] == [3, 4]
    first, second = systems
    assert abs(first.staff.top - 600) <= 2 and abs(second.staff.top - 1100) <= 2
    assert abs(first.notation.top - 768) <= 2 and abs(first.notation.bottom - 802) <= 2
    assert first.top <= 540 and first.bottom < second.diagram_top  # names, staff, digits, lyrics
    assert first.bottom > 830 and second.bottom > 1330  # the lyrics are inside the row
    for s in systems:
        crop = Image.open(s.crop)
        assert crop.size == (1200, s.bottom - s.top) and Path(s.crop).name.endswith(".png")


def test_worksheet_from_an_image(tmp_path: Path) -> None:
    page = tmp_path / "page.png"
    drawn_page().save(page)
    sheet = layout.worksheet(page, tmp_path / "work")
    assert (tmp_path / "work" / "worksheet.json").exists()
    pages = sheet["pages"]
    assert isinstance(pages, list) and len(pages) == 1
    rows = pages[0]["systems"]
    assert len(rows) == 1 and rows[0]["index"] == 0 and Path(rows[0]["crop"]).exists()
    assert Path(pages[0]["header_crop"]).exists()


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
    flagged = omr.repeated_passage_mismatches(readings, window=4)
    assert [(p.index, p.problem.split(",")[0]) for p in flagged] == [(6, "differs from bar 2")]
    # a passage that differs in most bars is another passage, not a misread
    other = [omr.BarReading(i, c, "5 5 5 5") for i, (c, _) in enumerate(verse, start=8)]
    assert omr.repeated_passage_mismatches(readings + other, window=4) == flagged
    # bars of rests carry no melody and never take part
    rests = [omr.BarReading(i, c, "0 0 0 0") for i, (c, _) in enumerate(verse, start=12)]
    assert omr.repeated_passage_mismatches(readings + rests, window=4) == flagged
    header = {"title": "t", "key": "D", "mode": "major", "capo": "2", "bpm": "100", "time": "4/4"}
    long_verse = [*verse, ("Am", "6 6 6 6"), ("F", "4 4 4 4"), ("C", "1 1 1 1"), ("G", "5 5 5 5")]
    readings = [omr.BarReading(i, c, m) for i, (c, m) in enumerate(long_verse + long_verse)]
    readings[10] = omr.BarReading(10, "C", "4 4 3, 3")  # the repeat differs in one bar of eight
    result = omr.assemble(header, readings, 4)
    assert result.text.startswith("title: t\nkey: D\n") and "C          | 1 1 5 5" in result.text
    assert [p.index for p in result.problems] == [10]
    omr.write_review(tmp_path / "review.md", result.problems, readings)
    assert "小节 10" in (tmp_path / "review.md").read_text(encoding="utf-8")
    # chords carry over bars that have none
    carried = omr.assemble(
        header, [omr.BarReading(0, "Am", "6 6 6 6"), omr.BarReading(1, "", "1 - - -")], 4
    )
    assert "Am         | 1 - - -" in carried.text and not carried.problems
    assert json.dumps(header)  # header fields are plain strings


def test_reconcile_header_fills_the_sung_key_and_reports_conflicts() -> None:
    header: dict[str, object] = {"key": None, "play_key": "C", "capo": 4}
    assert omr.reconcile_header(header) == [] and header["key"] == "E"
    header = {"key": "E", "play_key": "C", "capo": 4}
    assert omr.reconcile_header(header) == []
    header = {"key": "C", "play_key": "C", "capo": 4}  # the fingering key taken for the sung key
    assert omr.reconcile_header(header) == ["页眉不一致：选调 C + 变调夹 4 品 ≠ 原调 C"]
    header = {"key": "D", "play_key": "C", "capo": None}  # capo printed as a range: implied
    assert omr.reconcile_header(header) == [] and header["capo"] == 2
    header = {"key": "F", "play_key": None, "capo": 5}  # nothing to check against
    assert omr.reconcile_header(header) == [] and header["key"] == "F"
    assert omr.reconcile_header({"key": "H"}) == ["页眉的调名读不懂：not a note name: 'H'"]


def test_split_bars_are_joined_and_a_pickup_is_padded() -> None:
    bars = [
        omr.BarReading(0, "", "[0 5,]", row=0),  # pickup: one beat
        omr.BarReading(1, "C", "1 2 3 4", row=0),
        omr.BarReading(2, "G", "5 5", row=0),  # cut by the line break ...
        omr.BarReading(3, "", "6 6", row=1),  # ... and finished on the next row
        omr.BarReading(4, "F", "4 -", row=1),
        omr.BarReading(5, "G(2)", "2 2", row=2),  # both halves carry a chord (one with a count)
        omr.BarReading(6, "C", "1 1 1 1", row=2),
    ]
    joined = omr.join_split_bars(bars, 4)
    assert [(b.index, b.chords, b.melody, b.row) for b in joined] == [
        (0, "", "0 0 0 [0 5,]", 0),
        (1, "C", "1 2 3 4", 0),
        (2, "G", "5 5 6 6", 0),
        (3, "F(2) G(2)", "4 - 2 2", 1),
        (4, "C", "1 1 1 1", 2),
    ]
    header = {"title": "t", "key": "C", "mode": "major", "capo": "0", "bpm": "90", "time": "4/4"}
    result = omr.assemble(header, bars, 4)
    assert result.problems == [] and len(bars) == 5 and "F(2) G(2)  | 4 - 2 2" in result.text
    stale = [omr.BarReading(0, "G(3)(2) Bm7(2)", "1 1 1 1", row=0)]
    assert "G(2) Bm7(2)" in omr.assemble(header, stale, 4).text
    # a short bar inside a row, or not completed by the next row, is left for the checks
    bars = [omr.BarReading(0, "C", "1 2 3 4", row=0), omr.BarReading(1, "C", "1 2", row=0)]
    assert [b.melody for b in omr.join_split_bars(bars, 4)] == ["1 2 3 4", "1 2"]


def test_an_unreadable_melody_keeps_the_bar_chords() -> None:
    header = {"title": "t", "key": "C", "mode": "major", "capo": "0", "bpm": "90", "time": "4/4"}
    bars = [
        omr.BarReading(0, "C", "1 2 3 4", row=0),
        omr.BarReading(1, "G", "1 1 / 2 2", row=0),  # two alternatives written in one bar
    ]
    result = omr.assemble(header, bars, 4)
    assert "\nG\n" in result.text  # chords only, the sheet still parses
    assert [p.index for p in result.problems] == [1] and "melody" in result.problems[0].problem


def test_a_printed_fingering_key_is_told_from_the_chord_shapes() -> None:
    c_shapes = [omr.BarReading(i, c, "") for i, c in enumerate(["C", "Am", "F", "G7", "C", "Em"])]
    assert omr.shape_key(c_shapes) == (0, 1.0)
    assert omr.shape_key([]) == (None, 0.0)
    # "1=C, capo 2" over C shapes: the 1= is the fingering key, the song is in D
    header: dict[str, object] = {"key": "C", "play_key": None, "capo": 2}
    assert omr.reconcile_header(header, c_shapes) == []
    assert header["key"] == "D" and header["play_key"] == "C"
    # "1=D, capo 2" over C shapes: the 1= is the sung key already
    header = {"key": "D", "play_key": None, "capo": 2}
    assert omr.reconcile_header(header, c_shapes) == [] and header["key"] == "D"
    # no capo: nothing to tell apart
    header = {"key": "C", "capo": 0}
    assert omr.reconcile_header(header, c_shapes) == [] and header["key"] == "C"
    # a minor song in A minor shapes under capo 3, printed 1=C: still C shapes -> sung bE
    a_minor = [omr.BarReading(i, c, "") for i, c in enumerate(["Am", "Dm", "E7", "Am", "F", "G"])]
    header = {"key": "C", "capo": 3}
    assert omr.reconcile_header(header, a_minor) == [] and header["key"] == "D#"
