"""Read back written durations, attacks and meter, not just a count of SVG paths."""

import re
from fractions import Fraction
from xml.etree import ElementTree as ET

import pytest
from test_arrange_sheet import SONG, beats, chord

from openbapo.arrange import arrange
from openbapo.arrange.strums import StrumEvent, library
from openbapo.model import Arrangement, ChordQuality, TimeSignature
from openbapo.render import layout as L
from openbapo.render import render_musicxml, render_svg
from openbapo.render.rhythm import beam_groups, tab_symbols

NS = {"s": "http://www.w3.org/2000/svg"}
QUARTERS = {
    "whole": 4,
    "half": 2,
    "quarter": 1,
    "eighth": 0.5,
    "16th": 0.25,
    "32nd": 0.125,
    "64th": 0.0625,
}


def one_bar(pattern_id):
    pattern = next(p for p in library() if p.id == pattern_id)
    num, den = pattern.meter
    song = SONG.model_copy(
        update={
            "beats": beats(1, per_bar=num),
            "duration": num * 0.5,
            "chords": (chord(0.0, num * 0.5, 0, ChordQuality.MAJ),),
            "time_signature": TimeSignature(numerator=num, denominator=den),
        }
    )
    return arrange(song, Arrangement(strum_pattern=pattern_id))


def svg_symbols(sheet):
    return ET.fromstring(render_svg(sheet)[0]).findall('.//s:g[@class="tab-symbol"]', NS)


def xml_attacks(sheet):
    root = ET.fromstring(render_musicxml(sheet))
    measure = root.find('./part[@id="P2"]/measure')
    divisions = int(measure.findtext("attributes/divisions"))
    position, attacks = 0, []
    for note in measure.findall("note"):
        if note.find("chord") is not None:
            continue
        if note.find("rest") is None and note.find('./tie[@type="stop"]') is None:
            attacks.append(Fraction(position, divisions))
        position += int(note.findtext("duration"))
    return attacks


@pytest.mark.parametrize("pattern", library(), ids=lambda p: p.id)
def test_every_pattern_fills_its_meter_in_both_formats(pattern):
    sheet = one_bar(pattern.id)
    expected = pattern.meter[0] * 4 / pattern.meter[1]
    svg = svg_symbols(sheet)
    written = sum(
        QUARTERS[s.get("data-kind")] * (1.5 if s.get("data-dots") == "1" else 1) for s in svg
    )
    assert written == expected
    assert [float(s.get("data-beat")) for s in svg if s.get("data-attack") == "true"] == [
        e.beat for e in pattern.events
    ]
    root = ET.fromstring(render_musicxml(sheet))
    for measure in root.findall("./part/measure"):
        divisions = int(measure.findtext("attributes/divisions"))
        sequential = [n for n in measure.findall("note") if n.find("chord") is None]
        assert sum(int(n.findtext("duration")) for n in sequential) / divisions == expected
        for note in sequential:
            kind = note.findtext("type")
            if kind is not None:
                graphical = QUARTERS[kind] * (1.5 if note.find("dot") is not None else 1)
                assert int(note.findtext("duration")) / divisions == graphical
    assert xml_attacks(sheet) == [
        Fraction(str(e.beat)) * Fraction(4, pattern.meter[1]) for e in pattern.events
    ]


def test_folk_syncopation_is_tied_across_beat_three_without_an_extra_attack():
    sheet = one_bar("folk")
    written = svg_symbols(sheet)
    assert [float(s.get("data-duration")) for s in written] == [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    assert [s.get("data-kind") for s in written] == ["quarter"] + ["eighth"] * 6
    assert written[2].find('./s:path[@class="rhythm-tie"]', NS) is not None
    assert written[3].get("data-beat") == "2" and written[3].get("data-attack") == "false"
    assert written[3].find('./s:path[@class="stroke-direction"]', NS) is None
    assert written[3].find('./s:g[@class="rhythm-sustain"]', NS) is not None
    assert written[3].find('./s:g[@class="tab-notehead"]', NS) is None
    assert xml_attacks(sheet) == [0, 1, Fraction(3, 2), Fraction(5, 2), 3, Fraction(7, 2)]
    score = pytest.importorskip("music21").converter.parse(
        render_musicxml(sheet), format="musicxml"
    )
    measure = score.parts[1].measure(1)
    assert measure.duration.quarterLength == 4
    assert [n.tie.type if n.tie else None for n in measure.notes] == [
        None,
        None,
        "start",
        "stop",
        None,
        None,
        None,
    ]


@pytest.mark.parametrize(
    "pattern_id,groups",
    [
        ("eighth-down", [[0, 1], [2, 3], [4, 5], [6, 7]]),
        ("six-eight", [[0, 1, 2], [3, 4, 5]]),
    ],
)
def test_beams_follow_the_meter(pattern_id, groups):
    sheet = one_bar(pattern_id)
    symbols = tab_symbols(sheet.bars[0], sheet.strum, sheet.shapes, sheet.header.time_signature)
    assert beam_groups(symbols, sheet.header.time_signature) == groups
    root = ET.fromstring(render_musicxml(sheet))
    notes = [n for n in root.findall('./part[@id="P2"]/measure/note') if n.find("chord") is None]
    assert [n.findtext("beam") for n in notes] == [
        m for g in groups for m in ["begin", *["continue"] * (len(g) - 2), "end"]
    ]


def test_compound_meter_dots_offsets_and_tempo():
    sheet = one_bar("six-eight-slow")
    bar = sheet.bars[0]
    change = bar.chords[0].model_copy(update={"beat": 3})
    sheet = sheet.model_copy(
        update={"bars": (bar.model_copy(update={"chords": (*bar.chords, change)}),)}
    )
    assert [s.get("data-kind") for s in svg_symbols(sheet)] == ["quarter", "quarter"]
    assert render_svg(sheet)[0].count('class="rhythm-dot"') == 2
    assert "♪= 120" in render_svg(sheet)[0]
    root = ET.fromstring(render_musicxml(sheet))
    assert root.findtext(".//metronome/beat-unit") == "eighth"
    assert root.find(".//sound").get("tempo") == "60.0"
    p1 = root.find('./part[@id="P1"]/measure')
    assert (
        int(p1.findall("harmony")[1].findtext("offset")) / int(p1.findtext("attributes/divisions"))
        == 1.5
    )


def test_isolated_eighth_has_a_flag_and_leading_silence_is_a_rest():
    sheet = one_bar("folk")
    pattern = sheet.strum.model_copy(update={"events": (StrumEvent(beat=0.5, stroke="down"),)})
    sheet = sheet.model_copy(update={"strum": pattern})
    written = svg_symbols(sheet)
    assert written[0].get("data-rest") == "true"
    assert written[1].find('./s:path[@class="rhythm-flag"]', NS) is not None
    assert xml_attacks(sheet) == [Fraction(1, 2)]
    assert sum(float(s.get("data-duration")) for s in written) == 4


def test_upstrokes_and_muted_attacks_keep_the_same_strings_in_xml():
    sheet = one_bar("folk")
    root = ET.fromstring(render_musicxml(sheet))
    notes = root.findall('./part[@id="P2"]/measure/note')
    groups = []
    for note in notes:
        if note.find("chord") is None:
            groups.append([])
        groups[-1].append(note)
    assert [int(n.findtext("notations/technical/string")) for n in groups[2]] == [4, 3, 2, 1]
    root = ET.fromstring(render_musicxml(one_bar("mute")))
    assert not root.findall('./part[@id="P2"]/measure/note/rest')
    assert len(root.findall('./part[@id="P2"]/measure/note/notehead[.="x"]')) == 10


def test_strum_arrows_cross_the_played_strings_and_match_the_rhythm_stem_x():
    """Corpus style: an open staff arrow, no column of crosses or side arrow."""
    sheet = one_bar("folk")
    box = L.paginate(sheet)[0].systems[0].bars[0]
    top = box.top + L.STAFF_Y
    bottom = top + L.STAFF_H
    symbols = svg_symbols(sheet)
    for symbol in symbols:
        assert not symbol.findall('./s:g[@class="tab-notehead"]', NS)
    down = symbols[0].find('./s:path[@class="stroke-direction"]', NS)
    up = symbols[2].find('./s:path[@class="stroke-direction"]', NS)
    for symbol, arrow, low, upward in ((symbols[0], down, 1, True), (symbols[2], up, 2, False)):
        # Read the actual drawn shaft, not just semantic SVG attributes.
        coordinates = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", arrow.get("d"))]
        x, tail, tip_x, tip = coordinates[:4]
        assert x == tip_x == pytest.approx(box.beat_x(float(symbol.get("data-beat"))), abs=0.06)
        assert sorted((tail, tip)) == [top, bottom - low * L.STRING_GAP]
        assert (tip < tail) == upward
        assert arrow.get("fill") == "none"  # thin open head, not a filled triangle
        stem = symbol.find("./s:line", NS)
        assert float(stem.get("x1")) == float(stem.get("x2")) == x
        assert bottom < float(stem.get("y1")) < float(stem.get("y2"))


def test_arpeggios_have_string_crosses_and_mute_dots_sit_above_arrows():
    sheet = one_bar("arp-53231323")
    box = L.paginate(sheet)[0].systems[0].bars[0]
    symbols = svg_symbols(sheet)
    for symbol in symbols:
        assert symbol.find('./s:path[@class="stroke-direction"]', NS) is None
        head = symbol.find('./s:g[@class="tab-notehead"]', NS)
        assert head is not None
        cross = head.find("./s:line", NS)
        cross_y = (float(cross.get("y1")) + float(cross.get("y2"))) / 2
        stem = symbol.find("./s:line", NS)
        assert float(stem.get("y1")) == pytest.approx(cross_y + 2)
        assert float(stem.get("y2")) == box.top + L.STEM_Y
    sheet = one_bar("mute")
    for symbol in svg_symbols(sheet):
        dot = symbol.find('./s:circle[@class="stroke-mute"]', NS)
        arrow = symbol.find('./s:path[@class="stroke-direction"]', NS)
        assert (dot is not None) == (arrow.get("data-stroke") == "mute")
        if dot is not None:
            assert float(dot.get("cy")) < box.top + L.STAFF_Y
            assert symbol.find('./s:circle[@class="rhythm-dot"]', NS) is None


def test_long_chord_changes_use_sustain_dashes_instead_of_extra_strokes():
    sheet = one_bar("quarter-down").model_copy(update={"strum": None})
    (symbol,) = svg_symbols(sheet)
    assert symbol.get("data-duration") == "4"
    assert len(symbol.findall('./s:path[@class="stroke-direction"]', NS)) == 1
    assert len(symbol.findall('./s:g[@class="rhythm-sustain"]', NS)) == 3
    assert symbol.find("./s:ellipse", NS) is None


def test_silence_before_first_chord_does_not_play_a_future_shape():
    sheet = one_bar("eighth-down")
    bar = sheet.bars[0]
    sheet = sheet.model_copy(
        update={
            "bars": (
                bar.model_copy(
                    update={
                        "chords": (bar.chords[0].model_copy(update={"beat": 1}),),
                    }
                ),
            )
        }
    )
    assert xml_attacks(sheet)[0] == 1
    assert all(s.get("data-rest") == "true" for s in svg_symbols(sheet)[:2])


def test_dotted_eighth_and_sixteenth_keep_their_written_lengths():
    sheet = one_bar("folk")
    pattern = sheet.strum.model_copy(
        update={
            "events": tuple(StrumEvent(beat=b, stroke="down") for b in (0.0, 0.75, 1.0, 2.0, 3.0))
        }
    )
    sheet = sheet.model_copy(update={"strum": pattern})
    symbols = svg_symbols(sheet)
    assert [(s.get("data-kind"), s.get("data-dots")) for s in symbols[:2]] == [
        ("eighth", "1"),
        ("16th", "0"),
    ]
    root = ET.fromstring(render_musicxml(sheet))
    notes = [n for n in root.findall('./part[@id="P2"]/measure/note') if n.find("chord") is None]
    assert notes[1].findtext('beam[@number="2"]') == "backward hook"
    assert xml_attacks(sheet) == [0, Fraction(3, 4), 1, 2, 3]


def test_partial_bar_is_labelled_and_exports_its_actual_length():
    sheet = one_bar("folk")
    sheet = sheet.model_copy(update={"bars": (sheet.bars[0].model_copy(update={"beats": 1}),)})
    assert sum(float(s.get("data-duration")) for s in svg_symbols(sheet)) == 1
    assert "不完全小节" in render_svg(sheet)[0]
    root = ET.fromstring(render_musicxml(sheet))
    assert all(m.get("implicit") == "yes" for m in root.findall("./part/measure"))
