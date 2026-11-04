"""The lead-sheet text format and its rendering."""

import json
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from opencapo.analysis.audio import Audio
from opencapo.analysis.chords.chroma import ChromaTemplateExpert
from opencapo.analysis.chords.stage import ChordStage
from opencapo.analysis.grid import build_grid
from opencapo.model import ChordQuality, Mode
from opencapo.sheet import SheetError, load_sheet, parse_sheet, render, write_reference, write_wav
from opencapo.sheet.format import degree_midi, parse_chord_name, parse_chords, parse_melody
from opencapo.sheet.render import shape_frets, voicing

EXAMPLE = Path(__file__).parents[1] / "docs" / "examples" / "xiaoxingxing.sheet"


def test_parse_example_sheet() -> None:
    sheet = load_sheet(EXAMPLE)
    assert sheet.title == "小星星" and sheet.key == 2 and sheet.capo == 2 and sheet.bpm == 100
    assert sheet.fingering_key == 0 and sheet.beats_per_bar == 4 and len(sheet.bars) == 12
    assert sheet.duration == pytest.approx(12 * 4 * 60 / 100)
    first = sheet.bars[0]
    assert [(c.root, c.quality, c.beats) for c in first.chords] == [(0, ChordQuality.MAJ, 4)]
    assert [(n.degree, n.octave, n.beats) for n in first.notes] == [
        (1, 0, 1), (1, 0, 1), (5, 0, 1), (5, 0, 1)
    ]  # fmt: skip
    second = sheet.bars[1]
    assert [(c.root, c.beats) for c in second.chords] == [(5, 2), (0, 2)]
    assert [(n.degree, n.beats) for n in second.notes] == [(6, 1), (6, 1), (5, 2)]  # the tie


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("C", (0, ChordQuality.MAJ, None)),
        ("Am7", (9, ChordQuality.MIN7, None)),
        ("F#m", (6, ChordQuality.MIN, None)),
        ("Bb", (10, ChordQuality.MAJ, None)),
        ("G7", (7, ChordQuality.DOM7, None)),
        ("Csus4", (0, ChordQuality.SUS4, None)),
        ("C/E", (0, ChordQuality.MAJ, 4)),
        ("Em7b5", (4, ChordQuality.HDIM7, None)),
        ("#Fm", (6, ChordQuality.MIN, None)),  # accidental first, as Chinese sheets print it
        ("bB", (10, ChordQuality.MAJ, None)),
        ("D/#F", (2, ChordQuality.MAJ, 6)),
    ],
)
def test_parse_chord_name(text: str, expected: tuple) -> None:  # type: ignore[type-arg]
    assert parse_chord_name(text) == expected


def test_chord_and_melody_splitting() -> None:
    assert [c.beats for c in parse_chords("C F G Am", 4)] == [1, 1, 1, 1]
    assert [c.beats for c in parse_chords("C(3) G(1)", 4)] == [3, 1]
    assert [c.beats for c in parse_chords("C", 3)] == [3]
    assert parse_chords("", 4) == ()
    with pytest.raises(SheetError):
        parse_chords("C(3) G", 4)
    with pytest.raises(SheetError):
        parse_chords("C(2) G(1)", 4)
    with pytest.raises(SheetError):
        parse_chords("H", 4)
    notes = parse_melody("5 [6 6] 0 -", 4)
    assert [(n.degree, n.beats) for n in notes] == [
        (5, 1),
        (6, Fraction(1, 2)),
        (6, Fraction(1, 2)),
        (None, 2),
    ]
    notes = parse_melody("1' 1,, [1 [2 3]]", 3)
    half, quarter = Fraction(1, 2), Fraction(1, 4)
    assert [(n.degree, n.octave, n.beats) for n in notes] == [
        (1, 1, 1), (1, -2, 1), (1, 0, half), (2, 0, quarter), (3, 0, quarter)
    ]  # fmt: skip
    assert parse_melody("", 4) == ()
    dotted = parse_melody("[0 2] [2. 5,] [2 2] [5, 6,]", 4)
    assert [(n.degree, n.octave, n.beats) for n in dotted[1:4]] == [
        (2, 0, Fraction(1, 2)), (2, 0, Fraction(3, 4)), (5, -1, Fraction(1, 4))
    ]  # fmt: skip
    assert parse_melody("1. 2 3", 4)[0].beats == Fraction(3, 2) * 4 / 3  # 1.5 : 0.5 : 1 of 4
    opening_tie = parse_melody("- 0 0 0", 4)
    assert opening_tie[0].tied and opening_tie[0].degree is None and opening_tie[0].beats == 1
    with pytest.raises(SheetError):
        parse_melody("8", 4)
    with pytest.raises(SheetError):
        parse_melody("[1 2", 4)


def test_header_errors_and_defaults() -> None:
    sheet = parse_sheet("---\nC | 1 2 3 4\n")
    assert (sheet.key, sheet.capo, sheet.bpm, sheet.beats_per_bar, sheet.mode) == (
        0,
        0,
        90.0,
        4,
        Mode.MAJOR,
    )
    with pytest.raises(SheetError):
        parse_sheet("title: x\nC | 1\n")  # no '---'
    with pytest.raises(SheetError):
        parse_sheet("capo: 13\n---\nC | 1\n")
    with pytest.raises(SheetError):
        parse_sheet("---\n")
    minor = parse_sheet("key: A\nmode: minor\ntime: 3/4\n---\nAm | 6, 1 3\n")
    assert minor.mode == Mode.MINOR and minor.beats_per_bar == 3


def test_degree_midi_follows_the_sung_key() -> None:
    sheet = parse_sheet("key: D\ncapo: 2\n---\nC | 1 5 1' 1,\n")
    notes = sheet.bars[0].notes
    assert [degree_midi(sheet, n) for n in notes] == [62, 69, 74, 50]
    assert degree_midi(sheet, parse_melody("0", 1)[0]) is None


def test_render_reference_and_wav(tmp_path: Path) -> None:
    sheet = load_sheet(EXAMPLE)
    rendered = render(sheet)
    ref = rendered.reference
    assert ref["key"] == "D major" and ref["capo"] == 2 and ref["fingering_key"] == 0
    assert len(ref["beats"]) == 48 and len(ref["downbeats"]) == 12
    assert ref["chords"][0] == [0.0, pytest.approx(4 * 60 / 100), "D:maj"]  # C shape, capo 2
    labels = {c[2] for c in ref["chords"]}
    assert labels == {"D:maj", "G:maj", "A:maj"}
    assert ref["chords"][-1][1] == pytest.approx(sheet.duration)
    assert ref["melody"][0][2] == 62 and ref["melody"][2][2] == 69  # 1 and 5 in D
    assert rendered.samples.shape[0] == 2 and rendered.samples.shape[1] > sheet.duration * 44100
    assert 0.5 < float(np.abs(rendered.samples).max()) <= 0.9
    again = render(sheet)
    np.testing.assert_array_equal(again.samples, rendered.samples)  # deterministic
    write_wav(tmp_path / "x.wav", rendered.samples, rendered.sample_rate)
    with wave.open(str(tmp_path / "x.wav")) as f:
        assert f.getnchannels() == 2 and f.getframerate() == 44100
        assert f.getnframes() == rendered.samples.shape[1]
    write_reference(tmp_path / "x.json", ref)
    assert json.loads((tmp_path / "x.json").read_text(encoding="utf-8"))["title"] == "小星星"


def test_render_without_bass_or_melody_is_quieter() -> None:
    sheet = parse_sheet("key: C\n---\nC | 1 - - -\nG | 5 - - -\n")
    full = render(sheet)
    chords_only = render(sheet, bass=False, melody=False)
    # Every render is peak-normalised, so compare shape, not loudness.
    assert chords_only.samples.shape == full.samples.shape
    assert not np.array_equal(chords_only.samples, full.samples)
    assert render(sheet, melody=False).reference["melody"] == full.reference["melody"]
    strummed = render(sheet, accompaniment="strum", melody_voice="voice")
    without = lambda ref: {k: v for k, v in ref.items() if k != "render"}  # noqa: E731
    assert without(strummed.reference) == without(full.reference)  # the timing is sound-independent
    assert strummed.reference["render"]["melody_voice"] == "voice"  # ... but the settings are kept
    assert not np.array_equal(strummed.samples, full.samples)
    with pytest.raises(ValueError):
        render(sheet, accompaniment="organ")


def test_chroma_expert_hears_the_rendered_sheet() -> None:
    sheet = load_sheet(EXAMPLE)
    rendered = render(sheet)
    audio = Audio(samples=rendered.samples, sample_rate=rendered.sample_rate)
    grid = build_grid(rendered.reference["beats"], rendered.reference["downbeats"])
    result = ChordStage([ChromaTemplateExpert()]).run(audio, None, grid.beats, sheet.duration)
    assert result.key is not None and (result.key.tonic, result.key.mode) == (2, Mode.MAJOR)
    in_key = sum(c.end - c.start for c in result.chords if c.root in (2, 7, 9))
    assert in_key / sheet.duration > 0.7


def test_tie_across_a_bar_line_holds_one_note() -> None:
    sheet = parse_sheet("key: C\nbpm: 120\n---\nC | 1 2 3 6,\nC | - 0 0 0\nC | 1 1 1 1\n")
    held = sheet.bars[1].notes[0]
    assert held.tied and (held.degree, held.octave, held.beats) == (6, -1, 1)
    ref = render(sheet, bass=False).reference["melody"]
    assert ref[3] == [1.5, 2.5, 57]  # 6, lasts into the next bar as one note
    assert len(ref) == 4 + 3 + 4
    with pytest.raises(SheetError):
        parse_sheet("---\nC | - 1 1 1\n")


def test_strum_voicings_follow_the_written_shape_under_the_capo() -> None:
    # C shape with the capo on 4 sounds E major from the fifth string up, as on a guitar.
    assert voicing(0, "maj", 4) == [52, 56, 59, 64, 68]  # E3 G#3 B3 E4 G#4
    assert voicing(5, "maj", 4) == [45, 52, 57, 61, 64, 69]  # F barre -> A major, A2 at the bottom
    assert voicing(9, "min", 0) == [45, 52, 57, 60, 64]  # open Am
    assert (
        shape_frets(11, "maj") == (X, 2, 4, 4, 4, 2) if (X := None) is None else True
    )  # B: A-form barre at 2
    assert shape_frets(6, "min") == (2, 4, 4, 2, 2, 2)  # F#m: E-form barre at 2
    assert voicing(0, "maj", 0, bass=4)[0] == 40  # C/E puts E2 on the bottom
    assert all(p >= 40 for p in voicing(3, "7", 0))  # Eb7 via a barre, nothing below E2
