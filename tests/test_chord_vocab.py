import pytest

from opencapo.analysis.chords import vocab
from opencapo.analysis.chords.vocab import (
    N_LABELS,
    NO_CHORD,
    LabelError,
    bass_relation,
    chord_tones,
    degree_semitones,
    format_label,
    index_label,
    label_index,
    parse_label,
    pitch_class,
)
from opencapo.model import ChordQuality


def test_label_space_size() -> None:
    assert N_LABELS == 14 * 12 + 1
    assert index_label(NO_CHORD) is None
    for i in range(NO_CHORD):
        root, quality = index_label(i)  # type: ignore[misc]
        assert label_index(root, quality) == i
    with pytest.raises(IndexError):
        index_label(N_LABELS)


@pytest.mark.parametrize(
    ("name", "pc"), [("C", 0), ("C#", 1), ("Db", 1), ("E#", 5), ("Fb", 4), ("Cb", 11), ("B#", 0)]
)
def test_pitch_class(name: str, pc: int) -> None:
    assert pitch_class(name) == pc


@pytest.mark.parametrize("name", ["H", "", "c", "C$"])
def test_bad_note_name(name: str) -> None:
    with pytest.raises(LabelError):
        pitch_class(name)


@pytest.mark.parametrize(
    ("degree", "semis"), [("1", 0), ("3", 4), ("b3", 3), ("5", 7), ("#5", 8), ("b7", 10), ("9", 2)]
)
def test_degree_semitones(degree: str, semis: int) -> None:
    assert degree_semitones(degree) == semis


def test_every_quality_has_a_template_with_the_root() -> None:
    for quality in ChordQuality:
        assert 0 in vocab.INTERVALS[quality]
    assert chord_tones(7, ChordQuality.DOM7) == {7, 11, 2, 5}  # G7 = G B D F


@pytest.mark.parametrize(
    ("label", "root", "quality", "bass", "exact"),
    [
        ("C", 0, ChordQuality.MAJ, None, True),
        ("C:maj", 0, ChordQuality.MAJ, None, True),
        ("A:min", 9, ChordQuality.MIN, None, True),
        ("G:7", 7, ChordQuality.DOM7, None, True),
        ("D:maj/3", 2, ChordQuality.MAJ, 6, True),  # D/F#
        ("G/5", 7, ChordQuality.MAJ, 2, True),  # G/D
        ("C:min7/b7", 0, ChordQuality.MIN7, 10, True),
        ("C:maj/1", 0, ChordQuality.MAJ, None, True),  # explicit root bass is root position
        ("E:hdim7", 4, ChordQuality.HDIM7, None, True),
        ("F#:sus4", 6, ChordQuality.SUS4, None, True),
        ("Bb:5", 10, ChordQuality.POWER, None, True),
        ("C:minmaj7", 0, ChordQuality.MIN, None, False),
        ("C:maj9", 0, ChordQuality.MAJ7, None, False),
        ("C:9", 0, ChordQuality.DOM7, None, False),
        ("C:maj(9)", 0, ChordQuality.MAJ, None, False),
        ("C:(1,3,5)", 0, ChordQuality.MAJ, None, True),
        ("C:(1,b3,5,b7)", 0, ChordQuality.MIN7, None, True),
        ("C:(1,3,5,9)", 0, ChordQuality.MAJ, None, False),
        ("C:7(*3)", 0, ChordQuality.DOM7, None, False),
    ],
)
def test_parse_label(
    label: str, root: int, quality: ChordQuality, bass: int | None, exact: bool
) -> None:
    parsed = parse_label(label)
    assert parsed is not None
    assert (parsed.root, parsed.quality, parsed.bass, parsed.exact) == (root, quality, bass, exact)
    assert parsed.index == label_index(root, quality)


@pytest.mark.parametrize("label", ["N", "X", " N ", ""])
def test_no_chord_labels(label: str) -> None:
    assert parse_label(label) is None


@pytest.mark.parametrize("label", ["C:weird", "Q:maj", "C:maj/x"])
def test_bad_labels(label: str) -> None:
    with pytest.raises(LabelError):
        parse_label(label)


def test_format_label_round_trips() -> None:
    for i in range(NO_CHORD):
        root, quality = index_label(i)  # type: ignore[misc]
        parsed = parse_label(format_label(root, quality))
        assert parsed is not None and (parsed.root, parsed.quality) == (root, quality)
    assert format_label(None) == "N"
    assert format_label(2, ChordQuality.MAJ, 6) == "D:maj/3"
    assert format_label(7, ChordQuality.MAJ, 11) == "G:maj/3"
    assert format_label(0, ChordQuality.MIN7, 10) == "C:min7/b7"
    assert format_label(0, ChordQuality.MAJ, 0) == "C:maj"


def test_bass_relation() -> None:
    assert bass_relation(2, ChordQuality.MAJ, 2) == "root"  # D over D
    assert bass_relation(2, ChordQuality.MAJ, 6) == "chord_tone"  # D/F#
    assert bass_relation(2, ChordQuality.MAJ, 9) == "chord_tone"  # D/A
    assert bass_relation(2, ChordQuality.MAJ, 0) == "foreign"  # C is not in D major
    assert bass_relation(0, ChordQuality.MAJ, 0) == "root"  # pitch class 0 is not "false"
