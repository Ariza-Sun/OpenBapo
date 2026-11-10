"""Assembling the renderable sheet, and the invariants the roadmap asks for."""

import pytest

from openbapo.arrange import ArrangedSheet, arrange
from openbapo.model import (
    Analysis,
    Arrangement,
    Beat,
    Chord,
    ChordQuality,
    Key,
    Mode,
    Tempo,
    TimeSignature,
)


def beats(bars: int, per_bar: int = 4, period: float = 0.5) -> tuple[Beat, ...]:
    return tuple(
        Beat(
            time=(b * per_bar + i) * period,
            is_downbeat=i == 0,
            bar_index=b,
            beat_in_bar=i + 1,
        )
        for b in range(bars)
        for i in range(per_bar)
    )


def chord(start: float, end: float, root: int, quality: ChordQuality, conf: float = 0.9,
          bass: int | None = None) -> Chord:  # fmt: skip
    return Chord(start=start, end=end, root=root, quality=quality, bass=bass, confidence=conf)


SONG = Analysis(
    title="t",
    duration=10.0,
    beats=beats(5),
    tempo=Tempo(bpm=120.0),
    time_signature=TimeSignature(numerator=4, denominator=4),
    key=Key(tonic=0, mode=Mode.MAJOR, confidence=0.9),
    chords=(
        chord(0.0, 2.0, 0, ChordQuality.MAJ),
        chord(2.0, 4.0, 7, ChordQuality.MAJ, bass=11),  # G/B
        chord(4.0, 6.0, 9, ChordQuality.MIN, conf=0.3),
        chord(6.0, 7.0, 5, ChordQuality.MAJ7),
        chord(7.0, 9.0, 7, ChordQuality.DOM7),  # runs over the bar line into bar 4
    ),
)


def test_sheet_from_a_c_major_song() -> None:
    sheet = arrange(SONG)
    assert sheet.header.text == "原调 C · 演唱调 C · C 调指法 无变调夹"
    assert sheet.header.jianpu_key == "1=C" and sheet.header.capo == 0
    assert sheet.header.bpm == 120.0 and sheet.header.time_signature == (4, 4)
    assert [b.index for b in sheet.bars] == [0, 1, 2, 3, 4]
    assert [(c.beat, c.name) for c in sheet.bars[3].chords] == [(0, "F"), (2, "G")]
    assert [(c.beat, c.name) for c in sheet.bars[4].chords] == [(0, "G")]  # carried over
    assert sheet.bars[4].end == 10.0
    names = [c.name for b in sheet.bars for c in b.chords]
    assert names == ["C", "G", "Am", "F", "G", "G"]  # simplified: G/B -> G, Fmaj7 -> F, G7 -> G
    originals = [c.original_name for b in sheet.bars for c in b.chords]
    assert originals == ["C", "G/B", "Am", "Fmaj7", "G7", "G7"]
    assert set(sheet.shapes) == {"C", "G", "Am", "F"}
    assert sheet.shapes["C"].frets == (None, 3, 2, 0, 1, 0)
    assert sheet.low_confidence_count == 1
    assert [c.low_confidence for c in sheet.bars[2].chords] == [True]
    assert sheet.strum is not None and sheet.strum.id == "folk"
    assert [o.id for o in sheet.strum_options if o.recommended] == ["folk"]
    # lowest capo first: C 0, A 3, G 5, F 7 recommended; E 8, D 10 not
    assert [(o.name, o.capo) for o in sheet.fingering_options] == [
        ("C", 0), ("A", 3), ("G", 5), ("F", 7), ("E", 8), ("D", 10)
    ]  # fmt: skip
    assert [o.recommended for o in sheet.fingering_options] == [True] * 4 + [False] * 2
    assert [o.selected for o in sheet.fingering_options] == [True] + [False] * 5
    assert sheet.warnings == ()


def test_unsimplified_names_keep_sevenths_and_slash_basses() -> None:
    sheet = arrange(SONG, Arrangement(simplify_chords=False))
    names = [c.name for b in sheet.bars for c in b.chords]
    assert names == ["C", "G/B", "Am", "Fmaj7", "G7", "G7"]
    assert set(sheet.shapes) == {"C", "G/B", "Am", "Fmaj7", "G7"}
    assert sheet.shapes["G/B"].frets == sheet.shapes["G/B"].frets  # a slash chord uses G's shape


def test_transposing_keeps_the_chord_names_when_the_fingering_key_is_fixed() -> None:
    base = arrange(SONG, Arrangement(fingering_key=0))
    for transpose in range(-6, 7):
        moved = arrange(SONG, Arrangement(transpose=transpose, fingering_key=0))
        assert [c.name for b in moved.bars for c in b.chords] == [
            c.name for b in base.bars for c in b.chords
        ]
        assert moved.header.capo == transpose % 12
        assert moved.header.fingering_key == "C"
    up = arrange(SONG, Arrangement(transpose=2, fingering_key=0))
    assert (
        up.header.text == "原调 C · 演唱调 D · C 调指法 夹 2 品" and up.header.jianpu_key == "1=D"
    )


def test_changing_the_fingering_key_rewrites_names_but_not_the_sound() -> None:
    g_shapes = arrange(SONG, Arrangement(fingering_key=7))
    assert g_shapes.header.capo == 5  # C sung with G shapes: capo 5
    assert [c.name for b in g_shapes.bars for c in b.chords] == ["G", "D", "Em", "C", "D", "D"]
    placed = [c for b in g_shapes.bars for c in b.chords][:5]
    for chord_in_c, chord_in_g in zip(SONG.chords, placed, strict=True):
        written = g_shapes.shapes[chord_in_g.name].root
        assert (written + g_shapes.header.capo) % 12 == chord_in_c.root


def test_minor_song_uses_minor_fingering_keys() -> None:
    song = SONG.model_copy(update={"key": Key(tonic=9, mode=Mode.MINOR, confidence=0.8)})
    sheet = arrange(song)
    assert sheet.header.text == "原调 Am · 演唱调 Am · Am 调指法 无变调夹"
    assert sheet.header.jianpu_key == "1=C"
    assert [(o.name, o.capo) for o in sheet.fingering_options] == [("Am", 0), ("Em", 5), ("Dm", 7)]
    low = arrange(song, Arrangement(transpose=-2))  # G minor: Em + 3 beats Am + 10
    assert low.header.fingering_key == "Em" and low.header.capo == 3


def test_fallbacks_are_reported() -> None:
    sheet = arrange(SONG, Arrangement(fingering_key=11, strum_pattern="waltz"))
    assert sheet.header.fingering_key == "C"
    assert any("不在候选" in w for w in sheet.warnings)
    assert any("节奏型" in w for w in sheet.warnings) and sheet.strum is not None
    assert sheet.strum.id == "folk"
    silent = arrange(SONG, Arrangement(strum_pattern="none"))
    assert silent.strum is None and silent.strum_options
    bare = arrange(Analysis())
    assert bare.bars == () and bare.header.sung_key == "C" and "调性" in bare.warnings[0]
    no_grid = arrange(Analysis(chords=SONG.chords, key=SONG.key))
    assert any("拍网格" in w for w in no_grid.warnings)


def test_sheet_round_trips_through_json() -> None:
    sheet = arrange(SONG, Arrangement(transpose=3, fingering_key=9))
    assert ArrangedSheet.from_json(sheet.to_json()) == sheet
    with pytest.raises(ValueError):
        ArrangedSheet.from_json("{}")
