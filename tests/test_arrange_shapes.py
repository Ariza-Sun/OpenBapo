"""The chord-shape library: coverage, correctness and the default choice."""

import subprocess
import sys
from pathlib import Path

from openbapo.arrange import shapes
from openbapo.model import ChordQuality

Q = ChordQuality
TONES = {
    Q.MAJ: {0, 4, 7}, Q.MIN: {0, 3, 7}, Q.DOM7: {0, 4, 7, 10}, Q.MAJ7: {0, 4, 7, 11},
    Q.MIN7: {0, 3, 7, 10}, Q.MAJ6: {0, 4, 7, 9}, Q.MIN6: {0, 3, 7, 9}, Q.SUS2: {0, 2, 7},
    Q.SUS4: {0, 5, 7}, Q.POWER: {0, 7}, Q.DIM: {0, 3, 6}, Q.AUG: {0, 4, 8},
    Q.DIM7: {0, 3, 6, 9}, Q.HDIM7: {0, 3, 6, 10},
}  # fmt: skip


def test_library_covers_every_root_and_quality() -> None:
    assert shapes.missing() == []
    assert len(shapes.library()) >= 12 * 14


def test_every_shape_sounds_its_chord() -> None:
    for s in shapes.library():
        rel = {(p - s.root) % 12 for p in s.pitch_classes}
        assert rel <= TONES[s.quality] and 0 in rel, s
        assert s.quality == ChordQuality.POWER or rel & {2, 3, 4, 5}, s  # a third (or suspension)
        sounded = [f for f in s.frets if f is not None]
        assert len(sounded) >= 2 and len(s.frets) == 6
        assert s.base_fret <= max(sounded) + 1


def test_default_choice_prefers_open_then_nearby_position() -> None:
    c = shapes.choose_shape(0, ChordQuality.MAJ)
    assert c.frets == (None, 3, 2, 0, 1, 0) and c.difficulty == 1
    fsharp = shapes.choose_shape(6, ChordQuality.MAJ)
    assert fsharp.source == "E-form" and fsharp.frets == (2, 4, 4, 3, 2, 2)
    # Bb has an A-form barre at fret 1 and an E-form barre at fret 6: the
    # easier (equal) one nearest the previous chord wins.
    low = shapes.choose_shape(10, ChordQuality.MAJ, previous_base_fret=1)
    high = shapes.choose_shape(10, ChordQuality.MAJ, previous_base_fret=7)
    assert low.base_fret == 1 and high.base_fret == 6
    assert shapes.choose_shape(11, ChordQuality.HDIM7).frets == (None, 2, 3, 2, 3, None)


def test_committed_library_matches_the_generator() -> None:
    script = Path(__file__).parents[1] / "scripts" / "make_shapes.py"
    proc = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
