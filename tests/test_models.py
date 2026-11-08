"""End-to-end checks with the real models. Needs `pip install "openbapo[analysis]"`,
network access for the weights on first run, and CPU time; run with `pytest -m models`.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from openbapo.analysis.audio import decode
from openbapo.analysis.beats import BeatThisTracker
from openbapo.analysis.pipeline import analyze_file
from openbapo.analysis.separate import STEM_NAMES, DemucsSeparator
from openbapo.cli import main
from openbapo.model import Analysis

pytestmark = pytest.mark.models

MAKE_TONE = Path(__file__).parents[1] / "scripts" / "make_tone.py"
BPM = 120.0


@pytest.fixture(scope="module")
def click(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("audio") / "click.wav"
    subprocess.run(
        [sys.executable, str(MAKE_TONE), str(path), "--click", str(BPM), "--seconds", "12"],
        check=True,
    )
    return path


def test_demucs_produces_four_stems(click: Path) -> None:
    audio = decode(click)
    stems = DemucsSeparator(device="cpu").separate(audio)
    assert stems.sample_rate == 44100
    for name in STEM_NAMES:
        assert stems[name].shape == audio.samples.shape
        assert np.isfinite(stems[name]).all()
    total = sum(stems[name] for name in STEM_NAMES)
    # Demucs stems add up to (approximately) the mix.
    assert np.corrcoef(total.ravel(), audio.samples.ravel())[0, 1] > 0.9


def test_beat_this_tracks_the_click(click: Path) -> None:
    tracked = BeatThisTracker(device="cpu").track(decode(click))
    assert len(tracked.beats) > 10
    assert set(tracked.downbeats.tolist()) <= set(tracked.beats.tolist())
    intervals = np.diff(tracked.beats)
    assert 60.0 / np.median(intervals) == pytest.approx(BPM, abs=3.0)


def test_pipeline_end_to_end(click: Path) -> None:
    before = set(Path(tempfile.gettempdir()).glob("openbapo-stems-*"))
    result = analyze_file(
        click, separator=DemucsSeparator(device="cpu"), tracker=BeatThisTracker(device="cpu")
    )
    assert set(Path(tempfile.gettempdir()).glob("openbapo-stems-*")) == before
    analysis = result.analysis
    assert analysis.tempo is not None and analysis.tempo.bpm == pytest.approx(BPM, abs=3.0)
    assert analysis.beats and analysis.duration == pytest.approx(12.0)
    assert result.timings.separate > 0 and result.timings.beats > 0
    assert Analysis.from_json(analysis.to_json()) == analysis


def test_cli_with_real_models(click: Path, tmp_path: Path) -> None:
    out = tmp_path / "analysis.json"
    assert main(["analyze", str(click), "--out", str(out), "--device", "cpu"]) == 0
    analysis = Analysis.from_json(out.read_text(encoding="utf-8"))
    assert analysis.tempo is not None and analysis.beats


@pytest.fixture(scope="module")
def chord_progression(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("audio") / "progression.wav"
    subprocess.run(
        [sys.executable, str(MAKE_TONE), str(path), "--progression", str(BPM), "--seconds", "16"],
        check=True,
    )
    return path


def test_pipeline_with_chords_end_to_end(chord_progression: Path) -> None:
    from openbapo.analysis.chords.stage import default_chord_stage

    result = analyze_file(
        chord_progression,
        separator=DemucsSeparator(device="cpu"),
        tracker=BeatThisTracker(device="cpu"),
        chord_stage=default_chord_stage("cpu"),
    )
    analysis = result.analysis
    # C F G7 Am ends every loop on Am, so C major and its relative A minor are
    # equally fair readings (the key rules settle the pair on a near tie); the
    # tonal centre is what the product uses (decision 16 in docs/plans/step-2.md).
    assert analysis.key is not None
    assert (analysis.key.tonic, analysis.key.mode.value) in {(0, "major"), (9, "minor")}
    roots = [c.root for c in analysis.chords]
    # C F G7 Am per bar, repeating: every bar's root is one of these and all four appear.
    assert set(roots) <= {0, 5, 7, 9} and {0, 5, 7, 9} <= set(roots)
    beat_times = {b.time for b in analysis.beats}
    assert all(c.start in beat_times for c in analysis.chords)
    assert result.timings.chords > 0
    assert Analysis.from_json(analysis.to_json()) == analysis
