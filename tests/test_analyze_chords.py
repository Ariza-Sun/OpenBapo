"""The frame-level error decomposition behind scripts/analyze_chords.py."""

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
mir_eval_missing = importlib.util.find_spec("mir_eval") is None
pytestmark = pytest.mark.skipif(mir_eval_missing, reason="mir_eval not installed")


def load_script(name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analyze_chords = load_script("analyze_chords")


@pytest.mark.parametrize(
    ("ref", "est", "kind"),
    [
        ("C:maj", "C:maj", "correct"),
        ("C:maj7", "C:maj", "correct"),  # majmin ignores extensions
        ("C:sus4", "C:maj", "correct"),  # no third in the reference: not a third error
        ("N", "N", "correct"),
        ("N", "C:maj", "ref_N"),
        ("C:maj", "N", "est_N"),
        ("C:maj", "C:min", "third"),
        ("C:maj", "A:min", "relative"),
        ("A:min", "C:maj", "relative"),
        ("C:maj", "A:maj", "other_root"),
        ("C:maj", "F:maj", "fourth_fifth"),
        ("C:maj", "G:7", "fourth_fifth"),
        ("C:maj", "D:min", "other_root"),
    ],
)
def test_frame_kind(ref: str, est: str, kind: str) -> None:
    assert analyze_chords.frame_kind(ref, est) == kind


def test_label_at_and_boundaries() -> None:
    intervals = np.array([[0.0, 1.0], [1.0, 2.0]])
    times = np.array([0.0, 0.5, 1.0, 1.99, 2.0, 3.0])
    assert analyze_chords.label_at(intervals, ["C", "G"], times) == ["C", "C", "G", "G", "N", "N"]
    near = analyze_chords.near_boundary(times, intervals, tolerance=0.1)
    assert near.tolist() == [True, False, True, True, True, False]
    assert analyze_chords.label_at(np.zeros((0, 2)), [], times) == ["N"] * 6


def test_frame_kinds_counts_and_boundary_share() -> None:
    ref = np.array([[0.0, 1.0], [1.0, 2.0]])
    est = np.array([[0.0, 1.2], [1.2, 2.0]])  # the change comes 0.2 s late
    kinds, at_boundary = analyze_chords.frame_kinds(
        ref, ["C:maj", "G:maj"], est, ["C:maj", "G:maj"], 0.5
    )
    assert kinds["correct"] == 36 and kinds["fourth_fifth"] == 4  # 0.2 s / 0.05 s late frames
    assert at_boundary["fourth_fifth"] == 4  # all of them within a beat of the boundary
    kinds, _ = analyze_chords.frame_kinds(ref, ["C:maj", "G:maj"], est, ["C:min", "N"], 0.5)
    assert kinds["third"] == 20 and kinds["est_N"] == 16 and kinds["correct"] == 0
    assert analyze_chords.frame_kinds(np.zeros((0, 2)), [], est, ["C:maj"], 0.5) == (
        Counter(),
        Counter(),
    )


def test_compress_and_params(tmp_path: Path) -> None:
    text = analyze_chords.compress(np.array([[0, 1], [1, 2], [2, 2.5]]), ["C:maj", "C:maj", "G:7"])
    assert text == "C:maj(2.0) G:7(0.5)"
    report = {
        "systems": [
            {"system": "committee (tuned)", "params": {"weights": {"btc": 0.5}, "key_prior": 1.0}}
        ]
    }
    (tmp_path / "r.json").write_text(json.dumps(report), encoding="utf-8")
    params = analyze_chords.load_params(tmp_path / "r.json")
    assert params.key_prior == 1.0 and params.weights == {"btc": 0.5}
    assert analyze_chords.load_params(None).key_prior == 0.5
