"""The key-input dump written by eval_chords.py and the CPU scorer that reads it."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from opencapo.analysis.chords.fusion import FusionParams
from opencapo.analysis.chords.key import key_index, scores_from_chroma
from opencapo.analysis.chords.vocab import N_LABELS, label_index
from opencapo.model import Beat, ChordQuality, Mode

SCRIPTS = Path(__file__).parents[1] / "scripts"


def load_script(name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


eval_chords = load_script("eval_chords")
eval_key_inputs = load_script("eval_key_inputs")
mir_eval_missing = importlib.util.find_spec("mir_eval") is None


def outputs_for(progression: list[tuple[int, ChordQuality]], beats_per_chord: int = 4):  # type: ignore[no-untyped-def]
    """Fake committee outputs: one confident expert on a 120 bpm grid, chroma of the tonic."""
    labels = [label_index(r, q) for r, q in progression for _ in range(beats_per_chord)]
    probs = np.full((len(labels), N_LABELS), 0.01 / (N_LABELS - 1))
    for i, label in enumerate(labels):
        probs[i, label] = 0.99
    beats = [
        Beat(time=i * 0.5, is_downbeat=i % 4 == 0, bar_index=i // 4, beat_in_bar=i % 4 + 1)
        for i in range(len(labels))
    ]
    chroma = np.zeros(12)
    for pc in (progression[0][0], (progression[0][0] + 4) % 12, (progression[0][0] + 7) % 12):
        chroma[pc] = 1.0
    return eval_chords.TrackOutputs(
        beats=beats,
        duration=len(labels) * 0.5,
        beat_probs={"fake": probs},
        bass=None,
        chroma_scores=scores_from_chroma(chroma / np.linalg.norm(chroma)),
    )


def track(name: str, dataset: str, split: str, key: tuple[int, str], share: float = 1.0):  # type: ignore[no-untyped-def]
    return eval_chords.Track(
        dataset=dataset, split=split, name=name, group="x", audio=Path("x.wav"),
        chord_intervals=np.zeros((0, 2)), chord_labels=[], key=key,
        beats=np.zeros(0), downbeats=np.zeros(0), key_share=share,
        key_times={eval_chords.key_name(key): 10.0 * share, "G major": 10.0 * (1 - share)},
    )  # fmt: skip


@pytest.mark.skipif(mir_eval_missing, reason="mir_eval not installed")
def test_dump_round_trips_and_scores(tmp_path: Path) -> None:
    c_major = [
        (0, ChordQuality.MAJ),
        (5, ChordQuality.MAJ),
        (7, ChordQuality.DOM7),
        (0, ChordQuality.MAJ),
    ]
    tracks = [
        track("00_BN1-129-C_comp", "guitarset", "tune", (0, "major")),
        track("03_BN1-129-C_comp", "guitarset", "report", (0, "major")),
        track("0001", "aam", "report", (0, "major"), share=0.6),
        track("0002", "aam", "report", (9, "minor")),  # the same chords: a relative-key miss
    ]
    outputs = {t.name: outputs_for(c_major) for t in tracks}
    path = tmp_path / "keys.json.gz"
    eval_chords.dump_key_inputs(path, tracks, outputs, ["fake"], ["fake"], FusionParams())

    data = eval_key_inputs.load(path)
    assert data["params"]["key_prior"] == FusionParams().key_prior
    first = data["tracks"][0]
    assert first["key_ref"] == "C major" and first["chords"][0][2:] == [0, "maj"]
    assert len(first["chroma_scores"]) == 24
    assert data["tracks"][2]["key_share"] == 0.6
    assert eval_key_inputs.estimate(first) == "C major"

    scores = eval_key_inputs.evaluate(data["tracks"])
    assert scores["guitarset_tune_comp"]["key_exact"] == 1.0
    assert scores["guitarset_report_comp"]["tracks"] == 1
    assert scores["aam"]["tracks"] == 2 and scores["aam"]["key_exact"] == 0.5
    assert scores["aam"]["relations"] == {"exact": 1, "relative": 1}
    assert scores["aam_single_key"]["tracks"] == 1 and scores["aam_single_key"]["key_exact"] == 0.0
    # 0001 spends 40% in G major: an estimate of either key counts for "any >= 1/3".
    assert scores["aam"]["key_any_major"] == 0.5
    assert key_index(0, Mode.MAJOR) == 0  # sanity for the index helper used elsewhere


def test_relation_names() -> None:
    pytest.importorskip("mir_eval")
    assert eval_key_inputs.relation("C major", "C major") == "exact"
    assert eval_key_inputs.relation("C major", "G major") == "fifth"
    assert eval_key_inputs.relation("C major", "A minor") == "relative"
    assert eval_key_inputs.relation("C major", "C minor") == "parallel"
    assert eval_key_inputs.relation("C major", "F major") == "other"
    assert eval_key_inputs.relation("C major", None) == "none"
