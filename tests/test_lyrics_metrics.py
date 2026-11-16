"""Acceptance metrics must count missing predictions and actual characters."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "eval_lyrics", Path(__file__).resolve().parents[1] / "scripts/eval_lyrics.py"
)
assert spec is not None and spec.loader is not None
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)
error_rate, line_errors, summarise = metrics.error_rate, metrics.line_errors, metrics.summarise


def test_chinese_cer_does_not_dilute_errors_with_inserted_spaces():
    pytest.importorskip("jiwer")
    pytest.importorskip("opencc")
    rate, units = error_rate("zh", "我爱你", "我爱他")
    assert units == 3 and rate == pytest.approx(1 / 3)
    assert error_rate("zh", "我爱你", "")[0] == 1.0


def test_missing_lines_still_count_as_alignment_failures():
    entry = {"duration": 4.0, "lines": [{"start": 1.0}, {"start": 2.0}]}
    assert line_errors(entry, []) == [None, None]
    row = {
        "set": "test",
        "lang": "en",
        "condition": "real",
        "align": {"wav2vec2": {"errors": [0.1, None], "seconds": 1.0}},
    }
    result = summarise([row], ["wav2vec2"])["test/en/real"]["align/wav2vec2"]
    assert result["within"] == 0.5
    assert result["missing"] == 1 and result["lines"] == 2
    row["align"]["wav2vec2"]["errors"] = [None, None]
    result = summarise([row], ["wav2vec2"])["test/en/real"]["align/wav2vec2"]
    assert result["within"] == 0 and result["median"] is None


def test_gap_candidates_retain_times_and_do_not_suppress_vocalizations():
    from openbapo.analysis.lyrics.recognize import Recognition, RecognizedLine, RecognizedWord

    words = (RecognizedWord(1.0, 1.5, "oh"), RecognizedWord(5.0, 5.5, "song"))
    rec = Recognition("en", (RecognizedLine(1, 5.5, "oh song", words),))
    entry = {"duration": 10.0, "vocal_regions": [(4.0, 8.0)]}
    assert metrics.gap_candidates(entry, rec) == [{"text": "oh", "start": 1.0, "end": 1.5}]
    assert metrics.hallucinated(entry, rec) == ["oh"]  # keep the legacy count for comparison
    assert rec.text == "oh song"  # metric review never edits recognition output
