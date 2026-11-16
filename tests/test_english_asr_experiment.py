"""Experiment timestamps must map back to the uncut song without inventing coverage."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("compare_english_asr")


def test_candidate_chunks_neither_fill_silence_nor_duplicate_audio(experiment):
    assert list(experiment.chunks([(1.5, 64.0), (70.0, 72.0)])) == [
        (1.5, 31.5),
        (31.5, 61.5),
        (61.5, 64.0),
        (70.0, 72.0),
    ]
    assert list(experiment.chunks([])) == []


def test_subword_and_punctuation_timestamps_preserve_original_offset(experiment):
    words = experiment.token_words(
        [" Hel", "lo", ",", " world", "!"], [0.16, 0.24, 0.24, 0.56, 0.64], 100.0, 100.7
    )
    assert [w.text for w in words] == ["Hello,", "world!"]
    assert [(w.start, w.end) for w in words] == [(100.16, 100.32), (100.56, 100.7)]


def test_timestamp_loss_is_an_error_not_silent_truncation(experiment):
    with pytest.raises(ValueError, match="length mismatch"):
        experiment.token_words([" word"], [], 0, 1)


def test_missing_transcription_stays_in_error_denominator(experiment):
    pytest.importorskip("jiwer")
    result = experiment.Recognition("en", ())
    entry = {"reference_text": "hello world", "duration": 10, "vocal_regions": [(1, 9)]}
    measured = experiment.score(entry, result, [], [], "hash")
    assert measured["reference_words"] == measured["deletions"] == 2
    assert measured["wer"] == 1.0


def test_development_references_fall_back_to_annotated_lines(experiment):
    pytest.importorskip("jiwer")
    entry = {
        "reference_text": "",
        "lines": [{"text": "hello world"}],
        "duration": 10,
        "vocal_regions": [(1, 9)],
    }
    measured = experiment.score(entry, experiment.Recognition("en", ()), [], [], "hash")
    assert measured["reference_words"] == 2 and measured["wer"] == 1.0
