"""Missing annotations and agreement between models are not proof of hallucination."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "lyrics_gap_review", Path(__file__).resolve().parents[1] / "scripts/lyrics_gap_review.py"
)
assert spec is not None and spec.loader is not None
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)

WORD = {"start": 10.0, "end": 10.4}


def test_official_vocal_annotation_explains_gap_but_missing_labels_remain_unknown():
    assert review.classify(WORD, [{"start": 10.1, "end": 11}], [], "audio") == (
        "reference_supported_vocal"
    )
    assert review.classify(WORD, [], [], "audio") == "unresolved"


def test_model_agreement_does_not_resolve_pending_human_review():
    pending = {
        "start": 9,
        "end": 11,
        "audio_sha256": "audio",
        "verdict": "pending",
        "reviewer": None,
        "evidence": "two models agree",
    }
    assert review.classify(WORD, [], [pending], "audio") == "unresolved"


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [("vocal", "human_supported_vocal"), ("instrumental", "confirmed_hallucination")],
)
def test_human_verdict_is_bound_to_audio_and_needs_reviewer(verdict, expected):
    confirmed = {
        "start": 9,
        "end": 11,
        "audio_sha256": "audio",
        "verdict": verdict,
        "reviewer": "test reviewer",
    }
    assert review.classify(WORD, [], [confirmed], "audio") == expected
    with pytest.raises(ValueError, match="different audio"):
        review.classify(WORD, [], [confirmed], "different")
    confirmed["reviewer"] = None
    with pytest.raises(ValueError, match="identified reviewer"):
        review.classify(WORD, [], [confirmed], "audio")


def test_conflicting_reviews_cannot_silently_pass_acceptance():
    confirmed = {"start": 9, "end": 11, "audio_sha256": "audio", "reviewer": "test"}
    with pytest.raises(ValueError, match="conflicting"):
        review.classify(
            WORD, [], [dict(confirmed, verdict=v) for v in ("vocal", "instrumental")], "audio"
        )
