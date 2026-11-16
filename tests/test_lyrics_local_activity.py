import numpy as np
import pytest

from openbapo.analysis.lyrics.activity import ActivityParams, SileroActivity, supported_activity


def test_separated_short_detection_cannot_borrow_evidence_from_the_song():
    assert supported_activity([(0, 12), (65, 67)], context_gap_s=30, min_total_s=5) == [(0, 12)]


def test_short_phrases_keep_their_gaps_when_supported_by_nearby_singing():
    regions = [(2, 4), (8, 10), (17, 18), (60, 66)]
    assert supported_activity(regions, context_gap_s=30, min_total_s=5) == regions
    # Neither silence nor overlapping padding can satisfy the voiced-time gate.
    assert supported_activity([(2, 3), (20, 21)], context_gap_s=30, min_total_s=5) == []
    assert supported_activity([(2, 5), (3, 6)], context_gap_s=30, min_total_s=5) == []


def test_long_gap_boundary_and_minimum_duration_are_inclusive():
    assert supported_activity([(0, 5), (35, 36)], context_gap_s=30, min_total_s=5) == [(0, 5)]
    assert supported_activity([], context_gap_s=30, min_total_s=5) == []


def test_filter_is_opt_in_and_supplied_lyrics_detector_keeps_short_phrases(monkeypatch):
    regions = [(0, 12), (65, 67)]
    monkeypatch.setattr(SileroActivity, "_speech_regions", lambda self, audio: regions)
    audio = np.zeros(70 * 16000, dtype=np.float32)
    assert SileroActivity(ActivityParams(context_gap_s=30)).regions(audio) == [(0, 12)]
    assert SileroActivity(ActivityParams(energy_relative_db=-15)).regions(audio) == regions


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_context_is_rejected(value):
    with pytest.raises(ValueError, match="positive and finite"):
        supported_activity([], context_gap_s=value, min_total_s=5)
