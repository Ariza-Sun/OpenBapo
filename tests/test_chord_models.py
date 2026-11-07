"""Chord experts backed by real models. Label mapping tests run everywhere;
inference tests need the analysis extra (`pytest -m models`)."""

import numpy as np
import pytest
from synth import progression, voicing

from opencapo.analysis.chords.fusion import aggregate_to_beats
from opencapo.analysis.chords.models import btc, lvchordia
from opencapo.analysis.chords.vocab import NO_CHORD, index_label, label_index
from opencapo.model import ChordQuality

C_F_G7_AM = [
    (0, ChordQuality.MAJ),
    (5, ChordQuality.MAJ),
    (7, ChordQuality.DOM7),
    (9, ChordQuality.MIN),
]


def test_btc_label_map_covers_every_class() -> None:
    mapping = btc.label_map()
    assert mapping.shape == (170,)
    assert mapping[168] == NO_CHORD and mapping[169] == NO_CHORD  # X and N
    assert mapping[0 * 14 + 1] == label_index(0, ChordQuality.MAJ)  # C:maj
    assert mapping[7 * 14 + 9] == label_index(7, ChordQuality.DOM7)  # G:7
    assert mapping[9 * 14 + 7] == label_index(9, ChordQuality.MIN)  # A:minmaj7 -> A:min
    # 13 of our qualities are reachable (minmaj7 folds onto min; nothing yields "5"), plus N.
    assert len(set(mapping.tolist())) == 13 * 12 + 1


def test_btc_to_vocabulary_sums_folded_classes() -> None:
    probs = np.zeros((1, 170))
    probs[0, 9 * 14 + 0] = 0.5  # A:min
    probs[0, 9 * 14 + 7] = 0.3  # A:minmaj7 -> A:min
    probs[0, 168] = 0.2  # X -> N
    out = btc.to_vocabulary(probs)
    assert out[0, label_index(9, ChordQuality.MIN)] == pytest.approx(0.8)
    assert out[0, NO_CHORD] == pytest.approx(0.2)
    assert out.sum() == pytest.approx(1.0)


def test_lvchordia_template_and_name_mapping() -> None:
    text = lvchordia.template_text()
    assert "C:maj\n" in text and "C:hdim7\n" in text and text.count("\n") == 14
    indices = lvchordia.names_to_indices(["N", "C:maj", "Bb:min7", "F#:7"])
    assert indices.tolist() == [
        NO_CHORD,
        label_index(0, ChordQuality.MAJ),
        label_index(10, ChordQuality.MIN7),
        label_index(6, ChordQuality.DOM7),
    ]


def test_btc_checkpoint_url_is_pinned() -> None:
    assert "2682317be668032e6e4b269ded36adaa2ad57df0" in btc.CHECKPOINT_URL
    assert len(btc.CHECKPOINT_SHA256) == 64


@pytest.fixture(scope="module")
def synthetic_progression():  # type: ignore[no-untyped-def]
    return progression(
        [
            (voicing([0, 4, 7]), 1.0),
            (voicing([5, 9, 0]), 1.0),
            (voicing([7, 11, 2, 5]), 1.0),
            (voicing([9, 0, 4]), 1.0),
        ],
        harmonics=3,
    )


def top_labels(post, beat_times, end):  # type: ignore[no-untyped-def]
    beats = aggregate_to_beats(post, beat_times, end_time=end)
    return [index_label(int(i)) for i in beats.argmax(axis=1)]


@pytest.mark.models
def test_btc_recognises_the_synthetic_progression(synthetic_progression) -> None:  # type: ignore[no-untyped-def]
    post = btc.BTCExpert(device="cpu").estimate(synthetic_progression, None)
    np.testing.assert_allclose(post.probs.sum(axis=1), 1.0, atol=1e-5)
    assert top_labels(post, [0.0, 1.0, 2.0, 3.0], 4.0) == C_F_G7_AM


@pytest.mark.models
def test_lvchordia_recognises_the_synthetic_progression(synthetic_progression) -> None:  # type: ignore[no-untyped-def]
    post = lvchordia.LVChordiaExpert(device="cpu").estimate(synthetic_progression, None)
    np.testing.assert_allclose(post.probs.sum(axis=1), 1.0, atol=1e-5)
    labels = top_labels(post, [0.0, 1.0, 2.0, 3.0], 4.0)
    assert [root for root, _ in labels] == [0, 5, 7, 9]  # type: ignore[misc]
    assert labels[0] == (0, ChordQuality.MAJ) and labels[3] == (9, ChordQuality.MIN)
