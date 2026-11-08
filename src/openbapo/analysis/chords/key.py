"""Rule expert E5: key, estimated two ways and cross-checked.

Way one listens to the whole piece: the average chroma is correlated with
the Krumhansl-Schmuckler key profiles. Way two reads the chord sheet: each
candidate key is scored by how much of the chord sequence (by duration) it
explains. A transcriber does both ("the song sits on G, and these chords are
the ones G major has"), and when the two disagree the key is doubtful.

Key index convention: tonic * 2 + (0 for major, 1 for minor).

The chord view is scored in overlapping windows and averaged, so a piece
that modulates (or whose model output drifts) votes window by window
instead of pooling two keys into a third one. A relative major/minor pair
that scores within RELATIVE_MARGIN is settled by which tonic chord gets
more time, with the opening and closing chords (tonic, or its dominant)
as extra votes: those are what a listener uses to tell C major from A
minor. The weights were chosen on GuitarSet's tuning half with
scripts/eval_key_inputs.py (docs/plans/step-2.md, 2.7).
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from openbapo.analysis.chords.fusion import MAJOR_SCALE, MINOR_SCALE
from openbapo.analysis.chords.vocab import chord_tones
from openbapo.model import Chord, ChordQuality, Key, Mode

# Krumhansl & Kessler (1982) probe-tone profiles.
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

MAJOR_FAMILY = frozenset(
    {ChordQuality.MAJ, ChordQuality.MAJ7, ChordQuality.MAJ6, ChordQuality.DOM7}
)
MINOR_FAMILY = frozenset({ChordQuality.MIN, ChordQuality.MIN7, ChordQuality.MIN6})
DOMINANT = frozenset({ChordQuality.MAJ, ChordQuality.DOM7})

CHROMA_WEIGHT = 0.5  # the chroma view is the noisier of the two
WINDOW_SECONDS = 16.0  # chord-view windows (hop is half a window)
RELATIVE_MARGIN = 0.05  # relative keys closer than this (of the best score) are settled by votes
POSITION_VOTES = 2.0  # opening / closing chord votes, in mean chord durations


def key_index(tonic: int, mode: Mode) -> int:
    return tonic * 2 + (0 if mode == Mode.MAJOR else 1)


def key_from_index(index: int) -> tuple[int, Mode]:
    return index // 2, Mode.MAJOR if index % 2 == 0 else Mode.MINOR


def scale_of(tonic: int, mode: Mode) -> frozenset[int]:
    degrees = MAJOR_SCALE if mode == Mode.MAJOR else (*MINOR_SCALE, 11)  # harmonic minor's 7th
    return frozenset((tonic + d) % 12 for d in degrees)


def scores_from_chroma(mean_chroma: NDArray[np.float64]) -> NDArray[np.float64]:
    """(24,) Pearson correlation of the chroma with each rotated profile."""
    scores = np.zeros(24)
    if not np.any(mean_chroma):
        return scores
    for tonic in range(12):
        for mode, profile in ((Mode.MAJOR, KS_MAJOR), (Mode.MINOR, KS_MINOR)):
            rotated = np.roll(profile, tonic)
            scores[key_index(tonic, mode)] = float(np.corrcoef(mean_chroma, rotated)[0, 1])
    return scores


def scores_from_chords(
    chords: Sequence[Chord], window: float = WINDOW_SECONDS
) -> NDArray[np.float64]:
    """(24,) mean over windows of the normalised share of chords each key explains.

    Within a window a chord counts fully when all its tones are in the key,
    half when only its root is; the tonic chord of the right quality family
    earns a bonus. Each window's 24 scores are normalised to a distribution
    before averaging, so every window carries the same weight. `window <= 0`
    scores the whole piece at once.
    """
    if not chords:
        return np.zeros(24)
    end = max(c.end for c in chords)
    if window <= 0 or end <= window:
        return _normalise(_explained(chords, 0.0, end))
    acc = np.zeros(24)
    count = 0
    start = 0.0
    while start < end:
        scores = _explained(chords, start, min(start + window, end))
        if np.any(scores):
            acc += _normalise(scores)
            count += 1
        start += window / 2
    return acc / count if count else np.zeros(24)


def _explained(chords: Sequence[Chord], t0: float, t1: float) -> NDArray[np.float64]:
    """(24,) duration-weighted explanation of the chords clipped to [t0, t1)."""
    scores = np.zeros(24)
    clipped = [(max(c.start, t0), min(c.end, t1), c) for c in chords]
    clipped = [(s, e, c) for s, e, c in clipped if e > s]
    total = sum(e - s for s, e, _ in clipped)
    if total <= 0:
        return scores
    for index in range(24):
        tonic, mode = key_from_index(index)
        scale = scale_of(tonic, mode)
        family = MAJOR_FAMILY if mode == Mode.MAJOR else MINOR_FAMILY
        value = 0.0
        for s, e, chord in clipped:
            duration = e - s
            tones = chord_tones(chord.root, chord.quality)
            if tones <= scale:
                value += duration
            elif chord.root in scale:
                value += 0.5 * duration
            if chord.root == tonic and chord.quality in family:
                value += 0.5 * duration
        scores[index] = value / (1.5 * total)
    return scores


def _normalise(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = scores - scores.min()
    total = shifted.sum()
    return np.asarray(shifted / total) if total > 0 else np.full(24, 1 / 24)


def estimate_key(
    chroma_scores: NDArray[np.float64],
    chord_scores: NDArray[np.float64],
    chords: Sequence[Chord] = (),
) -> Key | None:
    """Combine both views; None when neither has any evidence.

    Confidence is the margin between the best and second-best key, halved
    when the two views disagree on the winner. When the best key and its
    relative major/minor are within RELATIVE_MARGIN, the pair is settled by
    tonic-chord time plus the opening / closing chord votes.
    """
    has_chroma, has_chords = bool(np.any(chroma_scores)), bool(np.any(chord_scores))
    if not has_chroma and not has_chords:
        return None
    combined = np.zeros(24)
    if has_chroma:
        combined += CHROMA_WEIGHT * _normalise(chroma_scores)
    if has_chords:
        # Already a distribution when it comes from scores_from_chords.
        combined += (
            chord_scores / chord_scores.sum()
            if chord_scores.min() >= 0
            else _normalise(chord_scores)
        )
    order = np.argsort(combined)[::-1]
    best, second = int(order[0]), int(order[1])
    settled = 0.0
    relative = _relative_of(best)
    if combined[best] - combined[relative] < RELATIVE_MARGIN * combined[best]:
        # The tie-break may rank the marginally lower-scoring key first, so the
        # score margin is clamped at zero and the tie-break's own decisiveness
        # (how lopsided the votes are) stands in for it.
        best, second, settled = _settle_relative(best, relative, chords)
    margin = max(0.0, (combined[best] - combined[second]) / max(combined[best], 1e-9))
    margin = max(margin, settled)
    confidence = float(min(1.0, 6.0 * margin))  # a 1/6 relative margin is already clear
    if has_chroma and has_chords and int(np.argmax(chroma_scores)) != int(np.argmax(chord_scores)):
        confidence /= 2.0
    tonic, mode = key_from_index(best)
    return Key(tonic=tonic, mode=mode, confidence=confidence)


def _relative_of(index: int) -> int:
    tonic, mode = key_from_index(index)
    if mode == Mode.MAJOR:
        return key_index((tonic + 9) % 12, Mode.MINOR)
    return key_index((tonic + 3) % 12, Mode.MAJOR)


def _settle_relative(a: int, b: int, chords: Sequence[Chord]) -> tuple[int, int, float]:
    """Order a relative major/minor pair by votes; also say how lopsided it was (0-1)."""
    ta, ma = key_from_index(a)
    tb, _ = key_from_index(b)
    major_tonic = ta if ma == Mode.MAJOR else tb
    minor_tonic = (major_tonic + 9) % 12
    major_votes = _tonic_votes(major_tonic, Mode.MAJOR, chords)
    minor_votes = _tonic_votes(minor_tonic, Mode.MINOR, chords)
    major_index = key_index(major_tonic, Mode.MAJOR)
    minor_index = key_index(minor_tonic, Mode.MINOR)
    total = major_votes + minor_votes
    strength = abs(major_votes - minor_votes) / total if total > 0 else 0.0
    if minor_votes > major_votes:
        return minor_index, major_index, strength
    return major_index, minor_index, strength


def _tonic_votes(tonic: int, mode: Mode, chords: Sequence[Chord]) -> float:
    """Time on the tonic chord, plus the opening and closing chords' votes.

    An opening or closing tonic chord counts POSITION_VOTES mean chord
    durations; the dominant (major or 7th on the fifth degree) counts half.
    """
    family = MAJOR_FAMILY if mode == Mode.MAJOR else MINOR_FAMILY
    votes = sum(c.end - c.start for c in chords if c.root == tonic and c.quality in family)
    if not chords or POSITION_VOTES <= 0:
        return votes
    mean_duration = sum(c.end - c.start for c in chords) / len(chords)
    for c in (chords[0], chords[-1]):
        if c.root == tonic and c.quality in family:
            votes += POSITION_VOTES * mean_duration
        elif c.root == (tonic + 7) % 12 and c.quality in DOMINANT:
            votes += 0.5 * POSITION_VOTES * mean_duration
    return votes
