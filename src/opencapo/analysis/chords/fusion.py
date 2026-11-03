"""Fuse chord experts on the beat grid into a Chord sequence.

The decision procedure mirrors how a person transcribes chords
(docs/plans/step-2.md, section 2):

1. every expert's frame-wise posterior is averaged inside each beat;
2. experts are combined log-linearly with per-expert weights;
3. roots inside the estimated key get a bonus;
4. a Viterbi pass makes chord changes cheap on downbeats and expensive
   mid-bar, so boundaries land where a player would write them;
5. "no chord" survives only when the experts jointly prefer it;
6. the bass line decides inversions and lowers confidence when it
   contradicts the chord;
7. confidence = fused posterior of the chosen label x share of model experts
   that agree on its root.

Everything here is plain numpy and runs in CI without any model.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from opencapo.analysis.audio import Audio
from opencapo.analysis.chords.vocab import (
    N_LABELS,
    NO_CHORD,
    bass_relation,
    chord_tones,
    index_label,
)
from opencapo.analysis.separate import Stems
from opencapo.model import Beat, Chord, Key, Mode

EPS = 1e-6

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)  # natural minor; the raised 7th is added below


@dataclass(frozen=True)
class ChordPosteriors:
    """Frame-wise output of one expert, already mapped into the shared vocabulary."""

    times: NDArray[np.float64]  # frame centre times in seconds, increasing
    probs: NDArray[np.float64]  # (frames, N_LABELS); rows sum to 1

    def __post_init__(self) -> None:
        if self.probs.ndim != 2 or self.probs.shape[1] != N_LABELS:
            raise ValueError(f"probs must be (frames, {N_LABELS}), got {self.probs.shape}")
        if self.times.shape != (self.probs.shape[0],):
            raise ValueError("times must have one entry per frame")


class ChordExpert(Protocol):
    name: str
    is_model: bool  # deep-learning experts count towards the agreement term

    def estimate(self, audio: Audio, stems: Stems | None) -> ChordPosteriors: ...


@dataclass(frozen=True)
class BassEstimate:
    """Per-beat bass pitch class from the bass stem; -1 where nothing was detected."""

    pitch_class: NDArray[np.int64]
    strength: NDArray[np.float64]  # in [0, 1]


@dataclass(frozen=True)
class FusionParams:
    """The few tunable knobs (section 2.3)."""

    # Defaults are the values tuned on GuitarSet's tuning half
    # (docs/benchmarks/step-2-chords.md, 调参结果); a missing weight means 1.
    weights: Mapping[str, float] = field(
        default_factory=lambda: {"btc": 0.5, "lvchordia": 0.5, "chroma": 0.5}
    )
    key_prior: float = 0.5  # added to the log score of chords whose root is in the key
    change_cost_downbeat: float = 0.1  # cost of changing chord on a downbeat
    change_cost_beat: float = 3.0  # ... on any other beat
    no_chord_bias: float = -1.0  # added to the log score of "N"
    foreign_bass_penalty: float = 0.7  # confidence multiplier when the bass is not a chord tone
    min_bass_strength: float = 0.3  # weaker bass evidence is ignored
    # A lead sheet holds the chord through a rest shorter than a bar; longer
    # silences (a break, a drum solo) stay "N". Not tuned: notation convention.
    fill_gaps_under_bar: bool = True
    # The detected bass note as evidence for the label (a transcriber's first
    # move): log-score bonus, scaled by the detection strength, for chords
    # whose root is the bass note; half of it when the bass is another chord tone.
    bass_weight: float = 1.0
    # Weight frames by 1 - P(N) when pooling an expert's output per beat (see
    # aggregate_to_beats); applied before decode by whoever aggregates.
    voiced_aggregation: bool = False
    # Per beat, scale each expert's weight by its own confidence (the top
    # posterior) to this power: 0 trusts all experts equally, higher values
    # let the surer expert decide that beat.
    confidence_power: float = 0.0


def aggregate_to_beats(
    posteriors: ChordPosteriors,
    beat_times: Sequence[float],
    end_time: float,
    *,
    voiced: bool = False,
) -> NDArray[np.float64]:
    """Average frame posteriors inside each beat interval [t_i, t_{i+1}).

    With `voiced`, each frame is weighted by how sure the expert is that a
    chord sounds there (1 - P(N)), so a sparse, syncopated accompaniment is
    read from the frames where the chord is struck rather than diluted by
    the rests in between. A beat that contains no frame takes the nearest
    frame, so short beats at coarse hop sizes still get a distribution.
    """
    n_beats = len(beat_times)
    out = np.zeros((n_beats, N_LABELS))
    if n_beats == 0:
        return out
    edges = [*beat_times, max(end_time, beat_times[-1] + EPS)]
    frame_bins = np.searchsorted(edges, posteriors.times, side="right") - 1
    for i in range(n_beats):
        rows = posteriors.probs[frame_bins == i]
        if len(rows) == 0:
            nearest = int(np.argmin(np.abs(posteriors.times - beat_times[i])))
            rows = posteriors.probs[nearest : nearest + 1]
        if voiced:
            weights = 1.0 - rows[:, NO_CHORD] + EPS
            out[i] = weights @ rows / weights.sum()
        else:
            out[i] = rows.mean(axis=0)
    return out


def fuse(
    beat_probs: Mapping[str, NDArray[np.float64]], params: FusionParams
) -> NDArray[np.float64]:
    """Weighted product of experts, renormalised per beat."""
    if not beat_probs:
        raise ValueError("no experts to fuse")
    log_score = None
    for name, probs in beat_probs.items():
        weight = params.weights.get(name, 1.0) * np.ones((len(probs), 1))
        if params.confidence_power > 0:
            weight = weight * probs.max(axis=1, keepdims=True) ** params.confidence_power
        weighted = weight * np.log(probs + EPS)
        log_score = weighted if log_score is None else log_score + weighted
    assert log_score is not None
    log_score -= log_score.max(axis=1, keepdims=True)
    fused = np.exp(log_score)
    return np.asarray(fused / fused.sum(axis=1, keepdims=True))


def key_prior(key: Key | None, bonus: float) -> NDArray[np.float64]:
    """Log-score bonus per label for roots that belong to the key's scale."""
    prior = np.zeros(N_LABELS)
    if key is None or bonus == 0.0:
        return prior
    scale = MAJOR_SCALE if key.mode == Mode.MAJOR else (*MINOR_SCALE, 11)
    in_key = {(key.tonic + degree) % 12 for degree in scale}
    for index in range(NO_CHORD):
        root, _ = index_label(index)  # type: ignore[misc]
        if root in in_key:
            prior[index] = bonus
    return prior


def bass_relations() -> NDArray[np.float64]:
    """(N_LABELS, 12): 1 where the pitch class is the label's root, 0.5 another chord tone."""
    table = np.zeros((N_LABELS, 12))
    for index in range(NO_CHORD):
        root, quality = index_label(index)  # type: ignore[misc]
        for pc in chord_tones(root, quality):
            table[index, pc] = 1.0 if pc == root else 0.5
    return table


_BASS_RELATIONS: NDArray[np.float64] | None = None


def bass_prior(
    bass: BassEstimate | None, weight: float, min_strength: float
) -> NDArray[np.float64]:
    """(n_beats, N_LABELS) log-score bonus from the detected bass line; zero rows where none."""
    global _BASS_RELATIONS
    if bass is None or weight == 0.0:
        return np.zeros((0, N_LABELS))
    if _BASS_RELATIONS is None:
        _BASS_RELATIONS = bass_relations()
    n = len(bass.pitch_class)
    prior = np.zeros((n, N_LABELS))
    for i in range(n):
        pc, strength = int(bass.pitch_class[i]), float(bass.strength[i])
        if pc >= 0 and strength >= min_strength:
            prior[i] = weight * strength * _BASS_RELATIONS[:, pc]
    return prior


def viterbi(
    log_scores: NDArray[np.float64], is_downbeat: Sequence[bool], params: FusionParams
) -> NDArray[np.int64]:
    """Best label path where changing label costs more mid-bar than on a downbeat."""
    n, n_labels = log_scores.shape
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    score = log_scores[0].copy()
    back = np.zeros((n, n_labels), dtype=np.int64)
    for i in range(1, n):
        cost = params.change_cost_downbeat if is_downbeat[i] else params.change_cost_beat
        best_prev = int(np.argmax(score))
        change = score[best_prev] - cost
        stay = score
        use_change = change > stay
        back[i] = np.where(use_change, best_prev, np.arange(n_labels))
        score = np.where(use_change, change, stay) + log_scores[i]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def agreement(
    fused_labels: NDArray[np.int64], model_probs: Sequence[NDArray[np.float64]]
) -> NDArray[np.float64]:
    """Share of model experts whose top root matches the fused label's root (1 if none)."""
    n = len(fused_labels)
    if not model_probs:
        return np.ones(n)
    fused_roots = np.array([_root_of(int(label)) for label in fused_labels])
    votes = np.zeros(n)
    for probs in model_probs:
        expert_roots = np.array([_root_of(int(label)) for label in probs.argmax(axis=1)])
        votes += expert_roots == fused_roots
    return votes / len(model_probs)


def _root_of(index: int) -> int:
    parsed = index_label(index)
    return -1 if parsed is None else parsed[0]


@dataclass(frozen=True)
class FusionResult:
    chords: tuple[Chord, ...]
    labels: NDArray[np.int64]  # per beat, index into the vocabulary
    confidence: NDArray[np.float64]  # per beat


def decode(
    beats: Sequence[Beat],
    beat_probs: Mapping[str, NDArray[np.float64]],
    model_names: Sequence[str],
    *,
    duration: float,
    key: Key | None = None,
    bass: BassEstimate | None = None,
    params: FusionParams | None = None,
) -> FusionResult:
    """Run steps 2-7 and build the Chord sequence. Every boundary is a beat time."""
    params = params or FusionParams()
    n = len(beats)
    if n == 0:
        return FusionResult((), np.zeros(0, dtype=np.int64), np.zeros(0))
    fused = fuse(beat_probs, params)
    log_scores = np.log(fused + EPS) + key_prior(key, params.key_prior)
    log_scores[:, NO_CHORD] += params.no_chord_bias
    from_bass = bass_prior(bass, params.bass_weight, params.min_bass_strength)
    if len(from_bass) == n:
        log_scores += from_bass
    labels = viterbi(log_scores, [b.is_downbeat for b in beats], params)
    if params.fill_gaps_under_bar:
        labels = fill_gaps(labels, beats_per_bar(beats))
    agree = agreement(labels, [beat_probs[name] for name in model_names if name in beat_probs])
    confidence = fused[np.arange(n), labels] * agree

    chords: list[Chord] = []
    start = 0
    while start < n:
        end = start
        while end + 1 < n and labels[end + 1] == labels[start]:
            end += 1
        parsed = index_label(int(labels[start]))
        if parsed is not None:
            root, quality = parsed
            segment_confidence = float(confidence[start : end + 1].min())
            chord_bass = None
            bass_pc = _segment_bass(bass, start, end, params.min_bass_strength)
            if bass_pc is not None:
                relation = bass_relation(root, quality, bass_pc)
                if relation == "chord_tone":
                    chord_bass = bass_pc
                elif relation == "foreign":
                    segment_confidence *= params.foreign_bass_penalty
            chord_end = beats[end + 1].time if end + 1 < n else max(duration, beats[end].time + EPS)
            chords.append(
                Chord(
                    start=beats[start].time,
                    end=chord_end,
                    root=root,
                    quality=quality,
                    bass=chord_bass,
                    confidence=min(1.0, max(0.0, segment_confidence)),
                )
            )
        start = end + 1
    return FusionResult(tuple(chords), labels, confidence)


def beats_per_bar(beats: Sequence[Beat]) -> int:
    """Median beats between downbeats; 4 when the grid has fewer than two downbeats."""
    downbeats = [i for i, b in enumerate(beats) if b.is_downbeat]
    if len(downbeats) < 2:
        return 4
    return int(np.median(np.diff(downbeats)))


def fill_gaps(labels: NDArray[np.int64], max_gap: int) -> NDArray[np.int64]:
    """Replace runs of N shorter than `max_gap` beats that sit between chords with the chord before.

    Leading and trailing N stay: nothing precedes a leading gap, and a tail
    that fades to silence is not a rest inside a phrase.
    """
    out = labels.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i] != NO_CHORD:
            i += 1
            continue
        j = i
        while j < n and out[j] == NO_CHORD:
            j += 1
        interior = i > 0 and j < n
        if interior and j - i < max_gap:
            out[i:j] = out[i - 1]
        i = j
    return out


def _segment_bass(
    bass: BassEstimate | None, start: int, end: int, min_strength: float
) -> int | None:
    """Strength-weighted vote of the detected bass pitch class over a segment."""
    if bass is None:
        return None
    votes = np.zeros(12)
    for i in range(start, end + 1):
        pc, strength = int(bass.pitch_class[i]), float(bass.strength[i])
        if pc >= 0 and strength >= min_strength:
            votes[pc] += strength
    if votes.sum() == 0:
        return None
    return int(np.argmax(votes))
