"""Turn beat and downbeat times into the Beat grid, global tempo and meter.

Pure computation on top of a beat tracker's output; no model code here.
Rules (docs/plans/step-1.md, section 2.3):

- beats per bar `n` is the mode of the beat count between consecutive downbeats;
- bars are numbered from the earliest one; beats before the first downbeat are
  put into implicit bars of `n` beats so that `bar_index` starts at 0;
- the tracker's downbeats are regularised: every beat gets its position in
  the bar from a Viterbi pass that expects bars of `n` beats, so an isolated
  spurious downbeat no longer cuts a 1-beat bar and an isolated missed one no
  longer makes a long bar; only a phase change the tracker keeps reporting
  for several bars moves it, and a shift that does not return (a genuine
  short or long bar in the song) is kept;
- fewer than two downbeats, or every beat flagged as one, means the bar length
  is unknown: bars of DEFAULT_BEATS_PER_BAR are laid out and no meter is claimed;
- bpm is 60 over the median beat interval;
- the meter is n/4, except that six beats per bar is written 6/8.
"""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from statistics import median

from openbapo.model import Beat, Tempo, TimeSignature

# Used when there are too few downbeats to measure the bar length.
DEFAULT_BEATS_PER_BAR = 4

# Downbeat regularisation, a hidden Markov model over the position in the bar.
# The numbers are a priori, not fitted: the tracker flags most true downbeats
# and few other beats, and the bar phase rarely jumps.
P_FLAGGED_DOWNBEAT = 0.8  # P(tracker flags a beat | it is a downbeat)
P_FLAGGED_OTHER = 0.05  # P(tracker flags a beat | it is not)
P_BEAT_SLIP = 0.0  # extra per-beat chance that the tracker added or dropped a beat
P_PHASE_JUMP = 1e-4  # per beat: any change of the bar phase


@dataclass(frozen=True)
class BeatGrid:
    beats: tuple[Beat, ...]
    tempo: Tempo | None
    time_signature: TimeSignature | None


def _match_downbeats(beat_times: Sequence[float], downbeat_times: Sequence[float]) -> list[int]:
    """Index of the beat closest to each downbeat; unmatched or duplicate ones dropped."""
    if not beat_times or not downbeat_times:
        return []
    tolerance = (
        median(b - a for a, b in pairwise(beat_times)) / 4 if len(beat_times) > 1 else float("inf")
    )
    indices: set[int] = set()
    cursor = 0
    for t in sorted(downbeat_times):
        while cursor + 1 < len(beat_times) and beat_times[cursor + 1] <= t:
            cursor += 1
        best = cursor
        if cursor + 1 < len(beat_times) and abs(beat_times[cursor + 1] - t) < abs(
            beat_times[cursor] - t
        ):
            best = cursor + 1
        if abs(beat_times[best] - t) <= tolerance:
            indices.add(best)
    return sorted(indices)


def beats_per_bar(downbeat_indices: Sequence[int]) -> int | None:
    """Mode of the beat count between consecutive downbeats; None if unmeasurable."""
    gaps = [b - a for a, b in pairwise(downbeat_indices)]
    if not gaps:
        return None
    counts = Counter(gaps)
    top = max(counts.values())
    # Ties go to the shorter bar: a missed downbeat doubles a gap more often
    # than a spurious one halves it.
    return min(gap for gap, count in counts.items() if count == top)


def regular_downbeats(
    count: int,
    flagged: Sequence[int],
    n: int,
    *,
    p_slip: float | None = None,
    p_jump: float | None = None,
) -> list[int]:
    """Indices of the beats that start a bar, given `count` beats and the flagged ones.

    Viterbi over each beat's position in an `n`-beat bar (0 = downbeat). A beat
    normally advances the position by one; any other move costs a share of
    P_PHASE_JUMP, more than one stray or missing downbeat flag can repay, so
    isolated flag errors are ignored. P_BEAT_SLIP adds probability to the
    moves of a dropped or added beat (off by default; see
    docs/benchmarks/step-6-bar-grid.md for the trade-off).
    """
    if count == 0 or n < 2:
        return []
    slip = P_BEAT_SLIP if p_slip is None else p_slip
    jump = P_PHASE_JUMP if p_jump is None else p_jump
    flags = set(flagged)
    hit, miss = math.log(P_FLAGGED_DOWNBEAT), math.log(1 - P_FLAGGED_DOWNBEAT)
    false, quiet = math.log(P_FLAGGED_OTHER), math.log(1 - P_FLAGGED_OTHER)
    # Transition probabilities by step (target - source) mod n: every phase
    # change costs at least a jump; a slip (step 0 or 2) adds its own share.
    steps = {d: jump / (n - 1) for d in range(n) if d != 1}
    steps[1] = 1.0 - jump
    if n > 2:
        for d in (0, 2 % n):
            steps[d] += slip
        steps[1] -= 2 * slip
    log_step = {d: math.log(p) for d, p in steps.items()}

    def emit(i: int, k: int) -> float:
        if k == 0:
            return hit if i in flags else miss
        return false if i in flags else quiet

    score = [emit(0, k) for k in range(n)]
    back: list[list[int]] = []
    for i in range(1, count):
        new, links = [], []
        for k in range(n):
            prev = max(
                ((k - d) % n for d in log_step), key=lambda j: score[j] + log_step[(k - j) % n]
            )
            new.append(score[prev] + log_step[(k - prev) % n] + emit(i, k))
            links.append(prev)
        score = new
        back.append(links)
    k = max(range(n), key=score.__getitem__)
    path = [k]
    for links in reversed(back):
        k = links[k]
        path.append(k)
    path.reverse()
    return [i for i, k in enumerate(path) if k == 0]


def build_grid(beat_times: Sequence[float], downbeat_times: Sequence[float] = ()) -> BeatGrid:
    """Build the Beat grid from tracker output. Times are seconds; order is not required."""
    times = sorted({float(t) for t in beat_times if t >= 0})
    if not times:
        return BeatGrid(beats=(), tempo=None, time_signature=None)

    tempo = None
    if len(times) > 1:
        tempo = Tempo(bpm=60.0 / median(b - a for a, b in pairwise(times)))

    downbeats = _match_downbeats(times, downbeat_times)
    measured = beats_per_bar(downbeats)
    if measured is not None and measured < 2:
        # Every beat flagged as a downbeat (seen on synthetic click tracks): no
        # usable bar information, so the downbeats are ignored altogether.
        downbeats, measured = [], None
    n = measured if measured is not None else DEFAULT_BEATS_PER_BAR
    if measured is not None:
        downbeats = regular_downbeats(len(times), downbeats, n) or downbeats
    anchor = downbeats[0] if downbeats else 0
    downbeat_set = set(downbeats)

    # Bar numbers relative to the anchor downbeat; shifted to start at 0 below.
    positions: list[tuple[int, int]] = []
    for i in range(anchor):
        offset = anchor - i
        positions.append((-((offset + n - 1) // n), n - (offset - 1) % n))
    # Without a measured bar length the tracker gave no bar information to
    # respect, so bars are also closed every `n` beats after the anchor.
    bar, beat_in_bar = 0, 0
    for i in range(anchor, len(times)):
        implied = measured is None and (i - anchor) % n == 0
        if i == anchor or i in downbeat_set or implied:
            bar, beat_in_bar = (bar + 1 if i > anchor else bar), 1
        else:
            beat_in_bar += 1
        positions.append((bar, beat_in_bar))
    first_bar = positions[0][0]

    beats = tuple(
        Beat(time=t, is_downbeat=pos == 1, bar_index=bar - first_bar, beat_in_bar=pos)
        for t, (bar, pos) in zip(times, positions, strict=True)
    )
    time_signature = None
    if measured is not None:
        time_signature = TimeSignature(numerator=measured, denominator=8 if measured == 6 else 4)
    return BeatGrid(beats=beats, tempo=tempo, time_signature=time_signature)
