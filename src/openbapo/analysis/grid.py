"""Turn beat and downbeat times into the Beat grid, global tempo and meter.

Pure computation on top of a beat tracker's output; no model code here.
Rules (docs/plans/step-1.md, section 2.3):

- beats per bar `n` is the mode of the beat count between consecutive downbeats;
- bars are numbered from the earliest one; beats before the first downbeat are
  put into implicit bars of `n` beats so that `bar_index` starts at 0;
- a bar that the tracker gave more than `n` beats keeps counting (a missed
  downbeat is reported as is, not repaired);
- fewer than two downbeats, or every beat flagged as one, means the bar length
  is unknown: bars of DEFAULT_BEATS_PER_BAR are laid out and no meter is claimed;
- bpm is 60 over the median beat interval;
- the meter is n/4, except that six beats per bar is written 6/8.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from statistics import median

from openbapo.model import Beat, Tempo, TimeSignature

# Used when there are too few downbeats to measure the bar length.
DEFAULT_BEATS_PER_BAR = 4


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
