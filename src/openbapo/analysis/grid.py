"""Turn beat and downbeat times into the Beat grid, global tempo and meter.

Pure computation on top of a beat tracker's output; no model code here.
Rules:

- the tracker's beats are first put on one pulse grid (`snap_to_pulse`): the
  pulse is half the song's beat, every interval is read as a whole number of
  pulses counted from the one before, and the sheet gets one beat every two
  pulses. A tracker that counts a stretch at twice the rate (a chorus, a few
  beats of a fill), skips a beat or shifts by half a beat is thereby counted
  at the song's one beat level; an interval that is not a whole number of
  pulses, or too long (a pause), starts the count afresh. A song with a
  stretch of 15 s or more counted at half the rate of the rest is written at
  that slower level (the level sheets are written at);
- beats per bar `n` is the mode of the beat count between consecutive downbeats,
  taken at the tracker's main level when it switched level (its bar length
  does not follow its level switches, so 4/4 counted at 125 stays 4/4 at
  62.5), with 2 read as half a 4/4 bar;
- bars are numbered from the earliest one; beats before the first downbeat are
  put into implicit bars of `n` beats so that `bar_index` starts at 0;
- the tracker's downbeats are regularised: every beat gets its position in
  the bar from a Viterbi pass that expects bars of `n` beats, so an isolated
  spurious downbeat no longer cuts a 1-beat bar and an isolated missed one no
  longer makes a long bar; only a phase change the tracker keeps reporting
  for several bars moves it, and a shift that does not return (a genuine
  short or long bar in the song) is kept. Downbeats the tracker reports off
  the pulse grid (a run shorter than a bar: noise, a stop) are no evidence;
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

# The pulse grid: one beat every two pulses, the pulse half the song's beat.
SNAP_TO_PULSE = True  # False: the tracker's beats as they are (for comparison in scripts)
PULSE_TOL = 0.25  # pulses: an interval further than this from a whole number breaks the count
PULSE_MAX_GAP = 3  # beats: a longer interval (a pause) is not filled with beats
PULSE_MIN_CHAIN = 4  # beats: downbeats in a shorter run on the grid carry no bar evidence
# The song's beat level: the one holding most of its time, or twice that when
# a stretch of LEVEL_MIN_S or more is counted at half the rate.
LEVEL_WINDOW = 8  # beats either side for the local interval (median)
LEVEL_RATIO = (1.7, 2.3)  # local / main interval taken as the slower level
LEVEL_MIN_S = 15.0  # shorter stretches (a break) do not make the song slower
LEVEL_PREFER_SLOWER = True  # two levels in one song: the slower (False: the main one)


@dataclass(frozen=True)
class BeatGrid:
    beats: tuple[Beat, ...]
    tempo: Tempo | None
    time_signature: TimeSignature | None


@dataclass(frozen=True)
class PulseGrid:
    """The tracker's beats counted on one pulse grid; (start, end) seconds for stretches."""

    times: list[float]
    changed: list[tuple[float, float]]  # counted by the tracker finer than the sheet's beat
    main: list[tuple[float, float]]  # counted by the tracker at its main level
    loose: list[tuple[float, float]]  # off the grid: too short a run to carry bar evidence


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


def _flagged(
    times: Sequence[float], beats: Sequence[int], downbeat_times: Sequence[float], tolerance: float
) -> int:
    """How many of `beats` lie within `tolerance` of a tracker downbeat."""
    return sum(any(abs(times[k] - t) <= tolerance for t in downbeat_times) for k in beats)


def _outside(times: Sequence[float], stretches: Sequence[tuple[float, float]]) -> list[float]:
    return [t for t in times if not any(a <= t <= b for a, b in stretches)]


def _stretches(flags: Sequence[bool], times: Sequence[float]) -> list[tuple[float, float]]:
    """(start, end) seconds of the runs of flagged intervals; flags[k] is for times[k..k+1]."""
    found: list[tuple[float, float]] = []
    for k, flag in enumerate(flags):
        if flag:
            if found and found[-1][1] == times[k]:
                found[-1] = (found[-1][0], times[k + 1])
            else:
                found.append((times[k], times[k + 1]))
    return found


def _has_slower_stretch(times: Sequence[float], local: Sequence[float], reference: float) -> bool:
    """Is some stretch of LEVEL_MIN_S or more counted at about half the rate of `reference`?"""
    low, high = LEVEL_RATIO
    slower = [low <= gap / reference <= high for gap in local]
    return any(b - a >= LEVEL_MIN_S for a, b in _stretches(slower, times))


def snap_to_pulse(beat_times: Sequence[float], downbeat_times: Sequence[float] = ()) -> PulseGrid:
    """Count the tracker's beats on one pulse grid. Sorted beat times in.

    The pulse is half the song's beat: the beat level holding most of the
    song's time, or twice that when the tracker counts a stretch of
    LEVEL_MIN_S or more at half the rate (the level sheets are written at).
    Each interval is read as a whole number of pulses (within PULSE_TOL),
    counted from the interval before, so a tempo
    drift does not add up. The sheet gets one beat every two pulses: the
    tracker's beat where it reported one, a beat halfway between its
    neighbours where it did not. Which half of the pulses is kept is the one
    holding more of the tracker's downbeats, so a tracker that shifted by
    half a beat for a while is put back in step. An interval that is not a
    whole number of pulses (a tempo change, noise) or longer than
    PULSE_MAX_GAP beats (a pause) starts the count afresh, and a run shorter
    than PULSE_MIN_CHAIN beats is reported as loose: the tracker was not on
    the grid there, so its downbeats are no evidence of the bar.
    """
    times = list(beat_times)
    gaps = [b - a for a, b in pairwise(times)]
    if not SNAP_TO_PULSE or len(gaps) < 2 * LEVEL_WINDOW + 1:
        return PulseGrid(times, [], [], [])
    local = [
        median(gaps[max(0, i - LEVEL_WINDOW) : i + LEVEL_WINDOW + 1]) for i in range(len(gaps))
    ]
    # The level holding most of the song's time, not most of its beats.
    covered, reference = 0.0, local[0]
    for i in sorted(range(len(gaps)), key=local.__getitem__):
        covered += gaps[i]
        if covered >= sum(gaps) / 2:
            reference = local[i]
            break
    beat = reference
    if LEVEL_PREFER_SLOWER and _has_slower_stretch(times, local, reference):
        beat = 2 * reference
    pulse = beat / 2
    main_step = round(reference / pulse)

    # Each beat's position in pulses within its chain; a chain ends at an
    # interval that is not a whole number of pulses or spans a pause.
    steps = [0] * len(gaps)  # 0: the chain breaks at this interval
    for k, gap in enumerate(gaps):
        ratio = gap / pulse
        n = round(ratio)
        if 1 <= n <= 2 * PULSE_MAX_GAP and abs(ratio - n) <= PULSE_TOL:
            steps[k] = n
    chains: list[list[int]] = [[0]]  # indices of the tracker's beats
    positions = [0] * len(times)
    for k, n in enumerate(steps):
        if n:
            chains[-1].append(k + 1)
            positions[k + 1] = positions[k] + n
        else:
            chains.append([k + 1])

    out: list[float] = []
    loose: list[tuple[float, float]] = []
    tolerance = reference / 4
    for chain in chains:
        at = {positions[k]: k for k in chain}
        last = positions[chain[-1]]
        if last // 2 + 1 < PULSE_MIN_CHAIN:
            loose.append((times[chain[0]], times[chain[-1]]))
        # The half of the pulses holding more of the tracker's downbeats; ties
        # go to the chain's first beat.
        on = _flagged(times, [k for k in chain if positions[k] % 2 == 0], downbeat_times, tolerance)
        off = _flagged(times, [k for k in chain if positions[k] % 2], downbeat_times, tolerance)
        for p in range(1 if off > on else 0, last + 1, 2):
            if p in at:
                out.append(times[at[p]])
                continue
            below = max(k for k in chain if positions[k] < p)
            above = min(k for k in chain if positions[k] > p)
            share = (p - positions[below]) / (positions[above] - positions[below])
            out.append(times[below] + share * (times[above] - times[below]))
    changed = _stretches([n == 1 for n in steps], times)
    main = _stretches([n == main_step for n in steps], times)
    return PulseGrid(out, changed, main, loose)


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
    moves of a dropped or added beat (off by default: on real songs it
    fixed fewer bars than it broke).
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
    return _assemble(times, downbeat_times, snap_to_pulse(times, downbeat_times))


def _assemble(original: list[float], downbeat_times: Sequence[float], pulse: PulseGrid) -> BeatGrid:
    """Tempo, meter and bar numbers for the tracker's beats put on the pulse grid."""
    times = pulse.times
    tempo = None
    if len(times) > 1:
        tempo = Tempo(bpm=60.0 / median(b - a for a, b in pairwise(times)))

    # Downbeats the tracker reported off the grid are no evidence of the bar.
    downbeat_times = _outside(downbeat_times, pulse.loose)
    downbeats = _match_downbeats(times, downbeat_times)
    measured = beats_per_bar(downbeats)
    if pulse.changed:
        # The tracker keeps its bar length when it switches level, so its bars
        # are half bars at the slower one: take its beats per bar at its main
        # level (4/4 counted at 125 stays 4/4 at 62.5, as a halved tempo is
        # written), and read 2 as half a 4/4 bar: 2/4 is rare on play-along
        # sheets.
        at_main = [
            i
            for i in _match_downbeats(original, downbeat_times)
            if any(a <= original[i] <= b for a, b in pulse.main)
        ]
        if len(at_main) >= 4:
            measured = beats_per_bar(at_main)
        if measured == 2:
            measured = 4
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
