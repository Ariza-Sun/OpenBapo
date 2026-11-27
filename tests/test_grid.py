from itertools import pairwise

import pytest

from openbapo.analysis.grid import beats_per_bar, build_grid, snap_to_pulse
from openbapo.model import Analysis


def positions(times: list[float], downbeats: list[float]) -> list[tuple[int, int]]:
    grid = build_grid(times, downbeats)
    Analysis(beats=grid.beats, tempo=grid.tempo, time_signature=grid.time_signature)  # validates
    return [(b.bar_index, b.beat_in_bar) for b in grid.beats]


def test_clean_four_four() -> None:
    times = [0.5 * i for i in range(8)]
    grid = build_grid(times, [0.0, 2.0])
    assert [(b.bar_index, b.beat_in_bar) for b in grid.beats] == [
        (0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 2), (1, 3), (1, 4),
    ]  # fmt: skip
    assert grid.tempo is not None and grid.tempo.bpm == pytest.approx(120.0)
    assert grid.time_signature is not None
    assert (grid.time_signature.numerator, grid.time_signature.denominator) == (4, 4)
    assert [b.is_downbeat for b in grid.beats] == [True, False, False, False] * 2


def test_three_four() -> None:
    times = [0.5 * i for i in range(9)]
    grid = build_grid(times, [0.0, 1.5, 3.0])
    assert grid.time_signature is not None
    assert (grid.time_signature.numerator, grid.time_signature.denominator) == (3, 4)


def test_six_beats_per_bar_is_written_six_eight() -> None:
    times = [0.25 * i for i in range(12)]
    grid = build_grid(times, [0.0, 1.5])
    assert grid.time_signature is not None
    assert (grid.time_signature.numerator, grid.time_signature.denominator) == (6, 8)


def test_pickup_beat_goes_into_bar_zero() -> None:
    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    assert positions(times, [0.5, 2.0]) == [(0, 3), (1, 1), (1, 2), (1, 3), (2, 1)]


def test_pickup_longer_than_a_bar() -> None:
    # 5 beats before the first downbeat with 4 per bar: two implicit bars.
    times = [0.5 * i for i in range(10)]
    got = positions(times, [2.5, 4.5])
    assert got[:5] == [(0, 4), (1, 1), (1, 2), (1, 3), (1, 4)]
    assert got[5] == (2, 1)


def test_two_downbeats_eight_apart_measure_eight() -> None:
    times = [0.5 * i for i in range(12)]
    got = positions(times, [0.0, 4.0])  # downbeat at beat 8 skipped by the tracker
    assert got[4:8] == [(0, 5), (0, 6), (0, 7), (0, 8)]
    assert got[8] == (1, 1)
    grid = build_grid(times, [0.0, 4.0])
    assert grid.time_signature is not None and grid.time_signature.numerator == 8


def test_bar_length_is_the_mode_and_ties_go_short() -> None:
    assert beats_per_bar([0, 4, 8, 12, 20]) == 4
    assert beats_per_bar([0, 4, 12]) == 4
    assert beats_per_bar([3]) is None
    assert beats_per_bar([]) is None


def test_no_downbeats_assumes_four_without_claiming_a_meter() -> None:
    times = [0.5 * i for i in range(6)]
    grid = build_grid(times, [])
    assert [(b.bar_index, b.beat_in_bar) for b in grid.beats] == [
        (0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 2),
    ]  # fmt: skip
    assert grid.time_signature is None
    assert grid.tempo is not None


def test_single_downbeat_anchors_but_does_not_measure() -> None:
    times = [0.5 * i for i in range(8)]
    grid = build_grid(times, [1.0])
    assert [(b.bar_index, b.beat_in_bar) for b in grid.beats] == [
        (0, 3), (0, 4), (1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2),
    ]  # fmt: skip
    assert grid.time_signature is None


def test_every_beat_a_downbeat_means_no_meter() -> None:
    times = [0.5 * i for i in range(6)]
    grid = build_grid(times, times)
    assert [(b.bar_index, b.beat_in_bar) for b in grid.beats] == [
        (0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 2),
    ]  # fmt: skip
    assert grid.time_signature is None


def test_single_beat() -> None:
    grid = build_grid([1.0], [1.0])
    assert [(b.bar_index, b.beat_in_bar) for b in grid.beats] == [(0, 1)]
    assert grid.tempo is None and grid.time_signature is None


def test_no_beats() -> None:
    grid = build_grid([], [])
    assert grid.beats == () and grid.tempo is None and grid.time_signature is None


def test_downbeats_snap_to_nearest_beat_within_tolerance() -> None:
    times = [0.5 * i for i in range(8)]
    # Slightly off downbeats snap; one far from any beat is ignored.
    assert positions(times, [0.02, 1.98, 1.25]) == positions(times, [0.0, 2.0])


def test_unsorted_and_duplicate_input_is_cleaned() -> None:
    times = [1.0, 0.0, 0.5, 0.5, 1.5]
    assert [b.time for b in build_grid(times, [0.0]).beats] == [0.0, 0.5, 1.0, 1.5]


def test_tempo_uses_median_interval() -> None:
    grid = build_grid([0.0, 0.5, 1.0, 1.5, 3.5], [0.0])  # one dropped beat
    assert grid.tempo is not None and grid.tempo.bpm == pytest.approx(120.0)


def bar_lengths(times: list[float], downbeats: list[float]) -> list[int]:
    grid = build_grid(times, downbeats)
    counts: dict[int, int] = {}
    for b in grid.beats:
        counts[b.bar_index] = counts.get(b.bar_index, 0) + 1
    return [counts[i] for i in sorted(counts)]


def test_isolated_spurious_downbeat_does_not_cut_a_short_bar() -> None:
    times = [0.5 * i for i in range(32)]
    flagged = [0.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]  # 3.0 is stray
    assert bar_lengths(times, flagged) == [4] * 8


def test_isolated_missed_downbeat_is_repaired() -> None:
    times = [0.5 * i for i in range(32)]
    flagged = [0.0, 2.0, 6.0, 8.0, 10.0, 12.0, 14.0]  # 4.0 missed
    assert bar_lengths(times, flagged) == [4] * 8


def test_noisy_start_settles_into_regular_bars() -> None:
    # The pattern seen on real songs: a burst of stray downbeats in the intro.
    times = [0.5 * i for i in range(40)]
    flagged = [0.0, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0, *[2.0 * k for k in range(3, 10)]]
    assert bar_lengths(times, flagged) == [4] * 10


def test_a_genuine_two_beat_bar_is_kept_once() -> None:
    # Bars of 4, then one 2-beat bar, then the phase stays shifted.
    times = [0.5 * i for i in range(34)]
    flagged = [0.0, 2.0, 4.0, 6.0, 7.0, 9.0, 11.0, 13.0, 15.0]
    assert bar_lengths(times, flagged) == [4, 4, 4, 2, 4, 4, 4, 4, 4]


def test_a_song_counted_at_two_levels_is_counted_at_the_slower(monkeypatch) -> None:
    import openbapo.analysis.grid as grid

    slow = [i * 1.0 for i in range(40)]  # 40 s counted at 60 bpm
    fast = [40.0 + i * 0.5 for i in range(240)]  # then 120 s at 120 bpm
    pulse = grid.snap_to_pulse(slow + fast)
    assert pulse.changed == [(40.0, 159.5)] and pulse.times[:40] == slow
    assert all(b - a == pytest.approx(1.0) for a, b in pairwise(pulse.times))
    # The alternative: the level most of the song is at; the slow part gets midpoints.
    monkeypatch.setattr(grid, "LEVEL_PREFER_SLOWER", False)
    pulse = grid.snap_to_pulse(slow + fast)
    assert len(pulse.times) == len(slow) * 2 + len(fast)
    assert all(b - a == pytest.approx(0.5) for a, b in pairwise(pulse.times))


def test_the_meter_keeps_the_trackers_beats_per_bar_at_its_main_level() -> None:
    from openbapo.analysis.grid import build_grid

    # As tracked on a real song: 40 s counted at 60 bpm, then 120 s at 120 bpm,
    # the tracker's bars 2 s long throughout (2 slow beats, then 4 fast ones).
    slow = [i * 1.0 for i in range(40)]
    fast = [40.0 + i * 0.5 for i in range(240)]
    downbeats = [t for t in slow + fast if t % 2 == 0]
    g = build_grid(slow + fast, downbeats)
    # Its main level read 4 beats a bar; at the slower level that is 4/4 of 4 s.
    assert g.time_signature is not None and g.time_signature.numerator == 4
    assert round(g.tempo.bpm) == 60
    bars = [b.time for b in g.beats if b.is_downbeat]
    assert all(b - a == pytest.approx(4.0) for a, b in pairwise(bars[1:-1]))


def test_a_tracker_flipping_between_two_levels_is_counted_at_one() -> None:
    # The pattern seen on a real chorus: a few half beats, a few beats, a few
    # half beats... far too short for a level of its own, yet 5-7 beat bars
    # unless the song is counted on one pulse grid.
    beats = [i * 1.0 for i in range(120)]
    flips = [40.5, 41.5, 42.5, 47.5, 48.5, 56.5, 57.5, 58.5, 59.5, 70.5, 81.5, 82.5]
    downbeats = [t for t in beats if t % 4 == 0]
    pulse = snap_to_pulse(sorted(beats + flips), downbeats)
    assert pulse.times == beats
    assert pulse.changed[0] == (40.0, 43.0) and len(pulse.changed) == 5
    assert bar_lengths(sorted(beats + flips), downbeats) == [4] * 30


def test_a_short_doubled_fill_is_counted_at_the_beat_and_a_steady_song_is_unchanged() -> None:
    steady = [i * 0.5 for i in range(300)]
    pulse = snap_to_pulse(steady)
    assert pulse.times == steady and pulse.changed == [] and pulse.loose == []
    fill = [i * 0.5 for i in range(200)] + [100.0 + i * 0.25 for i in range(1, 33)]
    fill += [108.0 + i * 0.5 for i in range(1, 200)]  # an 8 s doubled fill
    pulse = snap_to_pulse(fill)
    assert pulse.times == pytest.approx([i * 0.5 for i in range(416)])
    assert pulse.changed == [(100.25, 108.0)]


def test_a_skipped_beat_is_filled_but_a_pause_or_a_tempo_break_is_not() -> None:
    steady = [i * 1.0 for i in range(60)]
    pulse = snap_to_pulse([t for t in steady if t != 21.0])
    assert pulse.times == steady
    # A 6 s pause is left as it is: no beats are invented inside it.
    paused = [t for t in steady if not 20.0 < t < 26.0]
    assert snap_to_pulse(paused).times == paused
    # An interval that is no whole number of half beats (a rubato) starts the
    # count afresh: the beats after it keep their times, nothing is inserted.
    shifted = steady[:20] + [t + 0.3 for t in steady[20:]]  # 20.3, 21.3, ...
    assert snap_to_pulse(shifted).times == shifted


def test_a_stretch_shifted_by_half_a_beat_is_put_back_in_step() -> None:
    # The tracker lands on the off-beats for a while (a 1.5-beat interval in,
    # another out); the song's downbeats say which half is the beat.
    steady = [i * 1.0 for i in range(100)]
    off = [t + 0.5 for t in steady[40:60]]  # 40.5 .. 59.5
    times = steady[:40] + off + steady[60:]
    downbeats = [t for t in steady if t % 4 == 0]
    pulse = snap_to_pulse(times, downbeats)
    assert pulse.times == steady and pulse.changed == [(59.5, 60.0)]
    assert bar_lengths(times, downbeats) == [4] * 25


def test_the_half_holding_the_trackers_downbeats_is_kept() -> None:
    # Half beats throughout a stretch, its downbeats on the half: the stretch
    # alone would be counted on the half, but the rest of the song outvotes it.
    steady = [i * 1.0 for i in range(100)]
    doubled = [100.0 + i * 0.5 for i in range(1, 41)]
    tail = [120.0 + i * 1.0 for i in range(1, 30)]
    on_half = [100.5 + 4 * i for i in range(5)]
    pulse = snap_to_pulse(steady + doubled + tail, on_half)
    assert pulse.times == [i + 0.5 for i in range(149)]
    on_beat = [t for t in steady if t % 4 == 0]
    pulse = snap_to_pulse(steady + doubled + tail, on_beat + on_half)
    assert pulse.times == [i * 1.0 for i in range(150)]


def test_downbeats_off_the_grid_carry_no_bar_evidence(monkeypatch) -> None:
    import openbapo.analysis.grid as grid

    # An intro whose downbeats fix the bar phase, then 60 s counted at twice
    # the rate with the tracker's bars half as long (either phase fits), then
    # a noisy tail of beats off the grid whose flags favour the other phase.
    # Those flags count for nothing, so the song keeps the phase of its intro.
    intro = [i * 1.0 for i in range(20)]
    fast = [20.0 + i * 0.5 for i in range(120)]
    flags = [0.0, 4.0, 8.0, 12.0, 16.0, *[20.0 + 2 * k for k in range(30)]]
    tail, t = [], 80.0
    for k in range(40):
        t += (0.28, 0.33, 0.31, 0.27)[k % 4]
        tail.append(round(t, 2))
    times, downbeats = intro + fast + tail, flags + tail[2::4]
    pulse = grid.snap_to_pulse(times, downbeats)
    assert pulse.loose and pulse.loose[0][0] >= 80.0
    g = build_grid(times, downbeats)
    assert g.time_signature is not None and g.time_signature.numerator == 4
    assert bar_lengths(times, downbeats)[:20] == [4] * 20
    assert [b.time for b in g.beats if b.is_downbeat][:6] == [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]
    # Counted as evidence, the tail's flags would move the phase: a 2-beat bar.
    monkeypatch.setattr(grid, "PULSE_MIN_CHAIN", 0)
    assert 2 in bar_lengths(times, downbeats)[:20]
