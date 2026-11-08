import pytest

from openbapo.analysis.grid import beats_per_bar, build_grid
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


def test_missed_downbeat_keeps_counting() -> None:
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
