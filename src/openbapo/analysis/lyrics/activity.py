"""Singing activity: Silero speech evidence plus sustained vocal-stem energy.

Silero VAD (MIT, code and weights) is trained on speech; on a separated vocal
stem it also fires on sung phrases, but misses some styles of singing.
For supplied-lyrics alignment, energy can supplement independent speech evidence.
Automatic recognition uses speech-only regions with an optional local support gate.
Supplied-lyrics alignment does not use that gate. The energy candidate's
larger lyric-gap word count needs vocal-annotation review; it is not by itself
proof of increased hallucinations.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 16000

Region = tuple[float, float]
ENERGY_HOP_S = 0.05


def supported_activity(
    regions: list[Region], *, context_gap_s: float, min_total_s: float
) -> list[Region]:
    """Require enough voiced time in each group separated by a long quiet gap.

    Count the union of intervals, not padding overlaps or the intervening silence.
    Short phrases beside other singing survive; isolated short detections do not.
    This conservative recognition gate can also omit genuine isolated interjections.
    """
    if not isfinite(context_gap_s) or context_gap_s <= 0:
        raise ValueError("context gap must be positive and finite")
    if not isfinite(min_total_s) or min_total_s < 0:
        raise ValueError("minimum voiced time must be nonnegative and finite")
    merged: list[Region] = []
    for start, end in sorted(regions):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    groups: list[list[Region]] = []
    for start, end in merged:
        if not groups or start - groups[-1][-1][1] >= context_gap_s:
            groups.append([])
        groups[-1].append((start, end))
    return [
        region
        for group in groups
        if sum(end - start for start, end in group) >= min_total_s
        for region in group
    ]


def frame_rms(vocals: NDArray[np.float32]) -> NDArray[np.float32]:
    """50 ms energy frames; used only as a supplement to detected speech."""
    hop = round(SAMPLE_RATE * ENERGY_HOP_S)
    if not len(vocals):
        return np.empty(0, dtype=np.float32)
    padded = np.pad(vocals, (0, (-len(vocals)) % hop))
    return cast(
        NDArray[np.float32],
        np.sqrt(np.mean(padded.reshape(-1, hop).astype(np.float64) ** 2, axis=1)).astype(
            np.float32
        ),
    )


def extend_activity(
    energy: NDArray[np.float32],
    seeds: list[Region],
    duration: float,
    *,
    relative_db: float,
    floor_db: float,
    min_seed_s: float,
) -> list[Region]:
    """Recover sustained singing energy in a song with independent VAD evidence.

    Energy alone cannot establish that a track contains a singer: instrumental
    leakage also has energy. Require speech evidence first, then union speech
    with sustained energy above a relative and absolute noise floor.
    """
    if not len(energy) or sum(b - a for a, b in seeds) < min_seed_s:
        return list(seeds)
    threshold = max(
        10 ** (floor_db / 20), float(np.percentile(energy, 95)) * 10 ** (relative_db / 20)
    )
    active = energy >= threshold
    # Bridge brief dips within a sung phrase, then reject isolated transients.
    changes = np.flatnonzero(np.diff(np.r_[False, ~active, False]))
    for a, b in zip(changes[::2], changes[1::2], strict=True):
        if a > 0 and b < len(active) and (b - a) * ENERGY_HOP_S <= 0.3:
            active[a:b] = True
    changes = np.flatnonzero(np.diff(np.r_[False, active, False]))
    candidates = [(start, end) for start, end in seeds]
    for a, b in zip(changes[::2], changes[1::2], strict=True):
        a, b = int(a), int(b)
        if (b - a) * ENERGY_HOP_S >= 0.25:
            candidates.append(
                (max(0.0, a * ENERGY_HOP_S - 0.1), min(duration, b * ENERGY_HOP_S + 0.1))
            )
    merged: list[Region] = []
    for start, end in sorted(candidates):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


@dataclass(frozen=True)
class ActivityParams:
    threshold: float = 0.5
    min_speech_s: float = 0.25
    min_silence_s: float = 0.3
    pad_s: float = 0.1
    # A song with less voiced time than this has no lyrics.
    min_total_s: float = 5.0
    # Recognition only: require min_total_s in each group separated by this gap.
    context_gap_s: float | None = None
    # Opt-in recovery; automatic recognition currently defaults to speech-only.
    energy_relative_db: float | None = None
    energy_floor_db: float = -50.0
    energy_min_seed_s: float = 1.0


class ActivityDetector(Protocol):
    def regions(self, vocals: NDArray[np.float32]) -> list[Region]: ...

    def has_vocals(self, regions: list[Region]) -> bool: ...


class SileroActivity:
    def __init__(self, params: ActivityParams | None = None) -> None:
        self.params = params or ActivityParams()
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is None:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad()
        return self._model

    def regions(self, vocals: NDArray[np.float32]) -> list[Region]:
        seeds = self._speech_regions(vocals)
        p = self.params
        regions = seeds
        if p.energy_relative_db is not None:
            regions = extend_activity(
                frame_rms(vocals),
                seeds,
                len(vocals) / SAMPLE_RATE,
                relative_db=p.energy_relative_db,
                floor_db=p.energy_floor_db,
                min_seed_s=p.energy_min_seed_s,
            )
        if p.context_gap_s is not None:
            regions = supported_activity(
                regions, context_gap_s=p.context_gap_s, min_total_s=p.min_total_s
            )
        return regions

    def _speech_regions(self, vocals: NDArray[np.float32]) -> list[Region]:
        import torch
        from silero_vad import get_speech_timestamps

        p = self.params
        stamps = get_speech_timestamps(
            torch.from_numpy(np.ascontiguousarray(vocals)),
            self._load(),
            threshold=p.threshold,
            sampling_rate=SAMPLE_RATE,
            min_speech_duration_ms=int(p.min_speech_s * 1000),
            min_silence_duration_ms=int(p.min_silence_s * 1000),
            speech_pad_ms=int(p.pad_s * 1000),
            return_seconds=True,
        )
        return [(float(s["start"]), float(s["end"])) for s in stamps]

    def has_vocals(self, regions: list[Region]) -> bool:
        return sum(end - start for start, end in regions) >= self.params.min_total_s
