"""Beat and downbeat tracking: audio in, beat and downbeat times (seconds) out."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from openbapo.analysis.audio import Audio
from openbapo.analysis.device import AUTO, resolve_device

DEFAULT_BEAT_THIS_CHECKPOINT = "final0"


@dataclass(frozen=True)
class BeatTimes:
    beats: NDArray[np.float64]
    downbeats: NDArray[np.float64]  # a subset of `beats`


class BeatTracker(Protocol):
    def track(self, audio: Audio) -> BeatTimes: ...


class BeatThisTracker:
    """Beat This! (Foscarin, Schlüter, Widmer; ISMIR 2024) through the `beat_this` package.

    Default checkpoint `final0` was trained on everything except GTZAN, which
    keeps GTZAN usable as a test set. Code and weights are MIT. No DBN
    post-processing, so madmom is not needed.
    """

    def __init__(self, checkpoint: str = DEFAULT_BEAT_THIS_CHECKPOINT, device: str = AUTO) -> None:
        self.checkpoint = checkpoint
        self.device = resolve_device(device)
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is None:
            from beat_this.inference import Audio2Beats

            self._model = Audio2Beats(
                checkpoint_path=self.checkpoint, device=self.device, dbn=False
            )
        return self._model

    def track(self, audio: Audio) -> BeatTimes:
        model = self._load()
        beats, downbeats = model(audio.mono(), audio.sample_rate)  # type: ignore[operator]
        return BeatTimes(
            beats=np.asarray(beats, dtype=np.float64),
            downbeats=np.asarray(downbeats, dtype=np.float64),
        )
