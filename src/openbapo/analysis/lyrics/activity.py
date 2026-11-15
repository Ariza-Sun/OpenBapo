"""Where the vocal stem holds singing: Silero VAD on 16 kHz mono.

Silero VAD (MIT, code and weights) is trained on speech; on a separated vocal
stem it also fires on sung phrases. Its parameters here were chosen on the
step-5 development set (docs/benchmarks/step-5-lyrics.md).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 16000

Region = tuple[float, float]


@dataclass(frozen=True)
class ActivityParams:
    threshold: float = 0.5
    min_speech_s: float = 0.25
    min_silence_s: float = 0.3
    pad_s: float = 0.1
    # A song with less voiced time than this has no lyrics.
    min_total_s: float = 5.0


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
