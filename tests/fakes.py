"""Stand-ins for the model adapters so the pipeline runs on CPU without weights."""

import numpy as np

from opencapo.analysis.audio import Audio
from opencapo.analysis.beats import BeatTimes
from opencapo.analysis.chords.fusion import ChordPosteriors
from opencapo.analysis.chords.vocab import N_LABELS
from opencapo.analysis.separate import STEM_NAMES, Stems


class FakeSeparator:
    """Puts the whole mix into `other`; the rest is silence."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def separate(self, audio: Audio) -> Stems:
        self.calls += 1
        if self.fail:
            raise RuntimeError("separator crashed")
        silence = np.zeros_like(audio.samples)
        return Stems(
            sample_rate=audio.sample_rate,
            **{name: silence for name in STEM_NAMES if name != "other"},
            other=audio.samples,
        )


class FakeTracker:
    """Evenly spaced beats at a fixed BPM, four per bar, from time 0."""

    def __init__(self, bpm: float = 120.0, beats_per_bar: int = 4) -> None:
        self.bpm = bpm
        self.beats_per_bar = beats_per_bar
        self.inputs: list[Audio] = []

    def track(self, audio: Audio) -> BeatTimes:
        self.inputs.append(audio)
        beats = np.arange(0.0, audio.duration, 60.0 / self.bpm)
        return BeatTimes(beats=beats, downbeats=beats[:: self.beats_per_bar])


class FakeChordExpert:
    """Emits a fixed schedule of (label index, seconds) as peaky frame posteriors."""

    def __init__(self, schedule: list[tuple[int, float]], name: str = "fake", p: float = 0.9):
        self.schedule = schedule
        self.name = name
        self.is_model = True
        self.p = p

    def estimate(self, audio: Audio, stems: Stems | None) -> ChordPosteriors:
        hop = 0.1
        times, rows = [], []
        t = 0.0
        for label, seconds in self.schedule:
            for _ in range(round(seconds / hop)):
                row = np.full(N_LABELS, (1 - self.p) / (N_LABELS - 1))
                row[label] = self.p
                times.append(t)
                rows.append(row)
                t += hop
        return ChordPosteriors(times=np.array(times), probs=np.array(rows))
