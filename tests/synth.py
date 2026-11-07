"""Synthetic audio for the rule experts: sine-wave chords and bass lines."""

from collections.abc import Sequence

import numpy as np

from opencapo.analysis.audio import Audio

SR = 22050


def midi_hz(midi: float) -> float:
    return float(440.0 * 2 ** ((midi - 69) / 12))


def tone(midi: float, seconds: float, *, gain: float = 0.2, harmonics: int = 1) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    y = np.zeros_like(t)
    for h in range(1, harmonics + 1):
        y += (gain / h) * np.sin(2 * np.pi * midi_hz(midi) * h * t)
    return y


def chord_block(midis: Sequence[float], seconds: float, **kw: float) -> np.ndarray:
    if not midis:
        return np.zeros(int(seconds * SR))
    return sum(tone(m, seconds, **kw) for m in midis)  # type: ignore[no-any-return]


def progression(blocks: Sequence[tuple[Sequence[float], float]], **kw: float) -> Audio:
    """Concatenate (midi notes, seconds) blocks into mono Audio at 22.05 kHz."""
    y = np.concatenate([chord_block(m, s, **kw) for m, s in blocks]).astype(np.float32)
    return Audio(samples=y[None, :], sample_rate=SR)


def to_stereo(audio: Audio) -> Audio:
    return Audio(samples=np.repeat(audio.samples, 2, axis=0), sample_rate=audio.sample_rate)


# Pitch classes to MIDI notes around octave 4 (C4 = 60) and octave 2 for bass.
def voicing(pitch_classes: Sequence[int], octave: int = 4) -> list[int]:
    return [12 * (octave + 1) + pc for pc in pitch_classes]
