"""Rule expert E4: the bass line, from the bass stem.

A transcriber listens to the bass first: it almost always plays the root,
and when it does not, that is where an inversion (D/F#) belongs. The bass
stem is monophonic enough that a low-band constant-Q transform with a
"lowest strong bin" rule recovers its pitch class per frame; votes are then
pooled per beat. This is the only true transcription in the chord stage.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from opencapo.analysis.audio import Audio
from opencapo.analysis.chords.fusion import BassEstimate
from opencapo.analysis.separate import Stems

SAMPLE_RATE = 22050
HOP = 2048
FMIN_MIDI = 24  # C1 (32.7 Hz)
N_BINS = 48  # four octaves, up to B4
BINS_PER_OCTAVE = 12


def bass_pitch_classes(
    audio: Audio, *, strong_ratio: float = 0.6, silence_threshold: float = 0.02
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Per-frame (times, pitch class or -1, strength in [0, 1]) of the bass stem."""
    import librosa

    y = audio.mono()
    if audio.sample_rate != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=audio.sample_rate, target_sr=SAMPLE_RATE)
    cqt = np.abs(
        librosa.cqt(
            y,
            sr=SAMPLE_RATE,
            hop_length=HOP,
            fmin=librosa.midi_to_hz(FMIN_MIDI),
            n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE,
        )
    ).T  # (frames, bins)
    times = np.asarray(
        librosa.frames_to_time(np.arange(len(cqt)), sr=SAMPLE_RATE, hop_length=HOP),
        dtype=np.float64,
    )
    peak = float(cqt.max()) if cqt.size else 0.0
    pitch_class = np.full(len(cqt), -1, dtype=np.int64)
    strength = np.zeros(len(cqt))
    if peak <= 0:
        return times, pitch_class, strength
    for i, frame in enumerate(cqt):
        top = float(frame.max())
        if top < silence_threshold * peak:
            continue
        # The lowest bin that is nearly as strong as the maximum: the
        # fundamental rather than a louder harmonic (a fifth or octave above).
        lowest = int(np.argmax(frame >= strong_ratio * top))
        pitch_class[i] = (FMIN_MIDI + lowest) % 12
        strength[i] = min(1.0, top / peak)
    return times, pitch_class, strength


def pool_per_beat(
    times: NDArray[np.float64],
    pitch_class: NDArray[np.int64],
    strength: NDArray[np.float64],
    beat_times: Sequence[float],
    end_time: float,
) -> BassEstimate:
    """Strength-weighted vote per beat; strength = winner's share of the beat's frames."""
    n = len(beat_times)
    out_pc = np.full(n, -1, dtype=np.int64)
    out_strength = np.zeros(n)
    if n == 0:
        return BassEstimate(out_pc, out_strength)
    edges = [*beat_times, max(end_time, beat_times[-1] + 1e-6)]
    bins = np.searchsorted(edges, times, side="right") - 1
    for i in range(n):
        mask = bins == i
        if not mask.any():
            continue
        votes = np.zeros(12)
        for pc, s in zip(pitch_class[mask], strength[mask], strict=True):
            if pc >= 0:
                votes[pc] += s
        if votes.sum() == 0:
            continue
        winner = int(np.argmax(votes))
        out_pc[i] = winner
        out_strength[i] = float(votes[winner] / mask.sum())
    return BassEstimate(out_pc, out_strength)


def bass_source(audio: Audio, stems: Stems | None) -> Audio:
    """The bass stem when stems exist, else the mix.

    Falling back to the mix when the stem is merely quiet was tried on
    GuitarSet (solo guitar): the lowest note of a guitar voicing is often
    not the root, so it marked spurious inversions and helped nothing.
    """
    if stems is None:
        return audio
    return Audio(samples=stems.bass, sample_rate=stems.sample_rate)


class BassLineDetector:
    def detect(self, bass: Audio, beat_times: Sequence[float], end_time: float) -> BassEstimate:
        times, pitch_class, strength = bass_pitch_classes(bass)
        return pool_per_beat(times, pitch_class, strength, beat_times, end_time)
