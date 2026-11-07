"""Rule expert E3: chroma template matching.

Chroma (energy per pitch class over time) is the "partial transcription"
a chord needs: which pitch classes are sounding, regardless of instrument
or octave. Each frame is scored against a binary template per chord and
turned into a posterior with a softmax. No training data is involved, so
this expert fails differently from the learned ones, which is what makes it
worth having in the committee. It is also the only source of power chords.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from opencapo.analysis.audio import Audio
from opencapo.analysis.chords.fusion import ChordPosteriors
from opencapo.analysis.chords.vocab import N_LABELS, NO_CHORD, chord_tones, index_label
from opencapo.analysis.separate import Stems

SAMPLE_RATE = 22050
HOP = 2048  # ~93 ms frames; chords do not change faster than beats


def templates() -> NDArray[np.float64]:
    """(N_LABELS - 1, 12) unit-norm chord-tone templates in label order."""
    out = np.zeros((NO_CHORD, 12))
    for index in range(NO_CHORD):
        root, quality = index_label(index)  # type: ignore[misc]
        for pc in chord_tones(root, quality):
            out[index, pc] = 1.0
    return np.asarray(out / np.linalg.norm(out, axis=1, keepdims=True))


@dataclass(frozen=True)
class Chromagram:
    times: NDArray[np.float64]  # frame centres, seconds
    chroma: NDArray[np.float64]  # (frames, 12), each row L2-normalised (zeros for silence)
    energy: NDArray[np.float64]  # (frames,) RMS, relative to the loudest frame


def chromagram(audio: Audio) -> Chromagram:
    import librosa

    y = audio.mono()
    if audio.sample_rate != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=audio.sample_rate, target_sr=SAMPLE_RATE)
    chroma = librosa.feature.chroma_cqt(y=y, sr=SAMPLE_RATE, hop_length=HOP).T
    rms = librosa.feature.rms(y=y, frame_length=HOP * 2, hop_length=HOP)[0]
    n = min(len(chroma), len(rms))
    chroma, rms = chroma[:n], rms[:n]
    norms = np.linalg.norm(chroma, axis=1, keepdims=True)
    chroma = np.where(norms > 0, chroma / np.maximum(norms, 1e-9), 0.0)
    energy = rms / max(float(rms.max()), 1e-9)
    times = librosa.frames_to_time(np.arange(n), sr=SAMPLE_RATE, hop_length=HOP)
    return Chromagram(times=np.asarray(times, dtype=np.float64), chroma=chroma, energy=energy)


def mean_chroma(gram: Chromagram) -> NDArray[np.float64]:
    """Energy-weighted average chroma of the whole piece, for key estimation."""
    weights = gram.energy[:, None]
    total = (gram.chroma * weights).sum(axis=0)
    norm = np.linalg.norm(total)
    return np.asarray(total / norm) if norm > 0 else np.zeros(12)


class ChromaTemplateExpert:
    """Cosine similarity to chord templates, softmaxed; silence becomes N."""

    name = "chroma"
    is_model = False

    def __init__(self, temperature: float = 0.05, silence_threshold: float = 0.02) -> None:
        self.temperature = temperature
        self.silence_threshold = silence_threshold
        self._templates = templates()

    def estimate(self, audio: Audio, stems: Stems | None) -> ChordPosteriors:
        # Drums only add noise to the chroma; everything else carries harmony.
        source = stems.mix("bass", "other", "vocals") if stems is not None else audio
        return self.from_chromagram(chromagram(source))

    def from_chromagram(self, gram: Chromagram) -> ChordPosteriors:
        scores = gram.chroma @ self._templates.T  # cosine similarity, (frames, 168)
        full = np.zeros((len(gram.times), N_LABELS))
        full[:, :NO_CHORD] = scores
        # Silence or near-silence: the N template wins outright.
        silent = gram.energy < self.silence_threshold
        full[silent, :NO_CHORD] = 0.0
        full[silent, NO_CHORD] = 1.0
        logits = full / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return ChordPosteriors(times=gram.times, probs=probs)
