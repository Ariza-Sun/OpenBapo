"""Expert E2: lv-chordia (Jiang, Chen, Li & Xia, ISMIR 2019; pip port 2025).

An ensemble of five networks that decompose a chord into root/triad, bass,
seventh and extensions. We take the averaged per-frame head probabilities
and let the package's own decoder compose them into log-probabilities for a
template listing our 14 qualities (minus `5`, which its triad head lacks),
then normalise. Code and weights are MIT (weights undeclared separately).
"""

import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from opencapo.analysis.audio import Audio
from opencapo.analysis.chords.fusion import ChordPosteriors
from opencapo.analysis.chords.vocab import N_LABELS, QUALITIES, parse_label
from opencapo.analysis.device import AUTO, resolve_device
from opencapo.analysis.separate import Stems

SAMPLE_RATE = 22050
HOP = 512


def template_text() -> str:
    """One C-rooted label per quality; the decoder shifts them to all roots."""
    return "".join(f"C:{quality.value}\n" for quality in QUALITIES)


def names_to_indices(names: list[str]) -> NDArray[np.int64]:
    """Map the decoder's label names (Harte, flats allowed) to our label indices."""
    from opencapo.analysis.chords.vocab import NO_CHORD

    out = np.zeros(len(names), dtype=np.int64)
    for i, name in enumerate(names):
        parsed = parse_label(name)
        out[i] = NO_CHORD if parsed is None else parsed.index
    return out


class LVChordiaExpert:
    name = "lvchordia"
    is_model = True

    def __init__(self, device: str = AUTO) -> None:
        self.device = resolve_device(device)
        self._ensemble: list[object] | None = None
        self._decoder: object | None = None

    def _load(self) -> tuple[list[object], object]:
        if self._ensemble is None or self._decoder is None:
            from lv_chordia.chord_recognition import load_ensemble
            from lv_chordia.extractors.xhmm_ismir import XHMMDecoder

            self._ensemble = list(load_ensemble(self.device.startswith("cuda")))
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
                f.write(template_text())
                template = f.name
            try:
                self._decoder = XHMMDecoder(template_file=template)
            finally:
                Path(template).unlink(missing_ok=True)
        return self._ensemble, self._decoder

    def estimate(self, audio: Audio, stems: Stems | None) -> ChordPosteriors:
        import scipy.io.wavfile as wavfile
        from lv_chordia.extractors import CQTV2
        from lv_chordia.mir import DataEntry, io

        ensemble, decoder = self._load()
        # The package reads features from a file; the temporary wav never
        # outlives this call.
        with tempfile.TemporaryDirectory(prefix="opencapo-lvchordia-") as tmp:
            wav = Path(tmp) / "mix.wav"
            wavfile.write(str(wav), audio.sample_rate, audio.mono())
            entry = DataEntry()
            entry.prop.set("sr", SAMPLE_RATE)
            entry.prop.set("hop_length", HOP)
            entry.append_file(str(wav), io.MusicIO, "music")
            entry.append_extractor(CQTV2, "cqt")
            per_net = [net.inference(entry.cqt) for net in ensemble]  # type: ignore[attr-defined]
        heads = [np.mean([p[i] for p in per_net], axis=0) for i in range(len(per_net[0]))]
        names, log_prob = decoder.get_chord_tag_obs(heads)  # type: ignore[attr-defined]
        log_prob = np.asarray(log_prob, dtype=np.float64)
        log_prob -= log_prob.max(axis=1, keepdims=True)
        probs = np.exp(log_prob)
        probs /= probs.sum(axis=1, keepdims=True)
        out = np.zeros((len(probs), N_LABELS))
        np.add.at(out.T, names_to_indices(list(names)), probs.T)
        times = (np.arange(len(probs)) * HOP / SAMPLE_RATE).astype(np.float64)
        return ChordPosteriors(times=times, probs=out)
