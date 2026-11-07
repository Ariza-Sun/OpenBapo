"""Expert E1: BTC (Park, Choi, Jeon, Kim & Nam, ISMIR 2019), large vocabulary.

A bi-directional transformer over a 144-bin CQT, 170 classes (14 qualities x
12 roots, X, N). Its 14 qualities are ours with `minmaj7` in place of `5`.
The code is vendored (MIT); the checkpoint is downloaded once from the
authors' repository, pinned to a commit and verified by SHA-256 before it is
unpickled. Its license is not declared by the authors (docs/plans/backend.md).
"""

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from opencapo.analysis.audio import Audio
from opencapo.analysis.chords.fusion import ChordPosteriors
from opencapo.analysis.chords.vocab import N_LABELS, NO_CHORD, label_index
from opencapo.analysis.device import AUTO, resolve_device
from opencapo.analysis.separate import Stems
from opencapo.model import ChordQuality

CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/"
    "2682317be668032e6e4b269ded36adaa2ad57df0/test/btc_model_large_voca.pt"
)
CHECKPOINT_SHA256 = "1673d23f8f9a55ae7f9e8b80a51da616debb22675b8d8b67ea6ce0ef37b0ab51"
CHECKPOINT_NAME = "btc_model_large_voca.pt"

SAMPLE_RATE = 22050
HOP = 2048
N_BINS = 144
BINS_PER_OCTAVE = 24
TIMESTEP = 108  # frames per transformer window
CHUNK_SECONDS = 10.0  # the authors compute the CQT in 10 s pieces; kept for fidelity

MODEL_CONFIG = {
    "feature_size": 144,
    "timestep": TIMESTEP,
    "num_chords": 170,
    "input_dropout": 0.2,
    "layer_dropout": 0.2,
    "attention_dropout": 0.2,
    "relu_dropout": 0.2,
    "num_layers": 8,
    "num_heads": 4,
    "hidden_size": 128,
    "total_key_depth": 128,
    "total_value_depth": 128,
    "filter_size": 128,
    "loss": "ce",
    "probs_out": False,
}

# BTC's class layout: index = root * 14 + quality, then 168 = X, 169 = N.
BTC_QUALITIES = (
    "min", "maj", "dim", "aug", "min6", "maj6", "min7",
    "minmaj7", "maj7", "7", "dim7", "hdim7", "sus2", "sus4",
)  # fmt: skip


def cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "opencapo"


def checkpoint_path(download: bool = True) -> Path:
    """Local path of the verified checkpoint, downloading it on first use."""
    path = cache_dir() / CHECKPOINT_NAME
    if path.is_file() and _sha256(path) == CHECKPOINT_SHA256:
        return path
    if not download:
        raise FileNotFoundError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(CHECKPOINT_URL, tmp)
    if _sha256(tmp) != CHECKPOINT_SHA256:
        tmp.unlink()
        raise RuntimeError("BTC checkpoint download did not match the pinned SHA-256")
    tmp.replace(path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def label_map() -> NDArray[np.int64]:
    """(170,) BTC class index -> our label index; X is folded into N."""
    out = np.full(170, NO_CHORD, dtype=np.int64)
    for root in range(12):
        for q, name in enumerate(BTC_QUALITIES):
            if name == "minmaj7":
                # Not in our vocabulary: nearest is the minor triad.
                out[root * 14 + q] = label_index(root, ChordQuality.MIN)
            else:
                out[root * 14 + q] = label_index(root, ChordQuality(name))
    return out


def to_vocabulary(btc_probs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sum BTC's 170-way probabilities into our 169-way label space."""
    out = np.zeros((len(btc_probs), N_LABELS))
    np.add.at(out.T, label_map(), btc_probs.T)
    return out


def features(audio: Audio) -> NDArray[np.float64]:
    """Log-CQT exactly as the authors compute it, chunked in 10 s pieces."""
    import librosa

    y = audio.mono()
    if audio.sample_rate != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=audio.sample_rate, target_sr=SAMPLE_RATE)
    chunk = int(CHUNK_SECONDS * SAMPLE_RATE)
    pieces = []
    for start in range(0, max(len(y), 1), chunk):
        piece = y[start : start + chunk]
        if len(piece) == 0:
            break
        pieces.append(
            librosa.cqt(
                piece,
                sr=SAMPLE_RATE,
                n_bins=N_BINS,
                bins_per_octave=BINS_PER_OCTAVE,
                hop_length=HOP,
            )
        )
    cqt = np.concatenate(pieces, axis=1)
    return np.asarray(np.log(np.abs(cqt) + 1e-6).T, dtype=np.float64)


class BTCExpert:
    name = "btc"
    is_model = True

    def __init__(self, device: str = AUTO) -> None:
        self.device = resolve_device(device)
        self._model: object | None = None
        self._mean = 0.0
        self._std = 1.0

    def _load(self) -> object:
        if self._model is None:
            import torch

            from opencapo.analysis.chords.models.btc_vendor.btc_model import BTC_model

            path = checkpoint_path()
            # The pickle also holds numpy scalars, so weights_only cannot be
            # used; the file's hash was verified against the pinned value.
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = BTC_model(config=MODEL_CONFIG)  # type: ignore[no-untyped-call]
            model.load_state_dict(checkpoint["model"])
            model.to(self.device).eval()
            self._mean, self._std = float(checkpoint["mean"]), float(checkpoint["std"])
            self._model = model
        return self._model

    def estimate(self, audio: Audio, stems: Stems | None) -> ChordPosteriors:
        import torch

        model = self._load()
        feat = (features(audio) - self._mean) / self._std
        n_frames = len(feat)
        pad = (TIMESTEP - n_frames % TIMESTEP) % TIMESTEP
        feat = np.pad(feat, ((0, pad), (0, 0)))
        x = torch.tensor(feat, dtype=torch.float32, device=self.device)[None]
        probs = []
        with torch.no_grad():
            for t in range(len(feat) // TIMESTEP):
                hidden, _ = model.self_attn_layers(x[:, TIMESTEP * t : TIMESTEP * (t + 1)])  # type: ignore[attr-defined]
                logits = model.output_layer.output_projection(hidden)  # type: ignore[attr-defined]
                probs.append(torch.softmax(logits, -1)[0].cpu().numpy())
        btc_probs = np.concatenate(probs)[:n_frames].astype(np.float64)
        times = (np.arange(n_frames) * HOP / SAMPLE_RATE).astype(np.float64)
        return ChordPosteriors(times=times, probs=to_vocabulary(btc_probs))
