"""Stem separation: mix in, vocals / drums / bass / other out.

Stems are an internal intermediate (requirements 5.2): they live in memory
and, when a later stage needs files, inside a temporary directory that is
removed when the pipeline finishes, whether or not it succeeded. Nothing here
writes anywhere else, and the data model has no place to store them.
"""

import struct
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from openbapo.analysis.audio import Audio
from openbapo.analysis.device import AUTO, resolve_device

STEM_NAMES = ("vocals", "drums", "bass", "other")

DEFAULT_DEMUCS_MODEL = "htdemucs"


@dataclass(frozen=True)
class Stems:
    """Four stems with the same shape (channels, frames) and sample rate."""

    sample_rate: int
    vocals: NDArray[np.float32]
    drums: NDArray[np.float32]
    bass: NDArray[np.float32]
    other: NDArray[np.float32]

    def __post_init__(self) -> None:
        shapes = {name: self[name].shape for name in STEM_NAMES}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"stems must share one shape, got {shapes}")

    def __getitem__(self, name: str) -> NDArray[np.float32]:
        if name not in STEM_NAMES:
            raise KeyError(name)
        stem: NDArray[np.float32] = getattr(self, name)
        return stem

    def mix(self, *names: str) -> Audio:
        """Re-mix a subset of stems, e.g. drums + bass for rhythm-only beat tracking."""
        total = sum((self[name] for name in names), np.zeros_like(self.vocals))
        return Audio(samples=np.asarray(total, dtype=np.float32), sample_rate=self.sample_rate)

    def write(self, directory: Path) -> dict[str, Path]:
        """Write each stem as a 32-bit float WAV into `directory`; returns name -> path."""
        paths = {}
        for name in STEM_NAMES:
            path = directory / f"{name}.wav"
            _write_float_wav(path, self[name], self.sample_rate)
            paths[name] = path
        return paths


def _write_float_wav(path: Path, samples: NDArray[np.float32], sample_rate: int) -> None:
    channels = samples.shape[0]
    data = np.ascontiguousarray(samples.T, dtype="<f4").tobytes()
    block_align = channels * 4
    header = b"".join(
        [
            b"RIFF",
            struct.pack("<I", 36 + len(data)),
            b"WAVE",
            b"fmt ",
            struct.pack(
                "<IHHIIHH", 16, 3, channels, sample_rate, sample_rate * block_align, block_align, 32
            ),
            b"data",
            struct.pack("<I", len(data)),
        ]
    )
    path.write_bytes(header + data)


@contextmanager
def stems_workdir() -> Iterator[Path]:
    """A temporary directory for stem files, removed on exit even after an error."""
    with tempfile.TemporaryDirectory(prefix="openbapo-stems-") as name:
        yield Path(name)


class Separator(Protocol):
    def separate(self, audio: Audio) -> Stems: ...


class DemucsSeparator:
    """Demucs v4 (Hybrid Transformer) through the `demucs` package.

    Weights download on first use into the Hugging Face / torch cache. Their
    license is undeclared by the authors; see docs/plans/backend.md.
    """

    def __init__(
        self, model: str = DEFAULT_DEMUCS_MODEL, device: str = AUTO, shifts: int = 0
    ) -> None:
        self.model = model
        self.device = resolve_device(device)
        # Random shifts average several passes (+~0.2 dB SDR at `shifts` times the
        # cost); off by default so output is deterministic.
        self.shifts = shifts
        self._separator: object | None = None

    def _load(self) -> object:
        if self._separator is None:
            from demucs.api import Separator as _DemucsSeparator

            self._separator = _DemucsSeparator(
                model=self.model, device=self.device, shifts=self.shifts, progress=False
            )
        return self._separator

    def separate(self, audio: Audio) -> Stems:
        import torch

        separator = self._load()
        wav = torch.from_numpy(np.ascontiguousarray(audio.samples))
        _, sources = separator.separate_tensor(wav, audio.sample_rate)  # type: ignore[attr-defined]
        stems = {
            name: np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)
            for name, tensor in sources.items()
        }
        missing = set(STEM_NAMES) - stems.keys()
        if missing:
            raise RuntimeError(f"model {self.model!r} did not produce stems {sorted(missing)}")
        sample_rate = int(separator.samplerate)  # type: ignore[attr-defined]
        return Stems(sample_rate=sample_rate, **{name: stems[name] for name in STEM_NAMES})
