"""Audio decoding through the ffmpeg command-line tools.

Every supported container (mp3 / m4a / ogg / opus / wav / flac) goes through
the same path: ffprobe for a quick length check, then ffmpeg decoding to
float32 PCM in memory. Nothing decoded is ever written to disk here.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from openbapo.errors import InputError

# Requirements 5.1: songs longer than this are rejected, not truncated.
MAX_DURATION_SECONDS = 8 * 60

SAMPLE_RATE = 44100
CHANNELS = 2


@dataclass(frozen=True)
class Audio:
    """Decoded PCM audio: `samples` has shape (channels, frames), values in [-1, 1]."""

    samples: NDArray[np.float32]
    sample_rate: int

    @property
    def duration(self) -> float:
        return int(self.samples.shape[1]) / self.sample_rate

    def mono(self) -> NDArray[np.float32]:
        return np.asarray(self.samples.mean(axis=0), dtype=np.float32)


def _format_duration(seconds: float) -> str:
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes}:{rest:02d}"


def _run(command: list[str], path: Path) -> bytes:
    try:
        proc = subprocess.run(command, capture_output=True, check=False)
    except FileNotFoundError as e:
        raise InputError(f"{command[0]} is not installed; install ffmpeg to decode audio") from e
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        reason = detail[-1] if detail else f"{command[0]} exited with {proc.returncode}"
        raise InputError(f"cannot decode {path.name}: {reason}")
    return proc.stdout


@dataclass(frozen=True)
class Probe:
    duration: float
    channels: int


def probe(path: Path) -> Probe:
    """Length and channel count as reported by the container, without decoding."""
    out = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels:format=duration",
            "-of",
            "default=noprint_wrappers=1",
            str(path),
        ],
        path,
    )
    fields = dict(
        line.split("=", 1) for line in out.decode("ascii", "replace").splitlines() if "=" in line
    )
    try:
        return Probe(duration=float(fields["duration"]), channels=int(fields["channels"]))
    except (KeyError, ValueError) as e:
        raise InputError(f"cannot decode {path.name}: no audio stream found") from e


def probe_duration(path: Path) -> float:
    return probe(path).duration


def _check_duration(path: Path, seconds: float, max_duration: float) -> None:
    if seconds > max_duration:
        raise InputError(
            f"{path.name} is {_format_duration(seconds)} long; the limit is "
            f"{_format_duration(max_duration)}. Please trim it and upload again."
        )


def decode(
    path: Path,
    *,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
    max_duration: float | None = None,
) -> Audio:
    """Decode `path` to float32 PCM, rejecting files longer than `max_duration`.

    `max_duration` defaults to MAX_DURATION_SECONDS (requirements 5.1).
    """
    if max_duration is None:
        max_duration = MAX_DURATION_SECONDS
    if not path.is_file():
        raise InputError(f"audio file not found: {path}")
    # The container's own length is checked first so an over-long file is
    # rejected before any decoding work; the decoded length is checked again
    # because containers (VBR mp3 in particular) can under-report.
    info = probe(path)
    _check_duration(path, info.duration, max_duration)
    # Mono <-> stereo is done here rather than by ffmpeg, whose mono-to-stereo
    # upmix attenuates by 3 dB. Anything with more channels is downmixed by ffmpeg.
    decoded_channels = info.channels if info.channels in (1, 2) else channels
    raw = _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-vn",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            str(decoded_channels),
            "-ar",
            str(sample_rate),
            "-",
        ],
        path,
    )
    frames = np.frombuffer(raw, dtype="<f4")
    if frames.size == 0:
        raise InputError(f"cannot decode {path.name}: no audio samples")
    samples = frames.reshape(-1, decoded_channels).T
    if channels == 1 and decoded_channels == 2:
        samples = samples.mean(axis=0, keepdims=True)
    elif channels == 2 and decoded_channels == 1:
        samples = np.repeat(samples, 2, axis=0)
    elif channels != decoded_channels:
        raise ValueError("channels must be 1 or 2")
    audio = Audio(samples=np.ascontiguousarray(samples, dtype=np.float32), sample_rate=sample_rate)
    _check_duration(path, audio.duration, max_duration)
    return audio
