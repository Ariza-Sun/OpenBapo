import math
import struct
import subprocess
import wave
from pathlib import Path

import pytest

SAMPLE_RATE = 44100


def write_tone(path: Path, seconds: float = 3.0, *, frequency: float = 440.0) -> Path:
    """Write a mono 16-bit sine-wave WAV, like scripts/make_tone.py but in-process."""
    n = int(seconds * SAMPLE_RATE)
    frames = b"".join(
        struct.pack("<h", int(0.3 * 32767 * math.sin(2 * math.pi * frequency * i / SAMPLE_RATE)))
        for i in range(n)
    )
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(frames)
    return path


def convert(src: Path, dst: Path) -> Path:
    """Transcode with ffmpeg; the container is chosen from dst's extension."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(src), str(dst)],
        check=True,
        capture_output=True,
    )
    return dst


@pytest.fixture
def tone(tmp_path: Path) -> Path:
    return write_tone(tmp_path / "Example Song.wav")
