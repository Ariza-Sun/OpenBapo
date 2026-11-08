import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from conftest import convert, write_tone

from openbapo.analysis.audio import (
    MAX_DURATION_SECONDS,
    Audio,
    decode,
    probe,
    probe_duration,
)
from openbapo.errors import InputError

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_limit_is_eight_minutes() -> None:
    assert MAX_DURATION_SECONDS == 480


def test_decode_wav(tone: Path) -> None:
    audio = decode(tone)
    assert audio.sample_rate == 44100
    assert audio.samples.dtype == np.float32
    assert audio.samples.shape == (2, 3 * 44100)
    assert audio.duration == pytest.approx(3.0)
    # Mono input is duplicated to both channels; the tone peaks at 0.3.
    np.testing.assert_array_equal(audio.samples[0], audio.samples[1])
    assert np.abs(audio.samples).max() == pytest.approx(0.3, abs=0.01)
    assert audio.mono().shape == (3 * 44100,)


@pytest.mark.parametrize("extension", ["mp3", "m4a", "ogg", "opus", "flac"])
def test_decode_every_supported_container(tone: Path, tmp_path: Path, extension: str) -> None:
    audio = decode(convert(tone, tmp_path / f"song.{extension}"))
    # Lossy codecs pad the start or end slightly; the length must still be close.
    assert audio.duration == pytest.approx(3.0, abs=0.15)
    assert audio.samples.shape[0] == 2


def test_resamples_to_requested_rate(tone: Path) -> None:
    audio = decode(tone, sample_rate=22050, channels=1)
    assert audio.samples.shape == (1, 22050 * 3)


def test_probe(tone: Path) -> None:
    assert probe_duration(tone) == pytest.approx(3.0, abs=0.01)
    assert probe(tone).channels == 1


def test_stereo_to_mono_averages(tone: Path, tmp_path: Path) -> None:
    stereo = tmp_path / "stereo.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(tone),
            "-af",
            "pan=stereo|c0=c0|c1=0*c0",
            str(stereo),
        ],
        check=True,
    )
    assert probe(stereo).channels == 2
    left = decode(stereo).samples[0]
    mono = decode(stereo, channels=1).samples[0]
    np.testing.assert_allclose(mono, left / 2, atol=1e-6)


def test_no_audio_stream(tmp_path: Path) -> None:
    video = tmp_path / "silent.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:d=1",
            str(video),
        ],
        check=True,
    )
    with pytest.raises(InputError, match="no audio stream"):
        decode(video)


def test_rejects_too_long(tone: Path) -> None:
    with pytest.raises(InputError, match=r"is 0:03 long; the limit is 0:02"):
        decode(tone, max_duration=2.0)


def test_rejects_when_container_under_reports(tone: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("openbapo.analysis.audio.probe_duration", lambda _: 1.0)
    with pytest.raises(InputError, match="limit"):
        decode(tone, max_duration=2.0)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="not found"):
        decode(tmp_path / "nope.mp3")


def test_corrupt_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"\x00" * 1024)
    with pytest.raises(InputError, match=r"cannot decode bad\.mp3"):
        decode(bad)


def test_ffmpeg_missing(tone: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    with pytest.raises(InputError, match="install ffmpeg"):
        decode(tone)


def test_audio_duration_property() -> None:
    audio = Audio(samples=np.zeros((2, 44100), dtype=np.float32), sample_rate=22050)
    assert audio.duration == 2.0


def test_write_tone_matches_script(tmp_path: Path) -> None:
    assert probe_duration(write_tone(tmp_path / "t.wav", seconds=1.5)) == pytest.approx(
        1.5, abs=0.01
    )
