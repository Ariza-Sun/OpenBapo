import shutil
from pathlib import Path

import numpy as np
import pytest

from openbapo.analysis.audio import Audio, decode
from openbapo.analysis.separate import STEM_NAMES, Stems, stems_workdir


def zeros(frames: int = 100) -> Stems:
    return Stems(sample_rate=44100, **{n: np.zeros((2, frames), np.float32) for n in STEM_NAMES})


def test_stems_must_share_a_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        Stems(
            sample_rate=44100,
            vocals=np.zeros((2, 10), np.float32),
            drums=np.zeros((2, 10), np.float32),
            bass=np.zeros((2, 10), np.float32),
            other=np.zeros((2, 11), np.float32),
        )


def test_getitem_and_mix() -> None:
    stems = Stems(
        sample_rate=8000,
        vocals=np.full((1, 4), 1.0, np.float32),
        drums=np.full((1, 4), 2.0, np.float32),
        bass=np.full((1, 4), 4.0, np.float32),
        other=np.full((1, 4), 8.0, np.float32),
    )
    assert stems["bass"][0, 0] == 4.0
    with pytest.raises(KeyError):
        stems["mix"]
    rhythm = stems.mix("drums", "bass")
    assert isinstance(rhythm, Audio)
    assert rhythm.sample_rate == 8000
    np.testing.assert_array_equal(rhythm.samples, np.full((1, 4), 6.0, np.float32))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_write_produces_float_wavs_that_decode_exactly(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    stems = Stems(
        sample_rate=44100,
        **{n: rng.uniform(-1, 1, (2, 50)).astype(np.float32) for n in STEM_NAMES},
    )
    paths = stems.write(tmp_path)
    assert sorted(paths) == sorted(STEM_NAMES)
    for name, path in paths.items():
        back = decode(path)
        assert back.sample_rate == 44100
        np.testing.assert_array_equal(back.samples, stems[name])


def test_workdir_is_removed_on_success() -> None:
    with stems_workdir() as workdir:
        assert workdir.is_dir()
        zeros().write(workdir)
        assert len(list(workdir.iterdir())) == 4
    assert not workdir.exists()


def test_workdir_is_removed_on_error() -> None:
    with pytest.raises(RuntimeError), stems_workdir() as workdir:
        zeros().write(workdir)
        raise RuntimeError("model crashed")
    assert not workdir.exists()
