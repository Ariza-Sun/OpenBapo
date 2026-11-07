import json
import subprocess
import sys
from pathlib import Path

import pytest
from fakes import FakeSeparator, FakeTracker

from opencapo.cli import main
from opencapo.model import Analysis

MAKE_TONE = Path(__file__).parents[1] / "scripts" / "make_tone.py"


def run(argv: list[str]) -> int:
    return main(argv, separator=FakeSeparator(), tracker=FakeTracker())


def test_make_tone_script(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    subprocess.run([sys.executable, str(MAKE_TONE), str(path), "--seconds", "1"], check=True)
    assert path.stat().st_size > 44100
    click = tmp_path / "click.wav"
    subprocess.run(
        [sys.executable, str(MAKE_TONE), str(click), "--click", "120", "--seconds", "2"],
        check=True,
    )
    assert click.stat().st_size > 2 * 44100


def test_analyze_writes_valid_json(tone: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    assert run(["analyze", str(tone), "--out", str(out)]) == 0
    result = Analysis.from_json(out.read_text(encoding="utf-8"))
    assert result.title == "Example Song"
    assert result.artist is None
    assert result.beats and result.tempo and result.duration


def test_analyze_to_stdout(tone: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["analyze", str(tone), "--out", "-", "--title", "示例", "--artist", "X"]) == 0
    result = Analysis.from_json(capsys.readouterr().out)
    assert (result.title, result.artist) == ("示例", "X")


def test_timing_to_stderr(tone: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["analyze", str(tone), "--out", "-", "--timing", "-"]) == 0
    timing = json.loads(capsys.readouterr().err)
    assert {"decode", "separate", "beats", "chords", "key", "total"} <= set(timing)


def test_timing_to_file(tone: Path, tmp_path: Path) -> None:
    timing = tmp_path / "timing.json"
    assert (
        run(["analyze", str(tone), "--out", str(tmp_path / "o.json"), "--timing", str(timing)]) == 0
    )
    assert "total" in json.loads(timing.read_text(encoding="utf-8"))


def test_extension_is_case_insensitive(tone: Path, tmp_path: Path) -> None:
    audio = tmp_path / "song.WAV"
    audio.write_bytes(tone.read_bytes())
    assert run(["analyze", str(audio), "--out", str(tmp_path / "out.json")]) == 0


def test_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["analyze", str(tmp_path / "nope.mp3"), "--out", "-"]) == 1
    assert "audio file not found" in capsys.readouterr().err


def test_unsupported_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    audio = tmp_path / "song.mid"
    audio.write_bytes(b"")
    assert run(["analyze", str(audio), "--out", "-"]) == 1
    err = capsys.readouterr().err
    assert "unsupported audio format '.mid'" in err
    assert "mp3" in err


def test_undecodable_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"")
    assert run(["analyze", str(audio), "--out", "-"]) == 1
    assert "cannot decode song.mp3" in capsys.readouterr().err


def test_too_long(
    tone: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("opencapo.analysis.audio.MAX_DURATION_SECONDS", 2.0)
    assert run(["analyze", str(tone), "--out", "-"]) == 1
    assert "the limit is 0:02" in capsys.readouterr().err


def test_unwritable_output(tone: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "missing-dir" / "out.json"
    assert run(["analyze", str(tone), "--out", str(out)]) == 1
    assert "cannot write" in capsys.readouterr().err


def test_missing_analysis_extra(
    tone: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("opencapo.analysis.deps.ANALYSIS_PACKAGES", ("no_such_pkg",))
    assert main(["analyze", str(tone), "--out", "-"]) == 1
    assert "opencapo[analysis]" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["analyze"],
        ["analyze", "song.mp3"],
        ["analyze", "s.mp3", "--out", "-", "--device", "tpu"],
    ],
)
def test_usage_errors_exit_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_cli_imports_nothing_heavy() -> None:
    # A core-only install must reach the install hint before numpy/torch are needed.
    code = "import sys, opencapo.cli; assert 'numpy' not in sys.modules, 'numpy imported'"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_module_entry_point_reports_usage() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "opencapo", "--version"], capture_output=True, text=True, check=True
    )
    assert proc.stdout.startswith("opencapo ")
