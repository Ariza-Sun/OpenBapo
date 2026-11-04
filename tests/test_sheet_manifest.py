"""The sample-set manifest: rebuilt from the files, it lists what is complete and its origin."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from opencapo.sheet import (
    build_manifest,
    load_manifest,
    parse_sheet,
    render,
    write_manifest,
    write_reference,
    write_wav,
)
from opencapo.sheet.manifest import MANIFEST_VERSION, review_count

SHEET = "title: 歌\nkey: D\ncapo: 2\nbpm: 120\ntime: 4/4\n---\nC | 1 2 3 4\nG | 5 - - -\n"


def put_song(folder: Path, name: str, *, rendered: bool = True, **render_args: object) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.sheet").write_text(SHEET, encoding="utf-8")
    if rendered:
        out = render(parse_sheet(SHEET), **render_args)  # type: ignore[arg-type]
        write_wav(folder / f"{name}.wav", out.samples, out.sample_rate)
        write_reference(folder / f"{name}.json", out.reference)


def test_manifest_lists_complete_songs_with_their_origin(tmp_path: Path) -> None:
    put_song(tmp_path / "a", "a", melody=False)  # read from print: has readings + review
    (tmp_path / "a" / "readings.json").write_text("[]", encoding="utf-8")
    (tmp_path / "a" / "review.md").write_text(
        "# 待复核小节\n\n- 小节 3: x\n- 小节 7: y\n", encoding="utf-8"
    )
    put_song(tmp_path, "b", melody=False)  # typed by hand, rendered flat in the root
    put_song(tmp_path / "c", "c", rendered=False)  # never rendered
    put_song(tmp_path / "e", "e", melody=False)  # typed by hand, rendered into DIR/e/ (sheet above)
    (tmp_path / "e.sheet").write_text((tmp_path / "e" / "e.sheet").read_text(encoding="utf-8"))
    (tmp_path / "e" / "e.sheet").unlink()
    (tmp_path / "d.json").write_text("{}", encoding="utf-8")  # not a reference, no sheet

    stamp = dt.datetime(2026, 11, 4, 12, 0, tzinfo=dt.UTC)
    manifest = write_manifest(tmp_path, now=stamp)
    assert manifest == load_manifest(tmp_path)
    assert manifest["version"] == MANIFEST_VERSION
    assert manifest["generated"] == "2026-11-04T12:00:00+00:00"
    assert manifest["render"] == {
        "accompaniment": "strum",
        "melody_voice": "piano",
        "melody": False,
        "bass": False,
        "seed": 0,
    }
    songs = manifest["songs"]
    assert [s["name"] for s in songs] == ["a", "b", "e"]
    a, b, e = songs
    assert e["source"] == "manual" and e["files"] == {
        "sheet": "e.sheet",
        "audio": "e/e.wav",
        "reference": "e/e.json",
    }
    assert a["files"] == {"sheet": "a/a.sheet", "audio": "a/a.wav", "reference": "a/a.json"}
    assert a["source"] == "ocr" and a["to_review"] == 2
    assert b["source"] == "manual" and b["to_review"] == 0 and b["files"]["sheet"] == "b.sheet"
    for s in songs:
        assert (s["title"], s["key"], s["mode"], s["capo"], s["bpm"], s["time"], s["bars"]) == (
            "歌", "D", "major", 2, 120.0, "4/4", 2
        )  # fmt: skip
        assert s["duration"] == pytest.approx(4.0)
    assert manifest["incomplete"] == [{"name": "c", "problem": "c: no c.wav next to c.sheet"}]
    assert set(manifest["code"]) == {"opencapo", "commit"}


def test_manifest_is_rebuilt_identically_and_render_differences_are_kept_per_song(
    tmp_path: Path,
) -> None:
    put_song(tmp_path, "a", melody=False)
    put_song(tmp_path, "b", melody=False, bass=True)
    stamp = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    first = build_manifest(tmp_path, now=stamp)
    assert first == build_manifest(tmp_path, now=stamp)
    assert first["render"] is None  # the songs differ, so there is no common setting
    assert [s["render"]["bass"] for s in first["songs"]] == [False, True]


def test_manifest_loading_rejects_missing_or_foreign_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"sheet_ocr\.py manifest"):
        load_manifest(tmp_path)
    (tmp_path / "manifest.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a sample-set manifest"):
        load_manifest(tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"version": 99, "songs": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="version 99"):
        load_manifest(tmp_path)


def test_review_count_reads_only_bar_lines(tmp_path: Path) -> None:
    assert review_count(tmp_path / "missing.md") == 0
    path = tmp_path / "review.md"
    path.write_text("# 待复核小节\n\n- 小节 1: a\n- note\n- 小节 2: b\n", encoding="utf-8")
    assert review_count(path) == 2
