"""The manifest of a rendered song-book sample set (decision 12 in docs/plans/step-2.md).

A sample set is a directory holding, for every song, three files with
the same stem: `<name>.sheet` (the lead sheet as text), `<name>.wav`
(its rendering) and `<name>.json` (the timed reference). Sheets read
from print by `scripts/sheet_ocr.py` sit in a sub-directory each, next
to their `readings.json` and `review.md`; a sheet typed by hand may sit
one level above the folder `scripts/sheet_synth.py` rendered it into
(`DIR/<name>.sheet` + `DIR/<name>/<name>.wav`). `manifest.json` at the
root indexes them all.

The manifest is rebuilt from the files, never edited: running
`build_manifest` twice on the same directory gives the same result, and
a song whose render is missing is listed as incomplete rather than
silently dropped. The evaluation reads the sample set through the
manifest only, so what is scored is exactly what the manifest lists.
"""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path

from opencapo.sheet.format import SheetError, load_sheet

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"
PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def code_version(repo: Path | None = None) -> dict[str, str | None]:
    """The opencapo package version and, when run from a checkout, the git commit."""
    try:
        package = importlib.metadata.version("opencapo")
    except importlib.metadata.PackageNotFoundError:
        package = None
    commit = None
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo or Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode == 0:
            commit = proc.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return {"opencapo": package, "commit": commit}


def review_count(path: Path) -> int:
    """Bars listed in a review.md written by `opencapo.sheet.omr.write_review`."""
    if not path.exists():
        return 0
    return sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if re.match(r"- 小节 ", line)
    )


def song_entry(sheet_path: Path, root: Path) -> dict[str, object]:
    """The manifest entry for one complete song; raises if a file is missing or unreadable."""
    stem = sheet_path.stem
    # The render sits beside the sheet (sheet_ocr.py) or in a folder named
    # after it (sheet_synth.py SONG.sheet --out DIR/SONG).
    folder = sheet_path.parent
    if not (folder / f"{stem}.wav").exists() and (folder / stem / f"{stem}.wav").exists():
        folder = folder / stem
    audio = folder / f"{stem}.wav"
    reference_path = folder / f"{stem}.json"
    for p in (audio, reference_path):
        if not p.exists():
            raise FileNotFoundError(f"{stem}: no {p.name} next to {sheet_path.name}")
    sheet = load_sheet(sheet_path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if "chords" not in reference or "beats" not in reference:
        raise SheetError(f"{stem}: {reference_path.name} is not a sheet reference")
    source = "ocr" if (folder / "readings.json").exists() else "manual"
    return {
        "name": stem,
        "files": {
            "sheet": sheet_path.relative_to(root).as_posix(),
            "audio": audio.relative_to(root).as_posix(),
            "reference": reference_path.relative_to(root).as_posix(),
        },
        "title": sheet.title,
        "key": PITCH_NAMES[sheet.key],
        "mode": sheet.mode.value,
        "capo": sheet.capo,
        "bpm": sheet.bpm,
        "time": f"{sheet.beats_per_bar}/4",
        "bars": len(sheet.bars),
        "duration": float(reference.get("duration", sheet.duration)),
        "source": source,
        "to_review": review_count(folder / "review.md"),
        "render": reference.get("render"),
    }


def build_manifest(root: Path, *, now: dt.datetime | None = None) -> dict[str, object]:
    """Scan ROOT for complete songs; the result is a plain dict ready for JSON."""
    root = Path(root)
    songs: list[dict[str, object]] = []
    incomplete: list[dict[str, str]] = []
    for sheet_path in sorted(root.rglob("*.sheet")):
        try:
            songs.append(song_entry(sheet_path, root))
        except (FileNotFoundError, SheetError, ValueError, OSError) as e:
            incomplete.append({"name": sheet_path.stem, "problem": str(e)})
    renders = {json.dumps(s["render"], sort_keys=True) for s in songs}
    common_render = songs[0]["render"] if len(renders) == 1 else None
    stamp = (now or dt.datetime.now(dt.UTC)).isoformat(timespec="seconds")
    return {
        "version": MANIFEST_VERSION,
        "generated": stamp,
        "code": code_version(),
        "render": common_render,  # null when the songs were rendered with different settings
        "songs": songs,
        "incomplete": incomplete,
    }


def write_manifest(root: Path, *, now: dt.datetime | None = None) -> dict[str, object]:
    manifest = build_manifest(root, now=now)
    path = Path(root) / MANIFEST_NAME
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return manifest


def load_manifest(root: Path) -> dict[str, object]:
    path = Path(root) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `python scripts/sheet_ocr.py manifest {root}` first"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or "songs" not in manifest:
        raise ValueError(f"{path} is not a sample-set manifest")
    version = int(manifest.get("version", 0))
    if version != MANIFEST_VERSION:
        raise ValueError(f"{path}: manifest version {version}, expected {MANIFEST_VERSION}")
    return manifest
