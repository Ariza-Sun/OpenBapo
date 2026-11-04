"""One command from a printed sheet to a checked .sheet file, with Claude Code as the reader.

The deterministic parts (layout, checks, assembly) live in `layout` and
`omr`; this module runs the loop around them. The reading of each crop is
done by the Claude Code command-line tool in headless mode (`claude -p`),
which reads the image with its own Read tool under the user's own login,
so no API key is involved. Batches of bars run in parallel; bars that fail
the checks are cropped larger and read again on their own; what still
fails is listed for review.

The runner is injectable so the loop can be tested without a model.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from opencapo.sheet.layout import worksheet
from opencapo.sheet.omr import (
    HEADER_PROMPT,
    READING_PROMPT,
    BarReading,
    Check,
    assemble,
    check_readings,
    write_review,
)

Runner = Callable[[str], str]  # prompt -> model reply (text)

DEFAULT_MODEL = "claude-opus-5-5"
BATCH_SIZE = 10
PARALLEL = 3
ZOOM = 2  # magnification for the second reading of a doubtful bar


def claude_runner(model: str = DEFAULT_MODEL, timeout: float = 900.0) -> Runner:
    """A runner that calls `claude -p` with the Read tool allowed."""

    def run(prompt: str) -> str:
        result = subprocess.run(
            [
                "claude",
                "-p",
                "--model",
                model,
                "--allowedTools",
                "Read",
                "--output-format",
                "text",
                prompt,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"claude -p failed ({result.returncode}): {result.stderr[-500:]}")
        return result.stdout

    return run


def extract_json(text: str, kind: str = "array") -> object:
    """The JSON array / object in a reply that may carry prose or a code fence."""
    opener, closer = ("[", "]") if kind == "array" else ("{", "}")
    start, end = text.find(opener), text.rfind(closer)
    if start < 0 or end < start:
        raise ValueError(f"no JSON {kind} in reply: {text[:120]!r}")
    return json.loads(text[start : end + 1])


def batch_prompt(items: Sequence[tuple[int, str]]) -> str:
    listing = "\n".join(f"{index}\t{path}" for index, path in items)
    return (
        READING_PROMPT
        + "\n\n下面每行是一个小节：序号<TAB>图片路径。用 Read 工具逐张打开，逐个小节写出 JSON，"
        "不要跳过，不要凭上下文猜没看到的内容。最后只输出一个 JSON 数组，每个元素形如 "
        '{"index": 序号, "chords": "...", "melody": "...", "notes": "..."}，不要别的文字。\n\n'
        + listing
        + "\n"
    )


def header_prompt(path: str) -> str:
    return HEADER_PROMPT + f"\n用 Read 工具打开图片 {path} 再回答。\n"


def parse_batch(reply: str, expected: Sequence[int]) -> dict[int, BarReading]:
    data = extract_json(reply, "array")
    if not isinstance(data, list):
        raise ValueError("reply is not a JSON array")
    out: dict[int, BarReading] = {}
    for item in data:
        if not isinstance(item, dict) or "index" not in item:
            continue
        index = int(item["index"])
        if index in expected:
            out[index] = BarReading(
                index=index,
                chords=str(item.get("chords", "")).strip(),
                melody=str(item.get("melody", "")).strip(),
                notes=str(item.get("notes", "")).strip(),
            )
    missing = [i for i in expected if i not in out]
    if missing:
        raise ValueError(f"reply lacks bars {missing}")
    return out


def zoomed(path: Path, factor: int = ZOOM) -> Path:
    from PIL import Image

    image = Image.open(path)
    out = path.with_name(path.stem + f"-x{factor}.png")
    image.resize((image.width * factor, image.height * factor), Image.Resampling.LANCZOS).save(out)
    return out


@dataclass
class ReadResult:
    sheet_path: Path
    readings: list[BarReading]
    problems: list[Check]
    reread: list[int] = field(default_factory=list)
    seconds: float = 0.0
    header: dict[str, object] = field(default_factory=dict)


def read_sheet(
    source: Path,
    out_dir: Path,
    *,
    runner: Runner,
    bpm: float | None = None,
    batch_size: int = BATCH_SIZE,
    parallel: int = PARALLEL,
    retries: int = 1,
    log: Callable[[str], None] = print,
) -> ReadResult:
    """Layout -> read bars in batches -> check -> re-read doubtful bars -> assemble."""
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = worksheet(source, out_dir)
    pages = sheet["pages"]
    assert isinstance(pages, list)
    bars = [b for p in pages for s in p["systems"] for b in s["bars"]]
    crops = {int(b["index"]): str(b["stacked"]) for b in bars}
    log(f"{len(pages)} pages, {len(bars)} bars")

    header_json = extract_json(runner(header_prompt(str(pages[0]["header_crop"]))), "object")
    header: dict[str, object] = dict(header_json) if isinstance(header_json, dict) else {}
    (out_dir / "header.json").write_text(json.dumps(header, ensure_ascii=False), encoding="utf-8")
    beats_per_bar = int(str(header.get("time", "4/4")).split("/")[0])
    log(f"header: {header}")

    indices = sorted(crops)
    batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]

    def read_batch(batch: list[int]) -> dict[int, BarReading]:
        items = [(i, crops[i]) for i in batch]
        last: Exception | None = None
        for _ in range(retries + 1):
            try:
                return parse_batch(runner(batch_prompt(items)), batch)
            except (ValueError, RuntimeError) as e:  # a garbled reply: ask again
                last = e
        raise RuntimeError(f"batch {batch[0]}-{batch[-1]} unreadable: {last}")

    readings: dict[int, BarReading] = {}
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        for batch_readings in pool.map(read_batch, batches):
            readings.update(batch_readings)
    for index, r in readings.items():
        r.crop = crops[index]
    ordered = [readings[i] for i in indices]
    log(f"read {len(ordered)} bars in {time.perf_counter() - started:.0f} s")

    # Second look, one bar at a time and twice the size, for bars that fail a check.
    doubtful = sorted({c.index for c in check_readings(ordered, beats_per_bar) if c.index >= 0})
    for index in doubtful:
        big = zoomed(Path(crops[index]))
        try:
            again = parse_batch(runner(batch_prompt([(index, str(big))])), [index])
        except (ValueError, RuntimeError) as e:
            log(f"bar {index}: second reading failed ({e})")
            continue
        again[index].crop = crops[index]
        readings[index] = again[index]
    ordered = [readings[i] for i in indices]
    if doubtful:
        log(f"re-read {len(doubtful)} bars: {doubtful}")

    header_text = {
        "title": str(header.get("title", source.stem)),
        "key": str(header.get("key", "C")),
        "mode": "major",
        "capo": str(header.get("capo") or 0),
        "bpm": f"{bpm if bpm is not None else float(str(header.get('bpm') or 90)):g}",
        "time": str(header.get("time", "4/4")),
    }
    built = assemble(header_text, ordered, beats_per_bar)
    sheet_path = out_dir / f"{source.stem}.sheet"
    sheet_path.write_text(built.text, encoding="utf-8")
    (out_dir / "readings.json").write_text(
        json.dumps([r.__dict__ for r in ordered], ensure_ascii=False, indent=1), encoding="utf-8"
    )
    write_review(out_dir / "review.md", built.problems, ordered)
    seconds = time.perf_counter() - started
    log(f"{sheet_path}: {len(ordered)} bars, {len(built.problems)} to review, {seconds:.0f} s")
    return ReadResult(sheet_path, ordered, built.problems, doubtful, seconds, header)


def safe_stem(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text).strip("_") or "sheet"
