"""One command from a printed sheet to a checked .sheet file, with Claude Code as the reader.

The deterministic parts (layout, checks, assembly) live in `layout` and
`omr`; this module runs the loop around them. The reading of each row is
done by the Claude Code command-line tool in headless mode (`claude -p`),
which reads the image with its own Read tool under the user's own login,
so no API key is involved. The reader splits a row into bars itself
(decision 15: bar lines are not found by rule).

Octave dots and sixteenth groupings are the misreads the checks cannot
see (the bar still fills its beats), so every row is read twice, in
differently cut batches of enlarged row images, and the two readings are
compared bar by bar as music. Rows where they differ, or that hold a bar
failing a check, get a third reading on their own from a larger image;
the third reading settles it, and a row none of the three agree on goes
to review with all its bars.

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
BATCH_SIZE = 2  # rows per model call
PARALLEL = 4
ZOOM = 2  # magnification of the row images for the two full passes
ZOOM_AGAIN = 3  # ... and for the deciding reading of a disputed row

RowReading = list[BarReading]


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
        READING_PROMPT + "\n\n下面每行是谱子的一行：序号<TAB>图片路径。用 Read 工具逐张打开，"
        "逐行写出它的小节数组，不要跳过，不要凭上下文猜没看到的内容。"
        "最后只输出一个 JSON 数组，每个元素形如 "
        '{"row": 序号, "bars": [小节对象, ...]}，不要别的文字。\n\n' + listing + "\n"
    )


def header_prompt(path: str) -> str:
    return HEADER_PROMPT + f"\n用 Read 工具打开图片 {path} 再回答。\n"


def parse_batch(reply: str, expected: Sequence[int]) -> dict[int, RowReading]:
    """The reply's rows -> their bars (indices unset: they are assigned once rows are settled)."""
    data = extract_json(reply, "array")
    if not isinstance(data, list):
        raise ValueError("reply is not a JSON array")
    out: dict[int, RowReading] = {}
    for item in data:
        if not isinstance(item, dict) or "row" not in item:
            continue
        row = int(item["row"])
        bars = item.get("bars")
        if row not in expected or not isinstance(bars, list):
            continue
        out[row] = [
            BarReading(
                index=-1,
                chords=str(b.get("chords", "")).strip(),
                melody=str(b.get("melody", "")).strip(),
                notes=str(b.get("notes", "")).strip(),
                lyrics=str(b.get("lyrics", "")).strip(),
                row=row,
            )
            for b in bars
            if isinstance(b, dict)
        ]
    missing = [i for i in expected if i not in out or not out[i]]
    if missing:
        raise ValueError(f"reply lacks rows {missing}")
    return out


def zoomed(path: Path, factor: int = ZOOM) -> Path:
    from PIL import Image

    out = path.with_name(path.stem + f"-x{factor}.png")
    if not out.exists():
        image = Image.open(path)
        size = (image.width * factor, image.height * factor)
        image.resize(size, Image.Resampling.LANCZOS).save(out)
    return out


def same_reading(a: BarReading, b: BarReading, beats_per_bar: int) -> bool:
    """Equal as music (parsed chords and notes), not as text."""
    from opencapo.sheet.format import SheetError, parse_chords, parse_melody

    try:
        return parse_chords(a.chords, beats_per_bar) == parse_chords(b.chords, beats_per_bar) and (
            parse_melody(a.melody, beats_per_bar) == parse_melody(b.melody, beats_per_bar)
        )
    except SheetError:
        return a.chords == b.chords and a.melody == b.melody


def same_row(a: RowReading, b: RowReading, beats_per_bar: int) -> bool:
    return len(a) == len(b) and all(
        same_reading(x, y, beats_per_bar) for x, y in zip(a, b, strict=True)
    )


def row_checks(rows: dict[int, RowReading], beats_per_bar: int) -> set[int]:
    """Rows holding a bar that fails a deterministic check (ties are checked across rows)."""
    flat = flatten(rows)
    failing = {c.index for c in check_readings(flat, beats_per_bar) if c.index >= 0}
    return {flat[i].row for i in failing}


def flatten(rows: dict[int, RowReading]) -> list[BarReading]:
    """All bars in row order, numbered from 0."""
    out: list[BarReading] = []
    for row in sorted(rows):
        for bar in rows[row]:
            bar.index = len(out)
            out.append(bar)
    return out


@dataclass
class ReadResult:
    sheet_path: Path
    readings: list[BarReading]
    problems: list[Check]
    reread: list[int] = field(default_factory=list)  # rows that needed the deciding reading
    unsettled: list[int] = field(default_factory=list)  # ... and still had no two readings agree
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
    """Layout -> two readings of every row -> checks -> deciding reading of disputes -> assemble."""
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = worksheet(source, out_dir)
    pages = sheet["pages"]
    assert isinstance(pages, list)
    rows = [s for p in pages for s in p["systems"]]
    crops = {int(r["index"]): str(r["crop"]) for r in rows}
    log(f"{len(pages)} pages, {len(rows)} rows")
    if not rows:
        raise RuntimeError(f"{source}: no systems found on any page")

    header_json = extract_json(runner(header_prompt(str(pages[0]["header_crop"]))), "object")
    header: dict[str, object] = dict(header_json) if isinstance(header_json, dict) else {}
    (out_dir / "header.json").write_text(json.dumps(header, ensure_ascii=False), encoding="utf-8")
    beats_per_bar = int(str(header.get("time") or "4/4").split("/")[0])  # unprinted: assume 4/4
    log(f"header: {header}")

    indices = sorted(crops)
    enlarged = {i: str(zoomed(Path(crops[i]), ZOOM)) for i in indices}

    def read_batch(batch: list[int], paths: dict[int, str]) -> dict[int, RowReading]:
        items = [(i, paths[i]) for i in batch]
        last: Exception | None = None
        for _ in range(retries + 1):
            try:
                return parse_batch(runner(batch_prompt(items)), batch)
            except (ValueError, RuntimeError) as e:  # a garbled reply: ask again
                last = e
        raise RuntimeError(f"batch {batch[0]}-{batch[-1]} unreadable: {last}")

    # Two full passes with the batch boundaries shifted, so no row shares both
    # of its contexts with the same neighbours.
    passes: list[dict[int, RowReading]] = []
    for offset in (0, max(1, batch_size // 2)):
        head = indices[:offset]
        rest = indices[offset:]
        batches = ([head] if head else []) + [
            rest[i : i + batch_size] for i in range(0, len(rest), batch_size)
        ]
        readings: dict[int, RowReading] = {}
        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            for batch_readings in pool.map(lambda b: read_batch(b, enlarged), batches):
                readings.update(batch_readings)
        passes.append(readings)
    first, second = passes
    log(f"two readings of {len(indices)} rows in {time.perf_counter() - started:.0f} s")

    disputed = [i for i in indices if not same_row(first[i], second[i], beats_per_bar)]
    failing = row_checks(first, beats_per_bar) | row_checks(second, beats_per_bar)
    doubtful = sorted(set(disputed) | failing)
    chosen: dict[int, RowReading] = dict(first)
    unsettled: list[int] = []
    for index in doubtful:
        big = {index: str(zoomed(Path(crops[index]), ZOOM_AGAIN))}
        try:
            third = read_batch([index], big)[index]
        except RuntimeError as e:
            log(f"row {index}: deciding reading failed ({e})")
            unsettled.append(index)
            continue
        chosen[index] = third
        valid_before = [
            r for r in (first[index], second[index]) if not row_checks({index: r}, beats_per_bar)
        ]
        third_valid = not row_checks({index: third}, beats_per_bar)
        agrees = [r for r in valid_before if same_row(r, third, beats_per_bar)]
        # The third reading settles a dispute when it matches one of the two
        # valid readings, or when the earlier ones were invalid and it is not.
        if not third_valid or (valid_before and not agrees):
            unsettled.append(index)
            summary = " / ".join(
                " | ".join(b.melody for b in r) for r in (first[index], second[index], third)
            )
            for bar in third:
                bar.notes = f"三次读法不一致：{summary}。" + bar.notes
    ordered = flatten(chosen)
    for bar in ordered:
        bar.crop = crops[bar.row]
    if doubtful:
        log(f"deciding reading for {len(doubtful)} rows: {doubtful}; unsettled: {unsettled}")

    header_text = {
        "title": str(header.get("title", source.stem)),
        "key": str(header.get("key", "C")),
        "mode": "major",
        "capo": str(header.get("capo") or 0),
        "bpm": f"{bpm if bpm is not None else float(str(header.get('bpm') or 90)):g}",
        "time": str(header.get("time") or "4/4"),
    }
    built = assemble(header_text, ordered, beats_per_bar)
    sheet_path = out_dir / f"{source.stem}.sheet"
    sheet_path.write_text(built.text, encoding="utf-8")
    (out_dir / "readings.json").write_text(
        json.dumps([r.__dict__ for r in ordered], ensure_ascii=False, indent=1), encoding="utf-8"
    )
    problems = built.problems + [
        Check(b.index, "三次读法不一致，见 readings.json 的 notes")
        for b in ordered
        if b.row in unsettled
    ]
    write_review(out_dir / "review.md", problems, ordered)
    seconds = time.perf_counter() - started
    log(f"{sheet_path}: {len(ordered)} bars, {len(problems)} to review, {seconds:.0f} s")
    return ReadResult(sheet_path, ordered, problems, doubtful, unsettled, seconds, header)


def safe_stem(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text).strip("_") or "sheet"
