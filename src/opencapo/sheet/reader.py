"""One command from a printed sheet to a checked .sheet file, with Claude Code as the reader.

The deterministic parts (layout, checks, assembly) live in `layout` and
`omr`; this module runs the loop around them. The reading of each crop is
done by the Claude Code command-line tool in headless mode (`claude -p`),
which reads the image with its own Read tool under the user's own login,
so no API key is involved.

Octave dots and sixteenth groupings are the misreads the checks cannot
see (the bar still fills its beats), so every bar is read twice, in
differently cut batches of enlarged crops, and the two readings are
compared note by note. Bars where they differ, or that fail a check, get
a third reading on their own from a crop three times the size; the third
reading settles it, and a bar none of the three agree on goes to review.

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
BATCH_SIZE = 5
PARALLEL = 4
ZOOM = 2  # magnification of the crops for the two full passes
ZOOM_AGAIN = 3  # ... and for the deciding reading of a disputed bar


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


@dataclass
class ReadResult:
    sheet_path: Path
    readings: list[BarReading]
    problems: list[Check]
    reread: list[int] = field(default_factory=list)  # bars that needed the deciding reading
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
    """Layout -> two readings of every bar -> checks -> deciding reading of disputes -> assemble."""
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
    beats_per_bar = int(str(header.get("time") or "4/4").split("/")[0])  # unprinted: assume 4/4
    log(f"header: {header}")

    indices = sorted(crops)
    enlarged = {i: str(zoomed(Path(crops[i]), ZOOM)) for i in indices}

    def read_batch(batch: list[int], paths: dict[int, str]) -> dict[int, BarReading]:
        items = [(i, paths[i]) for i in batch]
        last: Exception | None = None
        for _ in range(retries + 1):
            try:
                return parse_batch(runner(batch_prompt(items)), batch)
            except (ValueError, RuntimeError) as e:  # a garbled reply: ask again
                last = e
        raise RuntimeError(f"batch {batch[0]}-{batch[-1]} unreadable: {last}")

    # Two full passes with the batch boundaries shifted, so no bar shares both
    # of its contexts with the same neighbours.
    passes: list[dict[int, BarReading]] = []
    for offset in (0, batch_size // 2):
        head = indices[:offset]
        rest = indices[offset:]
        batches = ([head] if head else []) + [
            rest[i : i + batch_size] for i in range(0, len(rest), batch_size)
        ]
        readings: dict[int, BarReading] = {}
        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            for batch_readings in pool.map(lambda b: read_batch(b, enlarged), batches):
                readings.update(batch_readings)
        passes.append(readings)
    first, second = passes
    log(f"two readings of {len(indices)} bars in {time.perf_counter() - started:.0f} s")

    disputed = [i for i in indices if not same_reading(first[i], second[i], beats_per_bar)]
    failing = {
        c.index
        for r in (first, second)
        for c in check_readings([r[i] for i in indices], beats_per_bar)
        if c.index >= 0
    }
    doubtful = sorted(set(disputed) | failing)
    chosen: dict[int, BarReading] = dict(first)
    unsettled: list[int] = []
    for index in doubtful:
        big = {index: str(zoomed(Path(crops[index]), ZOOM_AGAIN))}
        try:
            third = read_batch([index], big)[index]
        except RuntimeError as e:
            log(f"bar {index}: deciding reading failed ({e})")
            unsettled.append(index)
            continue
        chosen[index] = third
        valid_before = [
            r for r in (first[index], second[index]) if not check_readings([r], beats_per_bar)
        ]
        third_valid = not check_readings([third], beats_per_bar)
        agrees = [r for r in valid_before if same_reading(r, third, beats_per_bar)]
        # The third reading settles a dispute when it matches one of the two
        # valid readings, or when the earlier ones were invalid and it is not.
        if not third_valid or (valid_before and not agrees):
            unsettled.append(index)
            third.notes = (
                f"三次读法不一致：{first[index].melody!r} / {second[index].melody!r} / "
                f"{third.melody!r}。" + third.notes
            )
    for index, r in chosen.items():
        r.crop = crops[index]
    ordered = [chosen[i] for i in indices]
    if doubtful:
        log(f"deciding reading for {len(doubtful)} bars: {doubtful}; unsettled: {unsettled}")

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
        Check(i, "三次读法不一致，见 readings.json 的 notes") for i in unsettled
    ]
    write_review(out_dir / "review.md", problems, ordered)
    seconds = time.perf_counter() - started
    log(f"{sheet_path}: {len(ordered)} bars, {len(problems)} to review, {seconds:.0f} s")
    return ReadResult(sheet_path, ordered, problems, doubtful, unsettled, seconds, header)


def safe_stem(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text).strip("_") or "sheet"
