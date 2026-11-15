"""Lyrics alignment and recognition on the step-5 test set (docs/plans/step-5.md, section 4).

Usage:
    python scripts/eval_lyrics.py --manifest DIR/manifest.json --split dev|test \
        [--aligners wav2vec2,whisper,mms] [--no-recognition] [--sets zh-test,en-test,...] \
        [--device auto] [--limit N] [--out docs/benchmarks/step-5-lyrics]

The manifest comes from scripts/lyrics_testset.py. Needs `pip install -e
".[analysis]" -r scripts/requirements-eval.txt`. Writes OUT.json (every number, per song) and
OUT.md (the summary tables); song titles stay out of both.

Protocol:
- dry vocals go to the aligner as they are; mixes are separated with Demucs
  first and the vocal stem is used, as in the pipeline;
- alignment gets the annotated lines as the lyrics text; a line's predicted
  start is the start of its first token; error = |predicted - annotated|;
- recognition runs on the voiced regions of the same vocals; CER for Chinese
  (Simplified characters, letters and digits; no punctuation or spaces),
  WER for English (lower case, no punctuation); English test songs are
  scored against the Jam-ALT transcripts;
- hallucination = recognised words whose midpoint lies inside an annotated
  stretch of HALLUCINATION_GAP s or more without singing (shrunk by
  GAP_MARGIN s at each end);
- an instrumental track passes when the stage returns no lyrics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ACCEPTANCE_WITHIN = 0.5
ACCEPTANCE_RATE = 0.90
HALLUCINATION_GAP = 3.0
GAP_MARGIN = 0.5
SR = 16000


def load_vocals(entry: dict[str, Any], root: Path, separator: Any) -> tuple[np.ndarray, float]:
    """16 kHz mono vocals and the seconds spent separating (0 for dry vocals)."""
    import soxr

    from openbapo.analysis.audio import decode

    audio = decode(root / entry["audio"])
    spent = 0.0
    if entry["condition"] == "dry":
        mono = audio.mono()
    else:
        start = time.perf_counter()
        stems = separator.separate(audio)
        spent = time.perf_counter() - start
        mono = stems.vocals.mean(axis=0)
        return soxr.resample(mono, stems.sample_rate, SR).astype(np.float32), spent
    return soxr.resample(mono, audio.sample_rate, SR).astype(np.float32), spent


def manifest_lines(entry: dict[str, Any]) -> list[Any]:
    from openbapo.lyrics import Line, tokenize

    lines = []
    for i, line in enumerate(entry["lines"]):
        tokens = tokenize(line["text"])
        if tokens:
            lines.append(Line(i, line["text"], tokens))
    return lines


def line_errors(entry: dict[str, Any], timed: list[Any]) -> list[float]:
    from openbapo.analysis.lyrics.stage import fill_gaps

    spans = fill_gaps(timed, entry["duration"])
    first: dict[int, float] = {}
    for t, (start, _) in zip(timed, spans, strict=True):
        first.setdefault(t.line, start)
    return [abs(first[i] - line["start"]) for i, line in enumerate(entry["lines"]) if i in first]


def norm_zh(text: str) -> str:
    from openbapo.analysis.lyrics.recognize import to_simplified

    return "".join(c for c in to_simplified(text).lower() if c.isalnum())


def norm_en(text: str) -> str:
    text = re.sub(r"[^\w' ]+", " ", text.lower().replace("’", "'"))
    return " ".join(text.split())


def error_rate(lang: str, ref: str, hyp: str) -> tuple[float, int]:
    import jiwer

    if lang == "zh":
        r, h = norm_zh(ref), norm_zh(hyp)
        return (jiwer.cer(" ".join(r), " ".join(h)) if r else float("nan")), len(r)
    r, h = norm_en(ref), norm_en(hyp)
    return (jiwer.wer(r, h) if r else float("nan")), len(r.split())


def silent_gaps(entry: dict[str, Any]) -> list[tuple[float, float]]:
    edges = [0.0, *[t for r in entry["vocal_regions"] for t in r], entry["duration"]]
    gaps = []
    for start, end in zip(edges[::2], edges[1::2], strict=False):
        if end - start >= HALLUCINATION_GAP:
            gaps.append((start + GAP_MARGIN, end - GAP_MARGIN))
    return gaps


def hallucinated(entry: dict[str, Any], recognition: Any) -> list[str]:
    gaps = silent_gaps(entry)
    out = []
    for line in recognition.lines:
        for w in line.words:
            mid = (w.start + w.end) / 2
            if any(a <= mid <= b for a, b in gaps):
                out.append(w.text)
    return out


def summarise(rows: list[dict[str, Any]], aligners: list[str]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        lang = f"/{row['lang']}" if row["lang"] else ""
        groups[f"{row['set']}{lang}/{row['condition']}"].append(row)
    out: dict[str, Any] = {}
    for key, items in sorted(groups.items()):
        g: dict[str, Any] = {"songs": len(items)}
        for name in aligners:
            errs = [e for r in items for e in r.get("align", {}).get(name, {}).get("errors", [])]
            if errs:
                arr = np.array(errs)
                g[f"align/{name}"] = {
                    "lines": len(arr),
                    "within": float(np.mean(arr <= ACCEPTANCE_WITHIN)),
                    "median": float(np.median(arr)),
                    "p90": float(np.percentile(arr, 90)),
                    "seconds_per_song": float(
                        np.mean([r["align"][name]["seconds"] for r in items])
                    ),
                }
        rec = [r["recognition"] for r in items if "recognition" in r]
        if rec:
            units = sum(r["ref_units"] for r in rec)
            g["recognition"] = {
                "error_rate": (
                    float(sum(r["error_rate"] * r["ref_units"] for r in rec) / units)
                    if units
                    else None
                ),
                "language_correct": float(np.mean([r["language_ok"] for r in rec])),
                "hallucinated_words": int(sum(len(r["hallucinated"]) for r in rec)),
                "songs_with_hallucination": int(sum(bool(r["hallucinated"]) for r in rec)),
            }
        instr = [r for r in items if r["condition"] == "instrumental"]
        if instr:
            g["no_lyrics_rate"] = float(np.mean([r["no_lyrics"] for r in instr]))
        out[key] = g
    return out


def markdown(summary: dict[str, Any], aligners: list[str], meta: dict[str, Any]) -> str:
    lines = [
        f"# 第 5 步歌词评测（{meta['split']}）",
        "",
        f"> 由 `scripts/eval_lyrics.py` 生成（{meta['date']}，{meta['device']}）。"
        "协议见脚本说明与 `docs/plans/step-5.md` 第 4 节。",
        "",
        "## 行级对齐（行起始误差 ≤ 0.5 秒的比例）",
        "",
        "| 集合 | 歌数 | 行数 | " + " | ".join(aligners) + " |",
        "|---|---|---|" + "---|" * len(aligners),
    ]
    for key, g in summary.items():
        cells = []
        n = ""
        for name in aligners:
            a = g.get(f"align/{name}")
            if a:
                n = str(a["lines"])
                cells.append(f"{a['within']:.3f}（中位 {a['median']:.2f} s）")
            else:
                cells.append("—")
        if n:
            lines.append(f"| {key} | {g['songs']} | {n} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 识别",
        "",
        "| 集合 | 歌数 | 字 / 词错率 | 语种判对 | 幻听词数（歌数） |",
        "|---|---|---|---|---|",
    ]
    for key, g in summary.items():
        r = g.get("recognition")
        if r:
            er = "—" if r["error_rate"] is None else f"{r['error_rate']:.3f}"
            lines.append(
                f"| {key} | {g['songs']} | {er} | {r['language_correct']:.2f} | "
                f"{r['hallucinated_words']}（{r['songs_with_hallucination']}） |"
            )
    instr = {k: g for k, g in summary.items() if "no_lyrics_rate" in g}
    if instr:
        lines += ["", "## 纯音乐返回无歌词", "", "| 集合 | 曲数 | 无歌词比例 |", "|---|---|---|"]
        lines += [f"| {k} | {g['songs']} | {g['no_lyrics_rate']:.2f} |" for k, g in instr.items()]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--sets", help="comma-separated sets to run (default: all in the split)")
    parser.add_argument("--lang", choices=("zh", "en"), help="only songs in this language")
    parser.add_argument("--aligners", default="wav2vec2,whisper,mms")
    parser.add_argument("--no-recognition", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, help="first N entries of each set (dry run)")
    parser.add_argument("--out", default="docs/benchmarks/step-5-lyrics")
    args = parser.parse_args()

    from lyrics_candidates import WhisperAligner, mms_aligner

    from openbapo.analysis.device import resolve_device
    from openbapo.analysis.lyrics.activity import SileroActivity
    from openbapo.analysis.lyrics.align import wav2vec2_aligner
    from openbapo.analysis.lyrics.recognize import WhisperRecognizer, load_whisper
    from openbapo.analysis.separate import DemucsSeparator

    device = resolve_device(args.device)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.manifest.parent
    splits = {"dev": ("dev",), "test": ("test", "ref")}[args.split]
    entries = [e for e in manifest["entries"] if e["split"] in splits]
    if args.sets:
        wanted = set(args.sets.split(","))
        entries = [e for e in entries if e["set"] in wanted]
    if args.lang:
        entries = [e for e in entries if e["lang"] == args.lang]
    if args.limit:
        seen: dict[str, int] = defaultdict(int)
        kept = []
        for e in entries:
            key = f"{e['set']}/{e['condition']}"
            if seen[key] < args.limit:
                kept.append(e)
                seen[key] += 1
        entries = kept

    names = [n for n in args.aligners.split(",") if n]
    whisper = load_whisper(device=device)
    factories = {
        "wav2vec2": lambda: wav2vec2_aligner(device),
        "mms": lambda: mms_aligner(device),
        "whisper": lambda: WhisperAligner(device),
    }
    aligners = {n: factories[n]() for n in names}
    activity = SileroActivity()
    recognizer = WhisperRecognizer(whisper)
    separator = DemucsSeparator(device=device)

    rows = []
    for k, entry in enumerate(entries, 1):
        vocals, sep_s = load_vocals(entry, root, separator)
        regions = activity.regions(vocals)
        row: dict[str, Any] = {
            "id": entry["id"],
            "set": entry["set"],
            "condition": entry["condition"],
            "lang": entry["lang"],
            "duration": entry["duration"],
            "separate_seconds": sep_s,
            "voiced_seconds": sum(e - s for s, e in regions),
        }
        if entry["condition"] == "instrumental":
            no_lyrics = not activity.has_vocals(regions)
            words: list[str] = []
            if not no_lyrics:
                rec = recognizer.recognize(vocals, regions)
                words = [w.text for line in rec.lines for w in line.words]
                no_lyrics = not words
            row.update(no_lyrics=no_lyrics, recognized=words[:50])
        else:
            lines = manifest_lines(entry)
            row["align"] = {}
            for name, aligner in aligners.items():
                start = time.perf_counter()
                timed = aligner.align(vocals, lines, regions)
                spent = time.perf_counter() - start
                errors = line_errors(entry, timed)
                row["align"][name] = {
                    "seconds": spent,
                    "errors": [round(e, 3) for e in errors],
                    "untimed": sum(t.start is None for t in timed),
                }
            if not args.no_recognition:
                start = time.perf_counter()
                rec = recognizer.recognize(vocals, regions)
                spent = time.perf_counter() - start
                ref = entry.get("reference_text") or "\n".join(x["text"] for x in entry["lines"])
                er, units = error_rate(entry["lang"], ref, rec.text)
                row["recognition"] = {
                    "seconds": spent,
                    "language": rec.language,
                    "language_ok": rec.language == entry["lang"],
                    "error_rate": er,
                    "ref_units": units,
                    "dropped_segments": rec.dropped,
                    "hallucinated": hallucinated(entry, rec),
                }
        rows.append(row)
        brief = {
            n: f"{np.mean(np.array(a['errors']) <= ACCEPTANCE_WITHIN):.2f}"
            for n, a in row.get("align", {}).items()
            if a["errors"]
        }
        rec_brief = row.get("recognition", {}).get("error_rate")
        print(
            f"[{k}/{len(entries)}] {entry['id']} align {brief} rec {rec_brief}"
            f"{' no_lyrics=' + str(row['no_lyrics']) if 'no_lyrics' in row else ''}",
            flush=True,
        )

    summary = summarise(rows, names)
    import importlib.metadata as md

    meta = {
        "split": args.split,
        "date": dt.date.today().isoformat(),
        "device": device if device == "cpu" else _gpu_name(),
        "python": platform.python_version(),
        "packages": {
            p: _version(md, p)
            for p in (
                "faster-whisper",
                "stable-ts",
                "silero-vad",
                "torch",
                "torchaudio",
                "transformers",
                "demucs",
            )
        },
        "aligners": names,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "summary": summary, "songs": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    out.with_suffix(".md").write_text(markdown(summary, names, meta), encoding="utf-8")
    print(markdown(summary, names, meta))
    return 0


def _version(md: Any, name: str) -> str:
    try:
        return str(md.version(name))
    except md.PackageNotFoundError:
        return ""


def _gpu_name() -> str:
    import torch

    return str(torch.cuda.get_device_name(0))


if __name__ == "__main__":
    sys.exit(main())
