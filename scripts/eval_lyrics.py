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
- gap candidates = recognised words whose midpoint lies in a gap between lyric
  annotations of HALLUCINATION_GAP s or more (shrunk by GAP_MARGIN at each end).
  Missing lyric annotations do not prove absence of vocals: candidates require
  reference/audio review before being called hallucinations;
- an instrumental track passes when the stage returns no lyrics.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
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


def seed_inference(entry_id: str, seed: int) -> int:
    """Seed each song before creating its Whisper worker (also valid after resume).

    CTranslate2's CPU RNG is thread-local: reseeding an existing model's workers
    is insufficient. Callers must create a fresh Whisper model after this call.
    """
    import random

    import ctranslate2
    import torch

    value = int.from_bytes(hashlib.sha256(f"{seed}:{entry_id}".encode()).digest()[:4], "big")
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    ctranslate2.set_random_seed(value)
    return value


class ReusedActivity:
    """Reuse one song's deterministic activity result across evaluation modes.

    LyricsStage still runs its real gate in both modes. The standalone pipeline
    timing script measures runtime; this avoids repeating VAD for model comparisons.
    """

    def __init__(self, detector: Any) -> None:
        self.detector = detector
        self.audio: np.ndarray | None = None
        self.found: list[tuple[float, float]] = []

    def regions(self, vocals: np.ndarray) -> list[tuple[float, float]]:
        if self.audio is not vocals:
            self.found = self.detector.regions(vocals)
            self.audio = vocals
        return self.found

    def has_vocals(self, regions: list[tuple[float, float]]) -> bool:
        return bool(self.detector.has_vocals(regions))


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


def line_errors(entry: dict[str, Any], timed: list[Any]) -> list[float | None]:
    from openbapo.analysis.lyrics.stage import fill_gaps

    spans = fill_gaps(timed, entry["duration"])
    first: dict[int, float] = {}
    for t, (start, _) in zip(timed, spans, strict=True):
        first.setdefault(t.line, start)
    # An absent prediction is a failure, never a dropped denominator.
    return [
        abs(first[i] - line["start"]) if i in first else None
        for i, line in enumerate(entry["lines"])
    ]


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
        return (jiwer.cer(r, h) if r else float("nan")), len(r)
    r, h = norm_en(ref), norm_en(hyp)
    return (jiwer.wer(r, h) if r else float("nan")), len(r.split())


def silent_gaps(entry: dict[str, Any]) -> list[tuple[float, float]]:
    edges = [0.0, *[t for r in entry["vocal_regions"] for t in r], entry["duration"]]
    gaps = []
    for start, end in zip(edges[::2], edges[1::2], strict=False):
        if end - start >= HALLUCINATION_GAP:
            gaps.append((start + GAP_MARGIN, end - GAP_MARGIN))
    return gaps


def gap_candidates(entry: dict[str, Any], recognition: Any) -> list[dict[str, Any]]:
    """Preserve timestamps for review; lyric gaps alone cannot certify hallucinations."""
    gaps = silent_gaps(entry)
    out = []
    for line in recognition.lines:
        for w in line.words:
            mid = (w.start + w.end) / 2
            if any(a <= mid <= b for a, b in gaps):
                out.append({"text": w.text, "start": w.start, "end": w.end})
    return out


def hallucinated(entry: dict[str, Any], recognition: Any) -> list[str]:
    """Legacy field name: these are unreviewed gap candidates, not proven hallucinations."""
    return [w["text"] for w in gap_candidates(entry, recognition)]


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
                arr = np.array([float("inf") if e is None else e for e in errs])
                finite = arr[np.isfinite(arr)]
                g[f"align/{name}"] = {
                    "lines": len(arr),
                    "within": float(np.mean(arr <= ACCEPTANCE_WITHIN)),
                    "missing": int((~np.isfinite(arr)).sum()),
                    "median": float(np.median(finite)) if len(finite) else None,
                    "p90": float(np.percentile(finite, 90)) if len(finite) else None,
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
                "gap_candidate_words": int(sum(len(r["hallucinated"]) for r in rec)),
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
    if meta.get("notes"):
        lines[4:4] = [str(meta["notes"]), ""]
    for key, g in summary.items():
        cells = []
        n = ""
        for name in aligners:
            a = g.get(f"align/{name}")
            if a:
                n = str(a["lines"])
                median = f"{a['median']:.2f} s" if a["median"] is not None else "无预测"
                cells.append(
                    f"{a['within']:.3f}（已预测中位 {median}；缺失 {a.get('missing', 0)} 行）"
                )
            else:
                cells.append("—")
        if n:
            lines.append(f"| {key} | {g['songs']} | {n} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 识别",
        "",
        "| 集合 | 歌数 | 字 / 词错率 | 语种判对 | 未标注区间词数，待核验（歌数） |",
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
    parser.add_argument("--resume", action="store_true", help="resume an identical interrupted run")
    parser.add_argument(
        "--seed", type=int, default=0, help="base seed; each song gets a stable seed"
    )
    args = parser.parse_args()

    from lyrics_candidates import WhisperAligner, mms_aligner

    from openbapo.analysis.device import resolve_device
    from openbapo.analysis.lyrics.activity import ActivityParams, SileroActivity
    from openbapo.analysis.lyrics.align import wav2vec2_aligner
    from openbapo.analysis.lyrics.recognize import Recognition, WhisperRecognizer, load_whisper
    from openbapo.analysis.lyrics.stage import LyricsStage
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
    if not entries:
        parser.error("no entries match the selected split, set and language")

    names = [n for n in args.aligners.split(",") if n]
    if not names:
        parser.error("at least one aligner is required")
    factories = {
        "wav2vec2": lambda: wav2vec2_aligner(device),
        "mms": lambda: mms_aligner(device),
        "whisper": lambda: WhisperAligner(device),
    }
    aligners = {n: factories[n]() for n in names}
    detector = SileroActivity(ActivityParams(context_gap_s=30.0))
    activity = ReusedActivity(detector)
    alignment_detector = SileroActivity(ActivityParams(energy_relative_db=-15.0))
    alignment_activity = ReusedActivity(alignment_detector)
    recognizer = None
    separator = DemucsSeparator(device=device)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial.json")
    source = Path(__file__).resolve().parents[1] / "src/openbapo/analysis/lyrics"
    signature = {
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
            + b"".join(p.read_bytes() for p in sorted(source.glob("*.py")))
        ).hexdigest(),
        "ids": [e["id"] for e in entries],
        "aligners": names,
        "recognition": not args.no_recognition,
        "random_seed": args.seed,
    }
    rows = []
    if args.resume and partial.exists():
        previous = json.loads(partial.read_text())
        if previous["signature"] != signature:
            parser.error("the interrupted run used different code, data or settings")
        rows = previous["songs"]
    done = {row["id"] for row in rows}
    for k, entry in enumerate(entries, 1):
        if entry["id"] in done:
            continue
        recognizer = None  # release the previous worker before setting the next seed
        song_seed = seed_inference(entry["id"], args.seed)
        if not args.no_recognition or entry["condition"] == "instrumental":
            recognizer = WhisperRecognizer(load_whisper(device=device))
        vocals, sep_s = load_vocals(entry, root, separator)
        regions = activity.regions(vocals)
        row: dict[str, Any] = {
            "id": entry["id"],
            "set": entry["set"],
            "condition": entry["condition"],
            "lang": entry["lang"],
            "duration": entry["duration"],
            "random_seed": song_seed,
            "separate_seconds": sep_s,
            "voiced_seconds": sum(e - s for s, e in regions),
        }
        if entry["condition"] == "instrumental":
            result = LyricsStage(activity, recognizer, aligners[names[0]]).run(vocals)
            words = [s.text for s in result.syllables]
            no_lyrics = not words
            row.update(no_lyrics=no_lyrics, recognized=words[:50])
        else:
            lyrics_text = "\n".join(line["text"] for line in entry["lines"])
            row["align"] = {}
            for name, aligner in aligners.items():
                start = time.perf_counter()
                result = LyricsStage(
                    activity, None, aligner, alignment_activity=alignment_activity
                ).run(vocals, lyrics_text)
                timed = list(result.timed)
                spent = time.perf_counter() - start
                errors = line_errors(entry, timed)
                row["align"][name] = {
                    "seconds": spent,
                    "errors": [round(e, 3) if e is not None else None for e in errors],
                    "untimed": sum(t.start is None for t in timed),
                }
            row["alignment_voiced_seconds"] = sum(
                end - start for start, end in alignment_activity.regions(vocals)
            )
            if not args.no_recognition:
                start = time.perf_counter()
                result = LyricsStage(activity, recognizer, aligners[names[0]]).run(vocals)
                rec = result.recognition or Recognition("", ())
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
                    "gap_candidates": gap_candidates(entry, rec),
                }
        rows.append(row)
        temporary = partial.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"signature": signature, "songs": rows}, ensure_ascii=False)
        )
        temporary.replace(partial)
        brief = {
            n: f"{np.mean([e is not None and e <= ACCEPTANCE_WITHIN for e in a['errors']]):.2f}"
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
                "ctranslate2",
                "stable-ts",
                "silero-vad",
                "torch",
                "torchaudio",
                "transformers",
                "demucs",
            )
        },
        "aligners": names,
        "evaluation_protocol": 4,
        "activity_params": dataclasses.asdict(detector.params),
        "alignment_activity_params": dataclasses.asdict(alignment_detector.params),
        "ctc_activity_penalty": getattr(aligners.get("wav2vec2"), "activity_penalty", None),
        "run_signature": signature,
        "notes": "实际 LyricsStage；缺失行计失败；中文 CER 不插空格。"
        "歌词标注空隙中的词仅列为待核验候选，不能直接认定为幻听。"
        "逐曲固定随机种子并新建 Whisper 工作线程，便于中断续跑复现；不保证跨硬件一致。"
        "附歌词对齐允许能量补检；自动识别的语音区间按 30 秒空隙分组，每组至少 5 秒人声。"
        "评测复用各模式的人声检测结果。"
        "此处计时只用于评测，完整性能由 time_pipeline.py 单独测量。",
    }
    out.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "summary": summary, "songs": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    out.with_suffix(".md").write_text(markdown(summary, names, meta), encoding="utf-8")
    partial.unlink(missing_ok=True)
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
