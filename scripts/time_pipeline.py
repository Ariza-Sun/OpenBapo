"""Per-stage timing of the analysis pipeline on one song (step-1 acceptance).

Usage:
    python scripts/time_pipeline.py SONG.mp3 [--device auto] [--repeat 3] \
        [--separators htdemucs,htdemucs_ft] [--lyrics] [--out docs/benchmarks/step-1-timing]

The first run of each configuration is reported separately as "cold" (it
includes loading weights); the acceptance number is the median of the warm
runs. Model inference time depends on length, not content, so any ~4 minute
song will do; the song itself is not committed. Writes OUT.json and OUT.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import platform
import statistics
from pathlib import Path

ACCEPTANCE_SECONDS = 90.0  # step 1: separation + beats
ACCEPTANCE_CHORDS_SECONDS = 30.0  # step 2: chords + key
ANALYSIS_BUDGET_SECONDS = 300.0  # requirements: a 4-minute song in under 5 minutes, all stages


def run_once(
    song: Path, model: str, device: str, separator_cache: dict, chord_experts: str, lyrics: bool
) -> dict[str, float]:
    from openbapo.analysis.beats import BeatThisTracker
    from openbapo.analysis.pipeline import analyze_file
    from openbapo.analysis.separate import DemucsSeparator
    from openbapo.cli import build_chord_stage, build_lyrics_stage

    if model not in separator_cache:
        separator_cache[model] = DemucsSeparator(model=model, device=device)
    if "tracker" not in separator_cache:
        separator_cache["tracker"] = BeatThisTracker(device=device)
    if "chords" not in separator_cache:
        separator_cache["chords"] = build_chord_stage(chord_experts, device)
    if lyrics and "lyrics" not in separator_cache:
        separator_cache["lyrics"] = build_lyrics_stage(device)
    result = analyze_file(
        song,
        separator=separator_cache[model],
        tracker=separator_cache["tracker"],
        chord_stage=separator_cache["chords"],
        lyrics_stage=separator_cache.get("lyrics"),
    )
    timings = result.timings.as_dict()
    timings["beats_found"] = len(result.analysis.beats)
    timings["chords_found"] = len(result.analysis.chords)
    timings["syllables_found"] = len(result.analysis.lyrics)
    timings["bpm"] = result.analysis.tempo.bpm if result.analysis.tempo else float("nan")
    return timings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("song", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repeat", type=int, default=3, help="warm runs per configuration")
    parser.add_argument("--separators", default="htdemucs,htdemucs_ft")
    parser.add_argument("--chord-experts", default="btc,lvchordia,chroma", help="'none' to skip")
    parser.add_argument("--lyrics", action="store_true", help="include the lyrics stage (step 5)")
    parser.add_argument("--label", help="name to report instead of the file name")
    parser.add_argument("--out", default="docs/benchmarks/step-1-timing")
    args = parser.parse_args(argv)

    from openbapo.analysis.audio import probe
    from openbapo.analysis.device import resolve_device

    device = resolve_device(args.device)
    duration = probe(args.song).duration
    cache: dict = {}
    results = []
    for model in args.separators.split(","):
        cold = run_once(args.song, model, device, cache, args.chord_experts, args.lyrics)
        warm = [
            run_once(args.song, model, device, cache, args.chord_experts, args.lyrics)
            for _ in range(args.repeat)
        ]
        stages = ["decode", "separate", "beats", "chords", "key", "lyrics", "total"]
        results.append(
            {
                "separator": model,
                "cold": cold,
                "warm_runs": warm,
                "warm_median": {k: statistics.median(r[k] for r in warm) for k in stages},
                "beats_found": warm[-1]["beats_found"],
                "syllables_found": warm[-1]["syllables_found"],
                "bpm": warm[-1]["bpm"],
            }
        )
        print(model, json.dumps(results[-1]["warm_median"]))

    env = {
        "date": dt.date.today().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "packages": {
            n: importlib.metadata.version(n)
            for n in ("openbapo", "torch", "demucs", "beat-this", "lv-chordia")
        },
    }
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)

    payload = {
        "environment": env,
        "song": {"name": args.label or args.song.name, "duration_seconds": duration},
        "acceptance_total_seconds": ACCEPTANCE_SECONDS,
        "acceptance_chords_seconds": ACCEPTANCE_CHORDS_SECONDS,
        "analysis_budget_seconds": ANALYSIS_BUDGET_SECONDS,
        "lyrics": args.lyrics,
        "chord_experts": args.chord_experts,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    minutes, seconds = divmod(round(duration), 60)
    lines = [
        "# 流水线耗时（分轨 + 节拍 + 和弦 + 调性" + (" + 歌词）" if args.lyrics else "）"),
        "",
        f"由 `scripts/time_pipeline.py` 生成于 {env['date']}。",
        f"设备：{env.get('gpu') or device}；Python {env['python']}；"
        + "、".join(f"{k} {v}" for k, v in env["packages"].items()),
        "",
        f"歌曲时长 {minutes}:{seconds:02d}（文件不提交）。每个配置先跑 1 次冷启动（含加载权重），"
        f"再跑 {args.repeat} 次取中位数。验收：第 1 步分轨 + 节拍 ≤ {ACCEPTANCE_SECONDS:.0f} 秒，"
        f"第 2 步和弦 + 调性 ≤ {ACCEPTANCE_CHORDS_SECONDS:.0f} 秒（看 `htdemucs` 行的热启动数字）。"
        f"和弦专家：{args.chord_experts}。"
        + (
            f"歌词环节（人声活动 + 识别 + 对齐）计入；全部分析环节合计的预算是 "
            f"{ANALYSIS_BUDGET_SECONDS:.0f} 秒（需求：4 分钟歌曲 5 分钟内出谱）。"
            if args.lyrics
            else ""
        ),
        "",
        "| 分轨模型 | 解码 | 分轨 | 节拍 | 和弦 | 调性 | 歌词 | 合计（热） | 合计（冷） | 验收 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        w, c = r["warm_median"], r["cold"]
        step1_ok = w["separate"] + w["beats"] <= ACCEPTANCE_SECONDS
        step2_ok = w["chords"] + w["key"] <= ACCEPTANCE_CHORDS_SECONDS
        budget_ok = w["total"] <= ANALYSIS_BUDGET_SECONDS
        verdict = "通过" if step1_ok and step2_ok and budget_ok else "未通过"
        if r["separator"] != "htdemucs":
            verdict = "参考"
        cells = [
            r["separator"],
            f"{w['decode']:.1f} s",
            f"{w['separate']:.1f} s",
            f"{w['beats']:.1f} s",
            f"{w['chords']:.1f} s",
            f"{w['key']:.1f} s",
            f"{w['lyrics']:.1f} s" if args.lyrics else "—",
            f"**{w['total']:.1f} s**",
            f"{c['total']:.1f} s",
            verdict,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.with_suffix(".md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
