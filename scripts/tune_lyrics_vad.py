"""Voice-activity settings on the step-5 development set (docs/plans/step-5.md, 2.3 and 4.2).

Usage:
    python scripts/tune_lyrics_vad.py --manifest DIR/manifest.json [--device auto] \
        [--out docs/benchmarks/step-5-vad]

Runs Silero VAD on the dev songs' vocals (dry as they are, mixes through
Demucs, as in the pipeline) and on the dev instrumentals, for a small grid of
settings. Reports frame recall / precision against the annotated singing
(50 ms frames) and the voiced seconds found on each instrumental, which must
stay under ActivityParams.min_total_s for "no lyrics".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_lyrics import load_vocals

GRID = [(t, s) for t in (0.3, 0.5, 0.7) for s in (0.3, 1.0)]
HOP = 0.05


def frames(regions: list[Any], duration: float) -> np.ndarray:
    out = np.zeros(int(duration / HOP) + 1, dtype=bool)
    for start, end in regions:
        out[int(start / HOP) : int(end / HOP) + 1] = True
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="docs/benchmarks/step-5-vad")
    args = parser.parse_args()

    from openbapo.analysis.device import resolve_device
    from openbapo.analysis.lyrics.activity import ActivityParams, SileroActivity
    from openbapo.analysis.separate import DemucsSeparator

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = [e for e in manifest["entries"] if e["split"] == "dev"]
    separator = DemucsSeparator(device=resolve_device(args.device))
    vocals = {e["id"]: load_vocals(e, args.manifest.parent, separator)[0] for e in entries}

    results = []
    for threshold, silence in GRID:
        params = ActivityParams(threshold=threshold, min_silence_s=silence, energy_relative_db=None)
        activity = SileroActivity(params)
        row: dict[str, Any] = {
            "threshold": threshold,
            "min_silence_s": silence,
            "instrumental_voiced_s": [],
        }
        for condition in ("dry", "mix"):
            recall, precision = [], []
            for e in (e for e in entries if e["condition"] == condition):
                found = activity.regions(vocals[e["id"]])
                ref, hyp = frames(e["vocal_regions"], e["duration"]), frames(found, e["duration"])
                n = min(len(ref), len(hyp))
                ref, hyp = ref[:n], hyp[:n]
                recall.append((ref & hyp).sum() / max(ref.sum(), 1))
                precision.append((ref & hyp).sum() / max(hyp.sum(), 1))
            row[condition] = {
                "recall": float(np.mean(recall)),
                "precision": float(np.mean(precision)),
            }
        for e in (e for e in entries if e["condition"] == "instrumental"):
            found = activity.regions(vocals[e["id"]])
            row["instrumental_voiced_s"].append(round(sum(b - a for a, b in found), 2))
        row["instrumental_no_lyrics"] = float(
            np.mean([s < params.min_total_s for s in row["instrumental_voiced_s"]])
        )
        results.append(row)
        print(json.dumps(row), flush=True)

    default = ActivityParams()
    lines = [
        "# 第 5 步人声活动检测参数（开发集）",
        "",
        "> 由 `scripts/tune_lyrics_vad.py` 生成。召回 / 精确率按 50 ms 帧对照标注的演唱区间；"
        f"纯音乐一列是检出的人声总秒数（低于 {default.min_total_s:g} 秒判为无歌词）。",
        "",
        "| 阈值 | 最短静音（秒） | 干声 召回 / 精确 | 加伴奏 召回 / 精确 "
        "| 纯音乐判无歌词 | 纯音乐检出秒数（最大） |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        mark = (
            " **（采用）**"
            if (r["threshold"], r["min_silence_s"]) == (default.threshold, default.min_silence_s)
            else ""
        )
        lines.append(
            f"| {r['threshold']}{mark} | {r['min_silence_s']} | "
            f"{r['dry']['recall']:.3f} / {r['dry']['precision']:.3f} | "
            f"{r['mix']['recall']:.3f} / {r['mix']['precision']:.3f} | "
            f"{r['instrumental_no_lyrics']:.2f} | {max(r['instrumental_voiced_s']):.1f} |"
        )
    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
