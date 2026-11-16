"""Measure the cost of applying the existing activity gate to separated groups.

Uses independent dev features only. No parameter search and no stored stems.
The 5 s minimum comes from the existing song gate; 30 s is Whisper's context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lyrics_activity_dev import real_entries
from tune_lyrics_vad import frames

from openbapo.analysis.lyrics.activity import ActivityParams, supported_activity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jamendo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--asr-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    entries = real_entries(args.jamendo) + [
        e for e in json.loads(args.manifest.read_text())["entries"] if e["split"] == "dev"
    ]
    params = {"context_gap_s": 30.0, "min_total_s": ActivityParams().min_total_s}
    rows = []
    cache_unchanged = []
    for entry in entries:
        path = args.manifest.parent / entry["audio"]
        feature = json.loads((args.features / f"{entry['id']}.json").read_text())
        assert feature["audio_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        before = [tuple(r) for r in feature["speech"]["0.5"]]
        after = supported_activity(before, **params)
        truth = frames(entry["vocal_regions"], entry["duration"])
        old, new = (frames(r, entry["duration"]) for r in (before, after))
        removed = [r for r in before if r not in after]
        rows.append(
            {
                "id": entry["id"],
                "condition": entry["condition"],
                "reference_frames": int(truth.sum()),
                "before_correct_frames": int((truth & old).sum()),
                "after_correct_frames": int((truth & new).sum()),
                "removed_regions": removed,
                "removed_reference_seconds": sum(
                    max(0, min(b, y) - max(a, x))
                    for a, b in removed
                    for x, y in entry["vocal_regions"]
                ),
            }
        )
        if entry["condition"] != "real":
            raw = json.loads((args.asr_cache / f"{entry['id']}.json").read_text())
            assert raw["audio_sha256"] == feature["audio_sha256"]
            original = [tuple(r) for r in raw["regions"]]
            cache_unchanged.append(supported_activity(original, **params) == original)
    groups = {}
    for condition in ("real", "dry", "mix", "instrumental"):
        songs = [s for s in rows if s["condition"] == condition]
        ref = sum(s["reference_frames"] for s in songs)
        before = sum(s["before_correct_frames"] for s in songs)
        after = sum(s["after_correct_frames"] for s in songs)
        groups[condition] = {
            "entries": len(songs),
            "before_recall": before / ref if ref else None,
            "after_recall": after / ref if ref else None,
            "retained_detected_reference": after / before if before else None,
            "changed_entries": sum(bool(s["removed_regions"]) for s in songs),
            "removed_reference_seconds": sum(s["removed_reference_seconds"] for s in songs),
        }
    report = {
        "params": params,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "summary": groups,
        "original_asr_dev_entries": len(cache_unchanged),
        "original_asr_dev_unchanged": sum(cache_unchanged),
        "songs": rows,
        "notes": "Independent development only. Frame recall is pooled at 50 ms resolution; "
        "removed vocal seconds use original word intervals. Unchanged decoder inputs verified "
        "from saved metadata, not a new ASR run. Real isolated vocals can be lost.",
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# 第 5 步：孤立短人声区间过滤的开发检查",
        "",
        "对语音检测区间按 ≥30 秒空隙分组，每组沿用原有的至少 5 秒累计人声门槛。"
        "间隔时长不计入人声，重叠区间不重复计数。只用于自动识别；附歌词对齐不使用。"
        "本次未搜索参数，未读取英文验收曲目。",
        "",
        "| 数据 | 条目 | 修改前人声召回 | 修改后人声召回 | 丢失已标注人声 |",
        "|---|---|---|---|---|",
    ]
    for condition, s in groups.items():
        before = "—" if s["before_recall"] is None else f"{s['before_recall']:.4%}"
        after = "—" if s["after_recall"] is None else f"{s['after_recall']:.4%}"
        lines.append(
            f"| {condition} | {s['entries']} | {before} | {after} | "
            f"{s['removed_reference_seconds']:.3f} 秒 |"
        )
    lines += [
        "",
        f"原中英文及纯音乐开发集：{sum(cache_unchanged)}/{len(cache_unchanged)} 条"
        "的已缓存识别输入区间不变。"
        "这证明该过滤对这些输入没有影响，不冒充重新跑过全部模型。",
        "",
        "真实歌曲中被移除的已标注人声：",
        *[
            f"- {s['id']}：{s['removed_reference_seconds']:.3f} 秒。"
            for s in rows
            if s["removed_reference_seconds"] > 0
        ],
        "",
        "这是偏向减少误识别的取舍：真实的短句、呼喊或孤立演唱也可能被删去；"
        "不能宣称零漏词。提供歌词时保留原对齐逻辑。",
        "30 秒采用 [faster-whisper 1.2.1 的声学窗口](https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/feature_extractor.py)，"
        "5 秒沿用本项目既有无人声门槛；具体规则为本项目工程选择，并非上游推荐的通用阈值。",
    ]
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
