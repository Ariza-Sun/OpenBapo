"""Audit decoder words against VAD clips and annotation gaps without storing audio.

Raw decoder metadata goes to --cache outside the repository. No stems are saved.
Use --split dev for independent development, --ids for a diagnostic subset.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

from eval_lyrics import error_rate, load_vocals, seed_inference, silent_gaps


def summary(rows):
    groups = {}
    for row in rows:
        key = f"{row['lang'] or 'instrumental'}/{row['condition']}"
        groups.setdefault(key, []).append(row)
    result = {}
    for key, songs in sorted(groups.items()):
        units = sum(s.get("ref_units", 0) for s in songs)
        result[key] = {
            "songs": len(songs),
            "error_rate": sum(s.get("error_rate", 0) * s.get("ref_units", 0) for s in songs) / units
            if units
            else None,
            "gap_candidates": sum(w["annotation_gap"] for s in songs for w in s["suspect"]),
            "no_words": sum(s["stage_words"] == 0 for s in songs),
        }
    return result


class Recorder:
    def __init__(self, model):
        self.model = model
        self.segments = []

    def detect_language(self, *args, **kwargs):
        return self.model.detect_language(*args, **kwargs)

    def transcribe(self, *args, **kwargs):
        segments, info = self.model.transcribe(*args, **kwargs)
        self.segments = list(segments)
        return iter(self.segments), info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--ids")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--energy-relative-db", type=float)
    parser.add_argument("--context-gap-s", type=float)
    parser.add_argument("--seed", type=int, help="use a fresh seeded Whisper worker per song")
    parser.add_argument("--compare", type=Path, help="compare against a previous diagnostic report")
    parser.add_argument(
        "--repeat-cache", type=Path, help="verify seeded results against another run"
    )
    args = parser.parse_args()
    from openbapo.analysis.lyrics.activity import ActivityParams, SileroActivity
    from openbapo.analysis.lyrics.recognize import WhisperRecognizer, load_whisper
    from openbapo.analysis.separate import DemucsSeparator

    manifest = json.loads(args.manifest.read_text())
    entries = [e for e in manifest["entries"] if e["split"] == args.split]
    if args.ids:
        entries = [e for e in entries if e["id"] in args.ids.split(",")]
    if not entries:
        parser.error("no entries match the selected split and IDs")
    detector = SileroActivity(
        ActivityParams(energy_relative_db=args.energy_relative_db, context_gap_s=args.context_gap_s)
    )
    recorder = None
    recognizer = WhisperRecognizer(None)
    separator = DemucsSeparator(device=args.device)
    args.cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, entry in enumerate(entries, 1):
        audio_path = args.manifest.parent / entry["audio"]
        digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
        cached = args.cache / f"{entry['id']}.json"
        if cached.exists():
            raw = json.loads(cached.read_text())
            assert raw["audio_sha256"] == digest
            assert {"context_gap_s": None, **raw["activity_params"]} == dataclasses.asdict(
                detector.params
            )
            assert raw["recognize_params"] == dataclasses.asdict(recognizer.params)
            assert raw.get("base_seed") == args.seed
        else:
            song_seed = None
            if args.seed is not None:
                recorder = None
                recognizer = WhisperRecognizer(None)
                song_seed = seed_inference(entry["id"], args.seed)
            if recorder is None:
                recorder = Recorder(load_whisper(device=args.device))
                recognizer = WhisperRecognizer(recorder)
            vocals, _ = load_vocals(entry, args.manifest.parent, separator)
            regions = detector.regions(vocals)
            recognition = recognizer.recognize(vocals, regions)
            raw = {
                "id": entry["id"],
                "audio_sha256": digest,
                "activity_params": dataclasses.asdict(detector.params),
                "recognize_params": dataclasses.asdict(recognizer.params),
                "language": recognition.language,
                "base_seed": args.seed,
                "random_seed": song_seed,
                "regions": regions,
                "segments": [dataclasses.asdict(s) for s in recorder.segments] if regions else [],
            }
            cached.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n")
            del vocals
        gaps = silent_gaps(entry)
        suspect = []
        accepted_words = 0
        transcript = []
        for seg in raw["segments"]:
            p = recognizer.params
            if (
                seg["no_speech_prob"] > p.max_no_speech
                or seg["avg_logprob"] < p.min_avg_logprob
                or seg["compression_ratio"] > p.max_compression
            ):
                continue
            for word in seg["words"] or []:
                mid = (word["start"] + word["end"]) / 2
                outside = not any(a <= mid <= b for a, b in raw["regions"])
                gap = any(a <= mid <= b for a, b in gaps)
                accepted_words += 1
                transcript.append(word["word"].strip())
                if gap or outside:
                    suspect.append(
                        {
                            **word,
                            "outside_vad": outside,
                            "annotation_gap": gap,
                            "avg_logprob": seg["avg_logprob"],
                            "no_speech_prob": seg["no_speech_prob"],
                        }
                    )
        has_vocals = detector.has_vocals(raw["regions"])
        row = {
            "id": entry["id"],
            "words": accepted_words,
            "suspect": suspect,
            "lang": entry["lang"],
            "condition": entry["condition"],
            "stage_words": accepted_words if has_vocals else 0,
        }
        if entry["lang"]:
            ref = entry.get("reference_text") or "\n".join(line["text"] for line in entry["lines"])
            text = ("" if entry["lang"] == "zh" else " ").join(transcript) if has_vocals else ""
            rate, units = error_rate(entry["lang"], ref, text)
            row.update(error_rate=rate, ref_units=units)
        rows.append(row)
        print(
            f"[{i}/{len(entries)}] {entry['id']}: words={accepted_words} "
            f"gap_candidates={sum(w['annotation_gap'] for w in suspect)} "
            f"outside_vad={sum(w['outside_vad'] for w in suspect)}",
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    totals = summary(rows)
    report = {
        "split": args.split,
        "base_seed": args.seed,
        "activity_params": dataclasses.asdict(detector.params),
        "summary": totals,
        "songs": rows,
    }
    if args.repeat_cache:
        if args.seed is None:
            parser.error("repeat verification requires --seed")
        checks = []
        for row in rows:
            current_path = args.cache / f"{row['id']}.json"
            prior_path = args.repeat_cache / current_path.name
            current = json.loads(current_path.read_text())
            prior = json.loads(prior_path.read_text())
            for key in ("id", "audio_sha256", "activity_params", "recognize_params", "random_seed"):
                if prior[key] != current[key]:
                    raise ValueError(f"repeat inputs differ: {row['id']} / {key}")
            checks.append(
                {
                    "id": row["id"],
                    "segments": len(current["segments"]),
                    "full_metadata_equal": current == prior,
                    "current_sha256": hashlib.sha256(current_path.read_bytes()).hexdigest(),
                    "prior_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
                }
            )
        report["repeat_checks"] = checks
        args.out.with_suffix(".md").write_text(
            "# 第 5 步：逐曲随机种子的复现抽查\n\n"
            f"对比缓存 `{args.repeat_cache.name}` 与 `{args.cache.name}`，"
            f"基准种子 {args.seed}，能量补检参数 {args.energy_relative_db} dB。"
            "生成缓存时每首创建新的 Whisper 工作线程。\n\n"
            + "\n".join(
                f"- {c['id']}：{c['segments']} 个片段，完整解码元数据"
                f"{'一致' if c['full_metadata_equal'] else '存在差异'}（包括文本、时间戳及概率）。"
                for c in checks
            )
            + "\n\n这是同一环境的开发曲目抽查，不保证所有曲目或跨硬件完全一致。"
            "文件校验值见 JSON；未保存人声轨。\n"
        )
    if args.compare:
        previous = json.loads(args.compare.read_text())
        assert {s["id"] for s in previous["songs"]} == {s["id"] for s in rows}
        assert previous["split"] == args.split
        before = summary(previous["songs"])
        report["comparison"] = {
            "source_sha256": hashlib.sha256(args.compare.read_bytes()).hexdigest(),
            "before": before,
            "after": totals,
        }
        lines = [
            "# 第 5 步：独立开发集的识别策略比较",
            "",
            "同一开发集，当前语音区间与此前选定的能量补检参数比较；"
            "不使用验收曲目调整阈值。识别前沿用实际流水线的 5 秒人声门槛。",
            "",
            "| 集合 | 条目 | 语音区间字/词错率 | 能量补检字/词错率 | 未标注区间词（前 / 后） |",
            "|---|---|---|---|---|",
        ]

        def fmt(value):
            return "—" if value is None else f"{value * 100:.1f}%"

        for key, after in totals.items():
            old = before[key]
            lines.append(
                f"| {key} | {after['songs']} | {fmt(old['error_rate'])} | "
                f"{fmt(after['error_rate'])} | "
                f"{old['gap_candidates']} / {after['gap_candidates']} |"
            )
        inst = "instrumental/instrumental"
        lines += [
            "",
            f"纯音乐无词：{before[inst]['no_words']}/{before[inst]['songs']} → "
            f"{totals[inst]['no_words']}/{totals[inst]['songs']}。",
            "原始解码结果只缓存为仓库外的文本元数据；未保存人声轨。"
            "本次沿用原配置的温度回退抽样，可能有重复运行差异；固定随机种子的检查单独记录。",
        ]
        args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
