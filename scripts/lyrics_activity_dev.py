"""Evaluate singing activity on independent real mixes and the original dev set.

Run with --jamendo DIR --manifest TESTSET/manifest.json --cache DIR --out PREFIX.
--download obtains only the 59 non-English songs from the pinned public source.
Only dev entries are evaluated. Cached RMS curves and detected intervals contain
no audio; separated vocals stay in memory. Reports contain anonymous song IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from eval_lyrics import load_vocals
from tune_lyrics_vad import frames

from openbapo.analysis.lyrics.activity import (
    ActivityParams,
    SileroActivity,
    extend_activity,
    frame_rms,
)
from openbapo.analysis.separate import DemucsSeparator

REPO = "jamendolyrics/jamendolyrics"
REVISION = "de188c963fd4539bc769b3feb83582e5a9595e36"
LANGUAGES = ("de", "es", "fr")
THRESHOLDS = (0.1, 0.3, 0.5)


class RecordedSpeech:
    """Replay one Silero inference pass when comparing segmentation thresholds."""

    def __init__(self, model):
        self.model = model
        self.values = []
        self.position = 0

    def reset_states(self):
        self.position = 0
        self.model.reset_states()

    def __call__(self, chunk, sample_rate):
        if self.position == len(self.values):
            self.values.append(self.model(chunk, sample_rate).detach().clone())
        value = self.values[self.position]
        self.position += 1
        return value


def speech_settings(vocals, model):
    recorded = RecordedSpeech(model)
    found = {}
    for threshold in THRESHOLDS:
        detector = SileroActivity(ActivityParams(threshold=threshold, energy_relative_db=None))
        detector._model = recorded
        found[str(threshold)] = detector.regions(vocals)
    return found


def real_entries(root: Path) -> list[dict]:
    entries = []
    english = {
        json.loads(line)["name"]
        for line in (root / "subsets/en/metadata.jsonl").read_text().splitlines()
    }
    for lang in LANGUAGES:
        songs = [
            json.loads(line)
            for line in (root / f"subsets/{lang}/metadata.jsonl").read_text().splitlines()
        ]
        for i, song in enumerate(sorted(songs, key=lambda s: s["name"]), 1):
            assert song["name"] not in english
            path = root / f"subsets/{lang}" / song["file_name"]
            # Use word intervals, including finite ends in the pinned metadata.
            spans = sorted((float(w["start"]), float(w["end"])) for w in song["words"])
            entries.append(
                {
                    "id": f"real-dev-{lang}-{i:02d}",
                    "split": "dev",
                    "condition": "real",
                    "audio": str(path.resolve()),
                    "duration": sf.info(path).duration,
                    "vocal_regions": spans,
                }
            )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jamendo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        from huggingface_hub import snapshot_download

        snapshot_download(
            REPO,
            repo_type="dataset",
            revision=REVISION,
            local_dir=args.jamendo,
            allow_patterns=[
                "subsets/en/metadata.jsonl",
                *[f"subsets/{lang}/**" for lang in LANGUAGES],
            ],
            max_workers=4,
        )
    manifest = json.loads(args.manifest.read_text())
    entries = real_entries(args.jamendo) + [e for e in manifest["entries"] if e["split"] == "dev"]
    args.cache.mkdir(parents=True, exist_ok=True)
    separator = DemucsSeparator(device=args.device)
    speech_model = SileroActivity()._load()
    features = []
    for i, entry in enumerate(entries, 1):
        path = args.manifest.parent / entry["audio"]
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        cached = args.cache / f"{entry['id']}.json"
        feature = json.loads(cached.read_text()) if cached.exists() else None
        if feature is None or feature.get("audio_sha256") != digest:
            vocals, _ = load_vocals(entry, args.manifest.parent, separator)
            feature = {
                "id": entry["id"],
                "audio_sha256": digest,
                "rms": frame_rms(vocals).tolist(),
                "speech": speech_settings(vocals, speech_model),
            }
            temporary = cached.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(feature))
            temporary.replace(cached)
        features.append(feature)
        print(f"features {i}/{len(entries)} {entry['id']}", flush=True)
    settings = [{"threshold": t} for t in THRESHOLDS]
    settings += [
        {"threshold": 0.5, "relative_db": relative, "floor_db": floor, "min_seed_s": seed}
        for relative in (-15.0, -25.0, -35.0)
        for floor in (-50.0, -60.0)
        for seed in (1.0, 5.0)
    ]
    results = []
    for params in settings:
        songs = []
        for entry, feature in zip(entries, features, strict=True):
            predicted = feature["speech"][str(params["threshold"])]
            if "relative_db" in params:
                predicted = extend_activity(
                    np.array(feature["rms"], dtype=np.float32),
                    predicted,
                    entry["duration"],
                    **{k: v for k, v in params.items() if k != "threshold"},
                )
            truth, hyp = (
                frames(entry["vocal_regions"], entry["duration"]),
                frames(predicted, entry["duration"]),
            )
            true_positive = int((truth & hyp).sum())
            seconds = float(sum(b - a for a, b in predicted))
            songs.append(
                {
                    "id": entry["id"],
                    "condition": entry["condition"],
                    "recall": true_positive / max(int(truth.sum()), 1),
                    "precision": true_positive / max(int(hyp.sum()), 1),
                    "found_seconds": seconds,
                    "has_vocals": seconds >= 5,
                }
            )
        summary = {}
        for condition in ("real", "dry", "mix", "instrumental"):
            group = [s for s in songs if s["condition"] == condition]
            summary[condition] = {
                "songs": len(group),
                "recall": float(np.mean([s["recall"] for s in group])),
                "precision": float(np.mean([s["precision"] for s in group])),
                "no_vocals": sum(not s["has_vocals"] for s in group),
            }
        results.append({"params": params, "summary": summary, "songs": songs})
    report = {"source": REPO, "revision": REVISION, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    lines = [
        "# 第 5 步：独立真实歌曲开发集的人声检测",
        "",
        "59 首德语 / 西语 / 法语真实歌曲与英文验收集不重叠；"
        "另含原开发集 40 个干声 / 合成伴奏版本、10 首纯音乐。",
        "",
        f"来源：[JamendoLyrics](https://huggingface.co/datasets/{REPO})，固定版本 `{REVISION}`。",
        "数值为逐首平均的 50 ms 帧召回率 / 精确率；完全无人声曲目按检出时长 < 5 秒判断。",
        "",
        "| 参数 | 真实歌曲 | 干声 | 合成伴奏 | 纯音乐无歌词 |",
        "|---|---|---|---|---|",
    ]
    for row in results:
        cells = [
            f"{row['summary'][c]['recall']:.3f} / {row['summary'][c]['precision']:.3f}"
            for c in ("real", "dry", "mix")
        ]
        lines.append(
            f"| `{json.dumps(row['params'])}` | "
            + " | ".join(cells)
            + f" | {row['summary']['instrumental']['no_vocals']}/10 |"
        )
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
