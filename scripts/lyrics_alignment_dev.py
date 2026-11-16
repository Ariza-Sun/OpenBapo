"""Compare finite CTC activity penalties on the original dev songs only.

Acoustic inference covers the whole song. All candidates share the same
emissions and baseline Silero regions, isolating the effect of the penalty.
No acceptance entry is read. Vocal stems remain in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from eval_lyrics import line_errors, load_vocals, manifest_lines, summarise
from lyrics_activity_dev import speech_settings

from openbapo.analysis.lyrics.activity import SileroActivity, frame_rms
from openbapo.analysis.lyrics.align import (
    TimedToken,
    activity_prior,
    ctc_targets,
    emissions,
    forced_align,
    song_language,
    wav2vec2_aligner,
)
from openbapo.analysis.separate import DemucsSeparator

PENALTIES = (0.0, 2.0, 5.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    entries = [
        e
        for e in json.loads(args.manifest.read_text())["entries"]
        if e["split"] == "dev" and e["condition"] != "instrumental"
    ]
    separator = DemucsSeparator(device=args.device)
    aligner = wav2vec2_aligner(args.device)
    speech_model = SileroActivity()._load()
    names = [f"penalty-{p:g}" for p in PENALTIES]
    rows = []
    args.cache.mkdir(parents=True, exist_ok=True)
    for i, entry in enumerate(entries, 1):
        vocals, sep_s = load_vocals(entry, args.manifest.parent, separator)
        path = args.manifest.parent / entry["audio"]
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        cached = args.cache / f"{entry['id']}.json"
        feature = json.loads(cached.read_text()) if cached.exists() else None
        if feature is None or feature.get("audio_sha256") != digest:
            feature = {
                "id": entry["id"],
                "audio_sha256": digest,
                "rms": frame_rms(vocals).tolist(),
                "speech": speech_settings(vocals, speech_model),
            }
            temporary = cached.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(feature))
            temporary.replace(cached)
        lines = manifest_lines(entry)
        model = aligner.model(song_language(lines))
        start = time.perf_counter()
        raw = emissions(model, vocals, None)
        inference_s = time.perf_counter() - start
        pairs = [(line.index, token) for line in lines for token in line.tokens]
        targets = ctc_targets(model, [token for _, token in pairs])
        row = {
            "id": entry["id"],
            "set": entry["set"],
            "lang": entry["lang"],
            "condition": entry["condition"],
            "align": {},
            "separate_seconds": sep_s,
        }
        for name, penalty in zip(names, PENALTIES, strict=True):
            start = time.perf_counter()
            probs = activity_prior(
                raw, model.blank, model.frame_s, feature["speech"]["0.5"], penalty
            )
            spans = forced_align(probs, targets, model.blank, model.frame_s)
            timed = [
                TimedToken(line, token, *(span if span else (None, None, 0.0)))
                for (line, token), span in zip(pairs, spans, strict=True)
            ]
            row["align"][name] = {
                "errors": [
                    round(e, 3) if e is not None else None for e in line_errors(entry, timed)
                ],
                "seconds": inference_s + time.perf_counter() - start,
            }
        rows.append(row)
        print(f"alignment {i}/{len(entries)} {entry['id']}", flush=True)
    summary = summarise(rows, names)
    report = {"penalties": PENALTIES, "summary": summary, "songs": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    lines = [
        "# 第 5 步：CTC 软约束开发集比较",
        "",
        "只使用原 GTSinger 开发集。所有候选均推理完整人声轨，保留基线 Silero 区间，"
        "仅改变区间外非空白标签的有限惩罚；不包含验收集。",
        "",
        "| 集合 | 行数 | 惩罚 0 | 惩罚 2 | 惩罚 5 |",
        "|---|---|---|---|---|",
    ]
    for group, metrics in summary.items():
        values = [metrics[f"align/{n}"] for n in names]
        lines.append(
            f"| {group} | {values[0]['lines']} | "
            + " | ".join(f"{v['within']:.3f}" for v in values)
            + " |"
        )
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
