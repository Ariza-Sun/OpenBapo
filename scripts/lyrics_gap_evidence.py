"""Cross-check gap candidates with official revised lines and unconstrained CTC.

No candidate transcript is supplied to CTC. Decode original mixtures and vocal
stems independently, keeping only text metadata, never saving audio or stems.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import soxr
from eval_lyrics import load_vocals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--jam-alt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from openbapo.analysis.audio import decode
    from openbapo.analysis.lyrics.align import wav2vec2_english
    from openbapo.analysis.separate import DemucsSeparator

    entries = {e["id"]: e for e in json.loads(args.manifest.read_text())["entries"]}
    revised = {e["name"]: e for e in map(json.loads, args.jam_alt.read_text().splitlines())}
    model = wav2vec2_english(args.device)
    labels = {i: label for label, i in model.labels.items()}
    separator = DemucsSeparator(device=args.device)
    results = []
    for row in json.loads(args.audit.read_text())["songs"]:
        entry = entries[row["id"]]
        assert entry["lang"] == "en"
        path = args.manifest.parent / entry["audio"]
        audio = decode(path)
        mixture = soxr.resample(audio.mono(), audio.sample_rate, 16000).astype(np.float32)
        vocals, _ = load_vocals(entry, args.manifest.parent, separator)
        groups = []
        for word in row["suspect"]:
            if not word["annotation_gap"]:
                continue
            if groups and word["start"] - groups[-1][-1]["end"] < 2:
                groups[-1].append(word)
            else:
                groups.append([word])
        evidence = []
        for group in groups:
            start = max(0, group[0]["start"] - 1.5)
            end = min(len(vocals) / 16000, group[-1]["end"] + 1.5)
            decoded = {}
            for kind, samples in [("mixture", mixture), ("vocals", vocals)]:
                probs = model.forward(samples[int(start * 16000) : int(end * 16000)])
                tokens = [i for i, _ in itertools.groupby(probs.argmax(axis=1))]
                decoded[kind] = (
                    "".join(labels[i] for i in tokens if i != model.blank).replace("|", " ").strip()
                )
            overlap = [
                line
                for line in revised[path.stem]["lines"]
                if line["start"] < group[-1]["end"] and line["end"] > group[0]["start"]
            ]
            evidence.append(
                {
                    "window": [start, end],
                    "suspect_words": group,
                    "ctc_greedy": decoded,
                    "revised_lines": overlap,
                }
            )
        result = {
            "id": row["id"],
            "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "evidence": evidence,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        del audio, mixture, vocals
    args.out.write_text(
        json.dumps(
            {
                "method": "unprompted CTC greedy, original mixture and vocals",
                "reference_sha256": hashlib.sha256(args.jam_alt.read_bytes()).hexdigest(),
                "songs": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
