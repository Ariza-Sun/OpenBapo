"""Score the key estimator on inputs dumped by eval_chords.py --dump-key-inputs.

Usage:
    python scripts/eval_key_inputs.py docs/benchmarks/step-2-key-inputs.json.gz

The dump holds, per track, the committee's chords and the chroma key
scores (what `estimate_key` sees) plus the reference key, so changes to
the key rules can be measured in seconds on a CPU. Numbers here must
match the key columns of step-2-chords.md for the same code; the final
acceptance number still comes from eval_chords.py.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:  # type: ignore[type-arg]
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))  # type: ignore[no-any-return]


def to_chords(rows: list[list[object]]) -> list:  # type: ignore[type-arg]
    from openbapo.model import Chord, ChordQuality

    return [
        Chord(
            start=float(s), end=float(e), root=int(r), quality=ChordQuality(str(q)), confidence=1.0
        )  # type: ignore[arg-type]
        for s, e, r, q in rows
    ]


def estimate(record: dict) -> str | None:  # type: ignore[type-arg]
    from openbapo.analysis.chords.key import estimate_key, scores_from_chords

    chords = to_chords(record["chords"])
    key = estimate_key(np.array(record["chroma_scores"]), scores_from_chords(chords), chords)
    if key is None:
        return None
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[key.tonic]} {key.mode.value}"


def relation(ref: str, est: str | None) -> str:
    import mir_eval

    if est is None:
        return "none"
    weighted = round(float(mir_eval.key.weighted_score(ref, est)), 1)
    return {1.0: "exact", 0.5: "fifth", 0.3: "relative", 0.2: "parallel"}.get(weighted, "other")


SUBSETS = {
    "guitarset_tune_comp": lambda r: (
        r["dataset"] == "guitarset" and r["split"] == "tune" and r["name"].endswith("_comp")
    ),
    "guitarset_report_comp": lambda r: (
        r["dataset"] == "guitarset" and r["split"] == "report" and r["name"].endswith("_comp")
    ),
    "aam": lambda r: r["dataset"] == "aam",
    "aam_single_key": lambda r: r["dataset"] == "aam" and r["key_share"] >= 0.999,
}


def evaluate(records: list[dict]) -> dict[str, dict]:  # type: ignore[type-arg]
    """Per subset: exact accuracy, accuracy against any key with >= 1/3 of the time, relations."""
    scored = []
    for r in records:
        if r["key_ref"] is None:
            continue
        est = estimate(r)
        major = {k for k, t in r["key_times"].items() if t >= sum(r["key_times"].values()) / 3}
        scored.append(
            {
                **r,
                "key_est": est,
                "relation": relation(r["key_ref"], est),
                "any_major": float(est in (major or {r["key_ref"]})),
            }
        )
    out = {}
    for name, pick in SUBSETS.items():
        rows = [r for r in scored if pick(r)]
        if not rows:
            continue
        out[name] = {
            "tracks": len(rows),
            "key_exact": float(np.mean([r["relation"] == "exact" for r in rows])),
            "key_any_major": float(np.mean([r["any_major"] for r in rows])),
            "relations": dict(Counter(r["relation"] for r in rows)),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dump", type=Path)
    args = parser.parse_args(argv)
    data = load(args.dump)
    print(f"fusion params: {json.dumps(data['params'])}")
    for name, m in evaluate(data["tracks"]).items():
        print(
            f"{name:24s} n={m['tracks']:4d}  exact={m['key_exact']:.3f}  "
            f"any>=1/3={m['key_any_major']:.3f}  {m['relations']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
