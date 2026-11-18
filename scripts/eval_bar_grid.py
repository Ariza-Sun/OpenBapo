"""Bar-grid regularisation: downbeat accuracy versus regular bars (docs/plans/step-1.md, 2.3).

Usage:
    python scripts/eval_bar_grid.py --guitarset DIR --gtzan DIR --gtzan-annotations DIR \
        [--cache FILE] [--device auto] [--out docs/benchmarks/step-6-bar-grid]

Runs Beat This! once per track (cached in --cache), then builds the bar grid
with the old rule (a bar at every flagged downbeat) and with several
regulariser settings. GuitarSet (360 excerpts) is the development set the
setting was chosen on; GTZAN (998 tracks, the step-1 test set) is the check.

Metrics: downbeat F-measure (mir_eval, 70 ms, first 5 s trimmed, as in
scripts/eval_beats.py) and the share of inner bars (not the first or last)
whose beat count differs from the time signature.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_beats import load_tracks, score

# (label, P_BEAT_SLIP, P_PHASE_JUMP); None = the old rule, no regularisation.
SETTINGS: list[tuple[str, float | None, float | None]] = [
    ("旧规则（每个强拍开新小节）", None, None),
    ("A 少拍多拍 0.03 / 错位 1e-3", 0.03, 1e-3),
    ("B 少拍多拍 0.003 / 错位 1e-4", 0.003, 1e-4),
    ("C 只允许错位 1e-4（采用）", 0.0, 1e-4),
]


def predictions(args: argparse.Namespace) -> dict[str, tuple[Any, Any, Any, Any]]:
    """track -> (reference beats, reference downbeats, estimated beats, estimated downbeats)."""
    if args.cache and args.cache.exists():
        cached: dict[str, tuple[Any, Any, Any, Any]] = pickle.loads(args.cache.read_bytes())
        return cached
    from openbapo.analysis.audio import decode
    from openbapo.analysis.beats import BeatThisTracker
    from openbapo.errors import InputError

    tracker = BeatThisTracker(device=args.device)
    out: dict[str, tuple[Any, Any, Any, Any]] = {}
    for jams in sorted(args.guitarset.glob("*_comp.jams")) + sorted(
        args.guitarset.glob("*_solo.jams")
    ):
        data = json.loads(jams.read_text(encoding="utf-8"))
        rows = next(a for a in data["annotations"] if a["namespace"] == "beat_position")["data"]
        ref_beats = np.array([r["time"] for r in rows])
        ref_down = np.array([r["time"] for r in rows if r["value"]["position"] == 1])
        result = tracker.track(decode(jams.with_name(jams.stem + "_mic.wav")))
        out["guitarset/" + jams.stem] = (ref_beats, ref_down, result.beats, result.downbeats)
    tracks, _ = load_tracks(args.gtzan, args.gtzan_annotations)
    for track in tracks:
        try:
            result = tracker.track(decode(track.audio))
        except InputError:
            continue
        out["gtzan/" + track.name] = (track.beats, track.downbeats, result.beats, result.downbeats)
    if args.cache:
        args.cache.write_bytes(pickle.dumps(out))
    return out


def evaluate(
    items: dict[str, tuple[Any, Any, Any, Any]], slip: float | None, jump: float | None
) -> dict[str, float]:
    import openbapo.analysis.grid as grid

    original = grid.regular_downbeats
    if slip is None:
        grid.regular_downbeats = lambda count, flagged, n, **kw: list(flagged)  # type: ignore[assignment]
    else:
        grid.P_BEAT_SLIP, grid.P_PHASE_JUMP = slip, jump  # type: ignore[assignment]
    try:
        f1, irregular, inner = [], 0, 0
        for _, ref_down, beats, downbeats in items.values():
            g = grid.build_grid(beats.tolist(), downbeats.tolist())
            estimate = (
                np.array([b.time for b in g.beats if b.is_downbeat])
                if g.time_signature
                else downbeats
            )
            f1.append(score(ref_down, estimate))
            counts = collections.Counter(b.bar_index for b in g.beats)
            lengths = [counts[i] for i in sorted(counts)][1:-1]
            n = g.time_signature.numerator if g.time_signature else grid.DEFAULT_BEATS_PER_BAR
            irregular += sum(length != n for length in lengths)
            inner += len(lengths)
        return {"downbeat_f1": float(np.nanmean(f1)), "irregular_bars": irregular / max(inner, 1)}
    finally:
        grid.regular_downbeats = original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--guitarset", type=Path, required=True)
    parser.add_argument("--gtzan", type=Path, required=True)
    parser.add_argument("--gtzan-annotations", type=Path, required=True)
    parser.add_argument("--cache", type=Path, help="pickle of tracker outputs, reused if present")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="docs/benchmarks/step-6-bar-grid")
    args = parser.parse_args()

    preds = predictions(args)
    sets = {
        name: {k: v for k, v in preds.items() if k.startswith(name + "/")}
        for name in ("guitarset", "gtzan")
    }
    results = []
    for label, slip, jump in SETTINGS:
        row = {"setting": label, "p_beat_slip": slip, "p_phase_jump": jump}
        for name, items in sets.items():
            row[name] = evaluate(items, slip, jump)
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    lines = [
        "# 小节规整：强拍准确率与小节整齐度",
        "",
        f"由 `scripts/eval_bar_grid.py` 生成于 {dt.date.today().isoformat()}。"
        f"GuitarSet {len(sets['guitarset'])} 段为开发集（选参数），"
        f"GTZAN {len(sets['gtzan'])} 首为检查集（第 1 步验收集，只看不调）。",
        "强拍 F1 同 `scripts/eval_beats.py`；不规整小节 = 去掉首尾后拍数不等于拍号的小节比例。",
        "",
        "| 设置 | GuitarSet 强拍 F1 | GuitarSet 不规整小节 | GTZAN 强拍 F1 | GTZAN 不规整小节 |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        g, z = r["guitarset"], r["gtzan"]
        lines.append(
            f"| {r['setting']} | {g['downbeat_f1']:.3f} | {g['irregular_bars']:.1%} | "
            f"{z['downbeat_f1']:.3f} | {z['irregular_bars']:.1%} |"
        )
    lines += [
        "",
        "采用 C：谱面小节整齐优先。代价是节拍模型漏拍 / 多拍时，之后几个小节可能整体错开一拍，"
        "直到持续的证据触发一次错位；A、B 保留更多强拍细节但仍有零散的不规整小节。",
    ]
    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=1), "utf-8")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
