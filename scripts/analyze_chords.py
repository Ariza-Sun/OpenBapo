"""Error analysis for the step-2 chord and key results (reads the eval cache).

Usage:
    python scripts/analyze_chords.py --guitarset /data/guitarset --cache /data/eval-cache \
        [--subset report_comp] [--params docs/benchmarks/step-2-chords.json] [--worst 10]

Breaks the majmin error of the committee and of each expert into kinds
(frame by frame): the reference has no chord, the estimate says no chord,
same root but major/minor swapped, the relative chord (root a third away
with the opposite quality), the chord a fourth or fifth away, any other
root, and how much of each lies within a beat of a reference boundary.
Also lists how often at least one expert is right (the oracle the fusion
could reach), the worst tracks with reference and estimate side by side,
and every wrong key with the chords that led to it. Prints markdown; no
file is written. Run where the datasets and the cache live.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_chords import (
    Track,
    TrackOutputs,
    aam_tracks,
    guitarset_tracks,
    key_name,
    run_fusion,
    sheet_tracks,
)

STEP = 0.05  # seconds per analysis frame
KINDS = ("correct", "ref_N", "est_N", "third", "relative", "fourth_fifth", "other_root")


def encode(label: str) -> tuple[int, str] | None:
    """Harte label -> (root, "maj" | "min" | "none") for majmin comparison; None for N / X."""
    import mir_eval

    if label in ("N", "X"):
        return None
    root, semitones, _ = mir_eval.chord.encode(label, reduce_extended_chords=True)
    if root < 0:
        return None
    third = "maj" if semitones[4] else ("min" if semitones[3] else "none")
    return int(root), third


def frame_kind(ref: str, est: str) -> str:
    r, e = encode(ref), encode(est)
    if r is None and e is None:
        return "correct"
    if r is None:
        return "ref_N"
    if e is None:
        return "est_N"
    if r[0] == e[0]:
        return "correct" if r[1] == e[1] or "none" in (r[1], e[1]) else "third"
    delta = (e[0] - r[0]) % 12
    if {r[1], e[1]} == {"maj", "min"} and (
        (r[1] == "maj" and delta == 9) or (r[1] == "min" and delta == 3)
    ):
        return "relative"
    if delta in (5, 7):
        return "fourth_fifth"
    return "other_root"


def label_at(intervals: np.ndarray, labels: Sequence[str], times: np.ndarray) -> list[str]:
    """Label covering each time; N outside every interval."""
    out = ["N"] * len(times)
    if len(intervals) == 0:
        return out
    starts = intervals[:, 0]
    idx = np.searchsorted(starts, times, side="right") - 1
    for i, (t, j) in enumerate(zip(times, idx, strict=True)):
        if j >= 0 and intervals[j, 0] <= t < intervals[j, 1]:
            out[i] = labels[j]
    return out


def near_boundary(times: np.ndarray, intervals: np.ndarray, tolerance: float) -> np.ndarray:
    """True for frames within `tolerance` of a reference chord change."""
    if len(intervals) == 0:
        return np.zeros(len(times), dtype=bool)
    edges = np.unique(np.concatenate([intervals[:, 0], intervals[-1:, 1]]))
    distance = np.min(np.abs(times[:, None] - edges[None, :]), axis=1)
    return np.asarray(distance <= tolerance)


def frame_kinds(
    ref_intervals: np.ndarray,
    ref_labels: Sequence[str],
    est_intervals: np.ndarray,
    est_labels: Sequence[str],
    beat_period: float,
) -> tuple[Counter[str], Counter[str]]:
    """Frame counts per kind, and the subset of wrong frames within a beat of a boundary."""
    if len(ref_intervals) == 0:
        return Counter(), Counter()
    end = float(ref_intervals[:, 1].max())
    times = np.arange(0.0, end, STEP)
    refs = label_at(ref_intervals, ref_labels, times)
    ests = label_at(est_intervals, est_labels, times)
    boundary = near_boundary(times, ref_intervals, beat_period)
    kinds: Counter[str] = Counter()
    at_boundary: Counter[str] = Counter()
    for r, e, b in zip(refs, ests, boundary, strict=True):
        kind = frame_kind(r, e)
        kinds[kind] += 1
        if kind != "correct" and b:
            at_boundary[kind] += 1
    return kinds, at_boundary


def compress(intervals: np.ndarray, labels: Sequence[str]) -> str:
    parts = []
    for (s, e), label in zip(intervals.tolist(), labels, strict=True):
        if parts and parts[-1][0] == label:
            parts[-1][1] += e - s
        else:
            parts.append([label, e - s])
    return " ".join(f"{label}({d:.1f})" for label, d in parts)


def load_params(path: Path | None):  # type: ignore[no-untyped-def]
    from opencapo.analysis.chords.fusion import FusionParams

    if path is None:
        return FusionParams()
    data = json.loads(path.read_text(encoding="utf-8"))
    tuned = next(s for s in data["systems"] if s["system"] == "committee (tuned)")
    return FusionParams(**tuned["params"])


def pick_subset(tracks: list[Track], subset: str) -> list[Track]:
    if subset == "aam_single":
        return [t for t in tracks if t.dataset == "aam" and t.key_share >= 0.999]
    if subset in ("aam", "sheets"):
        return [t for t in tracks if t.dataset == subset]
    split, kind = subset.split("_")
    return [
        t
        for t in tracks
        if t.dataset == "guitarset" and t.split == split and t.name.endswith(f"_{kind}")
    ]


def percent(counter: Counter[str], total: int) -> dict[str, str]:
    return {k: f"{100 * counter.get(k, 0) / total:.1f}%" for k in KINDS if total}


def analyze(  # type: ignore[no-untyped-def]
    tracks: list[Track], outputs: dict[str, TrackOutputs], expert_names, model_names, params, worst
) -> list[str]:
    systems = {name: [name] for name in expert_names}
    systems["committee"] = list(expert_names)
    totals: dict[str, Counter[str]] = {s: Counter() for s in systems}
    boundaries: dict[str, Counter[str]] = {s: Counter() for s in systems}
    by_group: dict[str, Counter[str]] = defaultdict(Counter)
    group_frames: Counter[str] = Counter()
    per_track: list[tuple[float, Track, str, str]] = []
    oracle_hits = 0
    oracle_frames = 0
    key_rows: list[str] = []
    for t in tracks:
        out = outputs[t.name]
        beats = [b.time for b in out.beats]
        period = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
        estimates = {}
        for s, names in systems.items():
            intervals, labels, key = run_fusion(out, names, model_names, params)
            estimates[s] = (intervals, labels, key)
            kinds, at_boundary = frame_kinds(
                t.chord_intervals, t.chord_labels, intervals, labels, period
            )
            totals[s] += kinds
            boundaries[s] += at_boundary
            if s == "committee":
                by_group[t.group] += kinds
                group_frames[t.group] += sum(kinds.values())
                n = sum(kinds.values())
                per_track.append(
                    (
                        kinds["correct"] / n if n else 1.0,
                        t,
                        compress(t.chord_intervals, t.chord_labels),
                        compress(intervals, labels),
                    )
                )
                if t.key is not None and key != t.key:
                    est = "none" if key is None else key_name(key)
                    chords = compress(intervals, labels)
                    key_rows.append(
                        f"- `{t.name}` 参考 {key_name(t.key)}，估计 {est}；和弦：{chords}"
                    )
        # oracle: a frame counts when any model expert gets it right
        end = float(t.chord_intervals[:, 1].max()) if len(t.chord_intervals) else 0.0
        times = np.arange(0.0, end, STEP)
        refs = label_at(t.chord_intervals, t.chord_labels, times)
        right = np.zeros(len(times), dtype=bool)
        for s in model_names:
            intervals, labels, _ = estimates[s]
            ests = label_at(intervals, labels, times)
            right |= np.array(
                [frame_kind(r, e) == "correct" for r, e in zip(refs, ests, strict=True)]
            )
        oracle_hits += int(right.sum())
        oracle_frames += len(times)

    lines = [f"## 错误分解（{len(tracks)} 首，按帧，{STEP} s 一帧）", ""]
    lines.append("| 系统 | " + " | ".join(KINDS) + " | 错误中在参考边界一拍内 |")
    lines.append("|---|" + "---|" * (len(KINDS) + 1))
    for s in systems:
        total = sum(totals[s].values())
        wrong = total - totals[s]["correct"]
        share = f"{100 * sum(boundaries[s].values()) / wrong:.0f}%" if wrong else "—"
        cells = [s, *percent(totals[s], total).values(), share]
        lines.append("| " + " | ".join(cells) + " |")
    if oracle_frames:
        lines.append(
            f"\n任一模型专家正确（融合的上限）：{100 * oracle_hits / oracle_frames:.1f}%\n"
        )
    lines += [
        "## 按风格（committee）",
        "",
        "| 风格 | " + " | ".join(KINDS) + " |",
        "|---|" + "---|" * len(KINDS),
    ]
    for g in sorted(by_group):
        lines.append(
            f"| {g} | " + " | ".join(percent(by_group[g], group_frames[g]).values()) + " |"
        )
    lines += ["", f"## 最差的 {worst} 首（committee，majmin 帧正确率）", ""]
    for score, t, ref, est in sorted(per_track, key=lambda x: x[0])[:worst]:
        lines += [f"- `{t.name}` {score:.2f}", f"  - 参考：{ref}", f"  - 估计：{est}"]
    lines += ["", f"## 调性错误（committee，{len(key_rows)} 首）", "", *key_rows]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--guitarset", type=Path)
    parser.add_argument("--aam", type=Path)
    parser.add_argument("--sheets", type=Path)
    parser.add_argument("--aam-count", type=int, default=500)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--experts", default="btc,lvchordia,chroma")
    parser.add_argument(
        "--subset",
        default="report_comp",
        help="report_comp, tune_comp, report_solo, aam, aam_single, sheets",
    )
    parser.add_argument("--params", type=Path, help="step-2-chords.json whose tuned params to use")
    parser.add_argument("--worst", type=int, default=10)
    args = parser.parse_args(argv)

    tracks: list[Track] = []
    if args.guitarset:
        tracks += guitarset_tracks(args.guitarset)
    if args.aam:
        tracks += aam_tracks(args.aam, args.aam_count)
    if args.sheets:
        tracks += sheet_tracks(args.sheets)
    tracks = pick_subset(tracks, args.subset)
    if not tracks:
        print("no tracks in that subset", file=sys.stderr)
        return 1
    outputs = {}
    for t in tracks:
        path = args.cache / f"{t.dataset}_{t.name}.pkl"
        if not path.exists():
            print(f"missing cache for {t.name}: run eval_chords.py --cache first", file=sys.stderr)
            return 1
        outputs[t.name] = pickle.loads(path.read_bytes())
    expert_names = [n for n in args.experts.split(",") if n]
    model_names = [n for n in expert_names if n != "chroma"]
    params = load_params(args.params)
    print(f"# 和弦 / 调性错误分析：{args.subset}\n\n融合参数：`{params}`\n")
    print("\n".join(analyze(tracks, outputs, expert_names, model_names, params, args.worst)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
