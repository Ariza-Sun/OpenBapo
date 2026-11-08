"""Beat / downbeat F-measure of the beat tracker on GTZAN (step-1 acceptance).

Usage:
    python scripts/eval_beats.py --audio GTZAN_AUDIO_DIR --annotations BEATS_DIR \
        [--trackers beat_this] [--inputs mix,drums_bass] [--device auto] [--limit N] \
        [--out docs/benchmarks/step-1-beats]

Needs `pip install -e ".[analysis]" -r scripts/requirements-eval.txt` and the
dataset laid out as described in docs/benchmarks/README.md. Writes OUT.json
(all per-track numbers and environment details) and OUT.md (the summary table).

Metric: mir_eval.beat.f_measure with the default 70 ms tolerance, after
dropping the first 5 seconds of both reference and estimate (mir_eval's
trim_beats, the same convention as the Beat This! paper). Beat F-measure is
the acceptance number; downbeat F-measure is reported without a threshold.

Self-check: the annotations are also scored against themselves ("annotations"
row); anything but 1.0 there means the loader or the metric call is wrong.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import platform
import re
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TRIM_SECONDS = 5.0
ACCEPTANCE_BEAT_F1 = 0.85
GENRES = (
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
)  # fmt: skip
ANNOTATION_NAME = re.compile(r"^(?:gtzan_)?(?P<genre>[a-z]+)[_.](?P<number>\d{5})\.beats$")


@dataclass(frozen=True)
class Track:
    genre: str
    number: str
    audio: Path
    beats: np.ndarray
    downbeats: np.ndarray

    @property
    def name(self) -> str:
        return f"{self.genre}.{self.number}"


def parse_beats_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a `time<TAB>position` file; position 1 marks a downbeat."""
    beats, downbeats = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        t = float(parts[0])
        beats.append(t)
        if len(parts) > 1 and parts[1].split(".")[0] == "1":
            downbeats.append(t)
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


def find_audio(audio_dir: Path) -> dict[tuple[str, str], Path]:
    """Map (genre, number) -> audio file for both the Kaggle (.wav) and original (.au) layouts."""
    found = {}
    for path in audio_dir.rglob("*"):
        if path.suffix.lower() not in (".wav", ".au", ".mp3", ".flac"):
            continue
        match = re.match(r"^(?P<genre>[a-z]+)\.(?P<number>\d{5})$", path.stem)
        if match and match["genre"] in GENRES:
            found[(match["genre"], match["number"])] = path
    return found


def load_tracks(
    audio_dir: Path, annotations_dir: Path, *, verify_audio: bool = True
) -> tuple[list[Track], list[str]]:
    """Pair annotation files with audio; returns tracks and a list of problems.

    With `verify_audio`, each file is probed with ffprobe and undecodable ones
    (GTZAN's jazz.00054 is a known case) are skipped here, so that every system,
    including the annotations self-check, is scored on the same tracks.
    """
    audio = find_audio(audio_dir)
    tracks, problems = [], []
    for path in sorted(annotations_dir.glob("*.beats")):
        match = ANNOTATION_NAME.match(path.name)
        if not match:
            problems.append(f"unrecognised annotation name: {path.name}")
            continue
        key = (match["genre"], match["number"])
        if key not in audio:
            problems.append(f"no audio for {path.name}")
            continue
        beats, downbeats = parse_beats_file(path)
        if len(beats) == 0:
            problems.append(f"empty annotation: {path.name}")
            continue
        if verify_audio:
            from openbapo.analysis.audio import probe
            from openbapo.errors import InputError

            try:
                probe(audio[key])
            except InputError as e:
                problems.append(f"undecodable audio {audio[key].name}: {e}")
                continue
        tracks.append(Track(key[0], key[1], audio[key], beats, downbeats))
    if not tracks:
        problems.append("no tracks matched; check --audio and --annotations")
    return tracks, problems


def score(reference: np.ndarray, estimate: np.ndarray) -> float:
    import mir_eval

    reference = mir_eval.beat.trim_beats(reference, min_beat_time=TRIM_SECONDS)
    estimate = mir_eval.beat.trim_beats(estimate, min_beat_time=TRIM_SECONDS)
    if len(reference) == 0:
        return float("nan")
    return float(mir_eval.beat.f_measure(reference, estimate))


@dataclass
class Row:
    track: str
    genre: str
    beat_f1: float
    downbeat_f1: float
    seconds: float


@dataclass
class Summary:
    system: str
    rows: list[Row] = field(default_factory=list)

    def mean(self, attr: str, genre: str | None = None) -> float:
        values = [
            getattr(r, attr)
            for r in self.rows
            if (genre is None or r.genre == genre) and not np.isnan(getattr(r, attr))
        ]
        return float(np.mean(values)) if values else float("nan")

    def as_dict(self) -> dict:
        return {
            "system": self.system,
            "tracks": len(self.rows),
            "beat_f1": self.mean("beat_f1"),
            "downbeat_f1": self.mean("downbeat_f1"),
            "mean_seconds_per_track": self.mean("seconds"),
            "by_genre": {
                g: {"beat_f1": self.mean("beat_f1", g), "downbeat_f1": self.mean("downbeat_f1", g)}
                for g in GENRES
                if any(r.genre == g for r in self.rows)
            },
            "per_track": [r.__dict__ for r in self.rows],
        }


# --- systems under test -------------------------------------------------------
# Each entry builds a callable (track audio -> (beats, downbeats)). Adding a
# fallback candidate (madmom, all-in-one) means adding one entry here.

Predictor = Callable[[Path], tuple[np.ndarray, np.ndarray]]


def build_beat_this(device: str, beat_input: str) -> Predictor:
    from openbapo.analysis.audio import decode
    from openbapo.analysis.beats import BeatThisTracker

    tracker = BeatThisTracker(device=device)
    separator = None
    if beat_input == "drums_bass":
        from openbapo.analysis.separate import DemucsSeparator

        separator = DemucsSeparator(device=device)

    def predict(path: Path) -> tuple[np.ndarray, np.ndarray]:
        audio = decode(path)
        if separator is not None:
            audio = separator.separate(audio).mix("drums", "bass")
        result = tracker.track(audio)
        return result.beats, result.downbeats

    return predict


TRACKERS: dict[str, Callable[[str, str], Predictor]] = {
    "beat_this": build_beat_this,
}


def evaluate(system: str, predict: Predictor | None, tracks: Iterable[Track]) -> Summary:
    """predict=None scores the annotations against themselves (self-check)."""
    from tqdm import tqdm

    summary = Summary(system=system)
    for track in tqdm(list(tracks), desc=system, unit="track"):
        start = time.perf_counter()
        beats, downbeats = (
            (track.beats, track.downbeats) if predict is None else predict(track.audio)
        )
        seconds = time.perf_counter() - start
        summary.rows.append(
            Row(
                track=track.name,
                genre=track.genre,
                beat_f1=score(track.beats, beats),
                downbeat_f1=score(track.downbeats, downbeats),
                seconds=seconds,
            )
        )
    return summary


def environment(device: str) -> dict:
    info = {
        "date": dt.date.today().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "packages": {},
    }
    for name in ("openbapo", "torch", "beat-this", "demucs", "mir_eval", "numpy"):
        try:
            info["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][name] = None
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def markdown(summaries: list[Summary], env: dict, problems: list[str], n_tracks: int) -> str:
    lines = [
        "# 第 1 步：节拍评测（GTZAN）",
        "",
        f"由 `scripts/eval_beats.py` 生成于 {env['date']}。",
        f"设备：{env.get('gpu') or env['device']}；Python {env['python']}；"
        + "、".join(f"{k} {v}" for k, v in env["packages"].items() if v),
        "",
        f"曲目数：{n_tracks}（标注与音频都齐全的）。指标：mir_eval `beat.f_measure`，70 ms 容差，"
        f"去掉前 {TRIM_SECONDS:g} 秒。验收门槛：拍点 F1 ≥ {ACCEPTANCE_BEAT_F1:.0%}。",
        "",
        "| 系统 | 拍点 F1 | 强拍 F1 | 每曲耗时（秒） | 验收 |",
        "|---|---|---|---|---|",
    ]
    for s in summaries:
        verdict = (
            "自检"
            if s.system == "annotations"
            else ("通过" if s.mean("beat_f1") >= ACCEPTANCE_BEAT_F1 else "未通过")
        )
        lines.append(
            f"| {s.system} | {s.mean('beat_f1'):.3f} | {s.mean('downbeat_f1'):.3f} | "
            f"{s.mean('seconds'):.2f} | {verdict} |"
        )
    lines += ["", "## 按流派（拍点 F1 / 强拍 F1）", ""]
    systems = [s for s in summaries if s.system != "annotations"]
    lines.append("| 流派 | " + " | ".join(s.system for s in systems) + " |")
    lines.append("|---|" + "---|" * len(systems))
    for genre in GENRES:
        if not any(any(r.genre == genre for r in s.rows) for s in systems):
            continue
        cells = [
            f"{s.mean('beat_f1', genre):.3f} / {s.mean('downbeat_f1', genre):.3f}" for s in systems
        ]
        lines.append(f"| {genre} | " + " | ".join(cells) + " |")
    if problems:
        lines += ["", "## 跳过的文件", ""] + [f"- {p}" for p in problems]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audio", type=Path, required=True, help="GTZAN audio root")
    parser.add_argument("--annotations", type=Path, required=True, help="directory of .beats files")
    parser.add_argument("--trackers", default="beat_this", help=f"comma list of {sorted(TRACKERS)}")
    parser.add_argument("--inputs", default="mix", help="comma list of: mix, drums_bass")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, help="only the first N tracks (for a dry run)")
    parser.add_argument("--out", default="docs/benchmarks/step-1-beats", help="output path stem")
    args = parser.parse_args(argv)

    from openbapo.analysis.device import resolve_device

    device = resolve_device(args.device)
    tracks, problems = load_tracks(args.audio, args.annotations)
    for problem in problems:
        print(f"skip: {problem}", file=sys.stderr)
    if not tracks:
        return 1
    if args.limit:
        tracks = tracks[: args.limit]

    summaries = [evaluate("annotations", None, tracks)]
    self_check = summaries[0].mean("beat_f1")
    if abs(self_check - 1.0) > 1e-9:
        print(f"self-check failed: annotations vs themselves scored {self_check}", file=sys.stderr)
        return 1
    for name in args.trackers.split(","):
        for beat_input in args.inputs.split(","):
            predict = TRACKERS[name](device, beat_input)
            label = name if beat_input == "mix" else f"{name} ({beat_input})"
            summaries.append(evaluate(label, predict, tracks))

    env = environment(device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": env,
        "metric": {
            "tool": "mir_eval.beat.f_measure",
            "tolerance_seconds": 0.07,
            "trim_seconds": TRIM_SECONDS,
        },
        "acceptance_beat_f1": ACCEPTANCE_BEAT_F1,
        "tracks": len(tracks),
        "skipped": problems,
        "systems": [s.as_dict() for s in summaries],
    }
    out.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    out.with_suffix(".md").write_text(
        markdown(summaries, env, problems, len(tracks)), encoding="utf-8"
    )
    print(out.with_suffix(".md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
