"""Chord and key accuracy of the expert committee on GuitarSet and AAM (step-2 acceptance).

Usage:
    python scripts/eval_chords.py --guitarset DIR --aam DIR [--aam-count 500] \
        [--experts btc,lvchordia,chroma] [--device auto] [--limit N] [--no-tune] \
        [--no-stems] [--cache DIR] [--out docs/benchmarks/step-2-chords]
    python scripts/eval_chords.py --guitarset DIR --aam DIR --inspect   # show file formats and exit

Needs `pip install -e ".[analysis]" -r scripts/requirements-eval.txt` and the
datasets laid out as described in docs/benchmarks/README.md. Writes OUT.json
(every number) and OUT.md (the summary tables).

Protocol (docs/plans/step-2.md, decisions 2-4):
- the beat grid comes from the datasets' own beat annotations, so chord
  numbers are not confounded by beat-tracking errors;
- stems come from Demucs (for the bass line) unless --no-stems;
- every expert alone, the committee with default parameters, and the
  committee with parameters tuned on GuitarSet players 00-02 are scored;
  GuitarSet players 03-05 and AAM are the report sets;
- chord metrics are mir_eval's majmin / sevenths / majmin_inv /
  sevenths_inv / root (duration-weighted); the key metric is exact
  tonic+mode accuracy plus mir_eval's weighted key score.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import itertools
import json
import pickle
import platform
import re
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ACCEPTANCE_MAJMIN = 0.80
ACCEPTANCE_KEY = 0.85
TUNE_PLAYERS = ("00", "01", "02")
CHORD_METRICS = ("majmin", "sevenths", "majmin_inv", "sevenths_inv", "root")
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg")


# --- data ----------------------------------------------------------------------


@dataclass(frozen=True)
class Track:
    dataset: str  # "guitarset" | "aam"
    split: str  # "tune" | "report"
    name: str
    group: str  # GuitarSet style or AAM genre-ish bucket, for the breakdown
    audio: Path
    chord_intervals: np.ndarray  # (n, 2) seconds
    chord_labels: list[str]  # Harte
    key: tuple[int, str] | None  # (tonic pitch class, "major" | "minor")
    beats: np.ndarray  # seconds
    downbeats: np.ndarray  # seconds
    key_share: float = 1.0  # fraction of the piece the reference key covers (AAM modulates)
    key_times: dict[str, float] = field(default_factory=dict)  # "F major" -> seconds, per key
    # GuitarSet's second chord annotation: what the player actually sounded
    # (substitutions and all), scored alongside the lead sheet as a reference.
    performed_intervals: np.ndarray | None = None
    performed_labels: list[str] | None = None
    # Song-book samples: the manifest entry (origin, bars to review, capo ...), for the report.
    info: dict = field(default_factory=dict)


def read_jams(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def jams_observations(annotation: dict) -> list[dict]:
    """JAMS stores observations as a list of dicts (0.3) or a dict of lists (0.2)."""
    data = annotation["data"]
    if isinstance(data, dict):
        keys = list(data)
        return [dict(zip(keys, values, strict=True)) for values in zip(*data.values(), strict=True)]
    return list(data)


def guitarset_tracks(root: Path, *, chords: str = "instructed", audio: str = "mic") -> list[Track]:
    """GuitarSet from its Zenodo layout: annotation/*.jams + audio_mono-{mic,pickup_mix}/."""
    jams_files = sorted(root.rglob("*.jams"))
    tracks = []
    for jams_path in jams_files:
        jam = read_jams(jams_path)
        by_ns: dict[str, list[dict]] = {}
        for ann in jam["annotations"]:
            by_ns.setdefault(ann["namespace"], []).append(ann)
        chord_anns = by_ns.get("chord", [])
        if not chord_anns:
            continue
        # The first chord annotation is the lead sheet ("instructed"), the
        # second the performed chords inferred from the notes.
        ann = chord_anns[0 if chords == "instructed" else min(1, len(chord_anns) - 1)]
        obs = jams_observations(ann)
        intervals = np.array([[o["time"], o["time"] + o["duration"]] for o in obs], dtype=float)
        labels = [str(o["value"]) for o in obs]
        performed_intervals = performed_labels = None
        if chords == "instructed" and len(chord_anns) > 1:
            performed = jams_observations(chord_anns[1])
            performed_intervals = np.array(
                [[o["time"], o["time"] + o["duration"]] for o in performed], dtype=float
            )
            performed_labels = [str(o["value"]) for o in performed]
        key = None
        if by_ns.get("key_mode"):
            key = parse_key(str(jams_observations(by_ns["key_mode"][0])[0]["value"]))
        beats, downbeats = [], []
        for o in jams_observations(by_ns["beat_position"][0]) if by_ns.get("beat_position") else []:
            beats.append(float(o["time"]))
            if int(o["value"].get("position", 0)) == 1:
                downbeats.append(float(o["time"]))
        stem = jams_path.stem
        suffix = "_mic" if audio == "mic" else "_mix"
        candidates = list(root.rglob(f"{stem}{suffix}.wav")) or list(root.rglob(f"{stem}.wav"))
        if not candidates:
            print(f"skip: no audio for {stem}", file=sys.stderr)
            continue
        player = stem[:2]
        style = stem.split("_", 1)[1][:2] if "_" in stem else "??"
        tracks.append(
            Track(
                dataset="guitarset",
                split="tune" if player in TUNE_PLAYERS else "report",
                name=stem,
                group=style,
                audio=candidates[0],
                chord_intervals=intervals,
                chord_labels=labels,
                key=key,
                beats=np.array(beats, dtype=float),
                downbeats=np.array(downbeats, dtype=float),
                performed_intervals=performed_intervals,
                performed_labels=performed_labels,
            )
        )
    return tracks


def sheet_tracks(root: Path) -> list[Track]:
    """Renders of lead sheets, read through the sample set's manifest.json.

    The reference is the sheet itself in the sheet's timing: sounding chords,
    sung key and beat grid. These test the analysis on a song book without
    any recording; they are the acceptance set since decision 11. Only songs
    the manifest lists are scored (`scripts/sheet_ocr.py manifest DIR`
    rebuilds it from the files; decision 12).
    """
    from opencapo.sheet import load_manifest

    manifest = load_manifest(root)
    tracks = []
    for song in manifest["songs"]:
        files = song["files"]
        ref_path, audio = root / files["reference"], root / files["audio"]
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        intervals = np.array([[c[0], c[1]] for c in ref["chords"]], dtype=float).reshape(-1, 2)
        labels = [str(c[2]) for c in ref["chords"]]
        key = parse_key(str(ref.get("key", "")).replace(" ", ":"))
        tracks.append(
            Track(
                dataset="sheets",
                split="report",
                name=str(song["name"]),
                group=str(song.get("title") or song["name"]),
                audio=audio,
                chord_intervals=intervals,
                chord_labels=labels,
                key=key,
                beats=np.array(ref["beats"], dtype=float),
                downbeats=np.array(ref.get("downbeats", []), dtype=float),
                key_times={} if key is None else {key_name(key): float(ref.get("duration", 0.0))},
                info=dict(song),
            )
        )
    return tracks


def sheet_manifest_note(root: Path) -> str:
    """One line for the report's notes: which sample set the song-book numbers come from."""
    from opencapo.sheet import load_manifest

    m = load_manifest(root)
    songs = m["songs"]
    code = m.get("code") or {}
    to_review = sum(int(s.get("to_review", 0)) for s in songs)
    ocr = sum(1 for s in songs if s.get("source") == "ocr")
    manual = sum(1 for s in songs if s.get("source") == "manual")
    return (
        f"曲库合成样本集：{len(songs)} 首（识谱 {ocr}、人工录入 {manual}），"
        f"待复核 {to_review} 小节；manifest 生成于 {m.get('generated')}，"
        f"代码 {code.get('commit') or code.get('opencapo') or '未知'}，渲染参数 {m.get('render')}。"
    )


def read_arff(path: Path) -> tuple[list[str], list[list[str]]]:
    """Minimal ARFF reader: attribute names and the data rows as strings.

    AAM's beatinfo files omit the `@DATA` line, so the first line after the
    header that does not start with `@` also opens the data section.
    """
    attributes: list[str] = []
    rows: list[list[str]] = []
    in_data = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if not in_data and line.lower().startswith("@data"):
            in_data = True
        elif not in_data and line.startswith("@"):
            match = re.match(r"@attribute\s+('([^']*)'|\"([^\"]*)\"|(\S+))", line, re.IGNORECASE)
            if match:
                attributes.append(match.group(2) or match.group(3) or match.group(4) or "")
        else:
            in_data = True
            rows.append([cell.strip().strip("'\"") for cell in _split_csv(line)])
    return attributes, rows


def _split_csv(line: str) -> list[str]:
    out, current, quote = [], "", None
    for ch in line:
        if quote:
            if ch == quote:
                quote = None
            else:
                current += ch
        elif ch in "'\"":
            quote = ch
        elif ch == ",":
            out.append(current)
            current = ""
        else:
            current += ch
    out.append(current)
    return out


def _find_column(attributes: Sequence[str], *needles: str) -> int:
    for i, name in enumerate(attributes):
        lowered = name.lower()
        if any(n in lowered for n in needles):
            return i
    raise KeyError(f"no column matching {needles} in {list(attributes)}")


def aam_tracks(root: Path, count: int) -> list[Track]:
    """AAM v1.1.0 from its Zenodo layout (annotations zip + audio-mixes zip).

    Each track id has `<id>_beatinfo.arff` (one row per chord onset: start
    time, bar count, quarter count, chord name; not every beat is listed),
    `<id>_segments.arff` (one row per section: start, mark, tempo, key, ...,
    closed by an `end` row) and `<id>_onsets.arff` (unused). Chords are the
    beatinfo rows with repeats merged; the key is the section key that lasts
    longest; the beat grid is synthesised from the section tempi and
    anchored on the bar / quarter counts. Tracks are the first `count` ids in
    sorted order. Audio is any file under root whose name starts with the
    id, preferring one with "mix" in its name.
    """
    annotations: dict[str, dict[str, Path]] = {}
    for path in sorted(root.rglob("*.arff")):
        match = re.match(r"^(\d{4})_(beatinfo|segments)\.arff$", path.name, re.IGNORECASE)
        if match:
            annotations.setdefault(match.group(1), {})[match.group(2).lower()] = path
    audio_by_id: dict[str, Path] = {}
    candidates = [p for p in root.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES]
    for path in sorted(candidates, key=lambda p: ("mix" not in p.stem.lower(), p.name)):
        match = re.match(r"^(\d{4})", path.name)
        if match:
            audio_by_id.setdefault(match.group(1), path)
    tracks = []
    for track_id in sorted(annotations)[:count]:
        files = annotations[track_id]
        audio = audio_by_id.get(track_id)
        if audio is None or "beatinfo" not in files or "segments" not in files:
            print(f"skip: AAM {track_id} missing audio or annotations", file=sys.stderr)
            continue
        try:
            tracks.append(aam_track(track_id, audio, files["beatinfo"], files["segments"]))
        except (KeyError, ValueError, IndexError) as e:
            print(f"skip: AAM {track_id} unreadable annotations ({e})", file=sys.stderr)
    return tracks


def read_aam_beatinfo(path: Path) -> list[tuple[float, int, int, str]]:
    """(start, bar, quarter, chord name) per chord-onset row."""
    attrs, rows = read_arff(path)
    t_col = _find_column(attrs, "start", "time")
    bar_col = _find_column(attrs, "bar")
    quarter_col = _find_column(attrs, "quarter")
    chord_col = _find_column(attrs, "chord")
    return [
        (float(r[t_col]), int(float(r[bar_col])), int(float(r[quarter_col])), r[chord_col])
        for r in rows
    ]


def read_aam_segments(path: Path) -> tuple[list[tuple[float, float, str]], float | None]:
    """(start, tempo, key) per section with tempo > 0, and the `end` row's time."""
    attrs, rows = read_arff(path)
    t_col = _find_column(attrs, "start", "time")
    tempo_col = _find_column(attrs, "tempo")
    key_col = _find_column(attrs, "key")
    sections, end = [], None
    for r in rows:
        start, tempo = float(r[t_col]), float(r[tempo_col])
        if tempo > 0:
            sections.append((start, tempo, r[key_col]))
        else:
            end = start
    return sections, end


def aam_track(track_id: str, audio: Path, beatinfo: Path, segments: Path) -> Track:
    anchors = read_aam_beatinfo(beatinfo)
    sections, end = read_aam_segments(segments)
    if not anchors or not sections:
        raise ValueError("no chord rows or sections")
    if end is None:
        end = anchors[-1][0] + 60.0 / sections[-1][1]
    intervals, labels = aam_chord_segments(anchors, end)
    beats, downbeats = aam_beat_grid(sections, end, anchors)
    key, share = aam_key(sections, end)
    return Track(
        dataset="aam",
        split="report",
        name=track_id,
        group="aam",
        audio=audio,
        chord_intervals=intervals,
        chord_labels=labels,
        key=key,
        beats=beats,
        downbeats=downbeats,
        key_share=share,
        key_times=aam_key_times(sections, end),
    )


def aam_chord_segments(
    anchors: Sequence[tuple[float, int, int, str]], end: float
) -> tuple[np.ndarray, list[str]]:
    """Merge consecutive rows with the same label; each segment ends where the next starts."""
    merged: list[tuple[float, str]] = []
    for start, _, _, name in anchors:
        label = normalize_aam_label(name)
        if not merged or merged[-1][1] != label:
            merged.append((start, label))
    starts = [m[0] for m in merged]
    ends = [*starts[1:], max(end, starts[-1])]
    keep = [i for i, (s, e) in enumerate(zip(starts, ends, strict=True)) if e > s]
    intervals = np.array([[starts[i], ends[i]] for i in keep], dtype=float).reshape(-1, 2)
    return intervals, [merged[i][1] for i in keep]


def aam_key(
    sections: Sequence[tuple[float, float, str]], end: float
) -> tuple[tuple[int, str] | None, float]:
    """The key that covers the most time and its share of the keyed time.

    AAM pieces may modulate between sections; the share tells modulating
    pieces (share < 1) from single-key ones.
    """
    time_by_key = aam_key_times(sections, end)
    if not time_by_key:
        return None, 1.0
    best = max(time_by_key, key=lambda k: time_by_key[k])
    total = sum(time_by_key.values())
    return parse_key(best), (time_by_key[best] / total if total > 0 else 1.0)


def aam_key_times(sections: Sequence[tuple[float, float, str]], end: float) -> dict[str, float]:
    """Seconds spent in each key, keyed by the canonical "F major" spelling."""
    time_by_key: dict[str, float] = {}
    bounds = [*(s[0] for s in sections[1:]), end]
    for (start, _, key), next_start in zip(sections, bounds, strict=True):
        parsed = parse_key(key) if key.strip() else None
        if parsed is not None:
            name = key_name(parsed)
            time_by_key[name] = time_by_key.get(name, 0.0) + max(0.0, next_start - start)
    return time_by_key


def key_name(key: tuple[int, str]) -> str:
    return f"{KEY_NAMES[key[0]]} {key[1]}"


KEY_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def aam_beat_grid(
    sections: Sequence[tuple[float, float, str]],
    end: float,
    anchors: Sequence[tuple[float, int, int, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """Beats at each section's tempo; downbeats phased by the bar / quarter anchors."""
    beats_per_bar = aam_beats_per_bar(anchors, sections)
    beats: list[float] = []
    downbeats: list[float] = []
    bounds = [*(s[0] for s in sections[1:]), end]
    for (start, tempo, _), next_start in zip(sections, bounds, strict=True):
        period = 60.0 / tempo
        count = max(0, round((next_start - start) / period))
        phase = 0
        for t, _, quarter, _ in anchors:
            if start - 1e-6 <= t < next_start:
                phase = (round((t - start) / period) - (quarter - 1)) % beats_per_bar
                break
        for k in range(count):
            t = start + k * period
            beats.append(t)
            if (k - phase) % beats_per_bar == 0:
                downbeats.append(t)
    return np.array(beats, dtype=float), np.array(downbeats, dtype=float)


def aam_beats_per_bar(
    anchors: Sequence[tuple[float, int, int, str]], sections: Sequence[tuple[float, float, str]]
) -> int:
    """Infer the meter from pairs of anchors in different bars (fallback: the largest quarter)."""
    votes: dict[int, int] = {}
    for (t1, b1, q1, _), (t2, b2, q2, _) in itertools.pairwise(anchors):
        if b2 <= b1:
            continue
        tempo = next((s[1] for s in reversed(sections) if s[0] <= t1 + 1e-6), sections[0][1])
        n = (t2 - t1) * tempo / 60.0
        estimate = (n - (q2 - q1)) / (b2 - b1)
        if abs(estimate - round(estimate)) < 0.1 and 1 <= round(estimate) <= 12:
            votes[round(estimate)] = votes.get(round(estimate), 0) + 1
    if votes:
        return max(votes, key=lambda k: (votes[k], -k))
    return max((a[2] for a in anchors), default=4) or 4


PITCH = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def parse_key(text: str) -> tuple[int, str] | None:
    """`A:minor`, `Eb:major`, `F# minor`, `Cm` -> (pitch class, mode)."""
    t = text.strip()
    if not t or t.upper() in ("N", "NONE"):
        return None
    match = re.match(r"^([A-Ga-g])([#b]*)\s*[:\s]?\s*(major|minor|maj|min|m|M)?$", t)
    if not match:
        return None
    pc = PITCH[match.group(1).upper()]
    for acc in match.group(2):
        pc += 1 if acc == "#" else -1
    mode_text = match.group(3) or "major"
    mode = "minor" if mode_text in ("minor", "min", "m") else "major"
    return pc % 12, mode


def normalize_aam_label(text: str) -> str:
    """AAM chord labels to Harte; `N` for none. Unknown spellings pass through."""
    t = text.strip()
    if not t or t.upper().replace(".", "") in ("N", "NC", "NONE", "NO CHORD"):
        return "N"  # AAM writes no-chord as N.C.
    if ":" in t or t in ("X",):
        return t
    match = re.match(r"^([A-G][#b]?)(.*)$", t)
    if not match:
        return t
    root, rest = match.groups()
    rest = rest.strip()
    table = {"": "maj", "maj": "maj", "M": "maj", "m": "min", "min": "min", "dim": "dim",
             "aug": "aug", "7": "7", "maj7": "maj7", "M7": "maj7", "m7": "min7", "min7": "min7",
             "sus2": "sus2", "sus4": "sus4", "dim7": "dim7", "hdim7": "hdim7", "m7b5": "hdim7",
             "6": "maj6", "m6": "min6", "5": "5", "dom7": "7", "min7b5": "hdim7"}  # fmt: skip
    return f"{root}:{table.get(rest, rest)}"


# --- scoring -------------------------------------------------------------------


def chord_scores(
    ref_intervals: np.ndarray,
    ref_labels: Sequence[str],
    est_intervals: np.ndarray,
    est_labels: Sequence[str],
) -> dict[str, float]:
    import mir_eval

    if len(ref_intervals) == 0:
        return {m: float("nan") for m in CHORD_METRICS}
    est_intervals, est_labels = mir_eval.util.adjust_intervals(
        est_intervals, list(est_labels), ref_intervals.min(), ref_intervals.max(), "N", "N"
    )
    # An estimate that starts after the reference ends (the rendered song
    # book carries a second of tail) is trimmed to nothing: drop it.
    keep = est_intervals[:, 1] - est_intervals[:, 0] > 0
    est_intervals, est_labels = (
        est_intervals[keep],
        [label for label, k in zip(est_labels, keep, strict=True) if k],
    )
    scores = mir_eval.chord.evaluate(
        ref_intervals, list(ref_labels), est_intervals, list(est_labels)
    )
    return {m: float(scores[m]) for m in CHORD_METRICS}


def key_hit_major(track: Track, estimate: tuple[int, str] | None) -> float:
    """1.0 when the estimate is any key holding at least a third of the piece."""
    if estimate is None or track.key is None:
        return 0.0 if track.key is not None else float("nan")
    total = sum(track.key_times.values())
    major = {k for k, t in track.key_times.items() if t >= total / 3} or {key_name(track.key)}
    return float(key_name(estimate) in major)


def key_scores(
    reference: tuple[int, str] | None, estimate: tuple[int, str] | None
) -> dict[str, float]:
    import mir_eval

    est = None if estimate is None else key_name(estimate)
    if reference is None:
        return {
            "key_exact": float("nan"),
            "key_tonal": float("nan"),
            "key_weighted": float("nan"),
            "key_est": est,
        }
    ref = key_name(reference)
    if estimate is None:
        return {
            "key_exact": 0.0,
            "key_tonal": 0.0,
            "key_weighted": 0.0,
            "key_ref": ref,
            "key_relation": "none",
        }
    weighted = float(mir_eval.key.weighted_score(ref, est))
    relation = KEY_RELATIONS.get(round(weighted, 1), "other")
    return {
        "key_exact": float(reference == estimate),
        # The sung key's tonal centre: a sheet's "1=X" names the major key of
        # the notes in use even for a song in the relative minor, so for the
        # song book the relative key counts as correct (decision 16).
        "key_tonal": float(reference == estimate or relation == "relative"),
        "key_weighted": weighted,
        "key_ref": ref,
        "key_est": est,
        "key_relation": relation,
    }


# mir_eval's weighted key score names the mistake: 0.5 = a fifth away, 0.3 =
# the relative key, 0.2 = the parallel key.
KEY_RELATIONS = {1.0: "exact", 0.5: "fifth", 0.3: "relative", 0.2: "parallel", 0.0: "other"}
KEY_RELATION_NAMES = ("exact", "fifth", "relative", "parallel", "other", "none")


# --- running the experts ----------------------------------------------------------


@dataclass
class TrackOutputs:
    """Everything the fusion needs for one track, cached so tuning is cheap."""

    beats: list  # opencapo Beat objects
    duration: float
    beat_probs: dict[str, np.ndarray]  # plain per-beat means, as the stage computed them
    bass: object | None
    chroma_scores: np.ndarray
    seconds: dict[str, float] = field(default_factory=dict)
    # Frame-level posteriors (times, float16 probs) per expert, so the beat
    # pooling itself can be varied at scoring time; older caches lack them.
    frame_probs: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def pooled(self, names: Sequence[str], voiced: bool) -> dict[str, np.ndarray]:
        """Per-beat posteriors for these experts; voiced pooling needs frame_probs."""
        if not voiced:
            return {n: self.beat_probs[n] for n in names}
        from opencapo.analysis.chords.fusion import ChordPosteriors, aggregate_to_beats

        cache = self.__dict__.setdefault("_voiced", {})
        frames = getattr(self, "frame_probs", {})  # objects pickled before the field existed
        beat_times = [b.time for b in self.beats]
        for n in names:
            if n in cache:
                continue
            if n in frames:
                times, probs = frames[n]
                post = ChordPosteriors(times=times, probs=probs.astype(np.float64))
                cache[n] = aggregate_to_beats(post, beat_times, self.duration, voiced=True)
            else:
                cache[n] = self.beat_probs[n]  # no frames cached: same as plain pooling
        return {n: cache[n] for n in names}


def build_experts(names: Sequence[str], device: str) -> list:
    from opencapo.analysis.chords.chroma import ChromaTemplateExpert
    from opencapo.analysis.chords.models.btc import BTCExpert
    from opencapo.analysis.chords.models.lvchordia import LVChordiaExpert

    table: dict[str, Callable[[], object]] = {
        "btc": lambda: BTCExpert(device=device),
        "lvchordia": lambda: LVChordiaExpert(device=device),
        "chroma": lambda: ChromaTemplateExpert(),
    }
    return [table[n]() for n in names]


def compute_outputs(track: Track, experts: list, separator, device: str) -> TrackOutputs:  # type: ignore[no-untyped-def]
    from opencapo.analysis.audio import decode
    from opencapo.analysis.chords.bassline import BassLineDetector, bass_source
    from opencapo.analysis.chords.chroma import ChromaTemplateExpert, chromagram, mean_chroma
    from opencapo.analysis.chords.fusion import aggregate_to_beats
    from opencapo.analysis.chords.key import scores_from_chroma
    from opencapo.analysis.grid import build_grid

    audio = decode(track.audio)
    grid = build_grid(track.beats.tolist(), track.downbeats.tolist())
    beats = list(grid.beats)
    beat_times = [b.time for b in beats]
    seconds: dict[str, float] = {}
    stems = None
    if separator is not None:
        start = time.perf_counter()
        stems = separator.separate(audio)
        seconds["separate"] = time.perf_counter() - start
    harmonic = stems.mix("bass", "other", "vocals") if stems is not None else audio
    gram = chromagram(harmonic)
    beat_probs = {}
    frame_probs = {}
    for expert in experts:
        start = time.perf_counter()
        post = (
            expert.from_chromagram(gram)
            if isinstance(expert, ChromaTemplateExpert)
            else expert.estimate(audio, stems)
        )
        beat_probs[expert.name] = aggregate_to_beats(post, beat_times, audio.duration)
        frame_probs[expert.name] = (
            np.asarray(post.times),
            np.asarray(post.probs, dtype=np.float16),
        )
        seconds[expert.name] = time.perf_counter() - start
    bass = BassLineDetector().detect(bass_source(audio, stems), beat_times, audio.duration)
    return TrackOutputs(
        beats=beats,
        duration=audio.duration,
        beat_probs=beat_probs,
        bass=bass,
        chroma_scores=scores_from_chroma(mean_chroma(gram)),
        seconds=seconds,
        frame_probs=frame_probs,
    )


def preliminary_chords(
    outputs: TrackOutputs, expert_names: Sequence[str], model_names: Sequence[str], params
) -> list:  # type: ignore[no-untyped-def]
    """The chords decoded without a key prior; the key is estimated from these."""
    from opencapo.analysis.chords.fusion import decode

    probs = outputs.pooled(expert_names, params.voiced_aggregation)
    models = [n for n in model_names if n in probs]
    return list(
        decode(
            outputs.beats,
            probs,
            models,
            duration=outputs.duration,
            bass=outputs.bass,
            params=params,
        ).chords
    )


def dump_key_inputs(
    path: Path,
    tracks: list[Track],
    outputs: dict[str, TrackOutputs],
    expert_names: Sequence[str],
    model_names: Sequence[str],
    params,  # type: ignore[no-untyped-def]
) -> None:
    """Write what the key estimator sees per track, so its rules can be tuned on a CPU.

    Chords are the committee's own output (not annotations); the reference
    key per track is the only annotation-derived field. gzip when the name
    ends in .gz; read back with scripts/eval_key_inputs.py.
    """
    import gzip

    records = []
    for t in tracks:
        out = outputs[t.name]
        chords = preliminary_chords(out, expert_names, model_names, params)
        records.append(
            {
                "dataset": t.dataset,
                "split": t.split,
                "name": t.name,
                "group": t.group,
                "key_ref": None if t.key is None else key_name(t.key),
                "key_share": t.key_share,
                "key_times": t.key_times,
                "chroma_scores": [round(float(x), 6) for x in out.chroma_scores],
                "chords": [
                    [round(c.start, 3), round(c.end, 3), c.root, c.quality.value] for c in chords
                ],
            }
        )
    text = json.dumps({"params": params.__dict__ | {"weights": dict(params.weights)},
                       "tracks": records}, ensure_ascii=False)  # fmt: skip
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(text.encode("utf-8")))
    else:
        path.write_text(text + "\n", encoding="utf-8")


def run_fusion(
    outputs: TrackOutputs, expert_names: Sequence[str], model_names: Sequence[str], params
):  # type: ignore[no-untyped-def]
    """Chords + key for a subset of experts with given FusionParams (mirrors ChordStage)."""
    from opencapo.analysis.chords.fusion import decode
    from opencapo.analysis.chords.key import estimate_key, scores_from_chords
    from opencapo.analysis.chords.vocab import format_label

    probs = outputs.pooled(expert_names, params.voiced_aggregation)
    models = [n for n in model_names if n in probs]
    preliminary = preliminary_chords(outputs, expert_names, model_names, params)
    key = estimate_key(outputs.chroma_scores, scores_from_chords(preliminary), preliminary)
    final = decode(
        outputs.beats,
        probs,
        models,
        duration=outputs.duration,
        key=key,
        bass=outputs.bass,
        params=params,
    )
    intervals = np.array([[c.start, c.end] for c in final.chords], dtype=float).reshape(-1, 2)
    labels = [format_label(c.root, c.quality, c.bass) for c in final.chords]
    key_tuple = None if key is None else (key.tonic, key.mode.value)
    return intervals, labels, key_tuple


# --- aggregation ----------------------------------------------------------------


def mean_of(rows: Iterable[dict[str, float]], metric: str) -> float:
    values = [r[metric] for r in rows if not np.isnan(r[metric])]
    return float(np.mean(values)) if values else float("nan")


def score_system(
    name: str,
    tracks: list[Track],
    outputs: dict[str, TrackOutputs],
    expert_names,
    model_names,
    params,
) -> dict:  # type: ignore[no-untyped-def]
    rows = []
    for track in tracks:
        out = outputs[track.name]
        intervals, labels, key = run_fusion(out, expert_names, model_names, params)
        row = {
            "track": track.name,
            "dataset": track.dataset,
            "split": track.split,
            "group": track.group,
        }
        if track.info:
            row["info"] = track.info
        row.update(chord_scores(track.chord_intervals, track.chord_labels, intervals, labels))
        row["majmin_performed"] = row["root_performed"] = float("nan")
        if track.performed_intervals is not None and track.performed_labels is not None:
            performed = chord_scores(
                track.performed_intervals, track.performed_labels, intervals, labels
            )
            row["majmin_performed"], row["root_performed"] = performed["majmin"], performed["root"]
        row.update(key_scores(track.key, key))
        row["key_share"] = track.key_share
        row["key_hit_major"] = key_hit_major(track, key)
        row["seconds"] = sum(out.seconds.get(n, 0.0) for n in expert_names)
        rows.append(row)
    return {
        "system": name,
        "experts": list(expert_names),
        "params": params.__dict__ | {"weights": dict(params.weights)},
        "rows": rows,
    }


def summarize(system: dict) -> dict:
    rows = system["rows"]
    guitarset = [r for r in rows if r["dataset"] == "guitarset"]
    subsets = {
        "guitarset_tune": [r for r in guitarset if r["split"] == "tune"],
        "guitarset_report": [r for r in guitarset if r["split"] == "report"],
        "aam": [r for r in rows if r["dataset"] == "aam"],
        "sheets": [r for r in rows if r["dataset"] == "sheets"],
    }
    # GuitarSet pairs every progression as a comping take (chords are played)
    # and a soloing take (a single-note line over the same chart, so the
    # instructed chords are implied rather than sounded); report them apart.
    for split in ("report", "tune"):
        for kind in ("comp", "solo"):
            subsets[f"guitarset_{split}_{kind}"] = [
                r for r in subsets[f"guitarset_{split}"] if r["track"].endswith(f"_{kind}")
            ]
    summary = {}
    for subset, subset_rows in subsets.items():
        if not subset_rows:
            continue
        summary[subset] = {
            m: mean_of(subset_rows, m)
            for m in (*CHORD_METRICS, "key_exact", "key_tonal", "key_weighted", "seconds")
        }
        if any(not np.isnan(r.get("majmin_performed", float("nan"))) for r in subset_rows):
            summary[subset]["majmin_performed"] = mean_of(subset_rows, "majmin_performed")
            summary[subset]["root_performed"] = mean_of(subset_rows, "root_performed")
        summary[subset]["tracks"] = len(subset_rows)
        relations = [r["key_relation"] for r in subset_rows if r.get("key_relation")]
        summary[subset]["key_relations"] = {k: relations.count(k) for k in KEY_RELATION_NAMES}
        single = [r for r in subset_rows if r.get("key_share", 1.0) >= 0.999]
        if len(single) != len(subset_rows):
            # Modulating pieces have no single true key (decision 8): the gate
            # uses the single-key pieces; the rest get the relaxed hit rate.
            summary[subset]["key_exact_single_key"] = mean_of(single, "key_exact")
            summary[subset]["single_key_tracks"] = len(single)
            modulating = [r for r in subset_rows if r not in single]
            summary[subset]["key_hit_major_modulating"] = mean_of(modulating, "key_hit_major")
            summary[subset]["modulating_tracks"] = len(modulating)
        groups = sorted({r["group"] for r in subset_rows})
        if len(groups) > 1:
            summary[subset]["by_group"] = {
                g: {"majmin": mean_of([r for r in subset_rows if r["group"] == g], "majmin"),
                    "key_exact": mean_of([r for r in subset_rows if r["group"] == g], "key_exact")}
                for g in groups
            }  # fmt: skip
    return summary


def tune(tracks: list[Track], outputs: dict[str, TrackOutputs], expert_names, model_names):  # type: ignore[no-untyped-def]
    """Grid search FusionParams on the tuning tracks, maximising majmin."""
    from opencapo.analysis.chords.fusion import FusionParams

    # Model experts weigh 0.5-2; the rule-based chroma expert may also be dropped (0).
    weight_grids = [[0.5, 1.0, 2.0] if n in model_names else [0.0, 0.5, 1.0] for n in expert_names]
    # key_prior 0.5 and change_cost_downbeat 0.1 won every earlier search and are fixed.
    # Searched earlier and settled: bass_weight 1 (no measurable effect either
    # way), voiced pooling off (never chosen).
    grids = {
        "change_cost_beat": [0.5, 1.5, 3.0],
        "no_chord_bias": [0.0, -1.0, -2.0],
        "confidence_power": [0.0, 1.0, 2.0],
    }
    best, best_score = None, -1.0
    weight_combos = list(itertools.product(*weight_grids))
    for weights in weight_combos:
        for cost_b, bias, power in itertools.product(*grids.values()):
            params = FusionParams(
                weights=dict(zip(expert_names, weights, strict=True)),
                key_prior=0.5,
                change_cost_downbeat=0.1,
                change_cost_beat=cost_b,
                no_chord_bias=bias,
                bass_weight=1.0,
                confidence_power=power,
            )
            rows = score_system("tune", tracks, outputs, expert_names, model_names, params)["rows"]
            score = mean_of(rows, "majmin")
            if score > best_score:
                best, best_score = params, score
    return best, best_score


# --- reporting ------------------------------------------------------------------


def environment(device: str) -> dict:
    info = {
        "date": dt.date.today().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "packages": {},
    }
    for name in ("opencapo", "torch", "lv-chordia", "librosa", "mir_eval", "demucs", "numpy"):
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


def fmt(value: float) -> str:
    return "–" if np.isnan(value) else f"{value:.3f}"


def markdown(systems: list[dict], env: dict, notes: list[str]) -> str:
    lines = [
        "# 第 2 步：和弦与调性评测（GuitarSet + AAM）",
        "",
        f"由 `scripts/eval_chords.py` 生成于 {env['date']}。",
        f"设备：{env.get('gpu') or env['device']}；Python {env['python']}；"
        + "、".join(f"{k} {v}" for k, v in env["packages"].items() if v),
        "",
        "拍网格来自数据集自带的拍点标注；分轨来自 Demucs（低音线用）。"
        "指标：mir_eval `chord`（按时长加权）与调性精确匹配。"
        f"验收门槛：负责人曲库谱子渲染的样本集（曲库合成）上 majmin ≥ {ACCEPTANCE_MAJMIN:.0%}、"
        f"调性主音 ≥ {ACCEPTANCE_KEY:.0%}（`docs/plans/step-2.md` 决定 11、16）；"
        "合成音频没有人声和混音，分数是真录音的上限。"
        '"调性精确"要求大小调都对；"调性主音"把关系大小调也算对：'
        "谱子的 1=X 只写音级所在的大调，不区分大小调，曲库一组按它验收，"
        "公开数据集按精确匹配参考。"
        "GuitarSet 与 AAM 作公开数据集参考（决定 7、8：GuitarSet 只看伴奏曲目，"
        "独奏曲目是单音旋律；AAM 的调性只看全曲单一调性的曲目，转调曲目另报命中率）。"
        "调参只用 GuitarSet 吉他手 00–02。",
        "",
    ]
    for subset, title in (
        ("sheets", "曲库合成（负责人谱子渲染的样本集）——验收依据"),
        (
            "guitarset_report_comp",
            "GuitarSet 报数半（吉他手 03–05），仅伴奏（comp）曲目（公开数据集参考）",
        ),
        ("guitarset_report", "GuitarSet 报数半，伴奏 + 独奏（参考）"),
        (
            "guitarset_report_solo",
            "GuitarSet 报数半，仅独奏（solo）曲目：单音旋律，标注是隐含的和弦",
        ),
        ("aam", "AAM（500 首，公开数据集参考）"),
        ("guitarset_tune", "GuitarSet 调参半（吉他手 00–02，仅供对照）"),
        ("guitarset_tune_comp", "GuitarSet 调参半，仅伴奏（comp）曲目"),
    ):
        if not any(subset in s["summary"] for s in systems):
            continue
        lines += [
            f"## {title}",
            "",
            "| 系统 | majmin | sevenths | majmin_inv | sevenths_inv | root "
            "| 调性精确 | 调性主音 | 调性加权 | 每曲耗时 | 验收 |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for s in systems:
            if subset not in s["summary"]:
                continue
            m = s["summary"][subset]
            # The song book is gated on the tonal centre (relative keys count,
            # decision 16); the public sets on the exact key of single-key pieces.
            gated_key = (
                m["key_tonal"]
                if subset == "sheets"
                else m.get("key_exact_single_key", m["key_exact"])
            )
            verdict = (
                "自检"
                if s["system"] == "annotations"
                else (
                    "通过"
                    if m["majmin"] >= ACCEPTANCE_MAJMIN and gated_key >= ACCEPTANCE_KEY
                    else "未通过"
                )
            )
            if subset.startswith("guitarset_tune"):
                verdict = "对照"
            elif subset != "sheets":
                verdict = "参考"
            cells = [
                s["system"],
                *(fmt(m[k]) for k in (*CHORD_METRICS, "key_exact", "key_tonal", "key_weighted")),
                f"{m['seconds']:.2f} s",
                verdict,
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    # The production system: tuned when a tuning split ran, else the defaults
    # (which are the last tuning's result, so the two coincide in a full run).
    final = next(
        (s for s in systems if s["system"] == "committee (tuned)"),
        next((s for s in systems if s["system"] == "committee (default)"), None),
    )
    sheet_rows = [r for r in (final["rows"] if final else []) if r["dataset"] == "sheets"]
    if sheet_rows:
        assert final is not None
        lines += [
            f"## 曲库合成：逐首（{final['system']}）",
            "",
            "| 样本 | 标题 | 来源 | 变调夹 | 小节 | 待复核 | majmin | sevenths | root "
            "| 调性：参考 / 估计 | 每曲耗时 |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(sheet_rows, key=lambda r: r["track"]):
            info = r.get("info") or {}
            source = {"ocr": "识谱", "manual": "人工录入"}.get(str(info.get("source")), "—")
            cells = [
                r["track"],
                str(info.get("title") or r["group"]),
                source,
                str(info.get("capo", "—")),
                str(info.get("bars", "—")),
                str(info.get("to_review", "—")),
                *(fmt(r[k]) for k in ("majmin", "sevenths", "root")),
                f"{r.get('key_ref', '—')} / {r.get('key_est', '—')}"
                + ("（关系调）" if r.get("key_relation") == "relative" else ""),
                f"{r['seconds']:.2f} s",
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    performed_subsets = [
        (sub, title)
        for sub, title in (
            ("guitarset_report_comp", "报数半伴奏"),
            ("guitarset_tune_comp", "调参半伴奏"),
        )
        if any("majmin_performed" in s["summary"].get(sub, {}) for s in systems)
    ]
    if performed_subsets:
        lines += [
            "## GuitarSet 对实奏和弦（参考）",
            "",
            "谱面（lead sheet）是吉他手该弹什么，实奏是从音符转录出的他实际弹了什么"
            "（爵士里常有替代和弦）。"
            '对实奏的分数衡量"听得准不准"，对谱面的分数衡量"能不能还原谱子"；门槛只看谱面。',
            "",
            "| 系统 | " + " | ".join(f"{t} majmin | {t} root" for _, t in performed_subsets) + " |",
            "|---|" + "---|---|" * len(performed_subsets),
        ]
        for s in systems:
            cells = [s["system"]]
            for sub, _ in performed_subsets:
                m = s["summary"].get(sub, {})
                cells += [
                    fmt(m.get("majmin_performed", float("nan"))),
                    fmt(m.get("root_performed", float("nan"))),
                ]
            lines.append("| " + " | ".join(cells) + " |")  # fmt: skip
        lines.append("")
    if final is not None:
        lines += ["## 按风格 / 流派（committee (tuned) 的 majmin / 调性精确）", ""]
        for subset in ("guitarset_report_comp", "guitarset_report_solo", "aam", "sheets"):
            groups = final["summary"].get(subset, {}).get("by_group")
            if groups:
                lines.append(f"**{subset}**")
                lines.append("")
                lines.append("| 组 | majmin | 调性精确 |")
                lines.append("|---|---|---|")
                for g, v in groups.items():
                    lines.append(f"| {g} | {fmt(v['majmin'])} | {fmt(v['key_exact'])} |")
                lines.append("")
        lines += ["## 调性错误类型（committee (tuned)，按曲目数）", ""]
        lines.append(
            "| 子集 | 正确 | 差五度 | 关系大小调 | 同名大小调 | 其它 | 无估计 "
            "| 单一调性曲目的正确率（门槛依据） | 转调曲目命中任一主要调 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for subset in ("guitarset_report_comp", "guitarset_report_solo", "aam", "sheets"):
            m = final["summary"].get(subset)
            if not m or "key_relations" not in m:
                continue
            rel = m["key_relations"]
            single, modulating = "全部单一调性", "—"
            if "key_exact_single_key" in m:
                single = f"{fmt(m['key_exact_single_key'])}（{m['single_key_tracks']} 首）"
                modulating = f"{fmt(m['key_hit_major_modulating'])}（{m['modulating_tracks']} 首）"
            cells = [subset, *(str(rel[k]) for k in KEY_RELATION_NAMES), single, modulating]
            lines.append("| " + " | ".join(cells) + " |")  # fmt: skip
        lines.append("")
        lines += ["## 调参结果", "", "```json", json.dumps(final["params"], indent=2), "```", ""]
    if notes:
        lines += ["## 备注", ""] + [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"


# --- main -------------------------------------------------------------------------


def inspect(tracks: list[Track]) -> None:
    seen = set()
    for t in tracks:
        if t.dataset in seen:
            continue
        seen.add(t.dataset)
        first = list(zip(t.chord_intervals[:3].tolist(), t.chord_labels[:3], strict=True))
        print(f"[{t.dataset}] {t.name}: audio={t.audio}")
        print(f"  chords: {len(t.chord_labels)} segments, first: {first}")
        print(f"  key: {t.key}; beats: {len(t.beats)} (downbeats {len(t.downbeats)})")
        print(f"  first beats: {t.beats[:4].tolist()}")
    # Every reference label of every track must parse (ours and mir_eval's
    # grammar agree on Harte), or the self-check in main() would fail later.
    unknown: dict[str, set[str]] = {}
    for t in tracks:
        for label in t.chord_labels:
            if not _parses(label):
                unknown.setdefault(label, set()).add(t.name)
    for label, names in sorted(unknown.items()):
        print(
            f"  WARNING unparsable label {label!r} in {len(names)} tracks, e.g. {sorted(names)[0]}"
        )
    counts: dict[tuple[str, str], int] = {}
    for t in tracks:
        counts[(t.dataset, t.split)] = counts.get((t.dataset, t.split), 0) + 1
    print("counts:", counts)


def _parses(label: str) -> bool:
    from opencapo.analysis.chords.vocab import LabelError, parse_label

    try:
        parse_label(label)
        return True
    except LabelError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--guitarset", type=Path, help="GuitarSet root (annotation/ + audio_mono-*/)"
    )
    parser.add_argument("--aam", type=Path, help="AAM root (annotations + mixes)")
    parser.add_argument(
        "--sheets", type=Path, help="song-book sample set (directory with manifest.json)"
    )
    parser.add_argument("--aam-count", type=int, default=500)
    parser.add_argument(
        "--guitarset-chords", choices=("instructed", "performed"), default="instructed"
    )
    parser.add_argument("--guitarset-audio", choices=("mic", "mix"), default="mic")
    parser.add_argument("--experts", default="btc,lvchordia,chroma")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, help="first N tracks of each dataset (dry run)")
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--no-stems", action="store_true", help="skip Demucs (no bass line)")
    parser.add_argument("--cache", type=Path, help="directory to cache expert outputs between runs")
    parser.add_argument("--inspect", action="store_true", help="print dataset formats and exit")
    parser.add_argument("--out", default="docs/benchmarks/step-2-chords")
    parser.add_argument(
        "--dump-key-inputs",
        type=Path,
        help="also write the key estimator's inputs per track (JSON, gzip if .gz)",
    )
    args = parser.parse_args(argv)

    tracks: list[Track] = []
    if args.guitarset:
        tracks += guitarset_tracks(
            args.guitarset, chords=args.guitarset_chords, audio=args.guitarset_audio
        )
    if args.aam:
        tracks += aam_tracks(args.aam, args.aam_count)
    if args.sheets:
        tracks += sheet_tracks(args.sheets)
    if not tracks:
        print("no tracks found; check --guitarset / --aam / --sheets", file=sys.stderr)
        return 1
    if args.limit:
        tracks = [
            t
            for d in ("guitarset", "aam", "sheets")
            for t in [x for x in tracks if x.dataset == d][: args.limit]
        ]
    if args.inspect:
        inspect(tracks)
        return 0

    from opencapo.analysis.chords.fusion import FusionParams
    from opencapo.analysis.device import resolve_device

    device = resolve_device(args.device)
    expert_names = [n for n in args.experts.split(",") if n]
    experts = build_experts(expert_names, device)
    model_names = [e.name for e in experts if e.is_model]
    separator = None
    if not args.no_stems:
        from opencapo.analysis.separate import DemucsSeparator

        separator = DemucsSeparator(device=device)

    from tqdm import tqdm

    outputs: dict[str, TrackOutputs] = {}
    cache_dir = args.cache
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for track in tqdm(tracks, desc="experts", unit="track"):
        cache_file = cache_dir / f"{track.dataset}_{track.name}.pkl" if cache_dir else None
        if cache_file and cache_file.exists():
            outputs[track.name] = pickle.loads(cache_file.read_bytes())
            if set(expert_names) <= set(outputs[track.name].beat_probs):
                continue
        outputs[track.name] = compute_outputs(track, experts, separator, device)
        if cache_file:
            cache_file.write_bytes(pickle.dumps(outputs[track.name]))

    without_frames = sum(1 for o in outputs.values() if not getattr(o, "frame_probs", None))
    if without_frames:
        print(
            f"note: {without_frames} cached tracks have no frame posteriors; "
            "voiced pooling falls back to plain means there",
            file=sys.stderr,
        )
    # Self-check: the annotations against themselves must score 1.0.
    self_rows = []
    for t in tracks:
        row = {
            "track": t.name,
            "dataset": t.dataset,
            "split": t.split,
            "group": t.group,
            "seconds": 0.0,
        }
        row.update(
            chord_scores(t.chord_intervals, t.chord_labels, t.chord_intervals, t.chord_labels)
        )
        row.update(key_scores(t.key, t.key))
        self_rows.append(row)
    self_check = {"system": "annotations", "experts": [], "params": {}, "rows": self_rows}
    if abs(mean_of(self_rows, "majmin") - 1.0) > 1e-6:
        print(
            f"self-check failed: annotations vs themselves scored {mean_of(self_rows, 'majmin')}",
            file=sys.stderr,
        )
        return 1

    systems = [self_check]
    default = FusionParams()
    for name in expert_names:
        systems.append(score_system(name, tracks, outputs, [name], model_names, default))
    systems.append(
        score_system("committee (default)", tracks, outputs, expert_names, model_names, default)
    )
    notes = []
    if args.sheets:
        notes.append(sheet_manifest_note(args.sheets))
    tune_tracks = [t for t in tracks if t.split == "tune"]
    final_params = default
    if not args.no_tune and tune_tracks:
        tuned, tuned_score = tune(tune_tracks, outputs, expert_names, model_names)
        systems.append(
            score_system("committee (tuned)", tracks, outputs, expert_names, model_names, tuned)
        )
        notes.append(f"调参集上的最佳 majmin：{tuned_score:.3f}（{len(tune_tracks)} 段）")
        final_params = tuned
    if args.dump_key_inputs:
        dump_key_inputs(
            args.dump_key_inputs, tracks, outputs, expert_names, model_names, final_params
        )
    for s in systems:
        s["summary"] = summarize(s)

    env = environment(device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": env,
        "acceptance": {"majmin": ACCEPTANCE_MAJMIN, "key_tonal": ACCEPTANCE_KEY},
        "tracks": {
            "guitarset": sum(t.dataset == "guitarset" for t in tracks),
            "aam": sum(t.dataset == "aam" for t in tracks),
            "sheets": sum(t.dataset == "sheets" for t in tracks),
        },
        "systems": [{k: v for k, v in s.items()} for s in systems],
    }
    out.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8"
    )
    out.with_suffix(".md").write_text(markdown(systems, env, notes), encoding="utf-8")
    print(out.with_suffix(".md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
