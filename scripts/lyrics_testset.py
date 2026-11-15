"""Build the step-5 lyrics test set from open datasets (docs/plans/step-5.md, section 3).

Usage:
    python scripts/lyrics_testset.py --data DIR [--out DIR]

DIR holds the downloaded datasets:
    DIR/lyrics/jamendolyrics/   JamendoLyrics (HF jamendolyrics/jamendolyrics, subset en)
    DIR/lyrics/jam-alt/         Jam-ALT (HF jamendolyrics/jam-alt): reference text for WER
    DIR/lyrics/gtsinger/        GTSinger Chinese/ and English/ Control_Group folders
    DIR/lyrics/mir-1k/          MIR-1K UndividedWavfile/
    DIR/lyrics/lyricalignment/  MIR1k_partial_align.json (character timings for 17 songs)
    DIR/aam/                    AAM mixes, the accompaniment and instrumental source
    DIR/gtzan/genres/           GTZAN classical and jazz, more instrumentals

Writes OUT (default DIR/lyrics/testset) with the generated audio and
manifest.json. Nothing here goes into the repository: the manifest keeps the
song titles, the repository only the anonymised ids.

Sets (the `set` field):
    en-test    JamendoLyrics en, real mixes; line times are the human line annotations
    zh-test    GTSinger Chinese, dry vocals and the same vocals over an AAM accompaniment
    zh-ref     MIR-1K, amateur singing over real karaoke backing (reference only)
    dev        GTSinger Chinese and English songs not in zh-test (model selection, tuning)
    instr      instrumental tracks (AAM, GTZAN classical / jazz); split test / dev

Lines. JamendoLyrics has line annotations. GTSinger and MIR-1K have word or
character timings only, so a line is a run of words between pauses: a new line
starts at every GTSinger phrase clip, after a breath / silence mark of at least
LINE_PAUSE seconds (GTSinger) or a gap of at least LINE_PAUSE between two
characters (MIR-1K). Lines shorter than MIN_LINE_WORDS words join the line
before them.

GTSinger phrase clips are joined in order with a random 0.5-2.5 s gap; the
"mix" condition puts the joined vocals over an AAM track at a vocal-to-
accompaniment ratio of VOCAL_DB, with INTRO s of accompaniment before the
first phrase and OUTRO s after the last.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SR = 44100
SEED = 5
LINE_PAUSE = 0.4
MIN_LINE_WORDS = 3
CLIP_GAP = (0.5, 2.5)
VOCAL_DB = 0.0
INTRO, OUTRO = 8.0, 4.0
ZH_TEST, ZH_DEV, EN_DEV = 10, 10, 10
INSTR_TEST = {"aam": 10, "classical": 5, "jazz": 5}
INSTR_DEV = {"aam": 5, "classical": 3, "jazz": 2}
MARK = re.compile(r"^<\w+>$")  # GTSinger <AP> (breath) and <SP> (silence)


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Line:
    start: float
    end: float
    text: str


@dataclass
class Entry:
    id: str
    set: str
    split: str  # "test" | "dev" | "ref"
    lang: str  # "zh" | "en" | ""
    condition: str  # "real" | "dry" | "mix" | "instrumental"
    audio: str  # relative to the manifest
    duration: float
    lines: list[Line] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    vocal_regions: list[tuple[float, float]] = field(default_factory=list)
    reference_text: str = ""  # recognition reference when it differs from the lines
    source: dict[str, str] = field(default_factory=dict)  # dataset, title; local only


def stable_order(names: list[str]) -> list[str]:
    return sorted(names, key=lambda n: hashlib.sha1(f"{SEED}:{n}".encode()).hexdigest())


def load_mono(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    return soxr.resample(mono, sr, SR) if sr != SR else mono


def write(out: Path, rel: str, audio: np.ndarray) -> float:
    path = out / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(audio)))
    sf.write(path, audio * (0.98 / peak) if peak > 0.98 else audio, SR, subtype="PCM_16")
    return len(audio) / SR


def words_to_lines(words: list[Word], breaks: set[int], joiner: str) -> list[Line]:
    """Group words into lines; `breaks` are indices of words that start a line."""
    groups: list[list[Word]] = []
    for i, word in enumerate(words):
        if not groups or i in breaks:
            groups.append([])
        groups[-1].append(word)
    merged: list[list[Word]] = []
    for group in groups:
        if merged and len(group) < MIN_LINE_WORDS:
            merged[-1].extend(group)
        else:
            merged.append(group)
    if len(merged) > 1 and len(merged[0]) < MIN_LINE_WORDS:
        merged[1] = merged[0] + merged[1]
        merged.pop(0)
    return [Line(g[0].start, g[-1].end, joiner.join(w.text for w in g)) for g in merged]


def regions(items: list[Word] | list[Line], bridge: float = 0.3) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    for item in items:
        if spans and item.start - spans[-1][1] <= bridge:
            spans[-1] = (spans[-1][0], max(spans[-1][1], item.end))
        else:
            spans.append((item.start, item.end))
    return spans


# --- GTSinger ---------------------------------------------------------------------


@dataclass
class Song:
    title: str
    lang: str
    clips: list[Path]  # wav files in order


def gtsinger_songs(root: Path, lang: str) -> list[Song]:
    """One Control_Group folder per song: the one with the most phrase clips."""
    best: dict[str, list[Path]] = {}
    for folder in sorted(
        (root / {"zh": "Chinese", "en": "English"}[lang]).glob("*/*/*/Control_Group")
    ):
        clips = sorted(folder.glob("*.wav"))
        title = folder.parent.name
        if len(clips) > len(best.get(title, [])):
            best[title] = clips
    return [Song(t, lang, best[t]) for t in stable_order(list(best))]


def join_clips(song: Song, rng: np.random.Generator) -> tuple[np.ndarray, list[Word], set[int]]:
    pieces: list[np.ndarray] = []
    words: list[Word] = []
    breaks: set[int] = set()
    t = 0.0
    for k, wav in enumerate(song.clips):
        if k:
            gap = np.zeros(int(rng.uniform(*CLIP_GAP) * SR), dtype=np.float32)
            pieces.append(gap)
            t += len(gap) / SR
        clip = load_mono(wav)
        pause = LINE_PAUSE  # a clip always starts a line
        for item in json.loads(wav.with_suffix(".json").read_text(encoding="utf-8")):
            text = item["word"].strip()
            start, end = float(item["start_time"]), float(item["end_time"])
            if MARK.match(text) or not text:
                pause += end - start
                continue
            if pause >= LINE_PAUSE:
                breaks.add(len(words))
            pause = 0.0
            words.append(Word(round(t + start, 3), round(t + end, 3), text))
        pieces.append(clip)
        t += len(clip) / SR
    return np.concatenate(pieces), words, breaks


class Accompaniment:
    """AAM mixes handed out in a fixed order, joined when one is too short."""

    def __init__(self, files: list[Path]) -> None:
        self.files = files
        self.next = 0

    def take(self, seconds: float) -> tuple[np.ndarray, list[str]]:
        parts: list[np.ndarray] = []
        names: list[str] = []
        while sum(len(p) for p in parts) < seconds * SR:
            path = self.files[self.next % len(self.files)]
            self.next += 1
            parts.append(load_mono(path))
            names.append(path.name)
        return np.concatenate(parts)[: int(seconds * SR)], names


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) or 1e-9


def mix_over(
    vocals: np.ndarray, words: list[Word], backing: Accompaniment
) -> tuple[np.ndarray, list[str]]:
    total = INTRO + len(vocals) / SR + OUTRO
    acc, names = backing.take(total)
    voiced = np.concatenate([vocals[int(w.start * SR) : int(w.end * SR)] for w in words])
    acc = acc * (rms(voiced) / rms(acc)) * 10 ** (-VOCAL_DB / 20)
    out = acc.copy()
    start = int(INTRO * SR)
    out[start : start + len(vocals)] += vocals
    return out, names


def shifted(items: list, offset: float) -> list:
    return [type(i)(round(i.start + offset, 3), round(i.end + offset, 3), i.text) for i in items]


def gtsinger_entries(
    out: Path,
    songs: list[Song],
    split: str,
    set_name: str,
    prefix: str,
    backing: Accompaniment,
    rng: np.random.Generator,
) -> list[Entry]:
    entries: list[Entry] = []
    joiner = "" if songs and songs[0].lang == "zh" else " "
    for n, song in enumerate(songs, 1):
        vocals, words, breaks = join_clips(song, rng)
        lines = words_to_lines(words, breaks, joiner)
        sid = f"{prefix}-{n:02d}"
        dur = write(out, f"audio/{sid}-dry.wav", vocals)
        entries.append(
            Entry(
                f"{sid}-dry",
                set_name,
                split,
                song.lang,
                "dry",
                f"audio/{sid}-dry.wav",
                dur,
                lines,
                words,
                regions(words),
                source={"dataset": "gtsinger", "title": song.title},
            )
        )
        mixed, names = mix_over(vocals, words, backing)
        dur = write(out, f"audio/{sid}-mix.wav", mixed)
        entries.append(
            Entry(
                f"{sid}-mix",
                set_name,
                split,
                song.lang,
                "mix",
                f"audio/{sid}-mix.wav",
                dur,
                shifted(lines, INTRO),
                shifted(words, INTRO),
                regions(shifted(words, INTRO)),
                source={
                    "dataset": "gtsinger",
                    "title": song.title,
                    "accompaniment": ",".join(names),
                },
            )
        )
        print(f"  {sid}: {len(lines)} lines, {len(words)} words, {dur:.0f} s", flush=True)
    return entries


# --- JamendoLyrics ----------------------------------------------------------------


def jamendo_entries(out: Path, root: Path, jam_alt: Path) -> list[Entry]:
    meta = [
        json.loads(line)
        for line in (root / "subsets/en/metadata.jsonl").read_text().splitlines()
        if line
    ]
    entries: list[Entry] = []
    for n, song in enumerate(sorted(meta, key=lambda m: m["name"]), 1):
        name = song["name"]
        with open(root / "annotations/lines" / f"{name}.csv", newline="", encoding="utf-8") as f:
            lines = [
                Line(float(r["start_time"]), float(r["end_time"]), r["lyrics_line"].strip())
                for r in csv.DictReader(f)
                if r["lyrics_line"].strip()
            ]
        with open(root / "annotations/words" / f"{name}.csv", newline="", encoding="utf-8") as f:
            times = [(float(r["word_start"]), float(r["word_end"])) for r in csv.DictReader(f)]
        tokens = [w for line in lines for w in line.text.split()]
        words = (
            [Word(s, e, t) for (s, e), t in zip(times, tokens, strict=False)]
            if len(times) == len(tokens)
            else []
        )
        ref = jam_alt / "lyrics" / f"{name}.txt"
        audio = root / "subsets/en/mp3" / f"{name}.mp3"
        info = sf.info(audio)
        entries.append(
            Entry(
                f"en-test-{n:02d}",
                "en-test",
                "test",
                "en",
                "real",
                str(Path("..") / audio.relative_to(out.parent)),
                info.duration,
                lines,
                words,
                regions(words or lines),
                reference_text=ref.read_text(encoding="utf-8") if ref.exists() else "",
                source={
                    "dataset": "jamendolyrics",
                    "title": name,
                    "license": song.get("license_type", ""),
                },
            )
        )
    return entries


# --- MIR-1K -----------------------------------------------------------------------


def mir1k_entries(out: Path, root: Path, annotations: Path) -> list[Entry]:
    items = [x for x in json.loads(annotations.read_text(encoding="utf-8")) if "on_offset" in x]
    entries: list[Entry] = []
    for n, item in enumerate(sorted(items, key=lambda x: x["song_id"]), 1):
        chars = [c for c in item["lyric"] if not c.isspace()]
        if len(chars) != len(item["on_offset"]):
            print(
                f"  skip {item['song_id']}: "
                f"{len(chars)} characters, {len(item['on_offset'])} timings"
            )
            continue
        words = [
            Word(float(s), float(e), c) for c, (s, e) in zip(chars, item["on_offset"], strict=True)
        ]
        breaks = {
            i for i in range(1, len(words)) if words[i].start - words[i - 1].end >= LINE_PAUSE
        }
        sid = f"zh-ref-{n:02d}"
        dur = write(out, f"audio/{sid}.wav", load_mono(root / "UndividedWavfile" / item["song_id"]))
        entries.append(
            Entry(
                sid,
                "zh-ref",
                "ref",
                "zh",
                "real",
                f"audio/{sid}.wav",
                dur,
                words_to_lines(words, breaks, ""),
                words,
                regions(words),
                source={"dataset": "mir-1k", "title": item["song_id"]},
            )
        )
    return entries


# --- instrumentals ----------------------------------------------------------------


def instrumental_entries(
    out: Path, aam: list[Path], gtzan: Path, counts: dict[str, int], split: str
) -> list[Entry]:
    def genre(name: str) -> list[Path]:  # skips macOS "._" metadata files
        return [Path(p) for p in stable_order([str(p) for p in (gtzan / name).glob("[!.]*.wav")])]

    pools = {"aam": aam, "classical": genre("classical"), "jazz": genre("jazz")}
    entries: list[Entry] = []
    for kind, count in counts.items():
        offset = (
            0 if split == "test" else len(pools[kind]) // 2
        )  # disjoint picks for the two splits
        for path in pools[kind][offset : offset + count]:
            sid = f"instr-{split}-{len(entries) + 1:02d}"
            dur = write(out, f"audio/{sid}.wav", load_mono(path))
            entries.append(
                Entry(
                    sid,
                    "instr",
                    split,
                    "",
                    "instrumental",
                    f"audio/{sid}.wav",
                    dur,
                    source={
                        "dataset": "aam" if kind == "aam" else f"gtzan-{kind}",
                        "title": path.name,
                    },
                )
            )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    data: Path = args.data
    lyrics = data / "lyrics"
    out: Path = args.out or lyrics / "testset"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    aam = [Path(p) for p in stable_order([str(p) for p in (data / "aam").glob("*_mix.flac")])]
    backing = Accompaniment(
        aam[200:]
    )  # instrumentals take from the front, accompaniments from here

    zh = gtsinger_songs(lyrics / "gtsinger", "zh")
    en = gtsinger_songs(lyrics / "gtsinger", "en")
    entries: list[Entry] = []
    print("zh-test (GTSinger)")
    entries += gtsinger_entries(out, zh[:ZH_TEST], "test", "zh-test", "zh-test", backing, rng)
    print("dev (GTSinger zh / en)")
    entries += gtsinger_entries(
        out, zh[ZH_TEST : ZH_TEST + ZH_DEV], "dev", "dev", "zh-dev", backing, rng
    )
    entries += gtsinger_entries(out, en[:EN_DEV], "dev", "dev", "en-dev", backing, rng)
    print("en-test (JamendoLyrics)")
    entries += jamendo_entries(out, lyrics / "jamendolyrics", lyrics / "jam-alt")
    print("zh-ref (MIR-1K)")
    entries += mir1k_entries(
        out, lyrics / "mir-1k/MIR-1K", lyrics / "lyricalignment/MIR1k_partial_align.json"
    )
    print("instrumentals")
    entries += instrumental_entries(out, aam, data / "gtzan/genres", INSTR_TEST, "test")
    entries += instrumental_entries(out, aam, data / "gtzan/genres", INSTR_DEV, "dev")

    manifest = {
        "protocol": {
            "seed": SEED,
            "line_pause": LINE_PAUSE,
            "min_line_words": MIN_LINE_WORDS,
            "clip_gap": CLIP_GAP,
            "vocal_db": VOCAL_DB,
            "intro": INTRO,
            "outro": OUTRO,
        },
        "entries": [asdict(e) for e in entries],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    counts: dict[str, int] = {}
    for e in entries:
        counts[f"{e.set}/{e.condition}"] = counts.get(f"{e.set}/{e.condition}", 0) + 1
    print(json.dumps(counts, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
