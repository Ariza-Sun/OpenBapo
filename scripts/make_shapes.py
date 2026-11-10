"""Build the chord-shape library src/openbapo/arrange/data/shapes.json.

Usage:
    python scripts/make_shapes.py [--check]

Open-position shapes are listed by hand below; every other (root, quality)
gets movable shapes from the E-form, A-form and D-form templates, moved
to the root's fret. Each shape's sounding pitch classes are checked
against the quality's chord tones before it is written, so a typo here
fails the build rather than reaching a sheet. --check only verifies that
the committed file matches what this script would write.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "openbapo" / "arrange" / "data" / "shapes.json"
TUNING = (40, 45, 50, 55, 59, 64)  # E A D G B E, MIDI
NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
QUALITIES = ("maj", "min", "7", "maj7", "min7", "maj6", "min6", "sus2", "sus4", "5",
             "dim", "aug", "dim7", "hdim7")  # fmt: skip
TONES = {
    "maj": {0, 4, 7}, "min": {0, 3, 7}, "7": {0, 4, 7, 10}, "maj7": {0, 4, 7, 11},
    "min7": {0, 3, 7, 10}, "maj6": {0, 4, 7, 9}, "min6": {0, 3, 7, 9}, "sus2": {0, 2, 7},
    "sus4": {0, 5, 7}, "5": {0, 7}, "dim": {0, 3, 6}, "aug": {0, 4, 8},
    "dim7": {0, 3, 6, 9}, "hdim7": {0, 3, 6, 10},
}  # fmt: skip
X = None

# (root, quality): (frets low string first, difficulty, barre-from-string or None)
OPEN: dict[tuple[int, str], tuple[tuple[int | None, ...], int, int | None]] = {
    (0, "maj"): ((X, 3, 2, 0, 1, 0), 1, None),
    (0, "maj7"): ((X, 3, 2, 0, 0, 0), 1, None),
    (0, "7"): ((X, 3, 2, 3, 1, 0), 1, None),
    (0, "sus4"): ((X, 3, 3, 0, 1, 1), 2, None),
    (0, "maj6"): ((X, 3, 2, 2, 1, 0), 1, None),
    (2, "maj"): ((X, X, 0, 2, 3, 2), 1, None),
    (2, "min"): ((X, X, 0, 2, 3, 1), 1, None),
    (2, "7"): ((X, X, 0, 2, 1, 2), 1, None),
    (2, "min7"): ((X, X, 0, 2, 1, 1), 1, None),
    (2, "maj7"): ((X, X, 0, 2, 2, 2), 1, None),
    (2, "sus2"): ((X, X, 0, 2, 3, 0), 1, None),
    (2, "sus4"): ((X, X, 0, 2, 3, 3), 1, None),
    (2, "maj6"): ((X, X, 0, 2, 0, 2), 1, None),
    (2, "min6"): ((X, X, 0, 2, 0, 1), 1, None),
    (4, "maj"): ((0, 2, 2, 1, 0, 0), 1, None),
    (4, "min"): ((0, 2, 2, 0, 0, 0), 1, None),
    (4, "7"): ((0, 2, 0, 1, 0, 0), 1, None),
    (4, "min7"): ((0, 2, 0, 0, 0, 0), 1, None),
    (4, "maj7"): ((0, 2, 1, 1, 0, 0), 1, None),
    (4, "sus4"): ((0, 2, 2, 2, 0, 0), 1, None),
    (4, "maj6"): ((0, 2, 2, 1, 2, 0), 1, None),
    (4, "min6"): ((0, 2, 2, 0, 2, 0), 1, None),
    (5, "maj"): ((X, X, 3, 2, 1, 1), 2, 4),
    (5, "maj7"): ((X, X, 3, 2, 1, 0), 1, None),
    (7, "maj"): ((3, 2, 0, 0, 0, 3), 1, None),
    (7, "7"): ((3, 2, 0, 0, 0, 1), 1, None),
    (7, "maj7"): ((3, 2, 0, 0, 0, 2), 1, None),
    (7, "maj6"): ((3, 2, 0, 0, 0, 0), 1, None),
    (7, "sus4"): ((3, 3, 0, 0, 1, 3), 2, None),
    (9, "maj"): ((X, 0, 2, 2, 2, 0), 1, None),
    (9, "min"): ((X, 0, 2, 2, 1, 0), 1, None),
    (9, "7"): ((X, 0, 2, 0, 2, 0), 1, None),
    (9, "min7"): ((X, 0, 2, 0, 1, 0), 1, None),
    (9, "maj7"): ((X, 0, 2, 1, 2, 0), 1, None),
    (9, "sus2"): ((X, 0, 2, 2, 0, 0), 1, None),
    (9, "sus4"): ((X, 0, 2, 2, 3, 0), 1, None),
    (9, "maj6"): ((X, 0, 2, 2, 2, 2), 2, None),
    (9, "min6"): ((X, 0, 2, 2, 1, 2), 1, None),
    (9, "dim"): ((X, 0, 1, 2, 1, X), 1, None),
    (11, "7"): ((X, 2, 1, 2, 0, 2), 2, None),
    (11, "min7"): ((X, 2, 0, 2, 0, 2), 2, None),
    (11, "dim"): ((X, 2, 3, 4, 3, X), 2, None),
    (11, "hdim7"): ((X, 2, 3, 2, 3, X), 2, None),
}

# Movable templates: offsets from the root fret; the root string is where
# the root sits (0 = low E, 1 = A, 2 = D). barre = string the barre starts on.
TEMPLATES: dict[str, tuple[int, int | None, dict[str, tuple[int | None, ...]]]] = {
    "E-form": (
        0,
        0,
        {
            "maj": (0, 2, 2, 1, 0, 0),
            "min": (0, 2, 2, 0, 0, 0),
            "7": (0, 2, 0, 1, 0, 0),
            "maj7": (0, 2, 1, 1, 0, 0),
            "min7": (0, 2, 0, 0, 0, 0),
            "maj6": (0, 2, 2, 1, 2, 0),
            "min6": (0, 2, 2, 0, 2, 0),
            "sus4": (0, 2, 2, 2, 0, 0),
            "5": (0, 2, 2, X, X, X),
            "aug": (0, X, 2, 1, 1, X),
            "dim7": (0, X, 2, 3, 2, 3),
            "hdim7": (0, 1, 2, 0, 3, 0),
        },
    ),
    "A-form": (
        1,
        1,
        {
            "maj": (X, 0, 2, 2, 2, 0),
            "min": (X, 0, 2, 2, 1, 0),
            "7": (X, 0, 2, 0, 2, 0),
            "maj7": (X, 0, 2, 1, 2, 0),
            "min7": (X, 0, 2, 0, 1, 0),
            "maj6": (X, 0, 2, 2, 2, 2),
            "min6": (X, 0, 2, 2, 1, 2),
            "sus2": (X, 0, 2, 2, 0, 0),
            "sus4": (X, 0, 2, 2, 3, 0),
            "5": (X, 0, 2, 2, X, X),
            "dim": (X, 0, 1, 2, 1, X),
            "aug": (X, 0, 3, 2, 2, X),
            "dim7": (X, 0, 1, -1, 1, X),
            "hdim7": (X, 0, 1, 0, 1, X),
        },
    ),
    "D-form": (2, None, {"sus2": (X, X, 0, 2, 3, 0)}),
}


def pitch_classes(frets: tuple[int | None, ...]) -> set[int]:
    return {
        (open_pitch + f) % 12 for open_pitch, f in zip(TUNING, frets, strict=True) if f is not None
    }


def check(root: int, quality: str, frets: tuple[int | None, ...], where: str) -> None:
    rel = {(p - root) % 12 for p in pitch_classes(frets)}
    wants = TONES[quality]
    third = {3, 4, 2, 5} if quality in ("sus2", "sus4") else {3, 4}
    ok = (
        rel <= wants
        and 0 in rel
        and (quality == "5" or rel & third)
        and len(rel) >= 3 - (quality == "5")
    )
    if not ok:
        raise SystemExit(
            f"{where}: {NAMES[root]}{quality} {frets} sounds {sorted(rel)}, wants {sorted(wants)}"
        )


def build() -> list[dict[str, object]]:
    shapes: list[dict[str, object]] = []
    seen: set[tuple[int, str, tuple[int | None, ...]]] = set()

    def add(root: int, quality: str, frets: tuple[int | None, ...], difficulty: int,
            barre: int | None, source: str) -> None:  # fmt: skip
        check(root, quality, frets, source)
        if (root, quality, frets) in seen:
            return
        seen.add((root, quality, frets))
        sounded = [f for f in frets if f is not None and f > 0]
        base = min(sounded) if sounded and max(sounded) > 4 else 1
        shapes.append({
            "root": root, "quality": quality, "frets": list(frets), "base_fret": base,
            "barre": barre, "difficulty": difficulty, "source": source,
        })  # fmt: skip

    for (root, quality), (frets, difficulty, barre) in OPEN.items():
        add(root, quality, frets, difficulty, barre, "open")
    for form, (root_string, barre, by_quality) in TEMPLATES.items():
        for root in range(12):
            r = (root - TUNING[root_string]) % 12
            for quality, offsets in by_quality.items():
                frets = tuple(None if o is None else o + r for o in offsets)
                if any(f is not None and f < 0 for f in frets):
                    frets = tuple(None if f is None else f + 12 for f in frets)
                position = min(f for f in frets if f is not None and f > 0) if any(frets) else 0
                sounded = sum(f is not None for f in frets)
                if position == 0:
                    difficulty = 1
                else:
                    difficulty = 2 if sounded <= 4 else 3
                    if position > 7:
                        difficulty += 1
                add(
                    root, quality, frets, min(difficulty, 4), None if position == 0 else barre, form
                )

    def order(item: dict[str, object]) -> tuple[int, int, int, int]:
        root, quality = item["root"], item["quality"]
        difficulty, base = item["difficulty"], item["base_fret"]
        assert isinstance(root, int) and isinstance(quality, str)
        assert isinstance(difficulty, int) and isinstance(base, int)
        return root, QUALITIES.index(quality), difficulty, base

    shapes.sort(key=order)
    return shapes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify the committed file is current")
    args = parser.parse_args(argv)
    shapes = build()
    covered = {(s["root"], s["quality"]) for s in shapes}
    missing = [(r, q) for r in range(12) for q in QUALITIES if (r, q) not in covered]
    if missing:
        raise SystemExit(f"no shape for {missing}")
    text = json.dumps(shapes, indent=1) + "\n"
    if args.check:
        if OUT.read_text(encoding="utf-8") != text:
            print(f"{OUT} is out of date; run scripts/make_shapes.py", file=sys.stderr)
            return 1
        print(f"{OUT}: {len(shapes)} shapes, current")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"{OUT}: {len(shapes)} shapes for {len(covered)} chords")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
