"""Render lead sheets (docs/examples/*.sheet format) to WAV plus a timed reference.

Usage:
    python scripts/sheet_synth.py SHEET.sheet [SHEET2.sheet ...] --out DIR [--no-bass] [--no-melody]

Writes DIR/<stem>.wav and DIR/<stem>.json for each sheet. The JSON holds the
sounding chords (shapes transposed by the capo), the sung key, the beat grid
and the melody, so `scripts/eval_chords.py --sheets DIR` can score the
analysis against the sheet. Transcribed song sheets and their renders stay
out of the repository; only the public-domain example is committed.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sheets", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-bass", action="store_true")
    parser.add_argument("--no-melody", action="store_true")
    args = parser.parse_args(argv)

    from opencapo.sheet import load_sheet, render, write_reference, write_wav

    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.sheets:
        sheet = load_sheet(path)
        rendered = render(sheet, bass=not args.no_bass, melody=not args.no_melody)
        write_wav(args.out / f"{path.stem}.wav", rendered.samples, rendered.sample_rate)
        write_reference(args.out / f"{path.stem}.json", rendered.reference)
        print(
            f"{path.stem}: {len(sheet.bars)} bars, {sheet.duration:.1f} s, "
            f"key {rendered.reference['key']}, capo {sheet.capo}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
