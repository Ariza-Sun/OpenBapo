"""Read a printed guitar sheet (PDF / image) into the text sheet format, in two halves.

Usage:
    python scripts/sheet_ocr.py prepare SHEET.pdf --out WORK_DIR
    python scripts/sheet_ocr.py assemble WORK_DIR --readings READINGS.json \
        --header HEADER.json [--bpm 76] --out SONG.sheet

`prepare` rasterizes the pages, finds systems and bars, and writes crops
plus WORK_DIR/worksheet.json and the two prompts. An agent with vision
reads the page-1 header crop into HEADER.json (title, key, capo, time,
bpm) and each bar's stacked crop into READINGS.json, a list of
{"index", "chords", "melody"}, following those prompts.
`assemble` checks the readings (bars must fill the metre, chords must
parse, ties must continue, repeated passages should match), writes the
.sheet file and a review list (WORK_DIR/review.md) of bars to look at again.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("source", type=Path)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--dpi", type=int, default=300)
    assemble = commands.add_parser("assemble")
    assemble.add_argument("work_dir", type=Path)
    assemble.add_argument("--readings", type=Path, required=True)
    assemble.add_argument("--out", type=Path, required=True)
    assemble.add_argument("--header", type=Path, help="JSON from the header reading")
    assemble.add_argument("--title", default="")
    assemble.add_argument("--key", help="the sheet's 1=, e.g. E (overrides --header)")
    assemble.add_argument("--capo", type=int)
    assemble.add_argument("--bpm", type=float)
    assemble.add_argument("--time")
    assemble.add_argument("--mode", default="major")
    args = parser.parse_args(argv)

    if args.command == "prepare":
        from opencapo.sheet.layout import worksheet
        from opencapo.sheet.omr import HEADER_PROMPT, READING_PROMPT

        sheet = worksheet(args.source, args.out, args.dpi)
        (args.out / "reading_prompt.txt").write_text(READING_PROMPT, encoding="utf-8")
        (args.out / "header_prompt.txt").write_text(HEADER_PROMPT, encoding="utf-8")
        bars = sum(len(s["bars"]) for p in sheet["pages"] for s in p["systems"])  # type: ignore[index, union-attr]
        print(f"{len(sheet['pages'])} pages, {bars} bars; crops and worksheet.json in {args.out}")  # type: ignore[arg-type]
        return 0

    from opencapo.sheet.omr import BarReading, write_review
    from opencapo.sheet.omr import assemble as assemble_sheet

    worksheet_data = json.loads((args.work_dir / "worksheet.json").read_text(encoding="utf-8"))
    crops = {
        b["index"]: b["stacked"]
        for p in worksheet_data["pages"]
        for s in p["systems"]
        for b in s["bars"]
    }
    raw = json.loads(args.readings.read_text(encoding="utf-8"))
    readings = [
        BarReading(
            index=int(r["index"]),
            chords=str(r.get("chords", "")).strip(),
            melody=str(r.get("melody", "")).strip(),
            notes=str(r.get("notes", "")).strip(),
            crop=crops.get(int(r["index"]), ""),
        )
        for r in sorted(raw, key=lambda r: int(r["index"]))
    ]
    missing = sorted(set(crops) - {r.index for r in readings})
    if missing:
        print(f"no reading for bars {missing}", file=sys.stderr)
        return 1
    read = json.loads(args.header.read_text(encoding="utf-8")) if args.header else {}
    key = args.key or read.get("key")
    if not key:
        print("no key: pass --key or --header", file=sys.stderr)
        return 1
    bpm = args.bpm if args.bpm is not None else read.get("bpm") or 90.0
    time_sig = args.time or read.get("time") or "4/4"
    header = {
        "title": args.title or str(read.get("title", "")),
        "key": str(key),
        "mode": args.mode,
        "capo": str(args.capo if args.capo is not None else read.get("capo") or 0),
        "bpm": f"{float(bpm):g}",
        "time": str(time_sig),
    }
    result = assemble_sheet(header, readings, int(header["time"].split("/")[0]))
    args.out.write_text(result.text, encoding="utf-8")
    write_review(args.work_dir / "review.md", result.problems, readings)
    print(f"{len(readings)} bars -> {args.out}; {len(result.problems)} bars to review")
    for p in result.problems:
        print(f"  bar {p.index}: {p.problem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
