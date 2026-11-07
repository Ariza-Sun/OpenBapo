"""Read a printed guitar sheet (PDF / image) into the text sheet format.

Usage:
    python scripts/sheet_ocr.py read SHEET.pdf [SHEET2.png ...] --out DIR [--bpm 76] [--render]
    python scripts/sheet_ocr.py prepare SHEET.pdf --out WORK_DIR
    python scripts/sheet_ocr.py assemble WORK_DIR --readings READINGS.json \
        --header HEADER.json [--bpm 76] --out SONG.sheet
    python scripts/sheet_ocr.py manifest DIR

`read` does everything in one go: layout, reading every bar with the
Claude Code command-line tool in headless mode (`claude -p`, under your own
login), the checks, a second larger reading of doubtful bars, assembly,
and optionally the rendered sample. Each source gets DIR/<stem>/ with the
.sheet file, readings.json and review.md; with `--render`, DIR/manifest.json
is rebuilt afterwards.

`manifest` rebuilds DIR/manifest.json from the files in DIR: every
complete song (.sheet + .wav + .json with one stem) with its header, its
origin (read from print or typed by hand) and how many bars its review.md
lists; songs missing a file are listed as incomplete. The evaluation
(`scripts/eval_chords.py --sheets DIR`) reads the sample set through this
file only.

`prepare` rasterizes the pages, finds the systems, and writes one row
image per system plus WORK_DIR/worksheet.json and the two prompts. An
agent with vision reads the page-1 header crop into HEADER.json (title,
key, capo, time, bpm) and each row image into READINGS.json, a list of
bars in order, {"row", "chords", "melody"}, following those prompts.
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
    read = commands.add_parser("read")
    read.add_argument("sources", type=Path, nargs="+")
    read.add_argument("--out", type=Path, required=True)
    read.add_argument("--bpm", type=float, help="tempo when the sheet does not print one")
    read.add_argument("--model", default="claude-opus-5-5")
    read.add_argument("--batch", type=int, default=10, help="bars per model call")
    read.add_argument("--parallel", type=int, default=3)
    read.add_argument("--render", action="store_true", help="also write the WAV and reference")
    read.add_argument(
        "--skip-done", action="store_true", help="skip sources whose .sheet already exists in --out"
    )
    prepare = commands.add_parser("prepare")
    prepare.add_argument("source", type=Path)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--dpi", type=int, default=300)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("root", type=Path)
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

    if args.command == "manifest":
        from opencapo.sheet import write_manifest

        manifest = write_manifest(args.root)
        songs, incomplete = manifest["songs"], manifest["incomplete"]
        assert isinstance(songs, list) and isinstance(incomplete, list)
        to_review = sum(int(s["to_review"]) for s in songs)
        print(f"{args.root / 'manifest.json'}: {len(songs)} songs, {to_review} bars to review")
        for item in incomplete:
            print(f"incomplete: {item['name']}: {item['problem']}", file=sys.stderr)
        return 0

    if args.command == "read":
        from opencapo.sheet import (
            SheetError,
            load_sheet,
            render,
            write_manifest,
            write_reference,
            write_wav,
        )
        from opencapo.sheet.reader import claude_runner, read_sheet

        runner = claude_runner(args.model)
        failures = 0
        for source in args.sources:
            work = args.out / source.stem
            if args.skip_done and (work / f"{source.stem}.sheet").exists():
                continue
            try:
                result = read_sheet(
                    source, work, runner=runner, bpm=args.bpm,
                    batch_size=args.batch, parallel=args.parallel,
                )  # fmt: skip
            except Exception as e:  # one bad sheet must not stop the batch
                print(f"{source}: failed: {e}", file=sys.stderr)
                failures += 1
                continue
            if args.render:
                try:
                    sheet = load_sheet(result.sheet_path)
                except SheetError as e:  # a bar left for review keeps the sheet unparsable
                    print(f"{source}: not rendered, fix review.md first: {e}", file=sys.stderr)
                    failures += 1
                    continue
                rendered = render(sheet)
                write_wav(work / f"{source.stem}.wav", rendered.samples, rendered.sample_rate)
                write_reference(work / f"{source.stem}.json", rendered.reference)
        if args.render:
            manifest = write_manifest(args.out)
            print(f"manifest: {len(manifest['songs'])} songs")  # type: ignore[arg-type]
        return 1 if failures else 0

    if args.command == "prepare":
        from opencapo.sheet.layout import worksheet
        from opencapo.sheet.omr import HEADER_PROMPT, READING_PROMPT

        sheet = worksheet(args.source, args.out, args.dpi)
        (args.out / "reading_prompt.txt").write_text(READING_PROMPT, encoding="utf-8")
        (args.out / "header_prompt.txt").write_text(HEADER_PROMPT, encoding="utf-8")
        rows = sum(len(p["systems"]) for p in sheet["pages"])  # type: ignore[index, union-attr]
        print(
            f"{len(sheet['pages'])} pages, {rows} rows; row images and worksheet.json in {args.out}"
        )  # type: ignore[arg-type]
        return 0

    from opencapo.sheet.omr import BarReading, Check, reconcile_header, write_review
    from opencapo.sheet.omr import assemble as assemble_sheet

    worksheet_data = json.loads((args.work_dir / "worksheet.json").read_text(encoding="utf-8"))
    crops = {int(s["index"]): str(s["crop"]) for p in worksheet_data["pages"] for s in p["systems"]}
    raw = json.loads(args.readings.read_text(encoding="utf-8"))
    readings = [
        BarReading(
            index=i,
            chords=str(r.get("chords", "")).strip(),
            melody=str(r.get("melody", "")).strip(),
            notes=str(r.get("notes", "")).strip(),
            lyrics=str(r.get("lyrics", "")).strip(),
            row=int(r.get("row", -1)),
            crop=crops.get(int(r.get("row", -1)), ""),
        )
        for i, r in enumerate(raw)
    ]
    missing = sorted(set(crops) - {r.row for r in readings})
    if missing:
        print(f"no reading for rows {missing}", file=sys.stderr)
        return 1
    read = json.loads(args.header.read_text(encoding="utf-8")) if args.header else {}
    beats = int(str(args.time or read.get("time") or "4/4").split("/")[0])
    for problem in reconcile_header(read, readings, beats):
        print(f"header: {problem}", file=sys.stderr)
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
    # Bars the reading loop left open keep their mark when the sheet is re-assembled.
    unsettled = [
        Check(r.index, "三次读法不一致，见 readings.json 的 notes")
        for r in readings
        if r.notes.startswith("三次读法")
    ]
    write_review(args.work_dir / "review.md", result.problems + unsettled, readings)
    print(f"{len(readings)} bars -> {args.out}; {len(result.problems)} bars to review")
    for p in result.problems:
        print(f"  bar {p.index}: {p.problem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
