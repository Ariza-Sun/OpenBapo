"""Sort the owner's downloaded sheet images into one ordered page set per song.

Usage:
    python scripts/sheet_triage.py RAW_DIR --out DIR [--apply]

RAW_DIR holds one folder per song (pages as images or a PDF; duplicates,
second versions and stray files are tolerated) and possibly loose files
at the top. Without --apply only DIR/triage.md and DIR/triage.json are
written: one row per song saying which files were taken as its pages, in
what order, how sure the order is, the capo found in a file name, and
what was left out. --review FILE brings in what an agent saw on the
pages (the reading order with its evidence, the sheet's maker, whether
there is a numbered-notation line, the printed header), which settles
the guessed orders. With --apply the chosen pages are also written as
DIR/<name>/p01.png ... so `scripts/sheet_ocr.py read DIR/<name>` reads
the song. Nothing here reads a sheet; the table is for a person to check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--review",
        type=Path,
        action="append",
        default=[],
        help="JSON from an agent that looked at the pages (order, source, header); repeatable",
    )
    parser.add_argument("--apply", action="store_true", help="also write the page sets")
    parser.add_argument("--dpi", type=int, default=300, help="rasterisation of PDF sources")
    args = parser.parse_args(argv)

    from openbapo.sheet.triage import apply_review, materialize, plan_library, table, to_json

    plans = plan_library(args.raw)
    for path in args.review:
        review = json.loads(path.read_text(encoding="utf-8"))
        for problem in apply_review(plans, review):
            print(f"{path.name}: {problem}", file=sys.stderr)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "triage.md").write_text(table(plans), encoding="utf-8")
    (args.out / "triage.json").write_text(
        json.dumps(to_json(plans), ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    usable = [p for p in plans if p.pages]
    print(
        f"{len(plans)} songs, {len(usable)} with pages, "
        f"{sum(not p.order_evident for p in usable)} with guessed page order, "
        f"{sum(p.versions > 1 for p in usable)} with several versions -> {args.out / 'triage.md'}"
    )
    if not args.apply:
        return 0
    failures = 0
    for plan in usable:
        try:
            materialize(plan, args.out, args.dpi)
        except Exception as e:  # one bad file must not stop the batch
            print(f"{plan.name}: failed: {e}", file=sys.stderr)
            failures += 1
    print(f"wrote {len(usable) - failures} page sets under {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
