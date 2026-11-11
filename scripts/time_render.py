"""Time arranging and rendering a song (step 4: a 4-minute song in <= 5 s).

Usage:
    python scripts/time_render.py ANALYSIS.json [--repeat 5] [--out docs/benchmarks/step-4-timing]

Reads an analysis JSON (from `openbapo analyze`), then times arrange ->
SVG -> PNG (first page) -> PDF (all pages) -> MusicXML, reporting the
median of the warm runs. Writes OUT.md and OUT.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import statistics
import time
from pathlib import Path

ACCEPTANCE_SECONDS = 5.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--out", default="docs/benchmarks/step-4-timing")
    args = parser.parse_args(argv)

    from openbapo.arrange import arrange
    from openbapo.model import Analysis
    from openbapo.render import render_musicxml, render_svg, svg_pages_to_pdf, svg_to_png

    analysis = Analysis.from_json(args.analysis.read_text(encoding="utf-8"))
    runs: list[dict[str, float]] = []
    for _ in range(args.repeat + 1):  # the first run warms imports and font caches
        t: dict[str, float] = {}
        start = time.perf_counter()
        sheet = arrange(analysis)
        t["arrange"] = time.perf_counter() - start
        start = time.perf_counter()
        pages = render_svg(sheet)
        t["svg"] = time.perf_counter() - start
        start = time.perf_counter()
        svg_to_png(pages[0], dpi=150)
        t["png"] = time.perf_counter() - start
        start = time.perf_counter()
        svg_pages_to_pdf(pages)
        t["pdf"] = time.perf_counter() - start
        start = time.perf_counter()
        render_musicxml(sheet)
        t["musicxml"] = time.perf_counter() - start
        t["total"] = sum(t.values())
        runs.append(t)
    warm = runs[1:]
    median = {k: statistics.median(r[k] for r in warm) for k in runs[0]}
    info = {
        "date": dt.date.today().isoformat(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "analysis": str(args.analysis),
        "duration_seconds": analysis.duration,
        "bars": len(sheet.bars),
        "pages": len(pages),
        "repeat": args.repeat,
        "cold": runs[0],
        "warm_median": median,
        "acceptance_seconds": ACCEPTANCE_SECONDS,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    verdict = "通过" if median["total"] <= ACCEPTANCE_SECONDS else "未通过"
    lines = [
        "# 第 4 步：编配 + 渲染耗时",
        "",
        f"由 `scripts/time_render.py` 生成于 {info['date']}；"
        f"Python {info['python']}，{info['machine']}，CPU 单线程。",
        f"歌曲时长 {analysis.duration:.0f} 秒，{info['bars']} 小节，{info['pages']} 页；"
        f"热启动 {args.repeat} 次取中位数。",
        "",
        "| 环节 | 冷启动 | 热启动中位数 |",
        "|---|---|---|",
    ]
    for k in runs[0]:
        lines.append(f"| {k} | {runs[0][k]:.2f} s | {median[k]:.2f} s |")
    lines += [
        "",
        f"验收：4 分钟歌曲编配 + 渲染 ≤ {ACCEPTANCE_SECONDS:.0f} 秒：**{verdict}**"
        f"（{median['total']:.2f} s）。",
        "",
    ]
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
