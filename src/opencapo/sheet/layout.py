"""Deterministic layout analysis of a printed guitar sheet (PDF or image).

Finds, on each page, the systems (a chord-diagram strip, a six-line tab
staff, then the numbered-notation line with its lyrics), splits every
notation line into bars at the bar lines, and locates the chord diagrams
above each bar. Nothing here reads a digit: it produces crops and a
worksheet (JSON) for the reading step, which is done by an agent with
vision, and the geometry the assembler needs to put chords on beats.

Measurements are relative to the page width, so the same thresholds work
at any rasterisation resolution. Tuned on the jitahome / 小LV吉他 layout;
other layouts may need their own thresholds.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from PIL import Image

DPI = 300
INK_THRESHOLD = 110  # grey level below which a pixel is ink (the watermark is lighter)


@dataclass(frozen=True)
class Band:
    top: int
    bottom: int
    peak: int  # most ink pixels in one row

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass
class BarBox:
    index: int  # 0-based within the page
    left: int
    right: int
    top: int  # of the notation crop (with margins)
    bottom: int
    chords: list[dict[str, object]] = field(default_factory=list)  # {"x": fraction, "box": [...]}
    crop: str = ""  # path of the bar image
    stacked: str = ""  # chord strip over the notation, same x range: what the reader sees
    chord_crops: list[str] = field(default_factory=list)


@dataclass
class SystemBox:
    staff: Band
    notation: Band
    diagram_top: int
    bars: list[BarBox]


def rasterize(path: Path, dpi: int = DPI) -> list[Image.Image]:
    """Pages of a PDF, or the single image, as greyscale PIL images."""
    from PIL import Image

    if path.suffix.lower() == ".pdf":
        import pypdfium2

        pdf = pypdfium2.PdfDocument(str(path))
        pages = []
        for i in range(len(pdf)):
            page = pdf[i]
            pages.append(page.render(scale=dpi / 72).to_pil().convert("L"))
        return pages
    return [Image.open(path).convert("L")]


def ink_mask(image: Image.Image) -> NDArray[np.bool_]:
    import numpy as np

    return np.asarray(np.array(image) < INK_THRESHOLD)


def bands(mask: NDArray[np.bool_], axis: int, min_ink: int, min_size: int) -> list[Band]:
    """Runs along `axis` whose projected ink exceeds min_ink, at least min_size long."""
    profile = mask.sum(axis=1 - axis)
    out: list[Band] = []
    start = None
    for i, value in enumerate(profile):
        if value > min_ink and start is None:
            start = i
        elif value <= min_ink and start is not None:
            if i - start >= min_size:
                out.append(Band(start, i, int(profile[start:i].max())))
            start = None
    if start is not None and len(profile) - start >= min_size:
        out.append(Band(start, len(profile), int(profile[start:].max())))
    return out


def find_staves(mask: NDArray[np.bool_]) -> list[Band]:
    """Tab staves: groups of at least four page-wide ink rows within a few percent of the height."""
    import numpy as np

    height, width = mask.shape
    long_rows = np.flatnonzero(mask.sum(axis=1) >= 0.6 * width)
    staves: list[Band] = []
    group: list[int] = []
    for y in [*long_rows.tolist(), None]:
        if y is not None and (not group or y - group[-1] <= 0.02 * height):
            group.append(y)
            continue
        if group:
            distinct = 1 + int(np.sum(np.diff(group) > 2))
            span = group[-1] - group[0]
            if distinct >= 4 and 0.02 * height <= span <= 0.09 * height:
                staves.append(Band(group[0], group[-1] + 1, int(width)))
        group = [y] if y is not None else []
    return staves


def find_systems(mask: NDArray[np.bool_]) -> list[SystemBox]:
    """Each tab staff with the notation line below it and the chord-diagram strip above."""
    height = mask.shape[0]
    rows = bands(mask, axis=0, min_ink=3, min_size=4)
    staves = find_staves(mask)
    systems = []
    for i, staff in enumerate(staves):
        limit = staves[i + 1].top if i + 1 < len(staves) else height
        below = [b for b in rows if b.top > staff.bottom and b.bottom <= limit]
        notation = next((b for b in below if b.height >= 0.008 * height), None)
        if notation is None:
            continue
        # The diagram strip sits right above the staff: a band of fretboard
        # grids, with a thinner band of chord names just above it.
        above = [b for b in rows if b.bottom <= staff.top and staff.top - b.bottom < 0.03 * height]
        diagram_top = staff.top - int(0.03 * height)
        if above:
            grids = above[-1]
            diagram_top = grids.top - int(0.004 * height)
            names = [
                b for b in rows if b.bottom <= grids.top and grids.top - b.bottom < 0.012 * height
            ]
            if names and names[-1].height < 0.015 * height:
                diagram_top = names[-1].top - int(0.003 * height)
        systems.append(SystemBox(staff=staff, notation=notation, diagram_top=diagram_top, bars=[]))
    return systems


def longest_runs(strip: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Per column, the longest unbroken vertical run of ink."""
    import numpy as np

    best = np.zeros(strip.shape[1], dtype=np.int64)
    current = np.zeros(strip.shape[1], dtype=np.int64)
    for row in strip:
        current = np.where(row, current + 1, 0)
        best = np.maximum(best, current)
    return best


def bar_lines(mask: NDArray[np.bool_], notation: Band, width: int) -> list[int]:
    """x positions of the bar lines in a notation line.

    Bar lines are the tallest unbroken vertical strokes in the line (digits
    are shorter, tie arcs are thin and curved), so columns whose longest run
    comes within 80% of the tallest count. The band may include the lyrics
    below the digits, which is why the band height itself is not used.
    """
    import numpy as np

    runs = longest_runs(mask[notation.top : notation.bottom])
    tallest = float(np.percentile(runs[runs > 0], 99.5)) if np.any(runs > 0) else 0.0
    xs = np.flatnonzero(runs >= 0.8 * tallest) if tallest > 0 else np.zeros(0, dtype=np.int64)
    lines: list[int] = []
    for x in xs:
        if lines and x - lines[-1] <= max(3, int(0.004 * width)):
            continue
        lines.append(int(x))
    return lines


def diagram_boxes(
    mask: NDArray[np.bool_], top: int, bottom: int, width: int
) -> list[tuple[int, int]]:
    """x ranges of chord diagrams (dense ink blobs) in the strip above a staff."""
    strip = mask[top:bottom]
    columns = bands(strip, axis=1, min_ink=2, min_size=2)
    merged: list[list[int]] = []
    for c in columns:
        if merged and c.top - merged[-1][1] <= 0.006 * width:
            merged[-1][1] = c.bottom
        else:
            merged.append([c.top, c.bottom])
    return [(left, right) for left, right in merged if right - left >= 0.012 * width]


def analyze_page(image: Image.Image, page_index: int, out_dir: Path, stem: str) -> list[SystemBox]:
    """Split one page into systems and bars, writing bar and chord-name crops into out_dir."""
    from PIL import Image

    mask = ink_mask(image)
    height, width = mask.shape
    systems = find_systems(mask)
    margin_y = int(0.012 * height)
    margin_x = int(0.004 * width)
    counter = 0
    for system in systems:
        lines = bar_lines(mask, system.notation, width)
        if len(lines) < 2:
            continue
        diagrams = diagram_boxes(mask, system.diagram_top, system.staff.top, width)
        top = max(0, system.notation.top - margin_y)
        bottom = min(height, system.notation.bottom + margin_y)
        for left, right in itertools.pairwise(lines):
            if right - left < 0.03 * width:
                continue
            bar = BarBox(index=counter, left=left, right=right, top=top, bottom=bottom)
            box_x = (max(0, left - margin_x), min(width, right + margin_x))
            crop = image.crop((box_x[0], top, box_x[1], bottom))
            bar.crop = str(out_dir / f"{stem}-p{page_index + 1}-b{counter:03d}.png")
            crop.save(bar.crop)
            strip = image.crop((box_x[0], system.diagram_top, box_x[1], system.staff.top))
            stacked = Image.new("L", (crop.width, strip.height + crop.height + 8), 255)
            stacked.paste(strip, (0, 0))
            stacked.paste(crop, (0, strip.height + 8))
            bar.stacked = str(out_dir / f"{stem}-p{page_index + 1}-b{counter:03d}-stacked.png")
            stacked.save(bar.stacked)
            for d_left, d_right in diagrams:
                centre = (d_left + d_right) / 2
                if left <= centre < right:
                    box = [
                        d_left - margin_x,
                        system.diagram_top,
                        d_right + margin_x,
                        system.staff.top,
                    ]
                    bar.chords.append({"x": round((centre - left) / (right - left), 3), "box": box})
                    name_crop = image.crop((max(0, box[0]), box[1], min(width, box[2]), box[3]))
                    path = (
                        out_dir
                        / f"{stem}-p{page_index + 1}-b{counter:03d}-c{len(bar.chord_crops)}.png"
                    )
                    name_crop.save(path)
                    bar.chord_crops.append(str(path))
            system.bars.append(bar)
            counter += 1
    return systems


def worksheet(source: Path, out_dir: Path, dpi: int = DPI) -> dict[str, object]:
    """Rasterize, analyse every page and write <out_dir>/worksheet.json; returns its content."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = rasterize(source, dpi)
    stem = source.stem
    pages_out: list[dict[str, object]] = []
    sheet: dict[str, object] = {"source": str(source), "dpi": dpi, "pages": pages_out}
    for p, image in enumerate(pages):
        header = image.crop((0, 0, image.width, int(0.1 * image.height)))
        header_path = out_dir / f"{stem}-p{p + 1}-header.png"
        header.save(header_path)
        systems = analyze_page(image, p, out_dir, stem)
        pages_out.append(
            {
                "index": p + 1,
                "width": image.width,
                "height": image.height,
                "header_crop": str(header_path),
                "systems": [
                    {
                        "staff": asdict(s.staff),
                        "notation": asdict(s.notation),
                        "bars": [asdict(b) for b in s.bars],
                    }
                    for s in systems
                ],
            }
        )
    (out_dir / "worksheet.json").write_text(
        json.dumps(sheet, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return sheet
