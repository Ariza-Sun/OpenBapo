"""Deterministic layout analysis of a printed guitar sheet (PDF or image).

Finds, on each page, the systems: a chord-diagram strip, a six-line tab
staff, then the numbered-notation line with its lyrics. Each system is
cut out as one full-width row image for the reading step, which is done
by an agent with vision and splits the row into bars itself. Nothing
here reads a digit or looks for a bar line: bar lines are the same
height as a "1" in some layouts and as a "0" in others, and brackets or
repeat signs fool every rule tried, while a vision model reads a whole
row reliably (decision 15 in docs/plans/step-2.md).

Measurements are relative to the page size, so the same thresholds work
at any rasterisation resolution and across the layouts of the owner's
song book (tab staves are the longest horizontal lines on the page; the
notation line is the first substantial band of ink below a staff).
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from PIL import Image

DPI = 300
INK_THRESHOLD = 130  # ink is darker than this; thin bar lines render ~115, the watermark >150
STAFF_THRESHOLD = (
    170  # staff lines are thin and print grey; their regular spacing keeps watermarks out
)
PAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Band:
    top: int
    bottom: int
    peak: int  # most ink pixels in one row

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass
class SystemBox:
    staff: Band
    notation: Band
    diagram_top: int
    index: int = -1  # row number, running on across pages
    top: int = 0  # of the row crop
    bottom: int = 0
    crop: str = ""  # path of the row image: diagrams, staff, notation and lyrics


def rasterize(path: Path, dpi: int = DPI) -> list[Image.Image]:
    """Pages of a PDF, the images of a folder (in name order), or one image, as greyscale."""
    from PIL import Image

    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in PAGE_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"{path}: no page images")
        return [Image.open(p).convert("L") for p in files]
    if path.suffix.lower() == ".pdf":
        import pypdfium2

        pdf = pypdfium2.PdfDocument(str(path))
        pages = []
        for i in range(len(pdf)):
            page = pdf[i]
            pages.append(page.render(scale=dpi / 72).to_pil().convert("L"))
        return pages
    return [Image.open(path).convert("L")]


def ink_mask(image: Image.Image, threshold: int = INK_THRESHOLD) -> NDArray[np.bool_]:
    import numpy as np

    return np.asarray(np.array(image) < threshold)


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
    """Tab staves: four or more page-wide, evenly spaced ink rows a few percent of the height apart.

    The mask should be taken at STAFF_THRESHOLD: staff lines print grey and
    digits sit on them, so at the ink threshold a busy staff loses lines.
    Even spacing (every gap within 35% of the median) keeps watermark text
    and lyrics out.
    """
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
            lines: list[list[int]] = []
            for row in group:
                if lines and row - lines[-1][-1] <= 2:
                    lines[-1].append(row)
                else:
                    lines.append([row])
            centres = [sum(line) / len(line) for line in lines]
            gaps = np.diff(centres)
            span = group[-1] - group[0]
            even = len(gaps) >= 3 and bool(
                np.all(np.abs(gaps - np.median(gaps)) <= 0.35 * np.median(gaps))
            )
            if even and 0.02 * height <= span <= 0.09 * height:
                staves.append(Band(group[0], group[-1] + 1, int(width)))
        group = [y] if y is not None else []
    return staves


def widest_gap(mask: NDArray[np.bool_], top: int, bottom: int, min_ink: int) -> int | None:
    """The middle row of the longest run of blank rows between top and bottom, if any."""
    if bottom <= top:
        return None
    blank = mask[top:bottom].sum(axis=1) <= min_ink
    best: tuple[int, int] | None = None
    start = None
    for i, value in enumerate([*blank.tolist(), False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return None if best is None else top + (best[0] + best[1]) // 2


def find_systems(
    mask: NDArray[np.bool_], staff_mask: NDArray[np.bool_] | None = None
) -> list[SystemBox]:
    """Each tab staff with the notation line below it and the chord-diagram strip above.

    `staff_mask` is the page at STAFF_THRESHOLD (defaults to `mask`).
    """
    height, width = mask.shape
    # A row counts as inked when more than a few thin vertical strokes cross
    # it: some layouts run a bracket or the bar lines from the staff down to
    # the notation, which would otherwise fuse staff, digits and lyrics into
    # one band and leave the lyrics as the first band "below the staff".
    min_ink = max(3, int(0.015 * width))
    rows = bands(mask, axis=0, min_ink=min_ink, min_size=4)
    staves = find_staves(mask if staff_mask is None else staff_mask)
    systems = []
    for i, staff in enumerate(staves):
        limit = staves[i + 1].top if i + 1 < len(staves) else height
        # Strum stems hang right under the staff in some layouts and form a
        # band of their own; the digits sit further down.
        below = [b for b in rows if b.top > staff.bottom + 0.006 * height and b.bottom <= limit]
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
        systems.append(SystemBox(staff=staff, notation=notation, diagram_top=diagram_top))
    return systems


def analyze_page(
    image: Image.Image, page_index: int, out_dir: Path, stem: str, start: int = 0
) -> list[SystemBox]:
    """Find the systems of one page and write each as a row image into out_dir.

    A row runs from the top of its chord-name strip to just above the next
    system's strip (or a lyrics' height below the notation for the last
    one), across the full page width. Rows are numbered from `start` so
    indices run on across pages.
    """
    mask = ink_mask(image)
    height, width = mask.shape
    systems = find_systems(mask, ink_mask(image, STAFF_THRESHOLD))
    min_ink = max(3, int(0.015 * width))
    # Rows meet at the widest blank gap between one system's lyrics and the
    # next one's chord names; rows cut by a fixed margin lost octave dots and
    # underlines at the bottom, or chord names at the top, in tight layouts.
    splits: list[int] = []
    for a, b in itertools.pairwise(systems):
        gap = widest_gap(mask, a.notation.bottom, b.staff.top, min_ink)
        splits.append(gap if gap is not None else b.diagram_top - int(0.004 * height))
    for i, system in enumerate(systems):
        if i == 0:
            gap = widest_gap(
                mask, max(0, system.staff.top - int(0.12 * height)), system.diagram_top, min_ink
            )
            top = gap if gap is not None else system.diagram_top - int(0.01 * height)
        else:
            top = splits[i - 1]
        if i + 1 < len(systems):
            bottom = splits[i]
        else:
            bottom = min(height, system.notation.bottom + int(0.06 * height))
        system.index = start + i
        system.top = max(0, min(top, system.diagram_top))
        system.bottom = max(bottom, system.notation.bottom + int(0.02 * height))
        system.crop = str(out_dir / f"{stem}-p{page_index + 1}-r{system.index:03d}.png")
        image.crop((0, system.top, width, system.bottom)).save(system.crop)
    return systems


def worksheet(source: Path, out_dir: Path, dpi: int = DPI) -> dict[str, object]:
    """Rasterize, analyse every page and write <out_dir>/worksheet.json; returns its content."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = rasterize(source, dpi)
    stem = source.stem
    pages_out: list[dict[str, object]] = []
    sheet: dict[str, object] = {"source": str(source), "dpi": dpi, "pages": pages_out}
    next_index = 0
    for p, image in enumerate(pages):
        # Everything above the first staff: title, 1=, capo, tempo if printed.
        first_staff = find_staves(ink_mask(image, STAFF_THRESHOLD))
        header_bottom = first_staff[0].top if first_staff else int(0.2 * image.height)
        header = image.crop((0, 0, image.width, header_bottom))
        header_path = out_dir / f"{stem}-p{p + 1}-header.png"
        header.save(header_path)
        systems = analyze_page(image, p, out_dir, stem, start=next_index)
        next_index += len(systems)
        pages_out.append(
            {
                "index": p + 1,
                "width": image.width,
                "height": image.height,
                "header_crop": str(header_path),
                "systems": [
                    {
                        "index": s.index,
                        "top": s.top,
                        "bottom": s.bottom,
                        "crop": s.crop,
                        "staff": asdict(s.staff),
                        "notation": asdict(s.notation),
                    }
                    for s in systems
                ],
            }
        )
    (out_dir / "worksheet.json").write_text(
        json.dumps(sheet, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return sheet
