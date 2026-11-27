"""Re-render retained analysis without importing any inference packages."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

from openbapo.arrange import ArrangedSheet, StrumOption, arrange, strums
from openbapo.model import Analysis, Arrangement
from openbapo.render import (
    render_musicxml,
    render_strum_overview,
    render_svg,
    svg_pages_to_pdf,
    svg_to_png,
)
from openbapo.service.config import ServiceError
from openbapo.service.store import Job


@dataclass(frozen=True)
class SheetFiles:
    sheet: ArrangedSheet
    png: bytes
    pdf: bytes | None


def sheet_for(job: Job, options: Arrangement | None = None) -> ArrangedSheet:
    if job.status != "done" or job.analysis is None:
        raise ServiceError("pending")
    return arrange(
        Analysis.from_json(job.analysis), options or Arrangement.from_json(job.arrangement)
    )


def preview(job: Job, options: Arrangement | None = None) -> SheetFiles:
    sheet = sheet_for(job, options)
    pages = render_svg(sheet)
    return SheetFiles(
        sheet, svg_to_png(pages[0]), svg_pages_to_pdf(pages) if len(pages) > 1 else None
    )


@lru_cache(maxsize=64)
def _overview(options: tuple[StrumOption, ...], titles: tuple[str, ...]) -> bytes:
    patterns = {p.id: p for p in strums.library()}
    cards = [(title, patterns[o.id]) for o, title in zip(options, titles, strict=True)]
    highlight = next((k for k, o in enumerate(options) if o.recommended), None)
    return svg_to_png(render_strum_overview(cards, highlight), dpi=144)


def strum_overview(
    job: Job, title: Callable[[StrumOption], str]
) -> tuple[bytes, tuple[StrumOption, ...]]:
    """The numbered picture of this song's strum patterns, and the options in that order.

    Patterns have numbers, not names: `title` gives the
    short text after each number, e.g. "扫弦 · 推荐"."""
    options = sheet_for(job).strum_options
    return _overview(options, tuple(title(o) for o in options)), options


def export(job: Job, format: str) -> tuple[str, bytes]:
    sheet = sheet_for(job)
    if format == "musicxml":
        return "sheet.musicxml", render_musicxml(sheet).encode("utf-8")
    pages = render_svg(sheet)
    if format == "pdf":
        return "sheet.pdf", svg_pages_to_pdf(pages)
    if format != "png":
        raise ServiceError("option")
    if len(pages) == 1:
        return "sheet.png", svg_to_png(pages[0])
    out = BytesIO()
    with ZipFile(out, "w", ZIP_DEFLATED) as archive:
        for number, page in enumerate(pages, 1):
            archive.writestr(f"page-{number}.png", svg_to_png(page))
    return "sheet-png.zip", out.getvalue()


def changed(job: Job, action: str, value: str) -> Arrangement:
    sheet = sheet_for(job)
    current = Arrangement.from_json(job.arrangement)
    values = current.model_dump()
    try:
        if action == "transpose":
            # Resolve the default once so transposition preserves the user's chord shapes.
            values["fingering_key"] = next(o.tonic for o in sheet.fingering_options if o.selected)
            values["transpose"] = int(value)
        elif action == "fingering":
            key = int(value)
            if key not in {o.tonic for o in sheet.fingering_options}:
                raise ValueError("unavailable key")
            values["fingering_key"] = key
        elif action == "strum":
            if value not in {"auto", "none", *(o.id for o in sheet.strum_options)}:
                raise ValueError("unavailable strum")
            values["strum_pattern"] = value
        elif action == "simplify":
            values["simplify_chords"] = not current.simplify_chords
        elif action == "reset":
            return Arrangement()
        else:
            raise ValueError("unknown action")
        return Arrangement.model_validate(values)
    except ValueError as exc:
        raise ServiceError("option") from exc
