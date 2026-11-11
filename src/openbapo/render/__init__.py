"""Rendering: from an ArrangedSheet to SVG pages, PNG, PDF and MusicXML (step 4)."""

from openbapo.render.export import svg_pages_to_pdf, svg_to_png
from openbapo.render.layout import BARS_PER_SYSTEM, Page, paginate
from openbapo.render.musicxml import render_musicxml
from openbapo.render.svg import render_svg


def render_png(sheet, dpi: int = 150) -> bytes:  # type: ignore[no-untyped-def]
    """The first page as PNG."""
    return svg_to_png(render_svg(sheet)[0], dpi=dpi)


def render_pdf(sheet) -> bytes:  # type: ignore[no-untyped-def]
    return svg_pages_to_pdf(render_svg(sheet))


__all__ = [
    "BARS_PER_SYSTEM",
    "Page",
    "paginate",
    "render_musicxml",
    "render_pdf",
    "render_png",
    "render_svg",
    "svg_pages_to_pdf",
    "svg_to_png",
]
