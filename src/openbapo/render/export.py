"""SVG pages -> PNG (one page) and PDF (all pages)."""

from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from typing import Any

from openbapo.errors import InputError


def _converter(name: str) -> Callable[..., Any]:
    """cairosvg.svg2png / svg2pdf, with an install hint when the extra or libcairo is missing."""
    try:
        import cairosvg
    except (ImportError, OSError) as e:  # cairocffi raises OSError without libcairo
        raise InputError(
            "PNG / PDF export needs the render extra and libcairo: "
            'pip install "openbapo[render]" (Debian / Ubuntu: apt install libcairo2)'
        ) from e
    convert: Callable[..., Any] = getattr(cairosvg, name)
    return convert


def svg_to_png(svg: str, dpi: int = 150) -> bytes:
    return bytes(_converter("svg2png")(bytestring=svg.encode("utf-8"), dpi=dpi))


def svg_pages_to_pdf(pages: list[str]) -> bytes:
    """A multi-page vector PDF: each page converted on its own, then merged."""
    from pypdf import PdfWriter

    svg2pdf = _converter("svg2pdf")
    writer = PdfWriter()
    for svg in pages:
        page_pdf = bytes(svg2pdf(bytestring=svg.encode("utf-8")))
        writer.append(BytesIO(page_pdf))
    out = BytesIO()
    writer.write(out)
    return out.getvalue()
