"""Reproducible rhythm proofs from the public-domain Twinkle score, without inference.

Run: python scripts/render_samples.py [--out docs/samples]
The historical audio analysis is rendered separately so its extra trailing
beat remains visible rather than being silently removed from a comparison.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

from openbapo.arrange import arrange
from openbapo.model import Analysis, Arrangement, Beat, Chord, Key, Tempo, TimeSignature
from openbapo.render import layout as L
from openbapo.render import render_musicxml, render_svg, svg_pages_to_pdf, svg_to_png
from openbapo.render.svg import FONT
from openbapo.sheet.format import parse_sheet

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "http://www.w3.org/2000/svg"
ET.register_namespace("", NAMESPACE)


def twinkle() -> Analysis:
    """Known chord and beat data; this fixture does not claim an analysis score."""
    source = parse_sheet((ROOT / "docs/examples/xiaoxingxing.sheet").read_text())
    chords = []
    for index, bar in enumerate(source.bars):
        position = Fraction(index * source.beats_per_bar)
        for c in bar.chords:
            end = position + c.beats
            chords.append(
                Chord(
                    start=float(position) * source.seconds_per_beat,
                    end=float(end) * source.seconds_per_beat,
                    root=(c.root + source.capo) % 12,
                    quality=c.quality,
                    bass=None if c.bass is None else (c.bass + source.capo) % 12,
                    confidence=1.0,
                )
            )
            position = end
    return Analysis(
        title=source.title,
        artist="法国民歌",
        duration=source.duration,
        beats=tuple(
            Beat(
                time=i * source.seconds_per_beat,
                is_downbeat=i % source.beats_per_bar == 0,
                bar_index=i // source.beats_per_bar,
                beat_in_bar=i % source.beats_per_bar + 1,
            )
            for i in range(len(source.bars) * source.beats_per_bar)
        ),
        tempo=Tempo(bpm=source.bpm),
        time_signature=TimeSignature(numerator=4, denominator=4),
        key=Key(tonic=source.key, mode=source.mode, confidence=1.0),
        chords=tuple(chords),
    )


def write_sheet(
    out: Path, name: str, analysis: Analysis, pattern: str, *, full: bool = True
) -> str:
    sheet = arrange(analysis, Arrangement(strum_pattern=pattern, fingering_key=0))
    pages = render_svg(sheet)
    (out / f"{name}.png").write_bytes(svg_to_png(pages[0], dpi=150))
    if full:
        (out / f"{name}.pdf").write_bytes(svg_pages_to_pdf(pages))
        (out / f"{name}.musicxml").write_text(render_musicxml(sheet))
    return pages[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/samples")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    analysis = twinkle()
    examples = []
    for pattern, suffix, caption in (
        ("folk", "", "民谣扫弦：第二拍后半拍延续到第三拍；每小节 4 拍"),
        ("quarter-down", "-quarter", "四分下扫：四次下扫，每次 1 拍"),
        ("arp-53231323", "-arpeggio", "分解和弦：每拍两个八分音符，× 标明拨哪根弦"),
    ):
        svg = write_sheet(args.out, f"step-4-xiaoxingxing{suffix}", analysis, pattern)
        examples.append((caption, svg))
    compound = analysis.model_copy(
        update={
            "time_signature": TimeSignature(numerator=6, denominator=8),
            "beats": tuple(
                Beat(time=i * 0.5, is_downbeat=i % 6 == 0, bar_index=i // 6, beat_in_bar=i % 6 + 1)
                for i in range(24)
            ),
            "duration": 12.0,
            "chords": (analysis.chords[0].model_copy(update={"end": 12.0}),),
        }
    )
    # A meter demonstration, not a claim that Twinkle is in 6/8.
    examples.append(
        (
            "6/8 记谱检查：六个八分音符，按 3 + 3 连杠",
            render_svg(arrange(compound, Arrangement(strum_pattern="six-eight", fingering_key=0)))[
                0
            ],
        )
    )
    root = ET.Element(f"{{{NAMESPACE}}}svg", width="595pt", height="680pt", viewBox="0 0 595 680")
    ET.SubElement(root, f"{{{NAMESPACE}}}rect", width="595", height="680", fill="white")
    for row, (caption, svg) in enumerate(examples):
        label = ET.SubElement(
            root,
            f"{{{NAMESPACE}}}text",
            x="40",
            y=str(row * 162 + 28),
            attrib={"font-family": FONT, "font-size": "10"},
        )
        label.text = caption
        panel = ET.fromstring(svg)
        panel.attrib.update(
            x="0",
            y=str(row * 162 + 42),
            width="595",
            height="116",
            viewBox=f"0 {L.MARGIN + L.HEADER_H - 3} 595 116",
        )
        root.append(panel)
    proof = ET.tostring(root, encoding="unicode")
    (args.out / "rhythm-proof.svg").write_text(proof)
    (args.out / "rhythm-proof.png").write_bytes(svg_to_png(proof, dpi=180))
    original = Analysis.from_json((ROOT / "docs/examples/xiaoxingxing.analysis.json").read_text())
    original_sheet = arrange(original)
    (args.out / "step-4-xiaoxingxing-analysis.png").write_bytes(
        svg_to_png(render_svg(original_sheet)[0])
    )
    print(
        json.dumps(
            {
                "bars": len(arrange(analysis).bars),
                "patterns": [p[0] for p in examples],
                "historical_analysis_bars": len(original_sheet.bars),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
