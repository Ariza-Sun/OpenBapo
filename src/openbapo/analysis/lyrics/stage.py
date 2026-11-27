"""The lyrics stage: vocal stem (+ lyrics text) -> syllables with times.

With lyrics text: align it. Without: recognise, then align the recognised
lines the same way. No singing found: no lyrics, and a note saying so.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from openbapo.analysis.lyrics.activity import SAMPLE_RATE, ActivityDetector
from openbapo.analysis.lyrics.align import Aligner, TimedToken
from openbapo.analysis.lyrics.recognize import Recognition, Recognizer
from openbapo.lyrics import Line, clean_line, parse_lyrics, tokenize
from openbapo.model import LyricSyllable

NO_VOCALS = "no singing found; the sheet has no lyrics"


@dataclass(frozen=True)
class LyricsResult:
    syllables: tuple[LyricSyllable, ...]
    lines: tuple[Line, ...]
    timed: tuple[TimedToken, ...]
    recognition: Recognition | None = None
    warnings: tuple[str, ...] = field(default=())


class LyricsStage:
    def __init__(
        self,
        activity: ActivityDetector,
        recognizer: Recognizer | None,
        aligner: Aligner,
        *,
        alignment_activity: ActivityDetector | None = None,
    ) -> None:
        self.activity = activity
        self.alignment_activity = alignment_activity if alignment_activity is not None else activity
        self.recognizer = recognizer
        self.aligner = aligner

    def run(self, vocals: NDArray[np.float32], lyrics_text: str | None = None) -> LyricsResult:
        detector = self.alignment_activity if lyrics_text is not None else self.activity
        regions = detector.regions(vocals)
        if not detector.has_vocals(regions):
            return LyricsResult((), (), (), warnings=(NO_VOCALS,))
        recognition = None
        windows: list[tuple[float, float]] | None = None
        if lyrics_text is None:
            if self.recognizer is None:
                raise ValueError("no lyrics text and no recognizer")
            recognition = self.recognizer.recognize(vocals, regions)
            # Keep each recognised line with the window the recognizer heard it in.
            lines, windows = recognised_lines(recognition)
        else:
            lines = parse_lyrics(lyrics_text)
        if not lines:
            return LyricsResult((), (), (), recognition, (NO_VOCALS,))
        timed = self.aligner.align(vocals, lines, regions, windows)
        duration = len(vocals) / SAMPLE_RATE
        syllables, warnings = to_syllables(timed, duration)
        return LyricsResult(syllables, lines, tuple(timed), recognition, warnings)


def recognised_lines(
    recognition: Recognition,
) -> tuple[tuple[Line, ...], list[tuple[float, float]]]:
    """Lines of the recognised text, numbered from 0, each with its time window."""
    lines: list[Line] = []
    windows: list[tuple[float, float]] = []
    for heard in recognition.lines:
        text = clean_line(heard.text)
        tokens = tokenize(text)
        if tokens:
            lines.append(Line(len(lines), text, tokens))
            windows.append((heard.start, heard.end))
    return tuple(lines), windows


def fill_gaps(timed: Sequence[TimedToken], duration: float) -> list[tuple[float, float]]:
    """A span for every token: untimed ones share the gap between their timed neighbours."""
    spans: list[tuple[float, float] | None] = [
        (t.start, t.end) if t.start is not None and t.end is not None else None for t in timed
    ]
    out: list[tuple[float, float]] = []
    i = 0
    while i < len(spans):
        span = spans[i]
        if span is not None:
            out.append(span)
            i += 1
            continue
        j = i
        while j < len(spans) and spans[j] is None:
            j += 1
        left = out[-1][1] if out else 0.0
        right_span = spans[j] if j < len(spans) else None
        right = right_span[0] if right_span is not None else min(duration, left + 0.5 * (j - i))
        step = max(right - left, 0.0) / (j - i)
        out.extend((left + k * step, left + (k + 1) * step) for k in range(j - i))
        i = j
    return out


def to_syllables(
    timed: Sequence[TimedToken], duration: float
) -> tuple[tuple[LyricSyllable, ...], tuple[str, ...]]:
    spans = fill_gaps(timed, duration)
    untimed = sum(t.start is None for t in timed)
    syllables: list[LyricSyllable] = []
    last = 0.0
    for t, (start, end) in zip(timed, spans, strict=True):
        start = max(start, last)
        end = max(end, start)
        parts = t.token.syllables
        weights = np.array([len(p.rstrip("-")) or 1 for p in parts], dtype=float)
        edges = start + (end - start) * np.concatenate([[0.0], np.cumsum(weights) / weights.sum()])
        for k, text in enumerate(parts):
            syllables.append(
                LyricSyllable(
                    start=round(float(edges[k]), 3),
                    end=round(float(edges[k + 1]), 3),
                    text=text,
                    line_index=t.line,
                )
            )
        # The next word starts no earlier than this word's last syllable: aligned lines
        # can touch by a few milliseconds where a split word ends (seen with Qwen3-ASR).
        last = float(edges[len(parts) - 1])
    warnings = (
        (f"{untimed} of {len(timed)} words could not be timed and were interpolated",)
        if untimed
        else ()
    )
    return tuple(syllables), warnings
