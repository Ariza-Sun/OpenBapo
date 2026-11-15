"""Lyrics recognition: faster-whisper on the voiced regions of the vocal stem.

Whisper invents text on music without singing. Three guards keep that out:
only voiced regions are transcribed (`clip_timestamps`), no segment is
conditioned on the previous one, and segments that look like hallucinations
(high no-speech probability, low log-probability, repetitive text) are
dropped. Chinese output is converted to Simplified characters.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from openbapo.analysis.device import AUTO, resolve_device
from openbapo.analysis.lyrics.activity import SAMPLE_RATE, Region

DEFAULT_WHISPER_MODEL = "large-v3"
LANGUAGES = ("zh", "en")


@dataclass(frozen=True)
class RecognizedWord:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class RecognizedLine:
    start: float
    end: float
    text: str
    words: tuple[RecognizedWord, ...]


@dataclass(frozen=True)
class Recognition:
    language: str
    lines: tuple[RecognizedLine, ...]
    dropped: int = 0  # segments rejected as hallucinations

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


@dataclass(frozen=True)
class RecognizeParams:
    beam_size: int = 5
    max_no_speech: float = 0.6
    min_avg_logprob: float = -1.0
    max_compression: float = 2.4
    line_gap_s: float = 0.6  # split a segment where words are this far apart
    extra: dict[str, object] = field(default_factory=dict)


def load_whisper(model: str = DEFAULT_WHISPER_MODEL, device: str = AUTO) -> object:
    """A faster-whisper model wrapped by stable-ts, so it can also align (see align.py)."""
    import stable_whisper

    device = resolve_device(device)
    return stable_whisper.load_faster_whisper(
        model, device=device, compute_type="float16" if device == "cuda" else "int8"
    )


def to_simplified(text: str) -> str:
    import opencc

    return str(opencc.OpenCC("t2s").convert(text))


class WhisperRecognizer:
    def __init__(self, model: object, params: RecognizeParams | None = None) -> None:
        self.model = model
        self.params = params or RecognizeParams()

    def detect_language(self, vocals: NDArray[np.float32], regions: list[Region]) -> str:
        """zh or en, from up to 30 s of voiced audio."""
        voiced = [vocals[int(s * SAMPLE_RATE) : int(e * SAMPLE_RATE)] for s, e in regions]
        sample = (
            np.concatenate(voiced)[: 30 * SAMPLE_RATE] if voiced else vocals[: 30 * SAMPLE_RATE]
        )
        _, _, probs = self.model.detect_language(sample)  # type: ignore[attr-defined]
        scores = dict(probs)
        return max(LANGUAGES, key=lambda lang: scores.get(lang, 0.0))

    def recognize(
        self, vocals: NDArray[np.float32], regions: list[Region], language: str | None = None
    ) -> Recognition:
        if not regions:
            return Recognition(language or "", ())
        language = language or self.detect_language(vocals, regions)
        p = self.params
        segments, _ = self.model.transcribe(  # type: ignore[attr-defined]
            vocals,
            language=language,
            beam_size=p.beam_size,
            condition_on_previous_text=False,
            word_timestamps=True,
            clip_timestamps=[t for region in regions for t in region],
            vad_filter=False,
            **p.extra,
        )
        lines: list[RecognizedLine] = []
        dropped = 0
        for seg in segments:
            if (
                seg.no_speech_prob > p.max_no_speech
                or seg.avg_logprob < p.min_avg_logprob
                or seg.compression_ratio > p.max_compression
            ):
                dropped += 1
                continue
            words = [
                RecognizedWord(float(w.start), float(w.end), _clean(w.word, language))
                for w in (seg.words or [])
            ]
            lines.extend(_split(words, p.line_gap_s, language))
        return Recognition(language, tuple(lines), dropped)


def _clean(text: str, language: str) -> str:
    text = text.strip()
    return to_simplified(text) if language == "zh" else text


def _split(words: list[RecognizedWord], gap: float, language: str) -> list[RecognizedLine]:
    groups: list[list[RecognizedWord]] = []
    for word in words:
        if not word.text:
            continue
        if not groups or word.start - groups[-1][-1].end >= gap:
            groups.append([])
        groups[-1].append(word)
    joiner = "" if language == "zh" else " "
    return [
        RecognizedLine(g[0].start, g[-1].end, joiner.join(w.text for w in g).strip(), tuple(g))
        for g in groups
    ]
