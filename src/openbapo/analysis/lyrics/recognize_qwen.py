"""Lyrics recognition with Qwen3-ASR, the default recognizer (Whisper is the fallback).

Qwen3-ASR (Alibaba, Apache 2.0) decodes through a language model, so sung
Mandarin whose tones the melody bends is read for sense as well as sound;
it is also trained on songs. As with Whisper, only the voiced regions of the
vocal stem are transcribed, each on its own, and Chinese output is converted
to Simplified characters. Word times come from Qwen3-ForcedAligner (same
family, same licence) and serve only to split lines and give each line its
window: the syllables are timed by the CTC aligner in `align.py`, as before.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from openbapo.analysis.device import AUTO, resolve_device
from openbapo.analysis.lyrics.activity import SAMPLE_RATE, Region
from openbapo.analysis.lyrics.recognize import (
    LANGUAGES,
    Recognition,
    RecognizedLine,
    RecognizedWord,
    _clean,
    _split,
    stock_phrase,
)
from openbapo.errors import InputError

DEFAULT_QWEN_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_QWEN_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"
QWEN_LANGUAGE = {"zh": "Chinese", "en": "English"}  # ours -> Qwen's canonical names
MIN_REGION_S = 0.5  # Qwen3-ASR rejects shorter clips
SAMPLE_S = 30.0  # voiced seconds used to detect the language


@dataclass(frozen=True)
class QwenParams:
    line_gap_s: float = 0.6  # split a region's text where words are this far apart
    max_new_tokens: int = 1024  # per chunk of at most 180 s (qwen-asr splits longer audio)
    batch_size: int = 4  # regions decoded together; small values avoid running out of GPU memory


def load_qwen3_asr(
    model: str = DEFAULT_QWEN_ASR_MODEL,
    aligner: str = DEFAULT_QWEN_ALIGNER,
    device: str = AUTO,
    params: QwenParams | None = None,
) -> object:
    """The `qwen_asr` model wrapper with its forced aligner, both on `device`."""
    import torch

    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as e:
        raise InputError(
            "the qwen3-asr recognizer needs the qwen-asr package "
            '(pip install "openbapo[analysis]"); --recognizer whisper is the fallback'
        ) from e

    device = resolve_device(device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    p = params or QwenParams()
    return Qwen3ASRModel.from_pretrained(
        model,
        forced_aligner=aligner,
        forced_aligner_kwargs={"dtype": dtype, "device_map": device},
        max_inference_batch_size=p.batch_size,
        max_new_tokens=p.max_new_tokens,
        dtype=dtype,
        device_map=device,
    )


def _our_language(reported: str) -> str | None:
    """zh or en from Qwen's "Chinese", "Chinese,English" ...; None when it said nothing."""
    names = {name.strip() for name in reported.split(",") if name.strip()}
    for ours, theirs in QWEN_LANGUAGE.items():
        if theirs in names:
            return ours
    return None


class Qwen3ASRRecognizer:
    """Same interface as WhisperRecognizer: voiced regions in, timed lines out."""

    def __init__(self, model: object, params: QwenParams | None = None) -> None:
        self.model = model
        self.params = params or QwenParams()

    def _clips(self, vocals: NDArray[np.float32], regions: list[Region]) -> list[tuple[float, Any]]:
        clips = []
        for start, end in regions:
            clip = vocals[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
            if len(clip) >= MIN_REGION_S * SAMPLE_RATE:
                clips.append((start, (clip, SAMPLE_RATE)))
        return clips

    def detect_language(self, vocals: NDArray[np.float32], regions: list[Region]) -> str:
        """zh or en, from the language Qwen reports on up to 30 s of voiced audio."""
        voiced = [vocals[int(s * SAMPLE_RATE) : int(e * SAMPLE_RATE)] for s, e in regions]
        sample = (
            np.concatenate(voiced)[: int(SAMPLE_S * SAMPLE_RATE)]
            if voiced
            else vocals[: int(SAMPLE_S * SAMPLE_RATE)]
        )
        if len(sample) < MIN_REGION_S * SAMPLE_RATE:
            return LANGUAGES[0]
        (result,) = self.model.transcribe([(sample, SAMPLE_RATE)])  # type: ignore[attr-defined]
        return _our_language(result.language) or LANGUAGES[0]

    def recognize(
        self, vocals: NDArray[np.float32], regions: list[Region], language: str | None = None
    ) -> Recognition:
        clips = self._clips(vocals, regions)
        if not clips:
            return Recognition(language or "", ())
        results = self.model.transcribe(  # type: ignore[attr-defined]
            [clip for _, clip in clips],
            language=QWEN_LANGUAGE[language] if language else None,
            return_time_stamps=True,
        )
        if language is None:
            heard = Counter(
                ours for r in results if (ours := _our_language(r.language)) is not None
            )
            language = heard.most_common(1)[0][0] if heard else LANGUAGES[0]
        lines: list[RecognizedLine] = []
        dropped = 0
        for (offset, _), result in zip(clips, results, strict=True):
            items = result.time_stamps.items if result.time_stamps is not None else []
            words = [
                RecognizedWord(
                    offset + float(item.start_time),
                    offset + float(item.end_time),
                    _clean(item.text, language),
                )
                for item in items
            ]
            for line in _split(words, self.params.line_gap_s, language):
                if stock_phrase(line.text, likely_silence=False):
                    dropped += 1
                    continue
                lines.append(line)
        return Recognition(language, tuple(lines), dropped)
