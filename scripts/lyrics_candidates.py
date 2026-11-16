"""The two lyrics aligners that lost to wav2vec2 on the step-5 development set.

Kept for scripts/eval_lyrics.py so the comparison in
docs/benchmarks/step-5-lyrics-dev.md can be reproduced; not part of the package.
Needs scripts/requirements-eval.txt (stable-ts, pypinyin).

- MMS forced alignment (Meta, torchaudio `MMS_FA`), Chinese spelled as
  toneless pinyin. **The weights are CC BY-NC 4.0.**
- Whisper word timing through stable-ts: the known text is fed to Whisper
  and the words are timed from its cross-attention (MIT).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from openbapo.analysis.device import AUTO, resolve_device
from openbapo.analysis.lyrics.align import (
    CtcAligner,
    CtcModel,
    TimedToken,
    _torch_forward,
    song_language,
)
from openbapo.lyrics import Line, Token


def mms(device: str = AUTO) -> CtcModel:
    import torchaudio
    from pypinyin import lazy_pinyin

    device = resolve_device(device)
    bundle = torchaudio.pipelines.MMS_FA
    net = bundle.get_model(with_star=False).to(device).eval()
    labels = bundle.get_dict(star=None)

    def spell(token: Token) -> list[str]:
        text = "".join(lazy_pinyin(token.text)) if token.lang == "zh" else token.spoken
        return [c for c in text.lower() if c in labels and c != "-"]

    return CtcModel(_torch_forward(net, device, normalize=False), labels, 0, 0.02, spell)


def mms_aligner(device: str = AUTO) -> CtcAligner:
    shared: dict[str, CtcModel] = {}

    def load() -> CtcModel:
        if "m" not in shared:
            shared["m"] = mms(device)
        return shared["m"]

    return CtcAligner("mms", {"en": load, "zh": load})


class WhisperAligner:
    """stable-ts alignment with its own faster-whisper `large-v3`."""

    name = "whisper"

    def __init__(self, device: str = AUTO) -> None:
        import stable_whisper

        device = resolve_device(device)
        self.model: Any = stable_whisper.load_faster_whisper(
            "large-v3", device=device, compute_type="float16" if device == "cuda" else "int8"
        )

    def align(
        self, vocals: np.ndarray, lines: Sequence[Line], regions: Sequence[Any] | None = None
    ) -> list[TimedToken]:
        language = song_language(lines)
        joiner = "" if language == "zh" else " "
        text = "\n".join(joiner.join(t.text for t in line.tokens) for line in lines)
        result = self.model.align(vocals, text, language=language)
        if result is None:
            return [TimedToken(line.index, t, None, None) for line in lines for t in line.tokens]
        # Walk the result's words and ours over the same character stream.
        timed_chars: list[tuple[str, float, float, float]] = []
        for word in result.all_words():
            chars = list(_norm(word.word))
            if not chars:
                continue
            step = (word.end - word.start) / len(chars)
            for k, c in enumerate(chars):
                timed_chars.append(
                    (c, word.start + k * step, word.start + (k + 1) * step, word.probability)
                )
        out: list[TimedToken] = []
        pos = 0
        for line in lines:
            for token in line.tokens:
                want = _norm(token.text)
                span = timed_chars[pos : pos + len(want)]
                if want and "".join(c for c, *_ in span) == want:
                    pos += len(want)
                    score = float(np.mean([s for *_, s in span]))
                    out.append(TimedToken(line.index, token, span[0][1], span[-1][2], score))
                else:
                    out.append(TimedToken(line.index, token, None, None))
        return out


def _norm(text: str) -> str:
    return "".join(c for c in text.lower().replace("’", "'") if c.isalnum() or c == "'")
