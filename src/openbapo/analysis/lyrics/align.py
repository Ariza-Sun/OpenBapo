"""Lyrics alignment: lines of lyrics + vocal stem -> a time span per token.

CTC forced alignment with a wav2vec2 model per language: torchaudio's
LibriSpeech model for English (MIT), an XLSR-53 Chinese fine-tune for Chinese
(Apache 2.0), as in WhisperX. It beat Meta's MMS aligner and Whisper's own
word timing on the step-5 development set (docs/benchmarks/step-5-lyrics-dev.md);
those two candidates live on in scripts/lyrics_candidates.py.

CTC alignment runs over the whole song at once, so a line cannot be placed
before the line above it. Frames outside the voiced regions are forced to
the blank label, which keeps lyrics off instrumental passages. Tokens the
model cannot spell (a character missing from its vocabulary) get no time
here; the lyrics stage interpolates them.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from openbapo.analysis.device import AUTO, resolve_device
from openbapo.analysis.lyrics.activity import SAMPLE_RATE, Region
from openbapo.lyrics import Line, Token

CHUNK_S = 30.0  # longest stretch of audio given to a wav2vec2 model at once
REGION_PAD_S = 0.3
ZH_CTC_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
ZH_CTC_REVISION = "99ccb2737be22b8bb50dcfcc39ad4d567fb90cfd"
ZH_CTC_SHA256 = "de031fd4b29e0c0667e5346450fadfe1326c89936b888b59c4ede608db763ee4"


@dataclass(frozen=True)
class TimedToken:
    line: int
    token: Token
    start: float | None
    end: float | None
    score: float = 0.0  # mean label probability, 0..1; 0 when untimed


class Aligner(Protocol):
    name: str

    def align(
        self,
        vocals: NDArray[np.float32],
        lines: Sequence[Line],
        regions: Sequence[Region] | None = None,
    ) -> list[TimedToken]: ...


def song_language(lines: Sequence[Line]) -> str:
    tokens = [t for line in lines for t in line.tokens]
    zh = sum(t.lang == "zh" for t in tokens)
    return "zh" if tokens and zh * 3 >= len(tokens) else "en"


# --- CTC -------------------------------------------------------------------------


@dataclass
class CtcModel:
    """A CTC acoustic model: audio -> per-frame log-probabilities over `labels`."""

    forward: Callable[[NDArray[np.float32]], NDArray[np.float32]]  # (frames, labels)
    labels: dict[str, int]
    blank: int
    frame_s: float
    spell: Callable[[Token], list[str]]  # token -> label strings; [] when unknown
    # Word-boundary label the model was trained to emit between words ("|" for
    # the LibriSpeech wav2vec2); appended to every word but the last.
    separator: str | None = None


def ctc_targets(model: CtcModel, tokens: Sequence[Token]) -> list[list[int]]:
    out = []
    for k, token in enumerate(tokens):
        spelled = model.spell(token)
        ids = [model.labels[s] for s in spelled if s in model.labels]
        if len(ids) != len(spelled):
            ids = []
        if ids and model.separator is not None and k < len(tokens) - 1:
            ids.append(model.labels[model.separator])
        out.append(ids)
    return out


def emissions(
    model: CtcModel, vocals: NDArray[np.float32], regions: Sequence[Region] | None
) -> NDArray[np.float32]:
    """Log-probabilities for the whole song; blank outside the (padded) regions."""
    total = int(len(vocals) / SAMPLE_RATE / model.frame_s) + 1
    out = np.full((total, len(model.labels)), -1e4, dtype=np.float32)
    out[:, model.blank] = 0.0
    duration = len(vocals) / SAMPLE_RATE
    spans = regions if regions else [(0.0, duration)]
    for start, end in _merge_padded(spans, duration):
        t = start
        while t < end:
            stop = min(end, t + CHUNK_S)
            chunk = vocals[int(t * SAMPLE_RATE) : int(stop * SAMPLE_RATE)]
            if len(chunk) >= SAMPLE_RATE // 10:
                frames = model.forward(chunk)
                first = round(t / model.frame_s)
                n = min(len(frames), total - first)
                out[first : first + n] = frames[:n]
            t = stop
    return out


def _merge_padded(spans: Sequence[Region], duration: float) -> list[Region]:
    merged: list[Region] = []
    for start, end in sorted(spans):
        start, end = max(0.0, start - REGION_PAD_S), min(duration, end + REGION_PAD_S)
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def forced_align(
    log_probs: NDArray[np.float32], targets: list[list[int]], blank: int, frame_s: float
) -> list[tuple[float, float, float] | None]:
    """(start, end, score) per target group; None for an empty group."""
    import torch
    import torchaudio.functional as F

    flat = [i for group in targets for i in group]
    if not flat:
        return [None] * len(targets)
    # CTC needs a frame per label plus one between repeated labels.
    repeats = sum(a == b for a, b in pairwise(flat))
    if len(flat) + repeats > len(log_probs):
        raise ValueError("lyrics are too long for the audio")
    emission = torch.from_numpy(log_probs).unsqueeze(0)
    target = torch.tensor([flat], dtype=torch.int32)
    labels, scores = F.forced_align(emission, target, blank=blank)
    spans = F.merge_tokens(labels[0], scores[0].exp(), blank=blank)
    out: list[tuple[float, float, float] | None] = []
    k = 0
    for group in targets:
        if not group:
            out.append(None)
            continue
        mine = spans[k : k + len(group)]
        k += len(group)
        score = float(np.mean([s.score for s in mine]))
        out.append((mine[0].start * frame_s, mine[-1].end * frame_s, score))
    return out


class CtcAligner:
    def __init__(self, name: str, models: dict[str, Callable[[], CtcModel]]) -> None:
        """`models` maps a song language (zh / en) to a loader of its CTC model."""
        self.name = name
        self._loaders = models
        self._models: dict[str, CtcModel] = {}

    def model(self, language: str) -> CtcModel:
        if language not in self._models:
            self._models[language] = self._loaders[language]()
        return self._models[language]

    def align(
        self,
        vocals: NDArray[np.float32],
        lines: Sequence[Line],
        regions: Sequence[Region] | None = None,
    ) -> list[TimedToken]:
        model = self.model(song_language(lines))
        pairs = [(line.index, token) for line in lines for token in line.tokens]
        targets = ctc_targets(model, [token for _, token in pairs])
        log_probs = emissions(model, vocals, regions)
        timed = forced_align(log_probs, targets, model.blank, model.frame_s)
        return [
            TimedToken(i, token, *(t if t else (None, None, 0.0)))
            for (i, token), t in zip(pairs, timed, strict=True)
        ]


def _torch_forward(net: object, device: str, normalize: bool) -> Callable[..., NDArray[np.float32]]:
    import torch

    def forward(chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        x = torch.from_numpy(np.ascontiguousarray(chunk)).unsqueeze(0).to(device)
        if normalize:
            x = (x - x.mean()) / torch.sqrt(x.var() + 1e-7)
        with torch.inference_mode():
            out = net(x)  # type: ignore[operator]
            logits = out[0] if isinstance(out, tuple) else out.logits
            return torch.log_softmax(logits[0].float(), dim=-1).cpu().numpy()

    return forward


def wav2vec2_english(device: str = AUTO) -> CtcModel:
    import torchaudio

    device = resolve_device(device)
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H
    net = bundle.get_model().to(device).eval()
    labels = {label: i for i, label in enumerate(bundle.get_labels())}

    def spell(token: Token) -> list[str]:
        return list(token.spoken.upper()) if token.lang == "en" else []

    forward = _torch_forward(net, device, normalize=False)
    return CtcModel(forward, labels, 0, 0.02, spell, separator="|")


def wav2vec2_chinese(device: str = AUTO) -> CtcModel:
    """The Chinese model at a fixed revision, its weights checked against their SHA-256.

    The repository has only a pickled `pytorch_model.bin`, which recent
    transformers refuse to read on torch < 2.6; reading it here with
    `weights_only=True` after the checksum keeps every supported torch working.
    """
    import hashlib

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2Config, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC

    device = resolve_device(device)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(ZH_CTC_MODEL, revision=ZH_CTC_REVISION)
    config = Wav2Vec2Config.from_pretrained(ZH_CTC_MODEL, revision=ZH_CTC_REVISION)
    path = hf_hub_download(ZH_CTC_MODEL, "pytorch_model.bin", revision=ZH_CTC_REVISION)
    with open(path, "rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != ZH_CTC_SHA256:
            raise RuntimeError(f"{path} does not match the expected checksum; delete it and retry")
    net = Wav2Vec2ForCTC(config)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net = net.to(device).eval()  # type: ignore[arg-type]
    vocab = tokenizer.get_vocab()
    blank = int(tokenizer.pad_token_id)

    def spell(token: Token) -> list[str]:
        if token.lang == "zh":
            return [token.text]
        return list(token.spoken.upper()) if all(c.upper() in vocab for c in token.spoken) else []

    return CtcModel(_torch_forward(net, device, normalize=True), vocab, blank, 0.02, spell)


def wav2vec2_aligner(device: str = AUTO) -> CtcAligner:
    return CtcAligner(
        "wav2vec2",
        {"en": lambda: wav2vec2_english(device), "zh": lambda: wav2vec2_chinese(device)},
    )
