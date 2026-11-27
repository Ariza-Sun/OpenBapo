"""Fetch every model's weights ahead of the first song.

Each model downloads on first use anyway; this does it once up front, on the
CPU, so a new deployment does not make its first user wait (or time out).
Weights go to the usual Hugging Face / torch caches (HF_HOME, TORCH_HOME)
and ~/.cache/openbapo; nothing is bundled with the code.
"""

import gc
from collections.abc import Callable

RECOGNIZERS = ("qwen3-asr", "whisper", "all")


def _drop(load: Callable[[], object]) -> None:
    instance = load()
    del instance
    gc.collect()


def download_models(recognizer: str = "qwen3-asr", say: Callable[[str], None] = print) -> None:
    """Download (and check) the weights the pipeline uses with `recognizer`."""
    from openbapo.analysis.beats import BeatThisTracker
    from openbapo.analysis.chords.models.btc import BTCExpert
    from openbapo.analysis.chords.models.lvchordia import LVChordiaExpert
    from openbapo.analysis.lyrics.align import wav2vec2_aligner
    from openbapo.analysis.separate import DemucsSeparator

    for adapter in (DemucsSeparator, BeatThisTracker, BTCExpert, LVChordiaExpert):
        say(f"{adapter.__name__} ...")
        _drop(adapter(device="cpu")._load)
    aligner = wav2vec2_aligner("cpu")
    for language in ("en", "zh"):
        say(f"lyrics aligner ({language}) ...")
        _drop(lambda language=language: aligner.model(language))  # type: ignore[misc]
    if recognizer in ("qwen3-asr", "all"):
        from huggingface_hub import snapshot_download

        from openbapo.analysis.lyrics.recognize_qwen import (
            DEFAULT_QWEN_ALIGNER,
            DEFAULT_QWEN_ASR_MODEL,
        )

        for repo in (DEFAULT_QWEN_ASR_MODEL, DEFAULT_QWEN_ALIGNER):
            say(f"{repo} ...")
            snapshot_download(repo)
    if recognizer in ("whisper", "all"):
        from faster_whisper import download_model

        from openbapo.analysis.lyrics.recognize import DEFAULT_WHISPER_MODEL

        say(f"Whisper {DEFAULT_WHISPER_MODEL} ...")
        download_model(DEFAULT_WHISPER_MODEL)
    say("All model weights are ready.")
