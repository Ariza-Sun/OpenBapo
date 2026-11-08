"""The chord + key stage: run the experts, fuse on the beat grid, cross-check the key.

Order of operations (docs/plans/step-2.md, section 2):

1. every expert produces frame posteriors, aggregated per beat;
2. the bass stem gives a per-beat bass pitch class;
3. a first fusion without a key prior yields preliminary chords;
4. the key is estimated from the chroma and from those chords, cross-checked;
5. a second fusion with the key prior yields the final chords.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from openbapo.analysis.audio import Audio
from openbapo.analysis.chords.bassline import BassLineDetector, bass_source
from openbapo.analysis.chords.chroma import ChromaTemplateExpert, chromagram, mean_chroma
from openbapo.analysis.chords.fusion import (
    BassEstimate,
    ChordExpert,
    FusionParams,
    aggregate_to_beats,
    decode,
)
from openbapo.analysis.chords.key import estimate_key, scores_from_chords, scores_from_chroma
from openbapo.analysis.separate import Stems
from openbapo.model import Beat, Chord, Key


@dataclass(frozen=True)
class ChordStageResult:
    chords: tuple[Chord, ...]
    key: Key | None
    beat_probs: dict[str, NDArray[np.float64]]  # per expert, for evaluation and tuning
    bass: BassEstimate | None
    timings: dict[str, float] = field(default_factory=dict)  # seconds per expert and sub-step


class ChordStage:
    def __init__(
        self,
        experts: Sequence[ChordExpert],
        *,
        bass_detector: BassLineDetector | None = None,
        params: FusionParams | None = None,
    ) -> None:
        if not experts:
            raise ValueError("at least one chord expert is needed")
        self.experts = list(experts)
        self.bass_detector = bass_detector if bass_detector is not None else BassLineDetector()
        self.params = params or FusionParams()

    @property
    def model_names(self) -> list[str]:
        return [e.name for e in self.experts if e.is_model]

    def run(
        self, audio: Audio, stems: Stems | None, beats: Sequence[Beat], duration: float
    ) -> ChordStageResult:
        beat_times = [b.time for b in beats]
        timings: dict[str, float] = {}

        start = time.perf_counter()
        harmonic = stems.mix("bass", "other", "vocals") if stems is not None else audio
        gram = chromagram(harmonic)
        timings["chroma"] = time.perf_counter() - start

        beat_probs: dict[str, NDArray[np.float64]] = {}
        for expert in self.experts:
            start = time.perf_counter()
            if isinstance(expert, ChromaTemplateExpert):
                posteriors = expert.from_chromagram(gram)
            else:
                posteriors = expert.estimate(audio, stems)
            beat_probs[expert.name] = aggregate_to_beats(
                posteriors, beat_times, duration, voiced=self.params.voiced_aggregation
            )
            timings[expert.name] = time.perf_counter() - start

        start = time.perf_counter()
        bass = self.bass_detector.detect(bass_source(audio, stems), beat_times, duration)
        timings["bass"] = time.perf_counter() - start

        start = time.perf_counter()
        preliminary = decode(
            beats, beat_probs, self.model_names, duration=duration, bass=bass, params=self.params
        )
        key = estimate_key(
            scores_from_chroma(mean_chroma(gram)),
            scores_from_chords(preliminary.chords),
            preliminary.chords,
        )
        timings["key"] = time.perf_counter() - start

        start = time.perf_counter()
        final = decode(
            beats,
            beat_probs,
            self.model_names,
            duration=duration,
            key=key,
            bass=bass,
            params=self.params,
        )
        timings["fusion"] = time.perf_counter() - start
        return ChordStageResult(
            chords=final.chords, key=key, beat_probs=beat_probs, bass=bass, timings=timings
        )


def default_chord_stage(device: str = "auto") -> ChordStage:
    """The production committee: BTC + lv-chordia + chroma templates, bass line, key."""
    from openbapo.analysis.chords.models.btc import BTCExpert
    from openbapo.analysis.chords.models.lvchordia import LVChordiaExpert

    return ChordStage(
        [BTCExpert(device=device), LVChordiaExpert(device=device), ChromaTemplateExpert()]
    )
