"""The analysis pipeline: decode, separate, track beats, build the grid, chords, key.

Stages are injected as protocols so the pipeline itself runs under test with
fakes; the real adapters live in `separate.py`, `beats.py` and `chords/`.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from opencapo.analysis.audio import decode
from opencapo.analysis.beats import BeatTracker
from opencapo.analysis.grid import build_grid
from opencapo.analysis.separate import Separator, stems_workdir
from opencapo.model import Analysis, Chord, Key

if TYPE_CHECKING:
    from opencapo.analysis.chords.stage import ChordStage

BeatInput = Literal["mix", "drums_bass"]


@dataclass
class StageTimings:
    """Seconds spent in each stage; `total` covers everything including setup."""

    decode: float = 0.0
    separate: float = 0.0
    beats: float = 0.0
    chords: float = 0.0
    key: float = 0.0
    total: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    STAGES = ("decode", "separate", "beats", "chords", "key", "total")

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if name in self.STAGES:
                setattr(self, name, getattr(self, name) + elapsed)
            else:
                self.extra[name] = self.extra.get(name, 0.0) + elapsed

    def as_dict(self) -> dict[str, float]:
        return {
            "decode": self.decode,
            "separate": self.separate,
            "beats": self.beats,
            "chords": self.chords,
            "key": self.key,
            **self.extra,
            "total": self.total,
        }


@dataclass(frozen=True)
class AnalysisResult:
    analysis: Analysis
    timings: StageTimings


def analyze_file(
    path: Path,
    *,
    separator: Separator,
    tracker: BeatTracker,
    chord_stage: "ChordStage | None" = None,
    title: str | None = None,
    artist: str | None = None,
    beat_input: BeatInput = "mix",
) -> AnalysisResult:
    """Run every stage on one file. Stems never outlive this call.

    Without `chord_stage` the result stops after beats and tempo (step 1).
    """
    timings = StageTimings()
    with timings.stage("total"):
        with timings.stage("decode"):
            audio = decode(path)
        chords: tuple[Chord, ...] = ()
        key: Key | None = None
        with stems_workdir() as workdir:
            with timings.stage("separate"):
                stems = separator.separate(audio)
                stems.write(workdir)
            with timings.stage("beats"):
                tracked = tracker.track(
                    stems.mix("drums", "bass") if beat_input == "drums_bass" else audio
                )
            grid = build_grid(tracked.beats.tolist(), tracked.downbeats.tolist())
            if chord_stage is not None:
                with timings.stage("chords"):
                    result = chord_stage.run(audio, stems, grid.beats, audio.duration)
                chords, key = result.chords, result.key
                # The key is estimated inside the chord stage; report it separately.
                timings.key = result.timings.get("key", 0.0)
                timings.chords -= timings.key
                timings.extra.update(
                    {f"chords.{k}": v for k, v in result.timings.items() if k != "key"}
                )
        analysis = Analysis(
            title=title or path.stem,
            artist=artist,
            duration=audio.duration,
            beats=grid.beats,
            tempo=grid.tempo,
            time_signature=grid.time_signature,
            key=key,
            chords=chords,
        )
    return AnalysisResult(analysis=analysis, timings=timings)
