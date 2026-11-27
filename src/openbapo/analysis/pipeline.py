"""The analysis pipeline: decode, separate, track beats, build the grid, chords, key.

Stages are injected as protocols so the pipeline itself runs under test with
fakes; the real adapters live in `separate.py`, `beats.py` and `chords/`.
"""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from openbapo.analysis.audio import decode
from openbapo.analysis.beats import BeatTracker
from openbapo.analysis.grid import build_grid
from openbapo.analysis.separate import Separator, Stems, stems_workdir
from openbapo.errors import where
from openbapo.model import Analysis, Chord, Key, LyricSyllable
from openbapo.naming import title_from_filename

if TYPE_CHECKING:
    from openbapo.analysis.chords.stage import ChordStage
    from openbapo.analysis.lyrics.stage import LyricsStage

BeatInput = Literal["mix", "drums_bass"]


@dataclass
class StageTimings:
    """Seconds spent in each stage; `total` covers everything including setup."""

    decode: float = 0.0
    separate: float = 0.0
    beats: float = 0.0
    chords: float = 0.0
    key: float = 0.0
    lyrics: float = 0.0
    total: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)
    # Called with each stage's name as it starts (progress for the user); not timed.
    on_stage: Callable[[str], None] | None = field(default=None, repr=False, compare=False)

    STAGES = ("decode", "separate", "beats", "chords", "key", "lyrics", "total")

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self.on_stage is not None:
            self.on_stage(name)
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
            "lyrics": self.lyrics,
            **self.extra,
            "total": self.total,
        }


@dataclass(frozen=True)
class AnalysisResult:
    analysis: Analysis
    timings: StageTimings
    notes: tuple[str, ...] = ()  # for the user, e.g. "no singing found"
    problems: tuple[str, ...] = ()  # for the operator's log: what was dropped and where


def analyze_file(
    path: Path,
    *,
    separator: Separator,
    tracker: BeatTracker,
    chord_stage: "ChordStage | None" = None,
    title: str | None = None,
    artist: str | None = None,
    beat_input: BeatInput = "mix",
    lyrics_stage: "LyricsStage | None" = None,
    on_stage: Callable[[str], None] | None = None,
) -> AnalysisResult:
    """Run every stage on one file. Stems never outlive this call.

    Without `chord_stage` the result stops after beats and tempo;
    without `lyrics_stage` it has no lyrics. Lyrics are always recognised:
    users cannot attach their own.
    """
    timings = StageTimings(on_stage=on_stage)
    with timings.stage("total"):
        with timings.stage("decode"):
            audio = decode(path)
        chords: tuple[Chord, ...] = ()
        key: Key | None = None
        lyrics: tuple[LyricSyllable, ...] = ()
        notes: tuple[str, ...] = ()
        problems: list[str] = []
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
            if lyrics_stage is not None:
                # Lyrics are the least reliable stage: if they fail, the song still
                # gets its chord sheet, as a song without singing does.
                try:
                    with timings.stage("lyrics"):
                        sung = lyrics_stage.run(vocals_16k(stems))
                    lyrics, notes = sung.syllables, sung.warnings
                except Exception as exc:
                    problems.append(f"lyrics dropped: {where(exc)}")

        def build(lyrics: tuple[LyricSyllable, ...]) -> Analysis:
            return Analysis(
                title=title or title_from_filename(path.name),
                artist=artist,
                duration=audio.duration,
                beats=grid.beats,
                tempo=grid.tempo,
                time_signature=grid.time_signature,
                key=key,
                chords=chords,
                lyrics=lyrics,
            )

        try:
            analysis = build(lyrics)
        except ValueError as exc:
            if not lyrics:
                raise
            problems.append(f"lyrics dropped: {where(exc)}")
            analysis, notes = build(()), ()
    return AnalysisResult(analysis=analysis, timings=timings, notes=notes, problems=tuple(problems))


def vocals_16k(stems: Stems) -> "NDArray[np.float32]":
    """The vocal stem as 16 kHz mono, the rate the lyrics models take."""
    import soxr

    mono = stems.vocals.mean(axis=0)
    return np.asarray(soxr.resample(mono, stems.sample_rate, 16000), dtype=np.float32)
