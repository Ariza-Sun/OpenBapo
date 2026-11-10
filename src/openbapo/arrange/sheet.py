"""Assembling the renderable sheet from an Analysis and an Arrangement.

Everything a renderer needs is decided here: which key the chords are
written in, how each chord is spelled and fingered, which strum pattern
plays, what the header says. The renderer (step 4) draws; it makes no
musical choices.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr

from openbapo.arrange import strums
from openbapo.arrange.chords import chord_name, simplify, written_root
from openbapo.arrange.keys import (
    FINGERING_CANDIDATES,
    capo_for,
    default_fingering,
    fingering_options,
    key_name,
    relative_major,
    sung_key,
)
from openbapo.arrange.shapes import Shape, choose_shape
from openbapo.arrange.strums import StrumPattern
from openbapo.model import Analysis, Arrangement, LyricSyllable, Mode, Note
from openbapo.model._base import Model
from openbapo.model.music import Confidence

LOW_CONFIDENCE = 0.5  # chords below this are marked on the sheet
BEAT_TOLERANCE = 0.03  # seconds: a chord "starts on" the beat nearest within this
DEFAULT_METER = (4, 4)


class Header(Model):
    title: StrictStr | None
    artist: StrictStr | None
    original_key: StrictStr | None  # None when the analysis found no key
    sung_key: StrictStr
    fingering_key: StrictStr
    capo: Annotated[StrictInt, Field(ge=0, le=11)]
    transpose: StrictInt
    bpm: StrictFloat | None
    time_signature: tuple[StrictInt, StrictInt]
    jianpu_key: StrictStr  # "1=G": the sung key's major, as Chinese sheets print it
    text: StrictStr  # "原调 G · 演唱调 F · E 调指法 夹 1 品"


class FingeringChoice(Model):
    tonic: Annotated[StrictInt, Field(ge=0, le=11)]
    name: StrictStr
    capo: Annotated[StrictInt, Field(ge=0, le=11)]
    recommended: StrictBool
    selected: StrictBool


class StrumOption(Model):
    id: StrictStr
    name: StrictStr
    mnemonic: StrictStr
    kind: StrictStr
    recommended: StrictBool  # the one `auto` would pick


class BarChord(Model):
    beat: Annotated[StrictInt, Field(ge=0)]  # 0 = the bar's downbeat
    name: StrictStr  # as written on the sheet (simplified when the switch is on)
    original_name: StrictStr  # never simplified
    shape: StrictStr  # key into ArrangedSheet.shapes
    confidence: Confidence
    low_confidence: StrictBool


class Bar(Model):
    index: Annotated[StrictInt, Field(ge=0)]
    start: StrictFloat
    end: StrictFloat
    beats: Annotated[StrictInt, Field(ge=1)]
    chords: tuple[BarChord, ...]


class ArrangedSheet(Model):
    header: Header
    fingering_options: tuple[FingeringChoice, ...]
    strum: StrumPattern | None  # None: the tab line shows chord changes only
    strum_options: tuple[StrumOption, ...]
    shapes: dict[str, Shape]  # every chord name on the sheet -> its diagram
    bars: tuple[Bar, ...]
    notes: tuple[Note, ...]
    lyrics: tuple[LyricSyllable, ...]
    low_confidence_count: Annotated[StrictInt, Field(ge=0)]
    warnings: tuple[StrictStr, ...]


def _bars(analysis: Analysis) -> list[tuple[int, float, float, list[float]]]:
    """(index, start, end, beat times) per bar, from the beat grid."""
    groups: dict[int, list[float]] = {}
    for beat in analysis.beats:
        groups.setdefault(beat.bar_index, []).append(beat.time)
    indices = sorted(groups)
    out = []
    for n, index in enumerate(indices):
        times = groups[index]
        if n + 1 < len(indices):
            end = groups[indices[n + 1]][0]
        else:
            period = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else 0.5
            end = times[-1] + period
            if analysis.duration is not None:
                end = max(end, min(analysis.duration, times[-1] + 2 * period))
        out.append((index, times[0], end, times))
    return out


def _beat_in_bar(times: list[float], start: float) -> int:
    """Index of the beat a chord starts on (the nearest within tolerance, else the last before)."""
    nearest = min(range(len(times)), key=lambda i: abs(times[i] - start))
    if abs(times[nearest] - start) <= BEAT_TOLERANCE:
        return nearest
    before = [i for i, t in enumerate(times) if t <= start]
    return before[-1] if before else 0


def arrange(analysis: Analysis, arrangement: Arrangement | None = None) -> ArrangedSheet:
    """The renderable sheet for this analysis under these options."""
    arrangement = arrangement or Arrangement()
    warnings: list[str] = []

    if analysis.key is not None:
        original_tonic, mode = analysis.key.tonic, analysis.key.mode
        original_name: str | None = key_name(original_tonic, mode)
    else:
        original_tonic, mode, original_name = 0, Mode.MAJOR, None
        warnings.append("分析没有给出调性，按 C 大调编配")
    sung = sung_key(original_tonic, arrangement.transpose)

    options = fingering_options(sung, mode)
    fingering = arrangement.fingering_key
    if fingering is not None and fingering not in FINGERING_CANDIDATES[mode]:
        warnings.append(f"指法调 {key_name(fingering, mode)} 不在候选里，改用默认")
        fingering = None
    if fingering is None:
        fingering = default_fingering(sung, mode).tonic
    capo = capo_for(sung, fingering)
    choices = tuple(
        FingeringChoice(
            tonic=o.tonic,
            name=o.name,
            capo=o.capo,
            recommended=o.recommended,
            selected=o.tonic == fingering,
        )
        for o in options
    )

    meter = (
        (analysis.time_signature.numerator, analysis.time_signature.denominator)
        if analysis.time_signature is not None
        else DEFAULT_METER
    )
    if analysis.time_signature is None and analysis.beats:
        warnings.append("分析没有给出拍号，按 4/4 编配")
    bpm = analysis.tempo.bpm if analysis.tempo is not None else None
    strum, strum_warning = strums.resolve(arrangement.strum_pattern, meter, bpm)
    if strum_warning:
        warnings.append(strum_warning)
    auto = strums.recommend(meter, bpm)
    strum_options = tuple(
        StrumOption(
            id=p.id,
            name=p.name,
            mnemonic=p.mnemonic,
            kind=p.kind,
            recommended=auto is not None and p.id == auto.id,
        )
        for p in strums.patterns_for(meter)
    )

    # Chord names and shapes: written in the fingering key, one shape per name.
    shapes: dict[str, Shape] = {}
    named: list[tuple[float, float, str, str, float]] = []  # start, end, name, original, confidence
    previous_base: int | None = None
    transpose = arrangement.transpose
    for chord in analysis.chords:
        # The chord sounds `transpose` semitones from the recording once the
        # song is sung in the new key; the shape is written below the capo.
        root = written_root((chord.root + transpose) % 12, capo)
        bass = None if chord.bass is None else written_root((chord.bass + transpose) % 12, capo)
        original = chord_name(root, chord.quality, bass, fingering, mode)
        quality, shown_bass = (
            simplify(chord.quality, bass) if arrangement.simplify_chords else (chord.quality, bass)
        )
        name = chord_name(root, quality, shown_bass, fingering, mode)
        if name not in shapes:
            shapes[name] = choose_shape(root, quality, previous_base)
        previous_base = shapes[name].base_fret
        named.append((chord.start, chord.end, name, original, chord.confidence))

    bars = []
    low = 0
    for index, start, end, times in _bars(analysis):
        in_bar = []
        for c_start, c_end, name, original, confidence in named:
            if c_start >= end or c_end <= start:
                continue
            beat = _beat_in_bar(times, c_start) if c_start >= start else 0
            in_bar.append(
                BarChord(
                    beat=beat,
                    name=name,
                    original_name=original,
                    shape=name,
                    confidence=confidence,
                    low_confidence=confidence < LOW_CONFIDENCE,
                )
            )
        bars.append(Bar(index=index, start=start, end=end, beats=len(times), chords=tuple(in_bar)))
    low = sum(1 for _, _, _, _, c in named if c < LOW_CONFIDENCE)
    if analysis.chords and not analysis.beats:
        warnings.append("分析没有拍网格，和弦无法排进小节")

    sung_name = key_name(sung, mode)
    fingering_name = key_name(fingering, mode)
    capo_text = f"夹 {capo} 品" if capo else "无变调夹"
    header = Header(
        title=analysis.title,
        artist=analysis.artist,
        original_key=original_name,
        sung_key=sung_name,
        fingering_key=fingering_name,
        capo=capo,
        transpose=arrangement.transpose,
        bpm=bpm,
        time_signature=meter,
        jianpu_key="1=" + key_name(relative_major(sung, mode), Mode.MAJOR),
        text=f"原调 {original_name or '?'} · 演唱调 {sung_name} · "
        f"{fingering_name} 调指法 {capo_text}",
    )
    return ArrangedSheet(
        header=header,
        fingering_options=choices,
        strum=strum,
        strum_options=strum_options,
        shapes=shapes,
        bars=tuple(bars),
        notes=analysis.notes,
        lyrics=analysis.lyrics,
        low_confidence_count=low,
        warnings=tuple(warnings),
    )
