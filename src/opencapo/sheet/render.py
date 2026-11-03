"""Render a Sheet to audio plus a timed reference, for testing the analysis on a song book.

The guitar is a Karplus-Strong plucked string per chord tone, strummed in a
fixed pattern; the melody is a sustained harmonic tone with vibrato (a
stand-in for the voice); an optional bass plays roots on the strong beats.
Nothing is sampled or downloaded, so the output is deterministic. The
reference carries the sounding chords (shapes transposed by the capo), the
sung key, the beat grid and the melody, in the sheet's own timing.
"""

from __future__ import annotations

import json
import struct
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

from opencapo.analysis.chords.vocab import format_label
from opencapo.sheet.format import Sheet, degree_midi

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

SAMPLE_RATE = 44100
# Strum pattern per bar as (beat offset, down?) for 4/4 (D DU UDU without the
# last up) and 3/4 (D DU DU); other metres strum every beat downward.
PATTERNS = {
    4: ((0.0, True), (1.0, True), (1.5, False), (2.5, False), (3.0, True), (3.5, False)),
    3: ((0.0, True), (1.0, True), (1.5, False), (2.0, True), (2.5, False)),
}
STRING_STAGGER = 0.012  # seconds between strings in a strum
LOWEST_MIDI = 40  # E2, the guitar's low string


@dataclass(frozen=True)
class Rendered:
    samples: NDArray[np.float32]  # (2, n)
    sample_rate: int
    reference: dict  # type: ignore[type-arg]


def midi_hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def pluck(midi: float, seconds: float, *, seed: int, decay: float = 0.996) -> NDArray[np.float64]:
    """Karplus-Strong: a noise burst circulating through a short averaging loop."""
    import numpy as np

    n = max(1, int(seconds * SAMPLE_RATE))
    period = max(2, round(SAMPLE_RATE / midi_hz(midi)))
    rng = np.random.default_rng(seed)
    block = rng.uniform(-1.0, 1.0, period)
    blocks = [block]
    while sum(len(b) for b in blocks) < n:
        prev = blocks[-1]
        nxt = np.empty(period)
        nxt[:-1] = 0.5 * decay * (prev[:-1] + prev[1:])
        nxt[-1] = 0.5 * decay * (prev[-1] + nxt[0])
        blocks.append(nxt)
    return np.asarray(np.concatenate(blocks)[:n])


def voicing(root: int, quality_tones: frozenset[int], bass: int | None) -> list[int]:
    """Guitar-like voicing: bass (or root) low, then chord tones climbing over ~2 octaves."""
    lowest_pc = bass if bass is not None else root
    low = LOWEST_MIDI + (lowest_pc - LOWEST_MIDI) % 12
    pitches = [low]
    candidate = low + 1
    while len(pitches) < 5 and candidate < low + 24:
        if (candidate % 12 in quality_tones and candidate % 12 != lowest_pc) or (
            candidate % 12 == lowest_pc and candidate - low >= 12
        ):
            pitches.append(candidate)
        candidate += 1
    return pitches


def melody_tone(midi: float, seconds: float) -> NDArray[np.float64]:
    """A sustained tone with a few harmonics, slow vibrato and a soft envelope."""
    import numpy as np

    n = max(1, int(seconds * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    vibrato = np.where(t > 0.15, 2 ** ((15 / 1200) * np.sin(2 * np.pi * 5.5 * t)), 1.0)
    phase = 2 * np.pi * np.cumsum(midi_hz(midi) * vibrato) / SAMPLE_RATE
    tone = sum(a * np.sin(k * phase) for k, a in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12)))
    attack = np.minimum(1.0, t / 0.02)
    release = np.minimum(1.0, (seconds - t) / 0.06)
    return np.asarray(tone * attack * np.clip(release, 0.0, 1.0) / 1.9)


def bass_tone(midi: float, seconds: float) -> NDArray[np.float64]:
    import numpy as np

    n = max(1, int(seconds * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    env = np.exp(-3.0 * t) * np.minimum(1.0, t / 0.01)
    f = midi_hz(midi)
    return np.asarray((np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * env)


def render(sheet: Sheet, *, bass: bool = True, melody: bool = True, seed: int = 0) -> Rendered:
    import numpy as np

    from opencapo.analysis.chords.vocab import chord_tones

    spb = sheet.seconds_per_beat
    total = int(sheet.duration * SAMPLE_RATE) + SAMPLE_RATE  # a second of tail
    guitar = np.zeros(total)
    bass_track = np.zeros(total)
    voice = np.zeros(total)
    ref_chords: list[list[object]] = []
    ref_melody: list[list[object]] = []
    beats: list[float] = []
    downbeats: list[float] = []
    pattern = PATTERNS.get(sheet.beats_per_bar) or tuple(
        (float(b), True) for b in range(sheet.beats_per_bar)
    )

    def seconds(beat_position: Fraction) -> float:
        return round(float(beat_position) * spb, 6)

    def add(track: NDArray[np.float64], start: float, signal: NDArray[np.float64]) -> None:
        i = int(start * SAMPLE_RATE)
        j = min(total, i + len(signal))
        track[i:j] += signal[: j - i]

    plucks = 0
    for bar_index, bar in enumerate(sheet.bars):
        bar_start = seconds(Fraction(bar_index * sheet.beats_per_bar))
        for b in range(sheet.beats_per_bar):
            beats.append(seconds(Fraction(bar_index * sheet.beats_per_bar + b)))
        downbeats.append(bar_start)

        # Chords: each span sounds the capo-transposed shape. Times are derived
        # from exact beat positions so consecutive spans never overlap.
        position = Fraction(bar_index * sheet.beats_per_bar)
        spans = []
        for span in bar.chords:
            start, end = seconds(position), seconds(position + span.beats)
            root = (span.root + sheet.capo) % 12
            chord_bass = None if span.bass is None else (span.bass + sheet.capo) % 12
            spans.append((start, end, root, span.quality, chord_bass))
            label = format_label(root, span.quality, chord_bass if chord_bass != root else None)
            if ref_chords and ref_chords[-1][2] == label and ref_chords[-1][1] == start:
                ref_chords[-1][1] = end
            else:
                ref_chords.append([start, end, label])
            position += span.beats
        for offset, down in pattern:
            t = bar_start + offset * spb
            hit = next((s for s in spans if s[0] - 1e-9 <= t < s[1]), None)
            if hit is None:
                continue
            _, end, root, quality, chord_bass = hit
            pitches = voicing(root, chord_tones(root, quality), chord_bass)
            if not down:
                pitches = pitches[::-1][:4]
            # ring until the next strum of this chord, at most a bar
            ring = min(end - t, sheet.beats_per_bar * spb)
            for k, midi in enumerate(pitches):
                plucks += 1
                add(guitar, t + k * STRING_STAGGER, 0.35 * pluck(midi, ring, seed=seed + plucks))
            if bass and down and offset in (0.0, 2.0):
                low = LOWEST_MIDI - 12 + (root - LOWEST_MIDI) % 12
                add(bass_track, t, 0.5 * bass_tone(low, min(2 * spb, end - t)))

        # Melody: degrees in the sung key.
        position = Fraction(bar_index * sheet.beats_per_bar)
        for note in bar.notes:
            start, end = seconds(position), seconds(position + note.beats)
            pitch = degree_midi(sheet, note)
            ref_melody.append([start, end, pitch])
            if melody and pitch is not None:
                add(voice, start, 0.3 * melody_tone(pitch, end - start))
            position += note.beats

    if not bass:
        bass_track[:] = 0.0
    left = 0.6 * guitar + 0.5 * bass_track + 0.45 * voice
    right = 0.4 * guitar + 0.5 * bass_track + 0.45 * voice
    stereo = np.stack([left, right])
    peak = float(np.abs(stereo).max())
    if peak > 0:
        stereo *= 0.9 / peak
    key_tonic = sheet.key
    reference = {
        "title": sheet.title,
        "key": f"{['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][key_tonic]}"
        f" {sheet.mode.value}",
        "fingering_key": sheet.fingering_key,
        "capo": sheet.capo,
        "bpm": sheet.bpm,
        "beats_per_bar": sheet.beats_per_bar,
        "duration": sheet.duration,
        "beats": beats,
        "downbeats": downbeats,
        "chords": ref_chords,
        "melody": ref_melody,
    }
    return Rendered(samples=stereo.astype(np.float32), sample_rate=SAMPLE_RATE, reference=reference)


def write_wav(path: Path, samples: NDArray[np.float32], sample_rate: int) -> None:
    import numpy as np

    pcm = np.clip(samples.T.reshape(-1) * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(samples.shape[0])
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(struct.pack(f"<{len(pcm)}h", *pcm.tolist()))


def write_reference(path: Path, reference: dict) -> None:  # type: ignore[type-arg]
    path.write_text(json.dumps(reference, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
