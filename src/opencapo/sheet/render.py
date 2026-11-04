"""Render a Sheet to audio plus a timed reference, for testing the analysis on a song book.

Two accompaniments: "strum" (default) plays a Karplus-Strong plucked
string per chord tone in a fixed strumming pattern, the guitar the sheet is
written for; "piano" strikes each chord as a block of decaying harmonic
tones on the strong beats. The melody is a piano tone (a sustained vibrato
tone is available as "voice"). An optional bass plays roots on the strong beats. Nothing is
sampled or downloaded, so the output is deterministic. The reference
carries the sounding chords (shapes transposed by the capo), the sung key,
the beat grid and the melody, in the sheet's own timing.
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


def pluck(midi: float, seconds: float, *, seed: int, decay: float = 0.998) -> NDArray[np.float64]:
    """Karplus-Strong: a softened noise burst circulating through a short averaging loop.

    The burst is low-passed first so the attack is a nylon-string thump
    rather than a click, and the loop decays slowly enough to ring for a bar.
    """
    import numpy as np

    n = max(1, int(seconds * SAMPLE_RATE))
    period = max(2, round(SAMPLE_RATE / midi_hz(midi)))
    rng = np.random.default_rng(seed)
    burst = rng.uniform(-1.0, 1.0, period)
    kernel = np.hanning(7)
    kernel /= kernel.sum()
    block = np.convolve(np.concatenate([burst, burst[:10]]), kernel, mode="valid")[:period]
    block = block / (np.abs(block).max() or 1.0)
    blocks = [block]
    while sum(len(b) for b in blocks) < n:
        prev = blocks[-1]
        # A three-point loop filter damps the upper partials faster than the
        # fundamental: the string mellows as it rings instead of jangling.
        left = np.roll(prev, 1)
        right = np.roll(prev, -1)
        nxt = decay * (0.15 * left + 0.7 * prev + 0.15 * right)
        blocks.append(nxt)
    out = np.concatenate(blocks)[:n]
    body = np.hanning(9)
    body /= body.sum()
    return np.asarray(np.convolve(out, body, mode="same"))


TUNING = (40, 45, 50, 55, 59, 64)  # E2 A2 D3 G3 B3 E4, low string first
X = None
# Open-position shapes as frets per string (None = not played), keyed by the
# shape's written root and quality: what the sheet's diagrams show.
SHAPES: dict[tuple[int, str], tuple[int | None, ...]] = {
    (0, "maj"): (X, 3, 2, 0, 1, 0),  # C
    (0, "maj7"): (X, 3, 2, 0, 0, 0),  # Cmaj7
    (0, "7"): (X, 3, 2, 3, 1, 0),  # C7
    (2, "maj"): (X, X, 0, 2, 3, 2),  # D
    (2, "min"): (X, X, 0, 2, 3, 1),  # Dm
    (2, "7"): (X, X, 0, 2, 1, 2),  # D7
    (2, "min7"): (X, X, 0, 2, 1, 1),  # Dm7
    (2, "sus2"): (X, X, 0, 2, 3, 0),  # Dsus2
    (2, "sus4"): (X, X, 0, 2, 3, 3),  # Dsus4
    (4, "maj"): (0, 2, 2, 1, 0, 0),  # E
    (4, "min"): (0, 2, 2, 0, 0, 0),  # Em
    (4, "7"): (0, 2, 0, 1, 0, 0),  # E7
    (4, "min7"): (0, 2, 0, 0, 0, 0),  # Em7
    (5, "maj"): (1, 3, 3, 2, 1, 1),  # F
    (5, "maj7"): (X, X, 3, 2, 1, 0),  # Fmaj7
    (7, "maj"): (3, 2, 0, 0, 0, 3),  # G
    (7, "7"): (3, 2, 0, 0, 0, 1),  # G7
    (9, "maj"): (X, 0, 2, 2, 2, 0),  # A
    (9, "min"): (X, 0, 2, 2, 1, 0),  # Am
    (9, "7"): (X, 0, 2, 0, 2, 0),  # A7
    (9, "min7"): (X, 0, 2, 0, 1, 0),  # Am7
    (9, "sus2"): (X, 0, 2, 2, 0, 0),  # Asus2
    (9, "sus4"): (X, 0, 2, 2, 3, 0),  # Asus4
    (11, "7"): (X, 2, 1, 2, 0, 2),  # B7
    (11, "min"): (X, 2, 4, 4, 3, 2),  # Bm (barre)
}
# Movable barre shapes (E-form on the low string, A-form on the fifth) for the rest.
BARRE_E: dict[str, tuple[int | None, ...]] = {
    "maj": (0, 2, 2, 1, 0, 0),
    "min": (0, 2, 2, 0, 0, 0),
    "7": (0, 2, 0, 1, 0, 0),
    "min7": (0, 2, 0, 0, 0, 0),
    "maj7": (0, X, 1, 1, 0, X),
}
BARRE_A: dict[str, tuple[int | None, ...]] = {
    "maj": (X, 0, 2, 2, 2, 0),
    "min": (X, 0, 2, 2, 1, 0),
    "7": (X, 0, 2, 0, 2, 0),
    "min7": (X, 0, 2, 0, 1, 0),
    "maj7": (X, 0, 2, 1, 2, 0),
}


def shape_frets(root: int, quality: str) -> tuple[int | None, ...]:
    """Frets of the written chord shape: the open shape if there is one, else a barre."""
    if (root, quality) in SHAPES:
        return SHAPES[root, quality]
    family = quality if quality in BARRE_E else {"maj6": "maj", "min6": "min"}.get(quality, "maj")
    e_fret, a_fret = (root - 4) % 12, (root - 9) % 12
    if a_fret < e_fret:  # the lower position is what a player picks
        return tuple(None if f is None else f + a_fret for f in BARRE_A[family])
    return tuple(None if f is None else f + e_fret for f in BARRE_E[family])


def voicing(root: int, quality: str, capo: int, bass: int | None = None) -> list[int]:
    """Sounding pitches of the written shape under the capo, low string first.

    A slash chord puts its bass note on the lowest sounding string; the
    other strings keep the shape.
    """
    pitches = [
        open_pitch + fret + capo
        for open_pitch, fret in zip(TUNING, shape_frets(root, quality), strict=True)
        if fret is not None
    ]
    if bass is not None and pitches and pitches[0] % 12 != (bass + capo) % 12:
        low = TUNING[0] + capo + (bass - TUNING[0]) % 12
        pitches = [low, *[p for p in pitches if p > low]]
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


def piano_tone(midi: float, seconds: float) -> NDArray[np.float64]:
    """A struck string: harmonics that decay faster the higher they are, a quick attack."""
    import numpy as np

    n = max(1, int(seconds * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    f = midi_hz(midi)
    tone = np.zeros(n)
    # Two slightly detuned strings per note, as a piano's unison strings beat.
    for detune in (-0.0008, 0.0008):
        for k in range(1, 10):
            partial = k * f * (1 + detune) * (1 + 0.0003 * k * k)  # slight inharmonicity
            if partial > SAMPLE_RATE / 2:
                break
            amplitude = 1.0 / k**1.5
            decay = 0.5 + 0.0005 * partial + 0.25 * k
            tone += amplitude * np.exp(-decay * t) * np.sin(2 * np.pi * partial * t)
    attack = np.minimum(1.0, t / 0.003)
    release = np.clip((seconds - t) / 0.08, 0.0, 1.0)
    return np.asarray(tone * attack * release / 4.0)


def bass_tone(midi: float, seconds: float) -> NDArray[np.float64]:
    import numpy as np

    n = max(1, int(seconds * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    env = np.exp(-3.0 * t) * np.minimum(1.0, t / 0.01)
    f = midi_hz(midi)
    return np.asarray((np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * env)


def render(
    sheet: Sheet,
    *,
    bass: bool = False,
    melody: bool = True,
    accompaniment: str = "strum",
    melody_voice: str = "piano",
    seed: int = 0,
) -> Rendered:
    import numpy as np

    if accompaniment not in ("piano", "strum"):
        raise ValueError("accompaniment must be 'piano' or 'strum'")
    if melody_voice not in ("piano", "voice"):
        raise ValueError("melody_voice must be 'piano' or 'voice'")
    melody_fn = piano_tone if melody_voice == "piano" else melody_tone

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

    def add(
        track: NDArray[np.float64], start: float, signal: NDArray[np.float64], replace: bool = False
    ) -> None:
        i = int(start * SAMPLE_RATE)
        j = min(total, i + len(signal))
        if replace:
            track[i:j] = signal[: j - i]
        else:
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
            spans.append((start, end, root, span.quality, chord_bass, span.root, span.bass))
            label = format_label(root, span.quality, chord_bass if chord_bass != root else None)
            if ref_chords and ref_chords[-1][2] == label and ref_chords[-1][1] == start:
                ref_chords[-1][1] = end
            else:
                ref_chords.append([start, end, label])
            position += span.beats
        if accompaniment == "strum":
            hits = pattern
        else:
            # Block chords on the strong beats and wherever the chord changes.
            strong = {0.0, 2.0} if sheet.beats_per_bar == 4 else {0.0}
            changes = {round((s[0] - bar_start) / spb, 6) for s in spans}
            hits = tuple((offset, True) for offset in sorted(strong | changes))
        for offset, down in hits:
            t = bar_start + offset * spb
            hit = next((s for s in spans if s[0] - 1e-9 <= t < s[1]), None)
            if hit is None:
                continue
            _, end, root, quality, chord_bass, shape_root, shape_bass = hit
            pitches = voicing(shape_root, quality.value, sheet.capo, shape_bass)
            next_hit = next((bar_start + o * spb for o, _ in hits if o > offset), end)
            ring = min(end - t, sheet.beats_per_bar * spb, next_hit - t + 0.05)
            if accompaniment == "strum":
                if not down:
                    pitches = pitches[::-1][:4]
                gain = 0.3 if down else 0.22
                for k, midi in enumerate(pitches):
                    plucks += 1
                    add(
                        guitar, t + k * STRING_STAGGER, gain * pluck(midi, ring, seed=seed + plucks)
                    )
            else:
                for midi in pitches:
                    add(guitar, t, 0.22 * piano_tone(midi, ring))
            if bass and down and offset in (0.0, 2.0):
                low = LOWEST_MIDI - 12 + (root - LOWEST_MIDI) % 12
                add(bass_track, t, 0.5 * bass_tone(low, min(2 * spb, end - t)))

        # Melody: degrees in the sung key.
        position = Fraction(bar_index * sheet.beats_per_bar)
        for note in bar.notes:
            start, end = seconds(position), seconds(position + note.beats)
            pitch = degree_midi(sheet, note)
            if note.tied and ref_melody and ref_melody[-1][2] == pitch:
                # Held over the bar line: one reference note, one longer tone.
                held_start = float(ref_melody[-1][0])  # type: ignore[arg-type]
                ref_melody[-1][1] = end
                if melody and pitch is not None:
                    add(voice, held_start, 0.3 * melody_fn(pitch, end - held_start), replace=True)
            else:
                ref_melody.append([start, end, pitch])
                if melody and pitch is not None:
                    add(voice, start, 0.3 * melody_fn(pitch, end - start))
            position += note.beats

    if not bass:
        bass_track[:] = 0.0
    left = 0.6 * guitar + 0.5 * bass_track + 0.55 * voice
    right = 0.4 * guitar + 0.5 * bass_track + 0.55 * voice
    stereo = np.stack([room(left), room(right)])
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


def room(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    """A few early reflections so the dry tones do not sit in a vacuum."""
    import numpy as np

    out = signal.copy()
    for delay_ms, gain in ((23.0, 0.25), (37.0, 0.18), (53.0, 0.12), (79.0, 0.08)):
        d = int(delay_ms * SAMPLE_RATE / 1000)
        out[d:] += gain * signal[:-d]
    return np.asarray(out)


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
