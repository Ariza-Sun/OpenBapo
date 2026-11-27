"""Write a short synthetic WAV for smoke-testing the CLI.

Usage:
    python scripts/make_tone.py OUT.wav [--seconds 3]            # a 440 Hz sine
    python scripts/make_tone.py OUT.wav --click 120 [--seconds 12]  # a click track
    python scripts/make_tone.py OUT.wav --progression 120         # C F G7 Am + bass + clicks

The click track plays a short tone on every beat at the given BPM, with a
louder, lower click on every fourth beat, so a beat tracker should report the
BPM and 4 beats per bar. The progression adds one sine-wave chord per bar
(C, F, G7, Am, repeating) with a bass note on the root, so chord experts
should report those chords in C major. The repository ships no real
recordings; tests and CI generate this instead.
"""

import argparse
import math
import struct
import wave

SAMPLE_RATE = 44100
FREQUENCY = 440.0


def sine(seconds: float) -> list[float]:
    n = int(seconds * SAMPLE_RATE)
    return [0.3 * math.sin(2 * math.pi * FREQUENCY * i / SAMPLE_RATE) for i in range(n)]


def click_track(seconds: float, bpm: float, beats_per_bar: int = 4) -> list[float]:
    n = int(seconds * SAMPLE_RATE)
    samples = [0.0] * n
    period = 60.0 / bpm
    click_length = int(0.04 * SAMPLE_RATE)
    beat = 0
    while beat * period < seconds:
        start = int(beat * period * SAMPLE_RATE)
        accent = beat % beats_per_bar == 0
        freq, gain = (1000.0, 0.9) if accent else (2000.0, 0.5)
        for i in range(click_length):
            if start + i >= n:
                break
            envelope = 1.0 - i / click_length
            samples[start + i] = gain * envelope * math.sin(2 * math.pi * freq * i / SAMPLE_RATE)
        beat += 1
    return samples


PROGRESSION = [  # (chord tones as MIDI, bass as MIDI): C, F, G7, Am
    ([60, 64, 67], 36),
    ([65, 69, 72], 41),
    ([67, 71, 74, 77], 43),
    ([69, 72, 76], 45),
]


def midi_hz(midi: int) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def chord_progression(seconds: float, bpm: float, beats_per_bar: int = 4) -> list[float]:
    """Sine chords (with three harmonics) and a bass note, one chord per bar, over clicks."""
    samples = click_track(seconds, bpm, beats_per_bar)
    bar_seconds = beats_per_bar * 60.0 / bpm
    n = len(samples)
    bar = 0
    while bar * bar_seconds < seconds:
        tones, bass = PROGRESSION[bar % len(PROGRESSION)]
        start = int(bar * bar_seconds * SAMPLE_RATE)
        end = min(n, int((bar + 1) * bar_seconds * SAMPLE_RATE))
        for i in range(start, end):
            t = (i - start) / SAMPLE_RATE
            value = 0.0
            for midi in tones:
                for h in (1, 2, 3):
                    value += (0.06 / h) * math.sin(2 * math.pi * midi_hz(midi) * h * t)
            for h in (1, 2, 3):
                value += (0.12 / h) * math.sin(2 * math.pi * midi_hz(bass) * h * t)
            samples[i] = max(-1.0, min(1.0, samples[i] * 0.5 + value))
        bar += 1
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--click", type=float, metavar="BPM", help="write a click track instead")
    parser.add_argument(
        "--progression", type=float, metavar="BPM", help="write a chord progression with clicks"
    )
    args = parser.parse_args()

    if args.progression:
        samples = chord_progression(args.seconds or 16.0, args.progression)
    elif args.click:
        samples = click_track(args.seconds or 12.0, args.click)
    else:
        samples = sine(args.seconds or 3.0)
    frames = b"".join(struct.pack("<h", int(32767 * s)) for s in samples)
    with wave.open(args.out, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(frames)


if __name__ == "__main__":
    main()
