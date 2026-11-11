# OpenBapo

[![CI](https://github.com/Ariza-Sun/OpenBapo/actions/workflows/ci.yml/badge.svg)](https://github.com/Ariza-Sun/OpenBapo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Turn a song into a guitar play-along sheet.** Upload a recording and get chord diagrams, a strumming tab, the vocal melody in numbered notation (jianpu) and the lyrics on one sheet. Change the singing key, the fingering key or the strum pattern, and export to PNG, PDF or MusicXML. Works for Chinese and English songs, delivered as a Discord bot.

> ⚠️ **Work in progress.** What works today is the command line: `analyze` finds the beats, tempo, chords and key of a recording; `arrange` turns that into the sheet's content (capo, chord names and shapes in a fingering key of your choice, a strum pattern, the header); `render` draws it as PNG / PDF and exports MusicXML. The melody line, lyrics and the Discord bot are still to come; see [Progress](#progress).

## Why

Tools that transcribe chords from audio (Capo, Chordify, Moises and others) are closed and paid. Guitar play-along sheets on Chinese tab sites are arranged by hand, so less popular songs and most English songs have none. OpenBapo aims to be an open, self-hostable alternative that produces the numbered-notation play-along sheet Chinese guitarists are used to.

## Progress

| Step | What | Status |
|---|---|---|
| 0 | Project skeleton, data model, CLI | ✅ Done |
| 1 | Source separation, beats, tempo, time signature | ✅ Done |
| 2 | Chords and key | ✅ Done |
| 3 | Arranging: capo, fingering key, chord simplification, strum patterns | ✅ Done |
| 4 | Rendering and export: chord diagrams, tab, jianpu; PNG / PDF / MusicXML | ✅ Done (jianpu notes and lyrics arrive with steps 5 and 7) |
| 5 | Lyrics: alignment of pasted lyrics, and recognition | ⏳ Next |
| 6 | Discord bot | — |
| 7 | Vocal melody to jianpu | — |

The full plan is in the [roadmap](docs/ROADMAP.md) and the [requirements](docs/REQUIREMENTS.md) (both in Chinese).

## Quick start

You need Python 3.11 or newer and `ffmpeg`. A GPU makes the models faster, but a CPU works.

```bash
git clone https://github.com/Ariza-Sun/OpenBapo.git
cd OpenBapo
python -m venv .venv && source .venv/bin/activate

# Install torch and torchaudio for your CUDA version first (see pytorch.org); for CPU only:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[analysis]"

openbapo analyze your-song.mp3 --out analysis.json
```

Model weights are downloaded on first use. No audio at hand? Synthesise a C–F–G7–Am progression with the bundled script and analyse that:

```bash
python scripts/make_tone.py demo.wav --progression 120 --seconds 16
openbapo analyze demo.wav --out demo.json
```

The output holds the beats and downbeats, tempo, time signature, key, and a chord sequence aligned to the beats. Roots are pitch classes (0 = C … 11 = B); each chord also has a quality, an inversion bass and a confidence. An excerpt of `demo.json`:

```json
{
  "beats": [{"time": 0.0, "is_downbeat": true, "bar_index": 0, "beat_in_bar": 1}, ...],
  "tempo": {"bpm": 120.0},
  "time_signature": {"numerator": 4, "denominator": 4},
  "key": {"tonic": 9, "mode": "minor", "confidence": 0.004},
  "chords": [
    {"start": 0.0, "end": 2.0, "root": 0, "quality": "maj", "bass": null, "confidence": 0.96},
    {"start": 2.0, "end": 4.0, "root": 5, "quality": "maj", "bass": null, "confidence": 0.92},
    ...
  ]
}
```

Every loop of this progression ends on Am, so the key comes out as A minor, the relative minor of C major. The very low confidence says the two were nearly tied.

Useful options: `--device cpu|cuda`, `--timing -` to print the time spent in each stage, and `--chord-experts lvchordia` to use a single chord model.

Then arrange it for the guitar:

```bash
openbapo arrange demo.json --out sheet.json                      # easiest fingering key, auto strum
openbapo arrange demo.json --out sheet.json --transpose -2 --fingering-key G --strum folk
```

The sheet JSON holds the header (`原调 C · 演唱调 bB · G 调指法 夹 3 品`), every chord per bar as written in the fingering key with its diagram, the strum pattern with its mnemonic (`下 下上 上下上`), the list of fingering keys with their capos, and which chords the analysis was unsure about. Transposing changes the capo, not the chord names; changing the fingering key rewrites the names but not the sound.

Finally draw it:

```bash
pip install -e ".[render]"        # cairosvg + pypdf; needs libcairo (Debian / Ubuntu: apt install libcairo2)
openbapo render sheet.json --png sheet.png --pdf sheet.pdf --musicxml sheet.musicxml
```

`sheet.png` is the first page (chord diagrams, the strum pattern on a tab staff, the jianpu and lyric lines, the header); the PDF has every page; the MusicXML opens in MuseScore or Guitar Pro for fine editing. PNG text uses the system's CJK font (`fonts-noto-cjk` on Linux).

## How it works

```
audio ─decode─▶ source separation (Demucs) ─▶ beats / downbeats / tempo (Beat This!)
                          │
                          ▼
          chords: BTC + lv-chordia + chroma templates + bass line,
          fused on the beat grid and smoothed with Viterbi
                          │
                          ▼
          key: chord statistics cross-checked with chroma templates
                          │
                          ▼
          arranging → rendering → Discord   (in progress)
```

- **Decisions on the beat.** Two deep-learning chord models and two rule-based experts each give a posterior per beat. A fusion layer votes on the beat grid, then Viterbi smoothing makes chord changes cheap on downbeats and expensive mid-bar, so every chord starts and ends on a beat.
- **Stems stay internal.** Separated stems are an intermediate product, deleted after analysis; they are never stored or returned.
- **Analysis apart from arranging.** Model inference runs once. The resulting JSON can be re-arranged and re-rendered any number of times, which is plain computation taking seconds.

## Accuracy

Every number comes from a script in this repository; see [docs/benchmarks](docs/benchmarks/README.md). Measured on an RTX 3090.

| Task | Data | Result |
|---|---|---|
| Beats | GTZAN (998 tracks) | beat F-measure **0.891**, downbeat 0.786 |
| Chords (major/minor) | 220 Chinese and English pop songs: guitar sheets rendered to audio | **0.974** |
| Chords (major/minor) | AAM, synthetic full-band mixes (500 tracks) | **0.947** |
| Chords (major/minor) | GuitarSet, acoustic guitar comping (90 excerpts) | 0.775 |
| Key | the same 220 guitar sheets (relative major/minor counted as correct) | **0.936** |
| Key | GuitarSet comping / AAM single-key tracks | 0.789 / 0.931 |
| Speed | 4-minute song: separation, beats, chords and key | 13.4 s |
| Speed | 4-minute song: arranging and rendering (SVG, PNG, PDF, MusicXML, CPU) | 2.1 s |

Two caveats:

- The 220-song set is rendered from guitar sheets, with no vocals and no real mix. Treat its scores as an upper bound for real recordings.
- The GuitarSet gap is mostly funk and jazz comping, a known weak spot of the open chord models available today.

## Repository layout

```
src/openbapo/
  model/      data model: Beat, Tempo, Key, Chord, Analysis, Arrangement; JSON round-trip
  analysis/   analysis pipeline: decoding, separation, beats, chords and key (chords/)
  arrange/    arranging: keys and capo, chord names, shape and strum libraries (data/), the sheet
  render/     rendering: page layout, SVG drawing, PNG / PDF export, MusicXML
  sheet/      text sheet format, renderer and sheet reader: the evaluation sample generator
scripts/      evaluation, timing, sheet reading and rendering scripts
tests/        unit tests; tests marked `models` load the real models
docs/         requirements, roadmap, per-step plans and benchmark results (in Chinese)
```

## Development

```bash
pip install -e ".[dev]"
ruff format --check . && ruff check . && mypy && pytest   # must pass before a push
pip install -e ".[dev,analysis]" && pytest -m models      # end-to-end tests with the real models
```

- torch and torchaudio must come from the same build. lv-chordia declares `torch>=2.13`, so on machines limited to an older torch, pip will replace it; [docs/benchmarks/README.md](docs/benchmarks/README.md) shows how to avoid that.
- Design decisions and trade-offs are recorded in each step's plan under [docs/plans/](docs/plans/). Read the plan for the step you are working on first.

Issues and pull requests are welcome. Code, identifiers and commit messages are in English; the design documents are in Chinese.

## Models and licences

The code in this repository is released under the [MIT licence](LICENSE). Model weights are not distributed with it; each is downloaded from its source on first use.

| Use | Model | Weight licence |
|---|---|---|
| Source separation | Demucs `htdemucs` | not stated; **treat as non-commercial** |
| Beats | Beat This! | MIT |
| Chords | BTC | not stated separately (shipped in an MIT code repository) |
| Chords | lv-chordia | not stated separately (shipped in an MIT code repository) |

Details and training data are in section 2.1 of [docs/plans/backend.md](docs/plans/backend.md). Check each licence yourself before commercial use or a public deployment. OpenBapo outputs sheets only; it never returns separated, re-pitched or re-timed audio.
