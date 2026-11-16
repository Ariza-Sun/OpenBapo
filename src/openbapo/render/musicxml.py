"""ArrangedSheet -> MusicXML (score-partwise 4.0) for editing in MuseScore and friends.

Two parts: a standard staff that carries the chord symbols (and, later,
the melody with its lyrics) and a TAB staff with the strum pattern's
strokes written as string / fret notes. Chord symbols are written as they
sound; the TAB part declares the capo as a transposition so an editor
shows the shapes' own names.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

from openbapo.arrange import ArrangedSheet, Shape
from openbapo.arrange.keys import parse_key_name
from openbapo.model import Mode
from openbapo.render.rhythm import (
    DIVISIONS,
    TabSymbol,
    beam_marks,
    quarter_length,
    tab_symbols,
    xml_duration,
)

TUNING = (40, 45, 50, 55, 59, 64)  # MIDI of the open strings, low to high
STEPS = ("C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B")
ALTERS = (0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0)
QUALITY_KIND = {
    "": "major", "m": "minor", "7": "dominant", "maj7": "major-seventh", "m7": "minor-seventh",
    "6": "major-sixth", "m6": "minor-sixth", "sus2": "suspended-second", "sus4": "suspended-fourth",
    "5": "power", "dim": "diminished", "aug": "augmented", "dim7": "diminished-seventh",
    "m7-5": "half-diminished",
}  # fmt: skip


def _sub(parent: ET.Element, tag: str, text: str | None = None, **attrs: str) -> ET.Element:
    el = ET.SubElement(parent, tag, attrs)
    if text is not None:
        el.text = text
    return el


def _pitch(el: ET.Element, midi: int) -> None:
    pitch = _sub(el, "pitch")
    _sub(pitch, "step", STEPS[midi % 12])
    if ALTERS[midi % 12]:
        _sub(pitch, "alter", "1")
    _sub(pitch, "octave", str(midi // 12 - 1))


def _harmony(
    measure: ET.Element,
    name: str,
    sounding_root: int,
    sounding_bass: int | None,
    offset: int,
) -> None:
    harmony = _sub(measure, "harmony")
    root = _sub(harmony, "root")
    _sub(root, "root-step", STEPS[sounding_root])
    if ALTERS[sounding_root]:
        _sub(root, "root-alter", "1")
    stem = name.split("/")[0]
    suffix = stem[2:] if stem[:1] in "#b" else stem[1:]
    kind = _sub(harmony, "kind", QUALITY_KIND.get(suffix, "other"))
    kind.set("text", suffix)
    if sounding_bass is not None:
        bass = _sub(harmony, "bass")
        _sub(bass, "bass-step", STEPS[sounding_bass])
        if ALTERS[sounding_bass]:
            _sub(bass, "bass-alter", "1")
    if offset:
        _sub(harmony, "offset", str(offset))


def _tab_note(
    measure: ET.Element,
    symbol: TabSymbol,
    shape: Shape,
    string_index: int,
    chord: bool,
    beams: tuple[str, ...],
    denominator: int,
) -> None:
    fret = shape.frets[string_index]
    assert fret is not None
    note = _sub(measure, "note")
    if chord:
        _sub(note, "chord")
    _pitch(note, TUNING[string_index] + fret)
    _sub(note, "duration", str(xml_duration(symbol.length, denominator)))
    for tied, kind in ((symbol.tie_stop, "stop"), (symbol.tie_start, "start")):
        if tied:
            _sub(note, "tie", type=kind)
    _sub(note, "voice", "1")
    _sub(note, "type", symbol.value.kind)
    for _ in range(symbol.value.dots):
        _sub(note, "dot")
    _sub(note, "stem", "down")
    if symbol.event is not None and symbol.event.stroke == "mute":
        _sub(note, "notehead", "x")
    if not chord:
        for number, mark in enumerate(beams, 1):
            _sub(note, "beam", mark, number=str(number))
    notations = _sub(note, "notations")
    for tied, kind in ((symbol.tie_stop, "stop"), (symbol.tie_start, "start")):
        if tied:
            _sub(notations, "tied", type=kind)
    technical = _sub(notations, "technical")
    _sub(technical, "string", str(6 - string_index))
    _sub(technical, "fret", str(fret))
    if not symbol.tie_stop and symbol.event is not None and symbol.event.stroke in ("down", "up"):
        _sub(notations, "arpeggiate", direction="up" if symbol.event.stroke == "down" else "down")


def _rest(measure: ET.Element, duration: int, symbol: TabSymbol | None = None) -> None:
    note = _sub(measure, "note")
    _sub(note, "rest", **({"measure": "yes"} if symbol is None else {}))
    _sub(note, "duration", str(duration))
    _sub(note, "voice", "1")
    if symbol is not None:
        _sub(note, "type", symbol.value.kind)
        for _ in range(symbol.value.dots):
            _sub(note, "dot")


def render_musicxml(sheet: ArrangedSheet) -> str:
    h = sheet.header
    root = ET.Element("score-partwise", version="4.0")
    work = _sub(root, "work")
    _sub(work, "work-title", h.title or "")
    if h.artist:
        _sub(_sub(root, "identification"), "creator", h.artist, type="artist")
    parts = _sub(root, "part-list")
    for pid, name in (("P1", "Chords"), ("P2", "Guitar")):
        sp = _sub(parts, "score-part", id=pid)
        _sub(sp, "part-name", name)

    sung_tonic, sung_mode = parse_key_name(h.sung_key)
    fifths = _fifths(sung_tonic, sung_mode)
    num, den = h.time_signature

    p1 = _sub(root, "part", id="P1")
    p2 = _sub(root, "part", id="P2")
    bars = sheet.bars or ()
    line_starts = _line_starts(sheet)
    for n, bar in enumerate(bars):
        m1 = _sub(p1, "measure", number=str(n + 1))
        m2 = _sub(p2, "measure", number=str(n + 1))
        if bar.beats < num:
            m1.set("implicit", "yes")
            m2.set("implicit", "yes")
        if n == 0:
            for measure, tab in ((m1, False), (m2, True)):
                attrs = _sub(measure, "attributes")
                _sub(attrs, "divisions", str(DIVISIONS))
                key = _sub(attrs, "key")
                _sub(key, "fifths", str(fifths))
                _sub(key, "mode", sung_mode.value)
                time = _sub(attrs, "time")
                _sub(time, "beats", str(num))
                _sub(time, "beat-type", str(den))
                clef = _sub(attrs, "clef")
                _sub(clef, "sign", "TAB" if tab else "G")
                _sub(clef, "line", "5" if tab else "2")
                if tab:
                    details = _sub(attrs, "staff-details")
                    _sub(details, "staff-lines", "6")
                    for line, midi in enumerate(TUNING, start=1):
                        tuning = _sub(details, "staff-tuning", line=str(line))
                        _sub(tuning, "tuning-step", STEPS[midi % 12])
                        if ALTERS[midi % 12]:
                            _sub(tuning, "tuning-alter", "1")
                        _sub(tuning, "tuning-octave", str(midi // 12 - 1))
                    if h.capo:
                        transpose = _sub(attrs, "transpose")
                        _sub(transpose, "diatonic", "0")
                        _sub(transpose, "chromatic", str(h.capo))
            if h.bpm:
                direction = _sub(m1, "direction", placement="above")
                _sub(
                    _sub(_sub(direction, "direction-type"), "metronome"),
                    "beat-unit",
                    {2: "half", 4: "quarter", 8: "eighth", 16: "16th"}[den],
                )
                _sub(direction.find("direction-type/metronome"), "per-minute", str(round(h.bpm)))  # type: ignore[arg-type]
                _sub(direction, "sound", tempo=str(float(quarter_length(h.bpm, den))))
        beats = bar.beats
        # Chord symbols as they sound: the shape's root plus the capo.
        for c in bar.chords:
            shape = sheet.shapes[c.shape]
            bass_name = c.name.split("/")[1] if "/" in c.name else None
            sounding_bass = None
            if bass_name:
                sounding_bass = (parse_key_name(bass_name)[0] + h.capo) % 12
            _harmony(
                m1, c.name, (shape.root + h.capo) % 12, sounding_bass, xml_duration(c.beat, den)
            )
        # Lyrics, one line per text direction at its first syllable, until the
        # melody (step 7) gives the syllables notes to sit under.
        for beat, text in line_starts.get(n, []):
            direction = _sub(m1, "direction", placement="below")
            _sub(_sub(direction, "direction-type"), "words", text)
            if beat:
                _sub(direction, "offset", str(round(quarter_length(beat, den) * DIVISIONS)))
        _rest(m1, xml_duration(beats, den))
        symbols = tab_symbols(bar, sheet.strum, sheet.shapes, h.time_signature)
        marks = beam_marks(symbols, h.time_signature)
        for index, symbol in enumerate(symbols):
            if symbol.rest:
                _rest(m2, xml_duration(symbol.length, den), symbol)
                continue
            assert symbol.shape is not None
            shape = sheet.shapes[symbol.shape]
            for k, string in enumerate(symbol.strings):
                _tab_note(m2, symbol, shape, string, k > 0, marks.get(index, ()), den)
    ET.indent(root)
    body = ET.tostring(root, encoding="unicode")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n' + body + "\n"
    )


def lyric_lines(sheet: ArrangedSheet) -> list[tuple[float, str]]:
    """(start time, text) per lyric line; English syllables ending in "-" join the next one."""
    lines: dict[int, list[str]] = {}
    starts: dict[int, float] = {}
    for syl in sheet.lyrics:
        parts = lines.setdefault(syl.line_index, [])
        starts.setdefault(syl.line_index, syl.start)
        if parts and not parts[-1].endswith("-") and _latin(parts[-1]) and _latin(syl.text):
            parts.append(" ")
        parts.append(syl.text)
    return [
        (starts[i], "".join(p[:-1] if p.endswith("-") else p for p in lines[i]))
        for i in sorted(lines)
    ]


def _latin(s: str) -> bool:
    return any(c.isascii() and c.isalnum() for c in s)


def _line_starts(sheet: ArrangedSheet) -> dict[int, list[tuple[float, str]]]:
    """Bar index -> (beat offset, line text) for every lyric line starting in that bar."""
    out: dict[int, list[tuple[float, str]]] = {}
    for start, text in lyric_lines(sheet):
        for n, bar in enumerate(sheet.bars):
            if bar.start <= start < bar.end or (n == len(sheet.bars) - 1 and start >= bar.start):
                span = bar.end - bar.start
                beat = (start - bar.start) / span * bar.beats if span > 0 else 0.0
                out.setdefault(n, []).append((min(beat, bar.beats - 0.25), text))
                break
        else:
            if sheet.bars:
                out.setdefault(0, []).append((0.0, text))
    return out


def _fifths(tonic: int, mode: Mode) -> int:
    major = tonic if mode == Mode.MAJOR else (tonic + 3) % 12
    # number of sharps (positive) or flats (negative) of the major key
    circle = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 8: -4, 3: -3, 10: -2, 5: -1}
    return circle[major]
