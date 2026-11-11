"""ArrangedSheet -> MusicXML (score-partwise 4.0) for editing in MuseScore and friends.

Two parts: a standard staff that carries the chord symbols (and, later,
the melody with its lyrics) and a TAB staff with the strum pattern's
strokes written as string / fret notes. Chord symbols are written as they
sound; the TAB part declares the capo as a transposition so an editor
shows the shapes' own names.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

from openbapo.arrange import ArrangedSheet, Bar, Shape, StrumEvent, StrumPattern
from openbapo.arrange.keys import parse_key_name
from openbapo.model import Mode

TUNING = (40, 45, 50, 55, 59, 64)  # MIDI of the open strings, low to high
STEPS = ("C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B")
ALTERS = (0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0)
QUALITY_KIND = {
    "": "major", "m": "minor", "7": "dominant", "maj7": "major-seventh", "m7": "minor-seventh",
    "6": "major-sixth", "m6": "minor-sixth", "sus2": "suspended-second", "sus4": "suspended-fourth",
    "5": "power", "dim": "diminished", "aug": "augmented", "dim7": "diminished-seventh",
    "m7-5": "half-diminished",
}  # fmt: skip
DIVISIONS = 4  # per beat: down to sixteenths


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


def _duration_type(beats: float) -> tuple[int, str]:
    for value, name in (
        (4.0, "whole"),
        (2.0, "half"),
        (1.0, "quarter"),
        (0.5, "eighth"),
        (0.25, "16th"),
    ):
        if abs(beats - value) < 1e-6:
            return int(value * DIVISIONS), name
    return max(1, round(beats * DIVISIONS)), "quarter"


def _harmony(
    measure: ET.Element,
    name: str,
    sounding_root: int,
    sounding_bass: int | None,
    beat_offset: float,
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
    if beat_offset:
        _sub(harmony, "offset", str(int(beat_offset * DIVISIONS)))


def _tab_events(
    bar: Bar, pattern: StrumPattern | None
) -> list[tuple[float, float, StrumEvent | None, str]]:
    """(beat, length in beats, event, chord shape name) for the bar, filling the bar with rests."""
    chords = bar.chords
    if not chords:
        return [(0.0, float(bar.beats), None, "")]
    starts = [float(c.beat) for c in chords]
    points: list[tuple[float, StrumEvent | None]]
    if pattern is None:
        points = [(s, None) for s in starts]
    else:
        points = [(e.beat, e) for e in pattern.events if e.beat < bar.beats]
        if not points:
            points = [(0.0, None)]
    out: list[tuple[float, float, StrumEvent | None, str]] = []
    for n, (beat, event) in enumerate(points):
        nxt = points[n + 1][0] if n + 1 < len(points) else float(bar.beats)
        current = chords[max(i for i, s in enumerate(starts) if s <= beat or i == 0)]
        out.append((beat, nxt - beat, event, current.shape))
    if out[0][0] > 0:
        out.insert(0, (0.0, out[0][0], None, ""))
    return out


def _tab_note(
    measure: ET.Element, string_index: int, fret: int, duration: int, kind: str, chord: bool
) -> None:
    note = _sub(measure, "note")
    if chord:
        _sub(note, "chord")
    _pitch(note, TUNING[string_index] + fret)
    _sub(note, "duration", str(duration))
    _sub(note, "voice", "1")
    _sub(note, "type", kind)
    technical = _sub(_sub(note, "notations"), "technical")
    _sub(technical, "string", str(6 - string_index))
    _sub(technical, "fret", str(fret))


def _rest(measure: ET.Element, duration: int, kind: str) -> None:
    note = _sub(measure, "note")
    _sub(note, "rest")
    _sub(note, "duration", str(duration))
    _sub(note, "voice", "1")
    _sub(note, "type", kind)


def _strings_for(event: StrumEvent | None, shape: Shape) -> list[int]:
    sounded = [i for i, f in enumerate(shape.frets) if f is not None]
    if event is None or event.stroke is not None:
        return sounded
    out = []
    for ref in event.strings:
        if ref == "bass":
            if sounded:
                out.append(sounded[0])
        elif isinstance(ref, int) and shape.frets[6 - ref] is not None:
            out.append(6 - ref)
    return out


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
    bar_beats_default = num

    p1 = _sub(root, "part", id="P1")
    p2 = _sub(root, "part", id="P2")
    bars = sheet.bars or ()
    for n, bar in enumerate(bars):
        m1 = _sub(p1, "measure", number=str(n + 1))
        m2 = _sub(p2, "measure", number=str(n + 1))
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
                _sub(_sub(_sub(direction, "direction-type"), "metronome"), "beat-unit", "quarter")
                _sub(direction.find("direction-type/metronome"), "per-minute", str(round(h.bpm)))  # type: ignore[arg-type]
                _sub(direction, "sound", tempo=str(round(h.bpm)))
        beats = bar.beats or bar_beats_default
        # Chord symbols as they sound: the shape's root plus the capo.
        for c in bar.chords:
            shape = sheet.shapes[c.shape]
            bass_name = c.name.split("/")[1] if "/" in c.name else None
            sounding_bass = None
            if bass_name:
                sounding_bass = (parse_key_name(bass_name)[0] + h.capo) % 12
            _harmony(m1, c.name, (shape.root + h.capo) % 12, sounding_bass, float(c.beat))
        whole, kind = _duration_type(float(beats))
        _rest(m1, whole, kind if beats in (1, 2, 4) else "whole")
        for _beat, length, event, shape_name in _tab_events(bar, sheet.strum):
            duration, kind = _duration_type(length)
            strings = _strings_for(event, sheet.shapes[shape_name]) if shape_name else []
            if not strings or (event is not None and event.stroke == "mute"):
                _rest(m2, duration, kind)
                continue
            shape = sheet.shapes[shape_name]
            for k, index in enumerate(strings):
                fret = shape.frets[index]
                assert fret is not None
                _tab_note(m2, index, fret, duration, kind, chord=k > 0)
    ET.indent(root)
    body = ET.tostring(root, encoding="unicode")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n' + body + "\n"
    )


def _fifths(tonic: int, mode: Mode) -> int:
    major = tonic if mode == Mode.MAJOR else (tonic + 3) % 12
    # number of sharps (positive) or flats (negative) of the major key
    circle = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 8: -4, 3: -3, 10: -2, 5: -1}
    return circle[major]
