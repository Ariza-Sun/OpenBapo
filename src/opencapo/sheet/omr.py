"""Reading a printed sheet into the text sheet format: protocol, checks, assembly.

The reading itself is done by an agent with vision, one bar at a time,
from the stacked crops that `opencapo.sheet.layout` produces. This module
holds the other three parts of the loop:

- the reading protocol: what the reader is asked and the JSON it returns
  per bar (`READING_PROMPT`, `BarReading`);
- the checks that catch misreads without a human: every bar's melody
  must fill the metre, every chord name must parse, a tie into a bar must
  continue the previous pitch, and bars that repeat an earlier passage
  should match it;
- the assembly of the checked readings into a `.sheet` text.

Bars that fail a check go back to the reader once with a larger crop; what
still fails is listed for review.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from opencapo.sheet.format import (
    SheetError,
    _group_tokens,
    parse_chords,
    parse_melody,
    parse_sheet,
)

READING_PROMPT = """你在读一张中文吉他谱的一个小节。图片上半部分是和弦图（和弦名写在图上方），
下半部分是这一小节的简谱和歌词。请只输出一个 JSON 对象，不要解释：

{"chords": "<和弦>", "melody": "<简谱>", "notes": "<可选：不确定之处>"}

和弦：按和弦图上方的名字从左到右写，用空格分开；一小节一个和弦就写一个（如 "Am"）；
两个和弦按它们在小节里的起拍写拍数，如 "C(2) G(2)"；上半部分没有和弦图就写空字符串
（表示沿用上一小节的和弦）。

简谱：按下面的记法写这一小节的旋律，拍数要凑满拍号（4/4 就是 4 拍）：
- 数字 1-7 是音，0 是休止；
- 数字下面一条横线是八分音符，两条是十六分音符。把共享一拍的音用方括号括起来，
  如 "[6 6]" 是两个八分，"[6 [5 5]]" 是一个八分加两个十六分；
- 数字下面的点表示低八度，写成 "6,"；上面的点是高八度，写成 "1'"；点和横线挨得很近，
  要逐个数字确认有没有点；
- 数字后面的小圆点是附点，写成 "2."（附点八分加十六分写成 "[2. 5]"）；
- "-" 是延长上一个音一拍；如果小节开头有从上一小节连过来的弧线（连音线），
  这一小节的第一个音写成 "-" 表示延续；
- 弧线连着不同的音是连音（滑音），照常写音。

例：{"chords": "G", "melody": "[0 2] [2. 5,] [2 2] [5, 6,]"}
"""


HEADER_PROMPT = """这是一张中文吉他谱第一页的页眉。请只输出一个 JSON 对象，不要解释：

{"title": "<歌名>", "key": "<1= 后面的音名，如 E 或 bB 写成 Bb>", "capo": <变调夹品数，没有写 0>,
 "time": "<拍号，如 4/4>", "bpm": <速度数字，没有写 null>}
"""


@dataclass
class BarReading:
    index: int
    chords: str
    melody: str
    notes: str = ""
    crop: str = ""


@dataclass
class Check:
    index: int
    problem: str


@dataclass
class Assembled:
    text: str
    problems: list[Check] = field(default_factory=list)


def parse_reading(index: int, text: str, crop: str = "") -> BarReading:
    """The reader's JSON (possibly wrapped in prose or a code fence) -> BarReading."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise SheetError(f"bar {index}: no JSON object in reading: {text[:80]!r}")
    data = json.loads(text[start : end + 1])
    return BarReading(
        index=index,
        chords=str(data.get("chords", "")).strip(),
        melody=str(data.get("melody", "")).strip(),
        notes=str(data.get("notes", "")).strip(),
        crop=crop,
    )


def check_readings(readings: list[BarReading], beats_per_bar: int) -> list[Check]:
    """Deterministic checks over a whole sheet's readings."""
    problems: list[Check] = []
    previous_melody: str | None = None
    for r in readings:
        try:
            parse_chords(r.chords, beats_per_bar)
        except SheetError as e:
            problems.append(Check(r.index, f"chords: {e}"))
        try:
            notes = parse_melody(r.melody, beats_per_bar)
        except SheetError as e:
            problems.append(Check(r.index, f"melody: {e}"))
            previous_melody = None
            continue
        total = sum((n.beats for n in notes), Fraction(0))
        if notes and total != beats_per_bar:
            problems.append(Check(r.index, f"melody fills {total} beats, bar has {beats_per_bar}"))
        # The reader writes one top-level item per beat (groups subdivide a
        # beat), so a missed or invented note shows up as a wrong item count.
        items = len(_group_tokens(r.melody))
        if notes and items != beats_per_bar:
            problems.append(Check(r.index, f"{items} beat items written, bar has {beats_per_bar}"))
        if notes and notes[0].tied and not previous_melody:
            problems.append(Check(r.index, "tie into the bar but the previous bar has no melody"))
        previous_melody = r.melody or None
    return problems


def repeated_passage_mismatches(readings: list[BarReading], window: int = 4) -> list[Check]:
    """Bars whose chords repeat an earlier run of `window` bars but whose melody differs.

    A verse sung twice is printed twice; a melody that differs under the
    same chords is more often a misread than a variation, so it is flagged
    for a second look (not rejected).
    """
    problems: list[Check] = []
    seen: dict[tuple[str, ...], list[str]] = {}
    for i in range(len(readings) - window + 1):
        run = readings[i : i + window]
        key = tuple(r.chords for r in run)
        if not all(key):
            continue
        melody = [r.melody for r in run]
        if key in seen and seen[key] != melody:
            for r, earlier in zip(run, seen[key], strict=True):
                if r.melody != earlier:
                    problems.append(
                        Check(
                            r.index,
                            f"differs from an earlier bar with the same chords: {earlier!r}",
                        )
                    )
        seen.setdefault(key, melody)
    return problems


def assemble(header: dict[str, str], readings: list[BarReading], beats_per_bar: int) -> Assembled:
    """Header fields plus checked readings -> the .sheet text (chords carry over empty bars)."""
    lines = [f"{k}: {v}" for k, v in header.items() if v != ""]
    lines.append("---")
    last_chords = ""
    for r in readings:
        chords = r.chords or last_chords
        last_chords = chords
        melody = r.melody
        lines.append(f"{chords:<10} | {melody}" if melody else chords)
    text = "\n".join(lines) + "\n"
    problems = check_readings(readings, beats_per_bar) + repeated_passage_mismatches(readings)
    try:
        parse_sheet(text)
    except SheetError as e:
        problems.append(Check(-1, f"sheet does not parse: {e}"))
    return Assembled(text=text, problems=problems)


def write_review(path: Path, problems: list[Check], readings: list[BarReading]) -> None:
    """A markdown list of bars needing a second look, with their crops."""
    by_index = {r.index: r for r in readings}
    lines = ["# 待复核小节", ""]
    for p in problems:
        r = by_index.get(p.index)
        crop = f" `{r.crop}`" if r and r.crop else ""
        lines.append(f"- 小节 {p.index}: {p.problem}{crop}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
