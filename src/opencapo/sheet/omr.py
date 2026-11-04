"""Reading a printed sheet into the text sheet format: protocol, checks, assembly.

The reading itself is done by an agent with vision, one row (system) at
a time, from the row images that `opencapo.sheet.layout` produces; the
agent splits the row into bars. This module holds the other three parts
of the loop:

- the reading protocol: what the reader is asked and the JSON it returns
  per bar (`READING_PROMPT`, `BarReading`);
- the checks that catch misreads without a human: every bar's melody
  must fill the metre, every chord name must parse, a tie into a bar must
  continue the previous pitch, and bars that repeat an earlier passage
  should match it;
- the assembly of the checked readings into a `.sheet` text.

Rows holding a bar that fails a check go back to the reader once on their
own; what still fails is listed for review.
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

READING_PROMPT = """你在读一张中文吉他谱的一行。图片从上到下是：和弦图（和弦名写在图上方）、六线谱、
这一行的简谱、歌词。请按小节从左到右，把这一行写成一个 JSON 数组，每个小节一个对象，不要解释：

[{"chords": "<和弦>", "melody": "<简谱>", "lyrics": "<歌词>", "notes": "<可选：不确定之处>"}, ...]

小节：以简谱行上的竖线为界（反复记号 ‖: :‖ 也是小节线）；行首行尾的不完全小节也各算一个小节。
行首有大括号或谱号时不要把它当小节线。一行通常有 3 到 6 个小节，不要多切也不要漏切。

和弦：按和弦图上方的名字从左到右写，用空格分开；一小节一个和弦就写一个（如 "Am"）；
两个和弦按它们在小节里的起拍写拍数，如 "C(2) G(2)"（每个都要写拍数）；
这一小节上方没有和弦图就写空字符串（表示沿用上一小节的和弦）。
升降号写在音名前后都行（#F 或 F#）；斜线和弦要写全，如 "D/#F"、"C/G"。

简谱：按下面的记法写这一小节的旋律，拍数要凑满拍号（4/4 就是 4 拍）：
- 数字 1-7 是音，0 是休止；
- 数字下面一条横线是八分音符，两条是十六分音符。把共享一拍的音用方括号括起来，
  如 "[6 6]" 是两个八分，"[6 [5 5]]" 是一个八分加两个十六分；
- 数字下面的点表示低八度，写成 "6,"；上面的点是高八度，写成 "1'"；点和横线挨得很近，
  要逐个数字确认有没有点；
- 数字后面的小圆点是附点，写成 "2."（附点八分加十六分写成 "[2. 5]"）；
- "-" 是延长上一个音一拍；如果小节开头有从上一小节（或上一行末尾）连过来的弧线（连音线），
  这一小节的第一个音写成 "-" 表示延续；
- 弧线连着不同的音是连音（滑音），照常写音。

歌词：这一小节下方的歌词原样抄，没有就写空字符串；两行歌词用 " / " 隔开。

例：[{"chords": "G", "melody": "[0 2] [2. 5,] [2 2] [5, 6,]", "lyrics": "我 匆匆地走入"},
     {"chords": "Am", "melody": "[1 6,] - 0 0", "lyrics": "森林中"}]
"""


HEADER_PROMPT = """这是一张中文吉他谱第一页的页眉。请只输出一个 JSON 对象，不要解释：

{"title": "<歌名>", "key": "<演唱调：'原调 1=X' 的 X；页眉只印一个 1=X 时就是它。E 或 bB 写成 Bb>",
 "play_key": "<指法调：'选调 1=Y'、'用Y调指法'、'Y调和弦编配' 的 Y；没印写 null>",
 "capo": <变调夹品数；印成范围或男女分开的写 null>,
 "time": "<拍号，如 4/4>", "bpm": <速度数字，没有写 null>}

注意：同时印了原调和选调时，key 填原调，不要填选调；选调加上变调夹品数应等于原调。
"""


def reconcile_header(header: dict[str, object]) -> list[str]:
    """Fill the sung key from the fingering key and capo when it is missing, and report conflicts.

    Sheets print the sung key (原调 1=X), or the fingering key (选调 1=Y)
    with the capo, or both; both must agree as Y + capo = X. Returns the
    problems found; the header is updated in place.
    """
    from opencapo.sheet.format import pitch_class

    problems: list[str] = []
    try:
        play = pitch_class(str(header["play_key"])) if header.get("play_key") else None
        sung = pitch_class(str(header["key"])) if header.get("key") else None
    except SheetError as e:
        return [f"页眉的调名读不懂：{e}"]
    capo = header.get("capo")
    capo = int(capo) if isinstance(capo, int | float) else None
    if sung is None and play is not None and capo is not None:
        header["key"] = PITCH_NAMES[(play + capo) % 12]
    elif sung is not None and play is not None and capo is not None:
        if (play + capo) % 12 != sung:
            problems.append(
                f"页眉不一致：选调 {header['play_key']} + 变调夹 {capo} 品 ≠ 原调 {header['key']}"
            )
    elif sung is not None and play is not None and capo is None:
        header["capo"] = (sung - play) % 12  # a capo given as a range: take what the keys imply
    return problems


PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass
class BarReading:
    index: int
    chords: str
    melody: str
    notes: str = ""
    crop: str = ""  # the row image the bar was read from
    row: int = -1
    lyrics: str = ""


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


def repeated_passage_mismatches(readings: list[BarReading], window: int = 8) -> list[Check]:
    """Bars in a repeated passage whose melody differs from the earlier printing.

    A verse sung twice is printed twice. Where a run of `window` bars has the
    same chords as an earlier run, both runs carry a melody, and they agree
    on most bars, the few bars that differ are flagged for a second look
    (not rejected): a misread is more likely than a variation. Runs that
    differ in many bars are different passages and are left alone.
    """
    problems: dict[int, Check] = {}
    melodic = [r for r in readings if any(ch in "1234567" for ch in r.melody)]
    seen: dict[tuple[str, ...], list[BarReading]] = {}
    for i in range(len(melodic) - window + 1):
        run = melodic[i : i + window]
        key = tuple(r.chords for r in run)
        if not all(key):
            continue
        earlier = seen.setdefault(key, run)
        if earlier is run:
            continue
        differing = [(r, e) for r, e in zip(run, earlier, strict=True) if r.melody != e.melody]
        if 0 < len(differing) <= max(1, window // 4):
            for r, e in differing:
                problems.setdefault(
                    r.index,
                    Check(r.index, f"differs from bar {e.index}, same chords: {e.melody!r}"),
                )
    return list(problems.values())


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
        where = f"（第 {r.row} 行）" if r and r.row >= 0 else ""
        crop = f" `{r.crop}`" if r and r.crop else ""
        lines.append(f"- 小节 {p.index}{where}: {p.problem}{crop}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
