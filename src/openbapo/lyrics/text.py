"""Lyrics text: lines, the units an aligner times, and the syllables they hold.

A line of lyrics is split into tokens. A Chinese character is one token and
one syllable. A Latin-script word is one token; its syllables come from
`syllables.split_word` and share the word's time span when it is aligned.

Section labels (`[Chorus]`, `【副歌】`, `Verse 1:`, `副歌：`), repeat marks
(`x2`, `×2`) and punctuation are dropped; blank lines only separate lines.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from openbapo.lyrics.syllables import split_word

Lang = Literal["zh", "en"]

_CJK = r"㐀-䶿一-鿿豈-﫿"
_TOKEN = re.compile(rf"[{_CJK}]|[0-9]+|[^\W\d_{_CJK}]+(?:['’][^\W\d_{_CJK}]+)*")
_HAS_CJK = re.compile(rf"[{_CJK}]")
_LABEL_WORDS = (
    r"intro|outro|verse|pre[- ]?chorus|chorus|refrain|bridge|hook|interlude|instrumental|solo|"
    r"repeat|前奏|主歌|副歌|间奏|尾奏|桥段|过门|重复|独白|合唱|男|女|合"
)
_BRACKETED = re.compile(r"[\[【〔].*?[\]】〕]")
_LABEL_LINE = re.compile(rf"^\(?\s*(?:{_LABEL_WORDS})\s*[0-9一二三四]*\s*\)?\s*$", re.IGNORECASE)
_LABEL_PREFIX = re.compile(rf"^(?:{_LABEL_WORDS})\s*[0-9一二三四]*\s*[:：]\s*", re.IGNORECASE)
_REPEAT = re.compile(
    r"[\(（]?\s*[x×*]\s*[0-9]+\s*[\)）]?\s*$|^\s*[\(（]?\s*[0-9]+\s*[x×*]\s*[\)）]?\s*$"
)
_ZH_DIGITS = dict(zip("0123456789", "零一二三四五六七八九", strict=True))


@dataclass(frozen=True)
class Token:
    text: str  # as written in the lyrics
    lang: Lang
    syllables: tuple[str, ...]  # displayed syllables; non-final ones end with "-"

    @property
    def spoken(self) -> str:
        """The token as the aligner reads it: lower case, straight apostrophes."""
        return self.text.lower().replace("’", "'")


@dataclass(frozen=True)
class Line:
    index: int
    text: str  # the cleaned line as displayed
    tokens: tuple[Token, ...]

    @property
    def lang(self) -> Lang:
        return "zh" if any(t.lang == "zh" for t in self.tokens) else "en"


def clean_line(raw: str) -> str:
    """The sung part of a line; empty for a label, a repeat mark or punctuation only."""
    line = unicodedata.normalize("NFKC", raw).strip()
    line = _BRACKETED.sub(" ", line)
    line = _LABEL_PREFIX.sub("", line.strip())
    line = _REPEAT.sub("", line).strip()
    if _LABEL_LINE.match(line):
        return ""
    return " ".join(line.split())


def tokenize(line: str) -> tuple[Token, ...]:
    zh_line = bool(_HAS_CJK.search(line))
    tokens: list[Token] = []
    for match in _TOKEN.finditer(line):
        text = match.group()
        if _HAS_CJK.fullmatch(text):
            tokens.append(Token(text, "zh", (text,)))
        elif text.isdigit() and zh_line:
            tokens.extend(Token(_ZH_DIGITS[d], "zh", (_ZH_DIGITS[d],)) for d in text)
        else:
            tokens.append(Token(text, "en", split_word(text)))
    return tuple(tokens)


def parse_lyrics(text: str) -> tuple[Line, ...]:
    """Lines that hold at least one token, numbered from 0."""
    lines: list[Line] = []
    for raw in text.splitlines():
        cleaned = clean_line(raw)
        tokens = tokenize(cleaned)
        if tokens:
            lines.append(Line(len(lines), cleaned, tokens))
    return tuple(lines)
