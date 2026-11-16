import pytest

from openbapo.lyrics import clean_line, parse_lyrics, split_word, tokenize


def test_chinese_line_is_one_token_per_character() -> None:
    tokens = tokenize("让我掉下眼泪的，不止昨夜的酒")
    assert [t.text for t in tokens] == list("让我掉下眼泪的不止昨夜的酒")
    assert all(t.lang == "zh" and t.syllables == (t.text,) for t in tokens)


def test_english_words_and_syllables() -> None:
    tokens = tokenize("Where the north wind meets the sea, it's beautiful")
    assert [t.text for t in tokens] == [
        "Where",
        "the",
        "north",
        "wind",
        "meets",
        "the",
        "sea",
        "it's",
        "beautiful",
    ]
    assert tokens[-1].syllables == ("beau-", "ti-", "ful")
    assert tokens[-2].syllables == ("it's",)
    assert tokens[0].spoken == "where"


def test_mixed_line() -> None:
    tokens = tokenize("我说 baby 别走")
    assert [(t.text, t.lang) for t in tokens] == [
        ("我", "zh"),
        ("说", "zh"),
        ("baby", "en"),
        ("别", "zh"),
        ("走", "zh"),
    ]


def test_digits_read_out_in_chinese_lines_only() -> None:
    assert [t.text for t in tokenize("1 2 3 木头人")] == list("一二三木头人")
    assert [t.text for t in tokenize("route 66")] == ["route", "66"]


@pytest.mark.parametrize(
    ("raw", "cleaned"),
    [
        ("[Chorus]", ""),
        ("【副歌】", ""),
        ("(Verse 2)", ""),
        ("副歌：", ""),
        ("Chorus: hold me now", "hold me now"),
        ("女：我在等你", "我在等你"),
        ("hold me now (x2)", "hold me now"),
        ("一直到老 ×2", "一直到老"),
        ("（oh yeah）", "(oh yeah)"),  # sung backing vocals stay
        ("  spaced   out  ", "spaced out"),
    ],
)
def test_clean_line(raw: str, cleaned: str) -> None:
    assert clean_line(raw) == cleaned


def test_parse_lyrics_numbers_sung_lines() -> None:
    text = "[Verse]\n离别没说再见\n\n你是否心酸\n...\nChorus:\nsay goodbye ×2\n"
    lines = parse_lyrics(text)
    assert [(line.index, line.text, line.lang) for line in lines] == [
        (0, "离别没说再见", "zh"),
        (1, "你是否心酸", "zh"),
        (2, "say goodbye", "en"),
    ]


def test_split_word_keeps_short_and_odd_words_whole() -> None:
    assert split_word("sound") == ("sound",)
    assert split_word("rock'n'roll") == ("rock'n'roll",)
    assert split_word("memory") == ("mem-", "o-", "ry")
    assert "".join(s.rstrip("-") for s in split_word("continuing")) == "continuing"
