"""Qwen3-ASR recognizer: region offsets, line splitting, language mapping, no model needed."""

from dataclasses import dataclass

import numpy as np
import pytest

from openbapo.analysis.lyrics.activity import SAMPLE_RATE
from openbapo.analysis.lyrics.recognize_qwen import Qwen3ASRRecognizer, _our_language


@dataclass
class Item:
    text: str
    start_time: float
    end_time: float


@dataclass
class Stamps:
    items: list


@dataclass
class Result:
    language: str
    text: str
    time_stamps: object = None


class Model:
    """Answers per clip from a script; records what it was asked."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def transcribe(self, audio, language=None, return_time_stamps=False, **kwargs):
        self.calls.append((len(audio), language, return_time_stamps))
        return [self.answers[k % len(self.answers)] for k in range(len(audio))]


def test_lines_are_offset_to_the_song_split_at_gaps_and_simplified() -> None:
    pytest.importorskip("opencc")  # Traditional -> Simplified, an analysis dependency
    # Two voiced regions; the second one is heard as two lines 0.8 s apart.
    answers = [
        Result(
            "Chinese",
            "我愛你",
            Stamps([Item("我", 0.1, 0.4), Item("愛", 0.4, 0.7), Item("你", 0.7, 1.0)]),
        ),
        Result(
            "Chinese",
            "唱歌 跳舞",
            Stamps(
                [
                    Item("唱", 0.0, 0.3),
                    Item("歌", 0.3, 0.6),
                    Item("跳", 1.4, 1.7),
                    Item("舞", 1.7, 2.0),
                ]
            ),
        ),
    ]
    model = Model(answers)
    vocals = np.zeros(60 * SAMPLE_RATE, dtype=np.float32)
    rec = Qwen3ASRRecognizer(model).recognize(vocals, [(10.0, 12.0), (30.0, 33.0)])
    assert rec.language == "zh"
    assert [(line.text, line.start, line.end) for line in rec.lines] == [
        ("我爱你", 10.1, 11.0),
        ("唱歌", 30.0, 30.6),
        ("跳舞", 31.4, 32.0),
    ]
    assert model.calls == [(2, None, True)]


def test_a_forced_language_is_passed_in_qwens_name_and_credits_are_dropped() -> None:
    answers = [
        Result("English", "sky kite", Stamps([Item("sky", 0.0, 0.5), Item("kite", 0.5, 1.0)])),
        Result(
            "English",
            "subtitles by X",
            Stamps([Item("subtitles", 0.0, 0.5), Item("by", 0.5, 0.7), Item("X", 0.7, 0.9)]),
        ),
    ]
    model = Model(answers)
    vocals = np.zeros(20 * SAMPLE_RATE, dtype=np.float32)
    rec = Qwen3ASRRecognizer(model).recognize(vocals, [(1.0, 3.0), (5.0, 7.0)], language="en")
    assert model.calls == [(2, "English", True)]
    assert [line.text for line in rec.lines] == ["sky kite"] and rec.dropped == 1
    assert [w.text for w in rec.lines[0].words] == ["sky", "kite"]


def test_short_regions_are_skipped_and_the_language_is_read_from_qwen() -> None:
    model = Model([Result("Chinese,English", "", Stamps([]))])
    vocals = np.zeros(10 * SAMPLE_RATE, dtype=np.float32)
    recognizer = Qwen3ASRRecognizer(model)
    assert recognizer.recognize(vocals, [(0.0, 0.2)]).lines == ()
    assert model.calls == []
    assert recognizer.detect_language(vocals, [(0.0, 5.0)]) == "zh"
    assert _our_language("English") == "en" and _our_language("") is None
    assert _our_language("Japanese") is None
