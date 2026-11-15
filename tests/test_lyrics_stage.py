import numpy as np
import pytest

from openbapo.analysis.lyrics.align import (
    CtcModel,
    TimedToken,
    ctc_targets,
    emissions,
    song_language,
)
from openbapo.analysis.lyrics.stage import fill_gaps, to_syllables
from openbapo.lyrics import parse_lyrics, tokenize


def timed(text: str, spans: list[tuple[float, float] | None], line: int = 0) -> list[TimedToken]:
    return [
        TimedToken(line, tok, *(s if s else (None, None)))
        for tok, s in zip(tokenize(text), spans, strict=True)
    ]


def test_fill_gaps_interpolates_between_neighbours() -> None:
    tokens = timed("我爱你们", [(1.0, 1.5), None, None, (3.0, 3.5)])
    assert fill_gaps(tokens, 10.0) == [(1.0, 1.5), (1.5, 2.25), (2.25, 3.0), (3.0, 3.5)]


def test_fill_gaps_at_the_edges() -> None:
    tokens = timed("我爱你", [None, (2.0, 2.5), None])
    spans = fill_gaps(tokens, 10.0)
    assert spans[0] == (0.0, 2.0)
    assert spans[2][0] == 2.5 and spans[2][1] > 2.5


def test_syllables_split_a_word_by_letters() -> None:
    syllables, warnings = to_syllables(timed("beautiful day", [(1.0, 1.9), (2.0, 2.4)]), 10.0)
    assert [s.text for s in syllables] == ["beau-", "ti-", "ful", "day"]
    assert syllables[0].start == 1.0 and syllables[2].end == 1.9
    assert syllables[1].start == pytest.approx(1.4)
    assert not warnings


def test_syllables_never_go_backwards() -> None:
    syllables, warnings = to_syllables(timed("我爱你", [(2.0, 2.5), (1.0, 1.2), None]), 10.0)
    starts = [s.start for s in syllables]
    assert starts == sorted(starts)
    assert warnings and "1 of 3" in warnings[0]


def test_song_language() -> None:
    assert song_language(parse_lyrics("我说 baby\n别走")) == "zh"
    assert song_language(parse_lyrics("hold me 我\nnow and forever")) == "en"


def letters_model() -> CtcModel:
    labels = {"-": 0, "a": 1, "b": 2}

    def forward(chunk: np.ndarray) -> np.ndarray:
        raise AssertionError("not used")

    return CtcModel(forward, labels, 0, 0.02, lambda t: list(t.spoken))


def test_ctc_targets_leave_unknown_tokens_empty() -> None:
    tokens = tokenize("ab ba abc")
    assert ctc_targets(letters_model(), tokens) == [[1, 2], [2, 1], []]


def test_emissions_are_blank_outside_regions() -> None:
    model = letters_model()

    def forward(chunk: np.ndarray) -> np.ndarray:
        frames = len(chunk) // 320
        out = np.full((frames, 3), -5.0, dtype=np.float32)
        out[:, 1] = 0.0
        return out

    model.forward = forward
    log_probs = emissions(model, np.zeros(16000 * 4, dtype=np.float32), [(1.0, 2.0)])
    assert log_probs.shape[0] == 201
    assert (log_probs[:30, 0] == 0).all() and (log_probs[:30, 1] < -100).all()
    assert (log_probs[60:90, 1] == 0).all()
    assert (log_probs[130:, 0] == 0).all()


def test_forced_align_places_tokens_where_the_labels_are() -> None:
    pytest.importorskip("torchaudio")
    from openbapo.analysis.lyrics.align import forced_align

    log_probs = np.full((100, 3), -10.0, dtype=np.float32)
    log_probs[:, 0] = 0.0
    log_probs[20:25, 0], log_probs[20:25, 1] = -10.0, 0.0  # "a" at 0.40-0.50 s
    log_probs[60:70, 0], log_probs[60:70, 2] = -10.0, 0.0  # "b" at 1.20-1.40 s
    out = forced_align(log_probs, [[1], [], [2]], blank=0, frame_s=0.02)
    assert out[1] is None
    assert (
        out[0] is not None and out[0][0] == pytest.approx(0.40) and out[0][1] == pytest.approx(0.50)
    )
    assert out[2] is not None and out[2][0] == pytest.approx(1.20)


def test_ctc_targets_put_the_separator_between_words() -> None:
    model = letters_model()
    model.labels = {**model.labels, "|": 3}
    model.separator = "|"
    assert ctc_targets(model, tokenize("ab ba abc ab")) == [[1, 2, 3], [2, 1, 3], [], [1, 2]]
