"""The one-command reading loop, with a fake model."""

import json
from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image
from test_sheet_omr import drawn_page

from openbapo.sheet import reader


def test_extract_json_and_prompts() -> None:
    assert reader.extract_json('x ```json\n[{"row": 1}]\n``` y') == [{"row": 1}]
    assert reader.extract_json('{"a": 1} trailing', "object") == {"a": 1}
    with pytest.raises(ValueError):
        reader.extract_json("nothing")
    prompt = reader.batch_prompt([(3, "/a.png"), (4, "/b.png")])
    assert "3\t/a.png" in prompt and "4\t/b.png" in prompt and "JSON 数组" in prompt
    assert "/h.png" in reader.header_prompt("/h.png")
    reply = (
        '[{"row": 3, "bars": [{"chords": "C", "melody": "1 2 3 4", "lyrics": "la"},'
        ' {"melody": "5 - - -"}]}]'
    )
    parsed = reader.parse_batch(reply, [3])
    assert [(b.row, b.chords, b.melody, b.lyrics) for b in parsed[3]] == [
        (3, "C", "1 2 3 4", "la"),
        (3, "", "5 - - -", ""),
    ]
    with pytest.raises(ValueError, match="lacks rows"):
        reader.parse_batch(reply, [3, 4])  # row 4 missing
    with pytest.raises(ValueError, match="lacks rows"):
        reader.parse_batch('[{"row": 3, "bars": []}]', [3])  # a row without bars is no reading


def rows_in(prompt: str) -> list[tuple[int, str]]:
    return [
        (int(i), p) for i, p in (line.split("\t") for line in prompt.splitlines() if "\t" in line)
    ]


def fake_runner(calls: list[str]) -> reader.Runner:
    """Answers the header, then every row (three bars each).

    Row 1's middle bar is read without its octave dot in pass 1 (batched
    with row 0) and with it in pass 2 (alone) and in the deciding x3
    reading; row 0's last bar is read with three beat items in both passes
    (a check failure) and right from x3.
    """

    def run(prompt: str) -> str:
        calls.append(prompt)
        if "页眉" in prompt:
            return '{"title": "t", "key": "D", "capo": 2, "time": "4/4", "bpm": null}'
        items = rows_in(prompt)
        batch = {i for i, _ in items}
        out = []
        for row, path in items:
            deciding = "-x3" in path
            bars = [{"chords": "C", "melody": "1 2 3 4", "lyrics": "a"} for _ in range(3)]
            if row == 1:  # misses the dot only when read in the same batch as row 0 (pass 1)
                bars[1] = {
                    "chords": "G",
                    "melody": "1 2 3, 4" if deciding or 0 not in batch else "1 2 3 4",
                }
            if row == 0:
                bars[2] = {"chords": "C", "melody": "1 2 3 4" if deciding else "1 2 3"}
            out.append({"row": row, "bars": bars})
        return json.dumps(out)

    return run


def test_read_sheet_loop(tmp_path: Path) -> None:
    page = tmp_path / "song.png"
    drawn_page(systems=2).save(page)
    calls: list[str] = []
    logs: list[str] = []
    result = reader.read_sheet(
        page, tmp_path / "work", runner=fake_runner(calls), bpm=100, batch_size=2, log=logs.append
    )
    assert result.header["key"] == "D"
    assert result.reread == [0, 1] and result.unsettled == []
    assert [r.melody for r in result.readings] == ["1 2 3 4"] * 4 + ["1 2 3, 4", "1 2 3 4"]
    assert [r.index for r in result.readings] == list(range(6))
    assert [r.row for r in result.readings] == [0, 0, 0, 1, 1, 1]
    assert result.problems == []
    text = result.sheet_path.read_text(encoding="utf-8")
    assert text.startswith("title: t\nkey: D\nmode: major\ncapo: 2\nbpm: 100\ntime: 4/4\n---\n")
    assert "G          | 1 2 3, 4" in text
    assert (tmp_path / "work" / "readings.json").exists()
    assert (tmp_path / "work" / "review.md").exists()
    # header + pass 1 (batch [0,1]) + pass 2 ([0],[1]) + two deciding readings
    assert len(calls) == 1 + 1 + 2 + 2
    assert all("-x2.png" in c for c in calls[1:4]) and all("-x3.png" in c for c in calls[4:])
    big = Path(rows_in(calls[-1])[0][1])
    assert Image.open(big).width == 3 * Image.open(result.readings[-1].crop).width


def test_unsettled_rows_go_to_review(tmp_path: Path) -> None:
    page = tmp_path / "song.png"
    drawn_page(systems=2).save(page)
    answers = iter(["1 2 3 4", "1 2 3, 4", "1, 2 3 4"])  # three different readings of row 0

    def run(prompt: str) -> str:
        if "页眉" in prompt:
            return '{"title": "t", "key": "C", "capo": 0, "time": "4/4"}'
        out = []
        for row, _ in rows_in(prompt):
            melody = next(answers) if row == 0 else "5 5 5 5"
            out.append(
                {"row": row, "bars": [{"chords": "C", "melody": melody}, {"melody": "1 1 1 1"}]}
            )
        return json.dumps(out)

    result = reader.read_sheet(
        page, tmp_path / "work", runner=run, batch_size=3, log=lambda _: None
    )
    assert result.unsettled == [0] and [p.index for p in result.problems] == [0]
    assert "三次读法不一致：'1 2 3 4' / '1 2 3, 4' / '1, 2 3 4'" in result.readings[0].notes  # noqa: RUF001
    assert (
        result.readings[0].melody == "1, 2 3 4" and result.readings[1].notes == ""
    )  # bar 1 agreed
    review = (tmp_path / "work" / "review.md").read_text(encoding="utf-8")
    assert "小节 0（第 0 行）" in review and "小节 1" not in review  # noqa: RUF001


def test_settle_row_votes_bar_by_bar() -> None:
    def row(*melodies: str) -> reader.RowReading:
        return [reader.BarReading(-1, "C", m) for m in melodies]

    # bar 0 agreed by readings 1 and 3, bar 1 by 1 and 2, bar 2 by nobody
    settled, open_bars = reader.settle_row(
        [
            row("1 2 3 4", "5 5 5 5", "1 1 1 1"),
            row("1 2 3, 4", "5 5 5 5", "2 2 2 2"),
            row("1 2 3 4", "5 5 5, 5", "3 3 3 3"),
        ],
        4,
    )
    assert [b.melody for b in settled] == ["1 2 3 4", "5 5 5 5", "3 3 3 3"]
    assert [b.melody for b in open_bars] == ["3 3 3 3"] and "三次读法不一致" in open_bars[0].notes
    # the only reading that passes the checks wins even against two agreeing invalid ones
    settled, open_bars = reader.settle_row([row("1 2 3"), row("1 2 3"), row("1 2 3 4")], 4)
    assert [b.melody for b in settled] == ["1 2 3 4"] and open_bars == []
    # two valid readings that disagree: the deciding one is kept, the bar stays open
    settled, open_bars = reader.settle_row([row("1 2 3"), row("1 1 1 1"), row("2 2 2 2")], 4)
    assert [b.melody for b in settled] == ["2 2 2 2"] and len(open_bars) == 1
    # the majority bar count decides which readings take part
    settled, open_bars = reader.settle_row(
        [row("1 1 1 1", "2 2 2 2"), row("1 1 1 1 2 2 2 2"), row("1 1 1 1", "2 2 2 2")], 4
    )
    assert [b.melody for b in settled] == ["1 1 1 1", "2 2 2 2"] and open_bars == []
    # three different bar counts: the deciding reading is kept, the whole row is open
    settled, open_bars = reader.settle_row([row("1"), row("1", "2"), row("1", "2", "3")], 4)
    assert len(settled) == 3 and open_bars == settled and "小节数不同" in settled[0].notes


def test_read_sheet_gives_up_on_a_garbled_batch(tmp_path: Path) -> None:
    page = tmp_path / "song.png"
    drawn_page().save(page)

    def bad(prompt: str) -> str:
        return '{"title": "t", "key": "C"}' if "页眉" in prompt else "sorry"

    with pytest.raises(RuntimeError, match="unreadable"):
        reader.read_sheet(page, tmp_path / "work", runner=bad, retries=1, log=lambda _: None)


def test_read_sheet_needs_a_system(tmp_path: Path) -> None:
    page = tmp_path / "blank.png"
    Image.new("L", (1200, 1600), 255).save(page)
    with pytest.raises(RuntimeError, match="no systems"):
        reader.read_sheet(page, tmp_path / "work", runner=lambda _: "{}", log=lambda _: None)


def test_claude_runner_retries_transient_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    outcomes = iter([1, 1, 0])
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        code = next(outcomes)
        return subprocess.CompletedProcess(cmd, code, stdout="ok" if code == 0 else "", stderr="x")

    monkeypatch.setattr(subprocess, "run", fake_run)
    sleeps: list[float] = []
    monkeypatch.setattr(reader.time, "sleep", sleeps.append)
    assert reader.claude_runner(backoff=(1.0, 2.0))("p") == "ok"
    assert len(calls) == 3 and sleeps == [1.0, 2.0]
    outcomes = iter([1, 1, 1])
    with pytest.raises(RuntimeError, match="claude -p failed"):
        reader.claude_runner(backoff=(1.0, 2.0))("p")


def test_failure_pause_waits_for_a_reported_reset() -> None:
    assert reader.failure_pause("network error", 30.0, 1000.0) == 30.0
    assert reader.failure_pause("Claude AI usage limit reached|1000003600", 30.0, 1000000000.0) == (
        3660.0
    )
    assert reader.failure_pause("usage limit reached|1000000010", 300.0, 1000000000.0) == 300.0
