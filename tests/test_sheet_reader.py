"""The one-command reading loop, with a fake model."""

import json
from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image
from test_sheet_omr import drawn_page

from opencapo.sheet import reader


def test_extract_json_and_prompts() -> None:
    assert reader.extract_json('x ```json\n[{"index": 1}]\n``` y') == [{"index": 1}]
    assert reader.extract_json('{"a": 1} trailing', "object") == {"a": 1}
    with pytest.raises(ValueError):
        reader.extract_json("nothing")
    prompt = reader.batch_prompt([(3, "/a.png"), (4, "/b.png")])
    assert "3\t/a.png" in prompt and "4\t/b.png" in prompt and "JSON 数组" in prompt
    assert "/h.png" in reader.header_prompt("/h.png")
    parsed = reader.parse_batch('[{"index": 3, "chords": "C", "melody": "1 2 3 4"}]', [3])
    assert parsed[3].chords == "C"
    with pytest.raises(ValueError):
        reader.parse_batch('[{"index": 3, "chords": "C"}]', [3, 4])  # bar 4 missing


def fake_runner(calls: list[str]) -> reader.Runner:
    """Answers the header, then every bar.

    Bar 1 is read without its octave dot in pass 1 (batched with bar 0) and
    with it in pass 2 (batched with bar 2) and in the deciding x3 reading;
    bar 2 is read with three beat items in both passes (a check failure)
    and right from x3.
    """

    def run(prompt: str) -> str:
        calls.append(prompt)
        if "页眉" in prompt:
            return '{"title": "t", "key": "D", "capo": 2, "time": "4/4", "bpm": null}'
        items = [line.split("\t") for line in prompt.splitlines() if "\t" in line]
        batch = {int(index) for index, _ in items}
        out = []
        for index, path in items:
            i = int(index)
            deciding = "-x3" in path
            if i == 1:  # misses the dot only when read in the same batch as bar 0 (pass 1)
                melody = "1 2 3, 4" if deciding or 0 not in batch else "1 2 3 4"
            elif i == 2:
                melody = "1 2 3 4" if deciding else "1 2 3"
            else:
                melody = "1 2 3 4"
            out.append({"index": i, "chords": "C" if i != 1 else "G", "melody": melody})
        return json.dumps(out)

    return run


def test_read_sheet_loop(tmp_path: Path) -> None:
    page = tmp_path / "song.png"
    drawn_page().save(page)
    calls: list[str] = []
    logs: list[str] = []
    result = reader.read_sheet(
        page, tmp_path / "work", runner=fake_runner(calls), bpm=100, batch_size=2, log=logs.append
    )
    assert result.header["key"] == "D"
    assert result.reread == [1, 2] and result.unsettled == []
    assert [r.melody for r in result.readings] == ["1 2 3 4", "1 2 3, 4", "1 2 3 4"]
    assert result.problems == []
    text = result.sheet_path.read_text(encoding="utf-8")
    assert text.startswith("title: t\nkey: D\nmode: major\ncapo: 2\nbpm: 100\ntime: 4/4\n---\n")
    assert "G          | 1 2 3, 4" in text
    assert (tmp_path / "work" / "readings.json").exists()
    assert (tmp_path / "work" / "review.md").exists()
    # header + pass 1 (batches [0,1],[2]) + pass 2 ([0],[1,2]) + two deciding readings
    assert len(calls) == 1 + 2 + 2 + 2
    assert all("-x2.png" in c for c in calls[1:5]) and all("-x3.png" in c for c in calls[5:])
    big = next(Path(c.split("\t")[1].strip()) for c in calls[-1].splitlines() if "\t" in c)
    assert Image.open(big).width == 3 * Image.open(result.readings[2].crop).width


def test_unsettled_bars_go_to_review(tmp_path: Path) -> None:
    page = tmp_path / "song.png"
    drawn_page().save(page)
    answers = iter(["1 2 3 4", "1 2 3, 4", "1, 2 3 4"])  # three different readings of bar 0

    def run(prompt: str) -> str:
        if "页眉" in prompt:
            return '{"title": "t", "key": "C", "capo": 0, "time": "4/4"}'
        items = [line.split("\t") for line in prompt.splitlines() if "\t" in line]
        out = []
        for index, _ in items:
            i = int(index)
            melody = next(answers) if i == 0 else "5 5 5 5"
            out.append({"index": i, "chords": "C", "melody": melody})
        return json.dumps(out)

    result = reader.read_sheet(
        page, tmp_path / "work", runner=run, batch_size=3, log=lambda _: None
    )
    assert result.unsettled == [0] and [p.index for p in result.problems] == [0]
    assert "三次读法不一致" in result.readings[0].notes
    assert "小节 0" in (tmp_path / "work" / "review.md").read_text(encoding="utf-8")


def test_read_sheet_gives_up_on_a_garbled_batch(tmp_path: Path) -> None:
    page = tmp_path / "song.png"
    drawn_page().save(page)

    def bad(prompt: str) -> str:
        return '{"title": "t", "key": "C"}' if "页眉" in prompt else "sorry"

    with pytest.raises(RuntimeError, match="unreadable"):
        reader.read_sheet(page, tmp_path / "work", runner=bad, retries=1, log=lambda _: None)
