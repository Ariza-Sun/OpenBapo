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
    """Answers the header, then every bar; bar 1 is misread (3 beat items) until re-read larger."""

    def run(prompt: str) -> str:
        calls.append(prompt)
        if "页眉" in prompt:
            return '{"title": "t", "key": "D", "capo": 2, "time": "4/4", "bpm": null}'
        items = [line.split("\t") for line in prompt.splitlines() if "\t" in line]
        out = []
        for index, path in items:
            i = int(index)
            melody = "1 2 3" if i == 1 and "-x2" not in path else "1 2 3 4"
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
    assert result.header["key"] == "D" and result.reread == [1]
    assert [r.melody for r in result.readings] == ["1 2 3 4"] * 3  # bar 1 fixed on the second look
    assert result.problems == []
    text = result.sheet_path.read_text(encoding="utf-8")
    assert text.startswith("title: t\nkey: D\nmode: major\ncapo: 2\nbpm: 100\ntime: 4/4\n---\n")
    assert "G          | 1 2 3 4" in text
    assert (tmp_path / "work" / "readings.json").exists() and (
        tmp_path / "work" / "review.md"
    ).exists()
    # header + two batches (3 bars, batch size 2) + one re-read
    assert len(calls) == 4 and any("-x2.png" in c for c in calls)
    big = next(Path(c.split("\t")[1].strip()) for c in calls[-1].splitlines() if "\t" in c)
    assert Image.open(big).width == 2 * Image.open(result.readings[1].crop).width


def test_read_sheet_gives_up_on_a_garbled_batch(tmp_path: Path) -> None:
    page = tmp_path / "song.png"
    drawn_page().save(page)

    def bad(prompt: str) -> str:
        return '{"title": "t", "key": "C"}' if "页眉" in prompt else "sorry"

    with pytest.raises(RuntimeError, match="unreadable"):
        reader.read_sheet(page, tmp_path / "work", runner=bad, retries=1, log=lambda _: None)
