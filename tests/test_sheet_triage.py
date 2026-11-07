"""Sorting downloaded sheet files into ordered page sets: versions, duplicates, order, capo."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image

from opencapo.sheet import layout
from opencapo.sheet.triage import (
    apply_review,
    capo_hint,
    materialize,
    ordered_pages,
    plan_library,
    plan_song,
    probe,
    sample_name,
    source_family,
    table,
    to_json,
)


def page(path: Path, width: int = 1200, height: int = 1600, shade: int = 255) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (width, height), shade).save(path)
    return path


def test_names_and_capo_hints() -> None:
    assert sample_name("中学时代 capo=0") == "中学时代"
    assert sample_name("匆匆那年吉他谱") == "匆匆那年"
    assert sample_name("一次就好吉他谱(7056)") == "一次就好"
    assert sample_name("Dont‘ be so serious") == "Dont_be_so_serious"
    assert sample_name("???") == "song"
    assert capo_hint("出卖", "capo=4.png") == 4
    assert capo_hint("capO=2.png") == 2 and capo_hint("Capo 3") == 3 and capo_hint("x") is None


def test_page_order_is_evident_only_for_numbered_names(tmp_path: Path) -> None:
    def files(*names: str) -> list:
        return [probe(page(tmp_path / n)) for n in names]

    ordered, evident = ordered_pages(files("2.jpg", "10.jpg", "1.jpg"))
    assert [p.path.name for p in ordered] == ["1.jpg", "2.jpg", "10.jpg"] and evident
    ordered, evident = ordered_pages(files("下载 (2).png", "下载.png", "下载 (1).png"))
    assert [p.path.name for p in ordered] == ["下载.png", "下载 (1).png", "下载 (2).png"]
    assert evident
    ordered, evident = ordered_pages(files("tab-45-2.png", "tab-45-1.png"))
    assert [p.path.name for p in ordered] == ["tab-45-1.png", "tab-45-2.png"] and evident
    ordered, evident = ordered_pages(files("20230323081450.png", "20230323081448.png"))
    assert [p.path.name for p in ordered] == ["20230323081448.png", "20230323081450.png"]
    assert evident
    ordered, evident = ordered_pages(files("f14f56.png", "25dfd9.png"))
    assert [p.path.name for p in ordered] == ["25dfd9.png", "f14f56.png"] and not evident


def test_plan_song_picks_a_version_and_drops_duplicates_and_junk(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    song = raw / "单身情歌"
    for i in range(1, 4):
        page(song / f"a{i}.webp", width=1100, shade=250 - i)  # one version, 3 pages
    page(song / "640.jpg", width=1300, shade=200)  # another version, 2 pages, wider
    page(song / "640 (1).jpg", width=1300, shade=201)
    (song / "640 (2).jpg").write_bytes((song / "640.jpg").read_bytes())  # exact duplicate
    (song / "c.txt").write_text("https://example.com", encoding="utf-8")
    (song / "capo=3.png").write_bytes(b"not an image")

    plan = plan_song(song, raw)
    assert plan.name == "单身情歌" and plan.folder == "单身情歌"
    assert [p.name for p in plan.pages] == ["a1.webp", "a2.webp", "a3.webp"]
    assert plan.kind == "image" and plan.width == 1100 and plan.order_evident
    assert plan.versions == 2 and {p.name for p in plan.other_versions} == {
        "640.jpg",
        "640 (1).jpg",
    }
    assert [p.name for p in plan.duplicates] == ["640 (2).jpg"]
    assert {p.name for p in plan.junk} == {"c.txt", "capo=3.png"}
    assert plan.capo_hint == 3
    assert plan.problems == ["2 个版本，选了页数最多的"]


def test_duplicate_keeps_the_plain_name(tmp_path: Path) -> None:
    song = tmp_path / "s"
    page(song / "x (1).png")
    (song / "x.png").write_bytes((song / "x (1).png").read_bytes())
    plan = plan_song(song, tmp_path)
    assert [p.name for p in plan.pages] == ["x.png"]
    assert [p.name for p in plan.duplicates] == ["x (1).png"]


def test_library_plans_folders_loose_files_and_name_clashes(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    page(raw / "yellow" / "1.png")
    page(raw / "yellow " / "1.png")  # cleans to the same name
    page(raw / "Dont be so serious.png")
    (raw / "黑夜中_files").mkdir()
    (raw / "黑夜中_files" / "style.css").write_text("", encoding="utf-8")
    (raw / "empty").mkdir()
    (raw / "empty" / "1.txt").write_text("url", encoding="utf-8")
    (raw / ".DS_Store").write_bytes(b"\0")
    plans = plan_library(raw)
    assert [p.name for p in plans] == ["Dont_be_so_serious", "empty", "yellow_1", "yellow_2"]
    assert plans[0].folder == "Dont be so serious.png"
    assert plans[1].problems == ["没有可读的页面"] and not plans[1].pages
    assert "与另一目录同名（yellow）" in plans[2].problems
    text = table(plans)
    assert (
        "| Dont_be_so_serious | Dont be so serious.png | 1 | 1 | 确定 | — | 1200×1600 | — | — |  |"
        in text
    )
    assert "4 首；页序不确定 0 首；多版本 0 首；无页面 1 首。" in text
    assert "| 1200 | 3 |" in text
    rows = to_json(plans)
    assert rows[1]["pages"] == [] and rows[2]["name"] == "yellow_1"


def test_materialize_writes_ordered_greyscale_pages_the_layout_can_read(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    Image.new("RGB", (800, 1000), (255, 255, 255)).save(raw.parent / "b.png")  # not in raw
    page(raw / "song" / "2.jpg", width=800, height=1000, shade=100)
    page(raw / "song" / "1.jpg", width=800, height=1000, shade=250)
    plan = plan_song(raw / "song", raw)
    out = materialize(plan, tmp_path / "out")
    assert out == tmp_path / "out" / "song"
    assert sorted(p.name for p in out.iterdir()) == ["p01.png", "p02.png"]
    pages = layout.rasterize(out)
    assert [im.mode for im in pages] == ["L", "L"] and [im.getpixel((0, 0)) for im in pages] == [
        250,
        100,
    ]
    materialize(plan, tmp_path / "out")  # rerun replaces, never accumulates
    assert sorted(p.name for p in out.iterdir()) == ["p01.png", "p02.png"]
    with pytest.raises(FileNotFoundError):
        layout.rasterize(tmp_path / "out")  # a folder without pages


def test_review_settles_the_order_and_labels_the_source(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for name in ("f14f56.png", "25dfd9.png", "9ad6e5.png"):
        page(raw / "song" / name, shade=ord(name[0]))
    page(raw / "other" / "1.png")
    plans = plan_library(raw)
    song = next(p for p in plans if p.name == "song")
    assert not song.order_evident
    review = [
        {
            "name": "song",
            "order": [str(raw / "song" / "9ad6e5.png"), str(raw / "song" / "f14f56.png")],
            "evidence": "页码 1/2、2/2",
            "source": "山山吉他",
            "has_jianpu": True,
            "header": {"capo": 2},
            "notes": "第三张是缩略图",
        },
        {"name": "other", "source": "oguitar", "has_jianpu": False},
        {"name": "missing", "order": []},
        {"name": "other", "order": [str(raw / "other" / "nope.png")]},
    ]
    rejected = apply_review(plans, review)
    assert rejected == [
        "missing: not in the triage",
        f"other: pages not in the plan: ['{raw / 'other' / 'nope.png'}']",
    ]
    assert [p.name for p in song.pages] == ["9ad6e5.png", "f14f56.png"] and song.order_evident
    assert song.source == "山山吉他" and song.has_jianpu is True and song.header == {"capo": 2}
    assert song.problems == ["页序已核定（页码 1/2、2/2）", "核定时去掉 1 页", "第三张是缩略图"]
    other = next(p for p in plans if p.name == "other")
    assert other.source == "嗡嗡响乐谱" and other.has_jianpu is False and other.order_evident
    text = table(plans)
    assert "| song | song | 2 | 1 | 确定 | — | 1200×1600 | 山山吉他 | 有 | 页序已核定" in text
    assert "| other | other | 1 | 1 | 确定 | — | 1200×1600 | 嗡嗡响乐谱 | 无 |  |" in text
    assert "## 按来源（版式）分组" in text and "| 山山吉他 | 1 | song |" in text
    assert to_json(plans)[1]["header"] == {"capo": 2}


def test_source_family_strips_credits_domains_and_aliases() -> None:
    assert source_family("小LV吉他（革命吉他制谱）") == "小LV吉他"
    assert source_family("吉他世界网（guitarworld.com.cn）") == "吉他世界网"
    assert source_family("oguitar / 嗡嗡响乐谱（buzzmusic.cn）") == "嗡嗡响乐谱"
    assert source_family("17吉他 (17jita.com 中文吉他音乐社区)") == "17吉他"
    assert source_family("易谱库 yipuku.com（制谱：二少）") == "易谱库"
    assert source_family("毛瓜谱APP（知乎 @毛瓜）") == "未知"
    assert source_family("mzyege（知音琴谱）") == "知音琴谱"
    assert source_family("小磊吉他教室") == "小磊吉他" and source_family("未知") == "未知"
