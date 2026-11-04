"""Sorting a pile of downloaded sheet images into one ordered page set per song.

The owner's song book arrives as one folder per song, filled by hand over
years: pages as PNG / JPG / WebP / GIF or a PDF, often two versions of the
same sheet side by side, duplicate downloads (`x (1).png`), stray text
files with the source URL, the capo written only in a file name. Nothing
here reads a sheet; this module decides, deterministically, which files
are the pages of which song and in what order, and says what it was
unsure about so a person can check the table before the reading starts
(decision 13 in docs/plans/step-2.md).

Per song: open every file as an image (or count a PDF's pages), drop
exact duplicates, group the rest into *versions* (same type and page
width), keep the version with the most pages, order its pages by the
numbers in their names, and remember whether that order was evident.
Where it was not, an agent looks at the pages (printed page numbers,
the title block, where the lyrics continue) and its findings come back
through `apply_review`, which also brings the sheet's maker and whether
it has a numbered-notation line, the clues the pilot is grouped by.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
CAPO_PATTERN = re.compile(r"cap[oO]\s*[=＝]?\s*(\d{1,2})", re.IGNORECASE)
INDEX_PATTERN = re.compile(r"^(.*?)[ _-]*\((\d+)\)$")  # "下载 (1)" -> ("下载", 1)
NOISE_WORDS = ("吉他谱", "吉他譜", "guitar tab", "六线谱")
SMALL_PAGE = 600  # pages narrower than this are thumbnails, not readable sheets


@dataclass
class PageFile:
    path: Path
    kind: str  # "image" | "pdf"
    width: int
    height: int
    pages: int = 1  # a PDF's page count
    digest: str = ""


@dataclass
class SongPlan:
    name: str  # the sample name (a safe file stem)
    folder: str  # where it came from, relative to the raw root
    pages: list[Path] = field(default_factory=list)  # chosen version, in reading order
    kind: str = "image"
    width: int = 0
    height: int = 0
    versions: int = 1
    order_evident: bool = True
    capo_hint: int | None = None
    duplicates: list[Path] = field(default_factory=list)
    other_versions: list[Path] = field(default_factory=list)
    junk: list[Path] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    # From the review (an agent looking at the pages): who made the sheet,
    # whether it has a numbered-notation line, what the header prints.
    source: str = ""
    has_jianpu: bool | None = None
    header: dict[str, object] = field(default_factory=dict)
    review_notes: str = ""


def natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def capo_hint(*texts: str) -> int | None:
    for text in texts:
        m = CAPO_PATTERN.search(text)
        if m:
            return int(m.group(1))
    return None


def sample_name(folder_name: str) -> str:
    """A file stem for the song: the folder name without noise words, capo notes or symbols."""
    text = folder_name
    text = CAPO_PATTERN.sub(" ", text)
    for word in NOISE_WORDS:
        text = re.sub(re.escape(word), " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\d+\)", " ", text)  # a site's numeric id in parentheses
    text = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_")
    return text or "song"


def probe(path: Path) -> PageFile | None:
    """What this file is, judged by its content, or None when it is not a page."""
    if path.suffix.lower() == ".pdf":
        try:
            import pypdfium2

            pdf = pypdfium2.PdfDocument(str(path))
            count = len(pdf)
            if count == 0:
                return None
            w, h = pdf[0].get_size()
            return PageFile(path, "pdf", int(w), int(h), pages=count, digest=digest_of(path))
        except Exception:
            return None
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:  # pragma: no cover - the sheets extra is required for triage
        raise
    try:
        with Image.open(path) as image:
            image.load()
            return PageFile(path, "image", image.width, image.height, digest=digest_of(path))
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def digest_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def version_key(page: PageFile) -> tuple[str, str, int]:
    """Pages of one version share the type, the suffix and (roughly) the width."""
    suffix = page.path.suffix.lower() if page.kind == "image" else ".pdf"
    return page.kind, suffix, round(page.width / 50) * 50


def ordered_pages(pages: list[PageFile]) -> tuple[list[PageFile], bool]:
    """Pages in reading order, and whether the names made that order evident.

    Evident: the stems share a prefix and differ by a trailing number
    (`1.jpg, 2.jpg`; `tab-1.png, tab-2.png`; `下载.png, 下载 (1).png`), or
    are all numbers (timestamps, counters). Otherwise the files are sorted
    naturally by name, which is a guess the reader must confirm.
    """
    if len(pages) <= 1:
        return list(pages), True
    stems = [p.path.stem for p in pages]
    parsed = []
    for stem in stems:
        m = INDEX_PATTERN.match(stem)
        if m:
            parsed.append((m.group(1).strip(), int(m.group(2)) + 1))
        else:
            parsed.append((stem, 0))
    bases = {base for base, _ in parsed}
    if len(bases) == 1 and len({n for _, n in parsed}) == len(parsed):  # "下载", "下载 (1)" ...
        by_index = sorted(zip(parsed, pages, strict=True), key=lambda x: x[0][1])
        return [p for _, p in by_index], True
    if all(stem.isdigit() for stem in stems) and len(set(stems)) == len(stems):
        return sorted(pages, key=lambda p: int(p.path.stem)), True
    trailing = [re.match(r"^(.*?)(\d+)$", stem) for stem in stems]
    if all(trailing):
        prefixes = {m.group(1) for m in trailing if m}
        numbers = [int(m.group(2)) for m in trailing if m]
        if len(prefixes) == 1 and len(set(numbers)) == len(numbers):
            by_number = sorted(zip(numbers, pages, strict=True), key=lambda x: x[0])
            return [p for _, p in by_number], True
    return sorted(pages, key=lambda p: natural_key(p.path.name)), False


def plan_song(folder: Path, root: Path, name: str | None = None) -> SongPlan:
    """Decide the pages of the song in FOLDER (or of a single file)."""
    rel = folder.relative_to(root).as_posix() if folder != root else "."
    plan = SongPlan(name=name or sample_name(folder.stem), folder=rel)
    files = sorted(p for p in folder.rglob("*") if p.is_file()) if folder.is_dir() else [folder]
    pages: list[PageFile] = []
    seen: dict[str, PageFile] = {}
    for path in files:
        if path.name.startswith("."):
            continue
        page = probe(path)
        if page is None:
            plan.junk.append(path)
            continue
        if page.digest in seen:
            # Keep the plainer name ("x.png" over "x (1).png") as the page.
            kept = seen[page.digest]
            if INDEX_PATTERN.match(kept.path.stem) and not INDEX_PATTERN.match(path.stem):
                plan.duplicates.append(kept.path)
                kept.path = path
            else:
                plan.duplicates.append(path)
            continue
        seen[page.digest] = page
        pages.append(page)
    plan.capo_hint = capo_hint(folder.name, *(p.name for p in files))
    if not pages:
        plan.problems.append("没有可读的页面")
        return plan

    versions: dict[tuple[str, str, int], list[PageFile]] = {}
    for page in pages:
        versions.setdefault(version_key(page), []).append(page)
    # The most pages wins; among equals the wider (sharper) one.
    ranked = sorted(
        versions.values(),
        key=lambda v: (sum(p.pages for p in v), max(p.width for p in v)),
        reverse=True,
    )
    chosen, others = ranked[0], ranked[1:]
    plan.versions = len(ranked)
    plan.other_versions = [p.path for v in others for p in v]
    if others:
        plan.problems.append(f"{len(ranked)} 个版本，选了页数最多的")
    if chosen[0].kind == "pdf":
        best = max(chosen, key=lambda p: (p.pages, p.width))
        plan.pages = [best.path]
        plan.kind = "pdf"
        plan.width, plan.height = best.width, best.height
        if len(chosen) > 1:
            plan.other_versions += [p.path for p in chosen if p is not best]
            plan.problems.append(f"{len(chosen)} 个 PDF，选了页数最多的")
        return plan
    ordered, evident = ordered_pages(chosen)
    plan.pages = [p.path for p in ordered]
    plan.order_evident = evident
    plan.width, plan.height = ordered[0].width, ordered[0].height
    if not evident:
        plan.problems.append("页序按文件名猜的")
    if plan.width < SMALL_PAGE:  # images only: a PDF's size is in points, not pixels
        plan.problems.append(f"页面只有 {plan.width} 像素宽")
    return plan


def plan_library(root: Path) -> list[SongPlan]:
    """One plan per song folder, plus one per loose file at the root."""
    plans = []
    for entry in sorted(root.iterdir(), key=lambda p: natural_key(p.name)):
        if entry.name.startswith(".") or entry.name.startswith("__"):
            continue
        if entry.is_dir():
            if entry.name.endswith("_files"):  # a saved web page's assets
                continue
            plans.append(plan_song(entry, root))
        elif entry.suffix.lower() in IMAGE_SUFFIXES | {".pdf"}:
            plans.append(plan_song(entry, root, name=sample_name(entry.stem)))
    # Two folders can clean to the same name; keep both apart.
    counts: dict[str, int] = {}
    for plan in plans:
        counts[plan.name] = counts.get(plan.name, 0) + 1
    seen: dict[str, int] = {}
    for plan in plans:
        if counts[plan.name] > 1:
            seen[plan.name] = seen.get(plan.name, 0) + 1
            plan.problems.append(f"与另一目录同名（{plan.name}）")
            plan.name = f"{plan.name}_{seen[plan.name]}"
    return plans


def apply_review(plans: list[SongPlan], review: list[dict[str, object]]) -> list[str]:
    """Take page order and labels from a review of the pages; returns what did not apply.

    A review entry has the song's `name`, its pages in reading `order`
    (paths that must all come from the plan's pages), the `evidence`,
    `source`, `has_jianpu`, `header` and `notes`. A page the review left
    out is dropped (a thumbnail, an advert); a path it names that the
    plan does not have is an error and the song keeps its guessed order.
    """
    by_name = {p.name: p for p in plans}
    rejected = []
    for entry in review:
        name = str(entry.get("name", ""))
        plan = by_name.get(name)
        if plan is None:
            rejected.append(f"{name}: not in the triage")
            continue
        if entry.get("source"):
            plan.source = str(entry["source"])
        if entry.get("has_jianpu") is not None:
            plan.has_jianpu = bool(entry["has_jianpu"])
        header = entry.get("header")
        if isinstance(header, dict):
            plan.header = dict(header)
        if entry.get("notes"):
            plan.review_notes = str(entry["notes"])
        order = entry.get("order")
        if not isinstance(order, list) or not order:
            continue
        wanted = [Path(str(x)) for x in order]
        known = {p.resolve(): p for p in plan.pages}
        unknown = [str(x) for x in wanted if x.resolve() not in known]
        if unknown:
            rejected.append(f"{name}: pages not in the plan: {unknown}")
            continue
        dropped = len(plan.pages) - len(wanted)
        plan.pages = [known[x.resolve()] for x in wanted]
        plan.order_evident = True
        plan.problems = [x for x in plan.problems if x != "页序按文件名猜的"]
        plan.problems.append(f"页序已核定（{entry.get('evidence') or '无依据说明'}）")
        if dropped:
            plan.problems.append(f"核定时去掉 {dropped} 页")
        if plan.review_notes:
            plan.problems.append(plan.review_notes)
    return rejected


def materialize(plan: SongPlan, out_root: Path, dpi: int = 300) -> Path:
    """Write the chosen pages as OUT/<name>/p01.png ... (greyscale PNG), ready for the reader."""
    from PIL import Image

    folder = out_root / plan.name
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("p*.png"):
        old.unlink()
    images: list[Image.Image]
    if plan.kind == "pdf":
        from opencapo.sheet.layout import rasterize

        images = rasterize(plan.pages[0], dpi)
    else:
        images = []
        for path in plan.pages:
            with Image.open(path) as source:
                images.append(source.convert("L"))
    for i, page_image in enumerate(images, start=1):
        page_image.save(folder / f"p{i:02d}.png")
    return folder


def table(plans: list[SongPlan]) -> str:
    """A markdown table for the owner to check before anything is read."""
    lines = [
        "# 曲库整理表",
        "",
        f"{len(plans)} 首；页序"
        f"不确定 {sum(not p.order_evident for p in plans)} 首；"
        f"多版本 {sum(p.versions > 1 for p in plans)} 首；"
        f"无页面 {sum(not p.pages for p in plans)} 首。",
        "",
        "| 样本名 | 原目录 | 页数 | 版本 | 页序 | 变调夹提示 | 首页尺寸 | 来源 | 简谱行 | 问题 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in plans:
        count = str(len(p.pages)) if p.kind == "image" else "PDF"
        jianpu = "—" if p.has_jianpu is None else ("有" if p.has_jianpu else "无")
        lines.append(
            "| "
            + " | ".join(
                [
                    p.name,
                    p.folder,
                    count,
                    str(p.versions),
                    "确定" if p.order_evident else "猜的",
                    "—" if p.capo_hint is None else str(p.capo_hint),
                    f"{p.width}×{p.height}" if p.pages else "—",
                    p.source or "—",
                    jianpu,
                    "；".join(p.problems) or "",
                ]
            )
            + " |"
        )
    sources: dict[str, list[str]] = {}
    for p in plans:
        if p.pages:
            sources.setdefault(p.source or "未知", []).append(p.name)
    if any(p.source for p in plans):
        lines += ["", "## 按来源（版式）分组", "", "| 来源 | 首数 | 例子 |", "|---|---|---|"]
        for src, names in sorted(sources.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"| {src} | {len(names)} | {'、'.join(names[:4])} |")
    widths: dict[int, list[str]] = {}
    for p in plans:
        if p.pages and p.kind == "image":
            widths.setdefault(round(p.width / 100) * 100, []).append(p.name)
    lines += ["", "## 按首页宽度分组（版式线索）", "", "| 宽度 | 首数 | 例子 |", "|---|---|---|"]
    for w in sorted(widths):
        names = widths[w]
        lines.append(f"| {w} | {len(names)} | {'、'.join(names[:4])} |")
    return "\n".join(lines) + "\n"


def to_json(plans: list[SongPlan]) -> list[dict[str, object]]:
    return [
        {
            "name": p.name,
            "folder": p.folder,
            "kind": p.kind,
            "pages": [str(x) for x in p.pages],
            "width": p.width,
            "height": p.height,
            "versions": p.versions,
            "order_evident": p.order_evident,
            "capo_hint": p.capo_hint,
            "duplicates": [str(x) for x in p.duplicates],
            "other_versions": [str(x) for x in p.other_versions],
            "junk": [str(x) for x in p.junk],
            "problems": p.problems,
            "source": p.source,
            "has_jianpu": p.has_jianpu,
            "header": p.header,
        }
        for p in plans
    ]
