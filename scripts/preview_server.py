"""Local web preview of the backend: upload a song, watch the job, view and adjust the sheet.

Usage:
    python scripts/preview_server.py [--port 8765] [--data-dir var/preview] [--device cuda]

Then, from your own computer:
    ssh -L 8765:127.0.0.1:8765 USER@HOST     # and open http://127.0.0.1:8765

A developer tool, not part of the product. It listens on 127.0.0.1 only and
has no login, so never expose it publicly. It runs the same job store,
analysis child process and renderer as the Discord bot (src/openbapo/service),
with the daily quota lifted and its own data directory. As in the service,
the upload and its stems are deleted when analysis ends; results expire
after 30 days or when deleted on the page.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from aiohttp import BodyPartReader, web

from openbapo.arrange import ArrangedSheet
from openbapo.model import STRUM_AUTO, TRANSPOSE_RANGE, Analysis, Arrangement
from openbapo.render import render_musicxml, render_svg, svg_pages_to_pdf, svg_to_png
from openbapo.service.config import ServiceError, Settings
from openbapo.service.media import check_attachment, inspect_audio, work_path
from openbapo.service.rendering import sheet_for
from openbapo.service.store import Job, Store
from openbapo.service.worker import InstanceLock, Worker

OWNER = 0  # one local user
MAX_UPLOAD = 100 * 1024 * 1024

ERRORS = {
    "format": "不支持的格式（支持 mp3 / m4a / ogg / opus / wav / flac）",
    "size": "文件太大",
    "corrupt": "文件无法解码",
    "long": "超过 8 分钟",
    "busy": "队列已满",
    "timeout": "处理超时（15 分钟）",
    "service": "分析出错（服务端异常）",
    "restarted": "服务重启，任务中断",
    "expired": "任务不存在或已过期",
    "pending": "还没分析完",
    "option": "无效的选项",
}

T = TypeVar("T")


def options_from(query: Any) -> Arrangement:
    try:
        transpose = int(query.get("transpose", "0"))
        fingering = query.get("fingering", "")
        return Arrangement(
            transpose=max(-TRANSPOSE_RANGE, min(TRANSPOSE_RANGE, transpose)),
            fingering_key=int(fingering) if fingering not in ("", "auto") else None,
            strum_pattern=query.get("strum", STRUM_AUTO) or STRUM_AUTO,
            simplify_chords=query.get("simplify", "1") != "0",
        )
    except ValueError as exc:
        raise ServiceError("option") from exc


def job_row(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "title": job.title,
        "status": job.status,
        "error": ERRORS.get(job.error or "", job.error),
        "created": job.created,
        "timings": json.loads(job.timings) if job.timings else None,
    }


def summary(job: Job, sheet: ArrangedSheet) -> dict[str, Any]:
    analysis = Analysis.from_json(job.analysis or "{}")
    lines: dict[int, list[str]] = {}
    for syl in analysis.lyrics:
        lines.setdefault(syl.line_index, []).append(syl.text)
    lyric_lines = []
    for parts in lines.values():
        text = ""
        for p in parts:
            latin = p[:1].isascii() and p[:1].isalnum()
            if text and latin and not text.endswith("-"):
                text += " "
            text += p
        lyric_lines.append(text.replace("-", ""))
    key = analysis.key
    return {
        "duration": analysis.duration,
        "bpm": analysis.tempo.bpm if analysis.tempo else None,
        "time_signature": sheet.header.time_signature,
        "key_confidence": key.confidence if key else None,
        "chords": len(analysis.chords),
        "low_confidence_chords": sum(c.low_confidence for b in sheet.bars for c in b.chords),
        "bar_chords": sum(len(b.chords) for b in sheet.bars),
        "bars": len(sheet.bars),
        "syllables": len(analysis.lyrics),
        "lyrics": lyric_lines,
        "timings": json.loads(job.timings) if job.timings else None,
    }


class Preview:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = Store(settings)
        self.lock = InstanceLock(settings.data_dir)
        self.worker = Worker(self.store, lock_fd=self.lock.file.fileno())
        self.render_gate = asyncio.Semaphore(1)
        self.tasks: list[asyncio.Task[None]] = []

    async def offload(self, fn: Callable[[], T]) -> T:
        async with self.render_gate:
            return await asyncio.get_running_loop().run_in_executor(None, fn)

    async def start(self, _: web.Application) -> None:
        self.worker.recover()
        self.store.sweep()
        self.tasks = [asyncio.create_task(self.worker.run()), asyncio.create_task(self.janitor())]

    async def stop(self, _: web.Application) -> None:
        for task in self.tasks:
            task.cancel()
        for task in self.tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self.lock.close()

    async def janitor(self) -> None:
        while True:
            self.worker.expire_waiting()
            self.store.sweep()
            await asyncio.sleep(30)

    # --- routes ------------------------------------------------------------------

    async def index(self, _: web.Request) -> web.Response:
        page = (Path(__file__).with_name("preview_page.html")).read_text(encoding="utf-8")
        return web.Response(text=page, content_type="text/html", charset="utf-8")

    async def list_jobs(self, _: web.Request) -> web.Response:
        jobs = sorted(self.store.list_jobs(), key=lambda j: j.created, reverse=True)
        return web.json_response([job_row(j) for j in jobs])

    async def submit(self, request: web.Request) -> web.Response:
        job_id = uuid.uuid4().hex
        work = work_path(self.settings, job_id)
        work.mkdir(mode=0o700)
        queued = False
        try:
            fields: dict[str, str] = {}
            audio_path: Path | None = None
            reader = await request.multipart()
            async for part in reader:
                if not isinstance(part, BodyPartReader):
                    continue
                if part.name == "audio" and part.filename:
                    suffix = check_attachment(part.filename, 1, self.settings)
                    audio_path = work / ("input" + suffix)
                    size = 0
                    with audio_path.open("wb") as f:
                        while chunk := await part.read_chunk(1 << 16):
                            size += len(chunk)
                            if size > self.settings.max_upload_bytes:
                                raise ServiceError("size")
                            f.write(chunk)
                    fields["filename"] = part.filename
                elif part.name in ("title", "lyrics"):
                    fields[part.name] = (await part.text()).strip()
            if audio_path is None:
                raise ServiceError("format")
            check_attachment(audio_path.name, audio_path.stat().st_size, self.settings)
            await inspect_audio(audio_path)
            title = (fields.get("title") or fields["filename"].rsplit(".", 1)[0])[:200]
            payload = {
                "audio": audio_path.name,
                "title": title,
                "artist": None,
                "lyrics": fields.get("lyrics") or None,
            }
            (work / "request.json").write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
            self.store.submit(job_id, OWNER, "zh-CN", title)
            queued = True
            return web.json_response({"id": job_id})
        except ServiceError as exc:
            return web.json_response({"error": ERRORS.get(exc.code, exc.code)}, status=400)
        finally:
            if not queued:
                await asyncio.get_running_loop().run_in_executor(None, _rmtree, work)

    async def delete(self, request: web.Request) -> web.Response:
        job_id = request.match_info["id"]
        with contextlib.suppress(ServiceError):
            job = self.store.get(job_id)
            if job.status in {"queued", "running"}:
                self.store.fail(job_id, "service")
            await self.worker.erase([job])
        return web.json_response({"ok": True})

    async def sheet(self, request: web.Request) -> web.Response:
        try:
            job = self.store.get(request.match_info["id"])
            options = options_from(request.query)

            def build() -> dict[str, Any]:
                started = time.perf_counter()
                sheet = sheet_for(job, options)
                pages = render_svg(sheet)
                return {
                    "header": sheet.header.model_dump(mode="json"),
                    "fingering_options": [
                        o.model_dump(mode="json") for o in sheet.fingering_options
                    ],
                    "strum": sheet.strum.id if sheet.strum else "none",
                    "strum_options": [o.model_dump(mode="json") for o in sheet.strum_options],
                    "warnings": list(sheet.warnings),
                    "pages": pages,
                    "summary": summary(job, sheet),
                    "render_seconds": time.perf_counter() - started,
                }

            return web.json_response(await self.offload(build))
        except ServiceError as exc:
            return web.json_response({"error": ERRORS.get(exc.code, exc.code)}, status=400)

    async def export(self, request: web.Request) -> web.Response:
        try:
            job = self.store.get(request.match_info["id"])
            options = options_from(request.query)
            fmt = request.match_info["fmt"]

            def build() -> tuple[str, bytes, str]:
                sheet = sheet_for(job, options)
                if fmt == "musicxml":
                    return ".musicxml", render_musicxml(sheet).encode(), "application/xml"
                pages = render_svg(sheet)
                if fmt == "pdf":
                    return ".pdf", svg_pages_to_pdf(pages), "application/pdf"
                if fmt == "png":
                    return ".png", svg_to_png(pages[0], dpi=200), "image/png"
                raise ServiceError("option")

            suffix, body, mime = await self.offload(build)
            name = "".join(c for c in job.title if c.isalnum() or c in " -_")[:60] or "sheet"
            return web.Response(
                body=body,
                content_type=mime,
                headers={
                    "Content-Disposition": f"attachment; filename*=UTF-8''{_quote(name)}{suffix}"
                },
            )
        except ServiceError as exc:
            return web.json_response({"error": ERRORS.get(exc.code, exc.code)}, status=400)


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def _quote(text: str) -> str:
    from urllib.parse import quote

    return quote(text)


def build_app(settings: Settings) -> web.Application:
    preview = Preview(settings)
    app = web.Application(client_max_size=settings.max_upload_bytes + (1 << 20))
    app.on_startup.append(preview.start)
    app.on_cleanup.append(preview.stop)
    app.router.add_get("/", preview.index)
    app.router.add_get("/api/jobs", preview.list_jobs)
    app.router.add_post("/api/jobs", preview.submit)
    app.router.add_delete("/api/jobs/{id}", preview.delete)
    app.router.add_get("/api/jobs/{id}/sheet", preview.sheet)
    app.router.add_get("/api/jobs/{id}/export/{fmt}", preview.export)
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=Path("var/preview"))
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    args = parser.parse_args()
    settings = Settings(
        data_dir=args.data_dir.resolve(),
        device=args.device,
        max_upload_bytes=MAX_UPLOAD,
        daily_quota=100_000,
        queue_limit=50,
    )
    print(f"OpenBapo preview on http://127.0.0.1:{args.port}  (data: {settings.data_dir})")
    web.run_app(build_app(settings), host="127.0.0.1", port=args.port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
