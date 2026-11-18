"""Bounded upload inspection. Paths are generated IDs, never attachment names."""

import asyncio
import json
import math
import re
from pathlib import Path

from openbapo.cli import AUDIO_EXTENSIONS
from openbapo.service.config import ServiceError, Settings


def work_path(settings: Settings, job_id: str) -> Path:
    if re.fullmatch(r"[0-9a-f]{32}", job_id) is None:
        raise ValueError("invalid job id")
    return settings.data_dir / "work" / job_id


def check_attachment(filename: str, size: int, settings: Settings) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in AUDIO_EXTENSIONS:
        raise ServiceError("format")
    if size <= 0:
        raise ServiceError("corrupt")
    if size > settings.max_upload_bytes:
        raise ServiceError("size")
    return suffix


async def inspect_audio(path: Path) -> float:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels:format=duration",
        "-of",
        "json",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        raw, _ = await asyncio.wait_for(proc.communicate(), 15)
    except (TimeoutError, asyncio.CancelledError):
        proc.kill()
        await proc.wait()
        raise
    try:
        data = json.loads(raw)
        duration = float(data["format"]["duration"])
        channels = int(data["streams"][0]["channels"])
        if proc.returncode or not math.isfinite(duration) or duration <= 0 or channels <= 0:
            raise ValueError("invalid audio")
    except (KeyError, ValueError, IndexError, TypeError) as exc:
        raise ServiceError("corrupt") from exc
    if duration > 480:
        raise ServiceError("long")
    return duration
