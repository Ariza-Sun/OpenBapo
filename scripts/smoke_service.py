"""Exercise the real queued GPU worker and exports without logging into Discord."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from openbapo.service.config import Settings
from openbapo.service.media import check_attachment, inspect_audio, work_path
from openbapo.service.rendering import changed, export, preview
from openbapo.service.store import Store
from openbapo.service.worker import InstanceLock, Worker


async def run(audio: Path, device: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="openbapo-service-smoke-") as directory:
        settings = Settings(Path(directory), device=device)
        store = Store(settings)
        lock = InstanceLock(store.root)
        try:
            started = time.perf_counter()
            suffix = check_attachment(audio.name, audio.stat().st_size, settings)
            job_id = uuid.uuid4().hex
            work = work_path(settings, job_id)
            work.mkdir()
            path = work / ("input" + suffix)
            shutil.copyfile(audio, path)
            duration = await inspect_audio(path)
            (work / "request.json").write_text(
                json.dumps(
                    {
                        "audio": path.name,
                        "title": "Service smoke",
                        "lyrics": None,
                    }
                ),
                encoding="utf-8",
            )
            store.submit(job_id, 42, "zh-CN", "Service smoke")
            worker = Worker(store, lock_fd=lock.file.fileno())
            await worker.run_once()
            job = store.get(job_id)
            if job.status != "done":
                raise RuntimeError(f"worker failed: {job.error}")
            files = preview(job)
            total = time.perf_counter() - started
            adjust_start = time.perf_counter()
            options = changed(job, "transpose", "-2")
            adjusted = preview(job, options)
            adjust_seconds = time.perf_counter() - adjust_start
            exports = {fmt: len(export(job, fmt)[1]) for fmt in ("png", "pdf", "musicxml")}
            assert not work.exists(), "audio / stems survived completion"
            assert files.png.startswith(b"\x89PNG") and adjusted.png != files.png
            return {
                "generated_at": datetime.now(UTC).isoformat(),
                "source_sha256": source_sha256(),
                "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "duration_s": duration,
                "device": device,
                "local_submit_to_preview_s": total,
                "adjust_to_preview_s": adjust_seconds,
                "analysis_timings_s": json.loads(job.timings or "{}"),
                "export_bytes": exports,
                "audio_cleaned": True,
                "within_300s": total < 300,
                "adjust_within_5s": adjust_seconds <= 5,
                "includes": "local copy, validation, queue, fresh model process, analysis, preview",
                "excludes": (
                    "Discord upload/download, Gateway, Discord message delivery, image build"
                ),
                "weights": (
                    "already cached; includes model initialization, excludes weight downloads"
                ),
            }
        finally:
            lock.close()


def source_sha256() -> str:
    """Fingerprint the service, core and deployment files used by this run."""
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "src").rglob("*.py"))
    files += [root / "pyproject.toml", Path(__file__).resolve()]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks/step-6-service"))
    args = parser.parse_args()
    report = asyncio.run(run(args.audio, args.device))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(
        "# 第 6 步：本地任务到出谱验证\n\n"
        f"由 `scripts/smoke_service.py` 生成于 {report['generated_at']}。\n\n"
        f"- 音频时长：{report['duration_s']:.1f} 秒。\n"
        f"- 本地提交到预览：**{report['local_submit_to_preview_s']:.2f} 秒**。\n"
        f"- 调整演唱调到新预览：**{report['adjust_to_preview_s']:.2f} 秒**。\n"
        "- PNG、PDF、MusicXML 均生成；原音频与分轨临时目录已清理。\n\n"
        "从新子进程开始计时，包含模型初始化；权重已缓存，不含首次下载。\n"
        "不包含 Discord 网络上传、下载和消息投递，也不证明干净机器部署通过。\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
