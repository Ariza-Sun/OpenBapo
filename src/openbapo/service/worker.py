"""Serial GPU worker with cancellation, hard deadlines and crash recovery."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import shutil
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO

from openbapo.model import Analysis
from openbapo.service.config import ServiceError
from openbapo.service.media import work_path
from openbapo.service.store import Job, Store

logger = logging.getLogger(__name__)


class InstanceLock:
    def __init__(self, root: Path) -> None:
        self.file: IO[bytes] = (root / "service.lock").open("a+b")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError("another OpenBapo instance is using this data directory") from None

    def close(self) -> None:
        # Do not explicitly unlock: an inference child may still hold its inherited FD.
        self.file.close()


class Worker:
    def __init__(
        self,
        store: Store,
        *,
        lock_fd: int | None = None,
        command: Callable[[Path], list[str]] | None = None,
    ) -> None:
        self.store = store
        self.lock_fd = lock_fd
        self.command = command or (
            lambda path: [
                sys.executable,
                "-m",
                "openbapo.service.job",
                str(path),
                "--device",
                store.settings.device,
            ]
        )
        self.current: str | None = None
        self.idle = asyncio.Event()
        self.idle.set()

    def recover(self) -> None:
        self.store.recover()
        retained = {j.id for j in self.store.list_jobs() if j.status == "queued"}
        for path in (self.store.root / "work").iterdir():
            if path.is_dir() and path.name not in retained:
                shutil.rmtree(path)

    async def _stop(self, process: asyncio.subprocess.Process) -> None:
        # Kill the entire process group, including ffmpeg/model helper processes.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        await process.wait()

    async def run_once(self) -> bool:
        job = self.store.claim()
        if job is None:
            return False
        self.current = job.id
        self.idle.clear()
        work = work_path(self.store.settings, job.id)
        process: asyncio.subprocess.Process | None = None
        started = time.monotonic()
        try:
            if time.time() >= job.created + self.store.settings.deadline_seconds:
                self.store.fail(job.id, "timeout")
                return True
            temp = work / "tmp"
            temp.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            env.pop("DISCORD_TOKEN", None)
            env.update(TMPDIR=str(temp), OPENBAPO_PARENT_PID=str(os.getpid()))
            process = await asyncio.create_subprocess_exec(
                *self.command(work),
                env=env,
                start_new_session=True,
                pass_fds=() if self.lock_fd is None else (self.lock_fd,),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            while process.returncode is None:
                if self.store.get(job.id).status != "running":
                    await self._stop(process)
                    return True
                if time.time() >= job.created + self.store.settings.deadline_seconds:
                    await self._stop(process)
                    self.store.fail(job.id, "timeout")
                    return True
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(process.wait(), timeout=0.25)
            if time.time() >= job.created + self.store.settings.deadline_seconds:
                self.store.fail(job.id, "timeout")
            elif process.returncode != 0:
                self.store.fail(job.id, "service")
            else:
                payload = json.loads((work / "result.json").read_text(encoding="utf-8"))
                if "error" in payload:
                    self.store.fail(job.id, str(payload["error"]))
                else:
                    analysis = Analysis.model_validate(payload["analysis"])
                    self.store.complete(job.id, analysis, json.dumps(payload["timings"]))
        except asyncio.CancelledError:
            self.store.fail(job.id, "restarted")
            raise
        except (OSError, ValueError, ServiceError):
            self.store.fail(job.id, "service")
        finally:
            if process is not None:
                await self._stop(process)
            shutil.rmtree(work, ignore_errors=True)
            self.current = None
            self.idle.set()
            logger.info("job=%s elapsed=%.2fs", job.id, time.monotonic() - started)
        return True

    async def run(self) -> None:
        while True:
            if not await self.run_once():
                await asyncio.sleep(0.5)

    def expire_waiting(self) -> None:
        """Queued files have the same deadline even while the GPU is occupied."""
        for job in self.store.list_jobs():
            if job.status == "queued" and time.time() >= (
                job.created + self.store.settings.deadline_seconds
            ):
                self.store.fail(job.id, "timeout")
                shutil.rmtree(work_path(self.store.settings, job.id), ignore_errors=True)

    async def erase(self, jobs: list[Job]) -> None:
        ids = {j.id for j in jobs}
        if self.current in ids:
            await self.idle.wait()
        for job in jobs:
            shutil.rmtree(work_path(self.store.settings, job.id), ignore_errors=True)
        self.store.erase(list(ids))
