"""Single-instance service configuration (no credentials in serialized jobs)."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    device: str = "cuda"
    recognizer: str = "qwen3-asr"  # lyrics recognizer: qwen3-asr (default) or whisper
    max_upload_bytes: int = 20 * 1024 * 1024
    daily_quota: int = 5
    queue_limit: int = 20
    deadline_seconds: float = 900.0
    retention_seconds: float = 30 * 86400.0
    contact: str | None = None  # shown to users for help with a failed job; None: not offered
    unlimited: frozenset[int] = frozenset()  # user IDs without a daily quota (the operator)

    def __post_init__(self) -> None:
        if self.device not in {"cuda", "cpu", "auto"}:
            raise ValueError("OPENBAPO_DEVICE must be cuda, cpu or auto")
        if self.recognizer not in {"whisper", "qwen3-asr"}:
            raise ValueError("OPENBAPO_RECOGNIZER must be whisper or qwen3-asr")
        if min(self.max_upload_bytes, self.daily_quota, self.queue_limit) <= 0:
            raise ValueError("upload, quota and queue limits must be positive")
        if not 0 < self.deadline_seconds <= 900 or self.retention_seconds <= 0:
            raise ValueError("invalid deadline or retention")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(os.environ.get("OPENBAPO_DATA_DIR", "./var/openbapo")).resolve(),
            device=os.environ.get("OPENBAPO_DEVICE", "cuda"),
            recognizer=os.environ.get("OPENBAPO_RECOGNIZER", "qwen3-asr"),
            max_upload_bytes=int(os.environ.get("OPENBAPO_MAX_UPLOAD_BYTES", 20 * 1024 * 1024)),
            contact=os.environ.get("OPENBAPO_CONTACT") or None,
            unlimited=frozenset(
                int(u)
                for u in os.environ.get("OPENBAPO_UNLIMITED_USERS", "").split(",")
                if u.strip()
            ),
        )


class ServiceError(Exception):
    """Stable error code translated by the presentation layer."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)
