"""SQLite queue with transactional quota and owner-scoped access.

Only this service reads the database. Interaction tokens are short-lived delivery
credentials; erase them after delivery or expiry. Never log row contents.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

from openbapo.model import Analysis, Arrangement
from openbapo.service.config import ServiceError, Settings


@dataclass(frozen=True)
class Job:
    id: str
    owner: int
    locale: str
    status: str
    created: float
    expires: float
    title: str
    error: str | None
    analysis: str | None
    arrangement: str
    timings: str | None
    channel_id: int | None
    message_id: int | None
    application_id: int
    token: str | None
    token_expires: float
    notified: int
    retry_after: float
    stage: str


class Store:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.data_dir
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self.path = self.root / "jobs.sqlite3"
        (self.root / "work").mkdir(exist_ok=True, mode=0o700)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS quota (
                    day TEXT NOT NULL, identity TEXT NOT NULL, used INTEGER NOT NULL,
                    PRIMARY KEY (day, identity));
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, owner INTEGER NOT NULL, locale TEXT NOT NULL,
                    status TEXT NOT NULL, created REAL NOT NULL, expires REAL NOT NULL,
                    title TEXT NOT NULL, error TEXT, analysis TEXT, arrangement TEXT NOT NULL,
                    timings TEXT, channel_id INTEGER, message_id INTEGER,
                    application_id INTEGER NOT NULL, token TEXT, token_expires REAL NOT NULL,
                    notified INTEGER NOT NULL DEFAULT 0, retry_after REAL NOT NULL DEFAULT 0,
                    stage TEXT NOT NULL DEFAULT 'queued');
                CREATE INDEX IF NOT EXISTS jobs_owner ON jobs(owner);
                CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, created);
            """)
            db.execute(
                "INSERT OR IGNORE INTO settings VALUES ('quota_key', ?)",
                (secrets.token_hex(32),),
            )
            self._quota_key = bytes.fromhex(
                db.execute("SELECT value FROM settings WHERE key='quota_key'").fetchone()[0]
            )
        self.path.chmod(0o600)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA secure_delete=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def _identity(self, owner: int, now: float) -> tuple[str, str]:
        day = datetime.fromtimestamp(now, UTC).date().isoformat()
        identity = hmac.new(self._quota_key, f"{day}:{owner}".encode(), hashlib.sha256).hexdigest()
        return day, identity

    def remaining(self, owner: int, *, now: float | None = None) -> int | None:
        """Uploads left today; None for a user without a quota."""
        if owner in self.settings.unlimited:
            return None
        with self.connect() as db:
            row = db.execute(
                "SELECT used FROM quota WHERE day=? AND identity=?",
                self._identity(owner, time.time() if now is None else now),
            ).fetchone()
        return max(0, self.settings.daily_quota - (row[0] if row else 0))

    def submit(
        self,
        job_id: str,
        owner: int,
        locale: str,
        title: str,
        *,
        channel_id: int | None = None,
        message_id: int | None = None,
        application_id: int = 0,
        token: str | None = None,
        token_expires: float = 0,
        now: float | None = None,
    ) -> Job:
        now = time.time() if now is None else now
        day, identity = self._identity(owner, now)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            used = db.execute(
                "SELECT used FROM quota WHERE day=? AND identity=?", (day, identity)
            ).fetchone()
            unlimited = owner in self.settings.unlimited
            if used and used[0] >= self.settings.daily_quota and not unlimited:
                raise ServiceError("quota")
            pending = db.execute(
                "SELECT count(*) FROM jobs WHERE status IN ('queued','running')"
            ).fetchone()[0]
            if pending >= self.settings.queue_limit:
                raise ServiceError("busy")
            db.execute(
                "INSERT INTO jobs (id,owner,locale,status,created,expires,title,arrangement,"
                "channel_id,message_id,application_id,token,token_expires) "
                "VALUES (?,?,?,'queued',?,?,?,?,?,?,?,?,?)",
                (
                    job_id,
                    owner,
                    locale,
                    now,
                    now + self.settings.retention_seconds,
                    title,
                    Arrangement().to_json(),
                    channel_id,
                    message_id,
                    application_id,
                    token,
                    token_expires,
                ),
            )
            if not unlimited:
                db.execute(
                    "INSERT INTO quota VALUES (?,?,1) ON CONFLICT(day,identity) "
                    "DO UPDATE SET used=used+1",
                    (day, identity),
                )
        return self.get(job_id)

    def get(self, job_id: str, owner: int | None = None, *, now: float | None = None) -> Job:
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None or row["expires"] <= (time.time() if now is None else now):
            raise ServiceError("expired")
        if owner is not None and row["owner"] != owner:
            raise ServiceError("owner")
        return Job(**dict(row))

    def list_jobs(self, owner: int | None = None) -> list[Job]:
        query, args = "SELECT * FROM jobs WHERE expires>?", [time.time()]
        if owner is not None:
            query += " AND owner=?"
            args.append(owner)
        with self.connect() as db:
            return [Job(**dict(r)) for r in db.execute(query + " ORDER BY created", args)]

    def position(self, job: Job) -> int:
        with self.connect() as db:
            return int(
                db.execute(
                    "SELECT count(*) FROM jobs WHERE status IN ('queued','running') "
                    "AND (created<? OR (created=? AND rowid<(SELECT rowid FROM jobs WHERE id=?)))",
                    (job.created, job.created, job.id),
                ).fetchone()[0]
            )

    def claim(self) -> Job | None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT id FROM jobs WHERE status='queued' ORDER BY created,rowid LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running',stage='analysis' WHERE id=?", (row[0],))
        return self.get(row[0])

    def complete(self, job_id: str, analysis: Analysis, timings: str) -> None:
        # A cancelled/deleted job must never be resurrected by a late child result.
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status='done',stage='done',analysis=?,timings=? "
                "WHERE id=? AND status='running'",
                (analysis.to_json(), timings, job_id),
            )

    def fail(self, job_id: str, error: str) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status='failed',stage='failed',error=? "
                "WHERE id=? AND status IN ('queued','running')",
                (error, job_id),
            )

    def update_arrangement(self, job_id: str, owner: int, value: Arrangement) -> None:
        job = self.get(job_id, owner)
        if job.status != "done":
            raise ServiceError("pending")
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET arrangement=? WHERE id=? AND owner=?",
                (value.to_json(), job_id, owner),
            )

    def move(self, job_id: str, channel_id: int, message_id: int) -> None:
        """The job's progress now lives in another bot message (its song thread);
        the bot's own token edits it from here on, not the interaction's."""
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET channel_id=?,message_id=?,token=NULL WHERE id=?",
                (channel_id, message_id, job_id),
            )

    def delivery(
        self,
        job_id: str,
        *,
        channel_id: int | None = None,
        message_id: int | None = None,
        retry: bool = False,
    ) -> None:
        with self.connect() as db:
            if retry:
                db.execute("UPDATE jobs SET retry_after=? WHERE id=?", (time.time() + 60, job_id))
            else:
                db.execute(
                    "UPDATE jobs SET notified=1,token=NULL,channel_id=COALESCE(?,channel_id),"
                    "message_id=COALESCE(?,message_id) WHERE id=?",
                    (channel_id, message_id, job_id),
                )

    def recover(self) -> None:
        # Called only with the exclusive instance lock. Incomplete inference is not replayed.
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status='failed',stage='failed',error='restarted' "
                "WHERE status='running'"
            )

    def cancel_owner(self, owner: int) -> list[Job]:
        jobs = self.list_jobs(owner)
        with self.connect() as db:
            db.execute("UPDATE jobs SET status='cancelled',token=NULL WHERE owner=?", (owner,))
        return jobs

    def erase(self, job_ids: list[str]) -> None:
        with self.connect() as db:
            db.executemany("DELETE FROM jobs WHERE id=?", [(j,) for j in job_ids])
        self.checkpoint()

    def sweep(self, *, now: float | None = None) -> list[str]:
        now = time.time() if now is None else now
        with self.connect() as db:
            ids = [str(r[0]) for r in db.execute("SELECT id FROM jobs WHERE expires<=?", (now,))]
            db.execute("DELETE FROM jobs WHERE expires<=?", (now,))
            db.execute("DELETE FROM quota WHERE day<>?", (self._identity(0, now)[0],))
            db.execute("UPDATE jobs SET token=NULL WHERE token_expires<=?", (now,))
        self.checkpoint()
        return ids

    def checkpoint(self) -> None:
        with self.connect() as db:
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
