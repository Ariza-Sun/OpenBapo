"""Lifecycle boundaries: a real database, child processes, ffmpeg and renderers."""

import asyncio
import sys
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest
from test_render import long_song

from openbapo.model import Analysis, Arrangement
from openbapo.service.config import ServiceError, Settings
from openbapo.service.media import check_attachment, inspect_audio, work_path
from openbapo.service.rendering import changed, export, preview, sheet_for
from openbapo.service.store import Store
from openbapo.service.worker import InstanceLock, Worker


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(Settings(tmp_path / "service", device="cpu"))


def submit(store: Store, owner: int = 42, **kwargs):
    return store.submit(uuid.uuid4().hex, owner, "zh-CN", "Song", **kwargs)


def finished(store: Store, analysis: Analysis | None = None):
    job = submit(store)
    store.claim()
    store.complete(job.id, analysis or long_song(8), "{}")
    return store.get(job.id)


def test_quota_is_atomic_and_deleting_does_not_reset_it(store: Store):
    def attempt(_):
        try:
            return submit(store)
        except ServiceError as exc:
            assert exc.code == "quota"
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted = [j for j in pool.map(attempt, range(12)) if j is not None]
    assert len(accepted) == 5
    assert store.remaining(42) == 0
    store.erase([j.id for j in accepted])
    assert store.remaining(42) == 0
    assert store.remaining(42, now=time.time() + 86400) == 5
    with store.connect() as db:
        row = db.execute("SELECT identity,used FROM quota").fetchone()
        assert len(row[0]) == 64 and row[1] == 5


def test_owner_access_and_roundtrip_persist(store: Store):
    job = finished(store)
    with pytest.raises(ServiceError, match="owner"):
        store.get(job.id, 99)
    with pytest.raises(ServiceError, match="owner"):
        store.update_arrangement(job.id, 99, Arrangement(transpose=2))
    options = changed(job, "transpose", "2")
    store.update_arrangement(job.id, 42, options)
    reopened = Store(store.settings)
    assert Arrangement.from_json(reopened.get(job.id, 42).arrangement) == options
    assert reopened.get(job.id).analysis == job.analysis


def test_transpose_keeps_shapes_and_reset_restores_defaults(store: Store):
    job = finished(store)
    before = sheet_for(job)
    options = changed(job, "transpose", "-2")
    after = sheet_for(job, options)
    assert after.header.fingering_key == before.header.fingering_key
    assert after.header.capo == 10
    assert [c.name for b in after.bars for c in b.chords] == [
        c.name for b in before.bars for c in b.chords
    ]
    assert changed(job, "reset", "") == Arrangement()
    for action, value in (("transpose", "7"), ("fingering", "100"), ("strum", "bad")):
        with pytest.raises(ServiceError, match="option"):
            changed(job, action, value)


def test_real_previews_and_three_exports(store: Store):
    job = finished(store, long_song(40))
    files = preview(job)
    assert files.png.startswith(b"\x89PNG") and files.pdf.startswith(b"%PDF")
    assert export(job, "pdf")[1].startswith(b"%PDF")
    assert b"score-partwise" in export(job, "musicxml")[1]
    name, data = export(job, "png")
    assert name.endswith(".zip")
    with ZipFile(BytesIO(data)) as archive:
        assert len(archive.namelist()) > 1
        assert all(archive.read(n).startswith(b"\x89PNG") for n in archive.namelist())


def test_expiry_erases_text_and_credentials(store: Store):
    job = submit(store, token="private-token", token_expires=time.time() + 1)
    store.claim()
    store.complete(job.id, Analysis(title="private title"), "{}")
    store.sweep(now=time.time() + 2)
    assert store.get(job.id).token is None
    assert store.sweep(now=job.expires + 1) == [job.id]
    with pytest.raises(ServiceError, match="expired"):
        store.get(job.id)
    assert b"private title" not in store.path.read_bytes()
    assert b"private-token" not in store.path.read_bytes()


def test_queue_limit_and_fifo(store: Store):
    limited = Store(replace(store.settings, queue_limit=2))
    first, second = submit(limited), submit(limited)
    assert limited.position(second) == 1
    with pytest.raises(ServiceError, match="busy"):
        submit(limited)
    assert limited.remaining(42) == 3
    assert limited.claim().id == first.id
    assert limited.claim().id == second.id
    assert limited.claim() is None


def test_instance_lock_excludes_a_second_service(store: Store):
    lock = InstanceLock(store.root)
    try:
        with pytest.raises(RuntimeError, match="another"):
            InstanceLock(store.root)
    finally:
        lock.close()
    InstanceLock(store.root).close()


def fake_command(work: Path) -> list[str]:
    return [
        sys.executable,
        "-c",
        (
            "import json,os,sys; from pathlib import Path; "
            "assert 'DISCORD_TOKEN' not in os.environ; "
            "p=Path(sys.argv[1]); (p/'tmp'/'stem.wav').write_bytes(b'temporary'); "
            "(p/'result.json').write_text(json.dumps({'analysis':{},'timings':{'total':0.1}}))"
        ),
        str(work),
    ]


def test_worker_runs_and_removes_all_audio(store: Store, monkeypatch):
    monkeypatch.setenv("DISCORD_TOKEN", "must-not-reach-child")
    job = submit(store)
    work = work_path(store.settings, job.id)
    work.mkdir()
    (work / "input.wav").write_bytes(b"original")
    assert asyncio.run(Worker(store, command=fake_command).run_once())
    assert store.get(job.id).status == "done"
    assert not work.exists()
    assert not asyncio.run(Worker(store).run_once())


def test_timeout_kills_process_and_cleans_stems(store: Store):
    store = Store(replace(store.settings, deadline_seconds=0.6))
    job = submit(store)

    def command(p):
        return [sys.executable, "-c", "import time; time.sleep(30)"]

    started = time.monotonic()
    asyncio.run(Worker(store, command=command).run_once())
    assert time.monotonic() - started < 5
    assert store.get(job.id).error == "timeout"
    assert not work_path(store.settings, job.id).exists()


def test_queued_deadline_does_not_start_inference(store: Store):
    job = submit(store, now=time.time() - 901)

    def should_not_run(_):
        pytest.fail("expired queued job ran inference")

    asyncio.run(Worker(store, command=should_not_run).run_once())
    assert store.get(job.id).error == "timeout"


def test_waiting_audio_expires_while_worker_is_busy(store: Store):
    running = submit(store)
    store.claim()
    old = submit(store, now=time.time() - 901)
    work_path(store.settings, old.id).mkdir()
    Worker(store).expire_waiting()
    assert store.get(running.id).status == "running"
    assert store.get(old.id).error == "timeout"
    assert not work_path(store.settings, old.id).exists()


def test_deletion_stops_running_job_and_prevents_resurrection(store: Store):
    async def run():
        job = submit(store)
        worker = Worker(
            store, command=lambda p: [sys.executable, "-c", "import time; time.sleep(30)"]
        )
        task = asyncio.create_task(worker.run_once())
        while worker.current is None:
            await asyncio.sleep(0.01)
        jobs = store.cancel_owner(42)
        await worker.erase(jobs)
        await task
        store.complete(job.id, Analysis(), "{}")
        assert store.list_jobs() == []
        assert not work_path(store.settings, job.id).exists()

    asyncio.run(run())


def test_recovery_keeps_queued_and_cleans_interrupted_jobs(store: Store):
    running, queued = submit(store), submit(store)
    store.claim()
    orphan = uuid.uuid4().hex
    for identifier in (running.id, queued.id, orphan):
        work_path(store.settings, identifier).mkdir()
    Worker(store).recover()
    assert store.get(running.id).error == "restarted"
    assert store.get(queued.id).status == "queued"
    assert {p.name for p in (store.root / "work").iterdir()} == {queued.id}


def test_failed_child_is_reported_and_cleaned(store: Store):
    job = submit(store)
    asyncio.run(
        Worker(store, command=lambda p: [sys.executable, "-c", "raise SystemExit(1)"]).run_once()
    )
    assert store.get(job.id).error == "service"
    assert not work_path(store.settings, job.id).exists()


def test_media_validation_uses_real_probe(store: Store, tone: Path):
    assert asyncio.run(inspect_audio(tone)) == pytest.approx(3, abs=0.01)
    bad = tone.with_name("fake.mp3")
    bad.write_text("not audio")
    with pytest.raises(ServiceError, match="corrupt"):
        asyncio.run(inspect_audio(bad))
    for name, size, error in (
        ("a.exe", 1, "format"),
        ("a.wav", 0, "corrupt"),
        ("a.mp3", 11 * 1024**2, "size"),
    ):
        with pytest.raises(ServiceError, match=error):
            check_attachment(name, size, store.settings)
    with pytest.raises(ValueError):
        work_path(store.settings, "../../outside")


def test_media_rejects_long_audio_before_inference(tmp_path: Path):
    path = tmp_path / "long.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\0\0" * (8000 * 481))
    with pytest.raises(ServiceError, match="long"):
        asyncio.run(inspect_audio(path))
