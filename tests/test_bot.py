"""Offline Discord contracts with real services and mocked network boundaries."""

import asyncio
import shutil
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest
from test_service import fake_command, finished

from openbapo.bot.app import OpenBapo
from openbapo.bot.messages import Translator, sheet_embed, t
from openbapo.bot.views import SheetView
from openbapo.model import Arrangement
from openbapo.service.config import ServiceError, Settings
from openbapo.service.rendering import preview
from openbapo.service.worker import Worker


def interaction(owner=42):
    response = SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock(), is_done=lambda: False)
    message = SimpleNamespace(id=123, channel=SimpleNamespace(id=456), delete=AsyncMock())
    return SimpleNamespace(
        user=SimpleNamespace(id=owner),
        locale=discord.Locale.chinese,
        response=response,
        followup=SimpleNamespace(send=AsyncMock(return_value=message)),
        edit_original_response=AsyncMock(),
        filesize_limit=10 * 1024**2,
        channel_id=456,
        application_id=100,
        token="temporary-token",
        expires_at=SimpleNamespace(timestamp=lambda: time.time() + 900),
    )


def test_commands_contexts_and_localizations(tmp_path):
    async def run():
        app = OpenBapo(Settings(tmp_path))
        translator = Translator()
        for cmd in app.tree.get_commands():
            payload = await cmd.get_translated_payload(app.tree, translator)
            assert set(payload["integration_types"]) == {0, 1}
            assert set(payload["contexts"]) == {0, 1, 2}
            assert payload["name_localizations"]["zh-CN"]
        assert {c.name for c in app.tree.get_commands()} == {
            "transcribe",
            "status",
            "delete",
            "help",
            "Transcribe",
        }
        await app.close()

    asyncio.run(run())


def test_persistent_controls_fit_discord_and_restore_state(tmp_path):
    async def run():
        app = OpenBapo(Settings(tmp_path))
        job = finished(app.store)
        app.store.update_arrangement(
            job.id, job.owner, Arrangement(transpose=3, strum_pattern="none")
        )
        job = app.store.get(job.id)
        view = SheetView(app, job)
        assert view.is_persistent() and len(view.children) == 6
        assert len({item.custom_id for item in view.children}) == 6
        for item in view.children:
            assert len(item.custom_id) < 100
            if isinstance(item, discord.ui.Select):
                assert len(item.options) <= 25
                assert all(len(o.label) <= 100 for o in item.options)
        assert next(o.value for o in view.children[0].options if o.default) == "3"
        assert next(o.value for o in view.children[2].options if o.default) == "none"
        app.add_view(view, message_id=123)
        app.stop_views({job.id})
        assert view.is_finished()
        await app.close()

    asyncio.run(run())


def test_non_owner_cannot_edit_or_export(tmp_path):
    async def run():
        app = OpenBapo(Settings(tmp_path))
        job = finished(app.store)
        intruder = interaction(99)
        for action in ("transpose", "export", "simplify", "reset"):
            with pytest.raises(ServiceError, match="owner"):
                await app.control(intruder, job.id, action, "2")
        intruder.response.defer.assert_not_awaited()
        intruder.edit_original_response.assert_not_awaited()
        assert app.store.get(job.id).arrangement == job.arrangement
        await app.close()

    asyncio.run(run())


def test_concurrent_adjustments_serialize_and_do_not_reanalyze(tmp_path):
    async def run():
        app = OpenBapo(Settings(tmp_path))
        job = finished(app.store)
        one, two = interaction(), interaction()
        await asyncio.gather(
            app.control(one, job.id, "transpose", "2"), app.control(two, job.id, "simplify", "")
        )
        current = app.store.get(job.id)
        options = Arrangement.from_json(current.arrangement)
        assert options.transpose == 2 and not options.simplify_chords
        assert current.analysis == job.analysis
        one.edit_original_response.assert_awaited_once()
        two.edit_original_response.assert_awaited_once()
        exported = interaction()
        await app.control(exported, job.id, "export", "musicxml")
        uploaded = exported.edit_original_response.call_args.kwargs["attachments"][0]
        assert uploaded.filename == "sheet.musicxml"
        assert b"score-partwise" in uploaded.fp.read()
        await app.close()

    asyncio.run(run())


def test_upload_queue_status_and_delete_without_discord(tmp_path, tone):
    async def run():
        app = OpenBapo(Settings(tmp_path / "app", device="cpu"))
        called = interaction()

        async def save(path):
            shutil.copyfile(tone, path)

        audio = SimpleNamespace(filename="../../song.wav", size=tone.stat().st_size, save=save)
        calls = MagicMock()
        calls.attach_mock(called.edit_original_response, "finish_private_response")
        calls.attach_mock(called.followup.send, "public_receipt")
        await app.submit(called, audio, lyrics="hello world")
        assert [c[0] for c in calls.mock_calls[:2]] == ["finish_private_response", "public_receipt"]
        assert called.followup.send.call_args.kwargs["ephemeral"] is False
        called.response.defer.assert_awaited_once_with(ephemeral=True, thinking=True)
        job = app.store.list_jobs(42)[0]
        assert job.status == "queued" and app.store.remaining(42) == 4
        app.worker = Worker(app.store, command=fake_command)
        await app.worker.run_once()
        assert app.store.get(job.id).status == "done"
        check = interaction()
        await app.tree.get_command("status").callback(check, job.id)
        assert (
            check.edit_original_response.call_args.kwargs["attachments"][0].filename == "sheet.png"
        )
        app.delete_message = AsyncMock(return_value=True)
        await app.tree.get_command("delete").callback(interaction())
        assert not app.store.list_jobs()
        assert app.store.remaining(42) == 4
        assert not list((app.settings.data_dir / "work").iterdir())
        await app.close()

    asyncio.run(run())


def test_invalid_upload_does_not_charge_or_leave_files(tmp_path):
    async def run():
        app = OpenBapo(Settings(tmp_path))

        async def save(path):
            path.write_bytes(b"invalid")

        audio = SimpleNamespace(filename="song.mp3", size=7, save=save)
        with pytest.raises(ServiceError, match="corrupt"):
            await app.submit(interaction(), audio)
        assert app.store.remaining(42) == 5
        assert not list((app.settings.data_dir / "work").iterdir())
        await app.close()

    asyncio.run(run())


def forbidden():
    return discord.Forbidden(SimpleNamespace(status=403, reason="Forbidden"), "no permission")


def test_expired_webhook_falls_back_to_dm(tmp_path):
    async def run():
        app = OpenBapo(Settings(tmp_path))
        job = replace(
            finished(app.store),
            channel_id=456,
            message_id=123,
            token="expired",
            token_expires=time.time() - 1,
        )
        partial = MagicMock()
        partial.get_partial_message.return_value.edit = AsyncMock(side_effect=forbidden())
        app.get_partial_messageable = MagicMock(return_value=partial)
        user = SimpleNamespace(
            send=AsyncMock(return_value=SimpleNamespace(id=999, channel=SimpleNamespace(id=888)))
        )
        app.fetch_user = AsyncMock(return_value=user)
        await app.deliver(job)
        user.send.assert_awaited_once()
        saved = app.store.get(job.id)
        assert saved.notified and saved.channel_id == 888 and saved.message_id == 999
        assert saved.token is None
        await app.close()

    asyncio.run(run())


def test_failed_delivery_remains_retrievable(tmp_path):
    async def run():
        app = OpenBapo(Settings(tmp_path))
        job = finished(app.store)
        app.fetch_user = AsyncMock(side_effect=forbidden())
        with pytest.raises(discord.Forbidden):
            await app.deliver(job)
        assert not app.store.get(job.id).notified
        assert app.store.get(job.id).analysis == job.analysis
        await app.close()

    asyncio.run(run())


def test_pure_music_feedback_and_bilingual_help(tmp_path):
    async def run():
        app = OpenBapo(Settings(tmp_path))
        job = finished(app.store)
        files = preview(job)
        assert "未识别到歌词" in sheet_embed(job, files).description
        english = replace(job, locale="en-US")
        assert "No lyrics" in sheet_embed(english, files).description
        for language in ("zh-CN", "zh-TW", "en-US"):
            assert len(t(language, "help", mb=10)) < 2000
        await app.close()

    asyncio.run(run())
