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
from test_service import submit as submit_job

from openbapo.bot.app import OpenBapo
from openbapo.bot.messages import (
    ago,
    error_text,
    help_embed,
    labelled,
    sheet_embed,
    strum_label,
    t,
)
from openbapo.bot.views import ExportPicker, SheetView, StatusPicker, StrumPicker
from openbapo.model import Arrangement
from openbapo.service.config import ServiceError, Settings
from openbapo.service.rendering import preview
from openbapo.service.worker import Worker


def interaction(owner=42):
    response = SimpleNamespace(
        defer=AsyncMock(), send_message=AsyncMock(), edit_message=AsyncMock(), is_done=lambda: False
    )
    message = SimpleNamespace(
        id=123,
        channel=SimpleNamespace(id=456),
        content="receipt",
        delete=AsyncMock(),
        edit=AsyncMock(),
    )
    return SimpleNamespace(
        user=SimpleNamespace(id=owner),
        # A DM or a user install by default: no song thread.
        is_guild_integration=lambda: False,
        channel=None,
        app_permissions=discord.Permissions.none(),
        locale=discord.Locale.chinese,
        response=response,
        followup=SimpleNamespace(send=AsyncMock(return_value=message)),
        edit_original_response=AsyncMock(return_value=message),
        delete_original_response=AsyncMock(),
        filesize_limit=10 * 1024**2,
        channel_id=456,
        application_id=100,
        token="temporary-token",
        expires_at=SimpleNamespace(timestamp=lambda: time.time() + 900),
    )


def test_commands_are_english_everywhere_with_all_contexts(tmp_path):
    async def run():
        app = OpenBapo(Settings(tmp_path))
        for cmd in app.tree.get_commands():
            payload = cmd.to_dict(app.tree)
            assert set(payload["integration_types"]) == {0, 1}
            assert set(payload["contexts"]) == {0, 1, 2}
            # Every client shows the same names and descriptions.
            assert not payload.get("name_localizations")
            assert not payload.get("description_localizations")
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
        assert [b.action for b in view.children if isinstance(b, discord.ui.Button)] == [
            "strum",
            "export",
            "simplify",
            "reset",
        ]
        # Singing keys read as words, not "+3"; a closed menu says which menu it is.
        labels = [o.label for o in view.children[0].options]
        assert labelled("zh-CN", "transpose", t("zh-CN", "key_original", key="C")) in labels
        assert labelled("zh-CN", "transpose", t("zh-CN", "key_up", key="Eb", n=3)) in labels
        prefix = labelled("zh-CN", "fingering", "")
        assert all(o.label.startswith(prefix) for o in view.children[1].options)
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
        with pytest.raises(ServiceError, match="owner"):
            await app.strum_picker(intruder, job.id)
        with pytest.raises(ServiceError, match="owner"):
            await app.pick_strum(intruder, interaction(), job.id, "none")
        intruder.response.defer.assert_not_awaited()
        intruder.response.edit_message.assert_not_awaited()
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
        # Each click is answered at once: the new choice shown, every control greyed
        # out and the clicked button reading "更新中…" until the redraw lands.
        busy = two.response.edit_message.call_args.kwargs["view"]
        assert all(item.disabled for item in busy.children)
        assert [b.label for b in busy.children if getattr(b, "action", "") == "simplify"] == [
            t("zh-CN", "updating")
        ]
        picked = one.response.edit_message.call_args.kwargs["view"].children[0]
        assert next(o.value for o in picked.options if o.default) == "2"
        assert not any(
            i.disabled for i in one.edit_original_response.call_args.kwargs["view"].children
        )
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
        await app.submit(called, audio)
        # One public message from "thinking…" to the receipt: nothing deleted under it,
        # so Discord keeps its "X used /transcribe" line.
        called.response.defer.assert_awaited_once_with(thinking=True)
        assert "song" in called.edit_original_response.call_args.kwargs["content"]
        called.delete_original_response.assert_not_awaited()
        called.followup.send.assert_not_awaited()
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
        app.delete_thread = AsyncMock(return_value=True)
        asked = interaction()
        await app.tree.get_command("delete").callback(asked)
        assert app.store.list_jobs()  # nothing goes before the owner confirms
        sent = asked.response.send_message.call_args
        assert sent.kwargs["ephemeral"] is True and "不能恢复" in sent.args[0]
        confirm = interaction()
        confirm.response.edit_message = AsyncMock()
        await sent.kwargs["view"].confirmed(confirm)
        assert not app.store.list_jobs()
        assert app.store.remaining(42) == 4
        assert not list((app.settings.data_dir / "work").iterdir())
        await app.close()

    asyncio.run(run())


def in_server_channel(called):
    called.is_guild_integration = lambda: True
    called.channel = MagicMock(spec=discord.TextChannel)
    called.app_permissions = discord.Permissions(
        create_public_threads=True, send_messages_in_threads=True
    )
    return called


def test_a_song_in_a_server_channel_gets_its_own_thread(tmp_path, tone, monkeypatch):
    async def run():
        app = OpenBapo(Settings(tmp_path / "app", device="cpu"))

        async def save(path):
            shutil.copyfile(tone, path)

        audio = SimpleNamespace(filename="song.wav", size=tone.stat().st_size, save=save)
        thread = MagicMock(spec=discord.Thread, id=777, owner_id=1, parent_id=456)
        thread.add_user = AsyncMock()
        thread.send = AsyncMock(return_value=SimpleNamespace(id=778))
        thread.delete = AsyncMock(side_effect=forbidden())
        thread.edit = AsyncMock()
        partial = MagicMock()
        partial.get_partial_message.return_value.delete = AsyncMock()
        app.get_partial_messageable = MagicMock(return_value=partial)

        called = in_server_channel(interaction())
        starter = called.channel.get_partial_message.return_value
        starter.create_thread = AsyncMock(return_value=thread)
        await app.submit(called, audio)
        # The thread hangs from the receipt, reached through the interaction's own
        # channel: a bare partial channel has no server and cannot start one.
        called.channel.get_partial_message.assert_called_once_with(123)
        made = starter.create_thread.call_args.kwargs
        assert made == {"name": "song", "auto_archive_duration": 10080}
        thread.add_user.assert_awaited_once_with(called.user)
        job = app.store.list_jobs(42)[0]
        # Progress and the sheet live in the thread, edited with the bot's own token.
        assert (job.channel_id, job.message_id, job.token) == (777, 778, None)
        receipt = called.edit_original_response.return_value
        assert receipt.edit.call_args.kwargs["content"] == t(
            "zh-CN", "receipt_thread", title="song"
        )

        # /delete removes the title too: the thread, or without Manage Threads
        # the line it hangs from and its name.
        monkeypatch.setattr(OpenBapo, "user_id", property(lambda self: 1))
        app.fetch_channel = AsyncMock(return_value=thread)
        assert await app.delete_thread(job)
        partial.get_partial_message.return_value.delete.assert_awaited()
        thread.edit.assert_awaited_once_with(name="—", archived=True)

        # A thread that cannot be made leaves the upload accepted, on its receipt.
        broken = in_server_channel(interaction())
        broken.channel.get_partial_message.return_value.create_thread = AsyncMock(
            side_effect=ValueError("This message does not have guild info attached.")
        )
        await app.submit(broken, audio)
        newest = max(app.store.list_jobs(42), key=lambda j: j.created)
        assert (newest.channel_id, newest.message_id) == (456, 123) and newest.token

        # Without the thread permissions the receipt itself stays the progress message.
        plain = in_server_channel(interaction())
        plain.app_permissions = discord.Permissions(send_messages_in_threads=True)
        await app.submit(plain, audio)
        newest = max(app.store.list_jobs(42), key=lambda j: j.created)
        assert (newest.channel_id, newest.message_id) == (456, 123) and newest.token
        await app.close()

    asyncio.run(run())


def test_invalid_upload_does_not_charge_or_leave_files(tmp_path):
    async def run():
        app = OpenBapo(Settings(tmp_path))

        async def save(path):
            path.write_bytes(b"invalid")

        audio = SimpleNamespace(filename="song.mp3", size=7, save=save)
        rejected = interaction()
        rejected.response.is_done = lambda: rejected.response.defer.await_count > 0
        await app.submit(rejected, audio)
        # The public "thinking…" goes; only the uploader learns why.
        rejected.delete_original_response.assert_awaited_once()
        told = rejected.followup.send.call_args
        assert told.kwargs["ephemeral"] is True
        assert told.args[0] == error_text("zh-CN", "corrupt", app.settings)
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
            card = help_embed(language, Settings(tmp_path))
            assert (
                len(card) < 6000 and len(card.fields) == 6
            )  # three steps, commands, limits, privacy
            assert not any(f.inline for f in card.fields)
            assert "lyrics" not in card.fields[0].value  # lyrics can no longer be attached
            assert "<t:" in card.fields[4].value and card.footer.text is None  # no contact set
        await app.close()

    asyncio.run(run())


def test_strum_is_picked_from_a_private_numbered_picture(tmp_path):
    async def run():
        app = OpenBapo(Settings(tmp_path))
        job = finished(app.store)
        sheet = interaction()
        await app.strum_picker(sheet, job.id)
        sheet.response.defer.assert_awaited_once_with()
        sent = sheet.followup.send.call_args
        assert sent.kwargs["ephemeral"] is True
        assert sent.kwargs["file"].filename == "strums.png"
        picker = sent.kwargs["view"]
        assert isinstance(picker, StrumPicker) and picker.sheet is sheet
        options = preview(job).sheet.strum_options
        values = [o.value for o in picker.menu.options]
        assert values == ["auto", "none", *(o.id for o in options)] and len(values) <= 25
        # Patterns are numbered, not named: "1 · 扫弦", with the recommended one marked.
        assert [o.label for o in picker.menu.options[2:]] == [
            strum_label("zh-CN", options, o.id) for o in options
        ]
        assert picker.menu.options[2].label.startswith("1 · ")
        assert sum("推荐" in o.label for o in picker.menu.options[2:]) == 1

        chosen = options[-1]
        picked = interaction()
        picker.menu._values = [chosen.id]  # what Discord reports for the selection
        await picker.picked(picked)
        assert Arrangement.from_json(app.store.get(job.id).arrangement).strum_pattern == chosen.id
        redrawn = sheet.edit_original_response.call_args.kwargs
        assert isinstance(redrawn["view"], SheetView)
        assert redrawn["attachments"][0].filename == "sheet.png"
        assert picked.response.edit_message.call_args.kwargs["content"] == t("zh-CN", "updating")
        note = picked.edit_original_response.call_args.kwargs
        assert f"{len(options)} · " in note["content"] and note["view"] is None
        assert picker.is_finished()
        await app.close()

    asyncio.run(run())


def test_discord_proxy_reaches_the_client_and_its_webhooks(tmp_path, monkeypatch):
    async def run():
        monkeypatch.setenv("DISCORD_PROXY", "http://127.0.0.1:9")
        app = OpenBapo(Settings(tmp_path))
        assert app.http.proxy == "http://127.0.0.1:9"
        # The session exists once logged in; a stand-in is enough to build the webhook.
        app.http._HTTPClient__session = SimpleNamespace(close=AsyncMock())  # type: ignore[attr-defined]
        assert app.interaction_webhook(100, "token").proxy == "http://127.0.0.1:9"
        await app.close()
        monkeypatch.delenv("DISCORD_PROXY")
        plain = OpenBapo(Settings(tmp_path))
        plain.http._HTTPClient__session = SimpleNamespace(close=AsyncMock())  # type: ignore[attr-defined]
        assert plain.http.proxy is None and plain.interaction_webhook(100, "t").proxy is None
        await plain.close()

    asyncio.run(run())


def test_delete_can_be_cancelled_and_says_so_when_there_is_nothing(tmp_path):
    async def run():
        app = OpenBapo(Settings(tmp_path))
        empty = interaction()
        await app.tree.get_command("delete").callback(empty)
        assert empty.response.send_message.call_args.args[0] == t("zh-CN", "empty")
        job = finished(app.store)
        asked = interaction()
        await app.tree.get_command("delete").callback(asked)
        view = asked.response.send_message.call_args.kwargs["view"]
        assert not await view.interaction_check(interaction(99))  # only the asker
        cancel = interaction()
        cancel.response.edit_message = AsyncMock()
        await view.cancelled(cancel)
        assert cancel.response.edit_message.call_args.kwargs["view"] is None
        assert app.store.get(job.id).status == "done"  # kept
        await app.close()

    asyncio.run(run())


def test_status_lists_titles_and_opens_a_sheet_from_a_menu(tmp_path):
    async def run():
        app = OpenBapo(Settings(tmp_path))
        job = finished(app.store)
        asked = interaction()
        await app.tree.get_command("status").callback(asked)
        shown = asked.edit_original_response.call_args.kwargs
        assert job.title and job.title in shown["content"] and job.id not in shown["content"]
        picker = shown["view"]
        assert isinstance(picker, StatusPicker)
        assert [o.value for o in picker.menu.options] == [job.id]
        opened = interaction()
        picker.menu._values = [job.id]
        await picker.picked(opened)
        assert opened.edit_original_response.call_args.kwargs["attachments"][0].filename == (
            "sheet.png"
        )
        await app.close()

    asyncio.run(run())


def test_export_is_a_button_that_offers_each_format_privately(tmp_path):
    async def run():
        app = OpenBapo(Settings(tmp_path))
        job = finished(app.store)
        asked = interaction()
        await app.export_picker(asked, job.id)
        picker = asked.response.send_message.call_args.kwargs["view"]
        assert isinstance(picker, ExportPicker) and len(picker.children) == 3
        for button, name in zip(
            picker.children, ("sheet.pdf", "sheet", "sheet.musicxml"), strict=True
        ):
            clicked = interaction()
            await button.callback(clicked)
            sent = clicked.edit_original_response.call_args.kwargs["attachments"][0]
            assert sent.filename.startswith(name)
        await app.close()

    asyncio.run(run())


def test_receipt_follows_the_queue_and_the_analysis_step(tmp_path):
    from openbapo.service.job import write_stage
    from openbapo.service.media import work_path

    async def run():
        app = OpenBapo(Settings(tmp_path))
        app.edit_receipt = AsyncMock()
        job_id = "a" * 32
        app.store.submit(job_id, 42, "zh-CN", "song", channel_id=1, message_id=2)
        await app.update_receipts()
        assert "排队中" in app.edit_receipt.call_args.args[1]
        await app.update_receipts()  # unchanged: no second edit
        assert app.edit_receipt.await_count == 1
        with app.store.connect() as db:
            db.execute("UPDATE jobs SET status='running' WHERE id=?", (job_id,))
        work = work_path(app.settings, job_id)
        work.mkdir(parents=True)
        write_stage(work, "chords")
        app.receipts[job_id] = (app.receipts[job_id][0], 0.0)  # past the edit interval
        await app.update_receipts()
        step = t("zh-CN", "step_chords")
        assert (
            step in app.edit_receipt.call_args.args[1]
            and "4/5" in app.edit_receipt.call_args.args[1]
        )
        await app.close()

    asyncio.run(run())


def test_errors_name_the_contact_only_when_one_is_set(tmp_path):
    plain = Settings(tmp_path)
    assert "@" not in error_text("zh-CN", "service", plain)
    with_contact = Settings(tmp_path, contact="help@example.org")
    assert "help@example.org" in error_text("zh-CN", "service", with_contact)
    assert "help@example.org" in (help_embed("en-US", with_contact).footer.text or "")
    assert "<t:" in error_text("zh-CN", "quota", plain)  # reset time in the reader's zone


def test_status_lists_short_reasons_and_tells_same_titles_apart(tmp_path):
    async def run():
        app = OpenBapo(Settings(tmp_path, contact="help@example.org"))
        failed = submit_job(app.store)
        app.store.claim()
        app.store.fail(failed.id, "service")
        first, second = finished(app.store), finished(app.store)
        asked = interaction()
        await app.tree.get_command("status").callback(asked)
        sent = asked.edit_original_response.call_args.kwargs
        text = sent["content"]
        # The list says why in a word, nothing more.
        assert f"{t('zh-CN', 'failed')} · {t('zh-CN', 'why_service')}" in text
        assert error_text("zh-CN", "service", app.settings) not in text
        assert "help@example.org" not in text
        # Same titles in the menu differ by age, not by a UTC clock.
        options = sent["view"].menu.options
        assert {o.value for o in options} == {first.id, second.id}
        assert all("UTC" not in (o.description or "") for o in options)
        await app.close()

    asyncio.run(run())


def test_ago_reads_like_speech():
    now = 1_000_000.0
    assert ago("zh-CN", now - 5, now) == "刚刚"
    assert ago("zh-CN", now - 35 * 60, now) == t("zh-CN", "minutes_ago", n=35)
    assert ago("en-US", now - 3 * 3600, now) == "3 h ago"
    assert ago("en-US", now - 2 * 86400, now) == "2 days ago"
