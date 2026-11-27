"""Gateway application. Invoke explicitly with openbapo-bot; imports never connect."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import time
import uuid
from collections import defaultdict
from dataclasses import replace
from functools import partial
from io import BytesIO

import discord
from discord import app_commands

from openbapo.analysis.audio import MAX_DURATION_SECONDS
from openbapo.bot.messages import (
    Interaction,
    error_text,
    failure_reason,
    help_embed,
    local_time,
    next_reset,
    sheet_embed,
    strum_kind,
    strum_label,
    t,
)
from openbapo.bot.views import DeleteConfirm, ExportPicker, SheetView, StatusPicker, StrumPicker
from openbapo.cli import AUDIO_EXTENSIONS
from openbapo.model import Arrangement
from openbapo.naming import title_from_filename
from openbapo.service.config import ServiceError, Settings
from openbapo.service.job import read_stage
from openbapo.service.media import check_attachment, inspect_audio, work_path
from openbapo.service.rendering import (
    SheetFiles,
    changed,
    export,
    preview,
    sheet_for,
    strum_overview,
)
from openbapo.service.store import Job, Store
from openbapo.service.worker import InstanceLock, Worker

logger = logging.getLogger(__name__)

# Analysis steps shown on a receipt, in order (the stages analyze_file reports).
STEPS = ("decode", "separate", "beats", "chords", "lyrics")
RECEIPT_INTERVAL = 3.0  # seconds between edits of one receipt (Discord rate limits)
MINUTES_PER_SONG = 2  # rough expected time per queued song, for the receipt


def expected(ahead: int) -> str:
    """When the sheet should be ready, as a Discord timestamp ("in 4 minutes")."""
    return f"<t:{int(time.time()) + (ahead + 1) * MINUTES_PER_SONG * 60}:R>"


def attachments(files: SheetFiles, limit: int) -> list[discord.File]:
    if len(files.png) > limit or (files.pdf is not None and len(files.pdf) > limit):
        raise ServiceError("export_size")
    result = [discord.File(BytesIO(files.png), filename="sheet.png")]
    if files.pdf is not None:
        result.append(discord.File(BytesIO(files.pdf), filename="sheet.pdf"))
    return result


class OpenBapo(discord.Client):
    def __init__(self, settings: Settings) -> None:
        super().__init__(
            intents=discord.Intents.none(),
            allowed_mentions=discord.AllowedMentions.none(),
            # Optional HTTP proxy for every Discord connection, for hosts that
            # cannot reach Discord directly; unset, nothing changes.
            proxy=os.environ.get("DISCORD_PROXY") or None,
        )
        self.settings = settings
        self.store = Store(settings)
        self.worker = Worker(self.store)
        self.instance: InstanceLock | None = None
        self.owner_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.render_gate = asyncio.Semaphore(1)
        self.tasks: list[asyncio.Task[None]] = []
        self.receipts: dict[str, tuple[str, float]] = {}  # job -> (what it shows, when)
        self.tree = app_commands.CommandTree(
            self,
            allowed_installs=app_commands.AppInstallationType(guild=True, user=True),
            allowed_contexts=app_commands.AppCommandContext(
                guild=True, dm_channel=True, private_channel=True
            ),
        )
        self.register_commands()

    def register_commands(self) -> None:
        @self.tree.command(
            name="transcribe",
            description="Turn audio into a guitar sheet",
        )
        @app_commands.describe(
            audio=f"Audio file, up to {MAX_DURATION_SECONDS // 60} minutes",
            title="Song title",
            artist="Artist",
        )
        async def transcribe(
            interaction: Interaction,
            audio: discord.Attachment,
            title: app_commands.Range[str, 1, 200] | None = None,
            artist: app_commands.Range[str, 1, 200] | None = None,
        ) -> None:
            await self.submit(interaction, audio, title, artist)

        @self.tree.command(
            name="status",
            description="View your jobs or retrieve a result",
        )
        @app_commands.describe(job="Optional job ID from your receipt")
        async def status(interaction: Interaction, job: str | None = None) -> None:
            if job:
                await self.show_job(interaction, job)
                return
            await interaction.response.defer(ephemeral=True, thinking=True)
            locale = str(interaction.locale)
            jobs = self.store.list_jobs(interaction.user.id)[-10:]
            lines = [t(locale, "status_head")] if jobs else [t(locale, "empty")]
            for saved in jobs:
                label = t(locale, saved.status)
                if saved.status == "queued":
                    label += " · " + t(locale, "ahead", n=self.store.position(saved))
                if saved.status == "failed":
                    label += " · " + failure_reason(locale, saved.error)
                title = discord.utils.escape_markdown(saved.title or t(locale, "untitled"))
                lines.append(f"• **{title}** · {label} · <t:{int(saved.created)}:R>")
            left = self.store.remaining(interaction.user.id)
            lines.append(
                t(locale, "unlimited")
                if left is None
                else t(
                    locale,
                    "remaining",
                    n=left,
                    q=self.settings.daily_quota,
                    reset=local_time(next_reset()),
                )
            )
            done = [j for j in jobs if j.status == "done"]
            await interaction.edit_original_response(
                content="\n".join(lines)[:2000],
                view=StatusPicker(self, done, locale) if done else discord.utils.MISSING,
            )

        @self.tree.command(
            name="delete",
            description="Delete all your server data",
        )
        async def delete(interaction: Interaction) -> None:
            # Nothing is removed until the owner confirms (irreversible).
            locale = str(interaction.locale)
            count = len(self.store.list_jobs(interaction.user.id))
            if not count:
                await interaction.response.send_message(t(locale, "empty"), ephemeral=True)
                return
            await interaction.response.send_message(
                t(locale, "delete_confirm", n=count),
                view=DeleteConfirm(self, interaction),
                ephemeral=True,
            )

        @self.tree.command(
            name="help",
            description="Usage, limits and privacy",
        )
        async def help_command(interaction: Interaction) -> None:
            await interaction.response.send_message(
                embed=help_embed(str(interaction.locale), self.settings), ephemeral=True
            )

        async def transcribe_message(interaction: Interaction, message: discord.Message) -> None:
            from pathlib import Path

            audio = [
                a
                for a in message.attachments
                if Path(a.filename).suffix.lower() in AUDIO_EXTENSIONS
            ]
            if len(audio) != 1:
                raise ServiceError("many_audio" if audio else "no_audio")
            await self.submit(interaction, audio[0])

        self.tree.add_command(
            app_commands.ContextMenu(
                name="Transcribe",
                callback=transcribe_message,
            )
        )

        @self.tree.error
        async def on_error(interaction: Interaction, error: app_commands.AppCommandError) -> None:
            await self.report_error(interaction, getattr(error, "original", error))

    async def setup_hook(self) -> None:
        self.instance = InstanceLock(self.settings.data_dir)
        self.worker.lock_fd = self.instance.file.fileno()
        self.worker.recover()
        self.store.sweep()
        for job in self.store.list_jobs():
            if job.status == "done":
                self.add_view(SheetView(self, job), message_id=job.message_id)
        await self.tree.sync()
        self.tasks = [
            asyncio.create_task(self.worker.run()),
            asyncio.create_task(self.deliver_loop()),
        ]

    async def close(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await super().close()
        if self.instance is not None:
            self.instance.close()

    async def report_error(
        self,
        interaction: Interaction,
        error: Exception,
        job_id: str | None = None,
        *,
        private: bool = False,
    ) -> None:
        """Tell the user what went wrong; `private` after the public response was removed."""
        code = error.code if isinstance(error, ServiceError) else "service"
        logger.warning(
            "interaction_error code=%s type=%s job=%s", code, type(error).__name__, job_id
        )
        content = error_text(str(interaction.locale), code, self.settings)
        if job_id:
            content += f"\n`{job_id}`"
        if not interaction.response.is_done():
            await interaction.response.send_message(content, ephemeral=True)
        elif private:
            await interaction.followup.send(content, ephemeral=True)
        elif interaction.response.type == discord.InteractionResponseType.deferred_channel_message:
            await interaction.edit_original_response(content=content)
        else:
            await interaction.followup.send(content, ephemeral=True)

    async def submit(
        self,
        interaction: Interaction,
        audio: discord.Attachment,
        title: str | None = None,
        artist: str | None = None,
    ) -> None:
        # Public from the start: "thinking…" becomes the receipt, then the sheet, so the
        # message keeps Discord's "X used /transcribe" line.
        await interaction.response.defer(thinking=True)
        try:
            await self.accept(interaction, audio, title, artist)
        except Exception as exc:
            # Rejected uploads leave nothing public: the reason goes to the uploader only.
            with contextlib.suppress(discord.HTTPException):
                await interaction.delete_original_response()
            await self.report_error(interaction, exc, private=True)

    async def accept(
        self,
        interaction: Interaction,
        audio: discord.Attachment,
        title: str | None,
        artist: str | None,
    ) -> None:
        """Validate and queue an upload, turning the deferred response into its receipt.
        Raises only before the job is queued."""
        owner, locale = interaction.user.id, str(interaction.locale)
        async with self.owner_locks[owner]:
            suffix = check_attachment(audio.filename, audio.size, self.settings)
            if self.store.remaining(owner) == 0:
                raise ServiceError("quota")
            job_id = uuid.uuid4().hex
            work = work_path(self.settings, job_id)
            work.mkdir(mode=0o700)
            queued = False
            try:
                path = work / ("input" + suffix)
                await asyncio.wait_for(audio.save(path), 60)
                check_attachment(path.name, path.stat().st_size, self.settings)
                await inspect_audio(path)
                title = (title or title_from_filename(audio.filename))[:200]
                request = {"audio": path.name, "title": title, "artist": artist}
                (work / "request.json").write_text(
                    json.dumps(request, ensure_ascii=False), encoding="utf-8"
                )
                ahead = sum(j.status in {"queued", "running"} for j in self.store.list_jobs())
                receipt = await interaction.edit_original_response(
                    content=t(
                        locale,
                        "receipt",
                        title=discord.utils.escape_markdown(title),
                        ahead=ahead,
                        eta=expected(ahead),
                    ),
                    allowed_mentions=discord.AllowedMentions.none(),
                )
                self.store.submit(
                    job_id,
                    owner,
                    locale,
                    title,
                    channel_id=interaction.channel_id,
                    message_id=receipt.id,
                    application_id=interaction.application_id,
                    token=interaction.token,
                    token_expires=interaction.expires_at.timestamp(),
                )
                queued = True
            except TimeoutError as exc:
                raise ServiceError("timeout") from exc
            finally:
                if not queued:
                    shutil.rmtree(work, ignore_errors=True)
        await self.open_thread(interaction, receipt, job_id, title, locale)

    async def open_thread(
        self,
        interaction: Interaction,
        receipt: discord.InteractionMessage,
        job_id: str,
        title: str,
        locale: str,
    ) -> None:
        """In a server text channel, give the song its own thread under the receipt:
        progress, the sheet and talk about it go there, and the channel keeps one
        line per song. Anywhere else, or without the thread
        permissions, the receipt itself stays the progress message."""
        perms = interaction.app_permissions
        channel = interaction.channel
        if not (
            interaction.is_guild_integration()
            and isinstance(channel, discord.TextChannel)
            and perms.create_public_threads
            and perms.send_messages_in_threads
        ):
            return
        try:
            # The interaction's channel carries the server a thread needs; a bare
            # partial channel does not (the job is queued either way).
            thread = await channel.get_partial_message(receipt.id).create_thread(
                name=title[:100], auto_archive_duration=10080
            )
            with contextlib.suppress(discord.HTTPException):
                await thread.add_user(interaction.user)
            job = self.store.get(job_id)
            progress = await thread.send(
                receipt.content, allowed_mentions=discord.AllowedMentions.none()
            )
            self.store.move(job.id, thread.id, progress.id)
            await receipt.edit(
                content=t(locale, "receipt_thread", title=discord.utils.escape_markdown(title))
            )
        except Exception as exc:
            # Never fail an accepted upload over its thread: the receipt carries on.
            logger.warning("thread_failed job=%s type=%s", job_id, type(exc).__name__)

    async def render(self, job: Job, options: Arrangement | None = None) -> SheetFiles:
        async with self.render_gate:
            return await asyncio.to_thread(preview, job, options)

    async def control(self, interaction: Interaction, job_id: str, action: str, value: str) -> None:
        # Authorize before deferring an update, so unauthorized clicks cannot touch the message.
        self.store.get(job_id, interaction.user.id)
        if action == "export":
            await interaction.response.defer(ephemeral=True, thinking=True)
        else:
            # Answer at once with the new choice and greyed-out controls: the
            # redraw takes a few seconds.
            job = self.store.get(job_id, interaction.user.id)
            pending = replace(job, arrangement=changed(job, action, value).to_json())
            await interaction.response.edit_message(view=SheetView(self, pending, busy=action))
        async with self.owner_locks[interaction.user.id]:
            job = self.store.get(job_id, interaction.user.id)
            if action == "export":
                async with self.render_gate:
                    name, data = await asyncio.to_thread(export, job, value)
                if len(data) > interaction.filesize_limit:
                    raise ServiceError("export_size")
                await interaction.edit_original_response(
                    attachments=[discord.File(BytesIO(data), filename=name)],
                )
                return
            try:
                await self.apply(interaction, job, action, value)
            except Exception:
                # Give the controls back; report_error then explains what failed.
                with contextlib.suppress(discord.HTTPException):
                    await interaction.edit_original_response(view=SheetView(self, job))
                raise

    async def apply(self, sheet: Interaction, job: Job, action: str, value: str) -> Job:
        """Change one option and redraw the sheet message `sheet` was sent from."""
        options = changed(job, action, value)
        files = await self.render(job, options)
        # Persist before sending so even a network failure can be recovered via /status.
        self.store.update_arrangement(job.id, job.owner, options)
        updated = self.store.get(job.id, job.owner)
        await sheet.edit_original_response(
            embed=sheet_embed(updated, files),
            attachments=attachments(files, sheet.filesize_limit),
            view=SheetView(self, updated),
        )
        return updated

    async def show_job(self, interaction: Interaction, job_id: str) -> None:
        """Send the owner, privately, one of their sheets with its controls."""
        await interaction.response.defer(ephemeral=True, thinking=True)
        async with self.owner_locks[interaction.user.id]:
            saved = self.store.get(job_id, interaction.user.id)
            if saved.status == "failed":
                raise ServiceError(saved.error or "service")
            files = await self.render(saved)
            await interaction.edit_original_response(
                content=None,
                embed=sheet_embed(saved, files),
                attachments=attachments(files, interaction.filesize_limit),
                view=SheetView(self, saved),
            )

    async def export_picker(self, interaction: Interaction, job_id: str) -> None:
        self.store.get(job_id, interaction.user.id)
        job = self.store.get(job_id, interaction.user.id)
        await interaction.response.send_message(
            t(job.locale, "export_pick"), view=ExportPicker(self, job), ephemeral=True
        )

    async def update_receipts(self) -> None:
        """Keep each waiting job's receipt current: queue place, then analysis step."""
        now = time.time()
        for job in self.store.list_jobs():
            if job.status not in {"queued", "running"} or not job.message_id:
                continue
            if job.status == "queued":
                ahead = self.store.position(job)
                text = t(
                    job.locale,
                    "receipt",
                    title=discord.utils.escape_markdown(job.title or ""),
                    ahead=ahead,
                    eta=expected(ahead),
                )
                key = f"queued:{ahead}"
            else:
                step = read_stage(work_path(self.settings, job.id)) or "decode"
                if step not in STEPS:
                    continue
                text = t(
                    job.locale,
                    "receipt_running",
                    title=discord.utils.escape_markdown(job.title or ""),
                    step=t(job.locale, f"step_{step}"),
                    k=STEPS.index(step) + 1,
                    n=len(STEPS),
                )
                key = f"running:{step}"
            shown, at = self.receipts.get(job.id, ("", 0.0))
            if key == shown or now - at < RECEIPT_INTERVAL:
                continue
            self.receipts[job.id] = (key, now)
            with contextlib.suppress(discord.HTTPException):
                await self.edit_receipt(job, text)

    async def edit_receipt(self, job: Job, text: str) -> None:
        assert job.message_id is not None
        if job.token and job.token_expires > time.time():
            webhook = self.interaction_webhook(job.application_id, job.token)
            await webhook.edit_message(job.message_id, content=text)
        elif job.channel_id:
            channel = self.get_partial_messageable(job.channel_id)
            await channel.get_partial_message(job.message_id).edit(content=text)

    async def strum_picker(self, interaction: Interaction, job_id: str) -> None:
        """Show the owner, privately, the numbered strum picture with a matching menu."""
        self.store.get(job_id, interaction.user.id)
        # A deferred update keeps this interaction pointing at the sheet message.
        await interaction.response.defer()
        job = self.store.get(job_id, interaction.user.id)
        async with self.render_gate:
            png, options = await asyncio.to_thread(
                strum_overview, job, partial(strum_kind, job.locale)
            )
        await interaction.followup.send(
            t(job.locale, "strum_pick"),
            file=discord.File(BytesIO(png), filename="strums.png"),
            view=StrumPicker(self, job, options, interaction),
            ephemeral=True,
        )

    async def pick_strum(
        self, interaction: Interaction, sheet: Interaction, job_id: str, value: str
    ) -> None:
        job = self.store.get(job_id, interaction.user.id)
        await interaction.response.edit_message(
            content=t(job.locale, "updating"), attachments=[], view=None
        )
        async with self.owner_locks[interaction.user.id]:
            job = self.store.get(job_id, interaction.user.id)
            updated = await self.apply(sheet, job, "strum", value)
        sheet_now = sheet_for(updated)
        chosen = sheet_now.strum
        name = (
            t(updated.locale, "none")
            if chosen is None
            else strum_label(updated.locale, sheet_now.strum_options, chosen.id)
        )
        await interaction.edit_original_response(
            content=t(updated.locale, "strum_chosen", name=name), attachments=[], view=None
        )

    async def deliver_loop(self) -> None:
        await self.wait_until_ready()
        last_sweep = 0.0
        while True:
            self.worker.expire_waiting()
            if self.is_ready():
                (self.settings.data_dir / "heartbeat").touch()
            await self.update_receipts()
            if time.time() - last_sweep > 60:
                self.stop_views(set(self.store.sweep()))
                last_sweep = time.time()
            for job in self.store.list_jobs():
                if (
                    job.status not in {"done", "failed"}
                    or job.notified
                    or job.retry_after > time.time()
                ):
                    continue
                async with self.owner_locks[job.owner]:
                    try:
                        current = self.store.get(job.id)
                        if current.status in {"done", "failed"}:
                            await self.deliver(current)
                    except ServiceError:
                        self.store.delivery(job.id, retry=True)
                    except discord.HTTPException as exc:
                        logger.warning("delivery_failed job=%s status=%s", job.id, exc.status)
                        self.store.delivery(job.id, retry=True)
                    except Exception as exc:
                        logger.error("delivery_failed job=%s type=%s", job.id, type(exc).__name__)
                        self.store.delivery(job.id, retry=True)
            await asyncio.sleep(1)

    def stop_views(self, job_ids: set[str]) -> None:
        for view in self.persistent_views:
            if isinstance(view, SheetView) and view.job_id in job_ids:
                view.stop()

    async def deliver(self, job: Job) -> None:
        self.receipts.pop(job.id, None)
        files = await self.render(job) if job.status == "done" else None
        view = SheetView(self, job) if files else None
        embed = sheet_embed(job, files) if files else None
        content = (
            f"<@{job.owner}>"
            if files
            else f"{error_text(job.locale, job.error or 'service', self.settings)}\n`{job.id}`"
        )
        allowed = discord.AllowedMentions(users=[discord.Object(job.owner)])
        message: discord.Message
        if job.token and job.token_expires > time.time() and job.message_id:
            webhook = self.interaction_webhook(job.application_id, job.token)
            try:
                message = await webhook.edit_message(
                    job.message_id,
                    content=content,
                    embed=embed,
                    view=view,
                    attachments=attachments(files, self.settings.max_upload_bytes) if files else [],
                    allowed_mentions=allowed,
                )
                if view:
                    self.add_view(view, message_id=message.id)
                self.store.delivery(job.id, channel_id=message.channel.id, message_id=message.id)
                return
            except discord.HTTPException:
                pass
        if job.channel_id:
            try:
                channel = self.get_partial_messageable(job.channel_id)
                if job.message_id:
                    message = await channel.get_partial_message(job.message_id).edit(
                        content=content,
                        embed=embed if embed is not None else discord.utils.MISSING,
                        view=view if view is not None else discord.utils.MISSING,
                        attachments=attachments(files, self.settings.max_upload_bytes)
                        if files
                        else [],
                        allowed_mentions=allowed,
                    )
                else:
                    message = await channel.send(
                        content,
                        embed=embed if embed is not None else discord.utils.MISSING,
                        view=view if view is not None else discord.utils.MISSING,
                        files=attachments(files, self.settings.max_upload_bytes) if files else [],
                        allowed_mentions=allowed,
                    )
                self.store.delivery(job.id, channel_id=message.channel.id, message_id=message.id)
                return
            except discord.HTTPException:
                pass
        user = await self.fetch_user(job.owner)
        message = await user.send(
            content,
            embed=embed if embed is not None else discord.utils.MISSING,
            view=view if view is not None else discord.utils.MISSING,
            files=attachments(files, self.settings.max_upload_bytes) if files else [],
        )
        self.store.delivery(job.id, channel_id=message.channel.id, message_id=message.id)

    async def erase_owner(self, owner: int, locale: str) -> str:
        """Stop and delete everything the user has on the server; the reply text."""
        async with self.owner_locks[owner]:
            jobs = self.store.cancel_owner(owner)
            await self.worker.erase(jobs)
            self.stop_views({j.id for j in jobs})
            removed = True
            for job in jobs:
                removed = await self.delete_message(job) and removed
                removed = await self.delete_thread(job) and removed
        text = t(locale, "deleted")
        if not removed:
            text += "\n" + t(locale, "delete_messages")
        return text

    def interaction_webhook(self, application_id: int, token: str) -> discord.Webhook:
        """The interaction's webhook, through the client's proxy (discord.py shares
        the client's session with it but not the proxy)."""
        webhook = discord.Webhook.partial(application_id, token, client=self)
        webhook.proxy = self.http.proxy
        return webhook

    async def delete_thread(self, job: Job) -> bool:
        """A song thread the bot opened carries the title in its name and in the
        line it hangs from: delete the thread, or failing that (it needs Manage
        Threads) that line and the name."""
        if not job.channel_id:
            return True
        try:
            thread = await self.fetch_channel(job.channel_id)
        except discord.NotFound:
            return True
        except discord.HTTPException:
            return False
        if not isinstance(thread, discord.Thread) or thread.owner_id != self.user_id:
            return True
        try:
            await thread.delete()
            return True
        except discord.NotFound:
            return True
        except discord.HTTPException:
            pass
        try:
            with contextlib.suppress(discord.NotFound):
                await (
                    self.get_partial_messageable(thread.parent_id)
                    .get_partial_message(thread.id)  # a thread's id is its starting message's
                    .delete()
                )
            await thread.edit(name="—", archived=True)
            return True
        except discord.HTTPException:
            return False

    @property
    def user_id(self) -> int | None:
        return self.user.id if self.user else None

    async def delete_message(self, job: Job) -> bool:
        if job.message_id is None:
            return True
        if job.token and job.token_expires > time.time():
            try:
                await self.interaction_webhook(job.application_id, job.token).delete_message(
                    job.message_id
                )
                return True
            except discord.NotFound:
                return True
            except discord.HTTPException:
                pass
        if job.channel_id:
            try:
                await (
                    self.get_partial_messageable(job.channel_id)
                    .get_partial_message(job.message_id)
                    .delete()
                )
                return True
            except discord.NotFound:
                return True
            except discord.HTTPException:
                pass
        return False


def main() -> None:
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise SystemExit("Set DISCORD_TOKEN in the service environment (never commit it).")
    os.umask(0o077)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    OpenBapo(Settings.from_env()).run(token, log_handler=None)
