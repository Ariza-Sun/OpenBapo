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
from io import BytesIO

import discord
from discord import app_commands

from openbapo.bot.messages import Interaction, Translator, sheet_embed, t
from openbapo.bot.views import SheetView
from openbapo.cli import AUDIO_EXTENSIONS
from openbapo.model import Arrangement
from openbapo.service.config import ServiceError, Settings
from openbapo.service.media import check_attachment, inspect_audio, work_path
from openbapo.service.rendering import SheetFiles, changed, export, preview
from openbapo.service.store import Job, Store
from openbapo.service.worker import InstanceLock, Worker

logger = logging.getLogger(__name__)


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
            intents=discord.Intents.none(), allowed_mentions=discord.AllowedMentions.none()
        )
        self.settings = settings
        self.store = Store(settings)
        self.worker = Worker(self.store)
        self.instance: InstanceLock | None = None
        self.owner_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.render_gate = asyncio.Semaphore(1)
        self.tasks: list[asyncio.Task[None]] = []
        self.tree = app_commands.CommandTree(
            self,
            allowed_installs=app_commands.AppInstallationType(guild=True, user=True),
            allowed_contexts=app_commands.AppCommandContext(
                guild=True, dm_channel=True, private_channel=True
            ),
        )
        self.register_commands()

    def register_commands(self) -> None:
        local = app_commands.locale_str

        @self.tree.command(
            name=local("transcribe", zh="扒谱"),
            description=local("Turn audio into a guitar sheet", zh="上传音频生成吉他弹唱谱"),
        )
        @app_commands.describe(
            audio=local("Audio file, up to 8 minutes", zh="音频文件，最多 8 分钟"),
            title=local("Song title", zh="歌名"),
            artist=local("Artist", zh="艺术家"),
            lyrics=local(
                "Optional lyrics; aligned instead of recognized", zh="可附歌词，系统只做对齐"
            ),
        )
        async def transcribe(
            interaction: Interaction,
            audio: discord.Attachment,
            title: app_commands.Range[str, 1, 200] | None = None,
            artist: app_commands.Range[str, 1, 200] | None = None,
            lyrics: app_commands.Range[str, 1, 4000] | None = None,
        ) -> None:
            await self.submit(interaction, audio, title, artist, lyrics)

        @self.tree.command(
            name=local("status", zh="进度"),
            description=local("View your jobs or retrieve a result", zh="查看任务进度或取回结果"),
        )
        @app_commands.describe(
            job=local("Optional job ID from your receipt", zh="可填回执上的任务编号取回谱子")
        )
        async def status(interaction: Interaction, job: str | None = None) -> None:
            await interaction.response.defer(ephemeral=True, thinking=True)
            async with self.owner_locks[interaction.user.id]:
                if job:
                    saved = self.store.get(job, interaction.user.id)
                    if saved.status == "failed":
                        raise ServiceError(saved.error or "service")
                    files = await self.render(saved)
                    await interaction.edit_original_response(
                        content=None,
                        embed=sheet_embed(saved, files),
                        attachments=attachments(files, interaction.filesize_limit),
                        view=SheetView(self, saved),
                    )
                else:
                    locale = str(interaction.locale)
                    jobs = self.store.list_jobs(interaction.user.id)
                    lines = []
                    for saved in jobs[-10:]:
                        label = t(locale, saved.status)
                        if saved.status == "queued":
                            label += " · " + t(locale, "ahead", n=self.store.position(saved))
                        if saved.status == "failed":
                            label += ": " + t(locale, saved.error or "service")
                        lines.append(f"`{saved.id}` · {label}")
                    text = "\n".join(lines) if lines else t(locale, "empty")
                    text += "\n" + t(
                        locale, "remaining", n=self.store.remaining(interaction.user.id)
                    )
                    text += "\n`/status job:<ID>`"
                    await interaction.edit_original_response(content=text[:2000])

        @self.tree.command(
            name=local("delete", zh="删除"),
            description=local("Delete all your server data", zh="删除自己的全部服务端数据"),
        )
        async def delete(interaction: Interaction) -> None:
            await interaction.response.defer(ephemeral=True, thinking=True)
            async with self.owner_locks[interaction.user.id]:
                jobs = self.store.cancel_owner(interaction.user.id)
                await self.worker.erase(jobs)
                self.stop_views({j.id for j in jobs})
                removed = True
                for job in jobs:
                    removed = await self.delete_message(job) and removed
                text = t(str(interaction.locale), "deleted")
                if not removed:
                    text += "\n" + t(str(interaction.locale), "delete_messages")
                await interaction.edit_original_response(content=text)

        @self.tree.command(
            name=local("help", zh="帮助"),
            description=local("Usage, limits and privacy", zh="用法、限制与隐私说明"),
        )
        async def help_command(interaction: Interaction) -> None:
            await interaction.response.send_message(
                t(str(interaction.locale), "help", mb=self.settings.max_upload_bytes // 1024**2),
                ephemeral=True,
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
                name=local("Transcribe", zh="生成弹唱谱"),
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
        await self.tree.set_translator(Translator())
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
        self, interaction: Interaction, error: Exception, job_id: str | None = None
    ) -> None:
        code = error.code if isinstance(error, ServiceError) else "service"
        logger.warning(
            "interaction_error code=%s type=%s job=%s", code, type(error).__name__, job_id
        )
        content = t(str(interaction.locale), code)
        if job_id:
            content += f"\n`{job_id}`"
        if not interaction.response.is_done():
            await interaction.response.send_message(content, ephemeral=True)
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
        lyrics: str | None = None,
    ) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        owner, locale = interaction.user.id, str(interaction.locale)
        async with self.owner_locks[owner]:
            suffix = check_attachment(audio.filename, audio.size, self.settings)
            if self.store.remaining(owner) == 0:
                raise ServiceError("quota")
            job_id = uuid.uuid4().hex
            work = work_path(self.settings, job_id)
            work.mkdir(mode=0o700)
            queued = False
            receipt: discord.WebhookMessage | None = None
            try:
                path = work / ("input" + suffix)
                await asyncio.wait_for(audio.save(path), 60)
                check_attachment(path.name, path.stat().st_size, self.settings)
                await inspect_audio(path)
                title = (title or audio.filename.rsplit(".", 1)[0])[:200]
                request = {"audio": path.name, "title": title, "artist": artist, "lyrics": lyrics}
                (work / "request.json").write_text(
                    json.dumps(request, ensure_ascii=False), encoding="utf-8"
                )
                ahead = sum(j.status in {"queued", "running"} for j in self.store.list_jobs())
                # Finish the ephemeral defer before creating a *public* followup.
                # Discord otherwise treats the first followup as editing the private receipt.
                await interaction.edit_original_response(content=t(locale, "validated"))
                receipt = await interaction.followup.send(
                    t(
                        locale,
                        "receipt",
                        title=discord.utils.escape_markdown(title),
                        ahead=ahead,
                        minutes=max(1, (ahead + 1) * 2),
                        id=job_id,
                    ),
                    ephemeral=False,
                    wait=True,
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
                await interaction.edit_original_response(content=t(locale, "accepted"))
            except TimeoutError as exc:
                raise ServiceError("timeout") from exc
            finally:
                if not queued:
                    shutil.rmtree(work, ignore_errors=True)
                    if receipt is not None:
                        with contextlib.suppress(discord.HTTPException):
                            await receipt.delete()

    async def render(self, job: Job, options: Arrangement | None = None) -> SheetFiles:
        async with self.render_gate:
            return await asyncio.to_thread(preview, job, options)

    async def control(self, interaction: Interaction, job_id: str, action: str, value: str) -> None:
        # Authorize before deferring an update, so unauthorized clicks cannot touch the message.
        self.store.get(job_id, interaction.user.id)
        if action == "export":
            await interaction.response.defer(ephemeral=True, thinking=True)
        else:
            await interaction.response.defer()
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
            options = changed(job, action, value)
            files = await self.render(job, options)
            # Persist before sending so even a network failure can be recovered via /status.
            self.store.update_arrangement(job_id, interaction.user.id, options)
            updated = self.store.get(job_id, interaction.user.id)
            await interaction.edit_original_response(
                embed=sheet_embed(updated, files),
                attachments=attachments(files, interaction.filesize_limit),
                view=SheetView(self, updated),
            )

    async def deliver_loop(self) -> None:
        await self.wait_until_ready()
        last_sweep = 0.0
        while True:
            self.worker.expire_waiting()
            if self.is_ready():
                (self.settings.data_dir / "heartbeat").touch()
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
        files = await self.render(job) if job.status == "done" else None
        view = SheetView(self, job) if files else None
        embed = sheet_embed(job, files) if files else None
        content = (
            f"<@{job.owner}>" if files else f"{t(job.locale, job.error or 'service')}\n`{job.id}`"
        )
        allowed = discord.AllowedMentions(users=[discord.Object(job.owner)])
        message: discord.Message
        if job.token and job.token_expires > time.time() and job.message_id:
            webhook = discord.Webhook.partial(job.application_id, job.token, client=self)
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

    async def delete_message(self, job: Job) -> bool:
        if job.message_id is None:
            return True
        if job.token and job.token_expires > time.time():
            try:
                await discord.Webhook.partial(
                    job.application_id, job.token, client=self
                ).delete_message(job.message_id)
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
