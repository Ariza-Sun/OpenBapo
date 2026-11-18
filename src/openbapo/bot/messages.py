"""Short, localized Discord copy. Never include raw model exceptions."""

from typing import Any

import discord
from discord import app_commands

from openbapo.service.rendering import SheetFiles
from openbapo.service.store import Job

Interaction = discord.Interaction[Any]

COPY = {
    "format": (
        "支持 mp3 / m4a / ogg / opus / wav / flac，请重新上传。",
        "Upload mp3, m4a, ogg, opus, wav or flac audio.",
    ),
    "size": (
        "附件超过服务大小限制，请压缩为 mp3 后重试。",
        "This file exceeds the service limit. Please compress it to MP3.",
    ),
    "corrupt": (
        "无法读取音频，请检查文件后重新上传。",
        "Cannot read this audio. Please check the file.",
    ),
    "long": ("歌曲超过 8 分钟，请裁剪后重新上传。", "Audio exceeds 8 minutes. Please trim it."),
    "quota": (
        "今天的 5 首配额已用完，UTC 零点重置。",
        "Your 5 daily uploads are used. Resets at 00:00 UTC.",
    ),
    "busy": ("队列已满，请稍后再试。", "The queue is full. Please try again later."),
    "timeout": (
        "任务超过 15 分钟，已停止并清理音频。",
        "The 15-minute deadline was exceeded. Audio was removed.",
    ),
    "owner": (
        "只有发起者能操作这张谱。请用 /transcribe 生成自己的一份。",
        "Only the requester can use these controls. Use /transcribe to make your own sheet.",
    ),
    "expired": (
        "这份结果已删除或超过 30 天，请重新上传。",
        "This result was deleted or expired after 30 days. Upload again.",
    ),
    "pending": ("任务尚未完成，请稍后查看 /status。", "Still processing. Check /status shortly."),
    "option": (
        "这个选项已不可用，请重新选择。",
        "This option is unavailable. Please choose again.",
    ),
    "service": (
        "处理遇到服务错误，请带任务编号联系维护者。",
        "A service error occurred. Contact the operator with the job ID.",
    ),
    "restarted": (
        "服务中断，此任务已停止并清理音频，请重新上传。",
        "The service was interrupted. Audio was removed; please upload again.",
    ),
    "queued": ("排队中", "Queued"),
    "running": ("分析中", "Analyzing"),
    "done": ("已完成", "Done"),
    "failed": ("失败", "Failed"),
    "cancelled": ("已取消", "Cancelled"),
    "receipt": (
        "已收到《{title}》。前面还有 {ahead} 首，预计约 {minutes} 分钟。任务 `{id}`。",
        "Received {title}. {ahead} ahead; about {minutes} min. Job `{id}`.",
    ),
    "accepted": ("已加入队列。可用 /status 查看进度。", "Queued. Use /status to check progress."),
    "validated": ("音频已通过校验，正在加入队列。", "Audio validated. Adding it to the queue."),
    "ahead": ("前面还有 {n} 首", "{n} ahead"),
    "empty": ("没有保留中的任务。", "No retained jobs."),
    "remaining": (
        "今日剩余 {n}/5 首，UTC 零点重置。",
        "{n}/5 uploads left today; resets at 00:00 UTC.",
    ),
    "deleted": (
        "服务端数据已删除，处理任务已停止。当日配额不重置。Discord 中原先上传的附件请自行删除。",
        "Server data deleted and active jobs stopped. Daily quota is unchanged. "
        "Delete your original Discord attachments separately.",
    ),
    "delete_messages": (
        "部分历史消息无法撤回，请在 Discord 中手动删除。",
        "Some earlier messages could not be removed; delete them in Discord manually.",
    ),
    "no_audio": (
        "请选择带有一个音频附件的消息。",
        "Choose a message containing one audio attachment.",
    ),
    "many_audio": (
        "消息中有多个音频，请用 /transcribe 选定其中一首。",
        "This message has multiple audio files. Use /transcribe to choose one.",
    ),
    "export_size": (
        "导出文件超过当前频道附件限制，请选择 PDF 或 MusicXML。",
        "The export exceeds this channel's file limit. Try PDF or MusicXML.",
    ),
    "transpose": ("演唱调", "Singing key"),
    "fingering": ("指法调", "Chord shapes"),
    "strum": ("节奏型", "Strum pattern"),
    "export": ("导出", "Export"),
    "reset": ("重置", "Reset"),
    "auto": ("自动推荐", "Recommended"),
    "none": ("无节奏型", "No strum pattern"),
    "simplify_on": ("简化和弦：开", "Simplify: on"),
    "simplify_off": ("简化和弦：关", "Simplify: off"),
    "capo": ("夹 {n} 品", "Capo {n}"),
    "high_capo": ("不推荐", "High capo"),
    "keys": (
        "原调 {original} · 演唱调 {sung} · {fingering} 指法 · 夹 {capo} 品",
        "Original {original} · Singing {sung} · {fingering} shapes · Capo {capo}",
    ),
    "low": (
        "有 {n} 个和弦置信度较低，已在谱上标出。可导出 MusicXML 精修。",
        "{n} low-confidence chords are marked. Export MusicXML for fine editing.",
    ),
    "no_lyrics": (
        "未识别到歌词；和弦谱仍可使用。",
        "No lyrics detected; the chord sheet is still available.",
    ),
    "no_key": (
        "未确定调性，暂按 C 大调编配。",
        "Key could not be determined; arranged in C major.",
    ),
    "help": (
        "/transcribe 上传音频，可附歌名、艺术家、歌词；或右键音频消息 → 应用 → Transcribe。\n"
        "支持 mp3/m4a/ogg/opus/wav/flac，最多 8 分钟、{mb} MiB，每人每天 5 首（UTC）。\n"
        "出谱后可选演唱调、指法调、节奏型，切换简化、重置，导出 PNG/PDF/MusicXML。\n"
        "/status 查看进度和取回最近结果；/delete 删除自己的服务端数据。\n"
        "音频处理后即删，结果保留 30 天；不用于训练。只上传你有权处理的音频。\n"
        "自动识别可能漏掉孤立短句；上传时附歌词可避免这项过滤。简谱旋律尚未提供。",
        "/transcribe uploads audio with optional title, artist and lyrics; "
        "or right-click an audio message → Apps → Transcribe.\n"
        "mp3/m4a/ogg/opus/wav/flac; up to 8 minutes and {mb} MiB. 5 uploads per user per UTC day.\n"
        "Change singing key, chord shapes, strum pattern or simplification; "
        "reset; export PNG/PDF/MusicXML.\n"
        "/status shows progress and retrieves recent results; /delete removes your server data.\n"
        "Audio is removed after processing; results last 30 days. No training use. "
        "Only upload audio you may process.\n"
        "Automatic recognition may omit isolated short vocals; supplied lyrics bypass that filter. "
        "Vocal melody notation is not available yet.",
    ),
}


def t(locale: str, key: str, **values: object) -> str:
    return COPY[key][0 if locale.startswith("zh") else 1].format(**values)


def sheet_embed(job: Job, files: SheetFiles) -> discord.Embed:
    sheet = files.sheet
    h = sheet.header
    description = t(
        job.locale,
        "keys",
        original=h.original_key or "?",
        sung=h.sung_key,
        fingering=h.fingering_key,
        capo=h.capo,
    )
    description += f"\n{h.bpm:.0f} BPM" if h.bpm is not None else ""
    description += f" · {h.time_signature[0]}/{h.time_signature[1]}"
    if h.artist:
        description += "\n" + discord.utils.escape_markdown(h.artist[:200])
    if sheet.low_confidence_count:
        description += "\n" + t(job.locale, "low", n=sheet.low_confidence_count)
    if not sheet.lyrics:
        description += "\n" + t(job.locale, "no_lyrics")
    if h.original_key is None:
        description += "\n" + t(job.locale, "no_key")
    embed = discord.Embed(
        title=(h.title or "OpenBapo")[:200], description=description, colour=0x315B46
    )
    embed.set_image(url="attachment://sheet.png")
    embed.set_footer(text=f"OpenBapo · {job.id}")
    return embed


class Translator(app_commands.Translator):
    async def translate(
        self,
        string: app_commands.locale_str,
        locale: discord.Locale,
        context: app_commands.TranslationContextTypes,
    ) -> str | None:
        if str(locale).startswith("zh"):
            value = string.extras.get("zh")
            return str(value) if value is not None else None
        return None
