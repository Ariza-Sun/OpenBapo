"""Short, localized Discord copy. Never include raw model exceptions."""

import time
from collections.abc import Sequence
from typing import Any

import discord

from openbapo.analysis.audio import MAX_DURATION_SECONDS
from openbapo.arrange.sheet import StrumOption, english_key
from openbapo.service.config import Settings
from openbapo.service.rendering import SheetFiles
from openbapo.service.store import Job

Interaction = discord.Interaction[Any]
COLOUR = 0x315B46  # embeds: help and sheet messages
MAX_MINUTES = MAX_DURATION_SECONDS // 60  # longest song accepted

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
    "long": (
        "歌曲超过 {minutes} 分钟，请裁剪后重新上传。",
        "Audio exceeds {minutes} minutes. Please trim it.",
    ),
    "quota": (
        "今天的 {q} 首已经用完，{reset} 重置。",
        "Your {q} uploads for today are used. They reset {reset}.",
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
    "gpu": (
        "服务器显卡暂时被占满，请稍后重新上传。",
        "The server's GPU is busy right now. Please upload again later.",
    ),
    "service": (
        "处理时出了问题，请稍后重新上传。{contact}",
        "Something went wrong while processing. Please upload again later.{contact}",
    ),
    "contact": (
        "如果一直出错，请把任务编号发到 {contact}。",
        " If it keeps failing, send the job ID to {contact}.",
    ),
    "restarted": (
        "服务中断，此任务已停止并清理音频，请重新上传。",
        "The service was interrupted. Audio was removed; please upload again.",
    ),
    # Why a job failed, in a /status line; the full message went out with the job.
    "why_timeout": ("超时", "timed out"),
    "why_long": ("音频太长", "audio too long"),
    "why_corrupt": ("音频无法读取", "unreadable audio"),
    "why_gpu": ("服务器繁忙", "server busy"),
    "why_restarted": ("服务中断", "service interrupted"),
    "why_service": ("服务异常", "service error"),
    "just_now": ("刚刚", "just now"),
    "minutes_ago": ("{n} 分钟前", "{n} min ago"),
    "hours_ago": ("{n} 小时前", "{n} h ago"),
    "days_ago": ("{n} 天前", "{n} days ago"),
    "queued": ("排队中", "Queued"),
    "running": ("分析中", "Analyzing"),
    "done": ("已完成", "Done"),
    "failed": ("失败", "Failed"),
    "cancelled": ("已取消", "Cancelled"),
    "receipt": (
        "🎸 已收到《{title}》，排队中，前面还有 {ahead} 首，预计 {eta} 出谱。",
        "🎸 Received {title}. In the queue with {ahead} ahead; expected {eta}.",
    ),
    "receipt_thread": (
        "🎸 《{title}》：进度和谱子在下面的子区里。",
        "🎸 {title}: progress and the sheet are in the thread below.",
    ),
    "receipt_running": (
        "🎸 正在分析《{title}》：{step}（第 {k}/{n} 步）…",
        "🎸 Analyzing {title}: {step} (step {k} of {n})…",
    ),
    "step_decode": ("读取音频", "reading the audio"),
    "step_separate": ("分离人声和伴奏", "separating vocals from the band"),
    "step_beats": ("找节拍", "finding the beat"),
    "step_chords": ("识别和弦", "recognizing chords"),
    "step_lyrics": ("识别歌词", "recognizing lyrics"),
    "accepted": ("已加入队列。可用 /status 查看进度。", "Queued. Use /status to check progress."),
    "ahead": ("前面还有 {n} 首", "{n} ahead"),
    "empty": ("你还没有保留中的谱子。", "You have no sheets kept."),
    "unlimited": ("你的上传不限量。", "Your uploads are unlimited."),
    "remaining": (
        "今天还能上传 {n}/{q} 首，{reset} 重置。",
        "{n} of {q} uploads left today; they reset {reset}.",
    ),
    "status_head": ("你最近的谱子：", "Your recent sheets:"),
    "status_open": ("打开一份谱子…", "Open a sheet…"),
    "untitled": ("未命名", "Untitled"),
    "deleted": (
        "服务端数据已删除，处理任务已停止。当日配额不重置。Discord 中原先上传的附件请自行删除。",
        "Server data deleted and active jobs stopped. Daily quota is unchanged. "
        "Delete your original Discord attachments separately.",
    ),
    "delete_confirm": (
        "将停止并删除你在服务器上的全部 {n} 个任务和谱子，删除后不能恢复。确定吗？",
        "This stops and deletes all {n} of your jobs and sheets on the server. "
        "It cannot be undone. Delete?",
    ),
    "delete_yes": ("确认删除", "Delete"),
    "delete_no": ("取消", "Cancel"),
    "deleting": ("正在删除…", "Deleting…"),
    "delete_kept": ("已取消，什么都没有删除。", "Cancelled; nothing was deleted."),
    "delete_expired": (
        "确认已超时，什么都没有删除。需要时请重新输入 /delete。",
        "This confirmation timed out; nothing was deleted. Run /delete again if needed.",
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
    "strum_button": ("换节奏型…", "Strum pattern…"),
    "strum_pick": (
        "对照图里的编号选节奏型（只有你能看到这条消息，10 分钟内有效）。",
        "Pick a strum pattern by its number in the picture (only you see this; 10 minutes).",
    ),
    "strum_chosen": ("节奏型已换成：{name}", "Strum pattern: {name}"),
    "strum_kind_strum": ("扫弦", "Strum"),
    "strum_kind_arpeggio": ("分解", "Fingerpicking"),
    "recommended": ("推荐", "recommended"),
    "labelled": ("{name}：{value}", "{name}: {value}"),
    "export": ("导出…", "Export…"),
    "export_pick": (
        "选择导出格式（只有你能看到这条消息）：",
        "Choose a format (only you see this):",
    ),
    "export_png": ("图片 PNG", "Image (PNG)"),
    "export_musicxml": ("MusicXML（打谱软件用）", "MusicXML (for notation apps)"),
    "key_original": ("{key}（原调）", "{key} (original)"),
    "key_up": ("{key}（升 {n} 个半音）", "{key} (up {n} semitones)"),
    "key_down": ("{key}（降 {n} 个半音）", "{key} (down {n} semitones)"),
    "reset": ("重置", "Reset"),
    "updating": ("更新中…", "Updating…"),
    "auto": ("自动推荐", "Recommended"),
    "none": ("无节奏型", "No strum pattern"),
    "simplify_on": (
        "简化和弦：开（七和弦等弹成三和弦）",
        "Simple chords: on (7ths etc. as triads)",
    ),
    "simplify_off": ("简化和弦：关", "Simple chords: off"),
    "capo": ("夹 {n} 品", "Capo {n}"),
    "tune_down_1": ("降半音调弦", "Tune down ½ step"),
    "tune_down_2": ("降全音调弦", "Tune down 1 step"),
    "high_capo": ("不推荐", "High capo"),
    "retune": ("需要调弦", "Retune the strings"),
    "keys": (
        "原调 {original} · 演唱调 {sung} · {fingering} 指法 · {capo}",
        "Original {original} · Singing {sung} · {fingering} shapes · {capo}",
    ),
    "no_lyrics": (
        "未识别到歌词；和弦谱仍可使用。",
        "No lyrics detected; the chord sheet is still available.",
    ),
    "no_key": (
        "未确定调性，暂按 C 大调编配。",
        "Key could not be determined; arranged in C major.",
    ),
    "help_title": ("OpenBapo · 吉他弹唱谱", "OpenBapo · guitar play-along sheets"),
    "help_intro": (
        "上传一首歌，几分钟后得到带和弦图、节奏型和歌词的弹唱谱。",
        "Upload a song; a few minutes later you get a sheet with chord diagrams, "
        "a strum pattern and lyrics.",
    ),
    "help_upload_name": ("① 上传", "① Upload"),
    "help_upload": (
        "`/transcribe` → 在 **audio** 里选音频文件\n"
        "也可以：右键带音频的消息 → APP → Transcribe\n"
        "在服务器频道里用，每首歌会单独开一个子区；不想让别人看到，"
        "点 OpenBapo 头像 → 发消息，在私信里用",
        "`/transcribe` → pick the audio file as **audio**\n"
        "Or: right-click a message with audio → Apps → Transcribe\n"
        "In a server channel each song gets its own thread; to keep it private, "
        "click OpenBapo's avatar → Message and use it there",
    ),
    "help_wait_name": ("② 等待", "② Wait"),
    "help_wait": (
        "回执会实时显示排队和分析进度，通常几分钟。",
        "The receipt shows your place in the queue and each analysis step; usually a few minutes.",
    ),
    "help_adjust_name": ("③ 调整与导出", "③ Adjust and export"),
    "help_adjust": (
        "换演唱调 · 指法调 · 节奏型 · 简化和弦\n导出 PDF / 图片 / MusicXML",
        "Singing key · chord shapes · strum · simple chords\nExport PDF, image or MusicXML",
    ),
    "help_commands_name": ("其他命令", "Other commands"),
    "help_commands": (
        "`/status`  查看进度、打开以前的谱子\n`/delete`  删除你的全部数据",
        "`/status`  progress and earlier sheets\n`/delete`  delete all your data",
    ),
    "help_limits_name": ("限制", "Limits"),
    "help_limits": (
        "每首不超过 {minutes} 分钟、{mb} MB\n"
        "支持的音频格式：mp3、m4a、ogg、opus、wav、flac\n"
        "每人每天 {q} 首，{reset} 重置",
        "Each song: up to {minutes} minutes and {mb} MB\n"
        "Supported audio formats: mp3, m4a, ogg, opus, wav, flac\n"
        "{q} songs per person per day; resets {reset}",
    ),
    "help_privacy_name": ("隐私与版权", "Privacy and copyright"),
    "help_privacy": (
        "OpenBapo 服务器分析完音频后自动删除它；你在 Discord 里发出的附件存在 Discord 上，"
        "需要你自己删除\n"
        "谱子在服务器上保留 30 天后自动删除，不用于训练模型\n"
        "歌曲和歌词的版权归原作者。只上传你拥有版权或已获授权的音频；"
        "谱子仅供个人练习，请勿商用或公开传播",
        "The OpenBapo server deletes your audio as soon as it is analyzed; the attachment "
        "you posted stays on Discord until you delete it\n"
        "Sheets are deleted from the server after 30 days and never used for training\n"
        "Songs and lyrics belong to their copyright holders. Only upload audio you own "
        "or are licensed to use; sheets are for personal practice, not for sale or "
        "public sharing",
    ),
    "help_footer": ("问题与反馈：{contact}", "Questions and feedback: {contact}"),
}


def t(locale: str, key: str, /, **values: object) -> str:
    return COPY[key][0 if locale.startswith("zh") else 1].format(**values)


def failure_reason(locale: str, code: str | None) -> str:
    """A failed job's short reason for a list, e.g. "服务异常"."""
    key = f"why_{code}"
    return t(locale, key if key in COPY else "why_service")


def ago(locale: str, then: float, now: float | None = None) -> str:
    """ "35 分钟前": for menus, where Discord's local-time stamps do not render."""
    seconds = max(0, int((time.time() if now is None else now) - then))
    if seconds < 60:
        return t(locale, "just_now")
    if seconds < 3600:
        return t(locale, "minutes_ago", n=seconds // 60)
    if seconds < 86400:
        return t(locale, "hours_ago", n=seconds // 3600)
    return t(locale, "days_ago", n=seconds // 86400)


def next_reset(now: float | None = None) -> int:
    """The next UTC midnight, when daily quotas reset (Unix seconds)."""
    day = 86400
    return (int(time.time() if now is None else now) // day + 1) * day


def local_time(unix: int) -> str:
    """Discord shows this in each reader's own time zone and clock format."""
    return f"<t:{unix}:t>"


def error_text(locale: str, code: str, settings: Settings) -> str:
    """The user-facing text for a service error code, with the reset time and
    the operator's contact filled in where they apply."""
    contact = t(locale, "contact", contact=settings.contact) if settings.contact else ""
    return t(
        locale,
        code,
        q=settings.daily_quota,
        reset=local_time(next_reset()),
        contact=contact,
        minutes=MAX_MINUTES,
    )


def help_embed(locale: str, settings: Settings) -> discord.Embed:
    """/help as a card: three steps to a sheet, other commands, limits, privacy
    and copyright, and the operator's contact in the footer when one is set."""
    embed = discord.Embed(
        title=t(locale, "help_title"), description=t(locale, "help_intro"), colour=COLOUR
    )
    for name in ("upload", "wait", "adjust", "commands"):
        embed.add_field(
            name=t(locale, f"help_{name}_name"), value=t(locale, f"help_{name}"), inline=False
        )
    # Limits and privacy are full sentences now, too long to sit side by side.
    embed.add_field(
        name=t(locale, "help_limits_name"),
        value=t(
            locale,
            "help_limits",
            minutes=MAX_MINUTES,
            mb=settings.max_upload_bytes // 1024**2,
            q=settings.daily_quota,
            reset=local_time(next_reset()),
        ),
        inline=False,
    )
    embed.add_field(
        name=t(locale, "help_privacy_name"), value=t(locale, "help_privacy"), inline=False
    )
    if settings.contact:
        embed.set_footer(text=t(locale, "help_footer", contact=settings.contact))
    return embed


def strum_kind(locale: str, option: StrumOption) -> str:
    """What follows a pattern's number: "扫弦", "分解 · 推荐". Patterns have
    numbers, not names; the picture shows the rest."""
    kind = t(locale, f"strum_kind_{option.kind}")
    return f"{kind} · {t(locale, 'recommended')}" if option.recommended else kind


def strum_label(locale: str, options: Sequence[StrumOption], option_id: str) -> str:
    """A pattern as the menu and the confirmation show it: "3 · 分解 · 推荐"."""
    number, option = next((k, o) for k, o in enumerate(options, 1) if o.id == option_id)
    return f"{number} · {strum_kind(locale, option)}"


def labelled(locale: str, action: str, value: str) -> str:
    """A menu entry that says which menu it is in, since a closed Discord menu
    shows only its current entry: "演唱调：Ab（原调）"."""
    return t(locale, "labelled", name=t(locale, action), value=value)


def key_choice(locale: str, key: str, steps: int) -> str:
    """A singing-key menu entry: "F（升 3 个半音）", "G#m（原调）"."""
    if steps == 0:
        return t(locale, "key_original", key=key)
    return t(locale, "key_up" if steps > 0 else "key_down", key=key, n=abs(steps))


def capo_label(locale: str, capo: int, tune_down: int) -> str:
    """ "夹 4 品" / "降半音调弦": how the shapes are brought to the sung key."""
    return t(locale, f"tune_down_{tune_down}") if tune_down else t(locale, "capo", n=capo)


def sheet_embed(job: Job, files: SheetFiles) -> discord.Embed:
    sheet = files.sheet
    h = sheet.header
    description = t(
        job.locale,
        "keys",
        original=english_key(h.original_key) if h.original_key else "?",
        sung=english_key(h.sung_key),
        fingering=english_key(h.fingering_key),
        capo=capo_label(job.locale, h.capo, h.tune_down),
    )
    description += f"\n{h.bpm:.0f} BPM" if h.bpm is not None else ""
    description += f" · {h.time_signature[0]}/{h.time_signature[1]}"
    if h.artist:
        description += "\n" + discord.utils.escape_markdown(h.artist[:200])
    if not sheet.lyrics:
        description += "\n" + t(job.locale, "no_lyrics")
    if h.original_key is None:
        description += "\n" + t(job.locale, "no_key")
    embed = discord.Embed(
        title=(h.title or "OpenBapo")[:200], description=description, colour=COLOUR
    )
    embed.set_image(url="attachment://sheet.png")
    embed.set_footer(text=f"OpenBapo · {job.id}")
    return embed
