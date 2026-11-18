"""Six persistent sheet controls. Every callback is authorized again by the service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import discord

from openbapo.arrange.keys import key_name
from openbapo.bot.messages import Interaction, t
from openbapo.model import Analysis, Arrangement, Mode
from openbapo.service.rendering import sheet_for
from openbapo.service.store import Job

if TYPE_CHECKING:
    from openbapo.bot.app import OpenBapo


class Choice(discord.ui.Select["SheetView"]):
    def __init__(
        self, job: Job, action: str, row: int, options: list[discord.SelectOption]
    ) -> None:
        self.action = action
        super().__init__(
            custom_id=f"ob:{job.id}:{action}",
            placeholder=t(job.locale, action),
            options=options,
            row=row,
        )

    async def callback(self, interaction: Interaction) -> None:
        assert self.view is not None
        await self.view.app.control(interaction, self.view.job_id, self.action, self.values[0])


class Action(discord.ui.Button["SheetView"]):
    def __init__(self, job: Job, action: str, label: str) -> None:
        self.action = action
        super().__init__(
            custom_id=f"ob:{job.id}:{action}",
            label=label,
            row=4,
            style=discord.ButtonStyle.secondary,
        )

    async def callback(self, interaction: Interaction) -> None:
        assert self.view is not None
        await self.view.app.control(interaction, self.view.job_id, self.action, "")


class SheetView(discord.ui.View):
    def __init__(self, app: OpenBapo, job: Job) -> None:
        super().__init__(timeout=None)
        self.app, self.job_id = app, job.id
        sheet = sheet_for(job)
        options = Arrangement.from_json(job.arrangement)
        analysis = Analysis.from_json(job.analysis or "{}")
        tonic = analysis.key.tonic if analysis.key else 0
        mode = analysis.key.mode if analysis.key else Mode.MAJOR
        self.add_item(
            Choice(
                job,
                "transpose",
                0,
                [
                    discord.SelectOption(
                        label=f"{key_name((tonic + n) % 12, mode)} ({n:+d})",
                        value=str(n),
                        default=n == options.transpose,
                    )
                    for n in range(-6, 7)
                ],
            )
        )
        self.add_item(
            Choice(
                job,
                "fingering",
                1,
                [
                    discord.SelectOption(
                        label=f"{o.name} · {t(job.locale, 'capo', n=o.capo)}",
                        value=str(o.tonic),
                        default=o.selected,
                        description=None if o.recommended else t(job.locale, "high_capo"),
                    )
                    for o in sheet.fingering_options
                ],
            )
        )
        rhythms = [
            discord.SelectOption(
                label=t(job.locale, choice), value=choice, default=options.strum_pattern == choice
            )
            for choice in ("auto", "none")
        ]
        for rhythm in sheet.strum_options:
            rhythms.append(
                discord.SelectOption(
                    label=(rhythm.name if job.locale.startswith("zh") else rhythm.id)[:100],
                    description=(rhythm.mnemonic if job.locale.startswith("zh") else rhythm.kind)[
                        :100
                    ],
                    value=rhythm.id,
                    default=options.strum_pattern == rhythm.id,
                )
            )
        self.add_item(Choice(job, "strum", 2, rhythms))
        self.add_item(
            Choice(
                job,
                "export",
                3,
                [
                    discord.SelectOption(label=label, value=value)
                    for label, value in (("PDF", "pdf"), ("PNG", "png"), ("MusicXML", "musicxml"))
                ],
            )
        )
        key = "simplify_on" if options.simplify_chords else "simplify_off"
        self.add_item(Action(job, "simplify", t(job.locale, key)))
        self.add_item(Action(job, "reset", t(job.locale, "reset")))

    async def on_error(
        self, interaction: Interaction, error: Exception, item: discord.ui.Item[Any]
    ) -> None:
        await self.app.report_error(interaction, error, self.job_id)
