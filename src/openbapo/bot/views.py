"""Six persistent sheet controls. Every callback is authorized again by the service."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import discord

from openbapo.arrange import StrumOption
from openbapo.arrange.keys import key_name
from openbapo.arrange.sheet import english_key
from openbapo.bot.messages import (
    Interaction,
    ago,
    capo_label,
    key_choice,
    labelled,
    strum_label,
    t,
)
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
            row=3,
            style=discord.ButtonStyle.secondary,
        )

    async def callback(self, interaction: Interaction) -> None:
        assert self.view is not None
        if self.action == "strum":
            await self.view.app.strum_picker(interaction, self.view.job_id)
        elif self.action == "export":
            await self.view.app.export_picker(interaction, self.view.job_id)
        else:
            await self.view.app.control(interaction, self.view.job_id, self.action, "")


class SheetView(discord.ui.View):
    """The sheet message's controls. With `busy`, the same controls greyed out
    while that action redraws the sheet; a clicked button reads "更新中…"."""

    def __init__(self, app: OpenBapo, job: Job, busy: str | None = None) -> None:
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
                        label=labelled(
                            job.locale,
                            "transpose",
                            key_choice(
                                job.locale, english_key(key_name((tonic + n) % 12, mode)), n
                            ),
                        ),
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
                        label=labelled(
                            job.locale,
                            "fingering",
                            english_key(o.name)
                            + " · "
                            + capo_label(job.locale, o.capo, o.tune_down),
                        ),
                        value=str(o.tonic),
                        default=o.selected,
                        description=None
                        if o.recommended
                        else t(job.locale, "retune" if o.tune_down else "high_capo"),
                    )
                    for o in sheet.fingering_options
                ],
            )
        )
        self.add_item(Action(job, "strum", t(job.locale, "strum_button")))
        self.add_item(Action(job, "export", t(job.locale, "export")))
        key = "simplify_on" if options.simplify_chords else "simplify_off"
        self.add_item(Action(job, "simplify", t(job.locale, key)))
        self.add_item(Action(job, "reset", t(job.locale, "reset")))
        if busy is not None:
            for item in self.children:
                assert isinstance(item, Choice | Action)
                item.disabled = True
                if isinstance(item, Action) and item.action == busy:
                    item.label = t(job.locale, "updating")

    async def on_error(
        self, interaction: Interaction, error: Exception, item: discord.ui.Item[Any]
    ) -> None:
        await self.app.report_error(interaction, error, self.job_id)


class StrumPicker(discord.ui.View):
    """Ephemeral: a numbered picture of the song's patterns and a menu of the same numbers.

    `sheet` is the interaction from the sheet message's button; its token edits
    that message once a pattern is picked (valid 15 minutes, hence the timeout).
    """

    def __init__(
        self,
        app: OpenBapo,
        job: Job,
        options: tuple[StrumOption, ...],
        sheet: Interaction,
    ) -> None:
        super().__init__(timeout=600)
        self.app, self.job_id, self.sheet = app, job.id, sheet
        current = Arrangement.from_json(job.arrangement).strum_pattern
        recommended = next((o for o in options if o.recommended), None)
        choices = [
            discord.SelectOption(
                label=t(job.locale, "auto"),
                value="auto",
                description=strum_label(job.locale, options, recommended.id)
                if recommended
                else None,
                default=current == "auto",
            ),
            discord.SelectOption(
                label=t(job.locale, "none"), value="none", default=current == "none"
            ),
        ]
        choices += [
            discord.SelectOption(
                label=strum_label(job.locale, options, o.id),
                value=o.id,
                default=current == o.id,
            )
            for o in options
        ]
        menu: discord.ui.Select[StrumPicker] = discord.ui.Select(
            placeholder=t(job.locale, "strum"), options=choices
        )
        menu.callback = self.picked  # type: ignore[method-assign]
        self.menu = menu
        self.add_item(menu)

    async def picked(self, interaction: Interaction) -> None:
        await self.app.pick_strum(interaction, self.sheet, self.job_id, self.menu.values[0])
        self.stop()

    async def on_error(
        self, interaction: Interaction, error: Exception, item: discord.ui.Item[Any]
    ) -> None:
        await self.app.report_error(interaction, error, self.job_id)


class DeleteConfirm(discord.ui.View):
    """Ephemeral: delete everything only after the owner confirms."""

    def __init__(self, app: OpenBapo, origin: Interaction) -> None:
        super().__init__(timeout=120)
        self.app, self.origin = app, origin
        self.owner, self.locale = origin.user.id, str(origin.locale)
        confirm: discord.ui.Button[DeleteConfirm] = discord.ui.Button(
            label=t(self.locale, "delete_yes"), style=discord.ButtonStyle.danger
        )
        cancel: discord.ui.Button[DeleteConfirm] = discord.ui.Button(
            label=t(self.locale, "delete_no"), style=discord.ButtonStyle.secondary
        )
        confirm.callback = self.confirmed  # type: ignore[method-assign]
        cancel.callback = self.cancelled  # type: ignore[method-assign]
        self.add_item(confirm)
        self.add_item(cancel)

    async def interaction_check(self, interaction: Interaction) -> bool:
        return interaction.user.id == self.owner

    async def confirmed(self, interaction: Interaction) -> None:
        self.stop()
        await interaction.response.edit_message(content=t(self.locale, "deleting"), view=None)
        text = await self.app.erase_owner(self.owner, self.locale)
        await interaction.edit_original_response(content=text)

    async def cancelled(self, interaction: Interaction) -> None:
        self.stop()
        await interaction.response.edit_message(content=t(self.locale, "delete_kept"), view=None)

    async def on_timeout(self) -> None:
        with contextlib.suppress(discord.HTTPException):
            await self.origin.edit_original_response(
                content=t(self.locale, "delete_expired"), view=None
            )


class ExportPicker(discord.ui.View):
    """Ephemeral: one button per format; each click sends that file privately."""

    FORMATS = ("pdf", "png", "musicxml")

    def __init__(self, app: OpenBapo, job: Job) -> None:
        super().__init__(timeout=600)
        self.app, self.job_id = app, job.id
        for fmt in self.FORMATS:
            label = "PDF" if fmt == "pdf" else t(job.locale, f"export_{fmt}")
            button: discord.ui.Button[ExportPicker] = discord.ui.Button(
                label=label, style=discord.ButtonStyle.primary
            )
            button.callback = self._sender(fmt)  # type: ignore[method-assign]
            self.add_item(button)

    def _sender(self, fmt: str) -> Any:
        async def send(interaction: Interaction) -> None:
            await self.app.control(interaction, self.job_id, "export", fmt)

        return send

    async def on_error(
        self, interaction: Interaction, error: Exception, item: discord.ui.Item[Any]
    ) -> None:
        await self.app.report_error(interaction, error, self.job_id)


class StatusPicker(discord.ui.View):
    """Ephemeral: open one of the owner's finished sheets by its title."""

    def __init__(self, app: OpenBapo, jobs: list[Job], locale: str) -> None:
        super().__init__(timeout=600)
        self.app = app
        menu: discord.ui.Select[StatusPicker] = discord.ui.Select(
            placeholder=t(locale, "status_open"),
            options=[
                discord.SelectOption(
                    label=(job.title or t(locale, "untitled"))[:100],
                    # Same titles are told apart by age; menus cannot show local time.
                    description=ago(locale, job.created),
                    value=job.id,
                )
                for job in jobs[-25:]
            ],
        )
        menu.callback = self.picked  # type: ignore[method-assign]
        self.menu = menu
        self.add_item(menu)

    async def picked(self, interaction: Interaction) -> None:
        await self.app.show_job(interaction, self.menu.values[0])

    async def on_error(
        self, interaction: Interaction, error: Exception, item: discord.ui.Item[Any]
    ) -> None:
        await self.app.report_error(
            interaction, error, self.menu.values[0] if self.menu.values else None
        )
