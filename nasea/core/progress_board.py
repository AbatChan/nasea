"""
Rich plan board rendering for interactive progress updates.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


StatusTuple = Tuple[str, str]


@dataclass
class PlanItem:
    label: str
    status: str = "pending"  # pending, in_progress, done, failed


@dataclass
class PlanSection:
    title: str
    items: List[PlanItem] = field(default_factory=list)


class PlanBoard:
    """Render and update a live project plan with Rich."""

    STATUS_ICONS = {
        "pending": "[dim]○[/dim]",
        "in_progress": "[cyan]◐[/cyan]",
        "done": "[green]●[/green]",
        "failed": "[red]●[/red]",
    }

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.sections: List[PlanSection] = []
        self.events: Deque[StatusTuple] = deque(maxlen=5)
        self.live: Optional[Live] = None

    # Section / task management -------------------------------------------------

    def reset(self, sections: Optional[List[Tuple[str, List[str]]]] = None):
        """Reset sections and timeline."""
        self.sections.clear()
        if sections:
            for title, items in sections:
                self.set_section(title, items)
        else:
            self.refresh()

    def set_section(self, title: str, items: Optional[List[str]] = None):
        """Create or replace a section."""
        section = self._get_section(title)
        if section is None:
            section = PlanSection(title=title)
            self.sections.append(section)

        section.items = [
            PlanItem(label=item, status="pending") for item in (items or [])
        ]
        self.refresh()

    def append_item(self, title: str, label: str):
        section = self._ensure_section(title)
        section.items.append(PlanItem(label=label))
        self.refresh()

    def update_status(
        self,
        title: str,
        index: Optional[int] = None,
        status: str = "pending",
        label: Optional[str] = None,
    ):
        section = self._get_section(title)
        if not section:
            return

        item: Optional[PlanItem] = None
        if index is not None and 0 <= index < len(section.items):
            item = section.items[index]
        elif label is not None:
            for candidate in section.items:
                if candidate.label == label:
                    item = candidate
                    break

        if item:
            item.status = status
            self.refresh()

    # Timeline -----------------------------------------------------------------

    def add_event(self, text: str, style: str = "dim"):
        self.events.appendleft((text, style))
        self.refresh()

    # Live integration ---------------------------------------------------------

    def attach_live(self, live: Live):
        self.live = live
        self.refresh()

    def refresh(self):
        if self.live:
            self.live.update(self.render(), refresh=True)

    # Rendering ----------------------------------------------------------------

    def render(self) -> Panel:
        plan_table = Table.grid(padding=(0, 1))
        plan_table.expand = True

        for section in self.sections:
            plan_table.add_row(Text(section.title, style="bold cyan"))
            if not section.items:
                plan_table.add_row(Text("  —", style="dim"))
                continue

            for item in section.items:
                icon = self.STATUS_ICONS.get(item.status, "[dim]○[/dim]")
                style = {
                    "pending": "dim",
                    "in_progress": "cyan",
                    "done": "green",
                    "failed": "red",
                }.get(item.status, "white")
                plan_table.add_row(
                    Text.assemble(
                        Text("  "),
                        Text.from_markup(icon),
                        Text(" "),
                        Text(item.label, style=style),
                    )
                )

            plan_table.add_row(Text("", style="dim"))

        events_table = Table.grid(expand=True)
        events_table.add_row(Text("Latest activity", style="bold magenta"))
        if not self.events:
            events_table.add_row(Text("  (waiting for updates)", style="dim"))
        else:
            for message, style in list(self.events)[:4]:
                events_table.add_row(Text(f"  {message}", style=style))

        body = Group(plan_table, Text(""), events_table)
        return Panel(body, border_style="cyan", title="Project Plan", expand=True)

    # Helpers ------------------------------------------------------------------

    def _ensure_section(self, title: str) -> PlanSection:
        section = self._get_section(title)
        if section is None:
            section = PlanSection(title=title)
            self.sections.append(section)
        return section

    def _get_section(self, title: str) -> Optional[PlanSection]:
        for section in self.sections:
            if section.title == title:
                return section
        return None

