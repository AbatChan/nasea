"""
Status reporting system for friendly progress updates.
"""

from typing import Dict, List, Optional

from rich.console import Console

from nasea.core.progress_board import PlanBoard


class StatusReporter:
    """Handles friendly status messages during project generation."""

    def __init__(self, console: Optional[Console] = None, verbose: bool = False):
        """Initialize status reporter."""
        self.console = console or Console()
        self.verbose = verbose
        self.board = PlanBoard(console=self.console)
        self._task_index: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Live/plan helpers
    # ------------------------------------------------------------------
    def attach_live(self, live) -> None:
        """Attach a Rich Live instance so the board can refresh."""
        self.board.attach_live(live)

    def reset_plan(self) -> None:
        """Reset plan sections to a default state for a new generation."""
        self._task_index.clear()
        self.board.reset(
            [
                ("Planning", ["Analyze prompt", "Outline tasks"]),
                ("Implementation", []),
                ("Verification", ["Run tests"]),
                ("Finalization", ["Finalize project"]),
            ]
        )

    def register_tasks(self, tasks: List[Dict[str, str]]) -> None:
        """Register implementation tasks for progress tracking."""
        labels: List[str] = []
        self._task_index.clear()

        for idx, task in enumerate(tasks):
            label = task.get("description") or f"Task {idx + 1}"
            labels.append(label)
            task_id = task.get("id") or f"task_{idx + 1}"
            self._task_index[task_id] = idx

        self.board.set_section("Implementation", labels)

    def task_started(self, task_id: str) -> None:
        idx = self._task_index.get(task_id)
        if idx is not None:
            self.board.update_status("Implementation", index=idx, status="in_progress")

    def task_completed(self, task_id: str) -> None:
        idx = self._task_index.get(task_id)
        if idx is not None:
            self.board.update_status("Implementation", index=idx, status="done")

    def task_failed(self, task_id: str) -> None:
        idx = self._task_index.get(task_id)
        if idx is not None:
            self.board.update_status("Implementation", index=idx, status="failed")

    # ------------------------------------------------------------------
    # Public update helpers (mirrors previous API)
    # ------------------------------------------------------------------
    def update(self, message: str, emoji: str = "", level: str = "info") -> None:
        """Report a status update."""
        colors = {
            "info": "cyan",
            "success": "green",
            "warning": "yellow",
            "error": "red",
            "thinking": "magenta",
        }

        color = colors.get(level, "white")
        prefix = f"{emoji} " if emoji else ""
        text = f"{prefix}{message}".strip()

        if not self.verbose and self.board.live is None:
            self.console.print(f"[{color}]{text}[/{color}]")

        self.board.add_event(text, color)

    def thinking(self, message: str = "Thinking about your request...") -> None:
        self.update(message, emoji="💭", level="thinking")

    def planning(self, message: str = "Planning the project structure...") -> None:
        self.update(message, emoji="📋", level="info")

    def creating(self, message: str = "Creating project files...") -> None:
        self.update(message, emoji="📁", level="info")

    def writing(self, filename: str) -> None:
        self.update(f"Writing {filename}...", emoji="✍️", level="info")

    def testing(self, message: str = "Running tests...") -> None:
        self.update(message, emoji="🧪", level="info")

    def fixing(self, message: str = "Fixing issues...") -> None:
        self.update(message, emoji="🔧", level="warning")

    def success(self, message: str = "All done!") -> None:
        self.update(message, emoji="✅", level="success")

    def error(self, message: str) -> None:
        self.update(message, emoji="❌", level="error")
