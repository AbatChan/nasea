"""
Real-time streaming output parser and renderer.

Parses AI output as it streams and renders it beautifully in the terminal
with progress indicators, checkboxes, and stage markers.
"""

import re
from typing import Optional, List, Dict, Callable
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.markdown import Markdown
from rich.columns import Columns
from rich.tree import Tree
from rich.theme import Theme

MARKDOWN_THEME = Theme({
    "markdown.paragraph": "none",
    "markdown.em": "italic",
    "markdown.strong": "bold",
    "markdown.code": "dim",
    "markdown.s": "strike dim",
    "markdown.h1": "bold bright_white",
    "markdown.h1.border": "bright_black",
    "markdown.h2": "bold cyan",
    "markdown.h3": "bold magenta",
    "markdown.h4": "bold green",
    "markdown.h5": "bold yellow",
    "markdown.h6": "bold blue",
    "markdown.link": "underline cyan",
    "markdown.link_url": "cyan",
    "markdown.block_quote": "italic dim",
    "markdown.code_block": "dim",
    "markdown.item": "none",
    "markdown.item.bullet": "cyan",
    "markdown.item.number": "cyan",
    "markdown.hr": "bright_black",
})


class Stage:
    """Represents a workflow stage (Plan, Implementation, Testing, etc.)"""

    def __init__(self, name: str, emoji: str = "📋"):
        self.name = name
        self.emoji = emoji
        self.tasks: List[Dict[str, any]] = []
        self.status = "in_progress"  # in_progress, completed, failed
        self.output = ""

    def add_task(self, task_name: str):
        """Add a task to this stage"""
        self.tasks.append({
            "name": task_name,
            "status": "pending",  # pending, in_progress, completed, failed
        })

    def mark_task_complete(self, task_index: int):
        """Mark a task as completed"""
        if 0 <= task_index < len(self.tasks):
            self.tasks[task_index]["status"] = "completed"

    def mark_task_in_progress(self, task_index: int):
        """Mark a task as in progress"""
        if 0 <= task_index < len(self.tasks):
            self.tasks[task_index]["status"] = "in_progress"

    def complete(self):
        """Mark this stage as completed"""
        self.status = "completed"
        for task in self.tasks:
            if task["status"] != "completed":
                task["status"] = "completed"


class StreamingRenderer:
    """
    Renders streaming AI output with real-time updates.

    Detects patterns like:
    - ## Stage Name (creates new stage)
    - [ ] Task name (adds task)
    - [x] Task name (marks task complete)
    - ✅ Completed... (marks task complete)
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        if not getattr(self.console, "_nasea_markdown_theme", False):
            self.console.push_theme(MARKDOWN_THEME)
            setattr(self.console, "_nasea_markdown_theme", True)
        self.stages: List[Stage] = []
        self.current_stage: Optional[Stage] = None
        self.buffer = ""
        self.full_output = ""

        # Callbacks for when files are created/updated
        self.on_file_created: Optional[Callable] = None
        self.on_file_updated: Optional[Callable] = None

    def process_chunk(self, chunk: str) -> None:
        """
        Process a chunk of streamed text.

        Args:
            chunk: New text from the stream
        """
        self.buffer += chunk
        self.full_output += chunk

        # Check for stage markers
        self._detect_stages()

        # Check for task markers
        self._detect_tasks()

        # Check for completion markers
        self._detect_completions()

        # Check for file operations
        self._detect_file_operations()

    def _detect_stages(self):
        """Detect stage markers like ## Plan, ## Implementation"""
        stage_patterns = [
            (r"##\s*(Plan|Planning)", "📋", "Planning"),
            (r"##\s*(Implementat|Creating|Development)", "⚙️", "Implementation"),
            (r"##\s*(Test|Verification|Quality)", "🧪", "Testing"),
            (r"##\s*(Fix|Debug|Refactor)", "🔧", "Fixing"),
            (r"##\s*(Final|Complete|Done)", "✨", "Finalization"),
        ]

        for pattern, emoji, stage_name in stage_patterns:
            if re.search(pattern, self.buffer, re.IGNORECASE):
                # Complete previous stage
                if self.current_stage:
                    self.current_stage.complete()

                # Create new stage
                stage = Stage(stage_name, emoji)
                self.stages.append(stage)
                self.current_stage = stage

                # Clear buffer after the marker
                self.buffer = re.sub(pattern, "", self.buffer, flags=re.IGNORECASE)

    def _detect_tasks(self):
        """Detect task markers like [ ] or - Task name"""
        task_patterns = [
            r"\[\s*\]\s*(.+?)(?:\n|$)",  # [ ] Task name
            r"[-•]\s*(.+?)(?:\n|$)",      # - Task name or • Task name
        ]

        for pattern in task_patterns:
            matches = re.finditer(pattern, self.buffer)
            for match in matches:
                task_name = match.group(1).strip()
                if self.current_stage and task_name:
                    # Check if task already exists
                    existing = any(t["name"] == task_name for t in self.current_stage.tasks)
                    if not existing:
                        self.current_stage.add_task(task_name)

    def _detect_completions(self):
        """Detect completion markers like [x] or ✅"""
        completion_patterns = [
            r"\[x\]\s*(.+?)(?:\n|$)",     # [x] Task name
            r"✅\s*(.+?)(?:\n|$)",         # ✅ Task name
            r"✓\s*(.+?)(?:\n|$)",          # ✓ Task name
        ]

        for pattern in completion_patterns:
            matches = re.finditer(pattern, self.buffer, re.IGNORECASE)
            for match in matches:
                task_name = match.group(1).strip()
                if self.current_stage:
                    # Find and mark task as complete
                    for i, task in enumerate(self.current_stage.tasks):
                        if task_name.lower() in task["name"].lower():
                            self.current_stage.mark_task_complete(i)
                            break

    def _detect_file_operations(self):
        """Detect file creation/update operations"""
        file_patterns = [
            r"Created?\s+file[:\s]+([^\s\n]+)",
            r"Writing\s+to\s+([^\s\n]+)",
            r"Generated\s+([^\s\n]+\.py)",
        ]

        for pattern in file_patterns:
            matches = re.finditer(pattern, self.buffer, re.IGNORECASE)
            for match in matches:
                file_path = match.group(1).strip()
                if self.on_file_created:
                    self.on_file_created(file_path)

    def render(self) -> Panel:
        """Render current state as a Rich panel."""
        tree = Tree("🚀 [bold cyan]Project Generation")

        for stage in self.stages:
            # Stage icon and status
            if stage.status == "completed":
                stage_icon = "✅"
                stage_style = "green"
            elif stage.status == "in_progress":
                stage_icon = "⚙️"
                stage_style = "cyan"
            else:
                stage_icon = stage.emoji
                stage_style = "white"

            stage_node = tree.add(f"{stage_icon} [bold {stage_style}]{stage.name}[/]")

            # Add tasks
            for task in stage.tasks:
                if task["status"] == "completed":
                    task_icon = "✅"
                    task_style = "green"
                elif task["status"] == "in_progress":
                    task_icon = "⚙️"
                    task_style = "yellow"
                else:
                    task_icon = "⬜"
                    task_style = "dim"

                stage_node.add(f"{task_icon} [{task_style}]{task['name']}[/]")

        progress_panel = Panel(
            tree,
            title="[bold]Progress[/]",
            border_style="cyan",
            padding=(1, 2),
        )

        tail_chars = 4000
        md_window = self.full_output[-tail_chars:]
        if md_window:
            md_window = md_window.replace("\r\n", "\n").replace("\r", "\n")
            md_window = re.sub(r"(?<!\n)\n(\s*[-*+]\s)", r"\n\n\1", md_window)
            md_window = re.sub(r"(?<!\n)\n(\s*\d+\.\s)", r"\n\n\1", md_window)
            md_window = re.sub(r"^\s+(?=#+\s)", "", md_window, flags=re.MULTILINE)
            md_window = re.sub(r"^\s+(?=[-*+]\s)", "", md_window, flags=re.MULTILINE)
            md_window = re.sub(r"^\s+(?=\d+\.\s)", "", md_window, flags=re.MULTILINE)
        md_text = md_window.strip() if md_window else ""
        md_text = md_text or "_(waiting for output...)_"
        markdown_panel = Panel(
            Markdown(md_text, style="markdown"),
            title="[bold]Live Output[/]",
            border_style="magenta",
            padding=(1, 2),
        )

        content = Columns([progress_panel, markdown_panel], equal=True, expand=True)

        return Panel(content, border_style="cyan")

    def get_summary(self) -> str:
        """Get a text summary of progress"""
        total_tasks = sum(len(s.tasks) for s in self.stages)
        completed_tasks = sum(
            len([t for t in s.tasks if t["status"] == "completed"])
            for s in self.stages
        )

        return f"Progress: {completed_tasks}/{total_tasks} tasks completed"


def stream_with_renderer(
    stream_generator,
    console: Optional[Console] = None,
    title: str = "Generating Project"
) -> str:
    """
    Display a streaming LLM response with live rendering.

    Args:
        stream_generator: Generator yielding text chunks
        console: Rich console (optional)
        title: Title for the display

    Returns:
        Full accumulated text
    """
    console = console or Console()
    renderer = StreamingRenderer(console)

    with Live(renderer.render(), console=console, refresh_per_second=4) as live:
        bytes_since_update = 0
        for chunk in stream_generator:
            renderer.process_chunk(chunk)
            bytes_since_update += len(chunk)
            if "\n" in chunk or bytes_since_update > 200:
                live.update(renderer.render())
                bytes_since_update = 0
        live.update(renderer.render())

    # Show final summary
    console.print(f"\n{renderer.get_summary()}")

    return renderer.full_output
