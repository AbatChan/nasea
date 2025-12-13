"""
Enhanced CLI Input Display Module
Provides bordered input prompts with helper text, similar to opencoder.
"""

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from typing import Optional, Dict, Any
import math


def create_bordered_input_prompt(
    state: Optional[Dict[str, Any]] = None,
    show_context: bool = True,
    show_commands: bool = True
) -> None:
    """
    Display a beautifully bordered input prompt with helper text.

    Args:
        state: Application state dict containing context info
        show_context: Whether to show context usage in the title
        show_commands: Whether to show command hints
    """
    console = Console()

    # Create the main input indicator
    input_line = Text()
    input_line.append(">", style="bold dark_cyan")
    input_line.append(" Type your message here...", style="dim italic")

    # Create helper text
    helpers = []

    if show_commands:
        cmd_text = Text()
        cmd_text.append("  ", style="dim")
        cmd_text.append("/", style="bold dark_cyan")
        cmd_text.append(" commands", style="dim")
        helpers.append(cmd_text)

    history_text = Text()
    history_text.append("  ", style="dim")
    history_text.append("↑↓", style="bold dark_cyan")
    history_text.append(" history", style="dim")
    helpers.append(history_text)

    # Combine helpers on one line
    helper_line = Text()
    for helper in helpers:
        helper_line.append_text(helper)

    # Build content
    content = Group(
        input_line,
        Text(""),  # Empty line
        helper_line
    )

    # Build title with context info
    title = "[bold dark_cyan]NASEA[/bold dark_cyan]"
    subtitle = None

    if state and show_context:
        limit = state.get("context_limit", 0)
        used = min(limit, state.get("context_used", 0))
        if limit:
            percent = min(100, max(0, math.ceil((used / limit) * 100)))
            subtitle = f"[dim]Context: {percent}% ({used:,}/{limit:,} tokens)[/dim]"

    # Create the panel
    panel = Panel(
        content,
        border_style="dark_cyan",
        padding=(1, 2),
        title=title,
        title_align="left",
        subtitle=subtitle,
        subtitle_align="right",
        expand=False
    )

    console.print(panel)
    console.print()


def create_compact_input_border(console: Console, state: Optional[Dict[str, Any]] = None) -> None:
    """
    Create a compact ASCII border around the input area.
    Similar to opencoder's minimalist style.

    Args:
        console: Rich Console instance
        state: Application state dict
    """
    # Top border (simple, no context)
    border_width = 80
    top_border = "╭" + "─" * (border_width - 2) + "╮"

    console.print(top_border, style="dim cyan")

    # Input line with prompt
    console.print("│ ", style="dim cyan", end="")
    console.print("[cyan]❯[/cyan] ", end="")

    # This is where the actual input will appear
    # The caller will handle the prompt() call


def create_input_footer(console: Console, state: Optional[Dict[str, Any]] = None) -> None:
    """
    Create the bottom border and helper text for the input area.

    Args:
        console: Rich Console instance
        state: Application state dict for context info
    """
    border_width = 80
    console.print("│ " + " " * (border_width - 4) + " │", style="dim cyan")

    # Build bottom border with context if available
    footer_text = ""
    if state:
        limit = state.get("context_limit", 0)
        used = min(limit, state.get("context_used", 0))
        if limit:
            percent = min(100, max(0, math.ceil((used / limit) * 100)))
            footer_text = f" Context: {percent}% "

    # Calculate padding for centered footer
    if footer_text:
        padding_left = (border_width - len(footer_text) - 2) // 2
        padding_right = border_width - len(footer_text) - padding_left - 2
        bottom_border = "╰" + "─" * padding_left + footer_text + "─" * padding_right + "╯"
    else:
        bottom_border = "╰" + "─" * (border_width - 2) + "╯"

    console.print(bottom_border, style="dim cyan")

    # Helper text below the border
    helper = Text()
    helper.append("  ", style="dim")
    helper.append("/", style="cyan")
    helper.append(" commands  ", style="dim")
    helper.append("↑↓", style="cyan")
    helper.append(" history  ", style="dim")
    helper.append("Tab", style="cyan")

    console.print(helper)
    console.print()


def show_welcome_panel(
    version: str,
    model: str,
    output_dir: str,
    mock_enabled: bool = False
) -> None:
    """
    Show a beautiful welcome panel when starting NASEA.

    Args:
        version: NASEA version string
        model: Current LLM model name
        output_dir: Output directory path
        mock_enabled: Whether mock mode is enabled
    """
    console = Console()

    # Create welcome content
    title = Text()
    title.append("NASEA", style="bold dark_cyan")
    title.append(" ", style="white")
    title.append(f"v{version}", style="dim")

    subtitle = Text()
    subtitle.append("Natural-Language Autonomous Software-Engineering Agent", style="italic dim")

    # Info table
    info_table = Table.grid(padding=(0, 2))
    info_table.add_column(style="dark_cyan", justify="right")
    info_table.add_column(style="white")

    info_table.add_row("Model:", model)
    info_table.add_row("Output:", output_dir)
    if mock_enabled:
        info_table.add_row("Mode:", "[yellow]MOCK (Testing)[/yellow]")

    # Commands hint
    commands = Text()
    commands.append("\n💡 ", style="yellow")
    commands.append("Type ", style="dim")
    commands.append("/help", style="bold dark_cyan")
    commands.append(" to see available commands", style="dim")

    # Combine content
    content = Group(
        title,
        subtitle,
        Text(""),
        info_table,
        commands
    )

    # Create panel
    panel = Panel(
        content,
        border_style="dark_cyan",
        padding=(1, 2),
        title="[bold]🚀 Welcome[/bold]",
        title_align="left"
    )

    console.print()
    console.print(panel)
    console.print()


def show_command_hint_panel() -> None:
    """Show a helpful panel explaining available commands."""
    console = Console()

    # Create command hints
    commands = Table.grid(padding=(0, 2))
    commands.add_column(style="dark_cyan bold", justify="right")
    commands.add_column(style="dim")

    commands.add_row("/help", "Show all commands")
    commands.add_row("/model", "Switch LLM model")
    commands.add_row("/clear", "Clear conversation history")
    commands.add_row("/thinking", "Toggle thinking mode")
    commands.add_row("/exit", "Exit NASEA")

    panel = Panel(
        commands,
        border_style="dark_cyan",
        padding=(1, 2),
        title="[bold dark_cyan]Quick Commands[/bold dark_cyan]",
        title_align="left"
    )

    console.print(panel)
    console.print()


# Example usage demonstration
if __name__ == "__main__":
    import time

    console = Console()

    # Show welcome
    show_welcome_panel(
        version="0.1.0-alpha",
        model="gpt-4",
        output_dir="./output",
        mock_enabled=False
    )

    time.sleep(1)

    # Show command hints
    show_command_hint_panel()

    time.sleep(1)

    # Show input prompt examples
    console.print("[bold]Example 1: Full bordered input with context[/bold]")
    state = {"context_limit": 100000, "context_used": 45000}
    create_bordered_input_prompt(state, show_context=True, show_commands=True)

    time.sleep(1)

    console.print("[bold]Example 2: Compact ASCII border style[/bold]")
    create_compact_input_border(console, state)
    console.print("This is where prompt() would capture input...")
    create_input_footer(console, state)
