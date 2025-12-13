#!/usr/bin/env python3
"""Demo different spinner animations."""

from rich.console import Console
from rich.spinner import Spinner
from rich.live import Live
import time

console = Console()

spinners = [
    ("arc", "◜◠◝◞◡ - Arc/curved rotation (current)"),
    ("dots", "⠋⠙⠹⠸⠼ - Classic rotating dots"),
    ("dots3", "⠋⠙⠚⠞⠖ - Rotating dots variant"),
    ("dots12", "⢀⣀⠄⠂⠁ - Smooth dots"),
    ("line", "- \\ | / - Classic line spinner"),
    ("bouncingBall", "( ●    ) - Bouncing ball"),
    ("aesthetic", "▰▱ - Clean aesthetic bars"),
]

console.print("\n[bold cyan]🎨 Available Spinner Animations[/bold cyan]\n")

for name, desc in spinners:
    console.print(f"[cyan]{name}[/cyan]: {desc}")

    with Live(Spinner(name, style="cyan"), console=console, refresh_per_second=12):
        time.sleep(1.5)

    console.print()

console.print("\n[bold green]✅ Current spinner: arc[/bold green]")
console.print("[dim]To change, edit nasea/cli.py line 56[/dim]\n")
