"""
Dynamic loader with rotating messages and cancellation hints.
Cross-platform compatible (Windows + Unix).
"""

import time
import itertools
import threading
import sys
import platform
from typing import List, Optional
from rich.console import Console

# Check if we're on Windows (may need ASCII fallback for spinners)
IS_WINDOWS = platform.system() == "Windows"
from rich.text import Text
from rich.live import Live

# Unix-only modules (not available on Windows)
try:
    import termios
    import tty
    import select
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False
    termios = None
    tty = None
    select = None


class DynamicLoader:
    """
    Dynamic loader that shows rotating messages with cancellation hints.

    Example:
        loader = DynamicLoader(["Working...", "Almost there..."], style="cyan")
        loader.start()
        # Do work...
        loader.stop()
    """

    def __init__(
        self,
        messages: List[str],
        style: str = "cyan",
        spinner: str = "dots",
        show_cancel_hint: bool = True,
        console: Optional[Console] = None,
        min_message_seconds: float = 5.0
    ):
        """
        Args:
            messages: List of messages to rotate through
            style: Rich style for the loader (cyan, magenta, yellow, etc.)
            spinner: Spinner type (dots, dots2, arc, etc.)
            show_cancel_hint: Show "Ctrl+C to cancel" hint
            console: Rich console (creates new one if not provided)
        """
        self.messages = messages
        self.style = style
        self.spinner = spinner
        self.show_cancel_hint = show_cancel_hint
        self.console = console or Console()
        self.min_message_seconds = max(0.1, min_message_seconds)

        # Spinner frames - use ASCII on Windows for compatibility
        if IS_WINDOWS:
            self.spinner_frames = {
                "dots": ["-", "\\", "|", "/"],
                "dots2": ["-", "\\", "|", "/"],
                "arc": ["-", "\\", "|", "/"],
                "line": ["-", "\\", "|", "/"],
                "triangle": ["-", "\\", "|", "/"],
                "star": ["*", "+", "*", "+"],
            }
        else:
            self.spinner_frames = {
                "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
                "dots2": ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"],
                "arc": ["◜", "◠", "◝", "◞", "◡", "◟"],
                "line": ["-", "\\", "|", "/"],
                "triangle": ["◢", "◣", "◤", "◥"],
                "star": ["✶", "✷", "✸", "✹", "✺", "✹", "✸", "✷"],
            }

        self._stop_event = threading.Event()
        self._interrupted = threading.Event()  # Set when Esc pressed
        self._live = None
        self._message_index = 0
        self._frame_count = 0
        self._last_switch_time = 0.0
        self._start_time = 0.0
        self._old_terminal_settings = None
        self._current_message = messages[0] if messages else "Working..."
        self._message_lock = threading.Lock()

    def _generate_text(self, frame: str, message: str) -> Text:
        """Generate Rich Text for the loader."""
        text = Text()
        elapsed = max(0.0, time.perf_counter() - self._start_time)
        text.append(f"{frame} ", style=self.style)
        text.append(f"{message} ", style=self.style)
        text.append(f"({elapsed:0.1f}s)", style="dim")

        if self.show_cancel_hint:
            text.append("  ", style="dim")
            text.append("(esc to interrupt)", style="dim")

        return text

    def start(self):
        """Start the loader animation."""
        if self._live is None:
            # On Windows, use simple print mode (Rich Live can freeze)
            if IS_WINDOWS:
                self._start_time = time.perf_counter()
                self._stop_event.clear()
                self._interrupted.clear()
                # Print with \r so it can be overwritten, no newline
                print(f"\r* {self.messages[0]}...", end="", flush=True)
                self._windows_printed = True
                return

            # Disable terminal echo to prevent Enter presses from showing (Unix only)
            if HAS_TERMIOS:
                try:
                    if sys.stdin.isatty():
                        self._old_terminal_settings = termios.tcgetattr(sys.stdin)
                        new_settings = termios.tcgetattr(sys.stdin)
                        # Disable ECHO flag to prevent keypresses from appearing
                        new_settings[3] = new_settings[3] & ~termios.ECHO
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new_settings)
                except (AttributeError, OSError):
                    pass

            # Start with first message and frame
            frames = self.spinner_frames.get(self.spinner, self.spinner_frames["dots"])
            initial_text = self._generate_text(frames[0], self.messages[0])

            # Create Live display
            self._live = Live(
                initial_text,
                console=self.console,
                refresh_per_second=10,  # 10 FPS = smooth
                transient=True  # Disappears when stopped
            )
            self._live.start()

            # Start update thread
            self._stop_event.clear()
            self._interrupted.clear()
            self._frame_count = 0
            self._message_index = 0
            self._last_switch_time = time.perf_counter()
            self._start_time = self._last_switch_time
            threading.Thread(target=self._update_loop, daemon=True).start()
            # Start keyboard listener for Esc (Unix only)
            if HAS_TERMIOS:
                threading.Thread(target=self._keyboard_listener, daemon=True).start()

    def _keyboard_listener(self):
        """Listen for Esc key press in background (Unix only)."""
        if not HAS_TERMIOS:
            return

        try:
            if not sys.stdin.isatty():
                return

            # Set terminal to raw mode for keypress detection
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setraw(sys.stdin.fileno())

                while not self._stop_event.is_set():
                    # Check if there's input available (non-blocking)
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        char = sys.stdin.read(1)
                        if char == '\x1b':  # Could be Esc or start of escape sequence
                            # Wait longer to catch full escape sequences (arrow keys, etc)
                            sequence = char
                            for _ in range(5):
                                if select.select([sys.stdin], [], [], 0.02)[0]:
                                    sequence += sys.stdin.read(1)
                                else:
                                    break

                            # If we got more than just ESC, it's an escape sequence - ignore
                            if len(sequence) > 1:
                                continue

                            # Bare Esc key - interrupt!
                            self._interrupted.set()
                            self._stop_event.set()
                            if self._live:
                                self._live.stop()
                                self._live = None
                            # Restore terminal before printing
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                            termios.tcflush(sys.stdin, termios.TCIFLUSH)
                            self.console.print("\r\033[K", end="", highlight=False)
                            self.console.print("  [yellow]⎿[/yellow]  [red]Interrupted · What should Nasea do instead?[/red]\n")
                            return
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except (AttributeError, OSError):
            pass

    @property
    def was_interrupted(self) -> bool:
        """Check if loader was interrupted by Esc."""
        return self._interrupted.is_set()

    def update_message(self, message: str):
        """Update the loader message dynamically."""
        with self._message_lock:
            self._current_message = message

    def _update_loop(self):
        """Update loop running in background thread."""
        frames = self.spinner_frames.get(self.spinner, self.spinner_frames["dots"])
        frame_cycle = itertools.cycle(frames)
        long_wait_index = 0
        last_long_wait_time = 0.0

        while not self._stop_event.is_set() and not self._interrupted.is_set():
            frame = next(frame_cycle)
            elapsed_total = time.perf_counter() - self._start_time

            with self._message_lock:
                current_message = self._current_message

            if current_message == self.messages[0]:
                self._frame_count += 1
                if self._frame_count >= 50 and self._message_index < len(self.messages) - 1:
                    elapsed = time.perf_counter() - self._last_switch_time
                    if elapsed >= self.min_message_seconds:
                        self._message_index += 1
                        self._frame_count = 0
                        self._last_switch_time = time.perf_counter()
                        current_message = self.messages[self._message_index]
                        with self._message_lock:
                            self._current_message = current_message

            if elapsed_total >= 20.0:
                time_since_last = elapsed_total - last_long_wait_time
                if last_long_wait_time == 0.0 or time_since_last >= 10.0:
                    if long_wait_index < len(LONG_WAIT_MESSAGES):
                        current_message = LONG_WAIT_MESSAGES[long_wait_index]
                        with self._message_lock:
                            self._current_message = current_message
                        long_wait_index += 1
                        last_long_wait_time = elapsed_total

            text = self._generate_text(frame, current_message)
            if self._live:
                self._live.update(text)

            time.sleep(0.1)

    def stop(self):
        """Stop the loader animation."""
        self._stop_event.set()

        # On Windows, clear the line we printed
        if IS_WINDOWS:
            if getattr(self, '_windows_printed', False):
                # Clear the line with spaces and return cursor
                print("\r" + " " * 50 + "\r", end="", flush=True)
                self._windows_printed = False
            return

        time.sleep(0.15)

        if self._live:
            self._live.stop()
            self._live = None
            self.console.print("\r\033[K", end="", highlight=False)

        self._drain_input_buffer()

        # Restore terminal echo (Unix only)
        if HAS_TERMIOS and self._old_terminal_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_terminal_settings)
                self._old_terminal_settings = None
            except (AttributeError, OSError):
                pass

    def _drain_input_buffer(self):
        """Drain any pending input from stdin buffer."""
        if not HAS_TERMIOS:
            return
        try:
            if sys.stdin.isatty():
                while select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.read(1)
        except (AttributeError, OSError):
            pass

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


# Time-based encouraging messages
LONG_WAIT_MESSAGES = [
    "Still working...",
    "This is taking a bit...",
    "Hang tight...",
    "Almost there...",
    "That's a lot of code!",
    "Bear with me...",
    "Still on it...",
    "Making progress...",
]

# Predefined loader configurations
LOADER_CONFIGS = {
    "chat": {"messages": ["Thinking...", "Processing...", "Pondering..."], "style": "cyan", "spinner": "star"},
    "tool": {"messages": ["Working...", "Executing...", "Processing..."], "style": "cyan", "spinner": "dots"},
    "censor": {"messages": ["Working...", "Routing to uncensored model...", "Processing..."], "style": "magenta", "spinner": "dots"},
    "create": {"messages": ["Working...", "Creating project...", "Setting up files...", "Building..."], "style": "green", "spinner": "dots"},
    "edit": {"messages": ["Working...", "Analyzing code...", "Applying changes...", "Refactoring..."], "style": "yellow", "spinner": "dots"},
    "debug": {"messages": ["Working...", "Investigating issue...", "Analyzing errors...", "Finding fix..."], "style": "red", "spinner": "dots"},
    "test": {"messages": ["Working...", "Running tests...", "Checking results...", "Validating..."], "style": "blue", "spinner": "dots"},
    "view": {"messages": ["Working...", "Reading files...", "Inspecting code...", "Gathering info..."], "style": "cyan", "spinner": "dots"},
    "explain": {"messages": ["Working...", "Analyzing code...", "Understanding logic...", "Explaining..."], "style": "magenta", "spinner": "dots"},
    "run": {"messages": ["Working...", "Starting execution...", "Running project...", "Processing..."], "style": "green", "spinner": "dots"}
}


def get_loader(loader_type: str, console: Optional[Console] = None) -> DynamicLoader:
    """Get a pre-configured dynamic loader."""
    config = LOADER_CONFIGS.get(loader_type, LOADER_CONFIGS["chat"])
    return DynamicLoader(
        messages=config["messages"],
        style=config["style"],
        spinner=config["spinner"],
        console=console
    )
