"""
Bordered input implementation using prompt_toolkit Application
Like Ink's Box component but for Python - with inline command list like OpenCoder
"""

from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, VSplit, Window, ConditionalContainer
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import TextArea
from prompt_toolkit.layout.controls import FormattedTextControl, BufferControl
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from typing import Optional, Dict, Any, List
import math
import shutil
import time

# Track last Esc press for double-tap detection
_last_esc_time = 0

# Command definitions (like OpenCoder's)
COMMANDS = [
    {"name": "/build", "meta": "", "description": "Start a new generation prompt"},
    {"name": "/model", "meta": "[name]", "description": "Switch the active LLM model"},
    {"name": "/clear", "meta": "", "description": "Clear the screen and chat history"},
    {"name": "/continue", "meta": "", "description": "Continue the last incomplete task"},
    {"name": "/help", "meta": "", "description": "Show all keyboard shortcuts and commands"},
    {"name": "/exit", "meta": "", "description": "Leave the NASEA shell"},
]

def create_bordered_input(
    state: Optional[Dict[str, Any]] = None,
    completer=None,
    history=None,
    commands: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
    """
    Create a sleek input box with inline command list (like OpenCoder).
    Shows commands below the input when user types '/'.
    """

    # Use provided commands or default
    command_list = commands or COMMANDS

    # Theme colors (NASEA brand)
    border_color = '#555555'  # Darker gray border (more subtle)
    accent_color = '#cc66cc'  # Soft magenta for selected commands
    text_color = '#ffffff'    # White text
    dim_color = '#666666'     # Dimmed text

    # State for command selection
    _show_commands = [False]
    _selected_index = [0]
    _filtered_commands = [command_list.copy()]

    # --- Custom Style ---
    custom_style = Style.from_dict({
        'command-selected': f'fg:{accent_color}',
        'command-normal': f'fg:{dim_color}',
        'command-desc': f'fg:{dim_color}',
        'command-desc-selected': f'fg:{accent_color}',
        'command-meta': 'fg:#888888',
    })

    # --- Dynamic Width & Height Helpers ---

    def get_term_size():
        return shutil.get_terminal_size(fallback=(80, 24))

    def get_safe_width():
        """Returns full terminal width."""
        return get_term_size().columns

    def get_top_border():
        return HTML(f'<style fg="{border_color}">' + ("─" * get_safe_width()) + '</style>')

    def get_bottom_border():
        w = get_safe_width()
        context_text = ""
        if state:
            limit = state.get("context_limit", 0)
            used = min(limit, state.get("context_used", 0))
            if limit:
                used_percent = min(100, max(0, math.ceil((used / limit) * 100)))
                remaining_percent = 100 - used_percent
                context_text = f" Context: {remaining_percent}% "

        if not context_text:
            return HTML(f'<style fg="{border_color}">' + ("─" * w) + '</style>')

        text_len = len(context_text)
        padding_left = max(0, (w - text_len) // 2)
        padding_right = max(0, w - text_len - padding_left)

        return HTML(
            f'<style fg="{border_color}">{"─" * padding_left}</style>'
            f'<style fg="{border_color}">{context_text}</style>'
            f'<style fg="{border_color}">{"─" * padding_right}</style>'
        )

    # --- Layout Components ---

    top_border = Window(
        content=FormattedTextControl(text=get_top_border),
        height=1,
        always_hide_cursor=True,
    )

    # Store original text for very long pastes
    _original_long_text = [None]
    _placeholder_text = [None]
    _previous_text = [""]
    _is_replacing = [False]

    # Create buffer
    input_buffer = Buffer(
        completer=None,  # We handle completion ourselves now
        history=history,
        complete_while_typing=False,
        multiline=True,
    )

    PROMPT_PREFIX_WIDTH = 2

    def get_input_height():
        text = input_buffer.text
        if not text:
            return 1
        term_width = get_safe_width() - PROMPT_PREFIX_WIDTH
        lines = text.split('\n')
        visual_lines = 0
        for line in lines:
            if len(line) == 0:
                visual_lines += 1
            else:
                visual_lines += max(1, (len(line) + term_width - 1) // term_width)
        return min(10, max(1, visual_lines))

    prompt_prefix = Window(
        content=FormattedTextControl(HTML('<style fg="#00ffff">&gt;</style>')),
        width=2,
        dont_extend_width=True,
    )

    text_area_window = Window(
        content=BufferControl(buffer=input_buffer),
        height=get_input_height,
        wrap_lines=True,
        dont_extend_height=True,
        cursorline=False,
    )

    class TextAreaProxy:
        @property
        def text(self):
            return input_buffer.text
        @property
        def buffer(self):
            return input_buffer

    text_area = TextAreaProxy()

    def update_command_state(buffer):
        """Update command visibility and filtering based on input."""
        if _is_replacing[0]:
            return

        text = buffer.text
        prev_text = _previous_text[0]
        _previous_text[0] = text

        # Check if we should show commands
        if text.startswith("/"):
            _show_commands[0] = True
            query = text[1:].lower()  # Remove leading /

            # Filter commands
            if query:
                _filtered_commands[0] = [
                    cmd for cmd in command_list
                    if cmd["name"][1:].lower().startswith(query)  # Match without /
                ]
            else:
                _filtered_commands[0] = command_list.copy()

            # Reset selection if out of bounds
            if _selected_index[0] >= len(_filtered_commands[0]):
                _selected_index[0] = 0
        else:
            _show_commands[0] = False
            _filtered_commands[0] = command_list.copy()
            _selected_index[0] = 0

        # Handle long paste detection
        if _placeholder_text[0] and _placeholder_text[0] in text:
            return

        if len(text) <= len(prev_text) + 50:
            return

        term_width = get_safe_width() - PROMPT_PREFIX_WIDTH
        lines = text.split('\n')
        visual_lines = 0
        for line in lines:
            if len(line) == 0:
                visual_lines += 1
            else:
                visual_lines += max(1, (len(line) + term_width - 1) // term_width)

        if visual_lines > 10:
            prefix_len = 0
            for i in range(min(len(prev_text), len(text))):
                if prev_text[i] == text[i]:
                    prefix_len += 1
                else:
                    break

            suffix_len = 0
            for i in range(1, min(len(prev_text) - prefix_len, len(text) - prefix_len) + 1):
                if prev_text[-i] == text[-i]:
                    suffix_len += 1
                else:
                    break

            prefix = text[:prefix_len]
            suffix = text[len(text) - suffix_len:] if suffix_len > 0 else ""
            pasted = text[prefix_len:len(text) - suffix_len] if suffix_len > 0 else text[prefix_len:]

            pasted_lines = pasted.split('\n')

            _original_long_text[0] = pasted
            placeholder = f"[Pasted text: {len(pasted_lines)} lines]"
            _placeholder_text[0] = placeholder

            _is_replacing[0] = True
            new_text = prefix + placeholder + suffix
            buffer.text = new_text
            buffer.cursor_position = len(prefix) + len(placeholder)
            _is_replacing[0] = False
            _previous_text[0] = new_text

    text_area.buffer.on_text_changed += update_command_state

    bottom_border = Window(
        content=FormattedTextControl(text=get_bottom_border),
        height=1,
        always_hide_cursor=True,
    )

    def get_helper_text():
        """Generate helper text - only show in normal mode."""
        from prompt_toolkit.formatted_text import FormattedText

        # Hide helper when showing commands (cleaner look)
        if _show_commands[0]:
            return FormattedText([])

        w = get_safe_width()
        left_text = "  / commands  ↑↓ history"
        right_text = "esc twice to clear"

        padding = max(1, w - len(left_text) - len(right_text))
        space_padding = " " * padding

        return HTML(
            f'<style fg="#888888">  / </style>'
            f'<style fg="#00ffff">commands</style>'
            f'<style fg="#888888">  ↑↓ </style>'
            f'<style fg="#00ffff">history</style>'
            f'<style fg="#888888">{space_padding}</style>'
            f'<style fg="#00ffff">esc</style>'
            f'<style fg="#888888"> twice to clear</style>'
        )

    helper_text = Window(
        content=FormattedTextControl(text=get_helper_text),
        height=1,
        always_hide_cursor=True,
    )

    def escape_xml(text):
        """Escape special XML characters."""
        return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))

    def get_command_list():
        """Generate the inline command list (like OpenCoder)."""
        from prompt_toolkit.formatted_text import FormattedText

        if not _show_commands[0] or not _filtered_commands[0]:
            return FormattedText([])

        commands = _filtered_commands[0]
        selected = _selected_index[0]

        # Calculate column widths
        max_name_len = max(len(cmd["name"]) for cmd in commands) + 2
        max_meta_len = max(len(cmd.get("meta", "")) for cmd in commands) + 2 if any(cmd.get("meta") for cmd in commands) else 0

        # Build formatted text fragments instead of HTML
        fragments = []
        for idx, cmd in enumerate(commands):
            is_selected = idx == selected
            name = cmd["name"]
            meta = cmd.get("meta", "")
            desc = cmd["description"]

            # Styles for prompt_toolkit
            if is_selected:
                name_style = accent_color
                desc_style = accent_color
                meta_style = f'{accent_color} italic'
            else:
                name_style = dim_color
                desc_style = dim_color
                meta_style = '#555555 italic'

            # Pad name and meta for alignment
            name_padded = name.ljust(max_name_len)
            meta_padded = meta.ljust(max_meta_len) if meta else " " * max_meta_len

            # Add fragments for this line
            fragments.append((name_style, f"  {name_padded}"))
            fragments.append((meta_style, meta_padded))
            fragments.append((desc_style, desc))

            # Add newline if not last item
            if idx < len(commands) - 1:
                fragments.append(('', '\n'))

        return FormattedText(fragments)

    def get_command_list_height():
        """Dynamic height for command list."""
        if not _show_commands[0]:
            return 0
        return len(_filtered_commands[0])

    command_list_window = ConditionalContainer(
        content=Window(
            content=FormattedTextControl(text=get_command_list),
            height=get_command_list_height,
            always_hide_cursor=True,
        ),
        filter=Condition(lambda: _show_commands[0] and len(_filtered_commands[0]) > 0),
    )

    input_row = VSplit([
        prompt_prefix,
        text_area_window,
    ])

    body_container = HSplit([
        top_border,
        input_row,
        bottom_border,
        command_list_window,  # Inline below the input
        helper_text,
    ])

    layout = Layout(body_container)

    # Key bindings
    kb = KeyBindings()

    @kb.add('up')
    def _(event):
        """Navigate up in command list or history."""
        if _show_commands[0] and _filtered_commands[0]:
            _selected_index[0] = (_selected_index[0] - 1) % len(_filtered_commands[0])
            event.app.invalidate()
        else:
            # History navigation
            event.current_buffer.auto_up()

    @kb.add('down')
    def _(event):
        """Navigate down in command list or history."""
        if _show_commands[0] and _filtered_commands[0]:
            _selected_index[0] = (_selected_index[0] + 1) % len(_filtered_commands[0])
            event.app.invalidate()
        else:
            # History navigation
            event.current_buffer.auto_down()

    @kb.add('tab')
    def _(event):
        """Tab to complete selected command."""
        if _show_commands[0] and _filtered_commands[0]:
            cmd = _filtered_commands[0][_selected_index[0]]
            completion = cmd["name"] + (" " if not cmd.get("meta") else " ")
            input_buffer.text = completion
            input_buffer.cursor_position = len(completion)
            _show_commands[0] = False
            event.app.invalidate()

    @kb.add('enter', eager=True)
    def _(event):
        """Submit input or select command."""
        # If showing commands and one is selected, complete it first
        if _show_commands[0] and _filtered_commands[0]:
            cmd = _filtered_commands[0][_selected_index[0]]
            # If command has no meta (no args needed), submit directly
            if not cmd.get("meta"):
                text = cmd["name"]
                if history is not None and text.strip():
                    history.append_string(text)
                event.app.exit(result=text)
                return
            else:
                # Complete the command and wait for args
                completion = cmd["name"] + " "
                input_buffer.text = completion
                input_buffer.cursor_position = len(completion)
                _show_commands[0] = False
                event.app.invalidate()
                return

        # Normal submission
        if _original_long_text[0] is not None and _placeholder_text[0] is not None:
            current = text_area.text
            placeholder = _placeholder_text[0]

            if placeholder in current:
                idx = current.find(placeholder)
                before_added = current[:idx]
                after_added = current[idx + len(placeholder):]
                text = before_added + _original_long_text[0] + after_added
            else:
                text = current

            _original_long_text[0] = None
            _placeholder_text[0] = None
        else:
            text = text_area.text

        if history is not None and text.strip():
            history.append_string(text)
        event.app.exit(result=text)

    @kb.add('c-c')
    def _(event):
        """Cancel input."""
        event.app.exit(result=None)

    @kb.add('escape', eager=True)
    def _(event):
        """Escape to close commands or double-tap to clear."""
        global _last_esc_time

        if _show_commands[0]:
            # First escape closes command list
            _show_commands[0] = False
            input_buffer.text = ""
            input_buffer.cursor_position = 0
            event.app.invalidate()
            _last_esc_time = 0
            return

        now = time.time()
        if now - _last_esc_time < 2.0:
            text_area.buffer.reset()
            _original_long_text[0] = None
            _placeholder_text[0] = None
            _previous_text[0] = ""
            _last_esc_time = 0
            event.app.invalidate()
        else:
            _last_esc_time = now

    app = Application(
        layout=layout,
        key_bindings=kb,
        full_screen=False,
        mouse_support=False,
        refresh_interval=None,
        style=custom_style,
    )

    def calc_visual_lines(text):
        """Calculate how many visual lines the text takes."""
        if not text:
            return 1
        term_width = get_safe_width() - PROMPT_PREFIX_WIDTH
        lines = text.split('\n')
        visual = 0
        for line in lines:
            if len(line) == 0:
                visual += 1
            else:
                visual += max(1, (len(line) + term_width - 1) // term_width)
        return min(10, max(1, visual))

    try:
        result = app.run()

        # Cleanup
        import sys
        input_lines = calc_visual_lines(text_area.text)
        cmd_lines = len(_filtered_commands[0]) if _show_commands[0] else 0
        total_lines = 1 + input_lines + 1 + cmd_lines + 1  # top + input + bottom + commands + helper
        sys.stdout.write(f"\033[{total_lines}A")
        sys.stdout.write("\033[J")
        sys.stdout.flush()

        return result

    except (KeyboardInterrupt, EOFError):
        import sys
        input_lines = calc_visual_lines(text_area.text)
        cmd_lines = len(_filtered_commands[0]) if _show_commands[0] else 0
        total_lines = 1 + input_lines + 1 + cmd_lines + 1
        sys.stdout.write(f"\033[{total_lines}A")
        sys.stdout.write("\033[J")
        sys.stdout.flush()
        return None
