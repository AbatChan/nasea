"""
Test bordered input using prompt_toolkit properly
Based on how Ink does it (opencoder implementation)
"""

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.layout import Container, HSplit, Window, FormattedTextControl
from prompt_toolkit.widgets import Frame, TextArea, Box as PTBox
from rich.console import Console
import sys

console = Console()

def test_simple_bordered_prompt():
    """Test 1: Simple approach with manual border drawing"""
    print("\n" + "="*80)
    print("TEST 1: Manual border with prompt")
    print("="*80 + "\n")

    border_width = 80

    # Draw top border
    console.print("╭" + "─" * (border_width - 2) + "╮", style="dim cyan")

    # Create session and prompt with custom left margin
    session = PromptSession()

    try:
        # Print the left border and prompt on same line
        sys.stdout.write("\033[36m│\033[0m ")  # │ in cyan
        sys.stdout.flush()

        user_input = session.prompt(
            HTML('<ansicyan>></ansicyan> '),
            multiline=False,
        )

        # After input, add right border
        # Move cursor to end of line and print right border
        console.print(f"[dim cyan]│[/dim cyan]")

        # Print empty line with borders
        console.print("[dim cyan]│[/dim cyan]" + " " * (border_width - 4) + "[dim cyan]│[/dim cyan]")

        # Bottom border
        console.print("╰" + "─" * 30 + " Context: 0% " + "─" * 30 + "╯", style="dim cyan")

        print(f"\nYou entered: {user_input}")

    except KeyboardInterrupt:
        print("\nCancelled")
        console.print("╰" + "─" * (border_width - 2) + "╯", style="dim cyan")


def test_multiline_bordered():
    """Test 2: Multiline with borders (closer to what we want)"""
    print("\n" + "="*80)
    print("TEST 2: Multiline bordered input")
    print("="*80 + "\n")

    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout.containers import Window, HSplit, VSplit
    from prompt_toolkit.layout.layout import Layout
    from prompt_toolkit.widgets import TextArea
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.formatted_text import HTML

    # Create text area
    text_area = TextArea(
        height=1,
        multiline=False,
        wrap_lines=False,
        prompt=HTML('<ansicyan>> </ansicyan>'),
    )

    # Create border lines
    top_border = Window(
        content=FormattedTextControl(
            text=HTML('<dim-cyan>╭' + '─' * 78 + '╮</dim-cyan>')
        ),
        height=1,
    )

    bottom_border = Window(
        content=FormattedTextControl(
            text=HTML('<dim-cyan>╰' + '─' * 30 + ' Context: 0% ' + '─' * 30 + '╯</dim-cyan>')
        ),
        height=1,
    )

    # Create left/right borders for input line
    left_border = Window(content=FormattedTextControl(text=HTML('<dim-cyan>│</dim-cyan>')), width=1)
    right_border = Window(content=FormattedTextControl(text=HTML('<dim-cyan>│</dim-cyan>')), width=1)

    # Combine with borders
    input_with_borders = VSplit([
        left_border,
        text_area,
        right_border,
    ])

    # Create full layout
    root_container = HSplit([
        top_border,
        input_with_borders,
        bottom_border,
    ])

    layout = Layout(root_container)

    # Key bindings
    kb = KeyBindings()

    @kb.add('enter')
    def _(event):
        event.app.exit(result=text_area.text)

    @kb.add('c-c')
    def _(event):
        event.app.exit(result=None)

    # Create application
    app = Application(
        layout=layout,
        key_bindings=kb,
        full_screen=False,
    )

    # Run
    try:
        result = app.run()
        if result:
            print(f"\nYou entered: {result}")
    except KeyboardInterrupt:
        print("\nCancelled")


if __name__ == "__main__":
    print("Testing bordered input implementations")
    print("="*80)

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "2":
        test_multiline_bordered()
    else:
        print("\nTest 1: Simple bordered prompt (default)")
        print("Run with argument '2' for Test 2")
        print()
        test_simple_bordered_prompt()
