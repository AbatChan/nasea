"""
NASEA Command-Line Interface
Main entry point for user interaction.
"""

import sys
import re
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import math
import datetime

import typer

try:  # pragma: no cover - optional dependency
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.validation import Validator, ValidationError
    HAVE_PROMPT_TOOLKIT = True
except ImportError:  # Fallback to basic mode only
    PromptSession = None  # type: ignore
    HAVE_PROMPT_TOOLKIT = False

    class Completer:  # type: ignore
        pass

    class Completion:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    def HTML(text: str) -> str:  # type: ignore
        return text

    class Validator:  # type: ignore
        pass

    class ValidationError(Exception):  # type: ignore
        pass
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.markdown import Markdown
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from loguru import logger

from nasea import __version__
from nasea.core.control_unit import ControlUnit
from nasea.core.config import config
from nasea.core.status import StatusReporter
from nasea.core.intent import IntentClassifier, Intent
from nasea.llm.llm_factory import LLMFactory
from nasea.cli_enhanced_input import create_input_footer
from nasea.core.bordered_input import create_bordered_input
from nasea.prompts import load_system_prompt

CONTROL_TOKEN_CREATE = "[NASEA_CREATE]"
CONTROL_TOKEN_EDIT = "[NASEA_EDIT]"
CONTROL_TOKEN_DEBUG = "[NASEA_DEBUG]"
CONTROL_TOKEN_TEST = "[NASEA_TEST]"
CONTROL_TOKEN_LIST = "[NASEA_LIST]"
CONTROL_TOKEN_VIEW = "[NASEA_VIEW]"
CONTROL_TOKEN_EXPLAIN = "[NASEA_EXPLAIN]"
CONTROL_TOKEN_RUN = "[NASEA_RUN]"
CONTROL_TOKEN_SEARCH = "[NASEA_SEARCH]"
app = typer.Typer(
    name="nasea",
    help="NASEA - Natural-Language Autonomous Software-Engineering Agent",
    add_completion=False
)

console = Console(soft_wrap=True)

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None  # type: ignore

def _approximate_token_length(text: str) -> int:
    """
    Accurate token counting using tiktoken when available.
    Falls back to rough char-to-token conversion if tiktoken is not installed.
    """
    if not text:
        return 0

    if TIKTOKEN_AVAILABLE and tiktoken:
        try:
            # Use cl100k_base encoding (used by gpt-4, gpt-3.5-turbo, text-embedding-ada-002)
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fall through to approximation on any error
            pass

    # Fallback: rough approximation (4 chars ≈ 1 token)
    return max(1, (len(text) + 3) // 4)


def _recalculate_context_usage(state: Dict[str, Any]) -> None:
    """Recompute estimated context usage based on stored chat history."""
    limit = state.get("context_limit") or config.max_tokens or 4096
    history = state.get("chat_history", [])
    total = sum(_approximate_token_length(msg.get("content", "")) for msg in history)
    state["context_used"] = min(total, limit)


def _compress_chat_history(state: Dict[str, Any], llm_client=None) -> None:
    """
    Intelligently compress old chat history by summarizing it.
    Keeps recent messages intact, compresses older ones into a summary.
    """
    history = state.get("chat_history", [])
    if len(history) < 6:  # Need enough history to compress
        return

    limit = state.get("context_limit") or config.max_tokens or 4096
    total = sum(_approximate_token_length(msg.get("content", "")) for msg in history)

    # Compress when we hit 70% of context (before it's critical)
    compress_threshold = int(limit * 0.7)
    if total <= compress_threshold:
        state["context_used"] = min(total, limit)
        return

    # Keep last 4 messages intact, compress the rest
    recent_messages = history[-4:]
    old_messages = history[:-4]

    if not old_messages:
        return

    # Format old messages for compression
    old_conversation = "\n".join(
        f"{msg['role'].upper()}: {msg.get('content', '')[:500]}"
        for msg in old_messages
    )

    # Use LLM to compress if available, otherwise simple truncation
    if llm_client:
        try:
            compression_prompt = """Summarize this conversation history in 2-3 sentences, preserving:
- Key decisions made
- Files created/modified
- Current project state
- Any errors encountered

Conversation:
""" + old_conversation[:2000]

            response = llm_client.chat(
                messages=[{"role": "user", "content": compression_prompt}],
                max_tokens=200,
                temperature=0.3,
                disable_thinking=True,
                strip_thinking=True
            )
            summary = getattr(response, 'content', str(response))
        except Exception:
            # Fallback to simple summary
            summary = f"[Earlier: {len(old_messages)} messages about project work]"
    else:
        summary = f"[Earlier: {len(old_messages)} messages about project work]"

    # Create compressed history
    compressed_history = [
        {"role": "system", "content": f"Previous conversation summary: {summary}"}
    ] + recent_messages

    state["chat_history"] = compressed_history
    new_total = sum(_approximate_token_length(msg.get("content", "")) for msg in compressed_history)
    state["context_used"] = min(new_total, limit)
    safe_console_print("[dim]Compressed older conversation to save context space.[/dim]")


def _compact_chat_history(state: Dict[str, Any], threshold: float = 0.9) -> None:
    """Trim oldest history so we stay within a fraction of the model limit."""
    history = state.get("chat_history", [])
    if not history:
        state["context_used"] = state.get("context_used", 0)
        return

    limit = state.get("context_limit") or config.max_tokens or 4096
    max_allowed = int(limit * threshold)
    total = sum(_approximate_token_length(msg.get("content", "")) for msg in history)

    if total <= max_allowed:
        state["context_used"] = min(total, limit)
        return

    removed = False
    while history and total > max_allowed:
        first = history.pop(0)
        total -= _approximate_token_length(first.get("content", ""))
        removed = True

        # Remove the paired assistant reply if present to keep dialog aligned
        if history and history[0].get("role") != first.get("role"):
            partner = history.pop(0)
            total -= _approximate_token_length(partner.get("content", ""))

    state["chat_history"] = history
    state["context_used"] = min(total, limit)
    if removed:
        safe_console_print("[dim]Trimmed older chat history to stay within the model context window.[/dim]")


def _set_active_model(state: Dict[str, Any], model_name: str) -> None:
    """Update state for a newly selected model."""
    state["model"] = model_name
    state["context_limit"] = LLMFactory.get_context_window(model_name, config.max_tokens)
    _compact_chat_history(state)
    _recalculate_context_usage(state)


def _trim_router_text(s: str, n: int = 240) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n-1].rstrip() + "…"

def _strip_thinking_blocks(text: str) -> str:
    """Remove any <think>...</think> segments (even if unterminated)."""
    if not text:
        return text
    if "<think>" not in text:
        return text
    # Remove matching pairs
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove trailing open blocks
    open_idx = text.find("<think>")
    if open_idx != -1:
        text = text[:open_idx]
    return text.strip()

def safe_console_print(text: str = "", **kwargs):
    console.print(text, **kwargs)
    sys.stdout.flush()

def _load_project_memory(project_root: Path) -> str:
    """
    Load project-specific memory/instructions from NASEA.md or CLAUDE.md.
    These files contain persistent project conventions that the AI should follow.

    Returns:
        String containing the memory content to inject into system prompt, or empty string if no file found.
    """
    memory_files = ["NASEA.md", "CLAUDE.md", ".nasea.md"]

    for fname in memory_files:
        fpath = project_root / fname
        if fpath.exists():
            try:
                content = fpath.read_text(encoding="utf-8").strip()
                if content:
                    safe_console_print(f"[dim]📝 Loaded project memory from {fname}[/dim]\n")
                    return f"\n\n# PROJECT MEMORY\nThe project has specific conventions in {fname}:\n\n{content}\n\nFollow these conventions strictly when generating or modifying code.\n"
            except Exception as e:
                logger.warning(f"Failed to read {fname}: {e}")
                continue

    return ""


def _get_context_snapshot(state: Dict[str, Any]) -> str:
    """
    Generate a live state snapshot for the LLM to have "eyes" into the current context.
    This prevents hallucinations by showing what actually exists.

    Returns:
        XML-formatted context snapshot string
    """
    # 1. Get active project info
    project_name = state.get("current_project_name", "None")
    project_path = state.get("last_project_path")

    # 2. Get available projects in output folder
    output_dir = Path(state.get("output_dir") or config.output_dir)
    available_projects = []
    available_project_paths = []
    if output_dir.exists():
        try:
            for d in sorted(output_dir.iterdir()):
                if d.is_dir() and not d.name.startswith('.'):
                    available_projects.append(f"  - {d.name}")
                    available_project_paths.append(d)
        except Exception:
            pass
    projects_list = "\n".join(available_projects[:15]) if available_projects else "  (No projects yet)"

    # Auto-select project if only one exists and none is set
    if project_name == "None" and len(available_project_paths) == 1:
        project_path = available_project_paths[0]
        project_name = project_path.name
        state["last_project_path"] = project_path
        state["current_project_name"] = project_name

    # 3. Get file tree of current project (limit to save tokens)
    file_list = "(No project selected)"
    if project_path and Path(project_path).exists():
        try:
            files = []
            project_dir = Path(project_path)
            for item in sorted(project_dir.iterdir())[:30]:  # Limit to 30 items
                if item.name.startswith('.'):
                    continue
                prefix = "📁" if item.is_dir() else "📄"
                files.append(f"  {prefix} {item.name}")
            file_list = "\n".join(files) if files else "(Empty directory)"
        except Exception:
            file_list = "(Unable to read directory)"

    # 4. Get last error (crucial for auto-debug routing)
    last_error = state.get("last_error", "None")

    # 5. Get recent conversation context
    # Use ~80% of small model's context for conversation (rest for system prompt + response)
    # Small model (venice-uncensored) has ~8k context, so ~6k chars for conversation
    max_conversation_chars = 6000

    recent_context = []
    chat_history = state.get("chat_history", [])
    total_chars = 0

    # Work backwards from most recent to oldest
    for msg in reversed(chat_history):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if not content:
            continue

        entry = f"  [{role}]: {content}"
        entry_len = len(entry)

        # Stop if adding this would exceed limit
        if total_chars + entry_len > max_conversation_chars:
            break

        recent_context.insert(0, entry)  # Insert at front to maintain order
        total_chars += entry_len

    context_summary = "\n".join(recent_context) if recent_context else "  (No recent messages)"

    # 6. Get pending action (if any)
    pending = state.get("pending_generation")
    pending_action = "None"
    if pending:
        pending_action = f"Pending build: {pending.get('prompt', '')[:100]}"

    return f"""
<context_state>
  <active_project>{project_name}</active_project>
  <project_path>{project_path or 'None'}</project_path>
  <output_folder>{output_dir}</output_folder>
  <available_projects>
{projects_list}
  </available_projects>
  <current_project_files>
{file_list}
  </current_project_files>
  <last_system_error>{last_error}</last_system_error>
  <pending_action>{pending_action}</pending_action>
  <recent_conversation>
{context_summary}
  </recent_conversation>
</context_state>
"""


def _get_cached_client(state: Dict[str, Any], model_name: str, temperature: Optional[float] = None):
    """Return cached LLM client keyed by model name."""
    clients = state.setdefault("llm_clients", {})
    if model_name in clients:
        return clients[model_name]

    try:
        client = LLMFactory.create_client(
            model=model_name,
            config=config,
            temperature=temperature
        )
    except Exception as exc:
        raise RuntimeError(f"Unable to initialize client for {model_name}: {exc}") from exc

    clients[model_name] = client
    return client


def _should_enable_thinking(state: Dict[str, Any]) -> bool:
    """Return True when large-model thinking mode should be used."""
    if state.get("thinking_requested"):
        return True
    if state.get("failure_streak", 0) >= 2:
        return True
    return False


def _update_thinking_state(state: Dict[str, Any], user_input: str, small_reply: str) -> None:
    """Update thinking preferences based on user instructions."""
    lowered = user_input.lower()
    request_keywords = [
        "think step by step",
        "think carefully",
        "reason it out",
        "show your reasoning",
        "be thorough",
        "take your time",
        "use chain of thought",
        "enable thinking",
        "please think",
    ]
    disable_keywords = [
        "don't overthink",
        "no thinking",
        "just answer",
        "keep it short",
        "fast response",
        "disable thinking",
        "skip thinking",
        "think later",
    ]

    if any(phrase in lowered for phrase in disable_keywords):
        state["thinking_requested"] = False
    elif any(phrase in lowered for phrase in request_keywords):
        state["thinking_requested"] = True

    lowered_reply = small_reply.lower()
    if "[thinking:on]" in lowered_reply:
        state["thinking_requested"] = True
    elif "[thinking:off]" in lowered_reply:
        state["thinking_requested"] = False


def _build_router_prompt(state: Dict[str, Any]) -> str:
    """
    Build the dynamic router prompt with live context injection.
    Loads from modular prompt file with variable substitution.
    """
    context_snapshot = _get_context_snapshot(state)

    # Load router prompt from modular file with variable injection
    return load_system_prompt(
        "router",
        CONTEXT_SNAPSHOT=context_snapshot,
        CONTROL_TOKEN_CREATE=CONTROL_TOKEN_CREATE,
        CONTROL_TOKEN_EDIT=CONTROL_TOKEN_EDIT,
        CONTROL_TOKEN_DEBUG=CONTROL_TOKEN_DEBUG,
        CONTROL_TOKEN_TEST=CONTROL_TOKEN_TEST,
        CONTROL_TOKEN_LIST=CONTROL_TOKEN_LIST,
        CONTROL_TOKEN_VIEW=CONTROL_TOKEN_VIEW,
        CONTROL_TOKEN_EXPLAIN=CONTROL_TOKEN_EXPLAIN,
        CONTROL_TOKEN_RUN=CONTROL_TOKEN_RUN,
        CONTROL_TOKEN_SEARCH=CONTROL_TOKEN_SEARCH,
    )

SYSTEM_STATS_TEMPLATE = """
# SYSTEM STATS
- Current Date: {current_date}
- True Context Window: {context_limit:,} tokens
- Current Usage: {context_used:,} tokens
"""

TOKEN_RE = re.compile(r"(?:\s|\A)\[(?:NASEA_CREATE|NASEA_EDIT|NASEA_DEBUG|NASEA_TEST|NASEA_LIST|NASEA_VIEW|NASEA_EXPLAIN|NASEA_RUN|NASEA_SEARCH)\](?:\s*\[project_name=([a-z0-9-]+)\])?\s*$")

# Pattern to strip ANY [NASEA_*] tokens (including made-up ones like [NASEA_WEB_SEARCH ...])
STRIP_NASEA_TOKENS_RE = re.compile(r'\[/?NASEA_[A-Z_]+(?:\s+[^\]]+)?\]')

def _strip_nasea_tokens(text: str) -> str:
    """Remove all [NASEA_*] tokens from text (for display purposes)."""
    return STRIP_NASEA_TOKENS_RE.sub('', text).strip()

def _extract_trailing_token(text: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Extract control token and optional project name from text.

    Returns:
        Tuple of (clean_text, token, project_name)
    """
    m = TOKEN_RE.search(text)
    if not m:
        return text.strip(), None, None

    # Extract full match (token + optional project_name)
    full_match = m.group(0).strip()
    project_name = m.group(1) if m.group(1) else None

    # Clean text is everything before the token
    clean = text[:m.start()].rstrip()

    # Extract just the token part (without project_name)
    token_only = re.search(r"\[(NASEA_[A-Z]+)\]", full_match)
    if token_only:
        token = f"[{token_only.group(1)}]"
    else:
        token = full_match

    return clean, token, project_name

def _run_small_model(state: Dict[str, Any], user_input: str, retry_count: int = 0) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Run the lightweight model to determine routing/control tokens.

    Returns:
        Tuple of (response_text, token, project_name)
    """
    from nasea.core.dynamic_loader import get_loader

    try:
        # Use higher temperature for more natural, complete responses
        client = _get_cached_client(state, "venice-uncensored", temperature=0.7)
    except RuntimeError as exc:
        logger.error(exc)
        return "", None, None

    # Build dynamic router prompt with live context
    context_limit = state.get("context_limit") or config.max_tokens or 131072
    import datetime
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Use dynamic prompt with context injection (replaces static SMALL_MODEL_PROMPT)
    router_prompt = _build_router_prompt(state)
    system_content = router_prompt + SYSTEM_STATS_TEMPLATE.format(
        current_date=current_date,
        context_limit=context_limit,
        context_used=state.get("context_used", 0),
    )

    messages = [{"role": "system", "content": system_content}]

    # Include recent chat history as actual messages (not just in system prompt)
    # This gives the model proper conversational context
    chat_history = state.get("chat_history", [])

    # Calculate how much history we can include (stay under 80% of context)
    context_limit = state.get("context_limit") or config.max_tokens or 4096
    max_history_tokens = int(context_limit * 0.6)  # Leave room for system prompt and response

    history_tokens = 0
    history_to_include = []
    for msg in reversed(chat_history):
        msg_tokens = _approximate_token_length(msg.get("content", ""))
        if history_tokens + msg_tokens > max_history_tokens:
            break
        history_to_include.insert(0, msg)
        history_tokens += msg_tokens

    # Add history messages
    for msg in history_to_include:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current user input
    messages.append({"role": "user", "content": user_input})

    # Show loader while waiting for response
    loader = get_loader("chat", console=console)
    loader.start()

    # Use streaming so we can interrupt immediately
    stream = None
    text_chunks = []
    try:
        stream = client.stream_chat(
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
            disable_thinking=True,
            strip_thinking=True,
        )

        # Iterate through chunks, checking for interrupt each time
        for chunk in stream:
            if loader.was_interrupted:
                break
            if chunk.choices and chunk.choices[0].delta.content:
                text_chunks.append(chunk.choices[0].delta.content)

    except Exception as exc:
        if stream and hasattr(stream, 'close'):
            stream.close()
        loader.stop()
        if loader.was_interrupted:
            return "__INTERRUPTED__", None, None
        safe_console_print(f"[dim]  ⎿[/dim] [red] ⚠[/red] [dim]Connection error. Retrying may help.[/dim]\n")
        return "__ERROR__", None, None
    finally:
        # Always close stream to stop server-side generation
        if stream and hasattr(stream, 'close'):
            stream.close()
        loader.stop()

    # Check if user pressed Esc to interrupt
    if loader.was_interrupted:
        return "__INTERRUPTED__", None, None

    try:
        text = "".join(text_chunks).strip()

        # With streaming, we don't get finish_reason/tokens_used metadata
        # Check for truncation by text length heuristic instead
        if len(text) < 50 and retry_count == 0:
            logger.debug(f"Response may be truncated (length={len(text)}). Retrying...")
            return _run_small_model(state, user_input, retry_count=1)

        # Strip any residual thinking blocks for consistent display
        text = _strip_thinking_blocks(text)

        # Fallback if response is empty
        if not text:
            logger.debug(f"Empty response from small model for input: {user_input[:50]}")
            # Return a generic fallback so chat doesn't fail
            return "I'm here to help. What would you like to do?", None, None
    except Exception as exc:
        logger.error(f"Response processing failed: {exc}")
        return "", None, None

    text, detected_token, project_name = _extract_trailing_token(text)
    # Strip any remaining [NASEA_*] tokens that weren't recognized (e.g. made-up tokens)
    text = _strip_nasea_tokens(text)
    #text = _trim_router_text(text)
    logger.debug(f"Small model response: text='{text}', token={detected_token}, project_name={project_name}")

    return text, detected_token, project_name

def _handle_project_creation(state: Dict[str, Any], user_input: str, project_name: Optional[str] = None) -> None:
    """Route request to project creation workflow with optional AI-provided project name."""
    from pathlib import Path
    from nasea.core.tool_definitions import GENERATION_TOOLS
    from nasea.core.streaming_tool_handler import StreamingToolHandler, ToolUseConversation
    import re
    import hashlib

    output_base = Path(state.get("output_dir") or config.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # --- DYNAMIC PROJECT NAME ---
    if "current_project_name" not in state or state.get("last_intent") != "CREATE":
        # Use AI-provided project name if available, otherwise fallback
        if not project_name:
            project_name = state.get("project_name")

        if not project_name:
            # Fallback to hash-based name if AI didn't provide one
            digest = hashlib.md5(user_input.encode()).hexdigest()[:6]
            project_name = f"app-{digest}"
            logger.debug(f"Using fallback project name: {project_name}")

        # Sanitize name
        project_name = re.sub(r'[^a-z0-9-]', '-', project_name.lower())
        project_name = re.sub(r'-+', '-', project_name).strip('-')
        project_name = project_name[:25] or "app"

        # Collision check — if dir exists, append counter
        original_name = project_name
        counter = 2
        while (output_base / project_name).exists():
            project_name = f"{original_name}-{counter}"
            counter += 1

        state["current_project_name"] = project_name
        state["last_intent"] = "CREATE"

    # --- MODEL INITIALIZATION ---
    # Use coder-specialized model for better tool call accuracy
    try:
        llm_client = _get_cached_client(state, GENERATION_MODEL)
    except RuntimeError as exc:
        message = f"Coder model unavailable ({exc}). Try again shortly or switch models with `/model`."
        safe_console_print(f"[red]{message}[/red]")
        state["chat_history"].append({"role": "assistant", "content": message})
        return

    project_dir = output_base / state["current_project_name"]

    handler = StreamingToolHandler(
        console=console,
        project_root=project_dir,
        permission_mode=state.get("permission_mode", "ask"),
        loader_type="create"
    )
    # Carry forward "open_browser already used" across /continue runs to reduce noise.
    if state.get("open_browser_used"):
        handler.browsers_opened.append({"path": "<prior>"})

    # Load project memory (NASEA.md, CLAUDE.md, .nasea.md)
    project_memory = _load_project_memory(project_dir)

    # Load system prompt from modular file
    system_prompt = load_system_prompt(
        "create",
        PROJECT_NAME=state['current_project_name'],
        PROJECT_PATH=f"{project_dir}/",
        PROJECT_MEMORY=project_memory,
        RECENT_TOOL_CONTEXT=state.get("last_tool_context") or ""
    ) + f"\n\nUSER REQUEST:\n{user_input}"

    # Always show thinking to users so they see the planning process
    enable_thinking = True
    strip_thinking = False  # Don't strip - we filter and display it ourselves

    from nasea.core.tool_definitions import EDIT_TOOLS
    conversation = ToolUseConversation(
        client=llm_client.client if hasattr(llm_client, "client") else llm_client,
        handler=handler,
        console=console,
        model=GENERATION_MODEL,
        tools=EDIT_TOOLS,  # Excludes 'think' which causes issues with some models
        enable_thinking=enable_thinking,
        strip_thinking=strip_thinking
    )

    # Build truncated chat history for context (stay under 50% of context)
    chat_history = state.get("chat_history", [])
    context_limit = state.get("context_limit") or config.max_tokens or 131072
    max_history_tokens = int(context_limit * 0.4)  # Leave more room for creation

    history_tokens = 0
    history_to_include = []
    for msg in reversed(chat_history):
        msg_tokens = _approximate_token_length(msg.get("content", ""))
        if history_tokens + msg_tokens > max_history_tokens:
            break
        history_to_include.insert(0, msg)
        history_tokens += msg_tokens

    try:
        result = conversation.run(
            system_message=system_prompt,
            user_message=user_input,
            chat_history=history_to_include if history_to_include else None
        )
    except KeyboardInterrupt:
        safe_console_print("\n[yellow]Project creation cancelled by user.[/yellow]\n")
        state["last_project_path"] = handler.executor.project_root
        return  # Return to prompt instead of exiting the app
    except Exception as exc:
        message = f"Project creation failed: {exc}"
        safe_console_print(f"\n[red]{message}[/red]")
        logger.exception("Creation flow failed")
        state["chat_history"].append({"role": "assistant", "content": message})
        # Track error for context-aware routing
        state["last_error"] = str(exc)

        # Auto-escalate to debug mode for recovery
        safe_console_print("\n[yellow]Attempting automatic recovery...[/yellow]")
        state["last_project_path"] = handler.executor.project_root
        _handle_project_edit(
            state,
            f"Fix the following error and complete the project: {user_input}\nError: {exc}",
            mode="debug"
        )
        return
    finally:
        # Persist whether the browser was opened during this session.
        state["open_browser_used"] = bool(getattr(handler, "browsers_opened", []))

    # Check if operation actually succeeded
    if not result.get("success", True):  # Default to True for backwards compatibility
        error = result.get("error", "Unknown error")
        # Don't show error message for user interrupt - it's already shown
        if error == "User interrupted":
            state.pop("last_error", None)  # Clear error on interrupt
            return
        safe_console_print(f"\n[red]⏺ Project creation failed: {error}[/red]\n")
        state["chat_history"].append({"role": "assistant", "content": f"Failed: {error}"})
        # Track error for context-aware routing
        state["last_error"] = error
        return

    # Clear error on success
    state.pop("last_error", None)

    # Persist permission mode if user chose "Always"
    if handler.permission_mode == "always":
        state["permission_mode"] = "always"

    # Summary is already displayed by handler's _show_summary during complete_generation
    # Just record to chat history
    summary = result.get("summary") or "Project generation completed."
    state["chat_history"].append({"role": "assistant", "content": f"Created: {summary}"})
    state["last_project_path"] = handler.executor.project_root
    # Try intelligent compression first, then fallback to trimming
    _compress_chat_history(state, llm_client)
    _compact_chat_history(state)
    _recalculate_context_usage(state)

def _handle_web_search(state: Dict[str, Any], user_input: str) -> None:
    """Handle web search requests using the big model with web_search tool."""
    from pathlib import Path
    from nasea.core.tool_definitions import GENERATION_TOOLS
    from nasea.core.streaming_tool_handler import StreamingToolHandler, ToolUseConversation
    from nasea.core.dynamic_loader import get_loader

    # Use a temporary directory for the executor (no project context needed)
    temp_root = Path(state.get("output_dir") or config.output_dir) / ".search_temp"
    temp_root.mkdir(parents=True, exist_ok=True)

    try:
        llm_client = _get_cached_client(state, GENERATION_MODEL)
    except RuntimeError as exc:
        safe_console_print(f"[red]Search unavailable: {exc}[/red]\n")
        return

    # Create handler with minimal tools for search
    handler = StreamingToolHandler(
        project_root=temp_root,
        console=console,
        permission_mode="always",  # Auto-approve for searches
        loader_type="chat"
    )

    # Only provide web_search tool - nothing else
    search_tools = [t for t in GENERATION_TOOLS if t["function"]["name"] == "web_search"]

    # Create conversation with ONLY web search capability
    conversation = ToolUseConversation(
        client=llm_client.client if hasattr(llm_client, "client") else llm_client,
        handler=handler,
        console=console,
        model=GENERATION_MODEL,
        tools=search_tools,
        enable_thinking=False,
        strip_thinking=True
    )

    # Build search prompt with date context from modular prompt file
    import datetime
    current_date = datetime.datetime.now().strftime("%B %d, %Y")

    system_prompt = load_system_prompt("web_search", CURRENT_DATE=current_date)

    # Build chat history for context (so model understands follow-up questions)
    chat_history = state.get("chat_history", [])
    context_limit = state.get("context_limit") or config.max_tokens or 64000
    max_history_tokens = int(context_limit * 0.4)  # Leave room for search results

    history_tokens = 0
    history_to_include = []
    for msg in reversed(chat_history):
        msg_tokens = _approximate_token_length(msg.get("content", ""))
        if history_tokens + msg_tokens > max_history_tokens:
            break
        history_to_include.insert(0, msg)
        history_tokens += msg_tokens

    # Run the search conversation with chat history
    try:
        result = conversation.run(
            system_message=system_prompt,
            user_message=user_input,
            chat_history=history_to_include if history_to_include else None
        )
    except KeyboardInterrupt:
        safe_console_print("\n[dim]Search cancelled.[/dim]\n")
        return

    # Add to chat history
    if result.get("final_response"):
        state["chat_history"].append({"role": "user", "content": user_input})
        state["chat_history"].append({"role": "assistant", "content": result["final_response"]})
        _compact_chat_history(state)


def _handle_project_edit(state: Dict[str, Any], user_input: str, mode: str = "edit") -> None:
    """Route request to editing workflow using existing project context."""
    from pathlib import Path
    from nasea.core.tool_definitions import GENERATION_TOOLS, READ_ONLY_TOOLS, EDIT_TOOLS
    from nasea.core.streaming_tool_handler import StreamingToolHandler, ToolUseConversation

    output_dir = Path(state.get("output_dir") or config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_name(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def _find_project_from_prompt(prompt: str, candidates: List[Path]) -> Optional[Path]:
        normalized_prompt = _normalize_name(prompt)
        if not normalized_prompt:
            return None
        for candidate in candidates:
            variants = {
                candidate.name,
                candidate.name.replace("-", " "),
                candidate.name.replace("_", " "),
            }
            for variant in variants:
                normalized_variant = _normalize_name(variant)
                if normalized_variant and normalized_variant in normalized_prompt:
                    return candidate
        return None

    projects = [d for d in output_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    project_root = state.get("last_project_path")
    last_displayed_project = state.get("last_displayed_project_path")

    def _maybe_print_project_banner(label: str, path: Path) -> None:
        nonlocal last_displayed_project
        current = str(path)
        if last_displayed_project == current:
            return
        safe_console_print(f"[cyan]{label}:[/cyan] {path.name}\n")
        last_displayed_project = current
        state["last_displayed_project_path"] = current

    # Check if saved project still exists
    if project_root and not Path(project_root).exists():
        project_root = None
        state["last_project_path"] = None

    hinted = _find_project_from_prompt(user_input, projects)
    if hinted:
        project_root = hinted
        state["last_project_path"] = project_root
        state["current_project_name"] = project_root.name  # Track for router context
        _maybe_print_project_banner("Switching to project", project_root)
    elif project_root and project_root != output_dir:
        # Show which project we're using (was set from previous session)
        state["current_project_name"] = Path(project_root).name  # Track for router context
        _maybe_print_project_banner("Using project", Path(project_root))

    if not project_root:
        if projects:
            if len(projects) == 1:
                # Only one project - use it
                project_root = projects[0]
                state["last_project_path"] = project_root
                state["current_project_name"] = project_root.name  # Track for router context
                _maybe_print_project_banner("Using project", project_root)
            else:
                # Multiple projects - ask user to pick
                import questionary
                projects.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                choices = [p.name for p in projects[:10]]  # Limit to 10
                safe_console_print("[yellow]Multiple projects found. Which one?[/yellow]")
                selected = questionary.select(
                    "",
                    choices=choices,
                    style=questionary.Style([
                        ('highlighted', 'fg:cyan bold'),
                        ('pointer', 'fg:cyan bold'),
                    ])
                ).ask()
                if selected:
                    project_root = output_dir / selected
                    state["last_project_path"] = project_root
                    state["current_project_name"] = selected  # Track for router context
                    _maybe_print_project_banner("Using project", project_root)
                else:
                    # User cancelled
                    return
        else:
            project_root = output_dir
            safe_console_print(
                "[yellow]No existing project found. Working in output directory.[/yellow]\n"
            )

    # === EXPLORATION PHASE (Like Claude Code) ===
    # Only explore for certain modes where understanding context is critical
    continue_like = user_input.strip().lower().startswith("continue from where you stopped")
    should_explore = mode in ("edit", "debug") and project_root != output_dir and not continue_like

    if should_explore and not state.get("skip_exploration"):
        from nasea.core.modes.explore import ExploreMode

        try:
            # Use lightweight model for exploration
            explore_client = _get_cached_client(state, "venice-uncensored", temperature=0.3)

            explorer = ExploreMode(
                project_root=Path(project_root),
                console=console,
                llm_client=explore_client
            )

            exploration = explorer.explore(user_input)
            explorer.display_findings(exploration)

            # Store findings in state for use by main agent
            state["exploration_findings"] = exploration
            # Auto-proceed (no confirmation needed - user already requested the action)

        except Exception as e:
            # Don't fail the whole operation if exploration fails
            logger.warning(f"Exploration failed, continuing anyway: {e}")
            safe_console_print(f"[yellow]⚠[/yellow] Quick exploration skipped\n")
    # === END EXPLORATION PHASE ===

    try:
        llm_client = _get_cached_client(state, GENERATION_MODEL)
    except RuntimeError as exc:
        message = f"Coder model unavailable ({exc}). Try again shortly or switch models with `/model`."
        safe_console_print(f"[red]{message}[/red]")
        state["chat_history"].append({"role": "assistant", "content": message})
        return

    handler = StreamingToolHandler(
        console=console,
        project_root=Path(project_root),
        permission_mode=state.get("permission_mode", "ask"),
        loader_type=mode
    )
    # Carry forward "open_browser already used" across /continue runs to reduce noise.
    if state.get("open_browser_used"):
        handler.browsers_opened.append({"path": "<prior>"})

    # Load project memory (NASEA.md, CLAUDE.md, .nasea.md)
    project_memory = _load_project_memory(Path(project_root))

    # Pre-load file list to avoid wasteful list_files() call
    def _get_project_files(root: Path, max_files: int = 50) -> str:
        """Get project files list to inject into prompt."""
        skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build', '.next'}
        try:
            files = []
            for f in root.rglob("*"):
                if f.is_file():
                    # Skip hidden and cache directories
                    if any(p.startswith('.') or p in skip_dirs for p in f.parts):
                        continue
                    rel = f.relative_to(root)
                    files.append(str(rel))
                    if len(files) >= max_files:
                        break
            if files:
                file_list = "\n".join(f"  - {f}" for f in sorted(files))
                return f"\n# PROJECT FILES\n{file_list}\n"
            return ""
        except Exception:
            return ""

    project_files = _get_project_files(Path(project_root))

    # Get recent conversation context - include more history for debugging context
    chat_history = state.get("chat_history", [])
    recent_history = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in chat_history[-8:]
    ) or "No previous messages"

    # For short follow-up messages like "do it", "yes", "fix it", extract the real task from history
    # This helps the large model understand what the user actually wants
    actual_task = user_input
    if len(user_input) < 30 and len(chat_history) >= 2:
        # Look back for a more detailed user request
        for msg in reversed(chat_history[:-1]):  # Exclude current message
            if msg.get("role") == "user" and len(msg.get("content", "")) > 30:
                # Found a more detailed previous request - combine them
                actual_task = f"{msg['content']}\n\nUser confirmed: {user_input}"
                break

    # Get last error context
    last_error = state.get("last_error", "")
    error_context = f"\n# LAST ERROR\n{last_error}\n" if last_error else ""

    # Load system prompt from modular file based on mode
    recent_tool_context = state.get("last_tool_context") or ""

    # Get session state summary (tracks files read, issues fixed, checkpoints)
    session_state = ""
    if hasattr(handler, 'executor') and hasattr(handler.executor, 'get_session_state_summary'):
        session_state = handler.executor.get_session_state_summary()

    prompt_vars = {
        "PROJECT_MEMORY": project_memory,
        "PROJECT_FILES": project_files,
        "ERROR_CONTEXT": error_context,
        "RECENT_TOOL_CONTEXT": recent_tool_context,
        "USER_TASK": actual_task,
        "PROJECT_ROOT": str(handler.executor.project_root),
        "RECENT_HISTORY": recent_history,
        "SESSION_STATE": session_state
    }

    # Map mode to prompt file name
    mode_to_prompt = {
        "debug": "debug",
        "test": "test",
        "view": "view",
        "explain": "explain",
        "run": "run",
        "edit": "edit"
    }
    prompt_name = mode_to_prompt.get(mode, "edit")
    system_prompt = load_system_prompt(prompt_name, **prompt_vars)

    # Always show thinking to users so they see the planning process
    enable_thinking = True
    strip_thinking = False  # Don't strip - we filter and display it ourselves

    # Use appropriate tools for each mode
    if mode in ("view", "explain"):
        tools = READ_ONLY_TOOLS
    elif mode in ("edit", "debug", "test", "run"):
        tools = EDIT_TOOLS  # Excludes 'think' which causes issues with some models
    else:
        tools = GENERATION_TOOLS

    conversation = ToolUseConversation(
        client=llm_client.client if hasattr(llm_client, "client") else llm_client,
        handler=handler,
        console=console,
        model=GENERATION_MODEL,
        tools=tools,
        enable_thinking=enable_thinking,
        strip_thinking=strip_thinking
    )

    # Build truncated chat history for large model (stay under 60% of context)
    context_limit = state.get("context_limit") or config.max_tokens or 131072
    max_history_tokens = int(context_limit * 0.5)  # Leave room for system prompt, tools, and response

    history_tokens = 0
    history_to_include = []
    for msg in reversed(chat_history[:-1]):  # Exclude current message (already in user_input/actual_task)
        msg_tokens = _approximate_token_length(msg.get("content", ""))
        if history_tokens + msg_tokens > max_history_tokens:
            break
        history_to_include.insert(0, msg)
        history_tokens += msg_tokens

    try:
        result = conversation.run(
            system_message=system_prompt,
            user_message=actual_task,  # Use the enhanced task with context
            chat_history=history_to_include if history_to_include else None
        )
    except KeyboardInterrupt:
        safe_console_print(f"\n[yellow]{mode.capitalize()} cancelled by user.[/yellow]\n")
        state["last_project_path"] = handler.executor.project_root
        return  # Return to prompt instead of exiting the app
    except Exception as exc:
        message = f"{mode.capitalize()} workflow failed: {exc}"
        safe_console_print(f"\n[red]{message}[/red]")
        if logger:
            logger.exception("Edit flow failed")
        state["chat_history"].append({"role": "assistant", "content": message})
        # Track error for context-aware routing
        state["last_error"] = str(exc)
        return

    # Check if operation actually succeeded
    if not result.get("success", True):  # Default to True for backwards compatibility
        error = result.get("error", "Unknown error")
        # Don't show error message for user interrupt - it's already shown
        if error == "User interrupted":
            state.pop("last_error", None)  # Clear error on interrupt
            return
        safe_console_print(f"\n[red]⏺ {mode.capitalize()} failed: {error}[/red]\n")
        state["chat_history"].append({"role": "assistant", "content": f"Failed: {error}"})
        # Track error for context-aware routing
        state["last_error"] = error
        return

    # Clear error on success
    state.pop("last_error", None)

    # Persist permission mode if user chose "Always"
    if handler.permission_mode == "always":
        state["permission_mode"] = "always"

    # Build a summary of operations performed for chat history
    operations_summary = []
    if handler.files_created:
        operations_summary.append(f"Created: {', '.join(handler.files_created)}")
    if handler.files_updated:
        operations_summary.append(f"Updated: {', '.join(handler.files_updated)}")
    if handler.files_deleted:
        operations_summary.append(f"Deleted: {', '.join(handler.files_deleted)}")
    if handler.commands_run:
        operations_summary.append(f"Ran {len(handler.commands_run)} command(s)")
    if getattr(handler, "files_read", None):
        operations_summary.append(f"Read {len(handler.files_read)} file(s)")
    if getattr(handler, "searches_run", None):
        operations_summary.append(f"Searched {len(handler.searches_run)} time(s)")
    if getattr(handler, "syntax_checked", None):
        operations_summary.append(f"Linted {len(handler.syntax_checked)} file(s)")
    if getattr(handler, "browsers_opened", None):
        operations_summary.append(f"Opened browser {len(handler.browsers_opened)} time(s)")

    # Persist a compact tool-context summary to reduce re-reads on /continue
    try:
        state["last_tool_context"] = handler.format_recent_tool_context()
    except Exception:
        state["last_tool_context"] = state.get("last_tool_context", "")

    # Persist a compact findings summary to reduce repeated re-discovery on /continue
    try:
        state["last_findings"] = handler.format_recent_findings()
    except Exception:
        state["last_findings"] = state.get("last_findings", "")

    # Persist whether the browser was opened during this session.
    state["open_browser_used"] = bool(getattr(handler, "browsers_opened", []))

    # Add agent output to chat history for router context
    agent_output = handler.text_buffer if handler.text_buffer else ""
    agent_output_stripped = agent_output.strip()

    # Combine text output with operations summary
    history_content = agent_output_stripped
    if operations_summary:
        ops_text = " | ".join(operations_summary)
        if history_content:
            history_content = f"{history_content}\n[Operations: {ops_text}]"
        else:
            history_content = f"[Operations: {ops_text}]"

    if history_content:
        state["chat_history"].append({"role": "assistant", "content": history_content})
        # Router will handle follow-up routing based on conversation context
        return

    # Summary is already displayed by handler's _show_summary during complete_generation
    # Just record to chat history
    summary = result.get("summary") or f"{mode.capitalize()} complete."
    state["chat_history"].append({"role": "assistant", "content": f"Updated: {summary}"})
    state["last_project_path"] = handler.executor.project_root
    # Try intelligent compression first, then fallback to trimming
    _compress_chat_history(state, llm_client)
    _compact_chat_history(state)
    _recalculate_context_usage(state)


def _process_user_message(user_input: str, state: Dict[str, Any]) -> bool:
    """Process interactive user input with routing via control tokens.

    Returns:
        True if processing completed normally, False if interrupted/error.
    """
    # Short inputs (< 15 chars) with a last_intent are likely confirmations
    # Re-route them directly without going through the router
    last_intent = state.get("last_intent")
    if last_intent and len(user_input.strip()) < 15:
        mode_map = {"DEBUG": "debug", "EDIT": "edit", "CREATE": None}
        if last_intent in mode_map:
            if last_intent == "CREATE":
                _handle_project_creation(state, user_input, None)
            else:
                _handle_project_edit(state, user_input, mode=mode_map[last_intent])
            return True

    # Run lightweight model for routing
    small_reply, control_token, project_name = _run_small_model(state, user_input)

    # Check if user interrupted with Esc (message already shown by loader)
    if small_reply == "__INTERRUPTED__":
        return False

    # Check if there was a connection error (message already shown)
    if small_reply == "__ERROR__":
        return False

    _update_thinking_state(state, user_input, small_reply or "")

    # Append user message to history for future context
    state["chat_history"].append({"role": "user", "content": user_input})
    limit = state.get("context_limit") or config.max_tokens
    state["context_used"] = min(
        limit,
        state.get("context_used", 0) + _approximate_token_length(user_input)
    )

    if control_token == CONTROL_TOKEN_LIST:
        # Handle project listing directly
        from pathlib import Path
        if small_reply:
            _emit_small_reply(state, small_reply)

        output_dir = Path(state.get("output_dir") or config.output_dir).resolve()
        cwd_display = str(output_dir).replace(str(Path.home()), "~")

        # Show as tool call (Claude Code style)
        safe_console_print(f"[green]⏺[/green] [bold white]ListProjects[/bold white]({cwd_display})")

        if output_dir.exists():
            projects = [d for d in output_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            if projects:
                project_names = "\n".join(f"    - {p.name}" for p in sorted(projects, key=lambda x: x.stat().st_mtime, reverse=True))
                response = f"Found {len(projects)} project(s):\n{project_names}"
                safe_console_print(f"  [green]⎿[/green] Found {len(projects)} project(s)")
                safe_console_print(f"{project_names}\n")
            else:
                response = "No projects found in output folder."
                safe_console_print(f"  [dim]⎿  No projects found[/dim]\n")
        else:
            response = "Output folder doesn't exist yet."
            safe_console_print(f"  [yellow]⎿[/yellow] [dim]Output folder doesn't exist yet.[/dim]\n")

        state["chat_history"].append({"role": "assistant", "content": response})
        return True

    if control_token == CONTROL_TOKEN_CREATE:
        # Suppress router prose when routing to tool modes.
        _handle_project_creation(state, user_input, project_name)
        return True

    if control_token == CONTROL_TOKEN_EDIT:
        # Suppress router prose when routing to tool modes.
        state["last_intent"] = "EDIT"
        _handle_project_edit(state, user_input, mode="edit")
        return True

    if control_token == CONTROL_TOKEN_DEBUG:
        # Suppress router prose when routing to tool modes.
        state["last_intent"] = "DEBUG"
        _handle_project_edit(state, user_input, mode="debug")
        return True

    if control_token == CONTROL_TOKEN_TEST:
        # Suppress router prose when routing to tool modes.
        _handle_project_edit(state, user_input, mode="test")
        return True

    if control_token == CONTROL_TOKEN_VIEW:
        # Suppress router prose when routing to tool modes.
        _handle_project_edit(state, user_input, mode="view")
        return True

    if control_token == CONTROL_TOKEN_EXPLAIN:
        # Suppress router prose when routing to tool modes.
        _handle_project_edit(state, user_input, mode="explain")
        return True

    if control_token == CONTROL_TOKEN_RUN:
        # Suppress router prose when routing to tool modes.
        _handle_project_edit(state, user_input, mode="run")
        return True

    if control_token == CONTROL_TOKEN_SEARCH:
        # Suppress router prose when routing to tool modes.
        _handle_web_search(state, user_input)
        return True

    # If we get here, there's no routing token - the AI answered directly
    # or something went wrong
    if small_reply:
        # AI provided a direct answer (no routing to big model)
        _emit_small_reply(state, small_reply)
    else:
        # Something went wrong - no response from small model
        safe_console_print("[dim]  ⎿[/dim] [yellow] ⚠[/yellow] [dim]No response. Please try again.[/dim]\n")

    return True

class DynamicStatus:
    """Animated status indicator with rotating messages."""

    def __init__(
        self,
        messages: List[str],
        spinner: str = "arc",
        style: str = "cyan",
        interval: float = 5.0,
        show_cancel_hint: bool = True,
    ):
        self.messages = messages or ["Working…"]
        self.spinner = spinner
        self.style = style
        self.interval = interval
        self.show_cancel_hint = show_cancel_hint
        self._active = console.is_terminal

        if self.show_cancel_hint and self._active and self.messages:
            self.messages = self.messages.copy()
            self.messages[0] = (
                f"{self.messages[0]} [dim](Ctrl+C to cancel)[/dim]"
            )
        self._progress: Optional[Progress] = None
        self._task_id: Optional[int] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def __enter__(self):
        if not self._active:
            return self

        self._progress = Progress(
            SpinnerColumn(spinner_name=self.spinner, style=self.style),
            TextColumn("{task.fields[msg]}", style=f"bold {self.style}"),
            console=console,
            transient=True,  # Will automatically clear when done
            refresh_per_second=12,
        )
        self._progress.__enter__()
        self._task_id = self._progress.add_task("status", msg=self.messages[0])
        self._running = True
        self._thread = threading.Thread(target=self._rotate_messages, daemon=True)
        self._thread.start()
        return self

    def _rotate_messages(self):
        idx = 0
        while self._running:
            time.sleep(self.interval)
            if not self._progress or self._task_id is None:
                continue
            idx = (idx + 1) % len(self.messages)
            self._progress.update(self._task_id, msg=self.messages[idx])

    def __exit__(self, exc_type, exc, tb):
        if not self._active:
            return

        self._running = False
        if self._thread:
            self._thread.join(timeout=0.3)
        if self._progress:
            self._progress.__exit__(exc_type, exc, tb)

        # The cancel hint disappears automatically with transient=True
        # No need to manually clear it


def _intent_status_messages(prompt: str) -> List[str]:
    length = len(prompt.strip())
    if length <= 20:
        return [
            "Working…",
            "A sec…",
            "On it…",
            "Almost done…",
        ]
    if length <= 80:
        return [
            "Understanding your request…",
            "Mapping requirements…",
            "Hang tight…",
            "Working…",
        ]
    return [
        "Parsing this hefty brief…",
        "Lol, give me a minute…",
        "Still reading every line…",
        "Sip a coffee, I'm on it…",
        "Almost there…",
    ]


def _generation_status_messages(model_name: str) -> List[str]:
    name = model_name or config.default_model
    return [
        f"Handing specs to {name}…",
        "Sketching architecture…",
        "Implementing modules…",
        "Running quick tests…",
        "Polishing outputs…",
    ]

def _get_command_options(mock_mode: bool = False) -> List[Dict[str, str]]:
    """Get command options dynamically based on current mode."""
    options = [
        {"value": "__build__", "display": "/build", "meta": "", "description": "Start a new generation prompt"},
        {"value": "__model__", "display": "/model", "meta": "<name>", "description": "Switch the active LLM model"},
    ]

    # Only show the opposite mode command
    if mock_mode:
        options.append({"value": "/live", "display": "/live", "meta": "", "description": "Switch to live mode (needs API keys)"})
    else:
        options.append({"value": "/mock", "display": "/mock", "meta": "", "description": "Switch to offline demo mode"})

    options.extend([
        {"value": "/clear", "display": "/clear", "meta": "", "description": "Clear the screen and chat history"},
        {"value": "/continue", "display": "/continue", "meta": "", "description": "Continue the last incomplete task"},
        {"value": "/memory", "display": "/memory", "meta": "", "description": "Show saved memories"},
        {"value": "/help", "display": "/help", "meta": "", "description": "Show all keyboard shortcuts and commands"},
        {"value": "/exit", "display": "/exit", "meta": "", "description": "Leave the NASEA shell"},
    ])

    return options


# Default command options for completer (shows all)
COMMAND_OPTIONS = _get_command_options(mock_mode=False)


class NonEmptyValidator(Validator):
    """Validator that rejects empty input to prevent buffered Enter presses from submitting."""

    def validate(self, document):
        text = document.text.strip()
        if not text:
            # Don't raise ValidationError - just silently prevent submission
            # This way buffered Enter presses are ignored without showing error messages
            raise ValidationError(message="")


class CommandCompleter(Completer):
    """Custom completer that shows command descriptions."""

    def get_completions(self, document, complete_event):
        from prompt_toolkit.formatted_text import FormattedText

        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        word = text.strip()
        for opt in COMMAND_OPTIONS:
            # Show all commands when user types "/"
            command_text = opt["display"]
            if command_text.startswith(word) or word == "/":
                replacement = command_text + (" " if opt["meta"] else "")

                # Style: white command, dim description
                meta_parts = []
                if opt["meta"]:
                    meta_parts.append(("", opt["meta"] + "  "))
                meta_parts.append(("class:completion.meta", opt["description"]))

                yield Completion(
                    replacement,
                    start_position=-len(word),
                    display=FormattedText([("class:completion", command_text)]),
                    display_meta=FormattedText(meta_parts),
                    style="",  # No background for unselected
                    selected_style="fg:#00ffff",  # Cyan text when selected, no bg
                )


def _render_input_borders(top: bool = True, bottom: bool = True) -> None:
    """Legacy helper kept for compatibility (now renders nothing)."""
    return


def _render_status_line(context_used: int = 0, context_total: int = config.max_tokens) -> None:
    """Render status line showing context and commands hint."""
    total_tokens = max(1, context_total or config.max_tokens or 1)
    ratio = context_used / total_tokens
    context_percent = min(100, max(0, math.ceil(ratio * 100)))

    # Color based on usage
    if context_percent < 50:
        context_color = "green"
    elif context_percent < 80:
        context_color = "yellow"
    else:
        context_color = "red"

    status = (
        f"[dim]Context: [bold {context_color}]{context_percent}%[/bold {context_color}] "
        f"({context_used:,}/{context_total:,} tokens) • Type [cyan]/[/cyan] for commands[/dim]"
    )

    safe_console_print(status)


def _render_cancellation_hint() -> None:
    """Render hint about Ctrl+C to cancel during processing."""
    safe_console_print(
        "\n[dim]Press [bold red]Ctrl+C[/bold red] to cancel[/dim]\n"
    )


# Default generation model for code tasks
# Model options:
# - deepseek-chat: BEST VALUE - cheap, good function calling, 10-30x cheaper than OpenAI
# - qwen3-coder-480b-a35b-instruct: Best for agentic coding but slow/expensive
# - qwen3-235b: Complex reasoning, large context
# - llama-3.3-70b: Follows tool schemas well but struggles with multi-step fixes
GENERATION_MODEL = "deepseek-chat"
GENERATION_MODEL_DISPLAY = "deepseek"  # Short name for banner


def _render_banner(mock_enabled: bool, model_name: str, output_directory: Path, show_tip: bool) -> None:
    """Render the compact banner shown at startup (Claude Code style)."""
    # Style 4: Robot face with eyes
    logo = "[bold cyan]███████\n█ ◉ ◉ █\n▀▀███▀▀[/bold cyan]"

    # Show the generation model (coder model) - that's the one doing the real work
    model_display = GENERATION_MODEL_DISPLAY
    if mock_enabled:
        model_display += " [yellow](mock)[/yellow]"

    cwd_display = str(Path.cwd()).replace(str(Path.home()), "~")

    info_lines = [
        f"[bold cyan]NASEA[/bold cyan] [dim]v{__version__}[/dim]",
        f"[dim]{model_display}[/dim]",
        f"[dim]{cwd_display}[/dim]",
    ]

    # Create side-by-side layout: logo | info
    banner = Table.grid(padding=(0, 2), expand=False)
    banner.add_column(justify="left", no_wrap=True)  # Logo
    banner.add_column(justify="left", vertical="top")  # Info (top-aligned)

    banner.add_row(logo, "\n".join(info_lines))

    console.print()
    console.print(banner)
    console.print()
    sys.stdout.flush()

@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Use built-in mock LLM responses (offline demo mode)"
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="LLM model to use (kimi-k2, gpt-4-turbo, etc.)"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir", "-o",
        help="Where to save the generated project"
    ),
    project_name: Optional[str] = typer.Option(
        None,
        "--name", "-n",
        help="Project name (auto-generated if not provided)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    )
):
    """Interactive entry point when no subcommand is provided."""
    if ctx.invoked_subcommand is not None:
        return

    state = {
        "model": model or config.default_model,
        "mock": mock or getattr(config, "mock_mode", False),
        "output_dir": output_dir,
        "project_name": project_name,
        "verbose": verbose,
        "render_banner": True,
        "render_banner_tip": True,
        "chat_history": [],
        "pending_generation": None,
        "llm_clients": {},
        "last_project_path": None,
        "thinking_requested": False,
        "failure_streak": 0,
        "last_intent": None,
        "permission_mode": "ask",  # Persists "Always" choice across requests
    }
    _set_active_model(state, state["model"])

    use_prompt_toolkit = (
        HAVE_PROMPT_TOOLKIT
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    )

    if use_prompt_toolkit:
        _interactive_loop_prompt_toolkit(state)
    else:
        safe_console_print(
            "[yellow]Running in basic mode (no advanced command palette). "
            "If you're in a terminal, install prompt-toolkit>=3.0.39 for the full UX.[/yellow]"
        )
        _interactive_loop_basic(state)


def _confirm_generation(original_prompt: str, state: Dict[str, Any]) -> None:
    """Queue a build request and ask for confirmation on the next prompt."""
    state["pending_generation"] = {"prompt": original_prompt}
    safe_console_print(
        "[cyan]Build request queued.[/cyan] Type [green]y[/green] to confirm, [red]n[/red] to cancel, "
        "or append a stack hint, e.g. `y react`."
    )


def _interactive_loop_prompt_toolkit(state: Dict[str, Any]) -> None:
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style as PTKStyle
    try:
        from prompt_toolkit.shortcuts import CompleteStyle
    except ImportError:  # Older prompt_toolkit versions
        from prompt_toolkit.completion import CompleteStyle

    command_completer = CommandCompleter()

    state.setdefault(
        "context_limit",
        LLMFactory.get_context_window(state.get("model"), config.max_tokens)
    )
    state.setdefault("context_used", 0)

    # Create completion style once (doesn't change between loops)
    completion_style = PTKStyle.from_dict({
        'completion-menu': 'noinherit',
        'completion-menu.completion': 'noinherit',
        'completion-menu.completion.current': 'fg:#00ffff noinherit',
        'completion-menu.meta': 'fg:#aaaaaa noinherit',
        'completion-menu.multi-column-meta': 'fg:#aaaaaa noinherit',
        'scrollbar.background': 'noinherit',
        'scrollbar.button': 'noinherit',
    })

    # Create PromptSession ONCE for command history (Up/Down arrows)
    session = PromptSession(
        style=completion_style,
        history=FileHistory(str(Path.home() / ".nasea_history"))
    )
    state["prompt_session"] = session

    try:
        while True:
            if state.get("render_banner"):
                _render_banner(
                    mock_enabled=state["mock"],
                    model_name=state["model"],
                    output_directory=state["output_dir"] or config.output_dir,
                    show_tip=state.pop("render_banner_tip", False)
                )
                state["render_banner"] = False

            try:
                # Flush buffered input
                _flush_pending_prompt_input(state)
                time.sleep(0.05)

                # Use bordered input
                user_input = create_bordered_input(
                    state=state,
                    completer=command_completer,
                    history=session.history if hasattr(session, 'history') else None,
                )

                # === FIX LOGIC START ===

                # 1. Handle Ctrl+C (None) - exit immediately
                if user_input is None:
                    safe_console_print("[dim]Exiting NASEA. Goodbye![/dim]\n")
                    raise typer.Exit(0)

                # 2. Handle Empty Enter ("")
                user_input = user_input.strip()
                if not user_input:
                    continue

                # Show submitted input (white text, light grey bg)
                console.print(f"[cyan]>[/cyan] [white on #303030]{user_input}[/white on #303030]")
                console.print()  # Empty line for spacing

            except KeyboardInterrupt:
                # Ctrl+C exits immediately
                safe_console_print("[dim]Exiting NASEA. Goodbye![/dim]\n")
                raise typer.Exit(0)

            # Validator ensures we never get empty input, so no need to check

            if user_input == "/":
                selection = _show_command_palette(state)
                if selection is None:
                    continue
                if selection == "__build__":
                    build_session = PromptSession()
                    user_input = build_session.prompt("Describe project> ").strip()
                    if not user_input:
                        continue
                else:
                    _handle_interactive_command(selection, state)
                    continue

            if user_input.startswith("/"):
                _handle_interactive_command(user_input, state)
                continue

            _process_user_message(user_input, state)
            continue

    finally:
        state.pop("prompt_session", None)


def _flush_pending_prompt_input(state: Dict[str, Any]) -> None:
    """Clear any queued keystrokes typed while the app was busy (prompt_toolkit only)."""
    import sys
    import os

    # Method 1: Flush prompt_toolkit's input buffer
    session = state.get("prompt_session")
    if session:
        app = getattr(session, "app", None)
        if app:
            input_obj = getattr(app, "input", None)
            if input_obj:
                flush_fn = getattr(input_obj, "flush_keys", None) or getattr(input_obj, "flush", None)
                if flush_fn:
                    try:
                        flush_fn()
                    except Exception as exc:
                        logger.debug(f"Failed to flush prompt_toolkit input: {exc}")

    # Method 2: Aggressively drain stdin using select + read (Unix only)
    try:
        import termios
        import select
        import fcntl

        if sys.stdin.isatty():
            # First use tcflush to clear the terminal driver's input queue
            termios.tcflush(sys.stdin, termios.TCIFLUSH)

            # Then use select to check for any remaining buffered data and discard it
            while select.select([sys.stdin], [], [], 0.0)[0]:
                try:
                    old_flags = fcntl.fcntl(sys.stdin, fcntl.F_GETFL)
                    fcntl.fcntl(sys.stdin, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)
                    try:
                        sys.stdin.read(4096)
                    finally:
                        fcntl.fcntl(sys.stdin, fcntl.F_SETFL, old_flags)
                except (IOError, OSError):
                    break
    except (ImportError, AttributeError, OSError):
        # Not available on Windows
        pass


def _interactive_loop_basic(state: Dict[str, Any]) -> None:
    state.setdefault(
        "context_limit",
        LLMFactory.get_context_window(state.get("model"), config.max_tokens)
    )
    state.setdefault("context_used", 0)

    try:
        while True:
            if state.get("render_banner"):
                _render_banner(
                    mock_enabled=state["mock"],
                    model_name=state["model"],
                    output_directory=state["output_dir"] or config.output_dir,
                    show_tip=state.pop("render_banner_tip", False)
                )
                state["render_banner"] = False

            safe_console_print()
            _render_status_line(
                state.get("context_used", 0),
                state.get("context_limit") or config.max_tokens
            )

            try:
                user_input = input("> ").strip()
            except EOFError:
                safe_console_print("\n[dim]EOF reached. Exiting.[/dim]")
                raise typer.Exit(0)
            except KeyboardInterrupt:
                # Ctrl+C exits immediately
                safe_console_print("[dim]Exiting NASEA. Goodbye![/dim]\n")
                raise typer.Exit(0)

            if not user_input:
                continue

            if user_input == "/":
                selection = _show_command_palette(state)
                if selection is None:
                    continue
                if selection == "__build__":
                    user_input = input("Describe project> ").strip()
                    if not user_input:
                        continue
                else:
                    _handle_interactive_command(selection, state)
                    continue

            if user_input.startswith("/"):
                _handle_interactive_command(user_input, state)
                continue

            _process_user_message(user_input, state)
            continue
    except KeyboardInterrupt:
        safe_console_print("[dim]Exiting NASEA. Goodbye![/dim]\n")
        raise typer.Exit(0)


def _execute_generation_with_tools(
    prompt: str,
    output_dir: Optional[Path],
    project_name: Optional[str],
    model: Optional[str],
    mock: bool
):
    """
    Execute generation using streaming with tool calls (Claude Code style).

    Args:
        prompt: User's generation request
        output_dir: Output directory for project
        project_name: Project name
        model: Model to use
        mock: Whether to use mock mode
    """
    from pathlib import Path
    from nasea.core.tool_definitions import GENERATION_TOOLS
    from nasea.core.streaming_tool_handler import StreamingToolHandler, ToolUseConversation
    from nasea.llm.llm_factory import LLMFactory

    # Determine project directory
    if not project_name:
        # Generate project name from prompt
        import re
        words = re.findall(r'\b\w+\b', prompt.lower())
        project_name = '-'.join(words[:3]) if words else 'project'

    project_root = Path(output_dir or config.output_dir) / project_name
    project_root.mkdir(parents=True, exist_ok=True)

    safe_console_print(f"\n[cyan]Creating project in:[/cyan] [bold]{project_root}[/bold]\n")

    # Create LLM client
    if mock:
        safe_console_print("[yellow]Mock mode not supported with tool-based generation[/yellow]")
        safe_console_print("[dim]Falling back to live API...[/dim]\n")

    try:
        client = LLMFactory.create_client(
            model=model or config.default_model,
            config=config
        )
    except Exception as e:
        safe_console_print(f"[red]Failed to create LLM client: {e}[/red]")
        raise typer.Exit(1)

    # Create streaming tool handler
    handler = StreamingToolHandler(
        console=console,
        project_root=project_root,
        permission_mode="ask",  # Can be configured
        loader_type="create"
    )

    # Create conversation manager
    conversation = ToolUseConversation(
        client=client.client if hasattr(client, 'client') else client,
        handler=handler,
        console=console,
        model=model or config.default_model,
        tools=GENERATION_TOOLS
    )

    # Load project memory (NASEA.md/CLAUDE.md) if it exists
    project_memory = _load_project_memory(project_root)

    # System prompt
    system_prompt = f"""You are an expert software engineer helping to create a project.

You have access to tools to create files, edit files, delete/rename paths, run commands, and more. Use these tools step-by-step to build the requested project.

# TOOL USAGE RULES
**write_file**: Use to CREATE new files OR completely OVERWRITE existing files with new content.
  - Can be used multiple times on the same file (it will overwrite)
  - Use when you want to replace entire file content

**edit_file**: Use to MODIFY specific parts of an existing file.
  - Replaces old_content with new_content
  - Use for surgical changes to existing files

**General principle**: write_file is simpler and works for both new files and complete rewrites.

Guidelines:
1. Start with list_files() to understand the workspace, and treat that listing as ground truth.
2. Create necessary files (package.json, README, etc.) with write_file.
3. Write clean, well-documented code with proper validation.
4. Use rename_path/delete_path to fix wrong filenames instead of duplicating files.
5. Add comments explaining key sections when helpful.
6. When done, call complete_generation with a summary.

Current request: {prompt}
Project directory: {project_root}{project_memory}

Work methodically and call complete_generation when finished."""

    # Run conversation
    try:
        result = conversation.run(
            system_message=system_prompt,
            user_message=prompt
        )

        # Display final summary
        safe_console_print("\n" + "=" * 70)
        safe_console_print("\n[bold green]✅ Generation Complete![/bold green]\n")

        if result.get("summary"):
            safe_console_print(f"[cyan]Summary:[/cyan] {result['summary']}")

        if result.get("next_steps"):
            safe_console_print(f"\n[cyan]Next steps:[/cyan]")
            safe_console_print(f"  {result['next_steps']}")

        safe_console_print(f"\n[cyan]Project location:[/cyan] [bold]{project_root}[/bold]")
        safe_console_print()

    except KeyboardInterrupt:
        safe_console_print("\n\n[yellow]Generation cancelled by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        safe_console_print(f"\n[red]Error during generation: {e}[/red]")
        raise typer.Exit(1)

    return project_root


def _execute_generation(
    prompt: str,
    output_dir: Optional[Path],
    project_name: Optional[str],
    model: Optional[str],
    verbose: bool,
    mock: bool,
    stream: bool = False,
    show_banner: bool = True
):
    """Shared implementation used by both the CLI command and interactive entry."""
    # Update model configuration first so the banner reflects it
    if model:
        config.default_model = model
    model_name = config.default_model

    # Enable mock mode if requested
    if mock:
        config.mock_mode = True
    mock_enabled = bool(config.mock_mode)

    if mock:
        config.auto_generate_tests = False
        config.run_static_analysis = False
        config.run_security_scan = False
        config.enable_caching = False

    if show_banner:
        _render_banner(
            mock_enabled=mock_enabled,
            model_name=model_name,
            output_directory=output_dir or config.output_dir,
            show_tip=False
        )

    if mock and show_banner:
        safe_console_print("\n[yellow]Mock mode enabled – responses will be simulated[/yellow]\n")

    try:
        # Use streaming mode if enabled
        if stream:
            from nasea.core.streaming_control import StreamingControlUnit

            streaming_control = StreamingControlUnit(console=console)
            result = streaming_control.render_streaming_progress(
                prompt=prompt,
                project_name=project_name
            )
        else:
            # Traditional mode with rich live plan
            status = StatusReporter(console=console, verbose=verbose)
            control = ControlUnit(status_reporter=status)

            with Live(status.board.render(), console=console, refresh_per_second=8, transient=False) as live:
                status.attach_live(live)
                result = control.generate_project(
                    user_prompt=prompt,
                    output_dir=output_dir,
                    project_name=project_name
                )

        # Display results
        if result.success:
            safe_console_print("\n[bold green]✓ Project generated successfully![/bold green]\n")

            # Create results table
            table = Table(show_header=False, box=None)
            table.add_row("[cyan]Project Path:[/cyan]", str(result.project_path))
            table.add_row("[cyan]Files Generated:[/cyan]", str(result.files_generated))
            table.add_row("[cyan]Tests:[/cyan]", f"{result.tests_passed}/{result.tests_total} passed")
            table.add_row("[cyan]Iterations:[/cyan]", str(result.iterations))

            safe_console_print(table)

            if result.warnings:
                safe_console_print(f"\n[yellow]Warnings:[/yellow]")
                for warning in result.warnings:
                    safe_console_print(f"  - {warning}")

            safe_console_print(f"\n[dim]You can now navigate to the project and start using it:[/dim]")
            safe_console_print(f"[bold]  cd {result.project_path}[/bold]")

        else:
            safe_console_print(f"\n[bold red]✗ Project generation failed[/bold red]\n")
            safe_console_print(f"[red]Error: {result.error}[/red]")
            raise typer.Exit(code=1)

    except KeyboardInterrupt:
        safe_console_print("\n\n[yellow]Generation cancelled by user[/yellow]")
        raise typer.Exit(code=130)

    except Exception as e:
        safe_console_print(f"\n[bold red]✗ Unexpected error:[/bold red] {e}")
        if verbose:
            logger.exception("Full error trace:")
        raise typer.Exit(code=1)


def _render_command_table(command_options: List[Dict[str, str]] = None, selected_index: int = 0) -> Panel:
    if command_options is None:
        command_options = COMMAND_OPTIONS

    table = Table.grid(expand=True)
    table.add_column(justify="left", style="cyan", no_wrap=True)
    table.add_column(justify="left", style="dim", no_wrap=True)
    table.add_column(justify="left", style="white")

    for idx, opt in enumerate(command_options):
        command_label = opt["display"]
        meta = opt["meta"]
        description = opt["description"]

        if idx == selected_index:
            command_label = f"[bold cyan]{command_label}[/bold cyan]"
            if meta:
                meta = f"[bold magenta]{meta}[/bold magenta]"
            description = f"[white]{description}[/white]"
        else:
            if meta:
                meta = f"[magenta]{meta}[/magenta]"

        table.add_row(command_label, meta or "", description)

    header = Table.grid(expand=True)
    header.add_column(justify="left", style="dim")
    header.add_row("Type a command or pick from the list:")

    group = Group(header, table)
    return Panel(group, border_style="cyan", title="Command Palette", expand=True)


def _show_command_palette(state: Dict[str, Any]) -> Optional[str]:
    """Display an interactive menu for quick commands."""
    # Get dynamic commands based on current mode
    command_options = _get_command_options(mock_mode=state.get("mock", False))

    safe_console_print(_render_command_table(command_options))
    safe_console_print(
        "[dim]Enter the command number, type it manually (e.g. /mock), or press Enter to cancel.[/dim]"
    )

    prompt_func = PromptSession().prompt if HAVE_PROMPT_TOOLKIT and PromptSession else input

    try:
        raw_choice = prompt_func("Command> ").strip()
    except (KeyboardInterrupt, EOFError):
        safe_console_print("\n[dim]Cancelled command palette.[/dim]")
        return None

    if not raw_choice:
        return None

    selection: Optional[str] = None

    if raw_choice.isdigit() and 1 <= int(raw_choice) <= len(command_options):
        selection = command_options[int(raw_choice) - 1]["value"]
    elif raw_choice.startswith("/"):
        for opt in command_options:
            if raw_choice == opt["display"]:
                selection = opt["value"]
                break
        else:
            selection = raw_choice
    else:
        safe_console_print("[red]Invalid selection.[/red]")
        return None

    if selection == "__build__":
        return selection

    if selection == "__model__":
        models = [
            ("1", "deepseek-chat", "DeepSeek V3 - Cheapest, good quality"),
            ("2", "deepseek-reasoner", "DeepSeek R1 - Advanced reasoning"),
            ("3", "qwen3-235b", "Venice Qwen3 - Large context, complex tasks"),
            ("4", "venice-uncensored", "Venice Uncensored - No filters"),
            ("5", "llama-3.3-70b", "Venice Llama - Good for tools"),
            ("6", "qwen3-coder-480b-a35b-instruct", "Venice Qwen Coder - Agentic coding"),
        ]

        safe_console_print("\n[bold]Available Models[/bold]")
        for idx, model_key, description in models:
            safe_console_print(f"  {idx}. [cyan]{model_key}[/cyan] - {description}")
        safe_console_print("  [magenta]custom[/magenta] - Enter any model name")

        try:
            model_choice_input = prompt_func("Model (number/custom)> ").strip()
        except (KeyboardInterrupt, EOFError):
            safe_console_print("\n[dim]Cancelled model change.[/dim]")
            return None

        if not model_choice_input:
            safe_console_print("[yellow]Model unchanged.[/yellow]")
            return None

        if model_choice_input.isdigit() and 1 <= int(model_choice_input) <= len(models):
            _, model_value, _ = models[int(model_choice_input) - 1]
            _set_active_model(state, model_value)
        else:
            _set_active_model(state, model_choice_input)

        state["render_banner"] = True
        state["render_banner_tip"] = False
        safe_console_print(f"[green]Model set to[/green] {state['model']}")
        safe_console_print()  # Add spacing after palette
        return None

    safe_console_print()  # Add spacing after palette
    return selection


def _handle_interactive_command(command: str, state: Dict[str, Any]) -> None:
    """Process slash commands in interactive mode."""
    tokens = command[1:].strip().split()
    if not tokens:
        return

    cmd = tokens[0].lower()
    args = tokens[1:]

    if cmd in {"exit", "quit"}:
        safe_console_print("[dim]Exiting NASEA. Goodbye![/dim]\n")
        raise typer.Exit(0)

    if cmd in {"help", "?"}:
        safe_console_print()
        safe_console_print(_render_command_table())
        return

    if cmd == "model":
        if not args:
            session = PromptSession()
            try:
                new_model = session.prompt(
                    f"Model name (current: {state['model']})> "
                ).strip()
            except KeyboardInterrupt:
                safe_console_print("\n[dim]Cancelled model change.[/dim]")
                return
            if not new_model:
                safe_console_print("[yellow]Model unchanged.[/yellow]")
                return
            _set_active_model(state, new_model)
            state["render_banner"] = True
            state["render_banner_tip"] = False
            safe_console_print(f"[green]Model set to[/green] {new_model}")
            return
        new_model = args[0]
        _set_active_model(state, new_model)
        state["render_banner"] = True
        state["render_banner_tip"] = False
        safe_console_print(f"[green]Model set to[/green] {new_model}")
        return

    if cmd == "mock":
        state["mock"] = True
        state["render_banner"] = True
        state["render_banner_tip"] = False
        safe_console_print("[yellow]Mock mode enabled.[/yellow]\n")
        return

    if cmd == "live":
        state["mock"] = False
        state["render_banner"] = True
        state["render_banner_tip"] = False
        safe_console_print("[green]Live mode enabled. Ensure API keys are configured.[/green]\n")
        return

    if cmd == "clear":
        console.clear()
        state["render_banner"] = True
        state["render_banner_tip"] = False
        # Clear chat history / memory
        state["chat_history"] = []
        state["context_used"] = 0
        safe_console_print("[dim]Chat history cleared.[/dim]")
        return

    if cmd == "build":
        safe_console_print("[dim]Type your project description directly, or press / to open the command palette.[/dim]\n")
        return

    if cmd in {"continue", "c"}:
        # Continue the last task if it was interrupted or incomplete
        last_project = state.get("last_project_path")
        if not last_project:
            safe_console_print("[yellow]No previous task to continue. Start with a new request.[/yellow]\n")
            return
        # Build context from recent chat history
        chat_history = state.get("chat_history", [])
        last_user_task = None
        last_operations = None
        for msg in reversed(chat_history):
            if msg.get("role") == "user" and not last_user_task:
                last_user_task = msg.get("content", "")
            if msg.get("role") == "assistant" and "[Operations:" in msg.get("content", ""):
                last_operations = msg.get("content", "")
                break
        # Send a continue prompt with context
        continue_prompt = "Continue from where you stopped."
        if last_operations:
            continue_prompt += f"\n\nPrevious progress:\n{last_operations}"
        if last_user_task:
            continue_prompt += f"\n\nOriginal task: {last_user_task}"
        last_tool_context = state.get("last_tool_context")
        if last_tool_context:
            continue_prompt += (
                "\n\nTool context (avoid redoing; avoid full-file re-reads):\n"
                f"{last_tool_context}\n"
                "\nEfficiency reminder: use grep_search then read_file with offset+limit."
            )
        last_findings = state.get("last_findings")
        if last_findings:
            continue_prompt += (
                "\n\nFindings so far (do not re-discover):\n"
                f"{last_findings}\n"
            )
        # Skip exploration on /continue to avoid repeated "Found X files" banners and token waste.
        previous_skip = state.get("skip_exploration")
        state["skip_exploration"] = True
        try:
            _handle_project_edit(state, continue_prompt, mode="edit")
        finally:
            if previous_skip is None:
                state.pop("skip_exploration", None)
            else:
                state["skip_exploration"] = previous_skip
        return

    if cmd == "memory":
        # Show saved memories from current working directory
        import json
        memory_file = Path.cwd() / ".nasea_memory.json"
        if memory_file.exists():
            try:
                mem_data = json.loads(memory_file.read_text())
                entries = mem_data.get("entries", {})
                thoughts = mem_data.get("thoughts", [])
                if entries or thoughts:
                    safe_console_print("\n[bold cyan]Saved Memories:[/bold cyan]")
                    for key, val in entries.items():
                        preview = val.get("value", "")[:80]
                        category = val.get("category", "general")
                        safe_console_print(f"  [cyan]{key}[/cyan] ({category}): {preview}...")
                    if thoughts:
                        safe_console_print(f"\n[dim]{len(thoughts)} thinking steps recorded[/dim]")
                    safe_console_print()
                else:
                    safe_console_print("[dim]No memories saved yet.[/dim]\n")
            except Exception:
                safe_console_print("[dim]No memories saved yet.[/dim]\n")
        else:
            safe_console_print("[dim]No memories saved yet. Use memory_save tool to persist context.[/dim]\n")
        return

    safe_console_print(f"[red]Unknown command:[/red] {command}. Type /help for options.\n")


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="What to build (natural language description)"),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir", "-o",
        help="Where to save the generated project"
    ),
    project_name: Optional[str] = typer.Option(
        None,
        "--name", "-n",
        help="Project name (auto-generated if not provided)"
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="LLM model to use (kimi-k2, gpt-4-turbo, etc.)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Use built-in mock LLM responses (offline demo mode)"
    ),
    stream: bool = typer.Option(
        True,
        "--stream/--no-stream",
        help="Enable real-time streaming progress (like Claude Code CLI)"
    )
    ):
    """
    Generate a software project from a natural language prompt.

    Examples:
        nasea generate "Create a REST API for managing books"
        nasea generate "Build a CLI calculator" --name my-calculator
        nasea generate "Create a web scraper" --model kimi-k2 --verbose
        nasea generate "Build an API" --stream  # Real-time progress (default)
        nasea generate "Build an API" --no-stream  # Traditional mode
    """
    if stream:
        # Use new tool-based streaming (Claude Code style)
        _execute_generation_with_tools(
            prompt=prompt,
            output_dir=output_dir,
            project_name=project_name,
            model=model,
            mock=mock
        )
    else:
        # Use traditional generation
        _execute_generation(
            prompt=prompt,
            output_dir=output_dir,
            project_name=project_name,
            model=model,
            verbose=verbose,
            mock=mock,
            stream=False,
            show_banner=True
        )


@app.command()
def info():
    """Show NASEA configuration and status."""
    safe_console_print(Panel.fit(
        f"[bold cyan]NASEA[/bold cyan] v{__version__}",
        border_style="cyan"
    ))

    # Configuration info
    table = Table(title="Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Default Model", config.default_model)
    table.add_row("Fallback Model", config.fallback_model)
    table.add_row("Temperature", str(config.temperature))
    table.add_row("Max Iterations", str(config.max_iterations))
    table.add_row("Output Directory", str(config.output_dir))
    table.add_row("Mock Mode", "✓" if getattr(config, "mock_mode", False) else "✗")
    table.add_row("Auto Generate Tests", "✓" if config.auto_generate_tests else "✗")
    table.add_row("Static Analysis", "✓" if config.run_static_analysis else "✗")
    table.add_row("Security Scan", "✓" if config.run_security_scan else "✗")

    safe_console_print(table)

    # API status
    safe_console_print("\n[bold]API Keys Status:[/bold]")
    if config.venice_api_key:
        safe_console_print("  [green]✓[/green] Venice AI (uncensored)")
    else:
        safe_console_print("  [red]✗[/red] Venice AI (not configured)")

    if config.kimi_api_key:
        safe_console_print("  [green]✓[/green] Kimi K2")
    else:
        safe_console_print("  [red]✗[/red] Kimi K2 (not configured)")

    if config.openai_api_key:
        safe_console_print("  [green]✓[/green] OpenAI")
    else:
        safe_console_print("  [red]✗[/red] OpenAI (not configured)")

    if not config.venice_api_key and not config.kimi_api_key and not config.openai_api_key:
        safe_console_print("\n[yellow]Warning: No API keys configured![/yellow]")
        safe_console_print("[dim]Set VENICE_API_KEY, KIMI_API_KEY, or OPENAI_API_KEY in your .env file[/dim]")


@app.command()
def examples():
    """Show example prompts to try."""
    safe_console_print(Panel.fit(
        "[bold cyan]Example Prompts[/bold cyan]",
        border_style="cyan"
    ))

    examples_list = [
        ("Simple", [
            "Create a Python function that calculates Fibonacci numbers",
            "Build a CLI tool that converts CSV to JSON",
            "Create a password generator with customizable length"
        ]),
        ("Medium", [
            "Create a REST API for managing books with CRUD operations",
            "Build a Flask API with user authentication and JWT tokens",
            "Create a web scraper that extracts product data from e-commerce sites"
        ]),
        ("Complex", [
            "Create a full-stack todo application with React frontend and FastAPI backend",
            "Build a microservice architecture for user management with Docker deployment",
            "Create a data pipeline that processes CSV files and generates analytics reports"
        ])
    ]

    for difficulty, prompts in examples_list:
        safe_console_print(f"\n[bold]{difficulty} Examples:[/bold]")
        for prompt in prompts:
            safe_console_print(f"  • {prompt}")

    safe_console_print("\n[dim]Try: nasea generate \"<your prompt here>\"[/dim]")


@app.command()
def version():
    """Show NASEA version."""
    safe_console_print(f"NASEA version [bold cyan]{__version__}[/bold cyan]")


@app.command()
def setup():
    """Interactive setup wizard."""
    safe_console_print(Panel.fit(
        "[bold cyan]NASEA Setup Wizard[/bold cyan]",
        border_style="cyan"
    ))

    safe_console_print("\nThis wizard will help you configure NASEA.\n")

    # Check for .env file
    env_file = Path(".env")
    if env_file.exists():
        safe_console_print("[yellow]Found existing .env file[/yellow]")
        if not typer.confirm("Do you want to overwrite it?"):
            safe_console_print("Setup cancelled")
            return

    # Gather information
    safe_console_print("\n[bold]API Configuration:[/bold]")
    safe_console_print("You need at least one API key to use NASEA.\n")

    kimi_key = typer.prompt("Kimi K2 API Key (press Enter to skip)", default="", show_default=False)
    openai_key = typer.prompt("OpenAI API Key (press Enter to skip)", default="", show_default=False)

    if not kimi_key and not openai_key:
        safe_console_print("\n[red]Error: At least one API key is required[/red]")
        raise typer.Exit(code=1)

    # Model selection
    safe_console_print("\n[bold]Model Selection:[/bold]")
    model_options = ["kimi-k2", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
    default_model = typer.prompt(
        "Default model",
        type=typer.Choice(model_options),
        default="kimi-k2" if kimi_key else "gpt-4-turbo"
    )

    # Write .env file
    env_content = f"""# NASEA Configuration
# Auto-generated by setup wizard

# API Keys
KIMI_API_KEY={kimi_key}
OPENAI_API_KEY={openai_key}

# Model Configuration
DEFAULT_MODEL={default_model}
FALLBACK_MODEL=gpt-4-turbo
TEMPERATURE=0.7
MAX_TOKENS=4096

# Generation Settings
MAX_ITERATIONS=3
OUTPUT_DIR=./output

# Testing & Verification
AUTO_GENERATE_TESTS=true
RUN_STATIC_ANALYSIS=true
RUN_SECURITY_SCAN=true

# Logging
LOG_LEVEL=INFO
CONSOLE_LOGGING=true

# Caching
ENABLE_CACHING=true
"""

    env_file.write_text(env_content)

    safe_console_print("\n[bold green]✓ Setup complete![/bold green]")
    safe_console_print(f"\nConfiguration saved to: [cyan]{env_file.absolute()}[/cyan]")
    safe_console_print("\nYou can now start using NASEA:")
    safe_console_print('[bold]  nasea generate "Create a simple calculator"[/bold]')


def run():
    """Main entry point."""
    app()


def main():
    """Console script entry point (typer app shim)."""
    run()

if __name__ == "__main__":
    run()

def _emit_small_reply(state: dict, message: str) -> None:
    if not message:
        return

    import re

    # Safety: strip any control tokens that might have leaked through
    message = re.sub(r'\s*\[NASEA_(?:CREATE|EDIT|DEBUG|TEST|LIST|VIEW|EXPLAIN|RUN)\](?:\s*\[project_name=[a-z0-9-]+\])?\s*', '', message).strip()

    if not message:
        return

    # Check for uncensored content tags
    uncensored_match = re.search(r'<uncensored\s+type="([^"]+)">(.*?)</uncensored>', message, flags=re.DOTALL)

    if uncensored_match:
        content_type = uncensored_match.group(1)
        uncensored_content = uncensored_match.group(2).strip()

        # Style based on content type (only for truly sensitive content)
        type_styles = {
            "explicit": ("red", "⚠ Explicit"),
            "dangerous": ("red", "⚠ Dangerous"),
            "unethical": ("yellow", "⚠ Unethical"),
        }
        color, label = type_styles.get(content_type, ("yellow", "⚠ Sensitive"))

        # Display with styled label, then normal content
        console.print(f"[{color}]{label}[/{color}]")
        console.print()  # Line space after label
        console.print(f"[white]⏺[/white] {uncensored_content}")
        console.print()
        sys.stdout.flush()

        # Strip uncensored content from chat history (don't pollute big model context)
        # Just save a placeholder so conversation flow makes sense
        state["chat_history"].append({
            "role": "assistant",
            "content": f"[Responded to off-topic question - {content_type}]"
        })
        return

    # Use white dot for AI responses (green is for successful tool calls)
    console.print("[white]⏺[/white] ", end="")

    # Simple approach: Just print with proper wrapping, handle markdown manually
    # This avoids Rich's Markdown width limitations
    text = message

    # Handle code blocks (```...```) - indent and style them
    def format_code_block(match):
        code = match.group(2)
        # Indent each line and make it dim
        indented = '\n'.join('    ' + line for line in code.split('\n'))
        return f'\n[dim]{indented}[/dim]\n'

    text = re.sub(r'```(\w*)\n(.*?)```', format_code_block, text, flags=re.DOTALL)

    # Convert headers (### Header) to bold
    text = re.sub(r'^### (.+)$', r'\n[bold cyan]\1[/bold cyan]', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'\n[bold cyan]\1[/bold cyan]', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'\n[bold cyan]\1[/bold cyan]', text, flags=re.MULTILINE)

    # Convert **bold** to Rich markup [bold]text[/bold]
    text = re.sub(r'\*\*(.+?)\*\*', r'[bold]\1[/bold]', text)

    # Convert `code` to Rich markup [cyan]code[/cyan] (but not if already converted)
    text = re.sub(r'`([^`]+?)`', r'[cyan]\1[/cyan]', text)

    # Print with soft wrapping enabled
    console.print(text, soft_wrap=True, overflow="fold", no_wrap=False)
    console.print()  # Add newline after
    sys.stdout.flush()

    cleaned = _strip_thinking_blocks(message)
    state["chat_history"].append({"role": "assistant", "content": cleaned})
    # Try compression (without LLM for router - uses simple summary fallback)
    _compress_chat_history(state)
    _compact_chat_history(state)
    _recalculate_context_usage(state)
