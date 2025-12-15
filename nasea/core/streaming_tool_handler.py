"""
Streaming Tool Handler
Handles streaming API responses with tool calls, displaying them in Claude Code style.
"""

import json
import time
import re
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator, Set, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
import logging

from nasea.core.tool_executors import ToolExecutor
from nasea.core.dynamic_loader import get_loader
from nasea.llm.venice_client import retry_with_backoff
from nasea.core.tool_definitions import PARALLEL_SAFE_TOOLS

logger = logging.getLogger(__name__)


class StreamingToolHandler:
    """
    Handle streaming API responses with tool use.
    Displays tool calls in Claude Code style and manages permissions.
    """

    def __init__(
        self,
        console: Console,
        project_root: Path,
        permission_mode: str = "ask",
        loader_type: str = "create"
    ):
        """
        Initialize streaming tool handler.

        Args:
            console: Rich console for output
            project_root: Root directory for project generation
            permission_mode: 'ask', 'always', or 'never'
            loader_type: Type of loader to show (create, edit, debug, test)
        """
        self.console = console
        self.executor = ToolExecutor(project_root)
        self.permission_mode = permission_mode
        self.loader_type = loader_type

        # Track tool calls during streaming
        self.pending_tool_calls: List[Dict[str, Any]] = []
        self.tool_call_buffer: Dict[int, Dict[str, Any]] = {}

        # Track text output
        self.text_buffer = ""
        self.tracked_paths: Set[str] = set()
        self.directory_snapshots: Dict[str, Set[str]] = {}
        self.read_cache: Dict[str, Dict[str, Any]] = {}
        self._has_displayed_content = False

        # Real-time thinking block filter state
        self.in_thinking_block = False
        self.thinking_buffer = ""

        # Line buffer for markdown formatting (needs complete lines)
        self.line_buffer = ""

        # Track work done for summary
        self.files_created: List[str] = []
        self.files_updated: List[str] = []
        self.files_deleted: List[str] = []
        self.commands_run: List[str] = []
        self.files_read: List[Dict[str, Any]] = []
        self.searches_run: List[Dict[str, Any]] = []
        self.syntax_checked: List[Dict[str, Any]] = []
        self.browsers_opened: List[Dict[str, Any]] = []
        self._mutation_counter: int = 0
        self._read_history: List[Dict[str, Any]] = []
        self.user_cancelled: bool = False  # Track if user cancelled an operation
        self.on_first_output: Optional[Callable[[], None]] = None
        self.on_first_tool: Optional[Callable[[], None]] = None
        self.current_loader = None
        self.loader_idle_message = "Working..."

        # Parallel execution settings
        self.enable_parallel = True  # Enable parallel tool execution
        self.max_parallel_workers = 4  # Max concurrent tool executions

    def _notify_first_output(self) -> None:
        """Invoke the first-output callback (used to stop loaders gracefully)."""
        if self.on_first_output is None:
            return

        callback = self.on_first_output
        self.on_first_output = None

        try:
            callback()
        except Exception as exc:
            logger.debug(f"First-output callback failed: {exc}")

    def _notify_first_tool(self) -> None:
        """Invoke the first-tool callback (used to stop loaders before tool output)."""
        if self.on_first_tool is None:
            return

        callback = self.on_first_tool
        self.on_first_tool = None

        try:
            callback()
        except Exception as exc:
            logger.debug(f"First-tool callback failed: {exc}")

    def _set_loader_message(self, message: str) -> None:
        """Safely update the loader message if a loader is active."""
        if self.current_loader:
            self.current_loader.update_message(message)

    def _format_tool_loader_message(self, function_name: str, args: Dict[str, Any]) -> str:
        """Generate contextual loader text based on the tool and arguments."""
        # Handle case where model sends list instead of dict (some models do this)
        if isinstance(args, list):
            args = args[0] if args else {}
        if not isinstance(args, dict):
            args = {}
        path = args.get("file_path") or args.get("directory_path") or args.get("path") or ""
        if function_name == "write_file" and path:
            return f"Writing {path}..."
        if function_name == "edit_file" and path:
            return f"Editing {path}..."
        if function_name == "read_file" and path:
            return f"Reading {path}"
        if function_name == "list_files":
            directory = args.get("directory") or "."
            return f"Listing {directory}"
        if function_name == "create_directory" and path:
            return f"Creating dir {path}"
        if function_name == "delete_path" and path:
            return f"Deleting {path}"
        if function_name == "rename_path":
            source = args.get("source_path")
            target = args.get("target_path")
            if source and target:
                return f"Renaming {source} → {target}"
        if function_name == "run_command":
            cmd = args.get("command")
            if cmd:
                return f"Running: {cmd[:40]}{'…' if len(cmd) > 40 else ''}"
        if function_name == "complete_generation":
            return "Wrapping up project..."
        if function_name == "web_search":
            q = args.get("query", "")
            return f"Searching web for: {q[:20]}..."
        if function_name == "think":
            return "Planning..."
        if function_name == "grep_search":
            pattern = args.get("pattern", "")
            return f"Searching for: {pattern[:20]}..."
        if function_name == "memory_save":
            return "Saving to memory..."
        if function_name == "memory_read":
            return "Reading memory..."
        if function_name == "check_syntax":
            file_path = args.get("file_path", "")
            return f"Checking syntax: {file_path}" if file_path else "Checking syntax..."
        if function_name == "open_browser":
            path = args.get("path", "")
            return f"Opening in browser: {path[:30]}..." if path else "Opening browser..."
        return self.loader_idle_message

    def _generate_error_hint(self, error: str, tool_name: str) -> str:
        """Generate specific hint based on error message and tool."""
        error_lower = error.lower()

        # Missing argument hints
        if "missing required argument" in error_lower:
            if "file_path" in error_lower:
                return f"You must provide file_path argument. Example: {tool_name}(file_path=\"index.html\")"
            if "old_content" in error_lower or "new_content" in error_lower:
                return "edit_file needs: file_path, old_content, new_content. Read the file first to get exact content."
            if "content" in error_lower:
                return "write_file needs: file_path, content. Provide the full file content."
            if "summary" in error_lower:
                return "complete_generation needs: summary. Describe what you did."
            return f"Check required arguments for {tool_name}"

        # File not found
        if "not found" in error_lower or "no such file" in error_lower:
            return "File doesn't exist. Use list_files() to see available files."

        # Content not found (edit_file)
        if "content not found" in error_lower:
            return "Content mismatch - retrying..."

        # Path escapes project (trying to operate outside project folder)
        if "escapes" in error_lower or "must be inside" in error_lower:
            return "Can't operate outside project folder"

        # Directory vs file
        if "is a directory" in error_lower or "not a file" in error_lower:
            return "You passed a directory but need a file path. Add the filename (e.g., 'index.html' not just the folder)."

        # Permission/syntax
        if "syntax error" in error_lower:
            return "Your edit introduced a syntax error. Read the file again and fix it."

        # check_syntax specific
        if tool_name == "check_syntax" and "file_path" in error_lower:
            return "check_syntax needs a FILE path like 'index.html', not a directory. Use list_files() to see files."

        # Cannot complete due to errors
        if "cannot complete" in error_lower and "syntax" in error_lower:
            return "Fix ALL syntax errors shown above before calling complete_generation. Read the file and edit each error."

        return f"Try a different approach for {tool_name}"

    def _format_markdown(self, text: str) -> str:
        """
        Apply markdown formatting to text for better readability.
        Converts markdown syntax to Rich markup.
        """
        # Handle code blocks (```...```) - indent and style them
        def format_code_block(match):
            code = match.group(2)
            indented = '\n'.join('    ' + line for line in code.split('\n'))
            return f'\n[dim]{indented}[/dim]\n'

        text = re.sub(r'```(\w*)\n(.*?)```', format_code_block, text, flags=re.DOTALL)

        # Convert headers to bold cyan (remove # symbols)
        text = re.sub(r'^###\s+(.+)$', r'\n[bold cyan]\1[/bold cyan]', text, flags=re.MULTILINE)
        text = re.sub(r'^##\s+(.+)$', r'\n[bold cyan]\1[/bold cyan]', text, flags=re.MULTILINE)
        text = re.sub(r'^#\s+(.+)$', r'\n[bold cyan]\1[/bold cyan]', text, flags=re.MULTILINE)

        # Convert **bold** to Rich markup
        text = re.sub(r'\*\*(.+?)\*\*', r'[bold]\1[/bold]', text)

        # Convert *italic* to Rich markup (single asterisks, not double)
        text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'[italic]\1[/italic]', text)

        # Convert `code` to Rich markup
        text = re.sub(r'`([^`]+?)`', r'[cyan]\1[/cyan]', text)

        # Style bullet points
        text = re.sub(r'^(\s*)-\s+', r'\1• ', text, flags=re.MULTILINE)

        return text

    def process_stream(
        self,
        stream,
        messages: List[Dict[str, Any]]
    ) -> tuple[str, List[Dict[str, Any]], bool]:
        """
        Process streaming response with tool calls.

        Displays text in real-time as it streams, filtering thinking blocks dynamically.

        Args:
            stream: Streaming response from API
            messages: Conversation history

        Returns:
            Tuple of (assistant_message, tool_results, is_complete)
        """
        self.pending_tool_calls = []
        self.tool_call_buffer = {}
        self.text_buffer = ""
        self.in_thinking_block = False
        self.thinking_buffer = ""
        self.line_buffer = ""
        self._has_displayed_content = False
        if self.current_loader:
            # Reset loader to idle message at start of stream
            self.loader_idle_message = getattr(self.current_loader, "messages", ["Working..."])[0]
            self._set_loader_message(self.loader_idle_message)

        last_chunk = None
        import sys

        interrupted = False
        try:
            # Process streaming chunks - display text in real-time
            for chunk in stream:
                # Check if user pressed Esc to interrupt - stop immediately
                if self.current_loader and self.current_loader.was_interrupted:
                    interrupted = True
                    break
                last_chunk = chunk
                self._process_chunk(chunk)
        finally:
            # Always close stream to stop server-side generation
            if hasattr(stream, 'close'):
                stream.close()
            # Only flush if not interrupted (don't show partial output)
            if not interrupted:
                self._flush_line_buffer()
                # Add spacing after agent output
                if self._has_displayed_content:
                    self.console.print()
                sys.stdout.flush()

        # If interrupted, return early with empty results
        if interrupted:
            return "", [], False

        # Extract tool calls from streaming response (provider-agnostic)
        # Some providers (Venice AI) put tool calls in the last chunk's message
        # Others (OpenAI) stream tool calls incrementally in deltas
        # We've already accumulated tool calls from deltas in _accumulate_tool_call

        if last_chunk and hasattr(last_chunk, 'choices') and last_chunk.choices:
            choice = last_chunk.choices[0]

            # Check if there's a finish_reason of tool_calls
            if hasattr(choice, 'finish_reason') and choice.finish_reason == 'tool_calls':
                # Venice AI-style: Tool calls in the message (not delta)
                if hasattr(choice, 'message') and hasattr(choice.message, 'tool_calls'):
                    tool_calls = choice.message.tool_calls
                    if tool_calls:
                        # Only add if we haven't already accumulated from deltas
                        if not self.pending_tool_calls:
                            for tc in tool_calls:
                                raw_args = tc.function.arguments if hasattr(tc.function, 'arguments') else "{}"
                                # DEBUG: Log raw arguments from API
                                if os.environ.get("NASEA_DEBUG"):
                                    func_name = tc.function.name if hasattr(tc.function, 'name') else ""
                                    self.console.print(f"\n[yellow]DEBUG API raw tool call:[/yellow]")
                                    self.console.print(f"[dim]  function: {func_name}[/dim]")
                                    self.console.print(f"[dim]  raw arguments: {raw_args[:500]}...[/dim]" if len(raw_args) > 500 else f"[dim]  raw arguments: {raw_args}[/dim]")
                                self.pending_tool_calls.append({
                                    "id": tc.id if hasattr(tc, 'id') else None,
                                    "type": tc.type if hasattr(tc, 'type') else "function",
                                    "function": {
                                        "name": tc.function.name if hasattr(tc.function, 'name') else "",
                                        "arguments": raw_args
                                    }
                                })

        # Convert accumulated tool call buffer to pending tool calls (OpenAI-style)
        if not self.pending_tool_calls and self.tool_call_buffer:
            for tool_call_data in self.tool_call_buffer.values():
                if tool_call_data.get("function", {}).get("name"):
                    # DEBUG: Log accumulated tool calls
                    if os.environ.get("NASEA_DEBUG"):
                        func_name = tool_call_data.get("function", {}).get("name", "")
                        raw_args = tool_call_data.get("function", {}).get("arguments", "")
                        self.console.print(f"\n[yellow]DEBUG accumulated tool call:[/yellow]")
                        self.console.print(f"[dim]  function: {func_name}[/dim]")
                        self.console.print(f"[dim]  raw arguments: {raw_args[:500]}...[/dim]" if len(raw_args) > 500 else f"[dim]  raw arguments: {raw_args}[/dim]")
                    self.pending_tool_calls.append(tool_call_data)

        # Filter out thinking blocks from the complete response for history
        import re
        filtered_text = re.sub(r'<think>.*?</think>', '', self.text_buffer, flags=re.DOTALL).strip()

        # Execute tool calls if any
        tool_results = []
        is_complete = False

        if self.pending_tool_calls:
            # Deduplicate tool calls
            seen_signatures = set()
            unique_tool_calls = []
            for tool_call in self.pending_tool_calls:
                tool_signature = (
                    tool_call.get("function", {}).get("name", ""),
                    tool_call.get("function", {}).get("arguments", "")
                )
                if tool_signature not in seen_signatures:
                    seen_signatures.add(tool_signature)
                    unique_tool_calls.append(tool_call)
                else:
                    self.console.print(f"  [dim]⏭ Skipped duplicate tool call[/dim]")

            # Execute tools with parallel support
            tool_results, is_complete = self._execute_tool_calls(unique_tool_calls)

        # Return filtered text (without thinking blocks) for conversation history
        return filtered_text, tool_results, is_complete

    def _execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Execute tool calls with parallel support for independent operations.

        Groups parallel-safe tools together and executes them concurrently,
        while sequential tools run one at a time.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            Tuple of (results list, is_complete flag)
        """
        results = []
        is_complete = False

        if not tool_calls:
            return results, is_complete

        # Group tool calls into batches
        # Parallel-safe tools can run together, sequential tools run alone
        batches = self._group_tool_calls_into_batches(tool_calls)

        for batch in batches:
            # Check for interrupt
            if self.current_loader and self.current_loader.was_interrupted:
                break

            if len(batch) == 1:
                # Single tool - run normally
                result = self._handle_tool_call(batch[0])
                results.append(result)
                self.console.print()

                if result.get("user_cancelled"):
                    is_complete = True
                    break
                if result.get("is_complete"):
                    is_complete = True
            else:
                # Multiple parallel-safe tools - run concurrently
                batch_results = self._execute_parallel_batch(batch)
                results.extend(batch_results)

                # Check for completion/cancellation in any result
                for result in batch_results:
                    if result.get("user_cancelled"):
                        is_complete = True
                        break
                    if result.get("is_complete"):
                        is_complete = True

                if is_complete:
                    break

        return results, is_complete

    def _group_tool_calls_into_batches(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Group tool calls into batches for execution.

        Consecutive parallel-safe tools are grouped together.
        Sequential tools form their own single-item batch.

        Args:
            tool_calls: List of tool calls

        Returns:
            List of batches (each batch is a list of tool calls)
        """
        batches = []
        current_parallel_batch = []

        for tool_call in tool_calls:
            func_name = tool_call.get("function", {}).get("name", "")

            if func_name in PARALLEL_SAFE_TOOLS and self.enable_parallel:
                # Add to current parallel batch
                current_parallel_batch.append(tool_call)
            else:
                # Sequential tool - flush parallel batch first if any
                if current_parallel_batch:
                    batches.append(current_parallel_batch)
                    current_parallel_batch = []
                # Add sequential tool as its own batch
                batches.append([tool_call])

        # Don't forget remaining parallel batch
        if current_parallel_batch:
            batches.append(current_parallel_batch)

        return batches

    def _execute_parallel_batch(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute a batch of parallel-safe tools concurrently.

        Args:
            tool_calls: List of parallel-safe tool calls

        Returns:
            List of results in the same order as input
        """
        if len(tool_calls) == 1:
            result = self._handle_tool_call(tool_calls[0])
            self.console.print()
            return [result]

        # Prepare results dict to maintain order
        results_by_index = {}

        # Execute in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(tool_calls), self.max_parallel_workers)) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._execute_single_tool_for_parallel, tool_call): i
                for i, tool_call in enumerate(tool_calls)
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results_by_index[index] = result
                except Exception as e:
                    # Handle execution error
                    tool_call = tool_calls[index]
                    results_by_index[index] = {
                        "success": False,
                        "error": f"Parallel execution failed: {str(e)}",
                        "tool_call_id": tool_call.get("id"),
                        "name": tool_call.get("function", {}).get("name", "unknown")
                    }

        # Return results in original order
        ordered_results = [results_by_index[i] for i in range(len(tool_calls))]

        # Display results after parallel execution
        for i, result in enumerate(ordered_results):
            tool_call = tool_calls[i]
            func_name = tool_call.get("function", {}).get("name", "")
            args_str = tool_call.get("function", {}).get("arguments", "{}")
            try:
                arguments = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                arguments = {}
            self._display_tool_call(func_name, arguments)
            self._display_tool_result(result)
            self.console.print()

        return ordered_results

    def _execute_single_tool_for_parallel(
        self,
        tool_call: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single tool call for parallel execution.

        This is a simplified version of _handle_tool_call without display logic,
        since display happens after all parallel tools complete.

        Args:
            tool_call: Tool call data

        Returns:
            Tool execution result
        """
        function_name = tool_call["function"]["name"]
        arguments_str = tool_call["function"]["arguments"]

        # Parse arguments
        try:
            if arguments_str and arguments_str.startswith("{}"):
                arguments_str = arguments_str[2:]
            arguments = json.loads(arguments_str) if arguments_str else {}
            if isinstance(arguments, list):
                arguments = arguments[0] if arguments else {}
            if not isinstance(arguments, dict):
                arguments = {}
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "Response truncated - model output was cut off mid-generation",
                "tool_call_id": tool_call.get("id"),
                "name": function_name,
                "truncated": True
            }

        # Validate arguments
        validation_error = self._validate_arguments(function_name, arguments)
        if validation_error:
            return {
                "success": False,
                "error": validation_error,
                "tool_call_id": tool_call.get("id"),
                "name": function_name
            }

        # Execute tool (no permission check for parallel-safe read-only tools)
        result = self.executor.execute(function_name, arguments)
        result["tool_call_id"] = tool_call.get("id")
        result["name"] = function_name

        self._post_process_result(function_name, arguments, result)

        return result

    def _filter_and_display_chunk(self, content: str) -> str:
        """
        Filter thinking blocks in real-time and display visible content.

        Now shows thinking content in dim style so users can see the planning process.

        Args:
            content: Raw content chunk from stream

        Returns:
            Visible content (empty if inside thinking block)
        """
        import sys

        visible_content = ""
        i = 0

        while i < len(content):
            # Check for start of thinking block
            if not self.in_thinking_block:
                # Look for <think> tag
                if content[i:i+7] == '<think>':
                    self.in_thinking_block = True
                    self.thinking_buffer = '<think>'
                    # Don't change loader message for think blocks in text
                    # The think tool already shows the plan
                    i += 7
                    continue

                # Regular visible content - accumulate and display
                visible_content += content[i]
                i += 1
            else:
                # Inside thinking block - accumulate in buffer
                self.thinking_buffer += content[i]

                # Check for end of thinking block
                if content[i:i+8] == '</think>':
                    self.in_thinking_block = False
                    self.thinking_buffer = ""
                    self._set_loader_message(self.loader_idle_message)
                    i += 8
                    continue

                i += 1

        # Add visible content to line buffer for complete-line formatting
        self.line_buffer += visible_content

        # Process complete lines (markdown needs full lines to match patterns)
        while '\n' in self.line_buffer:
            # Split at first newline
            line, self.line_buffer = self.line_buffer.split('\n', 1)

            # Trim leading newlines on first output
            if not self._has_displayed_content:
                line = line.lstrip("\n")
                if not line:
                    continue

            # Skip lines that look like tool calls (LLM sometimes outputs these as text)
            tool_call_patterns = ['complete_generation(', 'read_file(', 'write_file(',
                                  'edit_file(', 'think(', 'run_command(', 'list_files(',
                                  'grep_search(', 'web_search(', 'memory_save(', 'memory_read(']
            if any(line.strip().startswith(p) for p in tool_call_patterns):
                continue
            # Also skip closing parens that might be part of tool calls
            if line.strip() == ')' or line.strip().startswith(')'):
                continue
            # Skip placeholder patterns that some models output instead of tool calls
            if '[NASEA_' in line or '[/NASEA_' in line:
                continue

            if not self._has_displayed_content:
                self._notify_first_output()
                # First line gets ⏺ prefix (like Claude Code)
                formatted_line = self._format_markdown(line)
                self.console.print(f"[white]⏺[/white] {formatted_line}", soft_wrap=True)
            else:
                # Continuation lines indented
                formatted_line = self._format_markdown(line)
                self.console.print(f"  {formatted_line}", soft_wrap=True)
            sys.stdout.flush()
            self._has_displayed_content = True

        return visible_content

    def _flush_line_buffer(self) -> None:
        """Flush any remaining text in the line buffer at end of stream."""
        import sys
        if self.line_buffer:
            # Trim leading newlines on first output
            text = self.line_buffer
            if not self._has_displayed_content:
                text = text.lstrip("\n")

            # Skip tool call syntax in final buffer
            tool_call_patterns = ['complete_generation(', 'read_file(', 'write_file(',
                                  'edit_file(', 'think(', 'run_command(', 'list_files(',
                                  'grep_search(', 'web_search(', 'memory_save(', 'memory_read(']
            if any(text.strip().startswith(p) for p in tool_call_patterns):
                self.line_buffer = ""
                return
            if text.strip() == ')' or text.strip().startswith(')'):
                self.line_buffer = ""
                return
            # Skip placeholder patterns that some models output instead of tool calls
            if '[NASEA_' in text or '[/NASEA_' in text:
                self.line_buffer = ""
                return

            if text:
                if not self._has_displayed_content:
                    self._notify_first_output()
                    formatted_text = self._format_markdown(text)
                    self.console.print(f"[white]⏺[/white] {formatted_text}", soft_wrap=True)
                else:
                    formatted_text = self._format_markdown(text)
                    self.console.print(f"  {formatted_text}", soft_wrap=True)
                sys.stdout.flush()
                self._has_displayed_content = True

            self.line_buffer = ""

    def _process_chunk(self, chunk):
        """Process a single streaming chunk with real-time display."""
        if not hasattr(chunk, 'choices') or not chunk.choices:
            return

        delta = chunk.choices[0].delta

        # Handle text content
        if hasattr(delta, 'content') and delta.content:
            # Filter thinking blocks and display in real-time
            self._filter_and_display_chunk(delta.content)
            # Also accumulate in buffer for history (we'll clean it later)
            self.text_buffer += delta.content

        # Handle tool calls
        if hasattr(delta, 'tool_calls') and delta.tool_calls:
            # DON'T stop loader here - keep it showing "Writing file..." etc.
            # Loader will be stopped when we actually display the tool result

            for tool_call_delta in delta.tool_calls:
                self._accumulate_tool_call(tool_call_delta)

    def _accumulate_tool_call(self, tool_call_delta):
        """Accumulate tool call information from streaming chunks."""
        index = tool_call_delta.index

        # Initialize tool call if new
        if index not in self.tool_call_buffer:
            self.tool_call_buffer[index] = {
                "id": getattr(tool_call_delta, 'id', None),
                "type": getattr(tool_call_delta, 'type', 'function'),
                "function": {
                    "name": "",
                    "arguments": ""
                }
            }

        tool_call = self.tool_call_buffer[index]

        # Accumulate function info
        if hasattr(tool_call_delta, 'function'):
            func_delta = tool_call_delta.function

            if hasattr(func_delta, 'name') and func_delta.name:
                tool_call["function"]["name"] = func_delta.name

                # Update loader with dynamic message based on tool being called
                tool_messages = {
                    "write_file": "Writing file...",
                    "edit_file": "Editing file...",
                    "create_directory": "Creating directory...",
                    "delete_path": "Deleting...",
                    "rename_path": "Renaming...",
                    "run_command": "Running command...",
                    "list_files": "Listing files...",
                    "read_file": "Reading file...",
                    "complete_generation": "Finalizing...",
                    "web_search": "Searching web...",
                    "open_browser": "Opening browser...",
                    "grep_search": "Searching code...",
                    "check_syntax": "Checking syntax...",
                    "think": "Planning...",
                    "memory_save": "Saving to memory...",
                    "memory_read": "Reading memory..."
                }
                message = tool_messages.get(func_delta.name, "Working...")
                self._set_loader_message(message)

            if hasattr(func_delta, 'arguments') and func_delta.arguments:
                tool_call["function"]["arguments"] += func_delta.arguments

        # Mark as complete if we have a name
        if tool_call["function"]["name"] and tool_call not in self.pending_tool_calls:
            self.pending_tool_calls.append(tool_call)

    def _handle_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a single tool call: display, get permission, execute.

        Args:
            tool_call: Tool call data

        Returns:
            Tool execution result
        """
        function_name = tool_call["function"]["name"]
        arguments_str = tool_call["function"]["arguments"]

        # Parse arguments
        try:
            # Venice AI sometimes prepends {} to arguments - strip it
            if arguments_str and arguments_str.startswith("{}"):
                arguments_str = arguments_str[2:]

            arguments = json.loads(arguments_str) if arguments_str else {}

            # Normalize: some models send list instead of dict
            if isinstance(arguments, list):
                arguments = arguments[0] if arguments else {}
            if not isinstance(arguments, dict):
                arguments = {}

            # DEBUG: Log raw edit_file arguments to see what AI actually sends
            if function_name == "edit_file" and os.environ.get("NASEA_DEBUG"):
                self.console.print(f"[dim]DEBUG raw arguments_str:[/dim]")
                self.console.print(f"[dim]{arguments_str}[/dim]")
                self.console.print(f"[dim]DEBUG old_content: {repr(arguments.get('old_content', '')[:100])}[/dim]")
                self.console.print(f"[dim]DEBUG new_content: {repr(arguments.get('new_content', '')[:100])}[/dim]")
                self.console.print(f"[dim]DEBUG identical? {arguments.get('old_content') == arguments.get('new_content')}[/dim]")
        except json.JSONDecodeError as e:
            # JSON parsing failed - likely truncated response from model
            # Stop any active loader first
            if self.current_loader:
                self.current_loader.stop()
                self.current_loader = None

            # Show user-friendly error (not technical JSON error)
            self._display_tool_call(function_name, {})
            self.console.print(f"  [yellow]⚠[/yellow] [dim]Response was cut off - retrying...[/dim]")

            result = {
                "success": False,
                "error": "Response truncated - model output was cut off mid-generation",
                "tool_call_id": tool_call.get("id"),
                "name": function_name,
                "truncated": True
            }
            return result

        # Update loader with contextual description
        self._set_loader_message(self._format_tool_loader_message(function_name, arguments))

        # Display tool call (Claude Code style)
        self._display_tool_call(function_name, arguments)

        # If no arguments, this might be an incomplete tool call
        if not arguments and arguments_str:
            # Stop loader before showing error
            if self.current_loader:
                self.current_loader.stop()
                self.current_loader = None

            self.console.print(f"  [yellow]⚠[/yellow] [dim]Incomplete response - retrying...[/dim]")
            result = {
                "success": False,
                "error": "Response truncated",
                "tool_call_id": tool_call.get("id"),
                "name": function_name,
                "truncated": True
            }
            return result

        validation_error = self._validate_arguments(function_name, arguments)
        if validation_error:
            result = {
                "success": False,
                "error": validation_error,
                "tool_call_id": tool_call.get("id"),
                "name": function_name
            }
            self._display_tool_result(result)
            return result

        # Check permission (returns tuple: is_allowed, feedback)
        is_allowed, feedback = self._get_permission(function_name, arguments)
        if not is_allowed:
            self.user_cancelled = True  # Track that user cancelled
            result = {
                "success": False,
                "error": feedback or "Operation cancelled by user",
                "tool_call_id": tool_call.get("id"),
                "name": function_name,
                "user_cancelled": True  # Flag to stop the loop
            }
            self._display_tool_result(result, cancelled=True)
            return result

        # Execute tool
        self._display_operation_progress(function_name)
        result = self.executor.execute(function_name, arguments)

        result["tool_call_id"] = tool_call.get("id")
        result["name"] = function_name

        self._post_process_result(function_name, arguments, result)

        # Display result
        self._display_tool_result(result)

        return result

    def _display_tool_call(self, function_name: str, arguments: Dict[str, Any]):
        """Display tool call in Claude Code style with full paths."""
        self._notify_first_tool()

        # Skip displaying these tool calls - only show the formatted result
        if function_name in ("think", "complete_generation"):
            return
        # If a browser preview was already opened in this session, suppress repeated
        # open_browser attempts to reduce noisy output.
        if function_name == "open_browser" and self.browsers_opened:
            return

        # Map function names to Claude Code-style operation names
        operation_map = {
            "read_file": "Read",
            "write_file": "Write",
            "edit_file": "Edit",
            "list_files": "List",
            "run_command": "Bash",
            "create_directory": "Create Directory",
            "delete_path": "Delete",
            "rename_path": "Rename",
            "complete_generation": "Complete",
            "think": "Think",
            "memory_save": "Remember",
            "memory_read": "Recall",
            "grep_search": "Search",
            "web_search": "Web Search",
            "check_syntax": "Lint",
            "open_browser": "Open Browser"
        }

        operation = operation_map.get(function_name, function_name)

        bullet = "[green]⏺[/green]"
        arg_summary = self._format_argument_summary(arguments)

        # Helper to convert paths to use ~ for home directory
        def to_display_path(path_str: str) -> str:
            from pathlib import Path
            # Resolve to absolute path first
            abs_path = str(Path(path_str).resolve())
            home = str(Path.home())
            if abs_path.startswith(home):
                return "~" + abs_path[len(home):]
            return abs_path

        # Check if required arguments are present to avoid showing "unknown" paths
        required_args = {
            "read_file": ["file_path"],
            "write_file": ["file_path"],
            "edit_file": ["file_path"],
            "create_directory": ["directory_path"],
            "delete_path": ["path"],
            "rename_path": ["source_path", "target_path"],
            "run_command": ["command"],
        }

        # If required args missing, show incomplete indicator
        if function_name in required_args:
            missing = [arg for arg in required_args[function_name] if arg not in arguments]
            if missing:
                self.console.print(f"{bullet} [yellow]{operation}[/yellow] [dim](missing {', '.join(missing)})[/dim]")
                return

        # Get file path for operations that work with files
        if function_name in ["read_file", "write_file", "edit_file"]:
            file_path = arguments.get("file_path")
            try:
                resolved = self.executor.resolve_path(file_path)
                display_path = to_display_path(str(resolved))
            except (ValueError, TypeError, AttributeError):
                display_path = to_display_path(file_path)
            self.console.print(f"{bullet} [bold white]{operation}[/bold white]({display_path})")

        elif function_name == "delete_path":
            path = arguments.get("path")
            try:
                resolved = self.executor.resolve_path(path)
                display_path = to_display_path(str(resolved))
            except (ValueError, TypeError, AttributeError):
                display_path = to_display_path(path)
            self.console.print(f"{bullet} [bold red]{operation}[/bold red]({display_path})")

        elif function_name == "rename_path":
            source = arguments.get("source_path")
            target = arguments.get("target_path")
            try:
                resolved_source = self.executor.resolve_path(source)
                resolved_target = self.executor.resolve_path(target)
                display_source = to_display_path(str(resolved_source))
                display_target = to_display_path(str(resolved_target))
            except (ValueError, TypeError, AttributeError):
                display_source = to_display_path(source)
                display_target = to_display_path(target)
            self.console.print(f"{bullet} [bold yellow]{operation}[/bold yellow]({display_source} → {display_target})")

        elif function_name == "create_directory":
            dir_path = arguments.get("directory_path")
            try:
                resolved = self.executor.resolve_path(dir_path)
                display_path = to_display_path(str(resolved))
            except (ValueError, TypeError, AttributeError):
                display_path = to_display_path(dir_path)
            self.console.print(f"{bullet} [bold white]{operation}[/bold white]({display_path})")

        elif function_name == "run_command":
            command = arguments.get("command", "")
            display_cmd = command[:80] + "..." if len(command) > 80 else command
            self.console.print(f"{bullet} [bold white]{operation}[/bold white]($ {display_cmd})")

        elif function_name == "list_files":
            directory = arguments.get("directory", ".")
            try:
                resolved = self.executor.resolve_path(directory)
                display_path = to_display_path(str(resolved))
            except (ValueError, TypeError, AttributeError):
                display_path = to_display_path(directory)
            self.console.print(f"{bullet} [bold white]{operation}[/bold white]({display_path})")

        elif function_name == "grep_search":
            # Search shows pattern and path like Claude Code
            pattern = arguments.get("pattern", "")
            path = arguments.get("path") or arguments.get("directory") or "."
            try:
                resolved = self.executor.resolve_path(path)
                display_path = to_display_path(str(resolved))
            except (ValueError, TypeError, AttributeError):
                display_path = to_display_path(path)
            self.console.print(f"{bullet} [bold white]{operation}[/bold white](pattern: \"{pattern}\", path: \"{display_path}\")")

        elif function_name == "web_search":
            query = arguments.get("query", "")
            self.console.print(f"{bullet} [bold white]{operation}[/bold white]({query})")

        elif function_name == "memory_save":
            key = arguments.get("key", "")
            self.console.print(f"{bullet} [bold white]{operation}[/bold white]({key})")

        elif function_name == "memory_read":
            key = arguments.get("key", "")
            self.console.print(f"{bullet} [bold white]{operation}[/bold white]({key})")

        elif function_name == "check_syntax":
            file_path = arguments.get("file_path")
            try:
                resolved = self.executor.resolve_path(file_path)
                display_path = to_display_path(str(resolved))
            except (ValueError, TypeError, AttributeError):
                display_path = to_display_path(file_path)
            self.console.print(f"{bullet} [bold white]{operation}[/bold white]({display_path})")

        elif function_name == "open_browser":
            path = arguments.get("path", "")
            # Check if it's a URL or file path
            if path.startswith(("http://", "https://")):
                display_path = path
            else:
                try:
                    resolved = self.executor.resolve_path(path)
                    display_path = to_display_path(str(resolved))
                except (ValueError, TypeError, AttributeError):
                    display_path = to_display_path(path)
            self.console.print(f"{bullet} [bold white]{operation}[/bold white]({display_path})")

        else:
            # Fallback: show function name with args
            args_display = arg_summary or ''
            if args_display:
                self.console.print(f"{bullet} [bold white]{operation}[/bold white]({args_display})")
            else:
                self.console.print(f"{bullet} [bold white]{operation}[/bold white]()")

        # Flush stdout to ensure immediate display
        import sys
        sys.stdout.flush()

    def _show_summary(self, result: Dict[str, Any]) -> None:
        """Show simple summary of work done - no borders, just clean text."""
        parts = []

        # Count what was done
        if self.files_created:
            parts.append(f"Created {len(self.files_created)} file(s)")
        if self.files_updated:
            parts.append(f"updated {len(self.files_updated)}")
        if self.files_deleted:
            parts.append(f"deleted {len(self.files_deleted)}")
        if self.commands_run:
            parts.append(f"ran {len(self.commands_run)} command(s)")

        if parts:
            self.console.print(f"{', '.join(parts)}.")

        # Next steps
        next_steps = result.get("next_steps")
        if next_steps:
            self.console.print(f"→ {next_steps}")

    def _format_argument_summary(self, arguments: Dict[str, Any]) -> str:
        """Format tool arguments for concise display."""
        if not arguments:
            return ""
        arg_strs = []
        for key, value in arguments.items():
            if isinstance(value, str):
                if len(value) > 50:
                    arg_strs.append(f'{key}="...{len(value)} chars..."')
                else:
                    arg_strs.append(f'{key}={json.dumps(value)}')
            else:
                arg_strs.append(f'{key}={json.dumps(value)}')
        return ', '.join(arg_strs)

    def _display_operation_progress(self, function_name: str) -> None:
        """Placeholder - progress is now shown inline with tool result."""
        self._notify_first_tool()

    def _display_tool_result(self, result: Dict[str, Any], cancelled: bool = False, denied: bool = False):
        """Display tool execution result in Claude Code style with detailed summaries."""
        from pathlib import Path

        if cancelled:
            self.console.print(f"  [yellow]⎿[/yellow] [dim]Cancelled[/dim]")
            sys.stdout.flush()
            return

        if denied:
            reason = result.get("error", "Permission denied")
            self.console.print(f"  [red]⎿[/red] [dim]{reason}[/dim]")
            sys.stdout.flush()
            return

        if result.get("success"):
            tool_name = result.get("name", "")

            # Format result message based on tool type
            if tool_name == "write_file":
                line_count = result.get("line_count", 0)
                file_path = result.get("file_path", "file")
                filename = Path(file_path).name
                is_overwrite = result.get("is_overwrite", False)

                action = "Overwrote" if is_overwrite else "Created"
                self.console.print(f"  [green]⎿[/green] {action} {filename} with {line_count} lines")

                # Show brief preview for new files
                preview_lines = result.get("preview_lines")
                if preview_lines:
                    for line in preview_lines:
                        # Truncate long lines
                        display_line = line[:80] + "..." if len(line) > 80 else line
                        self.console.print(f"      [dim]{display_line}[/dim]")
                    if line_count > 5:
                        self.console.print(f"      [dim]... ({line_count - 5} more lines)[/dim]")
                    sys.stdout.flush()  # Auto-scroll after preview

                # Show syntax warning if detected
                syntax_warning = result.get("syntax_warning")
                if syntax_warning:
                    self.console.print(f"  [yellow]⚠[/yellow] {syntax_warning}")

                # Track for summary
                if is_overwrite:
                    self.files_updated.append(filename)
                else:
                    self.files_created.append(filename)

            elif tool_name == "edit_file":
                file_path = result.get("file_path", "file")
                filename = Path(file_path).name
                self.files_updated.append(filename)

                # Count actual additions/removals from the changed content
                old_content = result.get("old_content", "")
                new_content = result.get("new_content", "")
                old_lines = len(old_content.splitlines()) if old_content else 0
                new_lines = len(new_content.splitlines()) if new_content else 0
                additions = max(0, new_lines - old_lines)
                removals = max(0, old_lines - new_lines)

                # Build message - only show what changed
                if additions > 0 and removals > 0:
                    self.console.print(f"  [green]⎿[/green] Updated {filename} with {additions} additions and {removals} removals")
                elif additions > 0:
                    self.console.print(f"  [green]⎿[/green] Updated {filename} with {additions} additions")
                elif removals > 0:
                    self.console.print(f"  [green]⎿[/green] Updated {filename} with {removals} removals")
                else:
                    self.console.print(f"  [green]⎿[/green] Updated {filename}")

                # Show unified diff if we have old and new content
                old_content = result.get("old_content")
                new_content = result.get("new_content")
                if old_content and new_content and old_content != new_content:
                    import difflib
                    from rich.syntax import Syntax
                    from rich.panel import Panel

                    # Generate unified diff
                    diff_lines = list(difflib.unified_diff(
                        old_content.splitlines(keepends=True),
                        new_content.splitlines(keepends=True),
                        fromfile=f"a/{filename}",
                        tofile=f"b/{filename}",
                        lineterm=""
                    ))

                    if diff_lines:
                        # Show simple +/- diff, truncate after 50 lines
                        displayed = 0
                        max_diff_lines = 50
                        remaining = 0
                        for line in diff_lines:
                            line = line.rstrip('\n')
                            # Skip metadata lines (---, +++, @@)
                            if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
                                continue
                            if displayed >= max_diff_lines:
                                remaining += 1
                                continue
                            if line.startswith('+'):
                                self.console.print(f"      [green]{line}[/green]")
                            elif line.startswith('-'):
                                self.console.print(f"      [red]{line}[/red]")
                            elif line.strip():
                                self.console.print(f"      [dim]{line}[/dim]")
                            displayed += 1
                        if remaining > 0:
                            self.console.print(f"      [dim]... +{remaining} more lines[/dim]")
                        sys.stdout.flush()  # Auto-scroll after diff

                # Show syntax warning if detected
                syntax_warning = result.get("syntax_warning")
                if syntax_warning:
                    self.console.print(f"  [yellow]⚠[/yellow] {syntax_warning}")

            elif tool_name == "read_file":
                line_count = result.get("line_count", 0)
                self.console.print(f"  [green]⎿[/green] Read {line_count} lines")

            elif tool_name == "list_files":
                count = result.get("count", 0)
                files = result.get("files", [])
                self.console.print(f"  [green]⎿[/green] Found {count} items")

                # Show file list (clean format without parentheses)
                if files and len(files) <= 10:
                    for file_info in files:
                        label = file_info.get("path", "<unknown>")
                        self.console.print(f"      [dim]• {label}[/dim]")
                elif files and len(files) > 10:
                    # Show first 10
                    for file_info in files[:10]:
                        label = file_info.get("path", "<unknown>")
                        self.console.print(f"      [dim]• {label}[/dim]")
                    self.console.print(f"      [dim]... and {len(files) - 10} more[/dim]")
                if files:
                    sys.stdout.flush()  # Auto-scroll after file list

            elif tool_name == "run_command":
                command = result.get("command", "")
                # Track for summary
                if command:
                    self.commands_run.append(command)

                output = result.get("output", "")
                if output:
                    # Show first few lines of output
                    lines = output.split("\n")[:5]
                    self.console.print(f"  [green]⎿[/green] Command executed successfully")
                    for line in lines:
                        if line.strip():
                            self.console.print(f"      [dim]{line}[/dim]")
                    if len(output.split("\n")) > 5:
                        self.console.print(f"      [dim]... ({len(output.split('\n')) - 5} more lines)[/dim]")
                    sys.stdout.flush()  # Auto-scroll after command output
                else:
                    self.console.print(f"  [green]⎿[/green] Command completed")

            elif tool_name == "create_directory":
                dir_path = result.get("directory_path", "directory")
                # Show just the directory name, not full path
                dir_name = Path(dir_path).name
                self.console.print(f"  [green]⎿[/green] Created directory: {dir_name}")

            elif tool_name == "delete_path":
                path = result.get("path", "item")
                filename = Path(path).name
                self.console.print(f"  [green]⎿[/green] Deleted {filename}")
                # Track for summary
                self.files_deleted.append(filename)

            elif tool_name == "rename_path":
                target = result.get("target_path", "target")
                filename = Path(target).name
                self.console.print(f"  [green]⎿[/green] Renamed to {filename}")

            elif tool_name == "run_command":
                command = result.get("command", "command")
                # Track for summary
                self.commands_run.append(command)

            elif tool_name == "complete_generation":
                # Don't show anything - the model already provides a summary
                # in its text response before calling complete_generation
                pass

            elif tool_name == "open_browser":
                url = result.get("url", result.get("path", ""))
                status = result.get("status", "Opened")

                # Show status with appropriate color
                if "error" in status.lower():
                    self.console.print(f"  [red]⎿[/red] {status}")
                elif "warning" in status.lower():
                    self.console.print(f"  [yellow]⎿[/yellow] {status}")
                else:
                    self.console.print(f"  [green]⎿[/green] {status}")

                # Show console errors if any
                console_errors = result.get("console_errors", [])
                page_errors = result.get("page_errors", [])

                if page_errors:
                    for err in page_errors[:5]:  # Limit display
                        self.console.print(f"    [red]✗[/red] {err[:100]}")

                if console_errors:
                    for log in console_errors[:5]:  # Limit display
                        log_type = log.get("type", "error")
                        text = log.get("text", "")[:100]
                        if log_type == "error":
                            self.console.print(f"    [red]✗[/red] {text}")
                        else:
                            self.console.print(f"    [yellow]![/yellow] {text}")

            elif tool_name == "think":
                # Display the plan prominently so user sees it before execution
                thought = result.get("thought", "")
                plan = result.get("plan", [])

                # Handle malformed thought (dict instead of string)
                if isinstance(thought, dict):
                    # Model returned a dict - convert to string representation
                    thought = str(thought)
                elif not isinstance(thought, str):
                    thought = str(thought) if thought else ""

                # Clean up tool names in plan text for user-friendliness
                tool_friendly_names = {
                    "grep_search": "search",
                    "read_file": "read",
                    "write_file": "write",
                    "edit_file": "edit",
                    "list_files": "list",
                    "run_command": "run",
                    "web_search": "web search",
                    "memory_save": "remember",
                    "memory_read": "recall",
                }
                for tool_name_raw, friendly in tool_friendly_names.items():
                    thought = thought.replace(tool_name_raw, friendly)

                self.console.print(f"  [white]⎿[/white] [bold]Plan:[/bold] {thought}")
                if plan:
                    self.console.print()
                    for i, step in enumerate(plan, 1):
                        # Also clean tool names in plan steps
                        clean_step = step
                        for tool_name_raw, friendly in tool_friendly_names.items():
                            clean_step = clean_step.replace(tool_name_raw, friendly)
                        self.console.print(f"    [dim]{i}.[/dim] {clean_step}")

            elif tool_name == "grep_search":
                # Display search results
                matches = result.get("matches", [])
                count = result.get("count", 0)
                if matches:
                    self.console.print(f"  [green]⎿[/green] Found {count} match(es)")
                    # Show first 5 results
                    for match in matches[:5]:
                        self.console.print(f"      [dim]{match}[/dim]")
                    if len(matches) > 5:
                        self.console.print(f"      [dim]... and {len(matches) - 5} more[/dim]")
                else:
                    self.console.print(f"  [yellow]⎿[/yellow] No matches found")

            elif tool_name == "check_syntax":
                # Display syntax check results
                has_errors = result.get("has_errors", False)
                file_path = result.get("file_path", "file")
                filename = Path(file_path).name
                if has_errors:
                    error_details = result.get("error_details", "Unknown error")
                    self.console.print(f"  [yellow]⎿[/yellow] {error_details}")
                else:
                    self.console.print(f"  [green]⎿[/green] No syntax errors in {filename}")

            elif tool_name == "web_search":
                # Display web search results - IMPORTANT: show actual content for AI to use
                message = result.get("message", "")
                output = result.get("output", "")
                self.console.print(f"  [green]⎿[/green] {message}")
                if output:
                    # Show first few results for user visibility
                    lines = output.strip().split("\n")[:15]
                    for line in lines:
                        if line.strip():
                            self.console.print(f"      [dim]{line[:100]}[/dim]")
                    if len(output.strip().split("\n")) > 15:
                        self.console.print(f"      [dim]... (more results available)[/dim]")

            else:
                # Fallback
                message = result.get("message", "Success")
                self.console.print(f"  [green]⎿[/green] {message}")

        else:
            # Error - red
            tool_name = result.get("name", "")
            error = result.get("error", "Unknown error")
            # Suppress noisy "already opened" browser message; it's an intentional guardrail.
            if tool_name == "open_browser" and "already used once" in error:
                return
            self.console.print(f"  [red]⎿[/red] {error}")

        # Flush to ensure terminal auto-scrolls to latest output
        sys.stdout.flush()

    def _normalize_rel_path(self, raw_path: Optional[str]) -> str:
        """Normalize relative path text for tracking."""
        if not raw_path or raw_path == ".":
            return "."
        normalized = Path(raw_path).as_posix()
        if normalized == ".":
            return "."
        if normalized.startswith("./"):
            normalized = normalized[2:]
        normalized = normalized.lstrip("/")
        return normalized or "."

    def _record_listing(self, directory: Optional[str], files: Optional[List[Dict[str, Any]]]) -> None:
        if files is None:
            return
        directory_key = self._normalize_rel_path(directory or ".")
        entries = set()
        for file_info in files:
            rel_path = self._normalize_rel_path(file_info.get("path", ""))
            entries.add(rel_path)
            if rel_path != ".":
                self.tracked_paths.add(rel_path)
        self.directory_snapshots[directory_key] = entries

    def _register_path(self, path_value: Optional[str]) -> None:
        rel = self._normalize_rel_path(path_value)
        if rel != ".":
            self.tracked_paths.add(rel)

    def _remove_path(self, path_value: Optional[str]) -> None:
        rel = self._normalize_rel_path(path_value)
        if rel in self.tracked_paths:
            self.tracked_paths.remove(rel)
        self.read_cache.pop(rel, None)

    def _invalidate_cache(self, path_value: Optional[str]) -> None:
        rel = self._normalize_rel_path(path_value)
        self.read_cache.pop(rel, None)

    def _path_exists(self, relative_path: str) -> tuple[bool, Optional[Path], Optional[str]]:
        try:
            resolved = self.executor.resolve_path(relative_path)
        except ValueError as error:
            return False, None, str(error)
        return resolved.exists(), resolved, None

    def _known_paths_summary(self, max_items: int = 6) -> str:
        if not self.tracked_paths:
            return "No files listed yet. Run list_files() first."
        sorted_paths = sorted(self.tracked_paths)
        if len(sorted_paths) > max_items:
            visible = ", ".join(sorted_paths[:max_items])
            return f"{visible}, … (+{len(sorted_paths) - max_items} more)"
        return ", ".join(sorted_paths)

    def _validate_arguments(self, function_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Lightweight validation before executing destructive or doomed calls."""
        # Validate write_file required arguments
        if function_name == "write_file":
            file_path = arguments.get("file_path")
            content = arguments.get("content")
            if not file_path:
                return "Missing required argument 'file_path'. Please retry with both file_path and content."
            if content is None:  # Allow empty string, but not missing
                return f"Missing required argument 'content' for file '{file_path}'. Please retry with file content."

        # Validate edit_file required arguments
        if function_name == "edit_file":
            file_path = arguments.get("file_path")
            old_content = arguments.get("old_content")
            new_content = arguments.get("new_content")
            if not file_path:
                return "Missing required argument 'file_path'. Please retry with file_path, old_content, and new_content."
            if old_content is None:
                return f"Missing required argument 'old_content' for file '{file_path}'. Please retry with old_content and new_content."
            if new_content is None:
                return f"Missing required argument 'new_content' for file '{file_path}'. Please retry with old_content and new_content."
            exists, resolved, error = self._path_exists(file_path)
            if error:
                return error
            if not exists:
                summary = self._known_paths_summary()
                return f"Cannot edit '{file_path}' because it does not exist. Known files: {summary}"
            rel = self._normalize_rel_path(file_path)
            is_hidden = Path(file_path).name.startswith(".")
            if is_hidden and rel not in self.tracked_paths and rel not in self.read_cache:
                return f"Ignoring hidden/system file '{file_path}'. Focus on project files listed via list_files()."

        if function_name == "read_file":
            file_path = arguments.get("file_path")
            if not file_path:
                return "Missing required argument 'file_path'."
            exists, resolved, error = self._path_exists(file_path)
            if error:
                return error
            if not exists:
                summary = self._known_paths_summary()
                return f"Cannot read '{file_path}' because it does not exist. Known files: {summary}"
            rel = self._normalize_rel_path(file_path)
            is_hidden = Path(file_path).name.startswith(".")
            if is_hidden and rel not in self.tracked_paths and rel not in self.read_cache:
                return f"Ignoring hidden/system file '{file_path}'. Focus on project files listed via list_files()."
            # Avoid repeating the exact same read when nothing has changed.
            offset = arguments.get("offset")
            limit = arguments.get("limit")
            for previous in reversed(self._read_history[-12:]):
                if (
                    (previous.get("file_path") or "") == file_path
                    and previous.get("offset") == offset
                    and previous.get("limit") == limit
                    and previous.get("mutation_counter") == self._mutation_counter
                ):
                    # If the previous read was truncated, allow a more targeted offset/limit read.
                    if not previous.get("truncated"):
                        return (
                            f"Already read '{file_path}' with the same offset/limit and no changes since. "
                            "Use the cached content, or use grep_search then read_file with a different offset+limit."
                        )
            if self._mutation_counter == 0:
                recent_activity = (len(self.searches_run) + len(self.files_read))
                if recent_activity >= 24:
                    return (
                        "Too many searches/reads without any edits. "
                        "Stop discovering and apply a fix with edit_file."
                    )

        if function_name == "open_browser":
            # Prevent repeated browser opens in a single run (token + attention waste).
            if self.browsers_opened:
                return "open_browser was already used once. Do not call it again."
            path = arguments.get("path")
            if not path:
                return "Missing required argument 'path'."

        if function_name == "grep_search":
            # Avoid repeating the exact same search when nothing has changed.
            pattern = (arguments.get("pattern") or "").strip()
            path = (arguments.get("path") or arguments.get("directory") or ".").strip()
            if pattern:
                for previous in reversed(self.searches_run[-12:]):
                    if (
                        (previous.get("pattern") or "").strip() == pattern
                        and (previous.get("path") or ".").strip() == path
                        and previous.get("mutation_counter") == self._mutation_counter
                    ):
                        return (
                            f"Already searched for pattern '{pattern}' in '{path}' with no changes since. "
                            "Use the previous results or refine the pattern/path."
                        )

            # Guard against "analysis paralysis" loops: too many searches/reads with no edits.
            if self._mutation_counter == 0:
                recent_activity = (len(self.searches_run) + len(self.files_read))
                if recent_activity >= 18:
                    return (
                        "Too many searches/reads without any edits. "
                        "Stop discovering and apply a fix with edit_file (or write_file if necessary)."
                    )
        if function_name == "delete_path":
            target = arguments.get("path")
            if not target:
                return "Missing required argument 'path'."
            exists, resolved, error = self._path_exists(target)
            if error:
                return error
            if not exists:
                return f"Nothing to delete at '{target}'."
            rel = self._normalize_rel_path(target)
            if resolved and resolved.is_file() and rel not in self.read_cache:
                return f"Read '{target}' first to confirm it's safe to delete."
            if resolved and resolved.is_dir() and rel not in self.tracked_paths:
                return f"Directory '{target}' has not been listed yet. Run list_files() first."
        if function_name == "create_directory":
            dir_path = arguments.get("directory_path")
            if not dir_path:
                return "Missing required argument 'directory_path'."

        if function_name == "rename_path":
            source = arguments.get("source_path")
            target = arguments.get("target_path")
            if not source:
                return "Missing required argument 'source_path'."
            if not target:
                return "Missing required argument 'target_path'."
            exists, _, error = self._path_exists(source)
            if error:
                return error
            if not exists:
                return f"Cannot rename '{source}' because it does not exist."

        if function_name == "run_command":
            command = arguments.get("command", "")
            if not command:
                return "Missing required argument 'command'."

            # Check for dangerous command patterns
            import re
            dangerous_patterns = [
                (r'\brm\s+-rf\s+/', "Recursive deletion from root directory"),
                (r'\brm\s+-rf\s+~', "Recursive deletion of home directory"),
                (r'\brm\s+-rf\s+\*', "Recursive deletion with wildcard"),
                (r'>\s*/dev/sd', "Direct disk write operation"),
                (r'\bcurl\b.*\|.*\bsh\b', "Piping curl output to shell"),
                (r'\bwget\b.*\|.*\bsh\b', "Piping wget output to shell"),
                (r'\beval\s+["\']?\$\(', "Using eval with command substitution"),
                (r'\bexec\s+["\']?\$\(', "Using exec with command substitution"),
                (r':\(\)\s*\{[^}]*\|[^}]*&[^}]*\};\s*:', "Fork bomb pattern"),
                (r'\bdd\s+if=/dev/(zero|random|urandom)\s+of=/dev/sd', "Disk overwrite with random data"),
                (r'\bmkfs\b', "Filesystem formatting"),
                (r'\bformat\b.*\bc:', "Windows disk formatting"),
                (r'\[(.*\]){100,}', "Excessive bracket expansion (potential DoS)"),
                (r'sudo\s+(rm|dd|mkfs|chmod\s+777)', "Destructive sudo command"),
            ]

            for pattern, description in dangerous_patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    return f"Dangerous command detected: {description}. Command: '{command[:100]}...'. Please review and modify."

            # Warn about potentially risky commands (don't block, just warn)
            risky_patterns = [
                r'\bsudo\b',
                r'\brm\b',
                r'\bchmod\s+777\b',
                r'>\s*/etc/',
            ]

            for pattern in risky_patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    # Don't block, but the permission system will ask user
                    pass

        return None

    def _post_process_result(self, function_name: str, arguments: Dict[str, Any], result: Dict[str, Any]) -> None:
        if not result.get("success"):
            return

        if function_name == "list_files":
            self._record_listing(arguments.get("directory", "."), result.get("files"))
        elif function_name == "write_file":
            self._mutation_counter += 1
            self._register_path(arguments.get("file_path"))
            self._invalidate_cache(arguments.get("file_path"))
        elif function_name == "create_directory":
            self._mutation_counter += 1
            self._register_path(arguments.get("directory_path"))
        elif function_name == "delete_path":
            self._mutation_counter += 1
            self._remove_path(arguments.get("path"))
        elif function_name == "rename_path":
            self._mutation_counter += 1
            self._remove_path(arguments.get("source_path"))
            self._register_path(arguments.get("target_path"))
        elif function_name == "read_file":
            rel = self._normalize_rel_path(arguments.get("file_path"))
            if "content" in result:
                self.read_cache[rel] = {
                    "content": result["content"],
                    "timestamp": time.time()
                }
            self.files_read.append(
                {
                    "file_path": arguments.get("file_path"),
                    "offset": arguments.get("offset"),
                    "limit": arguments.get("limit"),
                    "line_count": result.get("line_count"),
                    "total_lines": result.get("total_lines"),
                    "truncated": bool(result.get("truncated", False)),
                }
            )
            self._read_history.append(
                {
                    "file_path": arguments.get("file_path"),
                    "offset": arguments.get("offset"),
                    "limit": arguments.get("limit"),
                    "mutation_counter": self._mutation_counter,
                    "truncated": bool(result.get("truncated", False)),
                }
            )
        elif function_name == "edit_file":
            self._mutation_counter += 1
            self._invalidate_cache(arguments.get("file_path"))
        elif function_name == "grep_search":
            self.searches_run.append(
                {
                    "pattern": arguments.get("pattern"),
                    "path": arguments.get("path") or arguments.get("directory") or ".",
                    "count": result.get("count"),
                    "mutation_counter": self._mutation_counter,
                }
            )
        elif function_name == "check_syntax":
            self.syntax_checked.append(
                {
                    "file_path": arguments.get("file_path"),
                    "has_errors": result.get("has_errors"),
                    "error_details": result.get("error_details"),
                }
            )
        elif function_name == "open_browser":
            self.browsers_opened.append({"path": arguments.get("path")})

    def format_recent_tool_context(self, max_items: int = 8) -> str:
        """
        Return a compact summary of recent tool actions for resume/continue prompts.

        The goal is to prevent the model from re-reading/linting the same large files
        and to encourage grep_search + offset/limit reads.
        """
        parts: List[str] = []

        if self.files_read:
            shown = self.files_read[-max_items:]
            items = []
            for entry in shown:
                fp = entry.get("file_path") or "<unknown>"
                total = entry.get("total_lines")
                truncated = entry.get("truncated")
                offset = entry.get("offset")
                limit = entry.get("limit")
                span = ""
                if offset is not None or limit is not None:
                    span = f" (offset={offset or 0}, limit={limit})"
                elif total:
                    span = f" ({total} lines{' truncated' if truncated else ''})"
                items.append(f"{fp}{span}")
            parts.append("Read: " + ", ".join(items))

        if self.searches_run:
            shown = self.searches_run[-max_items:]
            items = []
            for entry in shown:
                pat = entry.get("pattern") or ""
                path = entry.get("path") or "."
                count = entry.get("count")
                suffix = f" → {count} match(es)" if count is not None else ""
                items.append(f"\"{pat}\" in {path}{suffix}")
            parts.append("Searches: " + "; ".join(items))

        if self.syntax_checked:
            shown = self.syntax_checked[-max_items:]
            items = []
            for entry in shown:
                fp = entry.get("file_path") or "<unknown>"
                has_errors = entry.get("has_errors")
                items.append(f"{fp} ({'errors' if has_errors else 'ok'})")
            parts.append("Lint: " + ", ".join(items))

        if self.browsers_opened:
            shown = self.browsers_opened[-max_items:]
            items = [entry.get("path") or "<unknown>" for entry in shown]
            parts.append("Open browser: " + ", ".join(items))

        return "\n".join(f"- {p}" for p in parts) if parts else ""

    def format_recent_findings(self, max_items: int = 8) -> str:
        """
        Extract a compact "findings so far" summary to feed into /continue prompts.

        This is intentionally conservative: it prefers tool-derived findings (check_syntax)
        and only lightly scrapes the assistant's own text for mismatch-like lines.
        """
        findings: List[str] = []
        seen: Set[str] = set()

        def _add(item: str) -> None:
            item = (item or "").strip()
            if not item:
                return
            if item in seen:
                return
            seen.add(item)
            findings.append(item)

        # Prefer tool-derived diagnostics.
        for entry in (self.syntax_checked or [])[-max_items * 3:]:
            file_path = entry.get("file_path") or "<unknown>"
            details = entry.get("error_details") or ""
            if not details:
                continue
            lowered = details.lower()
            if any(k in lowered for k in ("mismatch", "warning", "duplicate selector")):
                first_line = details.splitlines()[0].strip()
                _add(f"{file_path}: {first_line[:200]}")

        # Light scrape of visible assistant text for mismatch statements.
        try:
            text = (self.text_buffer or "")[-6000:]
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line or len(line) > 220:
                    continue
                lowered = line.lower()
                if any(
                    k in lowered
                    for k in (
                        "mismatch",
                        "missing",
                        "no matches found",
                        "doesn't have",
                        "does not have",
                        "not present",
                        "not found",
                        "found another issue",
                    )
                ):
                    # Avoid echoing tool call-like lines.
                    if any(lowered.startswith(p) for p in ("read(", "write(", "edit(", "search(", "lint(", "bash(")):
                        continue
                    _add(line)
        except Exception:
            pass

        if not findings:
            return ""

        # Keep it short and scannable.
        findings = findings[:max_items]
        return "\n".join(f"- {f}" for f in findings)
    def _generate_permission_context(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """
        Generate rich contextual information about what a tool will do.

        Args:
            function_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Formatted context string for display
        """
        context_lines = []

        if function_name == "write_file":
            file_path = arguments.get("file_path")
            content = arguments.get("content", "")

            # Should not happen due to validation, but handle gracefully
            if not file_path:
                context_lines.append(f"  [red]Invalid:[/red] Missing file_path")
                return "\n".join(context_lines)

            line_count = len(content.splitlines())
            char_count = len(content)

            # Check if file exists
            try:
                resolved_path = self.executor.resolve_path(file_path)
                exists = resolved_path.exists()
                action = "Replace" if exists else "Create new"
            except (ValueError, TypeError, AttributeError, OSError):
                action = "Write to"

            context_lines.append(f"  [cyan]{action} file:[/cyan] {file_path}")
            context_lines.append(f"  [dim]Size:[/dim] {line_count} lines, {char_count:,} characters")

        elif function_name == "edit_file":
            # No context needed - diff shown in result
            pass

        elif function_name == "delete_path":
            path = arguments.get("path")
            recursive = arguments.get("recursive", False)

            context_lines.append(f"  [red]Delete:[/red] {path}")
            if recursive:
                context_lines.append(f"  [yellow]⚠ Recursive deletion[/yellow]")

        elif function_name == "run_command":
            command = arguments.get("command", "")
            context_lines.append(f"  [dim]Command:[/dim] {command[:100]}")

        elif function_name == "rename_path":
            source = arguments.get("source_path")
            target = arguments.get("target_path")

            # Should not happen due to validation, but handle gracefully
            if not source or not target:
                context_lines.append(f"  [red]Invalid:[/red] Missing source_path or target_path")
                return "\n".join(context_lines)

            context_lines.append(f"  [cyan]Rename:[/cyan] {source} → {target}")

        elif function_name == "create_directory":
            dir_path = arguments.get("directory_path")

            # Should not happen due to validation, but handle gracefully
            if not dir_path:
                context_lines.append(f"  [red]Invalid:[/red] Missing directory_path")
                return "\n".join(context_lines)

            # Show just the directory name, not full path
            from pathlib import Path
            dir_name = Path(dir_path).name
            context_lines.append(f"  [cyan]Create directory:[/cyan] {dir_name}")

        return "\n".join(context_lines) if context_lines else ""

    def _get_permission(self, function_name: str, arguments: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Get user permission to execute tool with rich context.

        Args:
            function_name: Name of the function to execute
            arguments: Function arguments

        Returns:
            Tuple of (is_allowed, feedback_message)
            - feedback_message is empty if allowed, contains user guidance if rejected
        """
        # Auto-approve safe/read-only tools (no user confirmation needed)
        safe_tools = {
            "think",              # Just planning, no side effects
            "read_file",          # Read-only
            "list_files",         # Read-only
            "memory_read",        # Read-only
            "memory_save",        # Just saves context
            "grep_search",        # Read-only search
            "web_search",         # Read-only search
            "complete_generation", # Just marks completion
        }
        if function_name in safe_tools:
            return True, ""

        # Auto-approve if in 'always' mode
        if self.permission_mode == "always":
            return True, ""

        # Auto-reject if in 'never' mode
        if self.permission_mode == "never":
            return False, "Permission denied by configuration."

        # We are about to prompt the user; stop any active loader first to avoid UI overlap.
        self._notify_first_tool()

        # Show rich context about what will happen
        context = self._generate_permission_context(function_name, arguments)
        if context:
            self.console.print(context)
            import sys
            sys.stdout.flush()

        # Try to use questionary for inline arrow-key selection
        try:
            import questionary

            # Generate tool-specific prompt
            if function_name == "write_file":
                file_path = arguments.get("file_path", "file")
                prompt_text = f"Allow file write to {file_path}?"
            elif function_name == "edit_file":
                file_path = arguments.get("file_path", "file")
                prompt_text = f"Allow file edit to {file_path}?"
            elif function_name == "delete_path":
                path = arguments.get("path", "path")
                prompt_text = f"Allow deletion of {path}?"
            elif function_name == "run_command":
                cmd = arguments.get("command", "")[:40]
                prompt_text = f"Allow command: {cmd}...?" if len(arguments.get("command", "")) > 40 else f"Allow command: {cmd}?"
            elif function_name == "rename_path":
                source = arguments.get("source_path", "")
                target = arguments.get("target_path", "")
                prompt_text = f"Allow rename {source} → {target}?"
            elif function_name == "create_directory":
                dir_path = arguments.get("directory_path", "directory")
                prompt_text = f"Allow create directory {dir_path}?"
            elif function_name == "check_syntax":
                file_path = arguments.get("file_path", "file")
                prompt_text = f"Allow syntax check on {file_path}?"
            else:
                # Use friendly names for display
                friendly_names = {
                    "grep_search": "search",
                    "web_search": "web search",
                    "memory_save": "save to memory",
                    "memory_read": "read from memory",
                }
                display_name = friendly_names.get(function_name, function_name)
                prompt_text = f"Allow {display_name}?"

            # Show inline menu with arrow-key selection
            result = questionary.select(
                prompt_text,
                choices=[
                    "Yes",
                    "Always (don't ask again)",
                    "No",
                ],
                style=questionary.Style([
                    ('selected', 'fg:cyan'),
                    ('pointer', 'fg:cyan bold'),
                    ('question', 'bold'),
                ])
            ).ask()

            if result is None:
                # ESC was pressed - clean up and cancel
                import sys
                sys.stdout.write("\033[1A\033[2K")  # Clear the prompt line
                sys.stdout.flush()
                self.console.print("  [dim]Cancelled[/dim]")
                return False, "Cancelled"
            elif result == "Yes" or result == "Always (don't ask again)":
                # Clear the prompt line(s) - move up and clear
                import sys
                sys.stdout.write("\033[1A\033[2K")
                sys.stdout.flush()
                if result == "Always (don't ask again)":
                    self.permission_mode = "always"
                return True, ""
            else:
                # "No" selected
                return False, "Cancelled"

        except (ImportError, Exception) as e:
            # Fallback to simple prompt with colors
            self.console.print("  [yellow]Approve?[/yellow] [dim](y/n)[/dim]: ", end="")

            try:
                import sys
                response = sys.stdin.readline().strip().lower()

                if response in ["y", "yes"]:
                    return True, ""
                else:
                    return False, "User rejected this action."
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return False, "User cancelled."


class ToolUseConversation:
    """
    Manage multi-turn conversation with tool use.
    Continues calling the LLM until no more tools are needed.
    """

    def __init__(
        self,
        client,
        handler: StreamingToolHandler,
        console: Console,
        model: str,
        tools: List[Dict[str, Any]],
        enable_thinking: bool = False,
        strip_thinking: bool = False
    ):
        """
        Initialize tool use conversation.

        Args:
            client: LLM client (OpenAI/Venice compatible)
            handler: Streaming tool handler
            console: Rich console
            model: Model name
            tools: Tool definitions
            enable_thinking: Whether to request reasoning mode
        """
        self.client = client
        self.handler = handler
        self.console = console
        self.model = model
        self.tools = tools
        self.max_turns = 20  # Prevent infinite loops
        self.enable_thinking = enable_thinking
        self.strip_thinking = strip_thinking

    def run(
        self,
        system_message: str,
        user_message: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Run conversation with tool use until completion.

        Args:
            system_message: System prompt
            user_message: User's request
            chat_history: Optional list of previous messages for context

        Returns:
            Final result with summary
        """
        messages = [{"role": "system", "content": system_message}]

        # Include chat history if provided (for conversational context)
        if chat_history:
            for msg in chat_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Track if any tools were used (vs. just chatting)
        any_tools_used = False
        consecutive_errors = 0  # Track consecutive tool errors
        max_consecutive_errors = 3  # Stop if too many errors in a row
        last_error_message = None  # Track last error to detect loops

        for turn in range(self.max_turns):
            # Check if user already pressed Esc to interrupt
            if self.handler.current_loader and self.handler.current_loader.was_interrupted:
                return {
                    "success": False,
                    "error": "User interrupted",
                    "summary": "",
                    "next_steps": ""
                }

            # Stream response with tools
            # Use model-appropriate max_tokens from config
            from nasea.core.utils import get_max_output_tokens
            model_max_tokens = get_max_output_tokens(self.model)
            request_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "tools": self.tools,
                "stream": True,
                "max_tokens": model_max_tokens,
            }
            extra_body: Dict[str, Any] = {"venice_parameters": {}}
            if self.enable_thinking:
                extra_body["venice_parameters"]["disable_thinking"] = False
            else:
                extra_body["venice_parameters"]["disable_thinking"] = True
            if self.strip_thinking:
                extra_body["venice_parameters"]["strip_thinking_response"] = True
            request_kwargs["extra_body"] = extra_body

            # Keep a loader visible throughout the turn to avoid "silent gaps" while
            # the model is thinking/generating large tool payloads.
            # We intentionally do not stop the loader on first output.
            loader = get_loader(self.handler.loader_type or "chat", console=self.console)
            loader.start()
            self.handler.on_first_output = None
            self.handler.on_first_tool = None
            self.handler.current_loader = loader  # Store loader reference for dynamic updates
            idle_message = getattr(loader, "messages", ["Working..."])[0]
            self.handler.loader_idle_message = idle_message
            self.handler._set_loader_message(idle_message)

            def stop_loader():
                try:
                    loader.stop()
                except Exception:
                    pass

            # Stop the loader when the first tool call (or permission prompt) is displayed.
            self.handler.on_first_tool = stop_loader
            # Also stop the loader when the assistant starts printing visible text.
            # This prevents the spinner from overlapping the streamed output in text-only turns.
            self.handler.on_first_output = stop_loader
            try:
                # Use retry with backoff for network resilience
                def make_request():
                    return self.client.chat.completions.create(**request_kwargs)

                stream = retry_with_backoff(make_request, max_retries=3)

                # Process stream
                assistant_message, tool_results, is_complete = self.handler.process_stream(
                    stream, messages
                )
            except Exception as e:
                # Stop loader IMMEDIATELY before showing any error
                loader.stop()

                error_msg = str(e)

                # Check if user pressed Esc to interrupt (message already shown by loader)
                if loader.was_interrupted:
                    return {
                        "success": False,
                        "error": "User interrupted",
                        "summary": "",
                        "next_steps": ""
                    }

                if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    self.console.print("\n[red]⚠[/red] API timed out after 120 seconds")
                    self.console.print("[yellow]Tip:[/yellow] Large models can be slow. Try again or use a faster model.\n")
                    return {
                        "success": False,
                        "error": "API timeout",
                        "summary": "Request timed out",
                        "next_steps": ""
                    }
                elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                    self.console.print("[red]⚠[/red] Network connection failed after 3 retries\n")
                    self.console.print("[yellow]Tip:[/yellow] Check your internet connection and try again.\n")
                    return {
                        "success": False,
                        "error": "Connection error",
                        "summary": "Network connection failed",
                        "next_steps": ""
                    }
                else:
                    self.console.print(f"\n[red]⚠ API Error:[/red] {error_msg}\n")
                    return {
                        "success": False,
                        "error": error_msg,
                        "summary": "API call failed",
                        "next_steps": ""
                    }
            finally:
                loader.stop()
                self.handler.on_first_output = None
                self.handler.on_first_tool = None

            # Check if user pressed Esc to interrupt (message already shown by loader)
            if loader.was_interrupted:
                return {
                    "success": False,
                    "error": "User interrupted",
                    "summary": "",
                    "next_steps": ""
                }

            # Add assistant message
            if assistant_message.strip():
                messages.append({
                    "role": "assistant",
                    "content": assistant_message
                })

            # If no tool calls, we're done
            if not tool_results:
                break

            # Mark that tools were used
            any_tools_used = True

            # Check if all tools in this turn failed
            all_failed = all(not result.get("success", False) for result in tool_results)
            if all_failed:
                consecutive_errors += 1

                # Detect repeated error (death loop)
                current_error = tool_results[0].get("error", "") if tool_results else ""
                failed_tool = tool_results[0].get("name", "") if tool_results else ""

                if current_error and current_error == last_error_message and consecutive_errors >= 2:
                    # Generate specific hint based on error type
                    hint = self.handler._generate_error_hint(current_error, failed_tool)
                    self.console.print(f"[yellow]⚠ Repeated error - here's a hint:[/yellow] {hint}\n")
                    messages.append({
                        "role": "user",
                        "content": f"STOP! You made the same error twice: \"{current_error}\"\n\nHINT: {hint}\n\nDo NOT repeat the same call. Fix the issue first."
                    })
                last_error_message = current_error

                if consecutive_errors >= max_consecutive_errors:
                    self.console.print(f"[yellow]⚠ Stopping due to {consecutive_errors} consecutive tool errors[/yellow]\n")
                    break
            else:
                consecutive_errors = 0  # Reset on success
                last_error_message = None  # Clear error tracking

            # Add assistant tool call message
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": result["tool_call_id"],
                        "type": "function",
                        "function": {
                            "name": result["name"],
                            "arguments": "{}"  # Already executed
                        }
                    }
                    for result in tool_results
                ]
            })

            # Add tool results
            for result in tool_results:
                payload = {
                    "success": result["success"],
                    "message": result.get("message", ""),
                    "error": result.get("error")
                }
                for key in ("content", "files", "line_count", "count", "output", "results"):
                    if key in result:
                        payload[key] = result[key]
                messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": json.dumps(payload)
                })

            # Check if complete
            if is_complete:
                # Get final summary from the completed result
                for result in tool_results:
                    if result.get("is_complete"):
                        return {
                            "success": True,
                            "summary": result.get("summary", ""),
                            "next_steps": result.get("next_steps", ""),
                            "properly_completed": True
                        }
                break

        # Loop ended without complete_generation
        # Don't show warning if user cancelled or for chat/search mode (no completion expected)
        is_chat_mode = self.handler.loader_type == "chat"
        if any_tools_used and not self.handler.user_cancelled and not is_chat_mode:
            self.console.print()
            self.console.print("[yellow]⚠ Task may be incomplete. Type [bold]/continue[/bold] to resume.[/yellow]\n")
            return {
                "success": True,
                "summary": "Generation incomplete",
                "next_steps": "",
                "properly_completed": False
            }
        else:
            return {
                "success": True,
                "summary": "",
                "next_steps": "",
                "properly_completed": True
            }
