"""
Tool Executors
Executes the actual operations for each tool call.
"""

import os
import shutil
import subprocess
import time
import random
import re
import json
from pathlib import Path
from typing import Dict, Any, Union, Set, Optional, List
from datetime import datetime
from loguru import logger

# Import session state tracker
try:
    from nasea.core.session_state import SessionStateTracker
    HAS_SESSION_STATE = True
except ImportError:
    HAS_SESSION_STATE = False
    SessionStateTracker = None  # type: ignore

# Import config for session tracking setting
try:
    from nasea.core.config import config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    config = None  # type: ignore

try:
    from tavily import TavilyClient
    HAS_TAVILY = True
except ImportError:
    HAS_TAVILY = False

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

# Fallback to DuckDuckGo if Tavily not available
try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    try:
        from ddgs import DDGS
        HAS_DDG = True
    except ImportError:
        HAS_DDG = False

# Check for ripgrep
try:
    subprocess.run(["rg", "--version"], capture_output=True, check=True)
    HAS_RIPGREP = True
except (subprocess.CalledProcessError, FileNotFoundError):
    HAS_RIPGREP = False


class ToolExecutor:
    """Executes tool calls and returns results."""

    def __init__(self, project_root: Path, memory_db_path: Optional[Path] = None, session_tracker: Optional["SessionStateTracker"] = None):
        """
        Initialize tool executor.

        Args:
            project_root: Root directory for the project being generated
            memory_db_path: Path to memory database (defaults to project_root/.nasea_memory.json)
            session_tracker: Optional session state tracker for avoiding redundant work
        """
        self.project_root = Path(project_root).resolve()
        self.project_root.mkdir(parents=True, exist_ok=True)
        # Output directory (parent of project) - allowed for project-level operations like rename
        self.output_root = self.project_root.parent
        self.max_retries = 3  # Maximum retry attempts for transient failures
        self._written_paths: Set[str] = set()

        # Memory storage for think/memory tools
        self._memory_path = memory_db_path or (self.project_root / ".nasea_memory.json")
        self._memory: Dict[str, Any] = self._load_memory()
        self._thoughts: list = []  # Store thinking steps

        # Session state tracker for avoiding redundant work
        # Controlled by config.session_tracking: "auto", "always", "never"
        self._session_tracker = session_tracker
        self._session_tracking_mode = "auto"  # Default
        if HAS_CONFIG and config:
            self._session_tracking_mode = getattr(config, 'session_tracking', 'auto')

        if self._session_tracking_mode == "never":
            self._session_tracker = None
        elif self._session_tracking_mode == "always" and session_tracker is None and HAS_SESSION_STATE:
            self._session_tracker = SessionStateTracker(self.project_root)
        # For "auto" mode, we defer creation until we detect complexity
        self._files_touched = 0  # Track complexity for auto mode
        self._auto_threshold = 3  # Enable after 3+ files touched

        # read_file behavior:
        # - Small files are returned fully.
        # - Large files are truncated by default unless offset/limit is provided.
        # This reduces token blowups and nudges the model toward grep_search + targeted reads.
        self._default_full_read_max_lines = 240
        self._default_truncated_head_lines = 200
        self._default_truncated_tail_lines = 40

    def _load_memory(self) -> Dict[str, Any]:
        """Load memory from disk."""
        if self._memory_path.exists():
            try:
                return json.loads(self._memory_path.read_text())
            except Exception:
                return {"entries": {}, "thoughts": []}
        return {"entries": {}, "thoughts": []}

    def _save_memory(self) -> None:
        """Save memory to disk."""
        try:
            self._memory_path.write_text(json.dumps(self._memory, indent=2))
        except Exception:
            pass  # Non-critical failure

    def get_session_state_summary(self) -> str:
        """Get session state summary for injection into system prompt."""
        if self._session_tracker:
            return self._session_tracker.get_state_summary()
        return ""

    def add_issue(self, issue_id: str, description: str, file_path: Optional[str] = None, line_number: Optional[int] = None):
        """Track an identified issue."""
        if self._session_tracker:
            self._session_tracker.add_issue(issue_id, description, file_path, line_number)

    def mark_issue_fixed(self, issue_id: str, fix_description: Optional[str] = None):
        """Mark an issue as fixed."""
        if self._session_tracker:
            self._session_tracker.mark_issue_fixed(issue_id, fix_description)

    def set_checkpoint(self, task: str, current_step: str, completed: List[str], pending: List[str]):
        """Set a checkpoint for task continuation."""
        if self._session_tracker:
            self._session_tracker.set_checkpoint(task, current_step, completed, pending)

    def add_context_note(self, note: str):
        """Add an important finding/note."""
        if self._session_tracker:
            self._session_tracker.add_context_note(note)

    def _relative_str(self, path: Path) -> str:
        """Return a project-root-relative string for tracking purposes."""
        try:
            rel = path.relative_to(self.project_root)
        except ValueError:
            # Should never happen due to resolve_path guard, but fallback to absolute
            rel = path
        return str(rel)

    def _remember_write(self, path: Path) -> None:
        """Track files that have been written in this session."""
        self._written_paths.add(self._relative_str(path))

    def _forget_write(self, path: Path) -> None:
        """Stop tracking a file or directory that was deleted."""
        rel = self._relative_str(path)
        to_remove = {p for p in self._written_paths if p == rel or p.startswith(f"{rel}/")}
        self._written_paths.difference_update(to_remove)

    def _rename_tracked_path(self, source: Path, target: Path) -> None:
        """Update tracked write paths when files/directories are renamed."""
        source_rel = self._relative_str(source)
        target_rel = self._relative_str(target)
        updated = set()
        additions = set()
        for path in self._written_paths:
            if path == source_rel or path.startswith(f"{source_rel}/"):
                suffix = path[len(source_rel):]
                if suffix.startswith("/"):
                    suffix = suffix[1:]
                additions.add(f"{target_rel}/{suffix}" if suffix else target_rel)
                updated.add(path)
        if updated:
            self._written_paths.difference_update(updated)
            self._written_paths.update(additions)

    def resolve_path(self, path_value: Union[str, Path, None]) -> Path:
        """Public wrapper exposing safe path resolution."""
        return self._resolve_path(path_value)

    def _resolve_path(self, path_value: Union[str, Path, None]) -> Path:
        """
        Resolve user-provided paths relative to the project root while preventing escape.

        The agent sometimes prepends the project directory (e.g., "output/app") to paths
        even though tool calls are already sandboxed inside that directory. This helper
        normalizes those inputs and guards against traversal outside the project root.
        """
        if path_value in (None, "", "."):
            return self.project_root

        raw_path = Path(path_value)

        if raw_path.is_absolute():
            try:
                raw_path = raw_path.relative_to(self.project_root)
            except ValueError:
                raise ValueError("Absolute paths must be inside the project directory")
        else:
            # Check if the relative path starts with any suffix of the project root
            # e.g., if project_root is /Users/.../output/tic-tac-toe
            # and candidate is "output/tic-tac-toe/index.html"
            # we should strip "output/tic-tac-toe" from the beginning
            root_parts = self.project_root.parts
            candidate_parts = raw_path.parts

            # First check: full path match (original logic)
            if len(candidate_parts) >= len(root_parts) and candidate_parts[:len(root_parts)] == root_parts:
                remaining = candidate_parts[len(root_parts):]
                raw_path = Path(*remaining) if remaining else Path(".")
            else:
                # Second check: match against suffixes of the project root
                # This handles cases like "output/tic-tac-toe/file.txt" when project_root ends with "output/tic-tac-toe"
                for i in range(len(root_parts)):
                    suffix = root_parts[i:]
                    if len(candidate_parts) >= len(suffix) and candidate_parts[:len(suffix)] == suffix:
                        remaining = candidate_parts[len(suffix):]
                        raw_path = Path(*remaining) if remaining else Path(".")
                        break

        target_path = (self.project_root / raw_path).resolve()
        try:
            target_path.relative_to(self.project_root)
        except ValueError:
            raise ValueError("Path escapes the project directory")

        return target_path

    def _is_retryable_error(self, error_msg: str) -> bool:
        """
        Determine if an error is transient and worth retrying.

        Args:
            error_msg: Error message from failed tool execution

        Returns:
            True if the error is likely transient, False otherwise
        """
        # Don't retry validation errors, permission errors, or missing files
        non_retryable_patterns = [
            "Missing required argument",
            "Path escapes the project directory",
            "Absolute paths must be inside",
            "File not found",
            "not found",
            "does not exist",
            "Permission denied",
            "Invalid",
            "Validation failed",
        ]

        error_lower = error_msg.lower()
        for pattern in non_retryable_patterns:
            if pattern.lower() in error_lower:
                return False

        # Retry on I/O errors, network issues, temporary file locks
        retryable_patterns = [
            "busy",
            "locked",
            "timeout",
            "temporarily unavailable",
            "resource temporarily unavailable",
            "try again",
        ]

        for pattern in retryable_patterns:
            if pattern.lower() in error_lower:
                return True

        # Default: retry generic errors
        return True

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call with automatic retry on transient failures.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Result dictionary with success, message, and optional data
        """
        last_result = None

        for attempt in range(self.max_retries):
            result = self._execute_once(tool_name, arguments)

            # Success - return immediately
            if result.get("success"):
                return result

            # Failed - check if retryable
            last_result = result
            error_msg = result.get("error", "")

            # Don't retry if error is not transient
            if not self._is_retryable_error(error_msg):
                return result

            # Don't retry on last attempt
            if attempt < self.max_retries - 1:
                # Exponential backoff with jitter: 2^attempt + random(0, 1) seconds
                backoff = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(backoff)
                # Note: We don't log retries to avoid cluttering output
                # The streaming handler will show the final result

        # All retries exhausted - return last result
        return last_result

    def _execute_once(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call once (internal method without retry).

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Result dictionary with success, message, and optional data
        """
        executor_map = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "edit_file": self._edit_file,
            "list_files": self._list_files,
            "run_command": self._run_command,
            "create_directory": self._create_directory,
            "delete_path": self._delete_path,
            "rename_path": self._rename_path,
            "complete_generation": self._complete_generation,
            "web_search": self._web_search,
            # New tools (adapted from OpenCoder)
            "grep_search": self._grep_search,
            "think": self._think,
            "memory_save": self._memory_save,
            "memory_read": self._memory_read,
            "check_syntax": self._check_syntax_tool,
            "open_browser": self._open_browser,
        }

        executor = executor_map.get(tool_name)
        if not executor:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }

        try:
            return executor(arguments)
        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            }

    def _read_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Read a file's contents with optional offset and limit."""
        if "file_path" not in args:
            return {
                "success": False,
                "error": "Missing required argument 'file_path'"
            }

        try:
            file_path = self._resolve_path(args["file_path"])
        except ValueError as error:
            return {
                "success": False,
                "error": str(error)
            }

        if not file_path.exists():
            return {
                "success": False,
                "error": f"File not found: {args['file_path']}"
            }

        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.splitlines()
            total_lines = len(lines)

            # Auto-enable session tracking for complex tasks
            self._files_touched += 1
            if (self._session_tracking_mode == "auto" and
                self._session_tracker is None and
                self._files_touched >= self._auto_threshold and
                HAS_SESSION_STATE):
                self._session_tracker = SessionStateTracker(self.project_root)
                logger.debug(f"Auto-enabled session tracking (complexity: {self._files_touched} files)")

            # Check if file was already read and unchanged (avoid redundant work)
            redundant_read = False
            if self._session_tracker:
                if self._session_tracker.was_file_read(str(file_path)):
                    if not self._session_tracker.is_file_changed(str(file_path), content):
                        redundant_read = True
                        prev_summary = self._session_tracker.get_file_summary(str(file_path))
                        logger.debug(f"Redundant read detected: {file_path} (unchanged)")
                # Record this read
                self._session_tracker.record_file_read(str(file_path), content)

            # Support offset and limit for large files
            offset = args.get("offset", 0)
            limit = args.get("limit")

            if offset > 0 or limit is not None:
                start = max(0, offset)
                end = start + int(limit) if limit is not None else total_lines
                selected_lines = lines[start:end]
                content = "\n".join(selected_lines)
                line_count = len(selected_lines)
                message = f"Read lines {start+1}-{min(end, total_lines)} of {total_lines}"
                return {
                    "success": True,
                    "content": content,
                    "line_count": line_count,
                    "total_lines": total_lines,
                    "file_path": args["file_path"],
                    "message": message
                }

            # Default behavior when offset/limit not provided:
            # return full content only for smaller files; otherwise truncate to head+tail.
            if total_lines <= self._default_full_read_max_lines:
                message = f"Read {total_lines} lines (full file)"
                if redundant_read:
                    message += " ⚠️ NOTE: This file was already read and hasn't changed. Avoid re-reading unchanged files to save tokens."
                return {
                    "success": True,
                    "content": "\n".join(lines),
                    "line_count": total_lines,
                    "total_lines": total_lines,
                    "file_path": args["file_path"],
                    "message": message,
                    "redundant_read": redundant_read
                }

            head_n = min(self._default_truncated_head_lines, total_lines)
            tail_n = min(self._default_truncated_tail_lines, max(0, total_lines - head_n))
            head = lines[:head_n]
            tail = lines[-tail_n:] if tail_n else []
            truncated_content = "\n".join(head)
            if tail:
                truncated_content += "\n\n… (truncated) …\n\n" + "\n".join(tail)

            message = (
                f"Read {head_n}+{tail_n} lines (truncated) of {total_lines}. "
                "For large files, use grep_search to locate lines, then read_file with offset+limit."
            )
            return {
                "success": True,
                "content": truncated_content,
                "line_count": head_n + tail_n,
                "total_lines": total_lines,
                "file_path": args["file_path"],
                "message": message,
                "truncated": True,
                "truncation": {"head": head_n, "tail": tail_n}
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read file: {str(e)}"
            }

    def _write_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Write content to a file."""
        # Validate required arguments
        if "file_path" not in args:
            return {
                "success": False,
                "error": "Missing required argument 'file_path'. Please retry with both file_path and content."
            }
        if "content" not in args:
            return {
                "success": False,
                "error": f"Missing required argument 'content' for file '{args.get('file_path')}'. Please retry with file content."
            }

        try:
            file_path = self._resolve_path(args["file_path"])
        except ValueError as error:
            return {
                "success": False,
                "error": str(error)
            }

        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if overwriting existing file
        is_overwrite = file_path.exists() or self._relative_str(file_path) in self._written_paths

        try:
            content = args["content"]
            file_path.write_text(content, encoding="utf-8")
            lines = content.splitlines()
            line_count = len(lines)

            action = "Overwrote" if is_overwrite else "Created"
            result = {
                "success": True,
                "line_count": line_count,
                "file_path": args["file_path"],
                "message": f"{action} file ({line_count} lines)",
                "is_overwrite": is_overwrite,
                "preview_lines": lines[:5] if not is_overwrite else None,  # Brief preview for new files
            }
            self._remember_write(file_path)

            # Auto-verify syntax for supported file types
            syntax_error = self._check_syntax(file_path, content)
            if syntax_error:
                result["syntax_warning"] = syntax_error

            return result
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to write file: {str(e)}"
            }

    def _edit_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Edit a file by replacing content."""
        # Validate required arguments
        if "file_path" not in args:
            return {
                "success": False,
                "error": "Missing required argument 'file_path'. Please retry with file_path, old_content, and new_content."
            }
        if "old_content" not in args:
            return {
                "success": False,
                "error": f"Missing required argument 'old_content' for file '{args.get('file_path')}'. Please retry with old_content and new_content."
            }
        if "new_content" not in args:
            return {
                "success": False,
                "error": f"Missing required argument 'new_content' for file '{args.get('file_path')}'. Please retry with old_content and new_content."
            }

        try:
            file_path = self._resolve_path(args["file_path"])
        except ValueError as error:
            return {
                "success": False,
                "error": str(error)
            }

        if not file_path.exists():
            return {
                "success": False,
                "error": f"File not found: {args['file_path']}. Use write_file to create a new file instead."
            }

        try:
            content = file_path.read_text(encoding="utf-8")
            old_line_count = len(content.splitlines())

            # Check if old content exists
            if args["old_content"] not in content:
                return {
                    "success": False,
                    "error": "Content not found - re-reading file to get exact match"
                }

            # Reject no-op edits (old_content == new_content)
            if args["old_content"] == args["new_content"]:
                return {
                    "success": False,
                    "error": "No changes needed - content already matches"
                }

            # Replace content
            new_content = content.replace(args["old_content"], args["new_content"], 1)
            new_line_count = len(new_content.splitlines())
            file_path.write_text(new_content, encoding="utf-8")

            result = {
                "success": True,
                "message": "File updated successfully",
                "old_line_count": old_line_count,
                "new_line_count": new_line_count,
                "file_path": args["file_path"],
                "old_content": args["old_content"],  # For diff display
                "new_content": args["new_content"]   # For diff display
            }

            # Auto-verify syntax after edit
            syntax_error = self._check_syntax(file_path, new_content)
            if syntax_error:
                result["syntax_warning"] = syntax_error

            return result
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to edit file: {str(e)}"
            }

    def _list_files(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List files in a directory."""
        directory = args.get("directory", ".")
        try:
            dir_path = self._resolve_path(directory)
        except ValueError as error:
            return {
                "success": False,
                "error": str(error)
            }

        if not dir_path.exists():
            return {
                "success": False,
                "error": f"Directory not found: {directory}"
            }

        try:
            files = []
            for item in dir_path.iterdir():
                if item.name.startswith('.'):
                    continue
                rel_path = item.relative_to(self.project_root)
                files.append({
                    "path": str(rel_path),
                    "type": "directory" if item.is_dir() else "file"
                })

            return {
                "success": True,
                "files": files,
                "count": len(files),
                "message": f"Found {len(files)} items in {directory or '.'}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to list directory: {str(e)}"
            }

    def _run_command(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a shell command."""
        command = args["command"]
        try:
            working_dir = self._resolve_path(args.get("working_directory", "."))
        except ValueError as error:
            return {
                "success": False,
                "error": str(error)
            }

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=30
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": stdout or stderr,
                    "command": command,
                    "return_code": 0
                }
            else:
                return {
                    "success": False,
                    "error": stderr or stdout or f"Command failed with exit code {result.returncode}",
                    "command": command,
                    "return_code": result.returncode
                }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command timed out (30s limit)"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to run command: {str(e)}"
            }

    def _create_directory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create a directory."""
        try:
            dir_path = self._resolve_path(args["directory_path"])
        except ValueError as error:
            return {
                "success": False,
                "error": str(error)
            }

        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            return {
                "success": True,
                "directory_path": args["directory_path"],
                "message": f"Created directory: {args['directory_path']}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create directory: {str(e)}"
            }

    def _delete_path(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a file or directory."""
        path_arg = args.get("path")
        if not path_arg:
            return {
                "success": False,
                "error": "Missing required argument 'path'."
            }

        try:
            target_path = self._resolve_path(path_arg)
        except ValueError as error:
            return {"success": False, "error": str(error)}

        if not target_path.exists():
            return {
                "success": False,
                "error": f"Path not found: {path_arg}"
            }

        try:
            if target_path.is_dir():
                if args.get("recursive", False):
                    shutil.rmtree(target_path)
                elif any(target_path.iterdir()):
                    return {
                        "success": False,
                        "error": "Directory is not empty. Pass recursive=true to delete it."
                    }
                else:
                    target_path.rmdir()
                deleted_type = "directory"
            else:
                target_path.unlink()
                deleted_type = "file"
            self._forget_write(target_path)

            return {
                "success": True,
                "message": f"Deleted {deleted_type}: {path_arg}"
            }
        except Exception as exc:
            return {
                "success": False,
                "error": f"Failed to delete path: {exc}"
            }

    def _resolve_output_path(self, path_value: Union[str, Path, None]) -> Path:
        """
        Resolve paths within the output directory (allows project-level operations).
        Used for rename operations that may involve the project folder itself.
        """
        if path_value in (None, "", "."):
            return self.project_root

        raw_path = Path(path_value)

        if raw_path.is_absolute():
            resolved = raw_path.resolve()
        else:
            # Relative paths are resolved from output_root
            resolved = (self.output_root / raw_path).resolve()

        # Ensure it stays within output_root
        try:
            resolved.relative_to(self.output_root)
        except ValueError:
            raise ValueError("Path must be inside the output directory")

        return resolved

    def _rename_path(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Rename or move a file/directory."""
        source = args.get("source_path")
        target = args.get("target_path")

        if not source or not target:
            return {
                "success": False,
                "error": "Missing required arguments 'source_path' and 'target_path'."
            }

        # Check if this is a project-level rename (renaming the project folder itself)
        is_project_rename = False
        try:
            # First try normal resolution within project
            source_path = self._resolve_path(source)
            target_path = self._resolve_path(target)
        except ValueError:
            # If that fails, try output-level resolution for project renames
            try:
                source_path = self._resolve_output_path(source)
                target_path = self._resolve_output_path(target)
                is_project_rename = True
            except ValueError as error:
                return {"success": False, "error": str(error)}

        if not source_path.exists():
            return {
                "success": False,
                "error": f"Source not found: {source}"
            }

        if target_path.exists() and not args.get("overwrite", False):
            return {
                "success": False,
                "error": f"Target already exists: {target}. Pass overwrite=true to replace it."
            }

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.replace(target_path)

            # Update project_root if we renamed the project folder
            if is_project_rename and source_path == self.project_root:
                self.project_root = target_path
                self._written_paths.clear()  # Reset tracking for new project path
            else:
                self._rename_tracked_path(source_path, target_path)

            return {
                "success": True,
                "message": f"Renamed {source} → {target}"
            }
        except Exception as exc:
            return {
                "success": False,
                "error": f"Failed to rename path: {exc}"
            }

    def _complete_generation(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mark generation as complete after checking for syntax errors."""
        # Validate required arguments
        if "summary" not in args:
            return {
                "success": False,
                "error": "Missing required argument 'summary'. Please provide a brief summary of what was created/changed."
            }

        # Auto-check syntax on all project files before completing
        syntax_errors = []
        syntax_warnings = []
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file() and file_path.suffix in (".py", ".js", ".mjs", ".json", ".html", ".htm"):
                try:
                    content = file_path.read_text(encoding="utf-8")
                    error = self._check_syntax(file_path, content)
                    if error:
                        rel_path = file_path.relative_to(self.project_root)
                        # Separate warnings from errors - warnings don't block completion
                        if "warning" in error.lower():
                            syntax_warnings.append(f"{rel_path}: {error}")
                        else:
                            syntax_errors.append(f"{rel_path}: {error}")
                except Exception:
                    pass  # Skip unreadable files

        if syntax_errors:
            return {
                "success": False,
                "error": f"Cannot complete - syntax errors found:\n" + "\n".join(syntax_errors[:5]),
                "syntax_errors": syntax_errors,
                "message": "Fix syntax errors before completing"
            }

        result = {
            "success": True,
            "summary": args["summary"],
            "next_steps": args.get("next_steps", ""),
            "message": "Generation complete",
            "is_complete": True
        }

        # Include warnings in response (informational only, doesn't block)
        if syntax_warnings:
            result["warnings"] = syntax_warnings

        return result

    def _web_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search the web using Tavily (primary) or DuckDuckGo (fallback)."""
        query = args.get("query")
        if not query:
            return {"success": False, "error": "Missing query argument"}

        # Track what's available for error reporting
        tavily_status = "not_imported"
        ddg_status = "not_checked"

        # Try Tavily first (better for AI agents)
        try:
            from tavily import TavilyClient as TC
            tavily_status = "imported"

            # Try multiple sources for the API key
            api_key = os.getenv("TAVILY_API_KEY")
            key_source = "os.environ" if api_key else None

            if not api_key:
                # Try loading from .env directly
                from dotenv import dotenv_values
                env_paths = [
                    Path.cwd() / ".env",
                    Path(__file__).parent.parent.parent / ".env"
                ]
                for env_path in env_paths:
                    if env_path.exists():
                        env_vars = dotenv_values(env_path)
                        api_key = env_vars.get("TAVILY_API_KEY")
                        if api_key:
                            key_source = str(env_path)
                            break

            if api_key:
                tavily_status = f"has_key_from_{key_source}"
                try:
                    client = TC(api_key=api_key)
                    response = client.search(query, max_results=5)
                    results = response.get("results", [])

                    formatted = ""
                    for r in results:
                        formatted += f"Title: {r.get('title', 'No title')}\n"
                        formatted += f"URL: {r.get('url', '')}\n"
                        formatted += f"Content: {r.get('content', '')}\n\n"

                    return {
                        "success": True,
                        "output": formatted,
                        "results": results,
                        "message": f"Found {len(results)} results for '{query}'"
                    }
                except Exception as e:
                    tavily_status = f"search_error: {e}"
            else:
                tavily_status = "no_api_key"
        except ImportError as ie:
            tavily_status = f"import_error: {ie}"

        # Fallback to DuckDuckGo
        DDG = None
        try:
            from duckduckgo_search import DDGS as DDG
            ddg_status = "imported"
        except ImportError:
            try:
                from ddgs import DDGS as DDG
                ddg_status = "imported_ddgs"
            except ImportError:
                ddg_status = "not_available"

        if DDG:
            try:
                results = DDG().text(query, max_results=5)
                formatted = ""
                for r in results:
                    formatted += f"Title: {r['title']}\nURL: {r['href']}\nContent: {r['body']}\n\n"
                return {
                    "success": True,
                    "output": formatted,
                    "message": f"Found {len(results)} results for '{query}'"
                }
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "limit" in error_str:
                    return {"success": False, "error": "Search rate limit reached. Please wait a moment and try again."}
                elif "timeout" in error_str:
                    return {"success": False, "error": "Search timed out. Check your internet connection."}
                elif "connection" in error_str or "network" in error_str:
                    return {"success": False, "error": "No internet connection. Please check your network."}
                else:
                    return {"success": False, "error": f"Search failed. Try again or check internet connection."}

        return {"success": False, "error": "Web search not available. Run: pip install duckduckgo-search"}

    def _open_browser(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Open a file or URL in browser and capture console output."""
        import webbrowser

        path = args.get("path")
        if not path:
            return {"success": False, "error": "Missing required argument 'path'"}

        try:
            # Check if it's a URL or file path
            if path.startswith(("http://", "https://")):
                url = path
            else:
                # It's a file path - resolve it
                file_path = self._resolve_path(path)
                if not file_path.exists():
                    return {"success": False, "error": f"File not found: {path}"}
                # Convert to file:// URL
                url = f"file://{file_path.resolve()}"

            # Try to capture console with Playwright if available
            console_logs = []
            page_errors = []

            if HAS_PLAYWRIGHT:
                try:
                    with sync_playwright() as p:
                        browser = p.chromium.launch(headless=True)
                        page = browser.new_page()

                        # Capture console messages
                        def handle_console(msg):
                            console_logs.append({
                                "type": msg.type,
                                "text": msg.text
                            })

                        # Capture page errors
                        def handle_error(error):
                            page_errors.append(str(error))

                        page.on("console", handle_console)
                        page.on("pageerror", handle_error)

                        # Navigate and wait for page to load
                        page.goto(url, wait_until="networkidle", timeout=10000)

                        # Wait a bit for any async JS to run
                        page.wait_for_timeout(1000)

                        browser.close()
                except Exception as e:
                    # Playwright failed, continue without console capture
                    console_logs.append({"type": "warning", "text": f"Console capture failed: {str(e)}"})

            # Also open in real browser for user to see
            webbrowser.open(url)

            # Build result
            result = {
                "success": True,
                "message": f"Opened in browser: {path}",
                "url": url
            }

            # Add console output if any
            if console_logs:
                errors = [log for log in console_logs if log["type"] in ("error", "warning")]
                info = [log for log in console_logs if log["type"] not in ("error", "warning")]

                if errors:
                    result["console_errors"] = errors
                if info:
                    result["console_info"] = info[:10]  # Limit info messages

            if page_errors:
                result["page_errors"] = page_errors

            # Summary for agent
            error_count = len([l for l in console_logs if l["type"] == "error"]) + len(page_errors)
            warning_count = len([l for l in console_logs if l["type"] == "warning"])

            if error_count > 0:
                result["status"] = f"Page has {error_count} error(s) - check console_errors and page_errors"
            elif warning_count > 0:
                result["status"] = f"Page loaded with {warning_count} warning(s)"
            else:
                result["status"] = "Page loaded successfully with no errors"

            return result
        except Exception as e:
            return {"success": False, "error": f"Failed to open browser: {str(e)}"}

    # === NEW TOOLS (Adapted from OpenCoder) ===

    def _grep_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Fast regex search using ripgrep or fallback to Python."""
        pattern = args.get("pattern")
        if not pattern:
            return {"success": False, "error": "Missing required argument 'pattern'"}

        search_path = args.get("path", ".")
        try:
            search_dir = self._resolve_path(search_path)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        file_type = args.get("file_type")
        case_sensitive = args.get("case_sensitive", False)

        if HAS_RIPGREP:
            # Use ripgrep for speed
            cmd = ["rg", "--line-number", "--no-heading", "--color=never"]
            if not case_sensitive:
                cmd.append("-i")
            if file_type:
                cmd.extend(["-t", file_type])
            cmd.extend(["--max-count=50", pattern, str(search_dir)])

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30
                )
                matches = result.stdout.strip()
                if not matches:
                    return {
                        "success": True,
                        "matches": [],
                        "count": 0,
                        "message": f"No matches found for '{pattern}'"
                    }

                lines = matches.split("\n")
                return {
                    "success": True,
                    "matches": lines[:50],  # Limit results
                    "count": len(lines),
                    "message": f"Found {len(lines)} matches for '{pattern}'"
                }
            except subprocess.TimeoutExpired:
                return {"success": False, "error": "Search timed out (30s limit)"}
            except Exception as e:
                return {"success": False, "error": f"Search failed: {str(e)}"}
        else:
            # Python fallback using re
            try:
                regex = re.compile(pattern, re.IGNORECASE if not case_sensitive else 0)
            except re.error as e:
                return {"success": False, "error": f"Invalid regex: {str(e)}"}

            matches = []
            try:
                for file_path in search_dir.rglob("*"):
                    if file_path.is_file():
                        # Filter by type if specified
                        if file_type and not file_path.suffix.lstrip(".") == file_type:
                            continue
                        # Skip binary and hidden files
                        if file_path.name.startswith("."):
                            continue
                        try:
                            content = file_path.read_text(encoding="utf-8", errors="ignore")
                            for i, line in enumerate(content.splitlines(), 1):
                                if regex.search(line):
                                    rel_path = file_path.relative_to(self.project_root)
                                    matches.append(f"{rel_path}:{i}:{line.strip()}")
                                    if len(matches) >= 50:
                                        break
                        except Exception:
                            continue
                    if len(matches) >= 50:
                        break

                return {
                    "success": True,
                    "matches": matches,
                    "count": len(matches),
                    "message": f"Found {len(matches)} matches for '{pattern}'"
                }
            except Exception as e:
                return {"success": False, "error": f"Search failed: {str(e)}"}

    def _think(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Document reasoning before taking action."""
        thought = args.get("thought")
        if not thought:
            return {"success": False, "error": "Missing required argument 'thought'"}

        # Normalize thought to string (model sometimes sends dict)
        if isinstance(thought, dict):
            thought = str(thought)
        elif not isinstance(thought, str):
            thought = str(thought)

        plan = args.get("plan", [])
        # Normalize plan to list
        if isinstance(plan, str):
            plan = [plan]
        elif not isinstance(plan, list):
            plan = []
        timestamp = datetime.now().isoformat()

        thought_entry = {
            "timestamp": timestamp,
            "thought": thought,
            "plan": plan
        }

        # Store in memory
        self._thoughts.append(thought_entry)
        if "thoughts" not in self._memory:
            self._memory["thoughts"] = []
        self._memory["thoughts"].append(thought_entry)
        self._save_memory()

        # Format response
        response = f"Recorded thought: {thought}"
        if plan:
            response += f"\nPlan steps: {len(plan)}"
            for i, step in enumerate(plan, 1):
                response += f"\n  {i}. {step}"

        return {
            "success": True,
            "thought": thought,
            "plan": plan,
            "message": response
        }

    def _memory_save(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Save information to persistent memory."""
        key = args.get("key")
        value = args.get("value")

        if not key:
            return {"success": False, "error": "Missing required argument 'key'"}
        if not value:
            return {"success": False, "error": "Missing required argument 'value'"}

        category = args.get("category", "general")
        timestamp = datetime.now().isoformat()

        if "entries" not in self._memory:
            self._memory["entries"] = {}

        self._memory["entries"][key] = {
            "value": value,
            "category": category,
            "timestamp": timestamp
        }
        self._save_memory()

        return {
            "success": True,
            "key": key,
            "category": category,
            "message": f"Saved to memory: {key} = {value[:100]}{'...' if len(value) > 100 else ''}"
        }

    def _check_syntax_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool wrapper for syntax checking - can be called by AI."""
        file_path = args.get("file_path")
        if not file_path:
            return {
                "success": False,
                "error": "Missing required argument 'file_path'. Use a FILE name like 'index.html', not a directory."
            }

        try:
            resolved_path = self._resolve_path(file_path)
        except ValueError as error:
            return {
                "success": False,
                "error": str(error)
            }

        # Check if it's a directory instead of a file
        if resolved_path.is_dir():
            # List files in directory to help the model
            files = [f.name for f in resolved_path.iterdir() if f.is_file() and f.suffix in ('.html', '.js', '.css', '.py', '.json')][:5]
            file_list = ", ".join(files) if files else "no checkable files"
            return {
                "success": False,
                "error": f"'{file_path}' is a DIRECTORY, not a file. Use a filename like: {file_list}"
            }

        if not resolved_path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }

        try:
            content = resolved_path.read_text(encoding="utf-8")
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read file: {str(e)}"
            }

        error = self._check_syntax(resolved_path, content)
        if error:
            return {
                "success": True,
                "has_errors": True,
                "error_details": error,
                "file_path": file_path,
                "message": f"Syntax error found: {error}"
            }
        else:
            return {
                "success": True,
                "has_errors": False,
                "file_path": file_path,
                "message": f"No syntax errors found in {file_path}"
            }

    def _check_syntax(self, file_path: Path, content: str) -> Optional[str]:
        """
        Check syntax of a file after writing.
        Returns error message if syntax error found, None otherwise.
        """
        suffix = file_path.suffix.lower()

        # Python files
        if suffix == ".py":
            try:
                compile(content, str(file_path), "exec")
                return None
            except SyntaxError as e:
                return f"Python syntax error at line {e.lineno}: {e.msg}"

        # JavaScript/TypeScript files
        if suffix in (".js", ".mjs", ".cjs"):
            try:
                result = subprocess.run(
                    ["node", "--check", str(file_path)],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode != 0:
                    # Extract meaningful error
                    err = result.stderr.strip()
                    # Get first line of error
                    first_line = err.split("\n")[0] if err else "Syntax error"
                    return f"JavaScript syntax error: {first_line}"
                return None
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return None  # Node not available, skip check
            except Exception:
                return None

        # JSON files
        if suffix == ".json":
            try:
                json.loads(content)
                return None
            except json.JSONDecodeError as e:
                return f"JSON syntax error at line {e.lineno}: {e.msg}"

        # HTML files - check embedded JavaScript and basic tag matching
        if suffix in (".html", ".htm"):
            all_errors = []

            # Check embedded JavaScript in <script> tags using multi-pass approach
            # This finds ALL errors by masking found errors and re-checking
            for match in re.finditer(r'<script[^>]*>(.*?)</script>', content, re.DOTALL | re.IGNORECASE):
                script_raw = match.group(1)
                script = script_raw.strip()
                if not script:
                    continue

                # Calculate HTML line offset for this script block
                script_start_pos = match.start(1)
                leading_newlines = len(script_raw) - len(script_raw.lstrip('\n'))
                html_lines_before_script = content[:script_start_pos].count('\n') + 1 + leading_newlines

                # Multi-pass: find all errors by masking problematic lines
                script_lines = script.split('\n')
                masked_lines = set()  # Lines we've already found errors on
                max_passes = 20  # Prevent infinite loops

                for _ in range(max_passes):
                    # Create script with error lines masked (replaced with empty)
                    test_script = '\n'.join(
                        '' if i in masked_lines else line
                        for i, line in enumerate(script_lines)
                    )

                    try:
                        import tempfile
                        import os
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                            f.write(test_script)
                            temp_path = f.name
                        try:
                            result = subprocess.run(
                                ["node", "--check", temp_path],
                                capture_output=True,
                                text=True,
                                timeout=5
                            )
                            if result.returncode == 0:
                                break  # No more errors

                            err = result.stderr.strip()
                            line_match = re.search(r':(\d+)', err)
                            if line_match:
                                js_line = int(line_match.group(1))
                                js_line_idx = js_line - 1

                                # Skip if we already found this line (shouldn't happen, but safety)
                                if js_line_idx in masked_lines:
                                    break

                                # Calculate actual HTML line
                                html_line = html_lines_before_script + js_line - 1

                                # Get problem line content
                                if 0 <= js_line_idx < len(script_lines):
                                    # Extract error type
                                    error_type = "JS syntax error"
                                    if "SyntaxError:" in err:
                                        error_type = err.split("SyntaxError:")[1].split("\n")[0].strip()

                                    all_errors.append(f"Line {html_line}: {error_type}")
                                    masked_lines.add(js_line_idx)
                            else:
                                break  # Can't parse error, stop
                        finally:
                            os.unlink(temp_path)
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        break
                    except Exception:
                        break

            # Also check embedded CSS in <style> tags for common issues
            for match in re.finditer(r'<style[^>]*>(.*?)</style>', content, re.DOTALL | re.IGNORECASE):
                style_raw = match.group(1)
                style = style_raw.strip()
                if not style:
                    continue

                style_start_pos = match.start(1)
                leading_newlines = len(style_raw) - len(style_raw.lstrip('\n'))
                html_lines_before_style = content[:style_start_pos].count('\n') + 1 + leading_newlines

                style_lines = style.split('\n')
                for i, line in enumerate(style_lines):
                    html_line = html_lines_before_style + i
                    stripped = line.strip()
                    if not stripped:
                        continue

                    # Check for unclosed parentheses in CSS functions (e.g., rgba(, linear-gradient()
                    open_parens = line.count('(')
                    close_parens = line.count(')')
                    if open_parens > close_parens:
                        # Line has unclosed parenthesis - likely missing ); or )
                        all_errors.append(f"Line {html_line}: CSS unclosed '(' - add ')' and ';' at end")
                    elif ':' in stripped and not stripped.endswith((';', '{', '}', ',')):
                        # Only check semicolons if parens are balanced (otherwise paren error is primary)
                        all_errors.append(f"Line {html_line}: CSS missing ';' at end of line")

            # Return all errors found
            if all_errors:
                if len(all_errors) == 1:
                    return f"Syntax error: {all_errors[0]}"
                else:
                    return f"Found {len(all_errors)} syntax errors:\n" + "\n".join(all_errors)

            # Then check for unclosed HTML tags
            open_tags = re.findall(r'<([a-zA-Z][a-zA-Z0-9]*)[^>]*(?<!/)>', content)
            close_tags = re.findall(r'</([a-zA-Z][a-zA-Z0-9]*)>', content)

            # Exclude self-closing/void tags
            void_tags = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
                        'link', 'meta', 'param', 'source', 'track', 'wbr'}
            open_tags = [t.lower() for t in open_tags if t.lower() not in void_tags]
            close_tags = [t.lower() for t in close_tags]

            # Check for obvious mismatches
            if len(open_tags) != len(close_tags):
                diff = len(open_tags) - len(close_tags)
                if diff > 0:
                    return f"HTML warning: {diff} unclosed tag(s) detected"
                else:
                    return f"HTML warning: {-diff} extra closing tag(s) detected"

            # Basic HTML↔CSS selector consistency check for linked stylesheets.
            # This catches common drift where HTML structure changes but CSS selectors don't.
            try:
                link_tags = re.findall(r"<link\\b[^>]*>", content, flags=re.IGNORECASE)
                stylesheet_hrefs: list[str] = []
                for tag in link_tags:
                    if "stylesheet" not in tag.lower():
                        continue
                    m = re.search(r'href\\s*=\\s*[\"\\\']([^\"\\\']+)[\"\\\']', tag, flags=re.IGNORECASE)
                    if not m:
                        continue
                    href = m.group(1).strip()
                    if not href or href.startswith(("http://", "https://", "//")):
                        continue
                    stylesheet_hrefs.append(href)

                html_classes: set[str] = set()
                for m in re.finditer(r'class\\s*=\\s*[\"\\\']([^\"\\\']+)[\"\\\']', content, flags=re.IGNORECASE):
                    for token in m.group(1).split():
                        if token:
                            html_classes.add(token.strip())

                html_ids: set[str] = set()
                for m in re.finditer(r'id\\s*=\\s*[\"\\\']([^\"\\\']+)[\"\\\']', content, flags=re.IGNORECASE):
                    value = m.group(1).strip()
                    if value:
                        html_ids.add(value)

                css_classes: set[str] = set()
                css_ids: set[str] = set()

                for href in stylesheet_hrefs[:5]:  # cap work
                    # Resolve relative to the HTML file location.
                    candidate = (file_path.parent / href).resolve()
                    try:
                        candidate.relative_to(self.project_root)
                    except ValueError:
                        continue
                    if not candidate.exists() or not candidate.is_file():
                        continue
                    try:
                        css_text = candidate.read_text(encoding="utf-8")
                    except Exception:
                        continue

                    for cm in re.finditer(r"(?<![\\w-])\\.([a-zA-Z_][\\w-]*)", css_text):
                        css_classes.add(cm.group(1))
                    for im in re.finditer(r"(?<![\\w-])#([a-zA-Z_][\\w-]*)", css_text):
                        ident = im.group(1)
                        # Ignore hex colors like #fff/#ffffff
                        if len(ident) in (3, 6) and all(c in "0123456789abcdefABCDEF" for c in ident):
                            continue
                        css_ids.add(ident)

                missing_class_css = sorted(html_classes - css_classes)
                missing_id_css = sorted(html_ids - css_ids)

                # Only flag if it looks meaningful (avoid noise on tiny/unstyled HTML).
                if (missing_class_css or missing_id_css) and (len(html_classes) + len(html_ids)) >= 3:
                    parts: list[str] = []
                    if missing_class_css:
                        sample = ", ".join(missing_class_css[:10])
                        extra = f", …(+{len(missing_class_css) - 10})" if len(missing_class_css) > 10 else ""
                        parts.append(f"missing CSS for HTML class(es): {sample}{extra}")
                    if missing_id_css:
                        sample = ", ".join(missing_id_css[:10])
                        extra = f", …(+{len(missing_id_css) - 10})" if len(missing_id_css) > 10 else ""
                        parts.append(f"missing CSS for HTML id(s): {sample}{extra}")
                    return "HTML/CSS mismatch warning: " + " | ".join(parts)
            except Exception:
                pass

            return None

        # CSS files - basic brace matching + simple quality warnings
        if suffix == ".css":
            open_braces = content.count("{")
            close_braces = content.count("}")
            if open_braces != close_braces:
                # Try to pinpoint a likely mismatch line to avoid brute-force full-file reads.
                balance = 0
                in_block_comment = False
                in_string: Optional[str] = None
                last_open_line: Optional[int] = None

                lines = content.splitlines()
                for idx, line in enumerate(lines, start=1):
                    i = 0
                    while i < len(line):
                        ch = line[i]
                        nxt = line[i + 1] if i + 1 < len(line) else ""

                        if in_block_comment:
                            if ch == "*" and nxt == "/":
                                in_block_comment = False
                                i += 2
                                continue
                            i += 1
                            continue

                        if in_string:
                            if ch == "\\":
                                i += 2
                                continue
                            if ch == in_string:
                                in_string = None
                            i += 1
                            continue

                        # Not in comment/string
                        if ch == "/" and nxt == "*":
                            in_block_comment = True
                            i += 2
                            continue
                        if ch in ("'", '"'):
                            in_string = ch
                            i += 1
                            continue

                        if ch == "{":
                            balance += 1
                            last_open_line = idx
                        elif ch == "}":
                            balance -= 1
                            if balance < 0:
                                return (
                                    f"CSS syntax error: extra '}}' at line {idx} "
                                    f"(overall braces: {{ = {open_braces}, }} = {close_braces})"
                                )

                        i += 1

                if balance > 0:
                    # Missing close brace; best guess: last unmatched opener.
                    guess = last_open_line or 1
                    return (
                        f"CSS syntax error: missing '}}' for '{{' opened near line {guess} "
                        f"(overall braces: {{ = {open_braces}, }} = {close_braces})"
                    )

                return f"CSS syntax error: mismatched braces ({{ = {open_braces}, }} = {close_braces})"

            # Duplicate selector detection (often indicates iterative drift/overwrites).
            try:
                selector_counts: Dict[str, int] = {}
                # Capture selector blocks up to the opening brace (ignore @-rules).
                for match in re.finditer(r"([^{}]+)\\{", content):
                    raw = match.group(1).strip()
                    if not raw or raw.startswith("@"):
                        continue
                    # Remove common at-rule prelude accidentally captured.
                    if raw.lower().startswith(("from", "to")):
                        continue
                    for sel in raw.split(","):
                        sel = sel.strip()
                        if not sel or sel.startswith("@"):
                            continue
                        # Normalize whitespace
                        sel = re.sub(r"\\s+", " ", sel)
                        selector_counts[sel] = selector_counts.get(sel, 0) + 1

                duplicates = sorted(
                    ((s, c) for s, c in selector_counts.items() if c > 1),
                    key=lambda x: (-x[1], x[0]),
                )
                if duplicates:
                    shown = ", ".join(f"{s} ({c}x)" for s, c in duplicates[:8])
                    extra = f", …(+{len(duplicates) - 8})" if len(duplicates) > 8 else ""
                    return f"CSS warning: duplicate selector blocks detected: {shown}{extra}"
            except Exception:
                pass
            return None

        # YAML files
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
                yaml.safe_load(content)
                return None
            except ImportError:
                return None  # PyYAML not available
            except yaml.YAMLError as e:
                if hasattr(e, 'problem_mark'):
                    mark = e.problem_mark
                    return f"YAML syntax error at line {mark.line + 1}: {e.problem}"
                return f"YAML syntax error: {str(e)[:100]}"

        # Shell scripts
        if suffix in (".sh", ".bash"):
            try:
                result = subprocess.run(
                    ["bash", "-n", str(file_path)],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode != 0:
                    err = result.stderr.strip()
                    first_line = err.split("\n")[0] if err else "Syntax error"
                    return f"Shell syntax error: {first_line}"
                return None
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return None
            except Exception:
                return None

        # XML files
        if suffix == ".xml":
            try:
                import xml.etree.ElementTree as ET
                ET.fromstring(content)
                return None
            except ET.ParseError as e:
                return f"XML syntax error: {str(e)[:100]}"

        return None  # Unknown file type, skip check

    def _memory_read(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Read from persistent memory."""
        key = args.get("key")
        category = args.get("category")
        search = args.get("search")

        entries = self._memory.get("entries", {})

        # Specific key lookup
        if key:
            if key in entries:
                entry = entries[key]
                return {
                    "success": True,
                    "key": key,
                    "value": entry["value"],
                    "category": entry.get("category", "general"),
                    "timestamp": entry.get("timestamp"),
                    "message": f"Found memory: {key}"
                }
            else:
                return {
                    "success": False,
                    "error": f"No memory found for key: {key}"
                }

        # Filter and search
        results = []
        for k, entry in entries.items():
            # Category filter
            if category and entry.get("category") != category:
                continue
            # Search filter
            if search and search.lower() not in entry["value"].lower():
                continue
            results.append({
                "key": k,
                "value": entry["value"][:200] + ("..." if len(entry["value"]) > 200 else ""),
                "category": entry.get("category", "general")
            })

        if not results:
            if category or search:
                return {
                    "success": True,
                    "results": [],
                    "count": 0,
                    "message": "No matching memories found"
                }
            else:
                return {
                    "success": True,
                    "results": [],
                    "count": 0,
                    "message": "Memory is empty"
                }

        return {
            "success": True,
            "results": results,
            "count": len(results),
            "message": f"Found {len(results)} memories"
        }
