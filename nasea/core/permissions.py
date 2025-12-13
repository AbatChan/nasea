"""
Permission management system - mimics Claude Code's safety model.
Conservative by default, with allowlist overrides.

Key principles from Claude Code:
- Conservative approach to prioritize safety
- Always-allowed list for safe (read-only) operations
- Per-session allowlist for approved operations
- Persistent config file for user preferences
- Pattern matching for flexible rules (e.g., "git:*" for all git commands)
"""

from typing import Dict, Set, Optional, Tuple, Any
from pathlib import Path
import json
from dataclasses import dataclass, asdict, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PermissionConfig:
    """
    Permission configuration.

    Attributes:
        always_allowed: Tools that never require approval (read-only, safe)
        ask_each_time: Tools that require user approval each time
        allowed_this_session: Tools pre-approved for current session only
        denied: Tools that are explicitly denied
    """

    # Tools that are always allowed (read-only, safe operations)
    always_allowed: Set[str] = field(default_factory=set)

    # Tools that require user approval each time
    ask_each_time: Set[str] = field(default_factory=set)

    # Tools that are pre-approved for this session
    allowed_this_session: Set[str] = field(default_factory=set)

    # Tools that are explicitly denied
    denied: Set[str] = field(default_factory=set)


class PermissionManager:
    """
    Manages tool permissions with allowlist system.
    Mimics Claude Code's conservative-by-default approach.

    Security principles:
    1. Default DENY for destructive operations
    2. Default ALLOW for read-only operations
    3. User controls via interactive prompts
    4. Persistent storage of preferences
    5. Pattern matching for flexible rules
    """

    # Default safe tools (read-only, no state modification)
    SAFE_TOOLS = {
        'read_file',
        'list_files',
        'complete_generation',
        # Explicitly NO write/edit/delete in safe list
    }

    # Tools that modify state (require approval)
    DANGEROUS_TOOLS = {
        'write_file',
        'edit_file',
        'delete_path',
        'rename_path',
        'run_command',
        'create_directory'
    }

    def __init__(self, project_root: Path, console, auto_approve_safe: bool = True):
        """
        Initialize permission manager.

        Args:
            project_root: Root directory of the project
            console: Rich console for user interaction
            auto_approve_safe: Automatically approve safe (read-only) tools
        """
        self.root = project_root
        self.console = console
        self.auto_approve_safe = auto_approve_safe
        self.config = self._load_config()

    def _load_config(self) -> PermissionConfig:
        """
        Load permission config from .nasea/permissions.json if exists.

        Falls back to defaults if file doesn't exist or is invalid.
        """

        config_path = self.root / '.nasea' / 'permissions.json'

        if config_path.exists():
            try:
                data = json.loads(config_path.read_text())
                return PermissionConfig(
                    always_allowed=set(data.get('always_allowed', list(self.SAFE_TOOLS))),
                    ask_each_time=set(data.get('ask_each_time', list(self.DANGEROUS_TOOLS))),
                    allowed_this_session=set(),  # Never persisted
                    denied=set(data.get('denied', []))
                )
            except Exception as e:
                logger.warning(f"Failed to load permission config: {e}, using defaults")

        # Default config - conservative by default
        return PermissionConfig(
            always_allowed=self.SAFE_TOOLS.copy(),
            ask_each_time=self.DANGEROUS_TOOLS.copy(),
            allowed_this_session=set(),
            denied=set()
        )

    def save_config(self) -> None:
        """
        Save current permission config to disk.

        Only persists: always_allowed, ask_each_time, denied
        Does NOT persist: allowed_this_session (session-specific)
        """

        config_dir = self.root / '.nasea'
        config_dir.mkdir(exist_ok=True, parents=True)

        config_path = config_dir / 'permissions.json'

        data = {
            'always_allowed': sorted(list(self.config.always_allowed)),
            'ask_each_time': sorted(list(self.config.ask_each_time)),
            'denied': sorted(list(self.config.denied))
            # Explicitly NOT saving allowed_this_session
        }

        try:
            config_path.write_text(json.dumps(data, indent=2))
            logger.debug(f"Saved permission config to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save permission config: {e}")

    def check_permission(
        self,
        tool_name: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if tool execution is permitted.

        Decision flow:
        1. Check denied list → DENY
        2. Check always_allowed → ALLOW
        3. Check session allowlist → ALLOW
        4. Check pattern matching → ALLOW if matches
        5. Ask user → ALLOW/DENY based on response

        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters (optional, for pattern matching)

        Returns:
            (is_allowed, reason_if_denied)
        """

        parameters = parameters or {}

        # 1. Check denied list first (explicit block)
        if tool_name in self.config.denied:
            return False, f"Tool '{tool_name}' is explicitly denied in config"

        # 2. Check always allowed (safe tools)
        if tool_name in self.config.always_allowed:
            return True, None

        # 3. Check session allowlist
        if tool_name in self.config.allowed_this_session:
            return True, None

        # 4. For dangerous tools, check if pattern matches allowlist
        if self._matches_allowlist(tool_name, parameters):
            return True, None

        # 5. Default: ask user for permission
        return self._ask_user(tool_name, parameters)

    def _matches_allowlist(self, tool_name: str, parameters: Dict[str, Any]) -> bool:
        """
        Check if tool/parameters match any allowlist pattern.

        Pattern examples:
        - "write_file" → allows all write_file calls
        - "write_file(*.test.js)" → allows only test files
        - "run_command(git:*)" → allows all git commands
        - "run_command(npm:*)" → allows all npm commands

        Args:
            tool_name: Name of the tool
            parameters: Tool parameters

        Returns:
            True if matches any pattern, False otherwise
        """

        for pattern in self.config.always_allowed:
            # Simple tool name match
            if pattern == tool_name:
                return True

            # Pattern with parameters
            if ':' in pattern:
                # Format: "tool_name(param:pattern)"
                # Example: "run_command(git:*)"
                try:
                    tool_part, param_pattern = pattern.split(':', 1)

                    # Extract tool name from pattern
                    if '(' in tool_part:
                        pattern_tool = tool_part[:tool_part.index('(')]
                    else:
                        pattern_tool = tool_part

                    if pattern_tool == tool_name:
                        # Check if parameters match pattern
                        if param_pattern == '*':
                            return True

                        # Check specific parameter values
                        # For run_command, check if command starts with pattern
                        if tool_name == 'run_command' and 'command' in parameters:
                            cmd = parameters['command']
                            if param_pattern.endswith('*'):
                                prefix = param_pattern[:-1]
                                if cmd.startswith(prefix):
                                    return True
                            elif cmd == param_pattern:
                                return True

                except Exception as e:
                    logger.debug(f"Failed to parse pattern '{pattern}': {e}")
                    continue

        return False

    def _ask_user(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Ask user for permission to use tool.

        Shows tool name, parameters, and multiple choice options:
          - y: Allow once
          - a: Always allow (this session)
          - A: Always allow (saved to config)
          - n: Deny once
          - N: Never allow (saved to config)

        Args:
            tool_name: Name of the tool
            parameters: Tool parameters

        Returns:
            (is_allowed, reason_if_denied)
        """

        self.console.print(f"\n[yellow]⚠️  Permission required:[/yellow] [bold]{tool_name}[/bold]")

        # Show relevant parameters (not all, to avoid noise)
        relevant_params = self._get_relevant_params(tool_name, parameters)
        if relevant_params:
            self.console.print(f"[dim]Parameters:[/dim]")
            for key, value in relevant_params.items():
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 60:
                    value_str = value_str[:57] + "..."
                self.console.print(f"  [dim]{key}:[/dim] {value_str}")

        self.console.print()
        self.console.print("[bold]Options:[/bold]")
        self.console.print("  [green]y[/green] - Allow once")
        self.console.print("  [green]a[/green] - Always allow (this session)")
        self.console.print("  [green]A[/green] - Always allow (saved to config)")
        self.console.print("  [red]n[/red] - Deny once")
        self.console.print("  [red]N[/red] - Never allow (saved to config)")

        try:
            choice = input("\nChoice [y/a/A/n/N]: ").strip()
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]Permission prompt interrupted[/yellow]\n")
            return False, "User cancelled permission request"

        # Process choice
        if choice == 'y':
            return True, None

        if choice == 'a':
            self.config.allowed_this_session.add(tool_name)
            self.console.print(f"[green]✓[/green] Approved for this session\n")
            return True, None

        if choice == 'A':
            self.config.always_allowed.add(tool_name)
            self.save_config()
            self.console.print(f"[green]✓[/green] {tool_name} added to allowlist (permanent)\n")
            return True, None

        if choice == 'N':
            self.config.denied.add(tool_name)
            self.save_config()
            self.console.print(f"[red]✗[/red] {tool_name} added to denylist (permanent)\n")
            return False, f"Tool '{tool_name}' permanently denied by user"

        # Default (n or anything else): deny once
        self.console.print(f"[yellow]Permission denied once[/yellow]\n")
        return False, f"Permission denied by user for '{tool_name}'"

    def _get_relevant_params(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant parameters to show user.

        Avoids showing ALL parameters (too noisy), just the important ones.

        Args:
            tool_name: Name of the tool
            parameters: All parameters

        Returns:
            Subset of parameters relevant for this tool
        """

        relevant_keys = {
            'write_file': ['file_path'],
            'edit_file': ['file_path'],
            'delete_path': ['path'],
            'rename_path': ['old_path', 'new_path'],
            'run_command': ['command'],
            'create_directory': ['path'],
            'read_file': ['file_path'],
            'list_files': ['path', 'pattern']
        }

        keys = relevant_keys.get(tool_name, list(parameters.keys())[:3])
        return {k: parameters.get(k) for k in keys if k in parameters}

    def add_safe_patterns(self, patterns: list[str]) -> None:
        """
        Add multiple patterns to the allowlist at once.

        Useful for setting up common safe operations like git, npm, docker.

        Args:
            patterns: List of patterns to add (e.g., ["run_command(git:*)", "run_command(npm:*)"])
        """

        for pattern in patterns:
            self.config.always_allowed.add(pattern)

        self.save_config()
        logger.info(f"Added {len(patterns)} patterns to allowlist")

    def reset_session_permissions(self) -> None:
        """Clear all session-specific permissions."""
        self.config.allowed_this_session.clear()
        logger.debug("Reset session permissions")

    def get_stats(self) -> Dict[str, int]:
        """
        Get permission statistics.

        Returns:
            Dictionary with counts of different permission states
        """

        return {
            'always_allowed': len(self.config.always_allowed),
            'ask_each_time': len(self.config.ask_each_time),
            'session_allowed': len(self.config.allowed_this_session),
            'denied': len(self.config.denied)
        }
