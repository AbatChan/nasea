"""
NASEA Modular Prompts System
Inspired by Claude Code's modular prompt architecture.

This module provides centralized, well-organized prompts for all NASEA agents and tools.
"""

from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
import json
import re

# Prompt directory
PROMPTS_DIR = Path(__file__).parent


class PromptMetadata:
    """Metadata extracted from prompt frontmatter."""

    def __init__(
        self,
        name: str = "",
        description: str = "",
        version: str = "1.0.0",
        allowed_tools: Optional[List[str]] = None,
        forbidden_tools: Optional[List[str]] = None,
        variables: Optional[List[str]] = None,
        model: Optional[str] = None,
        color: Optional[str] = None,
        parallel: bool = False,
        extra: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.description = description
        self.version = version
        self.allowed_tools = allowed_tools or []
        self.forbidden_tools = forbidden_tools or []
        self.variables = variables or []
        self.model = model  # Preferred model for this agent (e.g., "gpt-4", "haiku")
        self.color = color  # CLI display color (e.g., "green", "red", "blue")
        self.parallel = parallel  # Can run in parallel with other agents
        self.extra = extra or {}

    def filter_tools(self, all_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter a list of tool definitions based on allowed/forbidden lists.

        Args:
            all_tools: List of tool definitions (OpenAI function format)

        Returns:
            Filtered list of tools this agent can use
        """
        if not self.allowed_tools and not self.forbidden_tools:
            return all_tools

        filtered = []
        for tool in all_tools:
            tool_name = tool.get("function", {}).get("name", "")

            # If allowed_tools is specified, tool must be in it
            if self.allowed_tools and tool_name not in self.allowed_tools:
                continue

            # If forbidden_tools is specified, tool must not be in it
            if self.forbidden_tools and tool_name in self.forbidden_tools:
                continue

            filtered.append(tool)

        return filtered


class PromptLoader:
    """Load and cache prompts from markdown files."""

    _cache: Dict[str, str] = {}
    _metadata_cache: Dict[str, PromptMetadata] = {}

    @classmethod
    def _parse_frontmatter(cls, content: str) -> Tuple[PromptMetadata, str]:
        """
        Parse YAML-like frontmatter from markdown content.

        Args:
            content: Full file content

        Returns:
            Tuple of (metadata, remaining_content)
        """
        metadata = PromptMetadata()

        if not content.startswith("<!--"):
            return metadata, content

        end_idx = content.find("-->")
        if end_idx == -1:
            return metadata, content

        frontmatter = content[4:end_idx].strip()
        remaining = content[end_idx + 3:].strip()

        # Parse YAML-like frontmatter (simple parser)
        current_list_key = None
        current_list: List[str] = []

        for line in frontmatter.split("\n"):
            line = line.strip()
            if not line:
                continue

            # List item (starts with -)
            if line.startswith("- "):
                if current_list_key:
                    current_list.append(line[2:].strip())
                continue

            # Key-value pair
            if current_list_key and current_list:
                # Save previous list
                if current_list_key == "allowed_tools":
                    metadata.allowed_tools = current_list
                elif current_list_key == "forbidden_tools":
                    metadata.forbidden_tools = current_list
                elif current_list_key == "variables":
                    metadata.variables = current_list
                current_list_key = None
                current_list = []

            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                # Check if this is a list header (value is empty or just whitespace)
                if not value or value == "":
                    if key in ("allowed_tools", "forbidden_tools", "variables"):
                        current_list_key = key
                        current_list = []
                    continue

                # Remove quotes from value
                if value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                elif value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]

                if key == "name":
                    metadata.name = value
                elif key == "description":
                    metadata.description = value
                elif key == "version":
                    metadata.version = value
                elif key == "model":
                    metadata.model = value
                elif key == "color":
                    metadata.color = value
                elif key == "parallel":
                    metadata.parallel = value.lower() in ("true", "yes", "1")
                else:
                    metadata.extra[key] = value

        # Don't forget the last list if file ends with a list
        if current_list_key and current_list:
            if current_list_key == "allowed_tools":
                metadata.allowed_tools = current_list
            elif current_list_key == "forbidden_tools":
                metadata.forbidden_tools = current_list
            elif current_list_key == "variables":
                metadata.variables = current_list

        return metadata, remaining

    @classmethod
    def load(cls, category: str, name: str) -> str:
        """
        Load a prompt from file.

        Args:
            category: Prompt category (system, agents, tools, utilities)
            name: Prompt file name without extension

        Returns:
            Prompt content as string
        """
        cache_key = f"{category}/{name}"

        if cache_key in cls._cache:
            return cls._cache[cache_key]

        prompt_path = PROMPTS_DIR / category / f"{name}.md"

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_path}")

        raw_content = prompt_path.read_text(encoding="utf-8")

        # Parse frontmatter and get clean content
        metadata, content = cls._parse_frontmatter(raw_content)

        cls._cache[cache_key] = content
        cls._metadata_cache[cache_key] = metadata

        return content

    @classmethod
    def load_with_metadata(cls, category: str, name: str) -> Tuple[str, PromptMetadata]:
        """
        Load a prompt and its metadata.

        Args:
            category: Prompt category
            name: Prompt file name

        Returns:
            Tuple of (prompt_content, metadata)
        """
        content = cls.load(category, name)
        cache_key = f"{category}/{name}"
        metadata = cls._metadata_cache.get(cache_key, PromptMetadata())
        return content, metadata

    @classmethod
    def get_metadata(cls, category: str, name: str) -> PromptMetadata:
        """
        Get just the metadata for a prompt (loads if not cached).

        Args:
            category: Prompt category
            name: Prompt file name

        Returns:
            PromptMetadata object
        """
        cache_key = f"{category}/{name}"

        if cache_key not in cls._metadata_cache:
            cls.load(category, name)

        return cls._metadata_cache.get(cache_key, PromptMetadata())

    @classmethod
    def load_with_vars(cls, category: str, name: str, variables: Dict[str, str]) -> str:
        """
        Load a prompt and substitute variables.

        Args:
            category: Prompt category
            name: Prompt file name
            variables: Dict of variable substitutions {VAR_NAME: value}

        Returns:
            Prompt with variables substituted
        """
        content = cls.load(category, name)

        for var_name, value in variables.items():
            content = content.replace(f"${{{var_name}}}", str(value))
            content = content.replace(f"${var_name}", str(value))

        return content

    @classmethod
    def clear_cache(cls):
        """Clear the prompt cache."""
        cls._cache.clear()
        cls._metadata_cache.clear()


# Convenience functions
def load_system_prompt(name: str, **variables) -> str:
    """Load a system prompt."""
    if variables:
        return PromptLoader.load_with_vars("system", name, variables)
    return PromptLoader.load("system", name)


def load_agent_prompt(name: str, **variables) -> str:
    """Load an agent prompt."""
    if variables:
        return PromptLoader.load_with_vars("agents", name, variables)
    return PromptLoader.load("agents", name)


def load_agent_prompt_with_metadata(name: str) -> Tuple[str, PromptMetadata]:
    """
    Load an agent prompt along with its metadata.

    Args:
        name: Agent name (manager, developer, verifier, explore)

    Returns:
        Tuple of (prompt_content, metadata) where metadata contains
        allowed_tools, forbidden_tools, etc.
    """
    return PromptLoader.load_with_metadata("agents", name)


def get_agent_metadata(name: str) -> PromptMetadata:
    """
    Get metadata for an agent prompt.

    Args:
        name: Agent name (manager, developer, verifier, explore)

    Returns:
        PromptMetadata object with allowed_tools, forbidden_tools, etc.
    """
    return PromptLoader.get_metadata("agents", name)


def filter_tools_for_agent(agent_name: str, all_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter tools based on an agent's allowed/forbidden tool lists.

    Args:
        agent_name: Name of the agent (manager, developer, verifier)
        all_tools: Complete list of tool definitions

    Returns:
        Filtered list of tools this agent can use
    """
    metadata = get_agent_metadata(agent_name)
    return metadata.filter_tools(all_tools)


def load_tool_prompt(name: str, **variables) -> str:
    """Load a tool description prompt."""
    if variables:
        return PromptLoader.load_with_vars("tools", name, variables)
    return PromptLoader.load("tools", name)


def load_utility_prompt(name: str, **variables) -> str:
    """Load a utility prompt."""
    if variables:
        return PromptLoader.load_with_vars("utilities", name, variables)
    return PromptLoader.load("utilities", name)
