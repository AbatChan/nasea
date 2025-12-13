"""
NASEA Dynamic Configuration System

Loads configuration from YAML files with support for:
- Default configs in nasea/config/*.yaml
- Project-level overrides in .nasea/config.yaml
- Environment variable overrides
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from loguru import logger

# Try to import yaml, fall back to basic parsing if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not installed - using fallback config loading")


# Config directory (where default configs live)
CONFIG_DIR = Path(__file__).parent

# Project config locations (checked in order)
PROJECT_CONFIG_PATHS = [
    ".nasea/config.yaml",
    ".nasea/config.yml",
    ".nasea.yaml",
    ".nasea.yml",
    "nasea.yaml",
    "nasea.yml",
]


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load a YAML file, returning empty dict if not found or invalid."""
    if not path.exists():
        return {}

    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        if YAML_AVAILABLE:
            return yaml.safe_load(content) or {}
        else:
            # Basic fallback parser for simple YAML
            return _parse_simple_yaml(content)
    except Exception as e:
        logger.warning(f"Failed to load config from {path}: {e}")
        return {}


def _parse_simple_yaml(content: str) -> Dict[str, Any]:
    """
    Simple YAML parser for basic key-value configs.
    Only handles flat structures and simple nested dicts.
    """
    result = {}
    current_section = result
    current_key = None
    indent_stack = [(0, result)]

    for line in content.split('\n'):
        # Skip empty lines and comments
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue

        # Calculate indent
        indent = len(line) - len(line.lstrip())

        # Handle list items
        if stripped.startswith('- '):
            if current_key and isinstance(current_section.get(current_key), list):
                value = stripped[2:].strip().strip('"').strip("'")
                current_section[current_key].append(value)
            continue

        # Handle key-value pairs
        if ':' in stripped:
            key, _, value = stripped.partition(':')
            key = key.strip()
            value = value.strip()

            # Pop indent stack to find correct parent
            while indent_stack and indent <= indent_stack[-1][0]:
                if len(indent_stack) > 1:
                    indent_stack.pop()
                else:
                    break

            current_section = indent_stack[-1][1]

            if value:
                # Simple value
                # Try to parse as number/bool
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
                else:
                    value = value.strip('"').strip("'")
                current_section[key] = value
            else:
                # Nested dict or list
                # Check next line to determine if list or dict
                current_section[key] = {}
                indent_stack.append((indent + 2, current_section[key]))
                current_key = key

    return result


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


class ConfigLoader:
    """
    Loads and merges configuration from multiple sources.

    Priority (highest to lowest):
    1. Environment variables (NASEA_*)
    2. Project-level config (.nasea/config.yaml)
    3. Default config (nasea/config/*.yaml)
    """

    _cache: Dict[str, Dict[str, Any]] = {}
    _project_root: Optional[Path] = None

    @classmethod
    def set_project_root(cls, path: Path):
        """Set the project root for loading project-level configs."""
        cls._project_root = Path(path)
        cls._cache.clear()  # Clear cache when project changes

    @classmethod
    def load(cls, config_name: str) -> Dict[str, Any]:
        """
        Load a configuration by name.

        Args:
            config_name: Name of config file (without .yaml extension)
                        e.g., "model_fixes", "confidence"

        Returns:
            Merged configuration dictionary
        """
        if config_name in cls._cache:
            return cls._cache[config_name]

        # Load default config
        default_path = CONFIG_DIR / f"{config_name}.yaml"
        config = _load_yaml_file(default_path)

        # Load project-level override
        if cls._project_root:
            for rel_path in PROJECT_CONFIG_PATHS:
                project_config_path = cls._project_root / rel_path
                if project_config_path.exists():
                    project_config = _load_yaml_file(project_config_path)
                    # Get section for this config name
                    if config_name in project_config:
                        config = _deep_merge(config, project_config[config_name])
                    break

        # Apply environment variable overrides
        config = cls._apply_env_overrides(config_name, config)

        cls._cache[config_name] = config
        return config

    @classmethod
    def _apply_env_overrides(cls, config_name: str, config: Dict) -> Dict:
        """Apply environment variable overrides."""
        prefix = f"NASEA_{config_name.upper()}_"

        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert NASEA_MODEL_FIXES_DEFAULT_THRESHOLD -> default_threshold
                config_key = key[len(prefix):].lower()

                # Try to parse value
                if value.lower() == 'true':
                    config[config_key] = True
                elif value.lower() == 'false':
                    config[config_key] = False
                elif value.isdigit():
                    config[config_key] = int(value)
                else:
                    config[config_key] = value

        return config

    @classmethod
    def reload(cls, config_name: Optional[str] = None):
        """Reload configuration(s) from disk."""
        if config_name:
            cls._cache.pop(config_name, None)
        else:
            cls._cache.clear()

    @classmethod
    def get_all_configs(cls) -> List[str]:
        """List all available config files."""
        configs = []
        for path in CONFIG_DIR.glob("*.yaml"):
            configs.append(path.stem)
        return configs


# Convenience functions
def load_config(name: str) -> Dict[str, Any]:
    """Load a configuration by name."""
    return ConfigLoader.load(name)


def set_project_root(path: Path):
    """Set the project root for config loading."""
    ConfigLoader.set_project_root(path)


def reload_configs():
    """Reload all configurations from disk."""
    ConfigLoader.reload()
