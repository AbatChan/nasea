"""
Shared utilities for NASEA.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml


# File extension to language mapping
EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
    ".md": "markdown",
    ".sql": "sql",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
    ".R": "r",
    ".lua": "lua",
    ".pl": "perl",
    ".pm": "perl",
}


def detect_language(filename: str, default: str = "text") -> str:
    """
    Detect programming language from filename extension.

    Args:
        filename: Name or path of the file
        default: Default language if extension not recognized

    Returns:
        Language identifier string
    """
    for ext, lang in EXTENSION_TO_LANGUAGE.items():
        if filename.endswith(ext):
            return lang
    return default


def get_file_extension(language: str) -> str:
    """
    Get the primary file extension for a language.

    Args:
        language: Language identifier

    Returns:
        File extension including the dot (e.g., ".py")
    """
    # Reverse lookup - find first matching extension
    for ext, lang in EXTENSION_TO_LANGUAGE.items():
        if lang == language:
            return ext
    return ".txt"


# Cache for model limits
_MODEL_LIMITS_CACHE: Optional[Dict] = None


def _load_model_limits() -> Dict:
    """Load model limits from YAML config file."""
    global _MODEL_LIMITS_CACHE
    if _MODEL_LIMITS_CACHE is not None:
        return _MODEL_LIMITS_CACHE

    config_path = Path(__file__).parent.parent / "config" / "model_limits.yaml"
    if config_path.exists():
        with open(config_path) as f:
            _MODEL_LIMITS_CACHE = yaml.safe_load(f)
    else:
        _MODEL_LIMITS_CACHE = {"defaults": {"context_window": 4096, "max_output_tokens": 4096}, "models": {}}

    return _MODEL_LIMITS_CACHE


def get_model_limits(model: str) -> Tuple[int, int]:
    """
    Get context window and max output tokens for a model.

    Args:
        model: Model name/identifier

    Returns:
        Tuple of (context_window, max_output_tokens)
    """
    config = _load_model_limits()
    defaults = config.get("defaults", {})
    models = config.get("models", {})

    # Normalize model name
    model_lower = model.lower() if model else ""

    # Try exact match first
    if model_lower in models:
        m = models[model_lower]
        return m.get("context_window", defaults.get("context_window", 4096)), m.get("max_output_tokens", defaults.get("max_output_tokens", 4096))

    # Try partial match
    for key, m in models.items():
        if key in model_lower or model_lower in key:
            return m.get("context_window", defaults.get("context_window", 4096)), m.get("max_output_tokens", defaults.get("max_output_tokens", 4096))

    # Return defaults
    return defaults.get("context_window", 4096), defaults.get("max_output_tokens", 4096)


def get_max_output_tokens(model: str) -> int:
    """
    Get max output tokens for a model.

    Args:
        model: Model name/identifier

    Returns:
        Max output tokens
    """
    _, max_output = get_model_limits(model)
    return max_output


def get_context_window(model: str) -> int:
    """
    Get context window for a model.

    Args:
        model: Model name/identifier

    Returns:
        Context window in tokens
    """
    context, _ = get_model_limits(model)
    return context
