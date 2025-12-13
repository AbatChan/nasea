"""
Model-Specific Prompt Fixes

Different LLM models have quirks and respond differently to certain prompt patterns.
This module provides model-specific adjustments to improve output quality.

Configuration is loaded from nasea/config/model_fixes.yaml and can be overridden
in project-level .nasea/config.yaml files.

Inspired by Claude Code's approach to handling model differences.
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from loguru import logger

from nasea.config import load_config


@dataclass
class ModelConfig:
    """Configuration for a specific model's prompt handling."""

    # Phrases to remove from prompts (model responds poorly to these)
    remove_phrases: List[str] = field(default_factory=list)

    # Phrase replacements {old: new}
    replace_phrases: Dict[str, str] = field(default_factory=dict)

    # Regex patterns to remove
    remove_patterns: List[str] = field(default_factory=list)

    # System prompt additions (prepended)
    system_prefix: str = ""

    # System prompt additions (appended)
    system_suffix: str = ""

    # Token adjustments
    max_tokens_multiplier: float = 1.0

    # Temperature adjustments
    temperature_offset: float = 0.0

    # Whether the model handles JSON well
    json_mode_supported: bool = True

    # Whether the model needs explicit instruction formatting
    needs_explicit_format: bool = False

    # Custom post-processor function name
    post_processor: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from dictionary (e.g., from YAML)."""
        return cls(
            remove_phrases=data.get("remove_phrases", []),
            replace_phrases=data.get("replace_phrases", {}),
            remove_patterns=data.get("remove_patterns", []),
            system_prefix=data.get("system_prefix", ""),
            system_suffix=data.get("system_suffix", ""),
            max_tokens_multiplier=data.get("max_tokens_multiplier", 1.0),
            temperature_offset=data.get("temperature_offset", 0.0),
            json_mode_supported=data.get("json_mode_supported", True),
            needs_explicit_format=data.get("needs_explicit_format", False),
            post_processor=data.get("post_processor"),
        )


# Cache for loaded configs
_config_cache: Dict[str, ModelConfig] = {}
_default_config: Optional[ModelConfig] = None


def _load_model_configs() -> Dict[str, ModelConfig]:
    """Load model configurations from YAML file."""
    global _config_cache, _default_config

    if _config_cache:
        return _config_cache

    try:
        yaml_config = load_config("model_fixes")
    except Exception as e:
        logger.warning(f"Could not load model_fixes.yaml: {e}")
        yaml_config = {}

    # Load defaults
    defaults = yaml_config.get("defaults", {})
    _default_config = ModelConfig.from_dict(defaults)

    # Load model-specific configs
    models = yaml_config.get("models", {})
    for model_name, model_data in models.items():
        # Merge with defaults
        merged = {**defaults, **model_data}
        _config_cache[model_name] = ModelConfig.from_dict(merged)

    logger.debug(f"Loaded {len(_config_cache)} model configurations")
    return _config_cache


def _get_default_config() -> ModelConfig:
    """Get the default config (loads if needed)."""
    global _default_config
    if _default_config is None:
        _load_model_configs()
    return _default_config or ModelConfig()


def reload_model_configs():
    """Reload model configurations from disk."""
    global _config_cache, _default_config
    _config_cache.clear()
    _default_config = None
    _load_model_configs()


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get configuration for a specific model.

    Loads from nasea/config/model_fixes.yaml with project-level overrides.

    Args:
        model_name: Model identifier (e.g., "gpt-4", "qwen3-235b")

    Returns:
        ModelConfig for the model, or default if unknown
    """
    configs = _load_model_configs()

    # Try exact match first
    if model_name in configs:
        return configs[model_name]

    # Try prefix match (e.g., "gpt-4-0125-preview" -> "gpt-4")
    for prefix, config in configs.items():
        if model_name.startswith(prefix):
            return config

    # Try substring match
    model_lower = model_name.lower()
    for key, config in configs.items():
        if key.lower() in model_lower:
            return config

    logger.debug(f"No specific config for model '{model_name}', using defaults")
    return _get_default_config()


def apply_model_fixes(
    prompt: str,
    model_name: str,
    is_system_prompt: bool = False
) -> str:
    """
    Apply model-specific fixes to a prompt.

    Args:
        prompt: The original prompt text
        model_name: Model identifier
        is_system_prompt: Whether this is a system prompt (vs user message)

    Returns:
        Modified prompt with model-specific adjustments
    """
    config = get_model_config(model_name)
    result = prompt

    # Apply phrase replacements first (sorted by length, longest first)
    # This prevents partial matches (e.g., "You should NEVER" before "NEVER")
    sorted_replacements = sorted(
        config.replace_phrases.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )
    for old, new in sorted_replacements:
        result = result.replace(old, new)

    # Remove problematic phrases
    for phrase in config.remove_phrases:
        result = result.replace(phrase, "")

    # Apply regex removals
    for pattern in config.remove_patterns:
        result = re.sub(pattern, "", result)

    # Add system prompt prefix/suffix
    if is_system_prompt:
        if config.system_prefix:
            result = config.system_prefix + "\n\n" + result
        if config.system_suffix:
            result = result + config.system_suffix

    # Clean up extra whitespace
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = result.strip()

    return result


def adjust_parameters(
    model_name: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None
) -> Dict[str, any]:
    """
    Adjust generation parameters based on model characteristics.

    Args:
        model_name: Model identifier
        max_tokens: Requested max tokens
        temperature: Requested temperature

    Returns:
        Adjusted parameters dict
    """
    config = get_model_config(model_name)

    params = {}

    if max_tokens is not None:
        params["max_tokens"] = int(max_tokens * config.max_tokens_multiplier)

    if temperature is not None:
        params["temperature"] = max(0.0, min(2.0, temperature + config.temperature_offset))

    return params


def supports_json_mode(model_name: str) -> bool:
    """Check if model supports JSON output mode."""
    config = get_model_config(model_name)
    return config.json_mode_supported


def needs_explicit_format(model_name: str) -> bool:
    """Check if model needs explicit format instructions."""
    config = get_model_config(model_name)
    return config.needs_explicit_format


class ModelAwarePromptProcessor:
    """
    Processor that automatically applies model-specific fixes.

    Usage:
        processor = ModelAwarePromptProcessor("qwen3-235b")
        fixed_prompt = processor.process_system_prompt(original_prompt)
        fixed_user_msg = processor.process_user_message(user_input)
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = get_model_config(model_name)

    def process_system_prompt(self, prompt: str) -> str:
        """Process a system prompt with model-specific fixes."""
        return apply_model_fixes(prompt, self.model_name, is_system_prompt=True)

    def process_user_message(self, message: str) -> str:
        """Process a user message (minimal fixes)."""
        # User messages typically need fewer adjustments
        return message

    def get_adjusted_params(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, any]:
        """Get adjusted generation parameters."""
        return adjust_parameters(self.model_name, max_tokens, temperature)

    @property
    def supports_json(self) -> bool:
        """Check JSON mode support."""
        return self.config.json_mode_supported

    @property
    def needs_format_hints(self) -> bool:
        """Check if explicit format hints needed."""
        return self.config.needs_explicit_format
