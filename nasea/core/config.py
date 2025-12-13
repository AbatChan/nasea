"""
Configuration management for NASEA.
Loads settings from environment variables and .env file.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file
# Try multiple locations: current directory, package directory, user home
_possible_env_paths = [
    Path.cwd() / ".env",  # Current working directory
    Path(__file__).parent.parent.parent / ".env",  # Project root (nasea/.env)
    Path.home() / ".nasea" / ".env",  # User config directory
]
for _env_path in _possible_env_paths:
    if _env_path.exists():
        load_dotenv(_env_path)
        break
else:
    # Fallback to default behavior
    load_dotenv()


class Config:
    """Configuration manager for NASEA."""

    def __init__(self):
        """Initialize configuration from environment variables."""

        # API Keys
        self.venice_api_key: Optional[str] = os.getenv("VENICE_API_KEY")
        self.kimi_api_key: Optional[str] = os.getenv("KIMI_API_KEY")
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

        # Model Configuration
        self.default_model: str = os.getenv("DEFAULT_MODEL", "gpt-4-turbo")
        self.fallback_model: str = os.getenv("FALLBACK_MODEL", "gpt-4")
        self.temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
        self.max_tokens: int = int(os.getenv("MAX_TOKENS", "4096"))

        # Generation Settings
        self.max_iterations: int = int(os.getenv("MAX_ITERATIONS", "3"))
        # Reserved for future use:
        # self.max_file_size: int = int(os.getenv("MAX_FILE_SIZE", "10000"))
        # self.max_files: int = int(os.getenv("MAX_FILES", "50"))
        # self.execution_timeout: int = int(os.getenv("EXECUTION_TIMEOUT", "60"))

        # Output Configuration
        self.output_dir: Path = Path(os.getenv("OUTPUT_DIR", "./output"))
        self.use_timestamps: bool = os.getenv("USE_TIMESTAMPS", "true").lower() == "true"
        # Reserved: self.save_intermediate: bool = os.getenv("SAVE_INTERMEDIATE", "false").lower() == "true"

        # Logging
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.console_log_level: str = os.getenv("CONSOLE_LOG_LEVEL", "WARNING")
        self.log_file: Path = Path(os.getenv("LOG_FILE", "./nasea.log"))
        self.console_logging: bool = os.getenv("CONSOLE_LOGGING", "true").lower() == "true"

        # Testing & Verification
        self.auto_generate_tests: bool = os.getenv("AUTO_GENERATE_TESTS", "true").lower() == "true"
        self.run_static_analysis: bool = os.getenv("RUN_STATIC_ANALYSIS", "true").lower() == "true"
        self.run_security_scan: bool = os.getenv("RUN_SECURITY_SCAN", "true").lower() == "true"
        # Reserved: self.min_coverage: int = int(os.getenv("MIN_COVERAGE", "70"))

        # Memory & Caching
        self.enable_caching: bool = os.getenv("ENABLE_CACHING", "true").lower() == "true"
        self.cache_dir: Path = Path(os.getenv("CACHE_DIR", "./.cache"))
        # Reserved: self.cache_ttl: int = int(os.getenv("CACHE_TTL", "86400"))

        # Session State Tracking (reduces redundant work)
        # "auto" = enable for complex tasks (>3 files or multi-step)
        # "always" = always enable
        # "never" = always disable
        self.session_tracking: str = os.getenv("SESSION_TRACKING", "auto").lower()
        if self.session_tracking not in ("auto", "always", "never"):
            self.session_tracking = "auto"

        # Advanced Settings
        self.debug_mode: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
        # Reserved for future use:
        # self.use_docker_sandbox: bool = os.getenv("USE_DOCKER_SANDBOX", "false").lower() == "true"
        # self.enable_web_ui: bool = os.getenv("ENABLE_WEB_UI", "false").lower() == "true"
        # self.rate_limit: int = int(os.getenv("RATE_LIMIT", "20"))

        # Intent/chat models (lightweight)
        self.intent_model: str = os.getenv("INTENT_MODEL", os.getenv("CHAT_MODEL", "venice-uncensored"))
        self.intent_temperature: float = float(os.getenv("INTENT_TEMPERATURE", "0.2"))

        # Mock / offline mode
        self.mock_mode: bool = os.getenv("MOCK_MODE", "false").lower() in {"true", "1", "yes"}

        # Database
        self.database_path: Path = Path(os.getenv("DATABASE_PATH", "./nasea.db"))
        # Reserved: self.vector_collection: str = os.getenv("VECTOR_COLLECTION", "nasea_memory")

        # Privacy & Security (reserved for future use)
        # self.redact_logs: bool = os.getenv("REDACT_LOGS", "true").lower() == "true"
        # self.telemetry_enabled: bool = os.getenv("TELEMETRY_ENABLED", "false").lower() == "true"

        # Validate configuration
        self._validate()

        # Setup logging
        self._setup_logging()

        # Apply mock mode overrides after validation/logging setup
        if self.mock_mode:
            self.auto_generate_tests = False
            self.run_static_analysis = False
            self.run_security_scan = False
            # Disable caching to keep behaviour transparent during offline demos
            self.enable_caching = False

    def _validate(self):
        """Validate configuration settings."""
        if not self.venice_api_key and not self.kimi_api_key and not self.openai_api_key:
            logger.warning(
                "No API keys found! Set VENICE_API_KEY, KIMI_API_KEY, or OPENAI_API_KEY in .env file"
            )

        # Create necessary directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Validate model selection
        valid_models = [
            "deepseek-chat", "deepseek-reasoner",  # DeepSeek (cheapest, good quality)
            "qwen3-235b", "qwen3-coder-480b-a35b-instruct", "venice-uncensored", "mistral-31-24b", "qwen3-4b", "llama-3.3-70b",  # Venice AI
            "kimi-k2",  # Kimi
            "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",  # OpenAI
            "claude-3-sonnet"  # Anthropic
        ]
        if self.default_model not in valid_models:
            logger.warning(
                f"Unknown model '{self.default_model}'. Valid options: {valid_models}"
            )

    def _setup_logging(self):
        """Configure logging based on settings."""
        logger.remove()  # Remove default handler

        if self.console_logging:
            logger.add(
                lambda msg: print(msg, end=""),
                level=self.console_log_level,
                colorize=True,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
            )

        logger.add(
            self.log_file,
            level=self.log_level,
            rotation="10 MB",
            retention="1 week",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}"
        )

        if self.debug_mode:
            logger.info("Debug mode enabled")
        if self.mock_mode:
            logger.info("Mock mode enabled (LLM responses will be simulated)")

    def get_api_key(self, model: Optional[str] = None) -> Optional[str]:
        """Get the appropriate API key for the specified model."""
        target_model = model or self.default_model

        if "kimi" in target_model.lower():
            return self.kimi_api_key
        elif "gpt" in target_model.lower():
            return self.openai_api_key
        elif "claude" in target_model.lower():
            return self.anthropic_api_key

        # Default to OpenAI if available
        return self.openai_api_key or self.kimi_api_key

    def __repr__(self) -> str:
        """String representation of config."""
        api_status = []
        if self.kimi_api_key:
            api_status.append("Kimi K2")
        if self.openai_api_key:
            api_status.append("OpenAI")
        if self.anthropic_api_key:
            api_status.append("Anthropic")

        return (
            f"Config(model={self.default_model}, "
            f"apis=[{', '.join(api_status)}], "
            f"max_iterations={self.max_iterations})"
        )


# Global config instance
config = Config()
