"""
LLM Factory - Creates and manages LLM clients with automatic fallback.
"""

import os
from typing import Optional
from loguru import logger

from nasea.llm.base_client import BaseLLMClient
from nasea.llm.openai_client import OpenAIClient
from nasea.llm.kimi_client import KimiClient
from nasea.llm.venice_client import VeniceLLMClient
from nasea.llm.mock_client import MockLLMClient
from nasea.core.config import Config
from nasea.core.utils import get_max_output_tokens


class LLMFactory:
    """Factory for creating LLM clients with automatic fallback."""

    @staticmethod
    def create_client(
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
        temperature: Optional[float] = None
    ) -> BaseLLMClient:
        """
        Create an LLM client based on model name.

        Args:
            model: Model identifier (e.g., 'kimi-k2', 'gpt-4-turbo')
            api_key: API key (will use config if not provided)
            config: Configuration object
            temperature: Sampling temperature

        Returns:
            Initialized LLM client

        Raises:
            ValueError: If no suitable client can be created
        """
        from nasea.core.config import config as default_config

        config = config or default_config
        model = model or config.default_model
        temperature = temperature or config.temperature

        logger.info(f"Creating LLM client for model: {model}")

        # Short-circuit to mock client when offline mode is enabled
        if getattr(config, "mock_mode", False):
            logger.warning("Mock mode active – using MockLLMClient.")
            return MockLLMClient(model="mock-llm", temperature=temperature)

        # Determine which client to use based on model name
        if "deepseek" in model.lower():
            return LLMFactory._create_deepseek_client(model, api_key, config, temperature)

        elif ("venice" in model.lower() or "qwen3" in model.lower() or
            "mistral-31" in model.lower() or "llama" in model.lower() or
            "dolphin" in model.lower()):
            return LLMFactory._create_venice_client(model, api_key, config, temperature)

        elif "kimi" in model.lower() or "moonshot" in model.lower():
            return LLMFactory._create_kimi_client(model, api_key, config, temperature)

        elif "gpt" in model.lower() or "chatgpt" in model.lower():
            return LLMFactory._create_openai_client(model, api_key, config, temperature)

        elif "claude" in model.lower():
            # Placeholder for future Anthropic support
            logger.warning("Claude support not yet implemented, falling back to OpenAI")
            return LLMFactory._create_openai_client(
                config.fallback_model, api_key, config, temperature
            )

        else:
            # Unknown model, try OpenAI as default
            logger.warning(f"Unknown model '{model}', attempting OpenAI client")
            return LLMFactory._create_openai_client(model, api_key, config, temperature)

    @staticmethod
    def _create_deepseek_client(
        model: str,
        api_key: Optional[str],
        config: Config,
        temperature: float
    ) -> BaseLLMClient:
        """Create DeepSeek client."""
        from nasea.llm.deepseek_client import DeepSeekLLMClient

        key = api_key or os.getenv("DEEPSEEK_API_KEY")

        if not key:
            logger.warning("DeepSeek API key not found, falling back to Venice")
            return LLMFactory._create_venice_client(
                "qwen3-235b", None, config, temperature
            )

        try:
            client = DeepSeekLLMClient(
                api_key=key,
                model=model,
                temperature=temperature,
                max_tokens=get_max_output_tokens(model)
            )
            logger.info(f"DeepSeek client initialized: {model}")
            return client

        except Exception as e:
            logger.error(f"Failed to create DeepSeek client: {e}")
            logger.info("Falling back to Venice")
            return LLMFactory._create_venice_client(
                "qwen3-235b", None, config, temperature
            )

    @staticmethod
    def _create_venice_client(
        model: str,
        api_key: Optional[str],
        config: Config,
        temperature: float
    ) -> BaseLLMClient:
        """Create Venice AI client."""
        key = api_key or config.venice_api_key

        if not key:
            if getattr(config, "mock_mode", False):
                logger.warning("Venice API key missing but mock mode enabled; using mock client.")
                return MockLLMClient(model="mock-llm", temperature=temperature)
            logger.warning("Venice API key not found, falling back to OpenAI")
            return LLMFactory._create_openai_client(
                config.fallback_model, None, config, temperature
            )

        try:
            client = VeniceLLMClient(
                api_key=key,
                model=model,
                temperature=temperature,
                max_tokens=get_max_output_tokens(model)
            )
            logger.info(f"Venice AI client initialized: {model}")
            return client

        except Exception as e:
            logger.error(f"Failed to create Venice client: {e}")
            if getattr(config, "mock_mode", False):
                logger.warning("Falling back to MockLLMClient because mock mode is enabled.")
                return MockLLMClient(model="mock-llm", temperature=temperature)
            logger.info("Falling back to OpenAI")
            return LLMFactory._create_openai_client(
                config.fallback_model, None, config, temperature
            )

    @staticmethod
    def _create_kimi_client(
        model: str,
        api_key: Optional[str],
        config: Config,
        temperature: float
    ) -> BaseLLMClient:
        """Create Kimi K2 client."""
        key = api_key or config.kimi_api_key

        if not key:
            if getattr(config, "mock_mode", False):
                logger.warning("Kimi API key missing but mock mode enabled; using mock client.")
                return MockLLMClient(model="mock-llm", temperature=temperature)
            logger.warning("Kimi API key not found, falling back to OpenAI")
            return LLMFactory._create_openai_client(
                config.fallback_model, None, config, temperature
            )

        try:
            client = KimiClient(api_key=key, model=model, temperature=temperature)
            logger.info(f"Kimi K2 client initialized: {model}")
            return client

        except Exception as e:
            logger.error(f"Failed to create Kimi client: {e}")
            if getattr(config, "mock_mode", False):
                logger.warning("Falling back to MockLLMClient because mock mode is enabled.")
                return MockLLMClient(model="mock-llm", temperature=temperature)
            logger.info("Falling back to OpenAI")
            return LLMFactory._create_openai_client(
                config.fallback_model, None, config, temperature
            )

    @staticmethod
    def _create_openai_client(
        model: str,
        api_key: Optional[str],
        config: Config,
        temperature: float
    ) -> BaseLLMClient:
        """Create OpenAI client."""
        key = api_key or config.openai_api_key

        if not key:
            if getattr(config, "mock_mode", False):
                logger.warning("OpenAI key missing but mock mode is enabled; using mock client.")
                return MockLLMClient(model="mock-llm", temperature=temperature)
            raise ValueError(
                "No OpenAI API key found. Set OPENAI_API_KEY in your .env file"
            )

        try:
            client = OpenAIClient(api_key=key, model=model, temperature=temperature)
            logger.info(f"OpenAI client initialized: {model}")
            return client

        except Exception as e:
            logger.error(f"Failed to create OpenAI client: {e}")
            if getattr(config, "mock_mode", False):
                logger.warning("Falling back to MockLLMClient due to error while mock mode is enabled.")
                return MockLLMClient(model="mock-llm", temperature=temperature)
            raise ValueError(
                f"Could not initialize any LLM client. Error: {e}\n"
                "Please check your API keys and internet connection."
            )

    @staticmethod
    def create_with_fallback(config: Optional[Config] = None) -> BaseLLMClient:
        """
        Create LLM client with automatic fallback chain.

        Tries in order:
        1. Default model from config
        2. Fallback model from config
        3. GPT-3.5-turbo as last resort

        Args:
            config: Configuration object

        Returns:
            Working LLM client

        Raises:
            ValueError: If no client can be created
        """
        from nasea.core.config import config as default_config

        config = config or default_config

        # Try default model
        try:
            return LLMFactory.create_client(
                model=config.default_model,
                config=config
            )
        except Exception as e:
            logger.warning(f"Default model failed: {e}")

        # Try fallback model
        try:
            logger.info(f"Trying fallback model: {config.fallback_model}")
            return LLMFactory.create_client(
                model=config.fallback_model,
                config=config
            )
        except Exception as e:
            logger.warning(f"Fallback model failed: {e}")

        # Last resort: GPT-3.5-turbo
        try:
            logger.info("Trying last resort: gpt-3.5-turbo")
            return LLMFactory.create_client(
                model="gpt-3.5-turbo",
                config=config
            )
        except Exception as e:
            if getattr(config, "mock_mode", False):
                logger.warning("All external models unavailable – using MockLLMClient.")
                return MockLLMClient(model="mock-llm", temperature=config.temperature)
            raise ValueError(
                f"All LLM clients failed. Please check:\n"
                f"1. Your API keys are set in .env file\n"
                f"2. You have internet connection\n"
                f"3. The API services are not down\n"
                f"Last error: {e}"
            )


    @staticmethod
    def get_context_window(model: Optional[str], default: int = 4096) -> int:
        """
        Return the approximate maximum context window for the given model.

        Args:
            model: Model identifier.
            default: Fallback window size when unknown.

        Returns:
            Context window in tokens.
        """
        if not model:
            return default

        if "mock" in model.lower():
            return default

        # Use centralized config from model_limits.yaml
        from nasea.core.utils import get_context_window as get_ctx
        return get_ctx(model) or default


def create_llm_client(
    model: Optional[str] = None,
    config: Optional[Config] = None
) -> BaseLLMClient:
    """
    Convenience function to create an LLM client.

    Args:
        model: Model name (optional, uses config default)
        config: Configuration (optional, uses global config)

    Returns:
        Initialized LLM client
    """
    if model:
        return LLMFactory.create_client(model=model, config=config)
    else:
        return LLMFactory.create_with_fallback(config=config)
