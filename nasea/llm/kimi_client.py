"""
Kimi K2 LLM client implementation.
Moonshot AI's Kimi K2 model with OpenAI-compatible API.
"""

from typing import List, Dict, Optional
from loguru import logger

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not installed. Run: pip install openai")

from nasea.llm.base_client import BaseLLMClient, LLMResponse


class KimiClient(BaseLLMClient):
    """
    Kimi K2 LLM client.
    Uses OpenAI-compatible API endpoint from Moonshot AI.
    """

    KIMI_BASE_URL = "https://api.moonshot.cn/v1"
    DEFAULT_MODEL = "moonshot-v1-8k"  # or kimi-k2-instruct if available

    def __init__(
        self,
        api_key: str,
        model: str = None,
        temperature: float = 0.7,
        base_url: str = None
    ):
        """
        Initialize Kimi K2 client.

        Args:
            api_key: Kimi API key from Moonshot AI
            model: Model name (moonshot-v1-8k, moonshot-v1-32k, etc.)
            temperature: Sampling temperature
            base_url: Custom API base URL
        """
        model = model or self.DEFAULT_MODEL
        super().__init__(api_key, model, temperature)

        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed (required for Kimi client)")

        self.base_url = base_url or self.KIMI_BASE_URL

        # Initialize OpenAI-compatible client with Kimi endpoint
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=60.0
        )

        logger.info(f"Kimi K2 client initialized: {model}")

    def complete(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> LLMResponse:
        """
        Generate a completion.

        Args:
            prompt: User prompt
            system_message: System message
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            LLMResponse
        """
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        return self.chat(messages, max_tokens, temperature)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> LLMResponse:
        """
        Generate a chat completion.

        Args:
            messages: List of messages
            max_tokens: Max tokens
            temperature: Temperature

        Returns:
            LLMResponse
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or 4096
            )

            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0

            logger.debug(f"Kimi K2 response: {tokens_used} tokens")

            return LLMResponse(
                content=content,
                model=self.model,
                tokens_used=tokens_used,
                finish_reason=response.choices[0].finish_reason,
                metadata={
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "provider": "kimi_k2"
                }
            )

        except Exception as e:
            logger.error(f"Kimi K2 API error: {e}")
            raise

    def is_available(self) -> bool:
        """Check if Kimi service is available."""
        try:
            # Simple test request
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.warning(f"Kimi K2 not available: {e}")
            return False

    def stream_complete(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ):
        """
        Stream a completion token-by-token.

        Args:
            prompt: User prompt
            system_message: System message
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Yields:
            String chunks as they arrive
        """
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        yield from self.stream_chat(messages, max_tokens, temperature)

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ):
        """
        Stream a chat completion token-by-token.

        Args:
            messages: List of messages
            max_tokens: Max tokens
            temperature: Temperature

        Yields:
            String chunks as they arrive
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or 4096,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Kimi K2 streaming error: {e}")
            raise

    @classmethod
    def list_models(cls, api_key: str) -> List[str]:
        """
        List available Kimi models.

        Args:
            api_key: Kimi API key

        Returns:
            List of model names
        """
        try:
            client = OpenAI(api_key=api_key, base_url=cls.KIMI_BASE_URL, timeout=60.0)
            models = client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.warning(f"Could not list Kimi models: {e}")
            return [cls.DEFAULT_MODEL]
