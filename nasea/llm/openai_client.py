"""
OpenAI LLM client implementation.
Supports GPT-4, GPT-4-turbo, GPT-3.5-turbo, etc.
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


class OpenAIClient(BaseLLMClient):
    """OpenAI LLM client."""

    def __init__(self, api_key: str, model: str = "gpt-4-turbo", temperature: float = 0.7):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model name (gpt-4, gpt-4-turbo, gpt-3.5-turbo, etc.)
            temperature: Sampling temperature
        """
        super().__init__(api_key, model, temperature)

        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed")

        self.client = OpenAI(api_key=api_key, timeout=60.0)
        logger.info(f"OpenAI client initialized: {model}")

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
            tokens_used = response.usage.total_tokens

            logger.debug(f"OpenAI response: {tokens_used} tokens")

            return LLMResponse(
                content=content,
                model=self.model,
                tokens_used=tokens_used,
                finish_reason=response.choices[0].finish_reason,
                metadata={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def is_available(self) -> bool:
        """Check if OpenAI service is available."""
        try:
            # Simple test request
            self.client.models.list()
            return True
        except Exception as e:
            logger.warning(f"OpenAI not available: {e}")
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
            logger.error(f"OpenAI streaming error: {e}")
            raise
