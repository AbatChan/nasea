"""
Base LLM client interface.
All LLM providers must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str
    model: str
    tokens_used: int
    finish_reason: str
    metadata: Dict[str, Any]


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.7):
        """
        Initialize the LLM client.

        Args:
            api_key: API key for the provider
            model: Model identifier
            temperature: Sampling temperature
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> LLMResponse:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: User prompt
            system_message: System message for role setup
            max_tokens: Maximum tokens to generate
            temperature: Override default temperature

        Returns:
            LLMResponse with generated content
        """
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> LLMResponse:
        """
        Generate a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Override default temperature

        Returns:
            LLMResponse with generated content
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the LLM service is available.

        Returns:
            True if available, False otherwise
        """
        pass

    def stream_complete(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ):
        """
        Stream a completion for the given prompt (token-by-token).

        Args:
            prompt: User prompt
            system_message: System message for role setup
            max_tokens: Maximum tokens to generate
            temperature: Override default temperature

        Yields:
            Chunks of generated content as they arrive
        """
        # Default implementation: yield full response at once (non-streaming fallback)
        response = self.complete(prompt, system_message, max_tokens, temperature)
        yield response.content

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ):
        """
        Stream a chat completion (token-by-token).

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Override default temperature

        Yields:
            Chunks of generated content as they arrive
        """
        # Default implementation: yield full response at once (non-streaming fallback)
        response = self.chat(messages, max_tokens, temperature)
        yield response.content

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(model={self.model})"
