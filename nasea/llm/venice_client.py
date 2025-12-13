"""
Venice AI LLM Client

Provides uncensored AI model access through Venice AI's API.
Venice uses OpenAI-compatible API format.
"""

import time
import logging
from typing import Optional, Dict, Any, Callable, TypeVar

from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError

from nasea.llm.base_client import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)

T = TypeVar('T')


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
) -> T:
    """
    Retry a function with exponential backoff on network errors.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        backoff_factor: Multiplier for delay after each retry

    Returns:
        Result of the function

    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except (APIConnectionError, APITimeoutError, ConnectionError, TimeoutError, InternalServerError) as e:
            last_exception = e
            if attempt == max_retries:
                logger.error(f"All {max_retries} retries failed: {e}")
                raise

            error_type = "Server error (500)" if isinstance(e, InternalServerError) else "Network error"
            logger.warning(f"{error_type} (attempt {attempt + 1}/{max_retries + 1}): {e}")
            logger.info(f"Retrying in {delay:.1f}s...")
            time.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)
        except RateLimitError as e:
            last_exception = e
            if attempt == max_retries:
                logger.error(f"Rate limit exceeded after {max_retries} retries")
                raise

            # Rate limits usually need longer waits
            wait_time = min(delay * 2, 60.0)
            logger.warning(f"Rate limited (attempt {attempt + 1}/{max_retries + 1})")
            logger.info(f"Waiting {wait_time:.1f}s before retry...")
            time.sleep(wait_time)
            delay = min(delay * backoff_factor, max_delay)

    raise last_exception


class VeniceLLMClient(BaseLLMClient):
    """
    Client for Venice AI uncensored models.

    Venice AI provides privacy-focused, uncensored language models
    with an OpenAI-compatible API interface.

    Available models:
    - qwen3-235b: Advanced reasoning and agents
    - mistral-31-24b: Vision and tool use
    - qwen3-4b: Fast, lightweight tasks
    - venice-uncensored: Unfiltered generation
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-235b",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """
        Initialize Venice AI client.

        Args:
            api_key: Venice API key
            model: Model name (qwen3-235b, venice-uncensored, etc.)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        super().__init__(api_key=api_key, model=model, temperature=temperature)

        # Venice uses OpenAI-compatible API with custom base URL
        # Note: qwen3-235b is a very large model and may take longer to respond
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.venice.ai/api/v1",
            timeout=120.0  # 120 second timeout (2 minutes) for large models like qwen3-235b
        )
        self.max_tokens = max_tokens

    def is_available(self) -> bool:
        """
        Check if Venice AI service is available.

        Returns:
            True if available, False otherwise
        """
        try:
            # Try a simple completion to check availability
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            return True
        except Exception:
            return False

    def complete(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        disable_thinking: Optional[bool] = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> LLMResponse:
        """
        Generate a completion using Venice AI.

        Args:
            prompt: User prompt
            system_message: System instructions (optional)
            max_tokens: Override default max tokens (optional)
            temperature: Override default temperature (optional)

        Returns:
            LLMResponse with generated content
        """
        # Build messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        # Use provided values or defaults
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                finish_reason=response.choices[0].finish_reason,
                metadata={"id": response.id}
            )

        except Exception as e:
            raise Exception(f"Venice AI API error: {str(e)}")

    def chat(
        self,
        messages: list,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        disable_thinking: Optional[bool] = None,
        strip_thinking: Optional[bool] = None
    ) -> LLMResponse:
        """
        Generate a chat completion using Venice AI.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Override default max tokens (optional)
            temperature: Override default temperature (optional)
            disable_thinking: Set True to disable chain-of-thought reasoning
            strip_thinking: Set True to strip thinking content from responses

        Returns:
            LLMResponse with generated content
        """
        # Use provided values or defaults
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        try:
            request_kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": tokens,
            }
            extra_body: Dict[str, Any] = {}
            venice_params: Dict[str, Any] = {}
            if disable_thinking is not None:
                venice_params["disable_thinking"] = disable_thinking
            if strip_thinking is not None:
                venice_params["strip_thinking_response"] = strip_thinking
            if venice_params:
                extra_body["venice_parameters"] = venice_params
            if extra_body:
                request_kwargs["extra_body"] = extra_body

            response = self.client.chat.completions.create(**request_kwargs)

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                finish_reason=response.choices[0].finish_reason,
                metadata={"id": response.id}
            )

        except Exception as e:
            raise Exception(f"Venice AI API error: {str(e)}")

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text using Venice AI model.

        Args:
            prompt: User prompt/query
            system_message: System instructions (optional)
            temperature: Override default temperature (optional)
            max_tokens: Override default max tokens (optional)

        Returns:
            Generated text response

        Raises:
            Exception: If API call fails
        """
        # Build messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        # Use provided values or defaults
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        try:
            # Call Venice API (OpenAI-compatible)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )

            # Extract and return the generated text
            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"Venice AI API error: {str(e)}")

    def generate_with_json(
        self,
        prompt: str,
        system_message: Optional[str] = None,
    ) -> str:
        """
        Generate JSON response using Venice AI.

        Args:
            prompt: User prompt
            system_message: System instructions

        Returns:
            JSON string response
        """
        # Venice supports JSON mode through system message
        json_instruction = "\n\nIMPORTANT: Respond ONLY with valid JSON. No additional text."

        enhanced_system = system_message or ""
        enhanced_system += json_instruction

        return self.generate(
            prompt=prompt,
            system_message=enhanced_system,
            temperature=0.3,  # Lower temperature for structured output
        )

    def stream_complete(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ):
        """
        Stream a completion token-by-token using Venice AI.

        Args:
            prompt: User prompt
            system_message: System instructions (optional)
            max_tokens: Override default max tokens (optional)
            temperature: Override default temperature (optional)

        Yields:
            String chunks as they arrive from the API
        """
        # Build messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        # Use provided values or defaults
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        try:
            # Call Venice API with streaming enabled
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                stream=True,  # Enable streaming
            )

            # Yield chunks as they arrive
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise Exception(f"Venice AI streaming error: {str(e)}")

    def stream_chat(
        self,
        messages: list,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        disable_thinking: Optional[bool] = None,
        strip_thinking: Optional[bool] = None
    ):
        """
        Stream a chat completion token-by-token using Venice AI.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Override default max tokens (optional)
            temperature: Override default temperature (optional)
            disable_thinking: Set True to disable chain-of-thought reasoning
            strip_thinking: Set True to strip thinking content from responses

        Yields:
            The raw stream object (caller should iterate and handle closing)
        """
        # Use provided values or defaults
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        # Build request kwargs
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
            "stream": True,
        }

        # Add Venice-specific parameters
        extra_body: Dict[str, Any] = {}
        venice_params: Dict[str, Any] = {}
        if disable_thinking is not None:
            venice_params["disable_thinking"] = disable_thinking
        if strip_thinking is not None:
            venice_params["strip_thinking_response"] = strip_thinking
        if venice_params:
            extra_body["venice_parameters"] = venice_params
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        # Return the stream object with retry on network errors
        def make_request():
            return self.client.chat.completions.create(**request_kwargs)

        return retry_with_backoff(make_request, max_retries=3)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate API call cost.

        Venice pricing varies by model. This provides rough estimates.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        # Venice uses "Diems" system - rough USD estimates
        # Prices approximate as of 2025
        pricing = {
            "qwen3-235b": {"input": 0.005, "output": 0.015},
            "qwen3-coder-480b-a35b-instruct": {"input": 0.006, "output": 0.018},
            "mistral-31-24b": {"input": 0.003, "output": 0.010},
            "qwen3-4b": {"input": 0.001, "output": 0.003},
            "venice-uncensored": {"input": 0.003, "output": 0.010},
        }

        model_pricing = pricing.get(self.model, pricing["qwen3-235b"])

        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]

        return input_cost + output_cost
