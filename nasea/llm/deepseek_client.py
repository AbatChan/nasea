"""
DeepSeek LLM Client

Provides cheap, high-quality AI model access through DeepSeek's API.
DeepSeek uses OpenAI-compatible API format.

Pricing (as of late 2025):
- Input: $0.28 per 1M tokens (new content), $0.028 cached
- Output: $0.42 per 1M tokens
- 10-30x cheaper than OpenAI
"""

import logging
from typing import Optional, Dict, Any

from openai import OpenAI

from nasea.llm.base_client import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


class DeepSeekLLMClient(BaseLLMClient):
    """
    Client for DeepSeek models.

    DeepSeek provides affordable, high-quality language models
    with an OpenAI-compatible API interface.

    Available models:
    - deepseek-chat: General chat and coding (DeepSeek V3)
    - deepseek-reasoner: Advanced reasoning (DeepSeek R1)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """
        Initialize DeepSeek client.

        Args:
            api_key: DeepSeek API key
            model: Model name (deepseek-chat, deepseek-reasoner)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        super().__init__(api_key=api_key, model=model, temperature=temperature)

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=120.0
        )
        self.max_tokens = max_tokens

    def is_available(self) -> bool:
        """Check if DeepSeek service is available."""
        try:
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
        """Generate a completion using DeepSeek."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

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
            raise Exception(f"DeepSeek API error: {str(e)}")

    def chat(
        self,
        messages: list,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        disable_thinking: Optional[bool] = None,
        strip_thinking: Optional[bool] = None
    ) -> LLMResponse:
        """Generate a chat completion using DeepSeek."""
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
            raise Exception(f"DeepSeek API error: {str(e)}")

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using DeepSeek model."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )
            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"DeepSeek API error: {str(e)}")

    def generate_with_json(
        self,
        prompt: str,
        system_message: Optional[str] = None,
    ) -> str:
        """Generate JSON response using DeepSeek."""
        json_instruction = "\n\nIMPORTANT: Respond ONLY with valid JSON. No additional text."
        enhanced_system = (system_message or "") + json_instruction

        return self.generate(
            prompt=prompt,
            system_message=enhanced_system,
            temperature=0.3,
        )

    def stream_chat(
        self,
        messages: list,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        disable_thinking: Optional[bool] = None,
        strip_thinking: Optional[bool] = None
    ):
        """Stream a chat completion token-by-token."""
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
            max_tokens=tokens,
            stream=True,
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate API call cost in USD."""
        # DeepSeek pricing (late 2025)
        pricing = {
            "deepseek-chat": {"input": 0.00028, "output": 0.00042},  # per 1K tokens
            "deepseek-reasoner": {"input": 0.00055, "output": 0.00219},
        }

        model_pricing = pricing.get(self.model, pricing["deepseek-chat"])
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]

        return input_cost + output_cost
