"""
Base Agent class - Foundation for all specialized agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from loguru import logger

from nasea.core.config import Config
from nasea.core.memory import ContextMemory
from nasea.llm.llm_factory import create_llm_client
from nasea.llm.base_client import BaseLLMClient, LLMResponse
from nasea.prompts import get_agent_metadata, PromptMetadata


class BaseAgent(ABC):
    """
    Abstract base class for all NASEA agents.
    Provides common functionality for LLM interaction and memory management.
    """

    def __init__(
        self,
        role: str,
        config: Config,
        memory: ContextMemory,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the base agent.

        Args:
            role: Agent role name (e.g., 'manager', 'developer', 'verifier')
            config: Configuration object
            memory: Shared context memory
            system_prompt: Custom system prompt for this agent
        """
        self.role = role
        self.config = config
        self.memory = memory
        self.system_prompt = system_prompt or self._default_system_prompt()

        # Initialize LLM client (lazy loaded)
        self._llm_client: Optional[BaseLLMClient] = None

        # Load tool permissions from prompt metadata
        self._metadata: Optional[PromptMetadata] = None
        try:
            self._metadata = get_agent_metadata(role)
        except FileNotFoundError:
            logger.debug(f"No prompt metadata found for {role}")

        logger.info(f"{self.role.upper()} agent initialized")

    @property
    def llm(self) -> BaseLLMClient:
        """Get LLM client (lazy loading)."""
        if self._llm_client is None:
            self._llm_client = create_llm_client(config=self.config)
        return self._llm_client

    @abstractmethod
    def _default_system_prompt(self) -> str:
        """
        Get the default system prompt for this agent.
        Must be implemented by subclasses.

        Returns:
            System prompt string
        """
        pass

    def query_llm(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        log_to_memory: bool = True
    ) -> str:
        """
        Query the LLM with caching and memory management.

        Args:
            prompt: User prompt
            system_message: Override system message
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            log_to_memory: Whether to log this interaction

        Returns:
            LLM response content
        """
        # Check cache first
        if self.config.enable_caching:
            cached = self.memory.get_cached_code(prompt)
            if cached:
                logger.debug(f"[{self.role}] Cache hit!")
                return cached[0]  # Return cached code

        # Log the query
        if log_to_memory:
            self.memory.add_entry(
                role=self.role,
                content=f"QUERY: {prompt}",
                metadata={"type": "query"}
            )

        # Query LLM
        try:
            logger.debug(f"[{self.role}] Querying LLM: {prompt[:100]}...")

            response: LLMResponse = self.llm.complete(
                prompt=prompt,
                system_message=system_message or self.system_prompt,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature or self.config.temperature
            )

            content = response.content

            # Log the response
            if log_to_memory:
                self.memory.add_entry(
                    role=self.role,
                    content=f"RESPONSE: {content}",
                    metadata={
                        "type": "response",
                        "tokens": response.tokens_used,
                        "model": response.model
                    }
                )

            # Cache if enabled
            if self.config.enable_caching:
                self.memory.cache_code(prompt, content, "code")

            logger.debug(f"[{self.role}] Response: {response.tokens_used} tokens")

            return content

        except Exception as e:
            logger.error(f"[{self.role}] LLM query failed: {e}")
            raise

    def extract_code_from_response(self, response: str) -> str:
        """
        Extract code from an LLM response that may contain markdown code blocks.

        Args:
            response: LLM response

        Returns:
            Extracted code
        """
        # Look for code blocks
        if "```" in response:
            # Find code between triple backticks
            parts = response.split("```")
            if len(parts) >= 3:
                code_block = parts[1]

                # Remove language identifier (e.g., ```python)
                lines = code_block.strip().split("\n")
                if lines and not lines[0].strip().startswith(("#", "//")):
                    # First line might be language identifier
                    first_line = lines[0].strip().lower()
                    if first_line in ["python", "javascript", "java", "go", "rust", "cpp", "c++", "c"]:
                        lines = lines[1:]

                return "\n".join(lines).strip()

        # No code blocks found, return entire response
        return response.strip()

    def validate_code_syntax(self, code: str, language: str = "python") -> bool:
        """
        Basic syntax validation for generated code.

        Args:
            code: Code to validate
            language: Programming language

        Returns:
            True if syntax is valid
        """
        if language.lower() == "python":
            try:
                compile(code, "<string>", "exec")
                return True
            except SyntaxError as e:
                logger.warning(f"Syntax error in Python code: {e}")
                return False

        # For other languages, just check if not empty
        return bool(code.strip())

    def format_context(self, context: Dict[str, Any]) -> str:
        """
        Format context dictionary into a prompt-friendly string.

        Args:
            context: Context dictionary

        Returns:
            Formatted string
        """
        lines = []
        for key, value in context.items():
            if isinstance(value, (list, dict)):
                import json
                value = json.dumps(value, indent=2)
            lines.append(f"{key}: {value}")

        return "\n".join(lines)

    def log(self, message: str, level: str = "info"):
        """
        Log a message with agent context.

        Args:
            message: Message to log
            level: Log level (debug, info, warning, error)
        """
        formatted_message = f"[{self.role.upper()}] {message}"

        if level == "debug":
            logger.debug(formatted_message)
        elif level == "info":
            logger.info(formatted_message)
        elif level == "warning":
            logger.warning(formatted_message)
        elif level == "error":
            logger.error(formatted_message)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(role={self.role})"

    @property
    def allowed_tools(self) -> Optional[List[str]]:
        """Get list of allowed tools for this agent."""
        if self._metadata:
            return self._metadata.allowed_tools
        return None

    @property
    def forbidden_tools(self) -> Optional[List[str]]:
        """Get list of forbidden tools for this agent."""
        if self._metadata:
            return self._metadata.forbidden_tools
        return None

    def filter_tools(self, all_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter tools based on agent's allowed/forbidden tool lists.

        Args:
            all_tools: List of tool definitions (OpenAI function format)

        Returns:
            Filtered list of tools this agent can use
        """
        if self._metadata:
            return self._metadata.filter_tools(all_tools)
        return all_tools

    def can_use_tool(self, tool_name: str) -> bool:
        """
        Check if this agent is allowed to use a specific tool.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if the agent can use this tool
        """
        # If no metadata, allow all tools
        if not self._metadata:
            return True

        # Check forbidden list first
        if self._metadata.forbidden_tools and tool_name in self._metadata.forbidden_tools:
            return False

        # If allowed_tools is set, tool must be in it
        if self._metadata.allowed_tools:
            return tool_name in self._metadata.allowed_tools

        return True
