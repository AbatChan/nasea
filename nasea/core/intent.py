""""
Intent classification for context-aware chat detection.
"""

from enum import Enum
from typing import Dict, List, Optional


class Intent(Enum):
    """User intent types."""
    CODE_GENERATION = "code_generation"
    CHAT = "chat"
    QUESTION = "question"
    CLARIFICATION = "clarification"


class IntentClassifier:
    """
    Classifies user intent to determine whether to generate code or respond conversationally.
    """

    # Keywords that indicate code generation intent
    CODE_KEYWORDS = [
        "create", "build", "make", "generate", "implement", "write",
        "develop", "code", "function", "class", "api", "app",
        "program", "script", "module", "package", "library",
        "website", "application", "tool", "service", "system",
        "add", "fix", "refactor", "optimize", "improve"
    ]

    # Greetings and casual chat patterns
    CHAT_PATTERNS = [
        "hello", "hi", "hey", "howdy", "sup", "yo",
        "how are you", "what's up", "how's it going",
        "good morning", "good afternoon", "good evening",
        "thanks", "thank you", "thx", "appreciate",
        "bye", "goodbye", "see you", "later",
        "lol", "haha", "nice", "cool", "awesome",
        "ok", "okay", "sure", "alright"
    ]

    # Question patterns
    QUESTION_KEYWORDS = [
        "what", "why", "how", "when", "where", "who",
        "can you", "could you", "would you", "will you",
        "do you", "does it", "is it", "are you",
        "tell me", "explain", "describe", "show me"
    ]

    @classmethod
    def classify(cls, prompt: str, llm_client=None) -> Intent:
        """
        Classify user intent from their prompt.

        Args:
            prompt: User input
            llm_client: Optional LLM client for advanced classification

        Returns:
            Intent classification
        """
        prompt_lower = prompt.lower().strip()

        # 1. Fast Path: Heuristics & Regex

        # Very short prompts are likely greetings
        if len(prompt_lower.split()) <= 3:
            if any(pattern in prompt_lower for pattern in cls.CHAT_PATTERNS):
                return Intent.CHAT

        # Check for chat patterns
        if any(pattern in prompt_lower for pattern in cls.CHAT_PATTERNS):
            # Unless it also contains code keywords
            if not any(keyword in prompt_lower for keyword in cls.CODE_KEYWORDS):
                return Intent.CHAT

        # Check for code generation keywords (High confidence)
        has_code_intent = any(keyword in prompt_lower for keyword in cls.CODE_KEYWORDS)

        # Check for questions
        has_question_intent = any(keyword in prompt_lower for keyword in cls.QUESTION_KEYWORDS)

        if has_code_intent and not has_question_intent:
            return Intent.CODE_GENERATION

        if has_question_intent and not has_code_intent:
            return Intent.QUESTION
            
        # 2. Slow Path: LLM Disambiguation (if client provided and intent ambiguous)
        if llm_client and (has_code_intent and has_question_intent) or len(prompt_lower.split()) > 5:
            try:
                response = llm_client.generate(
                    f"Classify the following user intent into one of these categories: GENERATE (writing new code), CHAT (casual conversation), or QUESTION (asking about existing concepts). \n\nUser Input: \"{prompt}\"\n\nRespond with ONLY one word.",
                    max_tokens=10,
                    temperature=0.1
                )
                response = response.strip().upper()
                if "GENERATE" in response:
                    return Intent.CODE_GENERATION
                if "QUESTION" in response:
                    return Intent.QUESTION
                if "CHAT" in response:
                    return Intent.CHAT
            except Exception:
                # Fallback to default heuristic if LLM fails
                pass

        # Fallback Heuristics
        if has_code_intent: 
            return Intent.CODE_GENERATION
            
        return Intent.CHAT

    @classmethod
    def get_chat_response(cls, prompt: str) -> str:
        """
        Generate appropriate chat response.

        Args:
            prompt: User input

        Returns:
            Conversational response
        """
        prompt_lower = prompt.lower().strip()

        # Greetings
        if any(word in prompt_lower for word in ["hello", "hi", "hey", "howdy"]):
            return ("Hello! I'm NASEA, your AI software engineer. 👋\n\n"
                    "I can help you build software projects from natural language descriptions. "
                    "Just tell me what you want to create!\n\n"
                    "For example: \"Create a REST API for managing tasks\" or \"Build a password validator function\"")

        # Goodbyes
        if any(word in prompt_lower for word in ["bye", "goodbye", "see you", "later"]):
            return "Goodbye! Feel free to come back anytime you need help building something. 👋"

        # Thanks
        if any(word in prompt_lower for word in ["thanks", "thank you", "thx"]):
            return "You're welcome! Happy to help. Let me know if you need anything else! 😊"

        # Questions about capabilities
        if "what can you" in prompt_lower or "what do you" in prompt_lower:
            return ("I can generate complete software projects from natural language descriptions.\n\n"
                    "I specialize in:\n"
                    "• Python applications (APIs, CLIs, libraries)\n"
                    "• Functions and classes with proper documentation\n"
                    "• Test generation and quality checks\n"
                    "• Project structure and best practices\n\n"
                    "Just describe what you want to build, and I'll create it for you!")

        # Friendly check-ins
        if ("how are" in prompt_lower and "you" in prompt_lower) or "how's it going" in prompt_lower:
            return ("I'm doing great—ready to code whenever you are! 🚀\n\n"
                    "Tell me what you'd like to build (e.g., \"Create a Flask API\" or \"Write a data validation function\") "
                    "and I'll get started.")

        # How questions
        if prompt_lower.startswith("how"):
            return ("To use me, simply describe what you want to build in plain English.\n\n"
                    "Examples:\n"
                    "• \"Create a function to check if a number is prime\"\n"
                    "• \"Build a REST API for managing books\"\n"
                    "• \"Make a password validator with specific rules\"\n\n"
                    "I'll analyze your request, plan the project, write the code, and test it!")

        # Generic help
        if "help" in prompt_lower:
            return ("I'm here to help you build software! Here's how to use me:\n\n"
                    "1. Describe what you want to create\n"
                    "2. I'll analyze and plan the project\n"
                    "3. I'll write the code with tests\n"
                    "4. You'll get a complete, working project\n\n"
                    "Commands:\n"
                    "• `/help` - Show all commands\n"
                    "• `/model <name>` - Switch AI model\n"
                    "• `/mock` or `/live` - Toggle demo mode\n"
                    "• `/exit` - Quit\n\n"
                    "What would you like to build?")

        # Default conversational response
        return ("I'm not sure how to respond to that, but I'm here to help you build software!\n\n"
                "Try asking me to create something specific, like:\n"
                "• \"Create a calculator function\"\n"
                "• \"Build a task management API\"\n"
                "• \"Make a data validation utility\"\n\n"
                "What would you like to build today?")