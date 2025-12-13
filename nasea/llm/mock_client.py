"""
Mock LLM client used for offline/demo mode.
Generates deterministic responses to exercise the agent pipeline without network access.
"""

import json
import re
from typing import List, Dict, Optional, Any, Iterator
from datetime import datetime
from dataclasses import dataclass

from nasea.llm.base_client import BaseLLMClient, LLMResponse


@dataclass
class MockChoice:
    """Mock choice object for OpenAI-style responses."""
    index: int = 0
    message: Any = None
    delta: Any = None
    finish_reason: Optional[str] = None


@dataclass
class MockMessage:
    """Mock message object."""
    role: str = "assistant"
    content: str = ""
    tool_calls: Optional[List[Any]] = None


@dataclass
class MockDelta:
    """Mock delta for streaming."""
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Any]] = None


@dataclass
class MockStreamChunk:
    """Mock streaming chunk."""
    choices: List[MockChoice] = None

    def __post_init__(self):
        if self.choices is None:
            self.choices = []


@dataclass
class MockChatCompletion:
    """Mock chat completion response."""
    choices: List[MockChoice] = None
    model: str = "mock-llm"
    usage: Dict[str, int] = None

    def __post_init__(self):
        if self.choices is None:
            self.choices = []
        if self.usage is None:
            self.usage = {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60}


class MockCompletions:
    """Mock completions endpoint that mimics OpenAI's interface."""

    def __init__(self, client: "MockLLMClient"):
        self.client = client

    def create(
        self,
        model: str = "mock-llm",
        messages: List[Dict[str, str]] = None,
        tools: List[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs
    ) -> Any:
        """Create a completion, optionally streaming."""
        messages = messages or []

        # Get the response content
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        system_message = next((m["content"] for m in messages if m["role"] == "system"), "")
        prompt = user_messages[-1] if user_messages else ""

        content = self.client._generate_chat_response(prompt, system_message, tools)

        if stream:
            return self._stream_response(content)
        else:
            return MockChatCompletion(
                choices=[MockChoice(
                    index=0,
                    message=MockMessage(role="assistant", content=content),
                    finish_reason="stop"
                )],
                model=model
            )

    def _stream_response(self, content: str) -> Iterator[MockStreamChunk]:
        """Stream the response word by word."""
        words = content.split(" ")
        for i, word in enumerate(words):
            chunk_content = word + (" " if i < len(words) - 1 else "")
            yield MockStreamChunk(
                choices=[MockChoice(
                    index=0,
                    delta=MockDelta(
                        role="assistant" if i == 0 else None,
                        content=chunk_content
                    ),
                    finish_reason=None if i < len(words) - 1 else "stop"
                )]
            )


class MockChat:
    """Mock chat namespace."""

    def __init__(self, client: "MockLLMClient"):
        self.completions = MockCompletions(client)


class MockLLMClient(BaseLLMClient):
    """Simple rule-based client that simulates LLM behaviour for demos."""

    def __init__(self, model: str = "mock-llm", temperature: float = 0.1):
        super().__init__(api_key="mock", model=model, temperature=temperature)
        self._call_count = 0
        # OpenAI-compatible interface
        self.chat = MockChat(self)

    # Public API -----------------------------------------------------------------
    def complete(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> LLMResponse:
        """Return a deterministic response tailored to the requesting agent."""
        self._call_count += 1

        content = self._generate_response(prompt, system_message or "")
        tokens = len(content.split())

        return LLMResponse(
            content=content,
            model=self.model,
            tokens_used=tokens,
            finish_reason="stop",
            metadata={
                "mock": True,
                "call": self._call_count,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> LLMResponse:
        """Support chat() by forwarding the last user message to complete()."""
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        system_message = next((m["content"] for m in messages if m["role"] == "system"), "")
        prompt = user_messages[-1] if user_messages else ""
        return self.complete(prompt=prompt, system_message=system_message, max_tokens=max_tokens, temperature=temperature)

    def is_available(self) -> bool:
        """Always available – no network call required."""
        return True

    # Internal helpers -----------------------------------------------------------
    def _generate_chat_response(self, prompt: str, system_message: str, tools: List[Dict] = None) -> str:
        """Generate response for chat mode with tool awareness."""
        prompt_lower = prompt.lower()

        # Handle common chat queries
        if "list" in prompt_lower and "file" in prompt_lower:
            return "I'll list the files in the current directory for you.\n\nBased on what I can see, this appears to be a Python project with the following structure:\n- `nasea/` - Main source code\n- `tests/` - Test files\n- `examples/` - Example usage\n- `requirements.txt` - Dependencies\n- `setup.py` - Package configuration\n\nWould you like me to explore any specific directory?"

        if "search" in prompt_lower or "find" in prompt_lower or "grep" in prompt_lower:
            return "I can search through the codebase for you. In mock mode, I'm simulating the search functionality.\n\nTo search for patterns in a real session, I would use the `grep_search` tool which supports regex patterns and can filter by file type."

        if "help" in prompt_lower or "what can you do" in prompt_lower:
            return """I'm NASEA's interactive chat assistant. I can help you with:

**File Operations:**
- Read, write, and edit files
- List directory contents
- Create and delete files/directories

**Code Analysis:**
- Search for patterns with `grep_search`
- Understand code structure
- Find function definitions

**Memory:**
- Remember important context with `memory_save`
- Recall saved information with `memory_read`

**Other:**
- Run shell commands
- Search the web for documentation

What would you like me to help you with?"""

        if "hello" in prompt_lower or "hi" in prompt_lower:
            return "Hello! I'm NASEA's chat assistant running in mock mode. I can help you explore and modify your codebase. What would you like to do?"

        # Default response
        return f"[Mock Mode] I received your message: \"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"\n\nIn a real session with an API key configured, I would process this request using the available tools. For now, I'm providing this simulated response.\n\nTry commands like:\n- \"list the files\"\n- \"search for TODO comments\"\n- \"what can you do?\""

    def _generate_response(self, prompt: str, system_message: str) -> str:
        """Choose a response template based on the agent requesting it."""
        system_message = system_message.lower()
        prompt_lower = prompt.lower()

        if "project manager" in system_message or "\"project_info\"" in prompt_lower:
            return self._manager_plan(prompt)

        if "expert software developer" in system_message or "implement the task" in prompt_lower:
            return self._developer_code(prompt)

        if "qa engineer" in system_message or "generate comprehensive pytest test cases" in prompt_lower:
            return self._verifier_tests(prompt)

        if "fix the syntax errors" in prompt_lower:
            return self._syntax_fix(prompt)

        # Generic fall-back: echo a helpful message
        return "This is a mock response used for offline testing."

    def _extract_user_request(self, prompt: str) -> str:
        """Extract the original request line from the manager/developer prompts."""
        match = re.search(r"USER REQUEST:\s*(.+)", prompt, re.IGNORECASE | re.DOTALL)
        if match:
            request = match.group(1).strip()
            # Stop at the next blank line if present
            request = request.split("\n\n")[0].strip()
            return request
        return "Create a simple demonstration application."

    def _manager_plan(self, prompt: str) -> str:
        """Return a simple but valid JSON plan."""
        request = self._extract_user_request(prompt)
        project_name = re.sub(r"[^a-z0-9]+", "_", request.lower())[:20].strip("_") or "demo_app"

        landing_keywords = [
            "landing page", "website", "web page", "homepage", "marketing site", "portfolio"
        ]

        if any(keyword in request.lower() for keyword in landing_keywords):
            plan = {
                "project_info": {
                    "name": project_name or "landing_page",
                    "description": request[:120],
                    "language": "html",
                    "framework": "vanilla"
                },
                "file_structure": [
                    {"path": "index.html", "purpose": "Main landing page markup"},
                    {"path": "styles.css", "purpose": "Responsive styling"},
                    {"path": "script.js", "purpose": "Optional interactivity"},
                    {"path": "README.md", "purpose": "Project overview and instructions"}
                ],
                "tasks": [
                    {
                        "id": "task_markup",
                        "description": "Create the landing page structure with hero, features, and CTA sections.",
                        "file": "index.html",
                        "dependencies": [],
                        "priority": 1
                    },
                    {
                        "id": "task_styles",
                        "description": "Add styling using modern responsive CSS.",
                        "file": "styles.css",
                        "dependencies": ["index.html"],
                        "priority": 2
                    },
                    {
                        "id": "task_interactions",
                        "description": "Implement smooth scrolling and CTA button behaviour in vanilla JavaScript.",
                        "file": "script.js",
                        "dependencies": ["index.html"],
                        "priority": 3
                    },
                    {
                        "id": "task_docs",
                        "description": "Document how to preview the landing page locally.",
                        "file": "README.md",
                        "dependencies": ["index.html"],
                        "priority": 4
                    }
                ]
            }
        else:
            plan = {
                "project_info": {
                    "name": project_name,
                    "description": request[:120],
                    "language": "python",
                    "framework": "standard-library"
                },
                "file_structure": [
                    {"path": "main.py", "purpose": "Application entry point"},
                    {"path": "utils.py", "purpose": "Helper functions"},
                    {"path": "README.md", "purpose": "Project documentation"}
                ],
                "tasks": [
                    {
                        "id": "task_1",
                        "description": request,
                        "file": "main.py",
                        "dependencies": [],
                        "priority": 1
                    },
                    {
                        "id": "task_2",
                        "description": "Add reusable helper utilities.",
                        "file": "utils.py",
                        "dependencies": ["main.py"],
                        "priority": 2
                    },
                    {
                        "id": "task_3",
                        "description": "Document how to run the generated project.",
                        "file": "README.md",
                        "dependencies": ["main.py"],
                        "priority": 3
                    }
                ]
            }

        return json.dumps(plan, indent=2)

    def _developer_code(self, prompt: str) -> str:
        """Generate a minimal Python implementation that matches the task."""
        task_match = re.search(r"TASK:\s*(.+)", prompt, re.IGNORECASE)
        task_description = task_match.group(1).strip() if task_match else "Demonstration task."

        file_match = re.search(r"TARGET FILE:\s*([^\s]+)", prompt, re.IGNORECASE)
        target_file = file_match.group(1) if file_match else "main.py"

        base_docstring = f"{task_description.capitalize()} (mock implementation)."

        if target_file.endswith("utils.py"):
            code = f'''"""
Utility helpers generated in mock mode.
"""

from typing import Iterable, List


def summarize_items(items: Iterable[str]) -> str:
    """
    Return a simple summary string for the provided items.

    Args:
        items: Iterable of strings to summarise.

    Returns:
        Human friendly description of the provided items.
    """
    values: List[str] = [str(item) for item in items]
    if not values:
        return "No items provided."
    return f"Received {{len(values)}} item(s): " + ", ".join(values)
'''
        elif target_file.endswith(".html"):
            code = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Mock Landing Page</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="hero">
    <h1>Launch your product with confidence</h1>
    <p>SaaS-friendly landing page generated in mock mode.</p>
    <button id="cta-button">Get Started</button>
  </header>
  <main>
    <section class="features">
      <article>
        <h2>Fast Setup</h2>
        <p>Deploy instantly with a clean, semantic HTML structure.</p>
      </article>
      <article>
        <h2>Responsive Design</h2>
        <p>Looks great on desktops, tablets, and phones.</p>
      </article>
      <article>
        <h2>Conversion Ready</h2>
        <p>Track CTA interactions with the bundled script.js.</p>
      </article>
    </section>
  </main>
  <footer>
    <small>&copy; {datetime.now().year} Mock Startup. All rights reserved.</small>
  </footer>
  <script src="script.js"></script>
</body>
</html>
"""
        elif target_file.endswith(".css"):
            code = """:root {
  --primary: #2d7ff9;
  --text: #1c1c1e;
  --muted: #6c6c70;
}

body {
  font-family: 'Inter', system-ui, sans-serif;
  margin: 0;
  color: var(--text);
  background: #f7f8ff;
}

.hero {
  text-align: center;
  padding: 6rem 1.5rem 4rem;
  background: linear-gradient(145deg, #eef3ff, #ffffff);
}

.hero h1 {
  font-size: clamp(2.5rem, 5vw, 3.75rem);
  margin-bottom: 0.75rem;
}

.hero p {
  color: var(--muted);
  font-size: 1.1rem;
  margin-bottom: 2rem;
}

#cta-button {
  background: var(--primary);
  color: #fff;
  border: none;
  padding: 0.9rem 2.4rem;
  border-radius: 999px;
  font-size: 1rem;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

#cta-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 30px rgba(45, 127, 249, 0.2);
}

.features {
  display: grid;
  gap: 1.5rem;
  max-width: 960px;
  margin: 0 auto;
  padding: 3rem 1.5rem 4rem;
}

.features article {
  background: #fff;
  border-radius: 1rem;
  padding: 2rem;
  box-shadow: 0 16px 40px rgba(0, 0, 0, 0.06);
}

@media (min-width: 768px) {
  .features {
    grid-template-columns: repeat(3, 1fr);
  }
}
"""
        elif target_file.endswith(".js"):
            code = """document.addEventListener('DOMContentLoaded', () => {
  const ctaButton = document.getElementById('cta-button');
  if (!ctaButton) return;

  ctaButton.addEventListener('click', () => {
    console.log('CTA clicked — mock analytics event fired.');
    ctaButton.textContent = "We'll be in touch!";
    ctaButton.disabled = true;
  });
});
"""
        elif target_file.endswith(".md"):
            code = f"""# Project Documentation (Mock)

This project was generated while running NASEA in **mock mode**.

## Request

> {task_description}

## Previewing

Open `index.html` in your browser or run a lightweight static server:

```bash
python -m http.server 8000
```
"""
        else:
            code = f'''"""
{base_docstring}
"""

from typing import List


def main() -> None:
    """Entry point for the mock implementation."""
    steps: List[str] = [
        "Interpret the user request.",
        "Call helper utilities.",
        "Display a confirmation message."
    ]
    for idx, step in enumerate(steps, start=1):
        print(f"[step {{idx}}] {{step}}")


if __name__ == "__main__":
    main()
'''
        return code

    def _verifier_tests(self, prompt: str) -> str:
        """Return a lightweight pytest suite that always passes."""
        return '''"""
Pytest suite generated in mock mode.
"""

import pytest

from main import main
from utils import summarize_items


def test_main_runs_without_error(capsys):
    main()
    captured = capsys.readouterr()
    assert "[step 3]" in captured.out


@pytest.mark.parametrize("items,expected", [
    ([], "No items provided."),
    (["a"], "Received 1 item(s): a"),
    (["a", "b"], "Received 2 item(s): a, b"),
])
def test_summarize_items(items, expected):
    assert summarize_items(items) == expected
'''

    def _syntax_fix(self, prompt: str) -> str:
        """If asked to fix syntax errors, just echo a simple valid script."""
        return '''def placeholder() -> None:
    """Placeholder fix returned by mock client."""
    pass
'''
