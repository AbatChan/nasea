"""
Tests for the new OpenCoder-inspired tools: grep_search, think, memory_save, memory_read
"""

import pytest
import tempfile
import json
from pathlib import Path
from nasea.core.tool_executors import ToolExecutor


@pytest.fixture
def temp_project():
    """Create a temporary project directory with sample files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create sample Python files
        (root / "main.py").write_text("""
import os
from utils import helper_function

def main():
    print("Hello, World!")
    result = helper_function(42)
    return result

if __name__ == "__main__":
    main()
""")

        (root / "utils.py").write_text("""
def helper_function(x):
    return x * 2

def another_function():
    pass

class MyClass:
    def __init__(self):
        self.value = 0
""")

        # Create a subdirectory with more files
        (root / "src").mkdir()
        (root / "src" / "api.py").write_text("""
from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello from API"

@app.route("/users")
def get_users():
    return []
""")

        yield root


class TestGrepSearch:
    """Tests for grep_search tool."""

    def test_basic_search(self, temp_project):
        executor = ToolExecutor(temp_project)
        result = executor.execute("grep_search", {"pattern": "def main"})

        assert result["success"] is True
        assert result["count"] >= 1
        assert any("main.py" in m for m in result["matches"])

    def test_regex_search(self, temp_project):
        executor = ToolExecutor(temp_project)
        result = executor.execute("grep_search", {"pattern": "def .*function"})

        assert result["success"] is True
        assert result["count"] >= 2  # helper_function and another_function

    def test_case_insensitive(self, temp_project):
        executor = ToolExecutor(temp_project)
        result = executor.execute("grep_search", {
            "pattern": "HELLO",
            "case_sensitive": False
        })

        assert result["success"] is True
        assert result["count"] >= 1

    def test_case_sensitive(self, temp_project):
        executor = ToolExecutor(temp_project)
        result = executor.execute("grep_search", {
            "pattern": "HELLO",  # Won't match "Hello"
            "case_sensitive": True
        })

        assert result["success"] is True
        assert result["count"] == 0

    def test_no_matches(self, temp_project):
        executor = ToolExecutor(temp_project)
        result = executor.execute("grep_search", {"pattern": "nonexistent_xyz_123"})

        assert result["success"] is True
        assert result["count"] == 0
        assert result["matches"] == []

    def test_missing_pattern(self, temp_project):
        executor = ToolExecutor(temp_project)
        result = executor.execute("grep_search", {})

        assert result["success"] is False
        assert "pattern" in result["error"].lower()

    def test_invalid_regex(self, temp_project):
        executor = ToolExecutor(temp_project)
        result = executor.execute("grep_search", {"pattern": "[invalid("})

        # ripgrep may handle some invalid patterns gracefully, so just check it doesn't crash
        assert "success" in result


class TestThink:
    """Tests for think tool."""

    def test_basic_thought(self, temp_project):
        executor = ToolExecutor(temp_project)
        result = executor.execute("think", {
            "thought": "I need to analyze the codebase structure first"
        })

        assert result["success"] is True
        assert "Recorded thought" in result["message"]
        assert result["thought"] == "I need to analyze the codebase structure first"

    def test_thought_with_plan(self, temp_project):
        executor = ToolExecutor(temp_project)
        result = executor.execute("think", {
            "thought": "Implementing the feature",
            "plan": [
                "Read existing code",
                "Identify modification points",
                "Make changes",
                "Test"
            ]
        })

        assert result["success"] is True
        assert len(result["plan"]) == 4
        assert "4" in result["message"]  # Should mention 4 plan steps

    def test_thought_persisted(self, temp_project):
        executor = ToolExecutor(temp_project)
        executor.execute("think", {"thought": "First thought"})
        executor.execute("think", {"thought": "Second thought"})

        # Check memory file
        memory_file = temp_project / ".nasea_memory.json"
        assert memory_file.exists()

        data = json.loads(memory_file.read_text())
        assert len(data["thoughts"]) == 2

    def test_missing_thought(self, temp_project):
        executor = ToolExecutor(temp_project)
        result = executor.execute("think", {})

        assert result["success"] is False
        assert "thought" in result["error"].lower()


class TestMemorySave:
    """Tests for memory_save tool."""

    def test_save_memory(self, temp_project):
        executor = ToolExecutor(temp_project)
        result = executor.execute("memory_save", {
            "key": "project_type",
            "value": "Flask REST API"
        })

        assert result["success"] is True
        assert result["key"] == "project_type"

    def test_save_with_category(self, temp_project):
        executor = ToolExecutor(temp_project)
        result = executor.execute("memory_save", {
            "key": "auth_method",
            "value": "JWT tokens",
            "category": "decision"
        })

        assert result["success"] is True
        assert result["category"] == "decision"

    def test_memory_persisted(self, temp_project):
        executor = ToolExecutor(temp_project)
        executor.execute("memory_save", {
            "key": "test_key",
            "value": "test_value"
        })

        # Check memory file
        memory_file = temp_project / ".nasea_memory.json"
        assert memory_file.exists()

        data = json.loads(memory_file.read_text())
        assert "test_key" in data["entries"]
        assert data["entries"]["test_key"]["value"] == "test_value"

    def test_overwrite_memory(self, temp_project):
        executor = ToolExecutor(temp_project)
        executor.execute("memory_save", {"key": "mykey", "value": "old_value"})
        executor.execute("memory_save", {"key": "mykey", "value": "new_value"})

        memory_file = temp_project / ".nasea_memory.json"
        data = json.loads(memory_file.read_text())
        assert data["entries"]["mykey"]["value"] == "new_value"

    def test_missing_key(self, temp_project):
        executor = ToolExecutor(temp_project)
        result = executor.execute("memory_save", {"value": "test"})

        assert result["success"] is False

    def test_missing_value(self, temp_project):
        executor = ToolExecutor(temp_project)
        result = executor.execute("memory_save", {"key": "test"})

        assert result["success"] is False


class TestMemoryRead:
    """Tests for memory_read tool."""

    def test_read_specific_key(self, temp_project):
        executor = ToolExecutor(temp_project)
        executor.execute("memory_save", {"key": "mykey", "value": "myvalue"})

        result = executor.execute("memory_read", {"key": "mykey"})

        assert result["success"] is True
        assert result["value"] == "myvalue"

    def test_read_nonexistent_key(self, temp_project):
        executor = ToolExecutor(temp_project)
        result = executor.execute("memory_read", {"key": "nonexistent"})

        assert result["success"] is False
        assert "no memory found" in result["error"].lower()

    def test_list_all_memories(self, temp_project):
        executor = ToolExecutor(temp_project)
        executor.execute("memory_save", {"key": "key1", "value": "value1"})
        executor.execute("memory_save", {"key": "key2", "value": "value2"})

        result = executor.execute("memory_read", {})

        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["results"]) == 2

    def test_filter_by_category(self, temp_project):
        executor = ToolExecutor(temp_project)
        executor.execute("memory_save", {"key": "a", "value": "v1", "category": "decision"})
        executor.execute("memory_save", {"key": "b", "value": "v2", "category": "discovery"})
        executor.execute("memory_save", {"key": "c", "value": "v3", "category": "decision"})

        result = executor.execute("memory_read", {"category": "decision"})

        assert result["success"] is True
        assert result["count"] == 2

    def test_search_memories(self, temp_project):
        executor = ToolExecutor(temp_project)
        executor.execute("memory_save", {"key": "a", "value": "Flask API with authentication"})
        executor.execute("memory_save", {"key": "b", "value": "React frontend"})

        result = executor.execute("memory_read", {"search": "Flask"})

        assert result["success"] is True
        assert result["count"] == 1

    def test_empty_memory(self, temp_project):
        executor = ToolExecutor(temp_project)
        result = executor.execute("memory_read", {})

        assert result["success"] is True
        assert result["count"] == 0
        assert "empty" in result["message"].lower()


class TestToolIntegration:
    """Integration tests combining multiple tools."""

    def test_think_then_search(self, temp_project):
        executor = ToolExecutor(temp_project)

        # Think about what to do
        think_result = executor.execute("think", {
            "thought": "Need to find all Flask routes in the codebase",
            "plan": ["Search for @app.route decorators", "Analyze results"]
        })
        assert think_result["success"] is True

        # Actually search
        search_result = executor.execute("grep_search", {"pattern": "@app.route"})
        assert search_result["success"] is True
        assert search_result["count"] >= 2  # We have 2 routes

    def test_search_and_save_to_memory(self, temp_project):
        executor = ToolExecutor(temp_project)

        # Search for classes
        search_result = executor.execute("grep_search", {"pattern": "class .*:"})
        assert search_result["success"] is True

        # Save findings to memory
        save_result = executor.execute("memory_save", {
            "key": "classes_found",
            "value": f"Found {search_result['count']} class definitions",
            "category": "discovery"
        })
        assert save_result["success"] is True

        # Retrieve later
        read_result = executor.execute("memory_read", {"key": "classes_found"})
        assert read_result["success"] is True
        assert "class" in read_result["value"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
