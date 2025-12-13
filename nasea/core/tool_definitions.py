"""
Tool Definitions for LLM Function Calling
Defines the tools/functions that the LLM can call during project generation.
"""

from typing import List, Dict, Any

# Tool definitions in OpenAI function calling format
GENERATION_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. You can call multiple tools in parallel - speculatively read multiple files at once if useful. For large files, use offset and limit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read (relative to project root)"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (0-indexed). Use with limit for large files."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read. Use with offset for large files."
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a new file or overwrite an existing file. If file exists, you MUST read_file first. Prefer edit_file for modifications.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path where the file should be created/written"
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete content to write to the file"
                    }
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file by replacing exact content. You MUST read_file first before editing. old_content must match exactly including whitespace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to edit"
                    },
                    "old_content": {
                        "type": "string",
                        "description": "The exact content to find and replace (must match exactly)"
                    },
                    "new_content": {
                        "type": "string",
                        "description": "The new content to insert in place of old_content"
                    }
                },
                "required": ["file_path", "old_content", "new_content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files in a directory. Use this to see what files exist before creating or editing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path to list files from (defaults to project root if not specified)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command (git, npm, python, pip, node, etc). Only for actual CLI tools. Cannot interact with browsers, GUIs, or see visual output. For file ops use read_file/write_file/edit_file instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The actual shell command to execute (must be a real CLI command)"
                    },
                    "working_directory": {
                        "type": "string",
                        "description": "Directory to run the command in (defaults to project root)"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "Create a new directory. Use this to organize files into folders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory_path": {
                        "type": "string",
                        "description": "Path of the directory to create"
                    }
                },
                "required": ["directory_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_path",
            "description": "Delete a file or directory. Use this to remove obsolete assets or folders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file or directory to delete"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Set true to delete non-empty directories"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rename_path",
            "description": "Rename or move a file/directory. Use this to resolve naming issues or reorganize files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_path": {
                        "type": "string",
                        "description": "Existing path to rename or move"
                    },
                    "target_path": {
                        "type": "string",
                        "description": "New path/name destination"
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Allow replacing an existing target"
                    }
                },
                "required": ["source_path", "target_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "complete_generation",
            "description": "Mark task as complete. IMPORTANT: Before calling this, you MUST first output a summary message to the user explaining what you did. Then call this tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of what was created"
                    },
                    "next_steps": {
                        "type": "string",
                        "description": "Instructions for the user on what to do next (e.g., 'Run npm install', 'Open index.html in browser')"
                    }
                },
                "required": ["summary"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for current documentation, news, or technical solutions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (e.g., 'python 3.12 release notes')"
                    }
                },
                "required": ["query"]
            }
        }
    },
    # === NEW TOOLS (Adapted from OpenCoder) ===
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Fast regex search across files. ALWAYS use this for searching - never use grep/rg via run_command. Supports regex patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for (e.g., 'def .*async', 'import.*flask')"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in (defaults to project root)"
                    },
                    "file_type": {
                        "type": "string",
                        "description": "Filter by file extension (e.g., 'py', 'js', 'ts')"
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether search is case-sensitive (default: false)"
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Document your reasoning before taking action. Use this to plan complex operations, analyze problems, or organize your approach. The thought is saved to memory for context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your reasoning, analysis, or plan"
                    },
                    "plan": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of planned steps"
                    }
                },
                "required": ["thought"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_save",
            "description": "Save important context or information to persistent memory for later reference. Use this to remember decisions, discoveries, or important facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Unique identifier for this memory (e.g., 'project_structure', 'user_preferences')"
                    },
                    "value": {
                        "type": "string",
                        "description": "The information to remember"
                    },
                    "category": {
                        "type": "string",
                        "description": "Category for organization (e.g., 'decision', 'discovery', 'preference')"
                    }
                },
                "required": ["key", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": "Retrieve saved information from persistent memory. Use this to recall previous decisions, context, or stored facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Specific key to retrieve (optional - omit to list all)"
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category (optional)"
                    },
                    "search": {
                        "type": "string",
                        "description": "Search term to find in memory values (optional)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_syntax",
            "description": "Check a file for syntax errors. Supports Python, JavaScript, JSON, HTML, CSS, YAML, Shell scripts, and XML. Use this to verify code is valid before or after edits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to check for syntax errors"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_browser",
            "description": "Open a file or URL in browser AND capture console output. Use this to preview HTML files and check for JavaScript errors. Returns console.log output, errors, and warnings so you can debug issues.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path (e.g., 'index.html') or URL (e.g., 'http://localhost:8000') to open"
                    }
                },
                "required": ["path"]
            }
        }
    }
]


# Read-only tools for VIEW/EXPLAIN modes (no file modifications)
READ_ONLY_TOOLS: List[Dict[str, Any]] = [
    tool for tool in GENERATION_TOOLS
    if tool["function"]["name"] in (
        "read_file", "list_files", "grep_search", "web_search",
        "think", "memory_read", "memory_save", "check_syntax"
    )
]

# Edit/debug tools - excludes 'think' which causes issues with some models
EDIT_TOOLS: List[Dict[str, Any]] = [
    tool for tool in GENERATION_TOOLS
    if tool["function"]["name"] not in ("think", "memory_save", "memory_read")
]

# Tools safe for parallel execution (no side effects or independent operations)
PARALLEL_SAFE_TOOLS: set = {
    "read_file",      # Reading files is always safe
    "list_files",     # Listing is safe
    "grep_search",    # Search is safe
    "web_search",     # Web searches are independent
    "check_syntax",   # Syntax checking is safe
    "think",          # Thinking doesn't modify anything
    "memory_read",    # Reading memory is safe
}

# Tools that must run sequentially (have side effects or dependencies)
SEQUENTIAL_ONLY_TOOLS: set = {
    "write_file",         # File writes may conflict
    "edit_file",          # Edits may conflict
    "create_directory",   # Directory creation order matters
    "delete_path",        # Deletions can affect other operations
    "rename_path",        # Renames affect file paths
    "run_command",        # Commands may have side effects
    "open_browser",       # Browser state
    "memory_save",        # Memory writes should be ordered
    "complete_generation", # Completion signal
}


def get_tool_by_name(name: str) -> Dict[str, Any] | None:
    """Get tool definition by function name."""
    for tool in GENERATION_TOOLS:
        if tool["function"]["name"] == name:
            return tool
    return None


def get_all_tool_names() -> List[str]:
    """Get list of all available tool names."""
    return [tool["function"]["name"] for tool in GENERATION_TOOLS]
