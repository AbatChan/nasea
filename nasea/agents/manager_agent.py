"""
Manager Agent - Decomposes prompts into tasks and creates project architecture.
"""

import json
import subprocess
import shlex
from typing import List, Dict, Any
from loguru import logger

from nasea.agents.base_agent import BaseAgent
from nasea.core.file_store import FileStore
from nasea.core.utils import detect_language
from nasea.prompts import load_agent_prompt


class ManagerAgent(BaseAgent):
    """
    Manager Agent: Breaks down user prompts into actionable tasks.
    Creates project structure and defines implementation plan.
    """

    def __init__(self, config, memory):
        super().__init__(role="manager", config=config, memory=memory)

    def _default_system_prompt(self) -> str:
        try:
            return load_agent_prompt("manager")
        except FileNotFoundError:
            # Fallback to inline prompt if file not found
            return (
                "You are a Senior Software Architect and Project Manager.\n"
                "Responsibilities:\n"
                "1. Understand user requirements\n"
                "2. Design project structure\n"
                "3. Break work into tasks\n"
                "4. Define dependencies\n"
                "5. Output JSON in the required format\n\n"
                "JSON Format:\n"
                "{\n"
                '  "project_info": {"name": "", "description": "", "language": "", "framework": ""},\n'
                '  "initialization_command": "optional scaffold command",\n'
                '  "file_structure": [{"path": "", "purpose": ""}],\n'
                '  "tasks": [{"id": "", "description": "", "file": "", "dependencies": [], "priority": 1}]\n'
                "}\n"
            )

    def decompose_prompt(self, user_prompt: str, file_store: FileStore) -> List[Dict[str, Any]]:
        self.log("Analyzing user prompt and creating project plan...")

        decomposition_prompt = (
            "Analyze this user request and create a detailed implementation plan.\n\n"
            f"USER REQUEST:\n{user_prompt}\n\n"
            "Create a JSON plan including:\n"
            "- project_info\n"
            "- initialization_command (if applicable)\n"
            "- file_structure\n"
            "- tasks\n\n"
            "Respond ONLY with valid JSON."
        )

        try:
            response = self.query_llm(decomposition_prompt)
            plan = self._parse_plan(response)

            if not plan or "tasks" not in plan:
                raise ValueError("Invalid plan structure")

            self.memory.add_entry(
                role=self.role,
                content=json.dumps(plan, indent=2),
                metadata={"type": "plan"}
            )

            self._initialize_file_structure(plan, file_store)

            tasks = plan["tasks"]
            self.log(f"Created {len(tasks)} tasks")
            return tasks

        except Exception as e:
            self.log(f"Error decomposing prompt: {e}", level="error")
            return self._create_fallback_plan(user_prompt)

    def _parse_plan(self, response: str) -> Dict[str, Any]:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            response = response.replace("'", '"')
            return json.loads(response.strip())

    def _initialize_file_structure(self, plan: Dict[str, Any], file_store: FileStore):
        init_cmd = plan.get("initialization_command")
        if init_cmd:
            self.log(f"Running scaffolding: {init_cmd}")
            try:
                subprocess.run(
                    init_cmd,
                    shell=True,
                    cwd=str(file_store.project_path),
                    timeout=120,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self.log("Scaffolding completed.")
            except Exception as e:
                self.log(f"Scaffolding failed: {e}", level="error")

        file_structure = plan.get("file_structure", [])

        for item in file_structure:
            path = item.get("path")
            purpose = item.get("purpose", "")

            if not path:
                continue

            if (file_store.project_path / path).exists():
                continue

            if path.endswith(".py"):
                content = f'"""\n{purpose}\n"""\n\n# TODO: Implement\n'
            elif path.endswith(".js"):
                content = f"/** {purpose} */\n\n// TODO\n"
            else:
                content = f"# {purpose}\n"

            file_store.save_file(
                path,
                content,
                language=detect_language(path),
                purpose="placeholder"
            )

        self.log(f"Initialized {len(file_structure)} files")

    def _create_fallback_plan(self, prompt: str) -> List[Dict[str, Any]]:
        self.log("Using fallback plan...", level="warning")

        lower = prompt.lower()

        landing_keywords = ["landing page", "homepage", "website", "web page"]

        if any(k in lower for k in landing_keywords):
            return [
                {
                    "id": "html",
                    "description": "Create landing page HTML",
                    "file": "index.html",
                    "dependencies": [],
                    "priority": 1
                },
                {
                    "id": "css",
                    "description": "Add responsive styling",
                    "file": "styles.css",
                    "dependencies": ["index.html"],
                    "priority": 2
                },
                {
                    "id": "js",
                    "description": "Add JS interactivity",
                    "file": "script.js",
                    "dependencies": ["index.html"],
                    "priority": 3
                },
                {
                    "id": "readme",
                    "description": "Document project",
                    "file": "README.md",
                    "dependencies": [],
                    "priority": 4
                }
            ]

        return [
            {
                "id": "single",
                "description": prompt,
                "file": "main.py",
                "dependencies": [],
                "priority": 1
            }
        ]