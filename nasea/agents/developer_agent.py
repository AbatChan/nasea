"""
Developer Agent - Implements code based on tasks from Manager Agent.
"""

from typing import List, Dict, Any
from loguru import logger

from nasea.agents.base_agent import BaseAgent
from nasea.core.file_store import FileStore
from nasea.core.utils import detect_language
from nasea.prompts import load_agent_prompt


class DeveloperAgent(BaseAgent):
    """
    Developer Agent: Generates code implementations for assigned tasks.
    """

    def __init__(self, config, memory):
        super().__init__(role="developer", config=config, memory=memory)

    def _default_system_prompt(self) -> str:
        try:
            return load_agent_prompt("developer")
        except FileNotFoundError:
            # Fallback to inline prompt if file not found
            return (
                "You are an Expert Software Developer.\n"
                "Write clean, production-ready code.\n"
                "Always use docstrings and comments.\n"
                "Return ONLY the full code file.\n"
            )

    def implement_tasks(
        self,
        tasks: List[Dict[str, Any]],
        file_store: FileStore,
        original_prompt: str,
        status=None
    ) -> bool:

        self.log(f"Implementing {len(tasks)} tasks...")

        sorted_tasks = sorted(tasks, key=lambda t: t.get("priority", 999))

        for i, task in enumerate(sorted_tasks, 1):
            task_id = task.get("id", f"task_{i}")
            description = task.get("description")
            target_file = task.get("file", "main.py")

            self.log(f"Task {i}/{len(tasks)}: {description}")

            try:
                code = self._generate_code_for_task(task, file_store, original_prompt)

                if not code:
                    self.log(f"Failed to generate code for task {task_id}", level="error")
                    continue

                language = detect_language(target_file)
                existing_file = file_store.files.get(target_file)
                was_placeholder = bool(existing_file and existing_file.purpose == "placeholder")

                if existing_file:
                    file_store.update_file(target_file, code)
                    file_store.files[target_file].purpose = task.get("purpose", "implementation")
                else:
                    file_store.save_file(target_file, code, language=language)

                self.log(f"Implemented: {target_file}")

            except Exception as e:
                self.log(f"Error implementing task {task_id}: {e}", level="error")
                return False

        return True

    def _create_code_skeleton(self, code: str, language: str) -> str:
        if not code or len(code) < 1000:
            return code

        if language == "python":
            skeleton = []
            for line in code.split("\n" ):
                stripped = line.strip()

                if (
                    stripped.startswith("import ")
                    or stripped.startswith("from ")
                    or stripped.startswith("class ")
                    or stripped.startswith("def ")
                    or stripped.startswith("@")
                    or stripped == ""
                    or stripped.startswith('"""')
                    or stripped.startswith("'''")
                ):
                    skeleton.append(line)

            return "\n".join(skeleton) + "\n\n# ... (body removed to save context) ..."
        
        return code[:500] + "\n\n# ... truncated ..."

    def _generate_code_for_task(
        self,
        task: Dict[str, Any],
        file_store: FileStore,
        original_prompt: str
    ) -> str:

        description = task.get("description")
        target_file = task.get("file", "main.py")
        dependencies = task.get("dependencies", [])

        context_segments = [
            f"ORIGINAL REQUEST:\n{original_prompt}",
            f"TASK:\n{description}",
            f"TARGET FILE: {target_file}"
        ]

        for dep in dependencies:
            dep_code = file_store.read_file(dep)
            if dep_code:
                lang = detect_language(dep)
                skeleton = self._create_code_skeleton(dep_code, lang)
                context_segments.append(f"\n--- DEPENDENCY: {dep} ---\n{skeleton}")

        existing = file_store.read_file(target_file)
        if existing and len(existing) > 50:
            context_segments.append(f"\nEXISTING CODE:\n{existing}")

        context = "\n".join(context_segments)

        prompt = (
            f"{context}\n\n"
            "Implement the task above. Return ONLY the full updated code file."
        )

        response = self.query_llm(prompt)
        code = self.extract_code_from_response(response)

        language = detect_language(target_file)
        if not self.validate_code_syntax(code, language):
            fix_prompt = (
                "The following code has syntax errors:\n\n"
                f"{code}\n\n"
                "Fix them and return ONLY corrected code."
            )
            fixed_response = self.query_llm(fix_prompt)
            code = self.extract_code_from_response(fixed_response)

        return code

    def fix_issues(
        self,
        issues: List[Dict[str, Any]],
        file_store: FileStore,
        original_prompt: str,
        status=None
    ) -> bool:

        for issue in issues:
            file_path = issue.get("file")
            description = issue.get("description")
            error_msg = issue.get("error", "")

            current = file_store.read_file(file_path)
            if not current:
                continue

            fixed = self._generate_fix(
                current,
                file_path,
                description,
                error_msg,
                original_prompt
            )

            if fixed and fixed != current:
                file_store.update_file(file_path, fixed)

        return True

    def _generate_fix(
        self,
        current_code: str,
        file_path: str,
        description: str,
        error_msg: str,
        original_prompt: str
    ) -> str:

        prompt = (
            f"ORIGINAL REQUEST:\n{original_prompt}\n\n"
            f"FILE: {file_path}\n\n"
            f"CURRENT CODE:\n{current_code}\n\n"
            f"ISSUE: {description}\n"
            f"ERROR: {error_msg}\n\n"
            "Fix the issue and return ONLY the corrected full file code."
        )

        response = self.query_llm(prompt)
        return self.extract_code_from_response(response)