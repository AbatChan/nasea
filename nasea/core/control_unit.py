"""
Control Unit - Main orchestrator for NASEA.
Coordinates agents, manages workflow, and ensures project completion.
Based on L2MAC architecture.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from loguru import logger

from nasea.core.config import config
from nasea.core.file_store import FileStore
from nasea.core.memory import ContextMemory
from nasea.core.status import StatusReporter


@dataclass
class GenerationResult:
    """Result of a code generation task."""
    success: bool
    project_path: Optional[Path]
    files_generated: int
    tests_passed: int
    tests_total: int
    iterations: int
    error: Optional[str] = None
    warnings: List[str] = None


class ControlUnit:
    """
    Main orchestrator for NASEA.
    Manages the workflow: User Prompt → Manager → Developer → Verifier → Output
    """

    def __init__(self, status_reporter: Optional[StatusReporter] = None):
        """Initialize the control unit."""
        self.config = config
        self.memory = ContextMemory(self.config.database_path)
        self.status = status_reporter or StatusReporter(verbose=True)  # Default to verbose

        # Agents will be initialized lazily
        self._manager_agent = None
        self._developer_agent = None
        self._verifier_agent = None

        logger.info("ControlUnit initialized")
        logger.info(f"Configuration: {self.config}")

    @property
    def manager_agent(self):
        """Lazy load manager agent."""
        if self._manager_agent is None:
            from nasea.agents.manager_agent import ManagerAgent
            self._manager_agent = ManagerAgent(self.config, self.memory)
        return self._manager_agent

    @property
    def developer_agent(self):
        """Lazy load developer agent."""
        if self._developer_agent is None:
            from nasea.agents.developer_agent import DeveloperAgent
            self._developer_agent = DeveloperAgent(self.config, self.memory)
        return self._developer_agent

    @property
    def verifier_agent(self):
        """Lazy load verifier agent."""
        if self._verifier_agent is None:
            from nasea.agents.verifier_agent import VerifierAgent
            self._verifier_agent = VerifierAgent(self.config, self.memory)
        return self._verifier_agent

    def generate_project(
        self,
        user_prompt: str,
        output_dir: Optional[Path] = None,
        project_name: Optional[str] = None
    ) -> GenerationResult:
        """
        Main entry point: Generate a complete project from a user prompt.

        Args:
            user_prompt: Natural language description of what to build
            output_dir: Where to save the project (defaults to config.output_dir)
            project_name: Project name (auto-generated if not provided)

        Returns:
            GenerationResult with project details
        """
        logger.info("=" * 80)
        logger.info("NEW PROJECT GENERATION STARTED")
        logger.info("=" * 80)
        logger.info(f"Prompt: {user_prompt}")

        if hasattr(self.status, 'reset_plan'):
            self.status.reset_plan()
            self.status.board.add_event('Starting new generation', 'cyan')

        # Generate project name if not provided
        if not project_name:
            project_name = self._generate_project_name(user_prompt)

        # Determine output directory
        if output_dir is None:
            output_dir = self.config.output_dir

        # Add timestamp if configured
        if self.config.use_timestamps:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = f"{project_name}_{timestamp}"

        # Create task ID for tracking
        task_id = str(uuid.uuid4())
        self.memory.add_task(task_id, user_prompt, status="in_progress")

        # Log user prompt in memory
        self.memory.add_entry("user", user_prompt, metadata={"task_id": task_id})

        try:
            # Initialize file store
            file_store = FileStore(output_dir, project_name)

            # Step 1: Manager decomposes the prompt into tasks
            if hasattr(self.status, "board"):
                self.status.board.update_status("Planning", index=0, status="in_progress")
            self.status.thinking("Analyzing your request...")
            logger.info("[STEP 1] Manager Agent: Decomposing prompt into tasks")
            tasks = self.manager_agent.decompose_prompt(user_prompt, file_store)

            if not tasks:
                raise ValueError("Manager failed to generate tasks")

            logger.info(f"Generated {len(tasks)} tasks")
            for i, task in enumerate(tasks, 1):
                logger.info(f"  Task {i}: {task.get('description', 'N/A')}")

            if hasattr(self.status, 'board'):
                self.status.board.update_status("Planning", index=0, status="done")
                self.status.board.update_status("Planning", index=1, status="in_progress")
            if hasattr(self.status, 'register_tasks'):
                self.status.register_tasks(tasks)
                if hasattr(self.status, 'board'):
                    self.status.board.update_status("Planning", index=1, status="done")

            self.status.update(
                f"Manager outlined {len(tasks)} task(s)",
                emoji="📋",
                level="info"
            )

            # Step 2: Developer implements each task
            self.status.creating("Creating your project files...")
            logger.info("[STEP 2] Developer Agent: Implementing tasks")
            implementation_success = self.developer_agent.implement_tasks(
                tasks, file_store, user_prompt, status=self.status
            )

            if not implementation_success:
                raise ValueError("Developer failed to implement tasks")

            # Step 3: Verification and refinement loop
            self.status.testing("Running tests and quality checks...")
            logger.info("[STEP 3] Verifier Agent: Testing and refinement")
            iteration = 0
            all_tests_passed = False

            while iteration < self.config.max_iterations and not all_tests_passed:
                iteration += 1
                logger.info(f"Iteration {iteration}/{self.config.max_iterations}")

                # Run verification
                verification_result = self.verifier_agent.verify_project(
                    file_store, user_prompt
                )

                tests_passed = verification_result.get("tests_passed", 0)
                tests_total = verification_result.get("tests_total", 0)
                issues = verification_result.get("issues", [])

                if hasattr(self.status, 'board'):
                    self.status.board.update_status("Verification", index=0, status="in_progress")

                logger.info(f"Tests: {tests_passed}/{tests_total} passed")

                if tests_passed == tests_total and not issues:
                    all_tests_passed = True
                    logger.info("All tests passed")
                    if hasattr(self.status, 'board'):
                        self.status.board.update_status("Verification", index=0, status="done")
                    break

                if iteration >= self.config.max_iterations:
                    logger.warning(f"Maximum iterations reached: {self.config.max_iterations}")
                    break

                # Step 4: Developer fixes issues
                self.status.fixing(f"Fixing {len(issues)} issue(s)...")
                logger.info(f"[STEP 4] Developer Agent fixing {len(issues)} issues")
                fix_success = self.developer_agent.fix_issues(
                    issues, file_store, user_prompt, status=self.status
                )

                if not fix_success:
                    logger.warning("Developer failed to fix issues")
                    break

            if hasattr(self.status, 'board') and not all_tests_passed:
                self.status.board.update_status("Verification", index=0, status="failed")

            # Step 5: Finalize project
            if hasattr(self.status, 'board'):
                self.status.board.update_status("Finalization", index=0, status="in_progress")
            self.status.update("Almost done! Finalizing project...", emoji="✨", level="info")
            logger.info("[STEP 5] Finalizing project")
            self._finalize_project(
                file_store,
                user_prompt,
                tests_passed=(tests_passed == tests_total)
            )
            if hasattr(self.status, 'board'):
                self.status.board.update_status("Finalization", index=0, status="done")

            # Update task status
            self.memory.update_task(
                task_id,
                status="completed" if all_tests_passed else "completed_with_warnings",
                result=f"Generated {len(file_store.files)} files"
            )

            # Create result
            result = GenerationResult(
                success=True,
                project_path=file_store.project_path,
                files_generated=len(file_store.files),
                tests_passed=tests_passed,
                tests_total=tests_total,
                iterations=iteration,
                warnings=[] if all_tests_passed else ["Some tests failed"]
            )

            logger.info("\n" + "=" * 80)
            logger.info("PROJECT GENERATION COMPLETED")
            logger.info("=" * 80)
            logger.info(f"Project: {file_store.project_path}")
            logger.info(f"Files: {result.files_generated}")
            logger.info(f"Tests: {result.tests_passed}/{result.tests_total}")
            logger.info(f"Iterations: {result.iterations}")

            self.status.success(f"Project ready! Generated {result.files_generated} file(s)")

            return result

        except Exception as e:
            logger.error(f"Project generation failed: {e}", exc_info=True)
            self.status.error(f"Generation failed: {str(e)}")

            # Update task with error
            self.memory.update_task(task_id, status="failed", error=str(e))

            return GenerationResult(
                success=False,
                project_path=None,
                files_generated=0,
                tests_passed=0,
                tests_total=0,
                iterations=0,
                error=str(e)
            )

    def _generate_project_name(self, prompt: str) -> str:
        """
        Generate a project name from the prompt.

        Args:
            prompt: User prompt

        Returns:
            Sanitized project name
        """
        # Extract first few words
        words = prompt.lower().split()[:5]

        # Filter out common words
        stop_words = {"a", "an", "the", "for", "with", "that", "this", "create", "build", "make"}
        meaningful_words = [w for w in words if w not in stop_words]

        # Join with underscores
        name = "_".join(meaningful_words[:3])

        # Sanitize
        name = "".join(c for c in name if c.isalnum() or c == "_")

        return name or "generated_project"

    def _finalize_project(
        self,
        file_store: FileStore,
        original_prompt: str,
        tests_passed: bool
    ):
        """
        Finalize the project by adding metadata and documentation.

        Args:
            file_store: FileStore instance
            original_prompt: Original user prompt
            tests_passed: Whether all tests passed
        """
        # Save manifest
        file_store.save_manifest(
            description=original_prompt,
            entry_point=self._detect_entry_point(file_store),
            dependencies=self._extract_dependencies(file_store),
            tests_passing=tests_passed
        )

        # Generate README if not exists
        if not file_store.read_file("README.md"):
            readme_content = self._generate_readme(file_store, original_prompt)
            file_store.save_file("README.md", readme_content, language="markdown", purpose="documentation")

        logger.info("Project finalized with manifest and README")

    def _detect_entry_point(self, file_store: FileStore) -> Optional[str]:
        """Detect the main entry point file."""
        # Common entry point names
        entry_points = [
            "main.py", "app.py", "server.py", "index.py",
            "cli.py", "__main__.py", "run.py"
        ]

        for entry in entry_points:
            if entry in file_store.files:
                return entry

        # Return first .py file as fallback
        py_files = [f for f in file_store.files if f.endswith(".py")]
        return py_files[0] if py_files else None

    def _extract_dependencies(self, file_store: FileStore) -> List[str]:
        """Extract project dependencies."""
        # Check for requirements.txt
        requirements = file_store.read_file("requirements.txt")
        if requirements:
            return [line.strip() for line in requirements.split("\n") if line.strip() and not line.startswith("#")]

        # Check for package.json (JavaScript)
        package_json = file_store.read_file("package.json")
        if package_json:
            import json
            try:
                data = json.loads(package_json)
                return list(data.get("dependencies", {}).keys())
            except json.JSONDecodeError:
                pass

        return []

    def _generate_readme(self, file_store: FileStore, prompt: str) -> str:
        """Generate a basic README for the project."""
        stats = file_store.get_stats()

        readme = f"""# {file_store.project_name}

**Generated by NASEA** - Natural-Language Autonomous Software-Engineering Agent

## Description

{prompt}

## Project Structure

- **Files**: {stats['total_files']}
- **Languages**: {', '.join(stats['by_language'].keys())}
- **Total Size**: {stats['total_size']} characters

## Setup

### Prerequisites

Install dependencies:

```bash
# Python projects
pip install -r requirements.txt

# JavaScript projects
npm install
```

### Running

```bash
# Check the entry point file and run accordingly
python main.py  # or appropriate entry point
```

## Files

"""
        # List all files
        for file_path in sorted(file_store.files.keys()):
            metadata = file_store.files[file_path]
            readme += f"- `{file_path}` - {metadata.purpose}\n"

        readme += f"""
## Generated

- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **By**: NASEA v0.1.0-alpha
- **From prompt**: "{prompt[:100]}{'...' if len(prompt) > 100 else ''}"
"""

        return readme

    def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a generation task.

        Args:
            task_id: Task identifier

        Returns:
            Task status information
        """
        return self.memory.get_task_status(task_id)

    def cleanup(self):
        """Cleanup resources."""
        self.memory.close()
        logger.info("ControlUnit cleaned up")

    def __repr__(self) -> str:
        """String representation."""
        return f"ControlUnit(model={self.config.default_model})"
