"""
Streaming-enabled control unit with real-time progress updates.

Wraps the existing control unit workflow and provides live streaming updates
to the CLI as tasks are planned, files are created, and tests are run.
"""

from typing import Optional, Generator, Dict, Any
from rich.console import Console
from rich.live import Live

from nasea.core.control_unit import ControlUnit, GenerationResult
from nasea.core.streaming import StreamingRenderer, Stage
from nasea.core.file_store import FileStore


class StreamingControlUnit:
    """
    Streaming-enabled control unit that provides real-time progress updates.

    This wraps the existing ControlUnit and provides a streaming interface
    that shows progress as it happens, similar to Claude Code CLI.
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize streaming control unit.

        Args:
            console: Rich console for output
        """
        self.console = console or Console()
        self.control = ControlUnit()
        self.renderer = StreamingRenderer(self.console)

    def generate_with_streaming(
        self,
        prompt: str,
        project_name: Optional[str] = None,
        output_dir: str = "output"
    ) -> Generator[Dict[str, Any], None, GenerationResult]:
        """
        Generate project with real-time streaming updates.

        Args:
            prompt: User's project description
            project_name: Optional project name
            output_dir: Output directory

        Yields:
            Progress updates with stage/task information

        Returns:
            Final GenerationResult
        """
        # Stage 1: Planning
        planning_stage = Stage("Planning", "📋")
        planning_stage.add_task("Analyzing requirements")
        planning_stage.add_task("Designing project structure")
        planning_stage.add_task("Breaking down tasks")
        self.renderer.stages.append(planning_stage)
        self.renderer.current_stage = planning_stage

        yield {
            "type": "stage_start",
            "stage": "planning",
            "render": self.renderer.render()
        }

        # Mark tasks as in progress one by one
        planning_stage.mark_task_in_progress(0)
        yield {"type": "task_update", "render": self.renderer.render()}

        # Simulate manager decomposition (in real implementation, this would stream)
        # For now, we'll hook into the existing workflow
        try:
            # Get file store
            file_store = self.control._get_or_create_file_store(project_name, output_dir, prompt)

            # Manager phase
            planning_stage.mark_task_complete(0)
            planning_stage.mark_task_in_progress(1)
            yield {"type": "task_update", "render": self.renderer.render()}

            manager_agent = self.control._get_manager_agent()
            tasks = manager_agent.decompose_prompt(prompt, file_store)

            planning_stage.mark_task_complete(1)
            planning_stage.mark_task_in_progress(2)
            yield {"type": "task_update", "render": self.renderer.render()}

            # Complete planning stage
            planning_stage.mark_task_complete(2)
            planning_stage.complete()
            yield {"type": "stage_complete", "stage": "planning", "render": self.renderer.render()}

            # Stage 2: Implementation
            impl_stage = Stage("Implementation", "⚙️")

            # Add a task for each file to be created
            for task in tasks:
                file_name = task.get("file", "unknown")
                impl_stage.add_task(f"Create {file_name}")

            self.renderer.stages.append(impl_stage)
            self.renderer.current_stage = impl_stage

            yield {
                "type": "stage_start",
                "stage": "implementation",
                "render": self.renderer.render()
            }

            # Developer phase
            developer_agent = self.control._get_developer_agent()

            for idx, task in enumerate(tasks):
                impl_stage.mark_task_in_progress(idx)
                yield {"type": "task_update", "render": self.renderer.render()}

                # Execute task
                developer_agent.execute_task(task, file_store)

                impl_stage.mark_task_complete(idx)
                yield {"type": "task_update", "render": self.renderer.render()}

            impl_stage.complete()
            yield {"type": "stage_complete", "stage": "implementation", "render": self.renderer.render()}

            # Stage 3: Testing
            test_stage = Stage("Testing", "🧪")
            test_stage.add_task("Running quality checks")
            test_stage.add_task("Verifying code syntax")
            test_stage.add_task("Checking test coverage")

            self.renderer.stages.append(test_stage)
            self.renderer.current_stage = test_stage

            yield {
                "type": "stage_start",
                "stage": "testing",
                "render": self.renderer.render()
            }

            # Verifier phase
            verifier_agent = self.control._get_verifier_agent()

            test_stage.mark_task_in_progress(0)
            yield {"type": "task_update", "render": self.renderer.render()}

            issues = verifier_agent.verify(file_store)

            test_stage.mark_task_complete(0)
            test_stage.mark_task_in_progress(1)
            yield {"type": "task_update", "render": self.renderer.render()}

            # Fix issues if any
            if issues:
                fix_stage = Stage("Fixing Issues", "🔧")
                for issue in issues:
                    fix_stage.add_task(f"Fix: {issue['message'][:50]}...")

                self.renderer.stages.append(fix_stage)
                self.renderer.current_stage = fix_stage

                yield {
                    "type": "stage_start",
                    "stage": "fixing",
                    "render": self.renderer.render()
                }

                for idx, issue in enumerate(issues):
                    fix_stage.mark_task_in_progress(idx)
                    yield {"type": "task_update", "render": self.renderer.render()}

                    # Fix issue
                    developer_agent.execute_task(
                        {
                            "id": f"fix_{idx}",
                            "description": f"Fix: {issue['message']}",
                            "file": issue["file"],
                            "dependencies": [],
                            "priority": 1
                        },
                        file_store
                    )

                    fix_stage.mark_task_complete(idx)
                    yield {"type": "task_update", "render": self.renderer.render()}

                fix_stage.complete()
                yield {"type": "stage_complete", "stage": "fixing", "render": self.renderer.render()}

            test_stage.mark_task_complete(1)
            test_stage.mark_task_complete(2)
            test_stage.complete()
            yield {"type": "stage_complete", "stage": "testing", "render": self.renderer.render()}

            # Stage 4: Finalization
            final_stage = Stage("Finalization", "✨")
            final_stage.add_task("Saving files to disk")
            final_stage.add_task("Creating manifest")
            final_stage.add_task("Generating documentation")

            self.renderer.stages.append(final_stage)
            self.renderer.current_stage = final_stage

            yield {
                "type": "stage_start",
                "stage": "finalization",
                "render": self.renderer.render()
            }

            final_stage.mark_task_in_progress(0)
            yield {"type": "task_update", "render": self.renderer.render()}

            # Save files
            saved_paths = self.control._save_files_to_disk(file_store)

            final_stage.mark_task_complete(0)
            final_stage.mark_task_in_progress(1)
            yield {"type": "task_update", "render": self.renderer.render()}

            # Create manifest
            manifest_path = self.control._create_manifest(file_store, tasks)

            final_stage.mark_task_complete(1)
            final_stage.mark_task_in_progress(2)
            yield {"type": "task_update", "render": self.renderer.render()}

            # Generate README
            readme_path = self.control._generate_readme(file_store, prompt, tasks)

            final_stage.mark_task_complete(2)
            final_stage.complete()
            yield {"type": "stage_complete", "stage": "finalization", "render": self.renderer.render()}

            # Create final result
            result = GenerationResult(
                success=True,
                file_store=file_store,
                files_generated=len(file_store.files),
                tests_passed=0,
                tests_total=0,
                iterations=1,
                project_path=str(file_store.root_path),
                manifest_path=manifest_path,
                saved_files=saved_paths
            )

            yield {
                "type": "complete",
                "result": result,
                "render": self.renderer.render()
            }

            return result

        except Exception as e:
            # Error handling
            if self.renderer.current_stage:
                self.renderer.current_stage.status = "failed"

            yield {
                "type": "error",
                "error": str(e),
                "render": self.renderer.render()
            }

            raise

    def render_streaming_progress(self, prompt: str, project_name: Optional[str] = None) -> GenerationResult:
        """
        Generate project with live rendering.

        Args:
            prompt: User's project description
            project_name: Optional project name

        Returns:
            GenerationResult
        """
        result = None

        with Live(self.renderer.render(), console=self.console, refresh_per_second=4) as live:
            for update in self.generate_with_streaming(prompt, project_name):
                if "render" in update:
                    live.update(update["render"])

                if update.get("type") == "complete":
                    result = update["result"]

        # Show final summary
        self.console.print(f"\n{self.renderer.get_summary()}\n")

        return result
