"""
Parallel Agent Executor - Run multiple agents concurrently.

Inspired by Claude Code's approach of launching 2-3 agents with different
perspectives for exploration and architecture tasks.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, TypeVar
from loguru import logger
import time

T = TypeVar('T')


@dataclass
class AgentTask:
    """A task to be executed by an agent."""
    name: str
    agent_type: str  # e.g., "explore", "architect"
    prompt: str
    context: Dict[str, Any] = field(default_factory=dict)
    model: Optional[str] = None  # Override model for this task
    priority: int = 0  # Higher = more important


@dataclass
class AgentResult:
    """Result from a parallel agent execution."""
    task_name: str
    agent_type: str
    success: bool
    result: Any
    error: Optional[str] = None
    duration_ms: int = 0


class ParallelExecutor:
    """
    Execute multiple agents in parallel.

    Usage:
        executor = ParallelExecutor(max_workers=3)

        tasks = [
            AgentTask("explore_api", "explore", "Find all API endpoints"),
            AgentTask("explore_models", "explore", "Find all data models"),
            AgentTask("explore_tests", "explore", "Find test patterns"),
        ]

        results = executor.run_parallel(tasks, agent_factory)
    """

    def __init__(self, max_workers: int = 3):
        """
        Initialize parallel executor.

        Args:
            max_workers: Maximum concurrent agents
        """
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def run_parallel(
        self,
        tasks: List[AgentTask],
        execute_fn: Callable[[AgentTask], Any],
        timeout: Optional[float] = 60.0
    ) -> List[AgentResult]:
        """
        Run multiple agent tasks in parallel.

        Args:
            tasks: List of agent tasks to execute
            execute_fn: Function to execute each task
            timeout: Timeout per task in seconds

        Returns:
            List of AgentResult objects
        """
        results: List[AgentResult] = []

        if not tasks:
            return results

        logger.info(f"Launching {len(tasks)} agents in parallel (max {self.max_workers} concurrent)")

        # Submit all tasks
        future_to_task = {}
        for task in tasks:
            future = self._executor.submit(self._execute_with_timing, execute_fn, task)
            future_to_task[future] = task

        # Collect results as they complete
        for future in as_completed(future_to_task, timeout=timeout):
            task = future_to_task[future]
            try:
                result, duration_ms = future.result(timeout=timeout)
                results.append(AgentResult(
                    task_name=task.name,
                    agent_type=task.agent_type,
                    success=True,
                    result=result,
                    duration_ms=duration_ms
                ))
                logger.debug(f"Agent '{task.name}' completed in {duration_ms}ms")
            except Exception as e:
                results.append(AgentResult(
                    task_name=task.name,
                    agent_type=task.agent_type,
                    success=False,
                    result=None,
                    error=str(e)
                ))
                logger.error(f"Agent '{task.name}' failed: {e}")

        return results

    async def run_parallel_async(
        self,
        tasks: List[AgentTask],
        execute_fn: Callable[[AgentTask], Any],
        timeout: Optional[float] = 60.0
    ) -> List[AgentResult]:
        """
        Async version of run_parallel.

        Args:
            tasks: List of agent tasks to execute
            execute_fn: Async function to execute each task
            timeout: Timeout per task in seconds

        Returns:
            List of AgentResult objects
        """
        if not tasks:
            return []

        logger.info(f"Launching {len(tasks)} agents in parallel (async)")

        async def execute_task(task: AgentTask) -> AgentResult:
            start = time.time()
            try:
                if asyncio.iscoroutinefunction(execute_fn):
                    result = await asyncio.wait_for(execute_fn(task), timeout=timeout)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        self._executor, execute_fn, task
                    )
                duration_ms = int((time.time() - start) * 1000)
                return AgentResult(
                    task_name=task.name,
                    agent_type=task.agent_type,
                    success=True,
                    result=result,
                    duration_ms=duration_ms
                )
            except Exception as e:
                duration_ms = int((time.time() - start) * 1000)
                return AgentResult(
                    task_name=task.name,
                    agent_type=task.agent_type,
                    success=False,
                    result=None,
                    error=str(e),
                    duration_ms=duration_ms
                )

        # Run all tasks concurrently
        results = await asyncio.gather(*[execute_task(t) for t in tasks])
        return list(results)

    def _execute_with_timing(
        self,
        execute_fn: Callable[[AgentTask], Any],
        task: AgentTask
    ) -> tuple:
        """Execute a task and return result with timing."""
        start = time.time()
        result = execute_fn(task)
        duration_ms = int((time.time() - start) * 1000)
        return result, duration_ms

    def shutdown(self):
        """Shutdown the executor."""
        self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


def merge_exploration_results(results: List[AgentResult]) -> Dict[str, Any]:
    """
    Merge results from multiple exploration agents.

    Args:
        results: List of AgentResult from parallel exploration

    Returns:
        Merged findings dictionary
    """
    merged = {
        "files_found": [],
        "patterns": [],
        "insights": [],
        "errors": []
    }

    for result in results:
        if result.success and result.result:
            if isinstance(result.result, dict):
                # Merge lists
                for key in ["files_found", "patterns", "insights"]:
                    if key in result.result:
                        merged[key].extend(result.result[key])
            elif isinstance(result.result, str):
                merged["insights"].append({
                    "source": result.task_name,
                    "content": result.result
                })
        elif result.error:
            merged["errors"].append({
                "task": result.task_name,
                "error": result.error
            })

    # Deduplicate
    merged["files_found"] = list(set(merged["files_found"]))

    return merged


def select_best_architecture(results: List[AgentResult]) -> Optional[AgentResult]:
    """
    Select the best architecture proposal from parallel architect agents.

    Uses simple heuristics - in production you might want LLM-based selection.

    Args:
        results: List of AgentResult from parallel architects

    Returns:
        The best AgentResult or None
    """
    successful = [r for r in results if r.success and r.result]

    if not successful:
        return None

    if len(successful) == 1:
        return successful[0]

    # Score each result (simple heuristic: longer = more detailed = better)
    def score_result(result: AgentResult) -> int:
        if isinstance(result.result, str):
            return len(result.result)
        elif isinstance(result.result, dict):
            return len(str(result.result))
        return 0

    return max(successful, key=score_result)
