"""
Session State Tracker - Tracks what the AI has done to avoid redundant work.

This module provides:
- File read tracking (avoid re-reading unchanged files)
- Issue/task tracking (know what's fixed vs pending)
- Checkpoint management (continue from where you left off)
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from loguru import logger


@dataclass
class FileReadRecord:
    """Record of a file that was read."""
    path: str
    content_hash: str
    read_at: str
    line_count: int
    summary: Optional[str] = None  # Brief summary of what was found


@dataclass
class IssueRecord:
    """Record of an identified issue."""
    id: str
    description: str
    file_path: Optional[str]
    line_number: Optional[int]
    status: str  # 'pending', 'in_progress', 'fixed', 'wont_fix'
    created_at: str
    fixed_at: Optional[str] = None
    fix_description: Optional[str] = None


@dataclass
class TaskCheckpoint:
    """Checkpoint for resuming interrupted tasks."""
    task_description: str
    current_step: str
    completed_steps: List[str]
    pending_steps: List[str]
    created_at: str
    updated_at: str


@dataclass
class SessionState:
    """Complete session state."""
    project_path: str
    started_at: str
    updated_at: str
    files_read: Dict[str, FileReadRecord] = field(default_factory=dict)
    issues: Dict[str, IssueRecord] = field(default_factory=dict)
    checkpoint: Optional[TaskCheckpoint] = None
    context_notes: List[str] = field(default_factory=list)  # Important findings


class SessionStateTracker:
    """
    Tracks session state to avoid redundant work and enable continuation.

    Usage:
        tracker = SessionStateTracker(project_path)

        # Record file read
        tracker.record_file_read("/path/to/file.js", content, summary="Contains AI logic")

        # Check if file was already read
        if tracker.was_file_read("/path/to/file.js"):
            content_hash = tracker.get_file_hash("/path/to/file.js")
            # Skip re-reading if unchanged

        # Track issues
        tracker.add_issue("display-bug", "Turn indicator shows wrong value", file_path="script.js")
        tracker.mark_issue_fixed("display-bug", "Updated updateDisplay function")

        # Checkpoint for continuation
        tracker.set_checkpoint(
            task="Add AI opponent",
            current_step="Implementing makeAIMove function",
            completed=["Added AI state variables", "Updated HTML"],
            pending=["Add difficulty selector", "Test AI"]
        )

        # Get summary for LLM prompt
        summary = tracker.get_state_summary()
    """

    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.state_file = self.project_path / ".nasea_session.json"
        self.state = self._load_state()

    def _load_state(self) -> SessionState:
        """Load state from file or create new."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                # Reconstruct dataclasses
                files_read = {
                    k: FileReadRecord(**v)
                    for k, v in data.get("files_read", {}).items()
                }
                issues = {
                    k: IssueRecord(**v)
                    for k, v in data.get("issues", {}).items()
                }
                checkpoint = None
                if data.get("checkpoint"):
                    checkpoint = TaskCheckpoint(**data["checkpoint"])

                return SessionState(
                    project_path=data.get("project_path", str(self.project_path)),
                    started_at=data.get("started_at", datetime.now().isoformat()),
                    updated_at=data.get("updated_at", datetime.now().isoformat()),
                    files_read=files_read,
                    issues=issues,
                    checkpoint=checkpoint,
                    context_notes=data.get("context_notes", [])
                )
            except Exception as e:
                logger.warning(f"Failed to load session state: {e}")

        return SessionState(
            project_path=str(self.project_path),
            started_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )

    def _save_state(self):
        """Save state to file."""
        try:
            data = {
                "project_path": self.state.project_path,
                "started_at": self.state.started_at,
                "updated_at": datetime.now().isoformat(),
                "files_read": {k: asdict(v) for k, v in self.state.files_read.items()},
                "issues": {k: asdict(v) for k, v in self.state.issues.items()},
                "checkpoint": asdict(self.state.checkpoint) if self.state.checkpoint else None,
                "context_notes": self.state.context_notes
            }
            self.state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save session state: {e}")

    def _hash_content(self, content: str) -> str:
        """Generate hash of file content."""
        return hashlib.md5(content.encode()).hexdigest()[:12]

    # -------------------------------------------------------------------------
    # File Tracking
    # -------------------------------------------------------------------------

    def record_file_read(self, file_path: str, content: str, summary: Optional[str] = None):
        """Record that a file was read."""
        rel_path = self._relative_path(file_path)
        self.state.files_read[rel_path] = FileReadRecord(
            path=rel_path,
            content_hash=self._hash_content(content),
            read_at=datetime.now().isoformat(),
            line_count=content.count('\n') + 1,
            summary=summary
        )
        self._save_state()

    def was_file_read(self, file_path: str) -> bool:
        """Check if file was already read in this session."""
        rel_path = self._relative_path(file_path)
        return rel_path in self.state.files_read

    def is_file_changed(self, file_path: str, current_content: str) -> bool:
        """Check if file content changed since last read."""
        rel_path = self._relative_path(file_path)
        if rel_path not in self.state.files_read:
            return True  # Never read = treat as changed

        old_hash = self.state.files_read[rel_path].content_hash
        new_hash = self._hash_content(current_content)
        return old_hash != new_hash

    def get_file_summary(self, file_path: str) -> Optional[str]:
        """Get summary of previously read file."""
        rel_path = self._relative_path(file_path)
        record = self.state.files_read.get(rel_path)
        return record.summary if record else None

    def _relative_path(self, file_path: str) -> str:
        """Convert to relative path for consistency."""
        try:
            return str(Path(file_path).relative_to(self.project_path))
        except ValueError:
            return file_path

    # -------------------------------------------------------------------------
    # Issue Tracking
    # -------------------------------------------------------------------------

    def add_issue(
        self,
        issue_id: str,
        description: str,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None
    ):
        """Add an identified issue."""
        self.state.issues[issue_id] = IssueRecord(
            id=issue_id,
            description=description,
            file_path=self._relative_path(file_path) if file_path else None,
            line_number=line_number,
            status="pending",
            created_at=datetime.now().isoformat()
        )
        self._save_state()

    def mark_issue_in_progress(self, issue_id: str):
        """Mark issue as being worked on."""
        if issue_id in self.state.issues:
            self.state.issues[issue_id].status = "in_progress"
            self._save_state()

    def mark_issue_fixed(self, issue_id: str, fix_description: Optional[str] = None):
        """Mark issue as fixed."""
        if issue_id in self.state.issues:
            self.state.issues[issue_id].status = "fixed"
            self.state.issues[issue_id].fixed_at = datetime.now().isoformat()
            self.state.issues[issue_id].fix_description = fix_description
            self._save_state()

    def get_pending_issues(self) -> List[IssueRecord]:
        """Get all pending/in-progress issues."""
        return [
            issue for issue in self.state.issues.values()
            if issue.status in ("pending", "in_progress")
        ]

    def get_fixed_issues(self) -> List[IssueRecord]:
        """Get all fixed issues."""
        return [
            issue for issue in self.state.issues.values()
            if issue.status == "fixed"
        ]

    # -------------------------------------------------------------------------
    # Checkpoint Management
    # -------------------------------------------------------------------------

    def set_checkpoint(
        self,
        task: str,
        current_step: str,
        completed: List[str],
        pending: List[str]
    ):
        """Set a checkpoint for task continuation."""
        now = datetime.now().isoformat()
        self.state.checkpoint = TaskCheckpoint(
            task_description=task,
            current_step=current_step,
            completed_steps=completed,
            pending_steps=pending,
            created_at=self.state.checkpoint.created_at if self.state.checkpoint else now,
            updated_at=now
        )
        self._save_state()

    def clear_checkpoint(self):
        """Clear checkpoint when task is complete."""
        self.state.checkpoint = None
        self._save_state()

    def add_context_note(self, note: str):
        """Add an important finding/note."""
        self.state.context_notes.append(f"[{datetime.now().strftime('%H:%M')}] {note}")
        # Keep only last 10 notes
        self.state.context_notes = self.state.context_notes[-10:]
        self._save_state()

    # -------------------------------------------------------------------------
    # State Summary for LLM Prompt
    # -------------------------------------------------------------------------

    def get_state_summary(self) -> str:
        """
        Generate a summary of session state for injection into LLM prompt.
        This tells the AI what it already knows to avoid redundant work.
        """
        lines = ["# SESSION STATE (avoid redundant work)"]

        # Files already read
        if self.state.files_read:
            lines.append("\n## Files Already Read:")
            for rel_path, record in self.state.files_read.items():
                summary_part = f" - {record.summary}" if record.summary else ""
                lines.append(f"- {rel_path} ({record.line_count} lines){summary_part}")
            lines.append("→ Don't re-read these unless checking for changes.")

        # Issues
        pending = self.get_pending_issues()
        fixed = self.get_fixed_issues()

        if pending:
            lines.append("\n## Pending Issues:")
            for issue in pending:
                loc = f" ({issue.file_path}:{issue.line_number})" if issue.file_path else ""
                status = "🔧 IN PROGRESS" if issue.status == "in_progress" else "⏳ PENDING"
                lines.append(f"- [{status}] {issue.description}{loc}")

        if fixed:
            lines.append("\n## Already Fixed:")
            for issue in fixed:
                lines.append(f"- ✅ {issue.description}")
            lines.append("→ Don't re-fix these issues.")

        # Checkpoint
        if self.state.checkpoint:
            cp = self.state.checkpoint
            lines.append(f"\n## Current Task: {cp.task_description}")
            lines.append(f"Current Step: {cp.current_step}")
            if cp.completed_steps:
                lines.append("Completed: " + ", ".join(cp.completed_steps))
            if cp.pending_steps:
                lines.append("Remaining: " + ", ".join(cp.pending_steps))
            lines.append("→ Continue from current step, don't restart.")

        # Context notes
        if self.state.context_notes:
            lines.append("\n## Important Findings:")
            for note in self.state.context_notes[-5:]:
                lines.append(f"- {note}")

        return "\n".join(lines) + "\n"

    def reset(self):
        """Reset session state (start fresh)."""
        self.state = SessionState(
            project_path=str(self.project_path),
            started_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        self._save_state()
