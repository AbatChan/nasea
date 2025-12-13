"""
File Store - Manages storage and retrieval of generated code files.
Based on L2MAC's persistent memory architecture.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class FileMetadata:
    """Metadata for a generated file."""
    path: str
    language: str
    size: int
    created_at: str
    modified_at: str
    purpose: str
    dependencies: List[str]
    test_status: Optional[str] = None
    quality_score: Optional[float] = None


@dataclass
class ProjectManifest:
    """Manifest describing the entire generated project."""
    name: str
    description: str
    created_at: str
    files: List[FileMetadata]
    entry_point: Optional[str] = None
    dependencies: List[str] = None
    language: str = "python"
    framework: Optional[str] = None
    tests_passing: bool = False
    total_lines: int = 0


class FileStore:
    """
    Manages the persistent storage of generated code files.
    Implements L2MAC-style external memory for code generation.
    """

    def __init__(self, base_path: Path, project_name: str):
        """
        Initialize the file store.

        Args:
            base_path: Base directory for storing projects
            project_name: Name of the current project
        """
        self.project_name = project_name
        self.base_path = Path(base_path)
        self.project_path = self.base_path / self._sanitize_name(project_name)
        self.manifest_path = self.project_path / "nasea_manifest.json"
        self.files: Dict[str, FileMetadata] = {}

        # Create project directory
        self.project_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"FileStore initialized at: {self.project_path}")

    def _sanitize_name(self, name: str) -> str:
        """Sanitize project name for filesystem."""
        # Remove invalid characters
        sanitized = "".join(c for c in name if c.isalnum() or c in ("-", "_", " "))
        sanitized = sanitized.replace(" ", "_").lower()
        return sanitized[:100]  # Limit length

    def save_file(
        self,
        relative_path: str,
        content: str,
        language: str = "python",
        purpose: str = "implementation",
        dependencies: Optional[List[str]] = None
    ) -> Path:
        """
        Save a file to the project directory.

        Args:
            relative_path: Path relative to project root (e.g., "src/main.py")
            content: File content
            language: Programming language
            purpose: Purpose of the file (implementation, test, config, etc.)
            dependencies: List of dependencies this file needs

        Returns:
            Absolute path to the saved file
        """
        file_path = self.project_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write content
        file_path.write_text(content, encoding="utf-8")

        # Create metadata
        metadata = FileMetadata(
            path=relative_path,
            language=language,
            size=len(content),
            created_at=datetime.now().isoformat(),
            modified_at=datetime.now().isoformat(),
            purpose=purpose,
            dependencies=dependencies or []
        )

        self.files[relative_path] = metadata
        logger.debug(f"Saved file: {relative_path} ({len(content)} chars)")

        return file_path

    def read_file(self, relative_path: str) -> Optional[str]:
        """
        Read a file from the project directory.

        Args:
            relative_path: Path relative to project root

        Returns:
            File content or None if not found
        """
        file_path = self.project_path / relative_path
        if not file_path.exists():
            logger.warning(f"File not found: {relative_path}")
            return None

        return file_path.read_text(encoding="utf-8")

    def update_file(self, relative_path: str, content: str) -> bool:
        """
        Update an existing file.

        Args:
            relative_path: Path relative to project root
            content: New content

        Returns:
            True if successful, False otherwise
        """
        if relative_path not in self.files:
            logger.warning(f"Cannot update non-existent file: {relative_path}")
            return False

        file_path = self.project_path / relative_path
        file_path.write_text(content, encoding="utf-8")

        # Update metadata
        self.files[relative_path].size = len(content)
        self.files[relative_path].modified_at = datetime.now().isoformat()

        logger.debug(f"Updated file: {relative_path}")
        return True

    def delete_file(self, relative_path: str) -> bool:
        """
        Delete a file from the project.

        Args:
            relative_path: Path relative to project root

        Returns:
            True if successful, False otherwise
        """
        if relative_path not in self.files:
            logger.warning(f"Cannot delete non-existent file: {relative_path}")
            return False

        file_path = self.project_path / relative_path
        if file_path.exists():
            file_path.unlink()

        del self.files[relative_path]
        logger.debug(f"Deleted file: {relative_path}")
        return True

    def list_files(self, purpose: Optional[str] = None, language: Optional[str] = None) -> List[FileMetadata]:
        """
        List files in the project with optional filtering.

        Args:
            purpose: Filter by purpose (implementation, test, config, etc.)
            language: Filter by language

        Returns:
            List of file metadata
        """
        files = list(self.files.values())

        if purpose:
            files = [f for f in files if f.purpose == purpose]

        if language:
            files = [f for f in files if f.language == language]

        return files

    def get_project_structure(self) -> Dict[str, Any]:
        """
        Get the directory structure of the project.

        Returns:
            Dictionary representing the project structure
        """
        structure = {}

        for file_path in self.files.keys():
            parts = Path(file_path).parts
            current = structure

            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    # Leaf node (file)
                    current[part] = {
                        "type": "file",
                        "metadata": self.files[file_path]
                    }
                else:
                    # Directory node
                    if part not in current:
                        current[part] = {"type": "directory"}
                    current = current[part]

        return structure

    def save_manifest(
        self,
        description: str,
        entry_point: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        language: str = "python",
        framework: Optional[str] = None,
        tests_passing: bool = False
    ):
        """
        Save project manifest with metadata.

        Args:
            description: Project description
            entry_point: Main entry point file
            dependencies: List of external dependencies
            language: Primary programming language
            framework: Framework used (if any)
            tests_passing: Whether all tests are passing
        """
        total_lines = sum(
            metadata.size // 80  # Rough estimate: ~80 chars per line
            for metadata in self.files.values()
        )

        manifest = ProjectManifest(
            name=self.project_name,
            description=description,
            created_at=datetime.now().isoformat(),
            files=list(self.files.values()),
            entry_point=entry_point,
            dependencies=dependencies or [],
            language=language,
            framework=framework,
            tests_passing=tests_passing,
            total_lines=total_lines
        )

        # Save as JSON
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(asdict(manifest), f, indent=2)

        logger.info(f"Manifest saved: {len(self.files)} files, ~{total_lines} lines")

    def export_project(self, destination: Optional[Path] = None) -> Path:
        """
        Export the entire project as a zip or copy to destination.

        Args:
            destination: Where to export (if None, returns project path)

        Returns:
            Path to the exported project
        """
        if destination is None:
            return self.project_path

        destination = Path(destination)
        if destination.suffix == ".zip":
            # Create zip archive
            shutil.make_archive(
                str(destination.with_suffix("")),
                "zip",
                self.project_path
            )
            logger.info(f"Project exported to: {destination}")
            return destination
        else:
            # Copy directory
            shutil.copytree(self.project_path, destination, dirs_exist_ok=True)
            logger.info(f"Project copied to: {destination}")
            return destination

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the project.

        Returns:
            Dictionary with project statistics
        """
        total_size = sum(m.size for m in self.files.values())
        by_language = {}
        by_purpose = {}

        for metadata in self.files.values():
            # Count by language
            by_language[metadata.language] = by_language.get(metadata.language, 0) + 1
            # Count by purpose
            by_purpose[metadata.purpose] = by_purpose.get(metadata.purpose, 0) + 1

        return {
            "total_files": len(self.files),
            "total_size": total_size,
            "by_language": by_language,
            "by_purpose": by_purpose,
            "project_path": str(self.project_path)
        }

    def cleanup(self):
        """Remove the entire project directory."""
        if self.project_path.exists():
            shutil.rmtree(self.project_path)
            logger.info(f"Cleaned up project: {self.project_name}")

    def __repr__(self) -> str:
        """String representation."""
        return f"FileStore(project={self.project_name}, files={len(self.files)})"
