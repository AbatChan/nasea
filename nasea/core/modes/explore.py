"""
Read-only exploration mode - mimics Claude Code's exploration phase.
NO file modifications allowed, only understanding.

This mode is activated BEFORE any code changes to:
1. Understand the existing codebase structure
2. Identify relevant files for the user's request
3. Extract existing code patterns and conventions
4. Generate informed recommendations

Based on Claude Code's "Explore" sub-agent architecture.
"""

from typing import Dict, Any, List, Optional
import re
from pathlib import Path
from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.table import Table
import json
import logging

logger = logging.getLogger(__name__)


class ExploreMode:
    """
    Read-only codebase exploration.
    Mimics Claude Code's Explore sub-agent (Haiku model, read-only tools).

    Key principles:
    - NO file modifications
    - Fast, lightweight analysis
    - Context gathering for informed decisions
    - Pattern recognition from existing code
    """

    def __init__(self, project_root: Path, console: Console, llm_client):
        """
        Initialize exploration mode.

        Args:
            project_root: Root directory of the project to explore
            console: Rich console for output
            llm_client: LLM client for analysis (lightweight model preferred)
        """
        self.root = project_root
        self.console = console
        self.llm = llm_client
        self.findings: List[Dict[str, Any]] = []

    def explore(self, user_request: str) -> Dict[str, Any]:
        """
        Explore codebase to understand before making changes.

        This is the main entry point that orchestrates the exploration process.

        Args:
            user_request: What the user wants to build/modify

        Returns:
            {
                'project_type': 'web-app|cli|api|library',
                'tech_stack': ['React', 'FastAPI', 'PostgreSQL'],
                'architecture': 'MVC|microservices|monolithic|flat',
                'existing_patterns': [...],
                'relevant_files': [...],
                'recommendations': [...]
            }
        """
        from rich.live import Live
        from rich.spinner import Spinner
        from rich.text import Text

        # Use animated spinner instead of emoji
        with Live(Spinner("dots", text="[cyan]Exploring codebase[/cyan]"), refresh_per_second=10, console=self.console, transient=True):
            try:
                # Step 1: Discover project structure
                structure = self._analyze_structure()

            except Exception as e:
                logger.error(f"Structure analysis failed: {e}")
                structure = {'type': 'unknown', 'architecture': 'unknown'}

        # Silent - no verbose output

        # Step 2: Identify tech stack
        with Live(Spinner("dots", text="[cyan]Analyzing tech stack[/cyan]"), refresh_per_second=10, console=self.console, transient=True):
            tech_stack = self._detect_tech_stack()

        tech_display = ', '.join(tech_stack[:3]) if tech_stack else 'Unknown'
        # Silent

        # Step 3: Find relevant files
        with Live(Spinner("dots", text="[cyan]Finding relevant files[/cyan]"), refresh_per_second=10, console=self.console, transient=True):
            relevant = self._find_relevant_files(user_request)

        # Silent

        # Step 4: Extract patterns
        with Live(Spinner("dots", text="[cyan]Extracting code patterns[/cyan]"), refresh_per_second=10, console=self.console, transient=True):
            patterns = self._extract_patterns(relevant)

        # Silent

        # Step 5: Generate recommendations
        with Live(Spinner("dots", text="[cyan]Generating recommendations[/cyan]"), refresh_per_second=10, console=self.console, transient=True):
            try:
                recommendations = self._generate_recommendations(
                    user_request, structure, tech_stack, patterns
                )
            except Exception as e:
                logger.error(f"Recommendation generation failed: {e}")
                recommendations = ["Proceed with caution - unable to generate recommendations"]

        # Silent - no verbose completion message

        return {
            'project_type': structure['type'],
            'tech_stack': tech_stack,
            'architecture': structure['architecture'],
            'existing_patterns': patterns,
            'relevant_files': relevant,
            'recommendations': recommendations
        }

    def _analyze_structure(self) -> Dict[str, str]:
        """
        Analyze project structure to determine type and architecture.

        Returns:
            {'type': 'web-app|cli|api|library', 'architecture': 'flat|modular|microservices'}
        """

        # Look for indicator files/directories
        indicators = {
            'web-app': ['package.json', 'index.html', 'webpack.config.js', 'vite.config.js', 'public/'],
            'cli': ['setup.py', 'cli.py', '__main__.py', 'bin/'],
            'api': ['app.py', 'main.py', 'routes/', 'api/', 'fastapi', 'flask', 'django'],
            'library': ['setup.py', '__init__.py', 'pyproject.toml', 'lib/'],
            'mobile': ['android/', 'ios/', 'App.tsx', 'app.json'],
        }

        detected_type = 'unknown'
        max_matches = 0

        for proj_type, indicators_list in indicators.items():
            matches = 0
            for indicator in indicators_list:
                if indicator.endswith('/'):
                    # Indicators ending with / represent directories
                    if (self.root / indicator.rstrip('/')).exists():
                        matches += 1
                else:
                    candidate = self.root / indicator
                    if candidate.exists() or any(self.root.rglob(indicator)):
                        matches += 1
            if matches > max_matches:
                max_matches = matches
                detected_type = proj_type

        # Detect architecture pattern
        architecture = 'flat'

        # Check for modular structure
        if (self.root / 'src').exists() and (self.root / 'tests').exists():
            architecture = 'modular'

        # Check for microservices
        if (self.root / 'services').exists() or (self.root / 'microservices').exists():
            architecture = 'microservices'

        # Check for MVC pattern
        if all((self.root / d).exists() for d in ['models', 'views', 'controllers']):
            architecture = 'MVC'

        return {
            'type': detected_type,
            'architecture': architecture
        }

    def _detect_tech_stack(self) -> List[str]:
        """
        Detect technologies used in the project.

        Returns:
            List of detected technologies (e.g., ['React', 'FastAPI', 'PostgreSQL'])
        """

        stack = []

        # Check package.json for JavaScript/Node technologies
        pkg_json = self.root / 'package.json'
        if pkg_json.exists():
            try:
                data = json.loads(pkg_json.read_text())
                deps = {**data.get('dependencies', {}), **data.get('devDependencies', {})}

                # Frontend frameworks
                if 'react' in deps:
                    stack.append('React')
                if 'vue' in deps:
                    stack.append('Vue')
                if '@angular/core' in deps:
                    stack.append('Angular')
                if 'svelte' in deps:
                    stack.append('Svelte')

                # Backend frameworks
                if 'express' in deps:
                    stack.append('Express')
                if 'next' in deps:
                    stack.append('Next.js')

                # Build tools
                if 'typescript' in deps:
                    stack.append('TypeScript')
                if 'webpack' in deps:
                    stack.append('Webpack')
                if 'vite' in deps:
                    stack.append('Vite')

            except Exception as e:
                logger.debug(f"Failed to parse package.json: {e}")

        # Check requirements.txt for Python technologies
        req_txt = self.root / 'requirements.txt'
        if req_txt.exists():
            try:
                content = req_txt.read_text().lower()

                if 'django' in content:
                    stack.append('Django')
                if 'flask' in content:
                    stack.append('Flask')
                if 'fastapi' in content:
                    stack.append('FastAPI')
                if 'pytest' in content:
                    stack.append('pytest')
                if 'sqlalchemy' in content:
                    stack.append('SQLAlchemy')
                if 'pandas' in content:
                    stack.append('Pandas')
                if 'numpy' in content:
                    stack.append('NumPy')

            except Exception as e:
                logger.debug(f"Failed to parse requirements.txt: {e}")

        # Check Cargo.toml for Rust
        cargo_toml = self.root / 'Cargo.toml'
        if cargo_toml.exists():
            stack.append('Rust')

        # Check go.mod for Go
        go_mod = self.root / 'go.mod'
        if go_mod.exists():
            stack.append('Go')

        return stack if stack else ['Unknown']

    def _find_relevant_files(self, user_request: str) -> List[Path]:
        """
        Find files relevant to the user's request using LLM analysis.

        This is a smart filter that avoids reading hundreds of files.

        Args:
            user_request: What the user wants to build/modify

        Returns:
            List of Path objects for relevant files
        """

        # Get all code files (limit to prevent overwhelming LLM)
        code_files = []
        for pattern in ['**/*.py', '**/*.js', '**/*.jsx', '**/*.ts', '**/*.tsx', '**/*.html', '**/*.css']:
            found = list(self.root.glob(pattern))
            # Filter out common exclusions
            filtered = [
                f for f in found
                if not any(excluded in str(f) for excluded in ['node_modules', 'venv', '__pycache__', '.git', 'dist', 'build'])
            ]
            code_files.extend(filtered)

        # Limit to first 100 for performance
        code_files = code_files[:100]

        if not code_files:
            logger.debug("No code files found in project")
            return []

        # Ask LLM which files are relevant
        file_list = "\n".join(f"- {f.relative_to(self.root)}" for f in code_files[:50])

        prompt = f"""Given this user request: "{user_request}"

And these project files:
{file_list}

Which 5-10 files are MOST relevant to this request?

Consider:
- Files that would need to be modified
- Files that provide context for understanding the change
- Configuration files that might need updating

Return ONLY the file paths, one per line, no explanations or markdown."""

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.2,
                disable_thinking=True,
                strip_thinking=True
            )

            # Parse response
            content = getattr(response, 'content', str(response))
            relevant_paths = [
                line.strip('- ').strip()
                for line in content.split('\n')
                if line.strip() and not line.startswith('#')
            ]

            # Validate and convert to Path objects
            valid_files = []
            for path_str in relevant_paths[:10]:  # Max 10 files
                file_path = self.root / path_str
                if file_path.exists() and file_path.is_file():
                    valid_files.append(file_path)

            return valid_files

        except Exception as e:
            logger.error(f"Failed to find relevant files: {e}")
            # Fallback: return first few Python/JS files
            return code_files[:5]

    def _extract_patterns(self, files: List[Path]) -> List[Dict[str, str]]:
        """
        Extract code patterns from relevant files.

        This helps maintain consistency with existing code style.

        Args:
            files: List of files to analyze

        Returns:
            List of pattern dictionaries: [{'file': 'path', 'pattern': 'description'}, ...]
        """

        patterns = []

        for file_path in files[:5]:  # Limit to prevent token overflow
            try:
                content = file_path.read_text(encoding='utf-8')

                # Extract patterns using LLM
                prompt = f"""Analyze this code and extract key patterns:

File: {file_path.name}
```
{content[:2000]}
```

What coding patterns, conventions, and styles are used?
List 2-3 key patterns, one per line, briefly.

Examples:
- Uses functional components with hooks
- Implements repository pattern for data access
- Error handling with try/except blocks

Keep each pattern to one line."""

                response = self.llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.3,
                    disable_thinking=True,
                    strip_thinking=True
                )

                content_response = getattr(response, 'content', str(response))
                for line in content_response.split('\n')[:3]:
                    clean_line = line.strip('- ').strip()
                    if clean_line and len(clean_line) > 10:
                        patterns.append({
                            'file': str(file_path.relative_to(self.root)),
                            'pattern': clean_line
                        })

            except Exception as e:
                logger.debug(f"Failed to extract patterns from {file_path}: {e}")
                continue

        return patterns

    def _generate_recommendations(
        self,
        user_request: str,
        structure: Dict,
        tech_stack: List[str],
        patterns: List[Dict]
    ) -> List[str]:
        """
        Generate implementation recommendations based on exploration.

        This is like having a senior developer review the plan before starting.

        Args:
            user_request: What the user wants to build
            structure: Project structure analysis
            tech_stack: Detected technologies
            patterns: Extracted code patterns

        Returns:
            List of recommendation strings
        """

        patterns_str = "\n".join(f"- {p['pattern']}" for p in patterns[:5])

        prompt = f"""Based on this codebase analysis:

Project Type: {structure['type']}
Architecture: {structure['architecture']}
Tech Stack: {', '.join(tech_stack)}

Existing Code Patterns:
{patterns_str}

User Request: "{user_request}"

Provide 3-5 specific, actionable recommendations for implementing this request.

Consider:
- Which files to modify vs create new
- What patterns to follow from existing code
- Potential gotchas or edge cases
- Testing strategy

Format: One recommendation per line, 1-2 sentences each.
Be specific and practical."""

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.4,
                disable_thinking=True,
                strip_thinking=True
            )

            content = getattr(response, 'content', str(response))
            recommendations = []

            for line in content.split('\n'):
                clean_line = line.strip('- 0123456789. ').strip()
                if clean_line and len(clean_line) > 20:
                    sanitized = self._sanitize_recommendation(clean_line)
                    if sanitized and sanitized not in recommendations:
                        recommendations.append(sanitized)

            return recommendations[:5]

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return [
                "Review existing code structure before implementing",
                "Follow established patterns in the codebase",
                "Test changes incrementally"
            ]

    def display_findings(self, exploration: Dict[str, Any]) -> None:
        """
        Display exploration findings in a nice format.

        Args:
            exploration: Results from explore() method
        """

        # Simple output - no bordered panels
        files = exploration.get('relevant_files', [])
        if files:
            file_names = [f.name if hasattr(f, 'name') else str(f).split('/')[-1] for f in files[:5]]
            self.console.print(f"[bold]Found {len(files)} files:[/bold] {', '.join(file_names)}\n")

    @staticmethod
    def _sanitize_recommendation(text: str) -> str:
        """Remove simple Markdown formatting markers from recommendation text."""
        text = text.strip()
        if not text:
            return text

        # Replace inline code/block markers
        text = re.sub(r'`([^`]*)`', r'\1', text)
        # Remove bold/italic markers
        text = text.replace('**', '').replace('__', '').replace('*', '').replace('_', '')
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
