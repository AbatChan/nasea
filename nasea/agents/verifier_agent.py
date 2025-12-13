"""
Verifier Agent - Tests and validates generated code.

Features:
- Confidence-based filtering (only surfaces high-confidence issues)
- Static analysis with pylint
- Auto-generated tests with pytest
- Security scanning with bandit

Configuration is loaded from nasea/config/confidence.yaml and can be
overridden in project-level .nasea/config.yaml files.
"""

import subprocess
import tempfile
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from loguru import logger

from nasea.agents.base_agent import BaseAgent
from nasea.core.file_store import FileStore
from nasea.prompts import load_agent_prompt
from nasea.config import load_config


@dataclass
class Issue:
    """
    Represents a code issue with confidence scoring.

    Confidence levels:
    - 90-100: Critical/definite issue (syntax errors, failing tests)
    - 70-89: High confidence (security vulnerabilities, type errors)
    - 50-69: Medium confidence (style issues, potential bugs)
    - 0-49: Low confidence (suggestions, minor style)
    """
    file: str
    type: str
    description: str
    error: str = ""
    line: Optional[int] = None
    confidence: int = 50  # 0-100 scale
    severity: str = "medium"  # low, medium, high, critical
    suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "file": self.file,
            "type": self.type,
            "description": self.description,
            "error": self.error,
            "line": self.line,
            "confidence": self.confidence,
            "severity": self.severity,
            "suggestion": self.suggestion
        }


# Load confidence configuration from YAML
def _load_confidence_config() -> Dict[str, Any]:
    """Load confidence configuration from YAML file."""
    try:
        return load_config("confidence")
    except Exception as e:
        logger.warning(f"Could not load confidence.yaml: {e}")
        return {}


def _get_thresholds() -> Dict[str, int]:
    """Get confidence thresholds from config."""
    config = _load_confidence_config()
    thresholds = config.get("thresholds", {})
    return {
        "high": thresholds.get("high", 80),
        "default_min": thresholds.get("default_min", 70),
        "noise": thresholds.get("noise", 40),
    }


def _get_pylint_confidence(code: str) -> int:
    """Get confidence level for a pylint error code from config."""
    config = _load_confidence_config()
    pylint_config = config.get("pylint", {})

    # Check specific code
    if code in pylint_config:
        return pylint_config[code]

    # Fall back to category defaults
    defaults = config.get("default_by_category", {})
    category = code[0] if code else "E"
    return defaults.get(category, 50)


def _get_bandit_config() -> Dict[str, Any]:
    """Get bandit security scan configuration."""
    config = _load_confidence_config()
    return config.get("bandit", {
        "severity_to_confidence": {"HIGH": 90, "MEDIUM": 75, "LOW": 55},
        "confidence_adjustment": {"HIGH": 5, "MEDIUM": 0, "LOW": -15}
    })


def _get_test_failure_config() -> Dict[str, Any]:
    """Get test failure configuration."""
    config = _load_confidence_config()
    return config.get("test_failures", {"confidence": 95, "severity": "high"})


# Convenience accessors for thresholds
def get_high_threshold() -> int:
    """Get high confidence threshold."""
    return _get_thresholds()["high"]


def get_default_min_confidence() -> int:
    """Get default minimum confidence for filtering."""
    return _get_thresholds()["default_min"]


# Legacy constants (for backward compatibility)
CONFIDENCE_THRESHOLD_HIGH = get_high_threshold()
DEFAULT_MIN_CONFIDENCE = get_default_min_confidence()


class VerifierAgent(BaseAgent):
    """
    Verifier Agent: Tests code quality, runs tests, and identifies issues.
    """

    def __init__(self, config, memory):
        super().__init__(role="verifier", config=config, memory=memory)

    def _default_system_prompt(self) -> str:
        try:
            return load_agent_prompt("verifier")
        except FileNotFoundError:
            # Fallback to inline prompt if file not found
            return (
                "You are a Senior QA Engineer and Code Reviewer.\n"
                "Responsibilities:\n"
                "1. Review code for bugs and issues\n"
                "2. Generate comprehensive test cases\n"
                "3. Verify code meets requirements\n"
                "4. Identify security vulnerabilities\n"
                "5. Check code quality and best practices\n\n"
                "Test Rules:\n"
                "- Use pytest\n"
                "- Test happy paths + edge cases\n"
                "- Include assertions\n"
                "- Test error handling\n"
            )

    def verify_project(
        self,
        file_store: FileStore,
        original_prompt: str,
        min_confidence: int = DEFAULT_MIN_CONFIDENCE
    ) -> Dict[str, Any]:
        """
        Verify the generated project with confidence-based filtering.

        Args:
            file_store: The file store containing generated files
            original_prompt: The original user prompt
            min_confidence: Minimum confidence level to report issues (0-100)
                           Default: 70 (filters out low-confidence noise)

        Returns:
            Dict with tests_passed, tests_total, issues, success, and filtered stats
        """
        self.log("Starting project verification...")

        all_issues: List[Issue] = []
        tests_passed = 0
        tests_total = 0

        if self.config.run_static_analysis:
            all_issues.extend(self._run_static_analysis(file_store))

        if self.config.auto_generate_tests:
            test_results = self._generate_and_run_tests(file_store, original_prompt)
            tests_passed = test_results["passed"]
            tests_total = test_results["total"]
            all_issues.extend(test_results["issues"])

        if self.config.run_security_scan:
            all_issues.extend(self._run_security_scan(file_store))

        # Apply confidence-based filtering
        filtered_issues = [i for i in all_issues if i.confidence >= min_confidence]
        low_confidence_count = len(all_issues) - len(filtered_issues)

        # Convert to dicts for backward compatibility
        issues_dicts = [i.to_dict() for i in filtered_issues]

        # Success only if no high-confidence issues
        critical_issues = [i for i in filtered_issues if i.confidence >= CONFIDENCE_THRESHOLD_HIGH]

        result = {
            "tests_passed": tests_passed,
            "tests_total": tests_total,
            "issues": issues_dicts,
            "success": len(critical_issues) == 0 and (tests_total == 0 or tests_passed == tests_total),
            "total_issues_found": len(all_issues),
            "issues_filtered_out": low_confidence_count,
            "min_confidence_used": min_confidence
        }

        self.log(
            f"Verification complete: {tests_passed}/{tests_total} tests passed, "
            f"{len(filtered_issues)} issues reported ({low_confidence_count} low-confidence filtered out)"
        )
        return result

    def _run_static_analysis(self, file_store: FileStore) -> List[Issue]:
        """
        Run static analysis with confidence-scored issues.

        Confidence levels are loaded from nasea/config/confidence.yaml:
        - Syntax errors: 100 (definite, code won't run)
        - Import errors: 90-95 (very likely to fail)
        - Undefined variables: 90 (high confidence bug)
        - Other pylint errors: 70-85 (depend on error code)
        - Pylint warnings: 35-65 (style/potential issues)
        """
        issues: List[Issue] = []
        py_files = [f for f in file_store.files.keys() if f.endswith(".py")]

        # Pylint error code confidence mapping - now loaded from YAML
        # This is kept as fallback in case config loading fails
        pylint_confidence_fallback = {
            # Fatal errors (F) - 95-100
            "F0001": 100,  # File not found
            "F0002": 100,  # Syntax error
            # Errors (E) - 80-95
            "E0001": 100,  # Syntax error
            "E0100": 90,   # __init__ returns something
            "E0101": 90,   # Return in __init__
            "E0102": 95,   # Function redefined
            "E0103": 90,   # Break/continue not in loop
            "E0104": 90,   # Return outside function
            "E0105": 90,   # Yield outside function
            "E0107": 85,   # Nonexistent operator
            "E0108": 85,   # Duplicate argument
            "E0110": 80,   # Abstract class instantiated
            "E0111": 80,   # Reversed call
            "E0112": 75,   # Method not callable
            "E0113": 75,   # Non-string item
            "E0114": 75,   # Non-string key
            "E0115": 80,   # Non-local without binding
            "E0116": 80,   # Continue in finally
            "E0117": 85,   # Nonlocal and global
            "E0118": 80,   # Named expr outside scope
            "E0119": 75,   # Return with argument
            "E0211": 85,   # Missing self argument
            "E0213": 85,   # Method should have self
            "E0401": 90,   # Import error
            "E0402": 85,   # Relative import beyond top-level
            "E0601": 90,   # Used before assignment
            "E0602": 90,   # Undefined variable
            "E0603": 85,   # Undefined all
            "E0604": 85,   # Invalid all object
            "E0611": 85,   # No name in module
            "E0701": 80,   # Bad except order
            "E0702": 85,   # Raising bad type
            "E0703": 80,   # Bad exception context
            "E0704": 85,   # Misplaced bare raise
            "E0710": 80,   # Raising non-exception
            "E0711": 85,   # NotImplemented raised
            "E0712": 75,   # Catching non-exception
            "E1003": 70,   # Bad first argument to super
            "E1101": 75,   # No member (can be false positive)
            "E1102": 80,   # Not callable
            "E1111": 75,   # Assignment from no return
            "E1120": 85,   # No value for argument
            "E1121": 85,   # Too many positional
            "E1123": 80,   # Unexpected keyword
            "E1124": 80,   # Redundant keyword
            "E1125": 80,   # Missing kwonly argument
            "E1126": 75,   # Invalid sequence index
            "E1127": 75,   # Invalid slice index
            "E1128": 75,   # Assignment from None
            "E1129": 75,   # Not context manager
            "E1130": 70,   # Invalid unary operand
            "E1131": 70,   # Not iterable
            "E1132": 75,   # Repeated keyword
            "E1133": 70,   # Not mapping
            "E1134": 70,   # Not iterable
            "E1135": 70,   # Unsupported membership
            "E1136": 70,   # Unsubscriptable
            "E1137": 70,   # Unsupported assignment
            "E1138": 70,   # Unsupported delete
            "E1139": 70,   # Invalid metaclass
            "E1140": 70,   # Unhashable member
            "E1141": 70,   # Dict iter delete
            "E1142": 75,   # Await outside async
            "E1200": 75,   # Logging unsupported format
            "E1201": 75,   # Logging bad format
            "E1205": 75,   # Logging too many args
            "E1206": 75,   # Logging too few args
            "E1300": 75,   # Bad format character
            "E1301": 75,   # Truncated format
            "E1302": 75,   # Mixed format string
            "E1303": 75,   # Format missing argument
            "E1304": 75,   # Missing format key
            "E1305": 75,   # Too many format args
            "E1306": 75,   # Too few format args
            "E1307": 70,   # Bad string format type
            "E1310": 70,   # Bad str strip call
            "E1507": 70,   # Invalid envvar value
            "E1519": 70,   # Singledispatch method
            "E1700": 80,   # Yield inside async
            # Warnings (W) - 40-65
            "W0101": 60,   # Unreachable code
            "W0102": 55,   # Dangerous default value
            "W0104": 40,   # Pointless statement
            "W0105": 35,   # Pointless string statement (docstrings)
            "W0106": 45,   # Expression not assigned
            "W0107": 30,   # Unnecessary pass
            "W0108": 40,   # Lambda may not be necessary
            "W0109": 50,   # Duplicate key
            "W0120": 45,   # Else clause on loop
            "W0122": 65,   # Use of exec
            "W0123": 65,   # Use of eval
            "W0124": 50,   # Self cls assignment
            "W0125": 40,   # Using constant test
            "W0126": 50,   # Invalid condition
            "W0127": 50,   # Self assigning variable
            "W0128": 50,   # Redeclared in handler
            "W0129": 40,   # Assert on string
            "W0131": 45,   # Named expression in except
            "W0199": 60,   # Assert on tuple
            "W0211": 50,   # Bad staticmethod argument
            "W0212": 45,   # Protected access
            "W0221": 55,   # Arguments differ
            "W0222": 55,   # Signature differs
            "W0223": 60,   # Abstract method not overridden
            "W0231": 55,   # Init not called
            "W0232": 50,   # No init
            "W0233": 55,   # Init from bad parent
            "W0235": 40,   # Useless super delegation
            "W0236": 50,   # Invalid overridden method
            "W0237": 50,   # Non parent init called
            "W0238": 45,   # Unused private member
            "W0301": 30,   # Unnecessary semicolon
            "W0311": 25,   # Bad indentation (style)
            "W0401": 50,   # Wildcard import
            "W0402": 40,   # Deprecated module
            "W0404": 50,   # Reimported
            "W0406": 55,   # Module import itself
            "W0410": 60,   # Future import wrong position
            "W0511": 20,   # Fixme/todo (informational)
            "W0601": 60,   # Global variable undefined
            "W0602": 45,   # Global variable not assigned
            "W0603": 50,   # Using global statement
            "W0604": 50,   # Using global at module level
            "W0611": 35,   # Unused import
            "W0612": 40,   # Unused variable
            "W0613": 35,   # Unused argument
            "W0614": 45,   # Unused wildcard import
            "W0621": 50,   # Redefined outer name
            "W0622": 50,   # Redefined builtin
            "W0631": 55,   # Undefined loop variable
            "W0632": 60,   # Unbalanced tuple unpacking
            "W0640": 55,   # Cell variable in loop
            "W0641": 45,   # Possibly unused variable
            "W0642": 50,   # Self cls variable
            "W0702": 55,   # Bare except
            "W0703": 50,   # Broad except
            "W0705": 55,   # Duplicate except
            "W0706": 50,   # Try except raise
            "W0707": 55,   # Raise missing from
            "W0711": 50,   # Except with identity
            "W0715": 50,   # String exception
            "W0716": 55,   # Wrong exception operation
            "W0718": 45,   # Broad exception caught
            "W0719": 45,   # Broad exception raised
            "W1113": 50,   # Keyword arg before vararg
            "W1114": 45,   # Arguments out of order
            "W1115": 50,   # Non string format
            "W1116": 50,   # Not number
            "W1117": 50,   # Bad keyword argument
            "W1201": 40,   # Logging not lazy
            "W1202": 40,   # Logging format interpolation
            "W1203": 40,   # Logging fstring interpolation
            "W1300": 45,   # Bad format string key
            "W1301": 50,   # Unused format key
            "W1302": 50,   # Bad format string
            "W1303": 50,   # Missing format argument key
            "W1304": 50,   # Unused format arg
            "W1305": 40,   # Format combined specs
            "W1308": 40,   # Duplicate string formatting
            "W1309": 45,   # f-string without interpolation
            "W1310": 45,   # Format string without args
            "W1401": 55,   # Anomalous backslash
            "W1402": 55,   # Anomalous unicode
            "W1404": 45,   # Implicit str concat in seq
            "W1405": 45,   # Using encoding for text
            "W1501": 45,   # Bad open mode
            "W1502": 50,   # Unspecified encoding
            "W1503": 40,   # Redundant unittest assert
            "W1506": 50,   # Bad thread instantiation
            "W1507": 50,   # Shallow copy environ
            "W1508": 50,   # Invalid envvar default
            "W1509": 60,   # Subprocess popen preexec_fn
            "W1510": 55,   # Subprocess run check
            "W1514": 45,   # Unspecified encoding
            "W1515": 40,   # Forgotten debug statement
            "W1518": 50,   # Method cache max size none
        }

        for file_path in py_files:
            full_path = file_store.project_path / file_path
            code = file_store.read_file(file_path)

            # Syntax check - 100% confidence (code won't run)
            try:
                compile(code, file_path, "exec")
            except SyntaxError as e:
                issues.append(Issue(
                    file=file_path,
                    type="syntax_error",
                    description=f"Syntax error on line {e.lineno}: {e.msg}",
                    error=str(e),
                    line=e.lineno,
                    confidence=100,
                    severity="critical",
                    suggestion="Fix the syntax error - code cannot run until this is resolved"
                ))

            # Pylint analysis with confidence mapping
            try:
                result = subprocess.run(
                    ["pylint", "--output-format=text", "--msg-template={path}:{line}:{column}: {msg_id}: {msg}", str(full_path)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.stdout:
                    # Parse pylint output
                    for line in result.stdout.strip().split("\n"):
                        # Match pattern: file.py:10:5: E0602: Undefined variable 'x'
                        match = re.match(r".*?:(\d+):(\d+): ([A-Z]\d{4}): (.+)", line)
                        if match:
                            line_num = int(match.group(1))
                            msg_id = match.group(3)
                            msg = match.group(4)

                            # Get confidence from YAML config, with fallback
                            confidence = _get_pylint_confidence(msg_id)

                            # If config returned default, try local fallback
                            if confidence == 50 and msg_id in pylint_confidence_fallback:
                                confidence = pylint_confidence_fallback[msg_id]

                            # Determine severity
                            if confidence >= 90:
                                severity = "critical"
                            elif confidence >= 70:
                                severity = "high"
                            elif confidence >= 50:
                                severity = "medium"
                            else:
                                severity = "low"

                            issues.append(Issue(
                                file=file_path,
                                type="lint_error" if msg_id.startswith("E") else "lint_warning",
                                description=f"[{msg_id}] {msg}",
                                error=line,
                                line=line_num,
                                confidence=confidence,
                                severity=severity
                            ))

            except FileNotFoundError:
                self.log("Pylint not found, skipping lint analysis", level="debug")
            except subprocess.TimeoutExpired:
                self.log("Pylint timed out", level="warning")

        return issues

    def _generate_and_run_tests(self, file_store: FileStore, original_prompt: str) -> Dict[str, Any]:
        """Generate and run tests, returning results with confidence-scored issues."""
        self.log("Generating tests...")

        py_files = [
            f for f in file_store.files.keys()
            if f.endswith(".py") and not f.startswith("test_")
        ]

        if not py_files:
            return {"passed": 0, "total": 0, "issues": []}

        test_code = self._generate_tests(py_files, file_store, original_prompt)

        if not test_code:
            return {"passed": 0, "total": 0, "issues": []}

        test_file = "tests/test_generated.py"
        file_store.save_file(test_file, test_code, language="python", purpose="test")

        return self._run_pytest(file_store)

    def _generate_tests(
        self,
        files: List[str],
        file_store: FileStore,
        original_prompt: str
    ) -> str:

        segments = []
        for file_path in files:
            segments.append(f"# File: {file_path}\n{file_store.read_file(file_path)}")

        context = "\n\n".join(segments)

        prompt = (
            "Generate pytest test cases for the following code.\n\n"
            f"ORIGINAL REQUEST:\n{original_prompt}\n\n"
            "CODE:\n" + context + "\n\n"
            "Rules:\n"
            "- Include this EXACT header at top:\n"
            "import sys\n"
            "import os\n"
            "sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
            "- Use pytest\n"
            "- Test all functions/classes\n"
            "- Include positive and negative test cases\n"
            "- Edge cases\n"
            "- Return ONLY the full test code."
        )

        response = self.query_llm(prompt)
        return self.extract_code_from_response(response)

    def _run_pytest(self, file_store: FileStore) -> Dict[str, Any]:
        """
        Run pytest and return results with confidence-scored issues.

        Test failures get high confidence because they represent
        actual runtime failures that must be fixed.
        Confidence level is loaded from nasea/config/confidence.yaml.
        """
        # Get test failure config from YAML
        test_config = _get_test_failure_config()
        test_confidence = test_config.get("confidence", 95)
        test_severity = test_config.get("severity", "high")

        try:
            result = subprocess.run(
                ["pytest", str(file_store.project_path), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(file_store.project_path)
            )

            output = result.stdout + result.stderr
            passed = output.count(" PASSED")
            failed = output.count(" FAILED")
            total = passed + failed

            issues: List[Issue] = []
            if failed > 0:
                for line in output.split("\n"):
                    if "FAILED" in line:
                        # Extract test name and error
                        test_match = re.search(r"FAILED (.+?) -", line)
                        test_name = test_match.group(1) if test_match else "unknown test"

                        issues.append(Issue(
                            file="test_generated.py",
                            type="test_failure",
                            description=f"Test failed: {test_name}",
                            error=line.strip(),
                            confidence=test_confidence,
                            severity=test_severity,
                            suggestion="Fix the code to make this test pass"
                        ))

            return {"passed": passed, "total": total, "issues": issues}

        except FileNotFoundError:
            self.log("pytest not found, skipping test execution", level="debug")
            return {"passed": 0, "total": 0, "issues": []}
        except subprocess.TimeoutExpired:
            self.log("pytest timed out", level="warning")
            return {"passed": 0, "total": 0, "issues": []}

    def _run_security_scan(self, file_store: FileStore) -> List[Issue]:
        """
        Run security scan with confidence-scored issues.

        Bandit severity mapping is loaded from nasea/config/confidence.yaml:
        - HIGH severity: ~90 confidence (serious vulnerabilities)
        - MEDIUM severity: ~75 confidence (potential issues)
        - LOW severity: ~55 confidence (minor concerns)
        """
        issues: List[Issue] = []
        py_files = [f for f in file_store.files.keys() if f.endswith(".py")]

        # Bandit config from YAML
        bandit_config = _get_bandit_config()
        severity_confidence = bandit_config.get("severity_to_confidence", {
            "HIGH": 90,
            "MEDIUM": 75,
            "LOW": 55,
        })
        confidence_adjustment = bandit_config.get("confidence_adjustment", {
            "HIGH": 5,
            "MEDIUM": 0,
            "LOW": -15,
        })

        try:
            for file_path in py_files:
                full_path = file_store.project_path / file_path

                # Use JSON output for easier parsing
                result = subprocess.run(
                    ["bandit", "-f", "json", "-r", str(full_path)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.stdout:
                    try:
                        import json
                        bandit_output = json.loads(result.stdout)

                        for finding in bandit_output.get("results", []):
                            severity = finding.get("issue_severity", "MEDIUM")
                            confidence_level = finding.get("issue_confidence", "MEDIUM")

                            # Calculate confidence based on both severity and bandit's confidence
                            base_confidence = severity_confidence.get(severity, 70)

                            # Adjust based on bandit's own confidence rating (from config)
                            adjustment = confidence_adjustment.get(confidence_level, 0)
                            confidence = max(40, min(100, base_confidence + adjustment))

                            issues.append(Issue(
                                file=file_path,
                                type="security",
                                description=f"[{finding.get('test_id', 'B???')}] {finding.get('issue_text', 'Security issue')}",
                                error=finding.get("code", ""),
                                line=finding.get("line_number"),
                                confidence=confidence,
                                severity=severity.lower(),
                                suggestion=finding.get("more_info", "Review this code for security implications")
                            ))

                    except json.JSONDecodeError:
                        # Fallback if JSON parsing fails
                        if "Issue:" in result.stdout:
                            issues.append(Issue(
                                file=file_path,
                                type="security",
                                description="Security issues detected (see details)",
                                error=result.stdout[:500],
                                confidence=70,
                                severity="medium"
                            ))

        except FileNotFoundError:
            self.log("Bandit not found, skipping security scan", level="debug")
        except subprocess.TimeoutExpired:
            self.log("Bandit timed out", level="warning")

        return issues