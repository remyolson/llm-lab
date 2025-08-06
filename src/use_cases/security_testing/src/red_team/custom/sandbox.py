"""
Sandbox environment for safe custom attack development and testing.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .attack_builder import AttackDefinition

logger = logging.getLogger(__name__)


class SandboxMode(Enum):
    """Sandbox execution modes."""

    STRICT = "strict"  # Maximum safety, no real execution
    MODERATE = "moderate"  # Some execution with safety checks
    PERMISSIVE = "permissive"  # More permissive, still safe
    DEBUG = "debug"  # Debug mode with detailed logging


class ValidationLevel(Enum):
    """Validation strictness levels."""

    BASIC = "basic"  # Basic syntax and structure
    STANDARD = "standard"  # Standard validation
    COMPREHENSIVE = "comprehensive"  # Full validation with testing


@dataclass
class SandboxConfig:
    """Configuration for attack sandbox."""

    mode: SandboxMode = SandboxMode.STRICT
    validation_level: ValidationLevel = ValidationLevel.STANDARD

    # Safety settings
    max_prompt_length: int = 2000
    max_execution_time_seconds: int = 30
    allow_external_calls: bool = False
    sanitize_outputs: bool = True

    # Testing settings
    test_iterations: int = 3
    mock_model_responses: bool = True
    record_interactions: bool = True

    # Restrictions
    blocked_keywords: List[str] = field(
        default_factory=lambda: [
            "rm -rf",
            "del /f",
            "format",
            "sudo",
            "admin",
            "password",
            "credential",
            "secret",
            "key",
        ]
    )

    blocked_patterns: List[str] = field(
        default_factory=lambda: [
            r"https?://[^\s]+",  # URLs
            r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",  # IP addresses
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Emails
        ]
    )

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_directory: str = "sandbox_logs"


@dataclass
class ValidationResult:
    """Result of attack validation in sandbox."""

    attack_id: str
    valid: bool
    validation_level: ValidationLevel

    # Validation details
    syntax_valid: bool = True
    structure_valid: bool = True
    safety_valid: bool = True
    execution_valid: bool = True

    # Issues found
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    # Test results
    test_executions: List[Dict[str, Any]] = field(default_factory=list)
    success_rate: float = 0.0

    # Metadata
    validation_time: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0

    def add_error(self, error: str):
        """Add validation error."""
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str):
        """Add validation warning."""
        self.warnings.append(warning)

    def add_suggestion(self, suggestion: str):
        """Add improvement suggestion."""
        self.suggestions.append(suggestion)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "attack_id": self.attack_id,
            "valid": self.valid,
            "validation_level": self.validation_level.value,
            "details": {
                "syntax_valid": self.syntax_valid,
                "structure_valid": self.structure_valid,
                "safety_valid": self.safety_valid,
                "execution_valid": self.execution_valid,
            },
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "test_results": {
                "executions": len(self.test_executions),
                "success_rate": self.success_rate,
            },
            "metadata": {
                "validation_time": self.validation_time.isoformat(),
                "duration_seconds": self.duration_seconds,
            },
        }


class AttackSandbox:
    """
    Secure sandbox environment for testing custom attacks.
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        """
        Initialize the attack sandbox.

        Args:
            config: Sandbox configuration
        """
        self.config = config or SandboxConfig()

        # Execution tracking
        self._execution_history: List[Dict[str, Any]] = []
        self._mock_responses: Dict[str, str] = self._initialize_mock_responses()

        # Safety checks
        self._safety_validator = SafetyValidator(self.config)

        logger.info("AttackSandbox initialized in %s mode", self.config.mode.value)

    def _initialize_mock_responses(self) -> Dict[str, str]:
        """Initialize mock model responses for testing."""
        return {
            "default": "This is a mock response for testing purposes.",
            "success": "Here is the requested information: [MOCK_DATA]",
            "failure": "I cannot provide that information due to security policies.",
            "error": "An error occurred while processing your request.",
            "blocked": "This request has been blocked for safety reasons.",
        }

    async def validate_attack(
        self, definition: AttackDefinition, model_interface: Optional[Callable[[str], str]] = None
    ) -> ValidationResult:
        """
        Validate a custom attack in the sandbox.

        Args:
            definition: Attack definition to validate
            model_interface: Optional model interface for testing

        Returns:
            ValidationResult with detailed findings
        """
        start_time = datetime.now()

        result = ValidationResult(
            attack_id=definition.attack_id,
            valid=True,
            validation_level=self.config.validation_level,
        )

        # Level 1: Basic validation
        if self.config.validation_level >= ValidationLevel.BASIC:
            self._validate_syntax(definition, result)
            self._validate_structure(definition, result)

        # Level 2: Standard validation
        if self.config.validation_level >= ValidationLevel.STANDARD:
            self._validate_safety(definition, result)
            self._validate_parameters(definition, result)

        # Level 3: Comprehensive validation
        if self.config.validation_level >= ValidationLevel.COMPREHENSIVE:
            await self._validate_execution(definition, result, model_interface)
            await self._run_test_cases(definition, result, model_interface)

        # Calculate duration
        end_time = datetime.now()
        result.duration_seconds = (end_time - start_time).total_seconds()

        # Log validation
        self._log_validation(definition, result)

        return result

    def _validate_syntax(self, definition: AttackDefinition, result: ValidationResult):
        """Validate attack syntax."""
        # Check prompt template syntax
        try:
            # Check for proper placeholder formatting
            placeholders = re.findall(r"\{(\w+)\}", definition.prompt_template)
            param_names = [p.name for p in definition.parameters]

            for placeholder in placeholders:
                if placeholder not in param_names:
                    result.add_warning(f"Placeholder '{{{placeholder}}}' not defined in parameters")

            # Check for unclosed placeholders
            if definition.prompt_template.count("{") != definition.prompt_template.count("}"):
                result.add_error("Unmatched braces in prompt template")
                result.syntax_valid = False

        except Exception as e:
            result.add_error(f"Syntax validation error: {str(e)}")
            result.syntax_valid = False

    def _validate_structure(self, definition: AttackDefinition, result: ValidationResult):
        """Validate attack structure."""
        # Check required fields
        if not definition.name:
            result.add_error("Attack name is required")
            result.structure_valid = False

        if not definition.prompt_template:
            result.add_error("Prompt template is required")
            result.structure_valid = False

        if not definition.attack_type:
            result.add_error("Attack type is required")
            result.structure_valid = False

        # Check field lengths
        if len(definition.prompt_template) > self.config.max_prompt_length:
            result.add_error(
                f"Prompt template exceeds maximum length ({self.config.max_prompt_length})"
            )
            result.structure_valid = False

        # Validate patterns
        if not definition.success_patterns and not definition.failure_patterns:
            result.add_warning(
                "No success or failure patterns defined - validation may be unreliable"
            )

        # Check metadata
        if not definition.author:
            result.add_warning("Author information missing")

        if not definition.organization:
            result.add_warning("Organization information missing")

    def _validate_safety(self, definition: AttackDefinition, result: ValidationResult):
        """Validate attack safety."""
        is_safe, safety_issues = self._safety_validator.check_safety(definition)

        if not is_safe:
            result.safety_valid = False
            for issue in safety_issues:
                result.add_error(f"Safety violation: {issue}")

        # Check severity alignment
        if definition.severity == "critical" and len(safety_issues) > 0:
            result.add_warning("Critical severity attack contains potential safety issues")

    def _validate_parameters(self, definition: AttackDefinition, result: ValidationResult):
        """Validate attack parameters."""
        for param in definition.parameters:
            # Check parameter definition
            if not param.name or not param.type:
                result.add_error(f"Invalid parameter definition: {param}")
                continue

            # Check default values for required parameters
            if param.required and param.default is None:
                result.add_warning(f"Required parameter '{param.name}' has no default value")

            # Validate default against validation rules
            if param.default is not None:
                if not param.validate(param.default):
                    result.add_error(f"Default value for '{param.name}' fails validation")

    async def _validate_execution(
        self,
        definition: AttackDefinition,
        result: ValidationResult,
        model_interface: Optional[Callable[[str], str]] = None,
    ):
        """Validate attack execution."""
        # Use mock or real model interface
        if self.config.mock_model_responses or not model_interface:
            model_interface = self._create_mock_model()

        try:
            # Generate test prompt
            test_params = self._generate_test_parameters(definition)
            test_prompt = definition.generate_prompt(**test_params)

            # Execute with timeout
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, model_interface, test_prompt),
                timeout=self.config.max_execution_time_seconds,
            )

            # Check response
            if response:
                result.execution_valid = True

                # Check success/failure patterns
                success_match = any(pattern in response for pattern in definition.success_patterns)
                failure_match = any(pattern in response for pattern in definition.failure_patterns)

                if not success_match and not failure_match:
                    result.add_warning("Response doesn't match any success or failure patterns")
            else:
                result.add_error("Empty response from model")
                result.execution_valid = False

        except asyncio.TimeoutError:
            result.add_error(f"Execution timeout ({self.config.max_execution_time_seconds}s)")
            result.execution_valid = False

        except Exception as e:
            result.add_error(f"Execution error: {str(e)}")
            result.execution_valid = False

    async def _run_test_cases(
        self,
        definition: AttackDefinition,
        result: ValidationResult,
        model_interface: Optional[Callable[[str], str]] = None,
    ):
        """Run test cases for the attack."""
        if not definition.test_cases:
            result.add_suggestion("Consider adding test cases for validation")
            return

        # Use mock or real model interface
        if self.config.mock_model_responses or not model_interface:
            model_interface = self._create_mock_model()

        successful_tests = 0

        for i, test_case in enumerate(definition.test_cases[: self.config.test_iterations]):
            try:
                # Run test
                test_result = await self._run_single_test(definition, test_case, model_interface)

                result.test_executions.append(test_result)

                if test_result.get("success", False):
                    successful_tests += 1

            except Exception as e:
                result.add_warning(f"Test case {i} failed: {str(e)}")

        # Calculate success rate
        if result.test_executions:
            result.success_rate = successful_tests / len(result.test_executions)

    async def _run_single_test(
        self,
        definition: AttackDefinition,
        test_case: Dict[str, Any],
        model_interface: Callable[[str], str],
    ) -> Dict[str, Any]:
        """Run a single test case."""
        # Generate prompt with test parameters
        test_params = test_case.get("parameters", {})
        prompt = definition.generate_prompt(**test_params)

        # Execute
        response = await asyncio.get_event_loop().run_in_executor(None, model_interface, prompt)

        # Check expected outcome
        expected_outcome = test_case.get("expected_outcome", "success")
        actual_outcome = self._determine_outcome(definition, response)

        return {
            "test_case": test_case.get("name", "unnamed"),
            "prompt": prompt[:200],  # Truncate for storage
            "response": response[:200],  # Truncate for storage
            "expected": expected_outcome,
            "actual": actual_outcome,
            "success": expected_outcome == actual_outcome,
        }

    def _determine_outcome(self, definition: AttackDefinition, response: str) -> str:
        """Determine test outcome from response."""
        # Check success patterns
        if any(pattern in response for pattern in definition.success_patterns):
            return "success"

        # Check failure patterns
        if any(pattern in response for pattern in definition.failure_patterns):
            return "failure"

        return "unknown"

    def _generate_test_parameters(self, definition: AttackDefinition) -> Dict[str, Any]:
        """Generate test parameters for an attack."""
        test_params = {}

        for param in definition.parameters:
            if param.default is not None:
                test_params[param.name] = param.default
            elif param.type == "string":
                test_params[param.name] = "test_value"
            elif param.type == "number":
                test_params[param.name] = 42
            elif param.type == "boolean":
                test_params[param.name] = True
            elif param.type == "list":
                test_params[param.name] = ["test1", "test2"]
            elif param.type == "dict":
                test_params[param.name] = {"key": "value"}
            else:
                test_params[param.name] = None

        return test_params

    def _create_mock_model(self) -> Callable[[str], str]:
        """Create mock model interface for testing."""

        def mock_model(prompt: str) -> str:
            # Check for patterns to determine response
            prompt_lower = prompt.lower()

            if any(word in prompt_lower for word in ["admin", "password", "secret"]):
                return self._mock_responses["blocked"]
            elif any(word in prompt_lower for word in ["error", "fail", "invalid"]):
                return self._mock_responses["error"]
            elif any(word in prompt_lower for word in ["deny", "refuse", "cannot"]):
                return self._mock_responses["failure"]
            elif any(word in prompt_lower for word in ["provide", "show", "give"]):
                return self._mock_responses["success"]
            else:
                return self._mock_responses["default"]

        return mock_model

    def _log_validation(self, definition: AttackDefinition, result: ValidationResult):
        """Log validation results."""
        if self.config.record_interactions:
            validation_record = {
                "timestamp": datetime.now().isoformat(),
                "attack_id": definition.attack_id,
                "attack_name": definition.name,
                "valid": result.valid,
                "errors": len(result.errors),
                "warnings": len(result.warnings),
                "duration": result.duration_seconds,
            }

            self._execution_history.append(validation_record)

            if self.config.log_to_file:
                # Would write to log file
                pass

        # Log summary
        if result.valid:
            logger.info(
                "Attack %s validated successfully (%.2fs)",
                definition.attack_id,
                result.duration_seconds,
            )
        else:
            logger.warning(
                "Attack %s validation failed with %d errors (%.2fs)",
                definition.attack_id,
                len(result.errors),
                result.duration_seconds,
            )

    async def test_attack_safely(
        self,
        definition: AttackDefinition,
        test_params: Dict[str, Any],
        model_interface: Callable[[str], str],
    ) -> Dict[str, Any]:
        """
        Safely test an attack with parameters.

        Args:
            definition: Attack to test
            test_params: Parameters for the attack
            model_interface: Model interface

        Returns:
            Test results
        """
        # Safety check
        is_safe, issues = self._safety_validator.check_prompt_safety(
            definition.generate_prompt(**test_params)
        )

        if not is_safe:
            return {"success": False, "error": "Safety check failed", "issues": issues}

        # Execute in sandbox
        try:
            prompt = definition.generate_prompt(**test_params)

            # Add safety wrapper if in strict mode
            if self.config.mode == SandboxMode.STRICT:
                prompt = f"[SANDBOX TEST] {prompt}"

            # Execute with timeout
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, model_interface, prompt),
                timeout=self.config.max_execution_time_seconds,
            )

            # Sanitize response if configured
            if self.config.sanitize_outputs:
                response = self._sanitize_response(response)

            return {
                "success": True,
                "prompt": prompt,
                "response": response,
                "outcome": self._determine_outcome(definition, response),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _sanitize_response(self, response: str) -> str:
        """Sanitize model response for safety."""
        # Remove potential sensitive information
        sanitized = response

        # Remove URLs
        sanitized = re.sub(r"https?://[^\s]+", "[URL_REMOVED]", sanitized)

        # Remove emails
        sanitized = re.sub(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL_REMOVED]", sanitized
        )

        # Remove potential credentials
        sanitized = re.sub(
            r"(password|token|key|secret)[\s:=]+[\w\-]+",
            "[CREDENTIAL_REMOVED]",
            sanitized,
            flags=re.IGNORECASE,
        )

        return sanitized

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get sandbox execution history."""
        return self._execution_history.copy()

    def clear_history(self):
        """Clear execution history."""
        self._execution_history.clear()
        logger.info("Sandbox history cleared")


class SafetyValidator:
    """
    Safety validation for custom attacks.
    """

    def __init__(self, config: SandboxConfig):
        """Initialize safety validator."""
        self.config = config

    def check_safety(self, definition: AttackDefinition) -> Tuple[bool, List[str]]:
        """
        Check if an attack definition is safe.

        Returns:
            Tuple of (is_safe, list_of_issues)
        """
        issues = []

        # Check prompt template
        prompt_safe, prompt_issues = self.check_prompt_safety(definition.prompt_template)
        issues.extend(prompt_issues)

        # Check for dangerous attack types in strict mode
        if self.config.mode == SandboxMode.STRICT:
            dangerous_types = ["code_execution", "privilege_escalation"]
            if definition.attack_type.value in dangerous_types:
                issues.append(
                    f"Attack type '{definition.attack_type.value}' not allowed in strict mode"
                )

        # Check severity
        if definition.severity == "critical" and self.config.mode == SandboxMode.STRICT:
            issues.append("Critical severity attacks not allowed in strict mode")

        return len(issues) == 0, issues

    def check_prompt_safety(self, prompt: str) -> Tuple[bool, List[str]]:
        """
        Check if a prompt is safe.

        Returns:
            Tuple of (is_safe, list_of_issues)
        """
        issues = []

        # Check blocked keywords
        for keyword in self.config.blocked_keywords:
            if keyword.lower() in prompt.lower():
                issues.append(f"Blocked keyword found: '{keyword}'")

        # Check blocked patterns
        for pattern in self.config.blocked_patterns:
            if re.search(pattern, prompt):
                issues.append(f"Blocked pattern found: {pattern}")

        # Check prompt length
        if len(prompt) > self.config.max_prompt_length:
            issues.append(f"Prompt exceeds maximum length ({self.config.max_prompt_length})")

        return len(issues) == 0, issues
