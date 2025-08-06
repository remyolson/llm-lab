"""Base validator class for benchmark validation."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..generators.base import TestCase

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation process."""

    is_valid: bool
    score: float  # 0.0 to 1.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "score": self.score,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
        }

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def update_score(self, new_score: float) -> None:
        """Update the validation score."""
        self.score = max(0.0, min(1.0, new_score))


@dataclass
class ValidationConfig:
    """Configuration for validators."""

    strict_mode: bool = True
    min_score_threshold: float = 0.7
    check_duplicates: bool = True
    check_completeness: bool = True
    check_format: bool = True
    check_difficulty: bool = True
    max_duplicate_similarity: float = 0.95
    min_question_length: int = 5
    max_question_length: int = 1000
    required_fields: List[str] = field(default_factory=lambda: ["question", "answer"])
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseValidator(ABC):
    """Abstract base class for all validators."""

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize validator.

        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()
        self.validation_history: List[ValidationResult] = []

    @abstractmethod
    def validate_single(self, test_case: TestCase) -> ValidationResult:
        """
        Validate a single test case.

        Args:
            test_case: Test case to validate

        Returns:
            Validation result
        """
        pass

    @abstractmethod
    def validate_batch(self, test_cases: List[TestCase]) -> ValidationResult:
        """
        Validate a batch of test cases.

        Args:
            test_cases: List of test cases

        Returns:
            Overall validation result
        """
        pass

    def validate(self, test_cases: Union[TestCase, List[TestCase]]) -> ValidationResult:
        """
        Validate test cases.

        Args:
            test_cases: Single test case or list of test cases

        Returns:
            Validation result
        """
        if isinstance(test_cases, TestCase):
            result = self.validate_single(test_cases)
        else:
            result = self.validate_batch(test_cases)

        self.validation_history.append(result)
        return result

    def check_completeness(self, test_case: TestCase) -> Tuple[bool, List[str]]:
        """
        Check if test case has all required fields.

        Args:
            test_case: Test case to check

        Returns:
            Tuple of (is_complete, missing_fields)
        """
        missing_fields = []

        for field in self.config.required_fields:
            value = getattr(test_case, field, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing_fields.append(field)

        return len(missing_fields) == 0, missing_fields

    def check_format(self, test_case: TestCase) -> Tuple[bool, List[str]]:
        """
        Check test case format.

        Args:
            test_case: Test case to check

        Returns:
            Tuple of (is_valid_format, format_issues)
        """
        issues = []

        # Check question length
        if test_case.question:
            q_len = len(test_case.question)
            if q_len < self.config.min_question_length:
                issues.append(f"Question too short ({q_len} chars)")
            elif q_len > self.config.max_question_length:
                issues.append(f"Question too long ({q_len} chars)")

        # Check answer format
        if test_case.choices and test_case.answer:
            if isinstance(test_case.answer, str):
                if test_case.answer not in test_case.choices:
                    issues.append("Answer not in choices")
            elif isinstance(test_case.answer, list):
                invalid = [a for a in test_case.answer if a not in test_case.choices]
                if invalid:
                    issues.append(f"Invalid answers: {invalid}")

        # Check for proper punctuation
        if test_case.question and not test_case.question.rstrip().endswith(("?", ".", ":", "!")):
            issues.append("Question lacks proper punctuation")

        return len(issues) == 0, issues

    def check_difficulty_distribution(
        self, test_cases: List[TestCase], expected_distribution: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check difficulty distribution of test cases.

        Args:
            test_cases: List of test cases
            expected_distribution: Expected distribution

        Returns:
            Tuple of (is_balanced, actual_distribution)
        """
        if not test_cases:
            return False, {}

        # Count difficulties
        difficulty_counts = {}
        for case in test_cases:
            difficulty = case.difficulty or "unspecified"
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

        # Calculate distribution
        total = len(test_cases)
        actual_distribution = {k: v / total for k, v in difficulty_counts.items()}

        # Check against expected if provided
        is_balanced = True
        if expected_distribution:
            tolerance = 0.1  # 10% tolerance
            for difficulty, expected_prop in expected_distribution.items():
                actual_prop = actual_distribution.get(difficulty, 0)
                if abs(actual_prop - expected_prop) > tolerance:
                    is_balanced = False
                    break

        return is_balanced, actual_distribution

    def calculate_diversity_score(self, test_cases: List[TestCase]) -> float:
        """
        Calculate diversity score for test cases.

        Args:
            test_cases: List of test cases

        Returns:
            Diversity score (0.0 to 1.0)
        """
        if len(test_cases) < 2:
            return 1.0

        # Simple diversity based on unique questions
        unique_questions = set()
        unique_domains = set()
        unique_types = set()

        for case in test_cases:
            # Normalize question for comparison
            q_normalized = case.question.lower().strip() if case.question else ""
            unique_questions.add(q_normalized)

            if case.domain:
                unique_domains.add(case.domain)

            case_type = case.metadata.get("type", "unknown")
            unique_types.add(case_type)

        # Calculate diversity metrics
        question_diversity = len(unique_questions) / len(test_cases)
        domain_diversity = len(unique_domains) / max(
            1, len(set(c.domain for c in test_cases if c.domain))
        )
        type_diversity = len(unique_types) / max(
            1, len(set(c.metadata.get("type") for c in test_cases))
        )

        # Weighted average
        diversity_score = 0.5 * question_diversity + 0.25 * domain_diversity + 0.25 * type_diversity

        return diversity_score

    def detect_duplicates(
        self, test_cases: List[TestCase], similarity_threshold: Optional[float] = None
    ) -> List[Tuple[int, int, float]]:
        """
        Detect duplicate or near-duplicate test cases.

        Args:
            test_cases: List of test cases
            similarity_threshold: Similarity threshold for duplicates

        Returns:
            List of (index1, index2, similarity) tuples
        """
        threshold = similarity_threshold or self.config.max_duplicate_similarity
        duplicates = []

        for i in range(len(test_cases)):
            for j in range(i + 1, len(test_cases)):
                similarity = self._calculate_similarity(test_cases[i], test_cases[j])
                if similarity >= threshold:
                    duplicates.append((i, j, similarity))

        return duplicates

    def _calculate_similarity(self, case1: TestCase, case2: TestCase) -> float:
        """
        Calculate similarity between two test cases.

        Args:
            case1: First test case
            case2: Second test case

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Simple similarity based on question text
        if not case1.question or not case2.question:
            return 0.0

        # Normalize questions
        q1 = case1.question.lower().strip()
        q2 = case2.question.lower().strip()

        # Exact match
        if q1 == q2:
            return 1.0

        # Calculate Jaccard similarity
        words1 = set(q1.split())
        words2 = set(q2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        jaccard_sim = len(intersection) / len(union)

        # Also check answer similarity
        answer_sim = 0.0
        if case1.answer == case2.answer:
            answer_sim = 1.0

        # Weighted combination
        return 0.7 * jaccard_sim + 0.3 * answer_sim

    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all validations performed.

        Returns:
            Summary dictionary
        """
        if not self.validation_history:
            return {"total_validations": 0}

        total = len(self.validation_history)
        valid_count = sum(1 for r in self.validation_history if r.is_valid)
        avg_score = sum(r.score for r in self.validation_history) / total

        all_errors = []
        all_warnings = []
        for result in self.validation_history:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

        return {
            "total_validations": total,
            "valid_count": valid_count,
            "valid_percentage": (valid_count / total) * 100,
            "average_score": avg_score,
            "total_errors": len(all_errors),
            "total_warnings": len(all_warnings),
            "unique_errors": len(set(all_errors)),
            "unique_warnings": len(set(all_warnings)),
        }

    def clear_history(self) -> None:
        """Clear validation history."""
        self.validation_history = []
        logger.info("Cleared validation history")
