"""Base generator class for benchmark test case generation."""

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Represents a single test case in a benchmark."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    answer: Any = None
    choices: Optional[List[str]] = None
    context: Optional[str] = None
    domain: Optional[str] = None
    difficulty: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert test case to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "choices": self.choices,
            "context": self.context,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        """Create test case from dictionary."""
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class GeneratorConfig:
    """Configuration for test case generators."""

    count: int = 100
    domain: Optional[str] = None
    difficulty_distribution: Dict[str, float] = field(
        default_factory=lambda: {"easy": 0.3, "medium": 0.5, "hard": 0.2}
    )
    task_types: List[str] = field(
        default_factory=lambda: ["multiple_choice", "true_false", "open_ended"]
    )
    include_context: bool = False
    random_seed: Optional[int] = None
    batch_size: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseGenerator(ABC):
    """Abstract base class for all test case generators."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """
        Initialize the generator.

        Args:
            config: Generator configuration
        """
        self.config = config or GeneratorConfig()
        self.generated_cases: List[TestCase] = []
        self._setup()

    def _setup(self) -> None:
        """Setup generator resources."""
        if self.config.random_seed is not None:
            import random

            import numpy as np

            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

    @abstractmethod
    def generate_single(self) -> TestCase:
        """
        Generate a single test case.

        Returns:
            Generated test case
        """
        pass

    @abstractmethod
    def validate_case(self, test_case: TestCase) -> bool:
        """
        Validate a generated test case.

        Args:
            test_case: Test case to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    def generate_batch(self, batch_size: Optional[int] = None) -> List[TestCase]:
        """
        Generate a batch of test cases.

        Args:
            batch_size: Number of cases to generate

        Returns:
            List of generated test cases
        """
        batch_size = batch_size or self.config.batch_size
        batch = []

        for _ in range(batch_size):
            try:
                test_case = self.generate_single()
                if self.validate_case(test_case):
                    batch.append(test_case)
                    self.generated_cases.append(test_case)
                else:
                    logger.warning(f"Generated invalid test case: {test_case.id}")
            except Exception as e:
                logger.error(f"Error generating test case: {e}")

        return batch

    def generate(self, count: Optional[int] = None) -> List[TestCase]:
        """
        Generate multiple test cases.

        Args:
            count: Number of test cases to generate

        Returns:
            List of generated test cases
        """
        count = count or self.config.count
        all_cases = []

        logger.info(f"Generating {count} test cases...")

        while len(all_cases) < count:
            remaining = count - len(all_cases)
            batch_size = min(self.config.batch_size, remaining)
            batch = self.generate_batch(batch_size)
            all_cases.extend(batch)

            logger.info(f"Generated {len(all_cases)}/{count} test cases")

        return all_cases

    def ensure_diversity(self, test_cases: List[TestCase]) -> List[TestCase]:
        """
        Ensure diversity in generated test cases.

        Args:
            test_cases: List of test cases

        Returns:
            Filtered list with improved diversity
        """
        # Simple diversity check - can be overridden in subclasses
        seen_questions = set()
        diverse_cases = []

        for case in test_cases:
            # Simple duplicate check
            question_key = case.question.lower().strip()
            if question_key not in seen_questions:
                seen_questions.add(question_key)
                diverse_cases.append(case)

        return diverse_cases

    def apply_difficulty_distribution(self, test_cases: List[TestCase]) -> List[TestCase]:
        """
        Apply difficulty distribution to test cases.

        Args:
            test_cases: List of test cases

        Returns:
            Test cases with assigned difficulties
        """
        import numpy as np

        total = len(test_cases)
        distribution = self.config.difficulty_distribution

        # Calculate counts for each difficulty
        counts = {level: int(total * proportion) for level, proportion in distribution.items()}

        # Adjust for rounding
        diff = total - sum(counts.values())
        if diff > 0:
            counts[list(counts.keys())[0]] += diff

        # Assign difficulties
        difficulty_assignments = []
        for level, count in counts.items():
            difficulty_assignments.extend([level] * count)

        # Shuffle assignments
        np.random.shuffle(difficulty_assignments)

        # Apply to test cases
        for case, difficulty in zip(test_cases, difficulty_assignments):
            case.difficulty = difficulty

        return test_cases

    def generate_edge_cases(self) -> List[TestCase]:
        """
        Generate edge cases for testing.

        Returns:
            List of edge case test cases
        """
        # Default implementation - should be overridden in subclasses
        edge_cases = []

        # Example edge cases
        edge_templates = [
            {"question": "", "answer": None},  # Empty question
            {"question": "A" * 10000, "answer": "Long input"},  # Very long question
            {"question": "Test?", "answer": ""},  # Empty answer
        ]

        for template in edge_templates:
            case = TestCase(
                question=template["question"],
                answer=template["answer"],
                metadata={"type": "edge_case"},
            )
            if self.validate_case(case):
                edge_cases.append(case)

        return edge_cases

    def export_cases(
        self, test_cases: Optional[List[TestCase]] = None, format: str = "json"
    ) -> Union[str, Dict, List]:
        """
        Export test cases in specified format.

        Args:
            test_cases: Test cases to export (uses generated if None)
            format: Export format (json, dict, list)

        Returns:
            Exported data in specified format
        """
        cases = test_cases or self.generated_cases

        if format == "dict":
            return [case.to_dict() for case in cases]
        elif format == "json":
            import json

            return json.dumps([case.to_dict() for case in cases], indent=2)
        elif format == "list":
            return cases
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about generated test cases.

        Returns:
            Dictionary of statistics
        """
        if not self.generated_cases:
            return {"total": 0}

        stats = {
            "total": len(self.generated_cases),
            "domains": {},
            "difficulties": {},
            "types": {},
        }

        for case in self.generated_cases:
            # Count domains
            domain = case.domain or "unspecified"
            stats["domains"][domain] = stats["domains"].get(domain, 0) + 1

            # Count difficulties
            difficulty = case.difficulty or "unspecified"
            stats["difficulties"][difficulty] = stats["difficulties"].get(difficulty, 0) + 1

            # Count types from metadata
            case_type = case.metadata.get("type", "unspecified")
            stats["types"][case_type] = stats["types"].get(case_type, 0) + 1

        return stats

    def clear(self) -> None:
        """Clear generated test cases."""
        self.generated_cases = []
        logger.info("Cleared all generated test cases")
