"""
Base Classes and Interfaces for Test Utilities

This module provides abstract base classes and interfaces that define
the contract for test utilities, ensuring consistency across all test helpers.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar

from src.types.protocols import DataType, T


class TestFactory(ABC, Generic[T]):
    """Abstract base class for test data factories."""

    @abstractmethod
    def create(self, **kwargs) -> T:
        """Create a single instance with optional overrides."""
        pass

    @abstractmethod
    def create_batch(self, count: int, **kwargs) -> List[T]:
        """Create multiple instances."""
        pass

    @abstractmethod
    def create_valid(self) -> T:
        """Create a valid instance with sensible defaults."""
        pass

    @abstractmethod
    def create_invalid(self) -> T:
        """Create an invalid instance for error testing."""
        pass

    @abstractmethod
    def create_edge_case(self) -> T:
        """Create an edge case instance for boundary testing."""
        pass


class MockProvider(ABC):
    """Abstract base class for mock providers."""

    def __init__(self):
        self.call_count = 0
        self.last_request = None
        self.responses = []
        self.should_fail = False
        self.failure_message = "Mock failure"
        self.delay = 0.0

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a mock response."""
        pass

    def set_response(self, response: str | List[str]):
        """Set the response(s) to return."""
        if isinstance(response, str):
            self.responses = [response]
        else:
            self.responses = response

    def set_failure(self, should_fail: bool = True, message: str = "Mock failure"):
        """Configure the mock to fail."""
        self.should_fail = should_fail
        self.failure_message = message

    def reset(self):
        """Reset the mock to initial state."""
        self.call_count = 0
        self.last_request = None
        self.responses = []
        self.should_fail = False
        self.delay = 0.0


class TestAssertion(Protocol):
    """Protocol for custom assertion functions."""

    def __call__(self, actual: Any, expected: Any, message: Optional[str] = None) -> None:
        """Perform assertion."""
        ...


@dataclass
class TestScenario:
    """Represents a test scenario with input and expected output."""

    name: str
    description: str
    input_data: Dict[str, Any]
    expected_output: Any
    expected_error: Optional[type[Exception]] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"TestScenario({self.name})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "input_data": self.input_data,
            "expected_output": self.expected_output,
            "expected_error": str(self.expected_error) if self.expected_error else None,
            "tags": self.tags,
            "metadata": self.metadata,
        }


class TestFixture(ABC):
    """Abstract base class for test fixtures."""

    @abstractmethod
    def setup(self) -> Any:
        """Set up the fixture."""
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Tear down the fixture."""
        pass

    @contextmanager
    def context(self):
        """Context manager for fixture lifecycle."""
        result = None
        try:
            result = self.setup()
            yield result
        finally:
            self.teardown()


class DatabaseFixture(TestFixture):
    """Base class for database test fixtures."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(tempfile.mktemp(suffix=".db"))
        self.connection = None

    @abstractmethod
    def create_schema(self) -> None:
        """Create database schema."""
        pass

    @abstractmethod
    def populate_data(self) -> None:
        """Populate test data."""
        pass

    def setup(self) -> Any:
        """Set up database fixture."""
        self.create_schema()
        self.populate_data()
        return self.connection

    def teardown(self) -> None:
        """Clean up database fixture."""
        if self.connection:
            self.connection.close()
        if self.db_path.exists():
            self.db_path.unlink()


class FileSystemFixture(TestFixture):
    """Base class for file system test fixtures."""

    def __init__(self):
        self.temp_dir = None

    def setup(self) -> Path:
        """Create temporary directory."""
        self.temp_dir = Path(tempfile.mkdtemp())
        return self.temp_dir

    def teardown(self) -> None:
        """Remove temporary directory."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def create_file(self, name: str, content: str) -> Path:
        """Create a file in the temp directory."""
        if not self.temp_dir:
            raise RuntimeError("Fixture not set up")
        file_path = self.temp_dir / name
        file_path.write_text(content)
        return file_path

    def create_json_file(self, name: str, data: Dict[str, Any]) -> Path:
        """Create a JSON file in the temp directory."""
        if not self.temp_dir:
            raise RuntimeError("Fixture not set up")
        file_path = self.temp_dir / name
        file_path.write_text(json.dumps(data, indent=2))
        return file_path


class TestPlugin(ABC):
    """Abstract base class for pytest plugins."""

    @abstractmethod
    def pytest_configure(self, config):
        """Configure the plugin."""
        pass

    def pytest_collection_modifyitems(self, items):
        """Modify collected test items."""
        pass

    def pytest_runtest_setup(self, item):
        """Called before running a test."""
        pass

    def pytest_runtest_teardown(self, item):
        """Called after running a test."""
        pass

    def pytest_runtest_makereport(self, item, call):
        """Create test report."""
        pass


@dataclass
class TestMetrics:
    """Container for test execution metrics."""

    test_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    memory_usage: Optional[float] = None
    api_calls: int = 0
    tokens_used: int = 0
    cost: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def calculate_duration(self) -> float:
        """Calculate test duration."""
        if self.end_time and self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
        return self.duration or 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "memory_usage": self.memory_usage,
            "api_calls": self.api_calls,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "errors": self.errors,
            "warnings": self.warnings,
            "custom_metrics": self.custom_metrics,
        }


class TestDataGenerator(ABC, Generic[DataType]):
    """Abstract base class for test data generators."""

    @abstractmethod
    def generate(self, seed: Optional[int] = None) -> DataType:
        """Generate test data with optional seed for reproducibility."""
        pass

    @abstractmethod
    def generate_valid(self) -> DataType:
        """Generate valid test data."""
        pass

    @abstractmethod
    def generate_invalid(self) -> DataType:
        """Generate invalid test data for error testing."""
        pass

    @abstractmethod
    def generate_edge_case(self) -> DataType:
        """Generate edge case test data."""
        pass

    def generate_batch(self, count: int, seed: Optional[int] = None) -> List[DataType]:
        """Generate multiple test data instances."""
        return [self.generate(seed=seed + i if seed else None) for i in range(count)]


class TestValidator:
    """Base class for test result validation."""

    @staticmethod
    def validate_response_format(response: Any, expected_format: Dict[str, type]) -> bool:
        """Validate response matches expected format."""
        if not isinstance(response, dict):
            return False

        for key, expected_type in expected_format.items():
            if key not in response:
                return False
            if not isinstance(response[key], expected_type):
                return False

        return True

    @staticmethod
    def validate_numeric_range(value: float, min_val: float, max_val: float) -> bool:
        """Validate numeric value is within range."""
        return min_val <= value <= max_val

    @staticmethod
    def validate_string_pattern(value: str, pattern: str) -> bool:
        """Validate string matches pattern."""
        import re

        return bool(re.match(pattern, value))

    @staticmethod
    def validate_list_length(
        lst: List[Any], min_length: int, max_length: Optional[int] = None
    ) -> bool:
        """Validate list length constraints."""
        if len(lst) < min_length:
            return False
        if max_length is not None and len(lst) > max_length:
            return False
        return True
