"""
Pytest configuration and shared fixtures for all tests.

This file provides:
- Test markers for categorizing tests
- Mock providers for testing without API keys
- Shared fixtures for common test data
- Performance tracking utilities
"""

import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from _pytest.config import Config
from _pytest.nodes import Item

# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config: Config) -> None:
    """Register custom markers for test categorization."""

    # Test category markers
    config.addinivalue_line("markers", "unit: Mark test as a unit test (fast, isolated)")
    config.addinivalue_line(
        "markers", "integration: Mark test as an integration test (may require external services)"
    )
    config.addinivalue_line("markers", "e2e: Mark test as an end-to-end test (full workflow)")
    config.addinivalue_line("markers", "benchmark: Mark test as a performance benchmark")

    # Provider-specific markers
    config.addinivalue_line("markers", "openai: Test requires OpenAI API")
    config.addinivalue_line("markers", "anthropic: Test requires Anthropic API")
    config.addinivalue_line("markers", "google: Test requires Google API")

    # Resource requirement markers
    config.addinivalue_line("markers", "slow: Mark test as slow (> 5 seconds)")
    config.addinivalue_line("markers", "expensive: Test consumes significant API quota")
    config.addinivalue_line("markers", "requires_gpu: Test requires GPU resources")
    config.addinivalue_line("markers", "requires_network: Test requires network access")

    # Environment markers
    config.addinivalue_line("markers", "ci_only: Test should only run in CI environment")
    config.addinivalue_line("markers", "local_only: Test should only run locally")


def pytest_collection_modifyitems(config: Config, items: List[Item]) -> None:
    """Modify test collection based on markers and environment."""

    # Skip expensive tests by default unless explicitly requested
    if not config.getoption("--run-expensive"):
        skip_expensive = pytest.mark.skip(reason="need --run-expensive option to run")
        for item in items:
            if "expensive" in item.keywords:
                item.add_marker(skip_expensive)

    # Skip slow tests in quick mode
    if config.getoption("--quick"):
        skip_slow = pytest.mark.skip(reason="skipping slow tests in quick mode")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Handle CI-only and local-only tests
    is_ci = os.environ.get("CI", "false").lower() == "true"
    for item in items:
        if "ci_only" in item.keywords and not is_ci:
            item.add_marker(pytest.mark.skip(reason="test only runs in CI"))
        if "local_only" in item.keywords and is_ci:
            item.add_marker(pytest.mark.skip(reason="test only runs locally"))


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-expensive",
        action="store_true",
        default=False,
        help="Run expensive tests that consume API quota",
    )
    parser.addoption(
        "--quick", action="store_true", default=False, help="Run only quick tests (skip slow tests)"
    )
    parser.addoption(
        "--benchmark-save",
        action="store",
        default=None,
        help="Save benchmark results to specified file",
    )


# =============================================================================
# Mock Providers
# =============================================================================


class MockLLMProvider:
    """Base mock provider for testing without API keys."""

    def __init__(self, name: str = "mock", model: str = "mock-model"):
        self.name = name
        self.model = model
        self.call_count = 0
        self.last_prompt = None
        self.response_delay = 0.1  # Simulate API latency

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a mock response."""
        self.call_count += 1
        self.last_prompt = prompt

        # Simulate processing time
        time.sleep(self.response_delay)

        # Generate deterministic response based on prompt
        response = f"Mock response from {self.name} for: {prompt[:50]}..."
        return response

    def generate_streaming(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Generate a mock streaming response."""
        self.call_count += 1
        self.last_prompt = prompt

        response = self.generate(prompt, **kwargs)
        # Simulate streaming by yielding words
        for word in response.split():
            time.sleep(0.01)
            yield word + " "

    def reset(self):
        """Reset mock state."""
        self.call_count = 0
        self.last_prompt = None


@pytest.fixture
def mock_openai_provider():
    """Mock OpenAI provider for testing."""
    provider = MockLLMProvider(name="openai", model="gpt-4")
    provider.response_delay = 0.2  # OpenAI typically slower
    return provider


@pytest.fixture
def mock_anthropic_provider():
    """Mock Anthropic provider for testing."""
    provider = MockLLMProvider(name="anthropic", model="claude-3")
    provider.response_delay = 0.15
    return provider


@pytest.fixture
def mock_google_provider():
    """Mock Google provider for testing."""
    provider = MockLLMProvider(name="google", model="gemini-pro")
    provider.response_delay = 0.1
    return provider


@pytest.fixture
def mock_all_providers(mock_openai_provider, mock_anthropic_provider, mock_google_provider):
    """Dictionary of all mock providers."""
    return {
        "openai": mock_openai_provider,
        "anthropic": mock_anthropic_provider,
        "google": mock_google_provider,
    }


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate fibonacci numbers.",
        "Translate 'Hello, world!' to Spanish.",
        "What are the main causes of climate change?",
    ]


@pytest.fixture
def sample_dataset():
    """Sample dataset for benchmark testing."""
    return {
        "name": "test_dataset",
        "version": "1.0",
        "prompts": [
            {"id": "test_001", "prompt": "What is 2+2?", "expected": "4", "category": "math"},
            {
                "id": "test_002",
                "prompt": "What color is the sky?",
                "expected": "blue",
                "category": "knowledge",
            },
            {
                "id": "test_003",
                "prompt": "Complete: Hello, ...",
                "expected": "world",
                "category": "completion",
            },
        ],
    }


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "providers": {
            "openai": {
                "api_key": "test-key-123",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000,
            },
            "anthropic": {"api_key": "test-key-456", "model": "claude-3", "temperature": 0.5},
        },
        "benchmarks": {"timeout": 30, "retries": 3, "batch_size": 10},
    }


# =============================================================================
# File System Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_config_file(temp_dir, sample_config):
    """Create a temporary config file."""
    config_path = temp_dir / "config.yaml"

    import yaml

    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)

    return config_path


@pytest.fixture
def temp_dataset_file(temp_dir, sample_dataset):
    """Create a temporary dataset file."""
    dataset_path = temp_dir / "dataset.json"

    with open(dataset_path, "w") as f:
        json.dump(sample_dataset, f)

    return dataset_path


# =============================================================================
# Performance Testing Fixtures
# =============================================================================


class PerformanceTracker:
    """Track performance metrics during tests."""

    def __init__(self):
        self.metrics = []
        self.start_time = None

    def start(self):
        """Start timing."""
        self.start_time = time.perf_counter()

    def stop(self, operation: str = "operation"):
        """Stop timing and record metric."""
        if self.start_time is None:
            return

        duration = time.perf_counter() - self.start_time
        self.metrics.append(
            {"operation": operation, "duration": duration, "timestamp": time.time()}
        )
        self.start_time = None
        return duration

    @contextmanager
    def measure(self, operation: str = "operation"):
        """Context manager for measuring performance."""
        self.start()
        try:
            yield self
        finally:
            self.stop(operation)

    def get_metrics(self):
        """Get all recorded metrics."""
        return self.metrics

    def get_summary(self):
        """Get summary statistics."""
        if not self.metrics:
            return {}

        durations = [m["duration"] for m in self.metrics]
        return {
            "count": len(durations),
            "total": sum(durations),
            "mean": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
        }


@pytest.fixture
def performance_tracker():
    """Create a performance tracker for the test."""
    return PerformanceTracker()


@pytest.fixture
def benchmark_results(request, performance_tracker):
    """Save benchmark results if requested."""
    yield performance_tracker

    # Save results if --benchmark-save option is provided
    save_path = request.config.getoption("--benchmark-save")
    if save_path and performance_tracker.metrics:
        results = {
            "test": request.node.name,
            "metrics": performance_tracker.get_metrics(),
            "summary": performance_tracker.get_summary(),
        }

        # Append to existing results file
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                all_results = json.load(f)
        else:
            all_results = []

        all_results.append(results)

        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2)


# =============================================================================
# Environment Management
# =============================================================================


@pytest.fixture
def clean_env():
    """Provide a clean environment for tests."""
    original_env = os.environ.copy()

    # Remove API keys to ensure tests don't accidentally use real APIs
    api_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "AZURE_API_KEY"]

    for key in api_keys:
        os.environ.pop(key, None)

    yield os.environ

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_env(clean_env):
    """Set up mock environment variables."""
    clean_env.update(
        {
            "OPENAI_API_KEY": "mock-openai-key",
            "ANTHROPIC_API_KEY": "mock-anthropic-key",
            "GOOGLE_API_KEY": "mock-google-key",
            "LLM_LAB_ENV": "test",
        }
    )
    return clean_env


# =============================================================================
# Network Mocking
# =============================================================================


@pytest.fixture
def mock_requests():
    """Mock requests library for network calls."""
    with patch("requests.Session") as mock_session:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_response.text = '{"status": "success"}'

        mock_session.return_value.get.return_value = mock_response
        mock_session.return_value.post.return_value = mock_response

        yield mock_session


# =============================================================================
# Assertion Helpers
# =============================================================================


class AssertionHelpers:
    """Custom assertion helpers for tests."""

    @staticmethod
    def assert_response_format(response: str, min_length: int = 10):
        """Assert that a response has the expected format."""
        assert isinstance(response, str)
        assert len(response) >= min_length
        assert response.strip() == response  # No leading/trailing whitespace

    @staticmethod
    def assert_performance(duration: float, max_duration: float):
        """Assert that an operation completed within time limit."""
        assert duration <= max_duration, (
            f"Operation took {duration:.2f}s, expected <= {max_duration}s"
        )

    @staticmethod
    def assert_api_response(response: Dict[str, Any]):
        """Assert that an API response has expected structure."""
        assert "status" in response
        assert "data" in response or "error" in response


@pytest.fixture
def assert_helpers():
    """Provide assertion helpers."""
    return AssertionHelpers()
