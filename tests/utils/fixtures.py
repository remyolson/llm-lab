"""
Pytest Fixtures and Assertion Helpers

Reusable pytest fixtures and custom assertion helpers for common test patterns,
including multi-provider setups, benchmarking environments, and monitoring configurations.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from pytest_factoryboy import register

from src.types.core import APIResponse, ConfigDict, ProviderInfo
from src.types.evaluation import EvaluationResult, MetricResult

from .base import DatabaseFixture, FileSystemFixture
from .factories import (
    BenchmarkDataFactory,
    ConfigFactory,
    EvaluationResultFactory,
    MetricFactory,
    ModelParametersFactory,
    ProviderFactory,
    ResponseFactory,
)
from .mocks import (
    MockAnthropicProvider,
    MockCache,
    MockEvaluator,
    MockGoogleProvider,
    MockLocalProvider,
    MockLogger,
    MockOpenAIProvider,
    create_mock_provider,
)

# Register factories with pytest-factoryboy
register(ProviderFactory)
register(ConfigFactory)
register(ResponseFactory)
register(ModelParametersFactory)
register(EvaluationResultFactory)
register(MetricFactory)
register(BenchmarkDataFactory)


# Provider Fixtures
@pytest.fixture
def mock_openai_provider():
    """Fixture for mock OpenAI provider."""
    provider = MockOpenAIProvider(model="gpt-4")
    provider.set_response("Test response from GPT-4")
    return provider


@pytest.fixture
def mock_anthropic_provider():
    """Fixture for mock Anthropic provider."""
    provider = MockAnthropicProvider(model="claude-3-opus")
    provider.set_response("Test response from Claude")
    return provider


@pytest.fixture
def mock_google_provider():
    """Fixture for mock Google provider."""
    provider = MockGoogleProvider(model="gemini-pro")
    provider.set_response("Test response from Gemini")
    return provider


@pytest.fixture
def mock_local_provider():
    """Fixture for mock local provider."""
    provider = MockLocalProvider(model="llama-2-7b")
    provider.set_response("Test response from Llama")
    return provider


@pytest.fixture
def all_mock_providers(
    mock_openai_provider,
    mock_anthropic_provider,
    mock_google_provider,
    mock_local_provider,
):
    """Fixture providing all mock providers."""
    return {
        "openai": mock_openai_provider,
        "anthropic": mock_anthropic_provider,
        "google": mock_google_provider,
        "local": mock_local_provider,
    }


@pytest.fixture(params=["openai", "anthropic", "google", "local"])
def parametrized_mock_provider(request, all_mock_providers):
    """Parametrized fixture for testing across all providers."""
    return all_mock_providers[request.param]


# Configuration Fixtures
@pytest.fixture
def test_config():
    """Fixture for test configuration."""
    factory = ConfigFactory()
    return factory.create_valid()


@pytest.fixture
def minimal_config():
    """Fixture for minimal configuration."""
    return ConfigDict(
        {
            "providers": {
                "openai": {
                    "api_key": "test-key",
                    "model": "gpt-4",
                }
            }
        }
    )


@pytest.fixture
def full_config():
    """Fixture for full configuration with all options."""
    factory = ConfigFactory()
    config = factory.create()
    # Add all possible configuration options
    config.update(
        {
            "providers": {
                "openai": {"api_key": "sk-test", "model": "gpt-4"},
                "anthropic": {"api_key": "sk-ant-test", "model": "claude-3"},
                "google": {"api_key": "google-test", "model": "gemini-pro"},
            },
            "evaluation": {
                "methods": ["semantic_similarity", "exact_match", "fuzzy_match"],
                "threshold": 0.8,
                "sample_size": 100,
            },
            "monitoring": {
                "enabled": True,
                "log_level": "INFO",
                "metrics_interval": 60,
            },
            "cache": {
                "enabled": True,
                "ttl": 3600,
                "max_size": 1000,
            },
        }
    )
    return config


# File System Fixtures
@pytest.fixture
def temp_dir():
    """Fixture for temporary directory."""
    fixture = FileSystemFixture()
    with fixture.context() as temp_path:
        yield temp_path


@pytest.fixture
def temp_config_file(temp_dir):
    """Fixture for temporary config file."""
    config_path = temp_dir / "config.json"
    config_data = {
        "providers": {
            "openai": {
                "api_key": "test-key",
                "model": "gpt-4",
            }
        }
    }
    config_path.write_text(json.dumps(config_data, indent=2))
    return config_path


@pytest.fixture
def test_data_dir(temp_dir):
    """Fixture for test data directory with sample files."""
    data_dir = temp_dir / "test_data"
    data_dir.mkdir()

    # Create sample test files
    (data_dir / "prompts.txt").write_text("Test prompt 1\nTest prompt 2\nTest prompt 3")
    (data_dir / "responses.json").write_text(
        json.dumps(
            [
                {"prompt": "Test prompt 1", "response": "Response 1"},
                {"prompt": "Test prompt 2", "response": "Response 2"},
            ]
        )
    )
    (data_dir / "config.yaml").write_text("providers:\n  openai:\n    model: gpt-4")

    return data_dir


# API Key Fixtures
@pytest.fixture
def mock_api_keys():
    """Fixture for mocking environment variable API keys."""
    keys = {
        "OPENAI_API_KEY": "sk-test-openai-key",
        "ANTHROPIC_API_KEY": "sk-ant-test-key",
        "GOOGLE_API_KEY": "google-test-key",
    }
    with patch.dict(os.environ, keys):
        yield keys


@pytest.fixture
def api_key_fixture():
    """Fixture providing API key management."""

    class APIKeyManager:
        def __init__(self):
            self.keys = {}

        def set(self, provider: str, key: str):
            self.keys[provider] = key

        def get(self, provider: str) -> Optional[str]:
            return self.keys.get(provider)

        def set_env(self, provider: str, key: str):
            env_var = f"{provider.upper()}_API_KEY"
            os.environ[env_var] = key

        def clear_env(self, provider: str):
            env_var = f"{provider.upper()}_API_KEY"
            os.environ.pop(env_var, None)

    manager = APIKeyManager()
    yield manager

    # Cleanup
    for provider in ["openai", "anthropic", "google"]:
        manager.clear_env(provider)


# Response Fixtures
@pytest.fixture
def mock_responses():
    """Fixture for pre-configured mock responses."""
    factory = ResponseFactory()
    return {
        "success": factory.create_valid(),
        "error": factory.create_error_response(),
        "timeout": factory.create(metadata={"error": "timeout"}),
        "rate_limit": factory.create_error_response(
            error={"type": "rate_limit", "message": "Rate limit exceeded"}
        ),
    }


@pytest.fixture
def response_sequence():
    """Fixture for a sequence of responses."""
    factory = ResponseFactory()
    return [factory.create(content=f"Response {i}") for i in range(10)]


# Benchmark Data Fixtures
@pytest.fixture
def benchmark_dataset():
    """Fixture for benchmark test dataset."""
    factory = BenchmarkDataFactory()
    return factory.create_batch(100)


@pytest.fixture
def small_benchmark_dataset():
    """Fixture for small benchmark dataset."""
    factory = BenchmarkDataFactory()
    return factory.create_batch(10)


@pytest.fixture
def categorized_benchmark_data():
    """Fixture for categorized benchmark data."""
    factory = BenchmarkDataFactory()
    categories = ["reasoning", "knowledge", "creativity", "ethics"]
    return {category: factory.create_batch(25, category=category) for category in categories}


# Evaluation Fixtures
@pytest.fixture
def mock_evaluator():
    """Fixture for mock evaluator."""
    return MockEvaluator()


@pytest.fixture
def evaluation_results():
    """Fixture for sample evaluation results."""
    factory = EvaluationResultFactory()
    return {
        "openai": factory.create(provider="openai", model="gpt-4"),
        "anthropic": factory.create(provider="anthropic", model="claude-3"),
        "google": factory.create(provider="google", model="gemini-pro"),
    }


@pytest.fixture
def metric_results():
    """Fixture for sample metric results."""
    factory = MetricFactory()
    return {
        "accuracy": factory.create(name="accuracy", value=85.5, unit="percentage"),
        "latency": factory.create(name="latency", value=125.3, unit="milliseconds"),
        "cost": factory.create(name="cost", value=0.05, unit="dollars"),
    }


# Monitoring Fixtures
@pytest.fixture
def mock_logger():
    """Fixture for mock logger."""
    return MockLogger()


@pytest.fixture
def mock_cache():
    """Fixture for mock cache."""
    return MockCache()


@pytest.fixture
def monitoring_setup(mock_logger, mock_cache):
    """Fixture for complete monitoring setup."""
    return {
        "logger": mock_logger,
        "cache": mock_cache,
        "metrics": {},
        "alerts": [],
    }


# Database Fixtures
@pytest.fixture
def test_database(tmp_path):
    """Fixture for test database."""

    class TestDB(DatabaseFixture):
        def create_schema(self):
            # Simple schema creation
            import sqlite3

            self.connection = sqlite3.connect(str(self.db_path))
            cursor = self.connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY,
                    model TEXT,
                    score REAL,
                    timestamp TEXT
                )
            """)
            self.connection.commit()

        def populate_data(self):
            cursor = self.connection.cursor()
            for i in range(10):
                cursor.execute(
                    "INSERT INTO results (model, score, timestamp) VALUES (?, ?, ?)",
                    (f"model_{i}", 0.5 + i * 0.05, datetime.now().isoformat()),
                )
            self.connection.commit()

    db = TestDB(tmp_path / "test.db")
    with db.context() as connection:
        yield connection


# Parameterized Fixtures
@pytest.fixture(
    params=[
        {"temperature": 0.0},  # Deterministic
        {"temperature": 0.7},  # Default
        {"temperature": 1.5},  # Creative
    ]
)
def temperature_params(request):
    """Parametrized fixture for temperature testing."""
    return request.param


@pytest.fixture(params=[10, 100, 1000])
def batch_sizes(request):
    """Parametrized fixture for batch size testing."""
    return request.param


@pytest.fixture(params=["semantic_similarity", "exact_match", "fuzzy_match", "bleu"])
def evaluation_methods(request):
    """Parametrized fixture for evaluation methods."""
    return request.param


# Context Manager Fixtures
@pytest.fixture
def environment_context():
    """Fixture providing environment context manager."""

    @contextmanager
    def set_env(**kwargs):
        old_env = dict(os.environ)
        os.environ.update(kwargs)
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    return set_env


@pytest.fixture
def mock_time():
    """Fixture for mocking time."""
    with patch("time.time") as mock_time_func:
        mock_time_func.return_value = 1234567890.0
        yield mock_time_func


@pytest.fixture
def capture_stdout():
    """Fixture for capturing stdout."""
    import sys
    from io import StringIO

    @contextmanager
    def capture():
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            yield sys.stdout
        finally:
            sys.stdout = old_stdout

    return capture


# Cleanup Fixtures
@pytest.fixture(autouse=True)
def cleanup_temp_files(request):
    """Auto-cleanup fixture for temporary files."""
    temp_files = []

    def register_temp_file(filepath):
        temp_files.append(filepath)

    request.addfinalizer(lambda: [Path(f).unlink(missing_ok=True) for f in temp_files])

    return register_temp_file


@pytest.fixture(autouse=True)
def reset_singletons():
    """Auto-reset fixture for singleton instances."""
    yield
    # Reset any singleton instances after each test
    # This would be implemented based on your singleton patterns
