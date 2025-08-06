"""
Provider Test Fixtures

Reusable fixtures for provider-related tests.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest


@pytest.fixture
def mock_openai_provider():
    """Create a mock OpenAI provider."""
    provider = Mock()
    provider.model_name = "gpt-4"
    provider.provider_name = "openai"
    provider.generate.return_value = "Mock OpenAI response"
    provider.get_model_info.return_value = {
        "model_name": "gpt-4",
        "provider": "openai",
        "max_tokens": 8192,
        "capabilities": ["text_generation", "chat"],
    }
    provider.validate_credentials.return_value = True
    return provider


@pytest.fixture
def mock_anthropic_provider():
    """Create a mock Anthropic provider."""
    provider = Mock()
    provider.model_name = "claude-3-opus"
    provider.provider_name = "anthropic"
    provider.generate.return_value = "Mock Anthropic response"
    provider.get_model_info.return_value = {
        "model_name": "claude-3-opus",
        "provider": "anthropic",
        "max_tokens": 200000,
        "capabilities": ["text_generation", "analysis"],
    }
    provider.validate_credentials.return_value = True
    return provider


@pytest.fixture
def mock_google_provider():
    """Create a mock Google provider."""
    provider = Mock()
    provider.model_name = "gemini-pro"
    provider.provider_name = "google"
    provider.generate.return_value = "Mock Google response"
    provider.get_model_info.return_value = {
        "model_name": "gemini-pro",
        "provider": "google",
        "max_tokens": 32768,
        "capabilities": ["text_generation", "multimodal"],
    }
    provider.validate_credentials.return_value = True
    return provider


@pytest.fixture
def provider_config():
    """Create a standard provider configuration."""
    return {
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.9,
        "timeout": 30,
        "max_retries": 3,
        "retry_delay": 1.0,
    }


@pytest.fixture
def all_providers(mock_openai_provider, mock_anthropic_provider, mock_google_provider):
    """Get all mock providers."""
    return [mock_openai_provider, mock_anthropic_provider, mock_google_provider]


@pytest.fixture
def provider_error_scenarios():
    """Common provider error scenarios for testing."""
    return [
        {
            "name": "rate_limit",
            "exception": Exception("Rate limit exceeded"),
            "retry": True,
        },
        {
            "name": "invalid_credentials",
            "exception": Exception("Invalid API key"),
            "retry": False,
        },
        {
            "name": "timeout",
            "exception": TimeoutError("Request timed out"),
            "retry": True,
        },
        {
            "name": "network_error",
            "exception": ConnectionError("Network error"),
            "retry": True,
        },
    ]


@pytest.fixture
def mock_provider_factory():
    """Factory for creating mock providers with specific behaviors."""

    def _create_provider(
        provider_name: str = "test",
        model_name: str = "test-model",
        responses: List[str] = None,
        should_fail: bool = False,
        failure_exception: Exception = None,
    ):
        provider = Mock()
        provider.provider_name = provider_name
        provider.model_name = model_name

        if should_fail:
            provider.generate.side_effect = failure_exception or Exception("Provider error")
        elif responses:
            provider.generate.side_effect = responses
        else:
            provider.generate.return_value = f"Mock {provider_name} response"

        provider.validate_credentials.return_value = not should_fail
        return provider

    return _create_provider


@pytest.fixture(scope="session")
def provider_test_suite():
    """Comprehensive test suite configuration for providers."""
    return {
        "test_prompts": [
            "What is 2+2?",
            "Translate 'hello' to French",
            "Write a haiku about testing",
        ],
        "expected_patterns": {
            "math": r"\b4\b|\bfour\b",
            "translation": r"bonjour|salut",
            "haiku": r"\d+\s+syllables?|\bhaiku\b",
        },
        "performance_thresholds": {
            "response_time_ms": 5000,
            "tokens_per_second": 10,
            "max_memory_mb": 500,
        },
    }
