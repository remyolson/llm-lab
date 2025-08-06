"""
Test suite for AnthropicProvider

This module contains comprehensive unit and integration tests for the Anthropic
provider implementation. It tests authentication, message format conversion,
API calls, error handling, and retry logic.
"""

import os
from typing import Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.providers.anthropic import ANTHROPIC_AVAILABLE, AnthropicProvider
from src.providers.exceptions import (
    InvalidCredentialsError,
    ProviderConfigurationError,
    ProviderResponseError,
    ProviderTimeoutError,
    RateLimitError,
)


class TestAnthropicProvider:
    """Test suite for AnthropicProvider class."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Mock environment variables."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        yield
        if "ANTHROPIC_API_KEY" in os.environ:
            monkeypatch.delenv("ANTHROPIC_API_KEY")

    @pytest.fixture
    def provider(self, mock_env):
        """Create a test provider instance."""
        if not ANTHROPIC_AVAILABLE:
            pytest.skip("anthropic package not installed")
        return AnthropicProvider("claude-3-sonnet-20240229")

    def test_provider_initialization(self, mock_env):
        """Test provider initialization with various models."""
        # Test with default model
        provider = AnthropicProvider()
        assert provider.model_name == "claude-3-sonnet-20240229"
        assert provider.provider_name == "anthropic"

        # Test with specific models
        for model in ["claude-3-opus-20240229", "claude-3-haiku-20240307", "claude-2.1"]:
            provider = AnthropicProvider(model)
            assert provider.model_name == model

    def test_provider_without_anthropic_package(self, mock_env):
        """Test error when anthropic package is not available."""
        with patch("llm_providers.anthropic.ANTHROPIC_AVAILABLE", False):
            with pytest.raises(ProviderConfigurationError) as exc_info:
                AnthropicProvider()
            assert "anthropic package not installed" in str(exc_info.value)

    def test_validate_credentials_missing_api_key(self, monkeypatch):
        """Test credential validation with missing API key."""
        if "ANTHROPIC_API_KEY" in os.environ:
            monkeypatch.delenv("ANTHROPIC_API_KEY")

        provider = AnthropicProvider()
        with pytest.raises(InvalidCredentialsError) as exc_info:
            provider.validate_credentials()
        assert "ANTHROPIC_API_KEY environment variable not set" in str(exc_info.value)

    @patch("llm_providers.anthropic.anthropic.Anthropic")
    def test_validate_credentials_invalid_key(self, mock_anthropic_class, mock_env):
        """Test credential validation with invalid API key."""
        # Mock the client to raise authentication error
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock authentication error
        mock_client.messages.create.side_effect = Exception(
            "Authentication failed: Invalid API key"
        )

        provider = AnthropicProvider()
        with pytest.raises(InvalidCredentialsError) as exc_info:
            provider.validate_credentials()
        assert "Invalid API key" in str(exc_info.value)

    @patch("llm_providers.anthropic.anthropic.Anthropic")
    def test_validate_credentials_success(self, mock_anthropic_class, mock_env):
        """Test successful credential validation."""
        # Mock successful API call
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_response = Mock()
        mock_response.content = [Mock(text="Hi")]
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider()
        assert provider.validate_credentials() is True

    def test_convert_to_messages_format_simple(self, provider):
        """Test conversion of simple prompt to messages format."""
        prompt = "What is the capital of France?"
        messages = provider._convert_to_messages_format(prompt)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == prompt

    def test_convert_to_messages_format_with_markers(self, provider):
        """Test conversion of prompt with Human/Assistant markers."""
        prompt = "\n\nHuman: Hello\n\nAssistant: Hi there!\n\nHuman: How are you?"
        messages = provider._convert_to_messages_format(prompt)

        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi there!"}
        assert messages[2] == {"role": "user", "content": "How are you?"}

    @patch("llm_providers.anthropic.anthropic.Anthropic")
    def test_generate_success(self, mock_anthropic_class, provider):
        """Test successful text generation."""
        # Mock the client and response
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        provider._client = mock_client
        provider._initialized = True

        # Mock successful response
        mock_content = Mock(text="Paris is the capital of France.")
        mock_response = Mock(content=[mock_content])
        mock_client.messages.create.return_value = mock_response

        result = provider.generate("What is the capital of France?")
        assert result == "Paris is the capital of France."

        # Verify API call
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        assert call_args["model"] == "claude-3-sonnet-20240229"
        assert call_args["messages"] == [
            {"role": "user", "content": "What is the capital of France?"}
        ]
        assert call_args["max_tokens"] == 1000  # default
        assert call_args["temperature"] == 0.7  # default

    @patch("llm_providers.anthropic.anthropic.Anthropic")
    def test_generate_empty_prompt(self, mock_anthropic_class, provider):
        """Test generation with empty prompt."""
        provider._initialized = True

        with pytest.raises(ProviderResponseError) as exc_info:
            provider.generate("")
        assert "Empty prompt provided" in str(exc_info.value)

        with pytest.raises(ProviderResponseError) as exc_info:
            provider.generate("   ")
        assert "Empty prompt provided" in str(exc_info.value)

    @patch("llm_providers.anthropic.anthropic")
    def test_generate_rate_limit_with_retry(self, mock_anthropic, provider):
        """Test rate limit handling with retry logic."""
        # Create mock client
        mock_client = Mock()
        provider._client = mock_client
        provider._initialized = True

        # Create a mock RateLimitError
        mock_rate_limit_error = type("RateLimitError", (Exception,), {})
        mock_anthropic.RateLimitError = mock_rate_limit_error

        # First call raises rate limit, second succeeds
        mock_content = Mock(text="Success after retry")
        mock_response = Mock(content=[mock_content])
        mock_client.messages.create.side_effect = [
            mock_rate_limit_error("Rate limit exceeded"),
            mock_response,
        ]

        # Mock time.sleep to speed up test
        with patch("time.sleep"):
            result = provider.generate("Test prompt")

        assert result == "Success after retry"
        assert mock_client.messages.create.call_count == 2

    @patch("llm_providers.anthropic.anthropic")
    def test_generate_rate_limit_max_retries_exceeded(self, mock_anthropic, provider):
        """Test rate limit handling when max retries exceeded."""
        # Create mock client
        mock_client = Mock()
        provider._client = mock_client
        provider._initialized = True
        provider.config.max_retries = 2

        # Create a mock RateLimitError
        mock_rate_limit_error = type("RateLimitError", (Exception,), {})
        mock_anthropic.RateLimitError = mock_rate_limit_error

        # All calls raise rate limit error
        mock_client.messages.create.side_effect = mock_rate_limit_error("Rate limit exceeded")

        # Mock time.sleep to speed up test
        with patch("time.sleep"):
            with pytest.raises(RateLimitError) as exc_info:
                provider.generate("Test prompt")

        assert "API rate limit" in str(exc_info.value)
        assert mock_client.messages.create.call_count == 3  # initial + 2 retries

    @patch("llm_providers.anthropic.anthropic")
    def test_generate_timeout_error(self, mock_anthropic, provider):
        """Test timeout error handling."""
        # Create mock client
        mock_client = Mock()
        provider._client = mock_client
        provider._initialized = True

        # Create mock exception types
        mock_timeout_error = type("APITimeoutError", (Exception,), {})
        mock_rate_limit_error = type("RateLimitError", (Exception,), {})
        mock_auth_error = type("AuthenticationError", (Exception,), {})

        mock_anthropic.APITimeoutError = mock_timeout_error
        mock_anthropic.RateLimitError = mock_rate_limit_error
        mock_anthropic.AuthenticationError = mock_auth_error

        mock_client.messages.create.side_effect = mock_timeout_error("Request timed out")

        with pytest.raises(ProviderTimeoutError) as exc_info:
            provider.generate("Test prompt")

        assert "Generate" in str(exc_info.value)

    @patch("llm_providers.anthropic.anthropic")
    def test_generate_authentication_error(self, mock_anthropic, provider):
        """Test authentication error handling during generation."""
        # Create mock client
        mock_client = Mock()
        provider._client = mock_client
        provider._initialized = True

        # Create mock exception types
        mock_auth_error = type("AuthenticationError", (Exception,), {})
        mock_rate_limit_error = type("RateLimitError", (Exception,), {})
        mock_timeout_error = type("APITimeoutError", (Exception,), {})

        mock_anthropic.AuthenticationError = mock_auth_error
        mock_anthropic.RateLimitError = mock_rate_limit_error
        mock_anthropic.APITimeoutError = mock_timeout_error

        mock_client.messages.create.side_effect = mock_auth_error("Invalid API key")

        with pytest.raises(InvalidCredentialsError) as exc_info:
            provider.generate("Test prompt")

        assert "Authentication failed" in str(exc_info.value)

    def test_get_model_info(self, provider):
        """Test getting model information."""
        info = provider.get_model_info()

        assert info["model_name"] == "claude-3-sonnet-20240229"
        assert info["provider"] == "anthropic"
        assert info["max_tokens"] == 4096
        assert info["context_window"] == 200000
        assert "text-generation" in info["capabilities"]
        assert "conversation" in info["capabilities"]
        assert info["supports_streaming"] is True
        assert info["supports_system_messages"] is True

    def test_get_model_info_different_models(self, mock_env):
        """Test model info for different Claude models."""
        # Test Claude 3.5 Sonnet with 8k output
        provider = AnthropicProvider("claude-3-5-sonnet-20241022")
        info = provider.get_model_info()
        assert info["max_tokens"] == 8192
        assert info["context_window"] == 200000

        # Test older Claude 2.0
        provider = AnthropicProvider("claude-2.0")
        info = provider.get_model_info()
        assert info["max_tokens"] == 4096
        assert info["context_window"] == 100000

    def test_get_default_parameters(self, provider):
        """Test getting default parameters."""
        params = provider.get_default_parameters()

        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 1000
        assert params["top_p"] == 1.0
        assert params["stop_sequences"] == []

    @patch("llm_providers.anthropic.anthropic.Anthropic")
    def test_generate_with_custom_parameters(self, mock_anthropic_class, provider):
        """Test generation with custom parameters."""
        # Mock the client and response
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        provider._client = mock_client
        provider._initialized = True

        # Mock successful response
        mock_content = Mock(text="Generated text")
        mock_response = Mock(content=[mock_content])
        mock_client.messages.create.return_value = mock_response

        # Generate with custom parameters
        result = provider.generate(
            "Test prompt", temperature=0.5, max_tokens=500, top_p=0.9, stop_sequences=["\n\n"]
        )

        assert result == "Generated text"

        # Verify custom parameters were passed
        call_args = mock_client.messages.create.call_args[1]
        assert call_args["temperature"] == 0.5
        assert call_args["max_tokens"] == 500
        assert call_args["top_p"] == 0.9
        assert call_args["stop_sequences"] == ["\n\n"]

    @patch("llm_providers.anthropic.anthropic.Anthropic")
    def test_generate_max_tokens_exceeds_limit(self, mock_anthropic_class, provider):
        """Test generation when requested tokens exceed model limit."""
        # Mock the client and response
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        provider._client = mock_client
        provider._initialized = True

        # Mock successful response
        mock_content = Mock(text="Generated text")
        mock_response = Mock(content=[mock_content])
        mock_client.messages.create.return_value = mock_response

        # Request more tokens than model supports
        with patch("llm_providers.anthropic.logger") as mock_logger:
            result = provider.generate("Test prompt", max_tokens=10000)

        # Should log warning and use model limit
        mock_logger.warning.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        assert call_args["max_tokens"] == 4096  # Model limit for claude-3-sonnet


class TestAnthropicProviderIntegration:
    """Integration tests for AnthropicProvider (optional, requires valid API key)."""

    @pytest.mark.skipif(
        not os.getenv("TEST_ANTHROPIC_INTEGRATION", "").lower() == "true",
        reason="Integration tests disabled. Set TEST_ANTHROPIC_INTEGRATION=true to enable.",
    )
    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
    def test_real_api_call(self):
        """Test actual API call to Anthropic."""
        provider = AnthropicProvider("claude-3-haiku-20240307")  # Use cheapest model

        # Test credential validation
        assert provider.validate_credentials() is True

        # Test simple generation
        response = provider.generate("Say 'Hello, World!' and nothing else.")
        assert "Hello, World!" in response
        assert len(response) < 100  # Should be a short response

    @pytest.mark.skipif(
        not os.getenv("TEST_ANTHROPIC_INTEGRATION", "").lower() == "true",
        reason="Integration tests disabled. Set TEST_ANTHROPIC_INTEGRATION=true to enable.",
    )
    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
    def test_real_api_model_info(self):
        """Test model info with real provider."""
        provider = AnthropicProvider("claude-3-haiku-20240307")
        info = provider.get_model_info()

        assert info["model_name"] == "claude-3-haiku-20240307"
        assert info["provider"] == "anthropic"
        assert info["max_tokens"] > 0
        assert info["context_window"] > 0
