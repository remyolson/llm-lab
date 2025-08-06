"""Tests for the OpenAI provider implementation."""

import os
from unittest.mock import MagicMock, Mock, patch

import openai
import pytest
from openai import OpenAI

from llm_providers.exceptions import (
    InvalidCredentialsError,
    ModelNotSupportedError,
    ProviderConfigurationError,
    ProviderResponseError,
    ProviderTimeoutError,
    RateLimitError,
)
from llm_providers.openai import OpenAIProvider


class TestOpenAIProvider:
    """Test the OpenAI provider implementation."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Mock environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123")

    @pytest.fixture
    def provider(self, mock_env):
        """Create a provider instance with mocked environment."""
        return OpenAIProvider("gpt-4")

    def test_provider_initialization(self, provider):
        """Test successful provider initialization."""
        assert provider.model_name == "gpt-4"
        assert provider.provider_name == "openai"
        assert "gpt-4" in provider.supported_models
        assert "gpt-3.5-turbo" in provider.supported_models
        assert provider._client is None

    def test_supported_models(self):
        """Test that all expected models are supported."""
        # This will trigger the decorator registration
        expected_models = [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-4o",
            "gpt-4o-mini",
        ]

        provider = OpenAIProvider("gpt-4")
        for model in expected_models:
            assert model in provider.supported_models

    def test_validate_credentials_no_api_key(self, monkeypatch):
        """Test credential validation with missing API key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        provider = OpenAIProvider("gpt-4")
        with pytest.raises(InvalidCredentialsError) as exc_info:
            provider.validate_credentials()

        assert "OPENAI_API_KEY environment variable not set" in str(exc_info.value)

    @patch("llm_providers.openai.OpenAI")
    def test_validate_credentials_invalid_key(self, mock_openai_class, mock_env):
        """Test credential validation with invalid API key."""
        # Mock the client to raise an authentication error
        mock_client = Mock()
        mock_client.models.list.side_effect = Exception("Error code: 401 - Invalid authentication")
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider("gpt-4")
        with pytest.raises(InvalidCredentialsError) as exc_info:
            provider.validate_credentials()

        assert "Invalid API key" in str(exc_info.value)

    @patch("llm_providers.openai.OpenAI")
    def test_validate_credentials_success(self, mock_openai_class, mock_env):
        """Test successful credential validation."""
        # Mock successful API call
        mock_client = Mock()
        mock_client.models.list.return_value = Mock()
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider("gpt-4")
        assert provider.validate_credentials() is True

    def test_initialize_client_no_api_key(self, monkeypatch):
        """Test client initialization with missing API key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        provider = OpenAIProvider("gpt-4")
        with pytest.raises(InvalidCredentialsError):
            provider._initialize_client()

    @patch("llm_providers.openai.OpenAI")
    def test_initialize_client_success(self, mock_openai_class, mock_env):
        """Test successful client initialization."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider("gpt-4")
        provider._initialize_client()

        assert provider._client is mock_client
        mock_openai_class.assert_called_once_with(
            api_key="sk-test-key-123", timeout=30, max_retries=0
        )

    @patch("llm_providers.openai.OpenAI")
    def test_generate_empty_prompt(self, mock_openai_class, mock_env):
        """Test generate with empty prompt."""
        provider = OpenAIProvider("gpt-4")
        provider._initialized = True

        with pytest.raises(ProviderResponseError) as exc_info:
            provider.generate("")

        assert "Empty prompt provided" in str(exc_info.value)

    @patch("llm_providers.openai.OpenAI")
    def test_generate_success(self, mock_openai_class, mock_env):
        """Test successful text generation."""
        # Mock the response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Paris is the capital of France."))]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider("gpt-4")
        provider._initialized = True

        response = provider.generate("What is the capital of France?")

        assert response == "Paris is the capital of France."
        mock_client.chat.completions.create.assert_called_once()

        # Check the call arguments
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4"
        assert call_args.kwargs["messages"][0]["content"] == "What is the capital of France?"

    @patch("llm_providers.openai.OpenAI")
    def test_generate_with_parameters(self, mock_openai_class, mock_env):
        """Test generation with custom parameters."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider("gpt-3.5-turbo")
        provider._initialized = True

        response = provider.generate("Test prompt", temperature=0.5, max_tokens=500, top_p=0.9)

        assert response == "Test response"

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0.5
        assert call_args.kwargs["max_tokens"] == 500
        assert call_args.kwargs["top_p"] == 0.9

    @patch("llm_providers.openai.OpenAI")
    @patch("time.sleep")
    def test_generate_rate_limit_retry(self, mock_sleep, mock_openai_class, mock_env):
        """Test retry logic for rate limit errors."""

        # Create a mock OpenAI rate limit error
        class MockRateLimitError(Exception):
            def __init__(self, message):
                super().__init__(message)
                self.__class__.__name__ = "RateLimitError"

        # First call raises rate limit error, second succeeds
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Success after retry"))]

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            MockRateLimitError("Rate limit exceeded"),
            mock_response,
        ]
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider("gpt-4")
        provider._initialized = True

        response = provider.generate("Test prompt")

        assert response == "Success after retry"
        assert mock_client.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once_with(1.0)  # First retry delay

    @patch("llm_providers.openai.OpenAI")
    @patch("time.sleep")
    def test_generate_rate_limit_max_retries(self, mock_sleep, mock_openai_class, mock_env):
        """Test rate limit error after max retries."""

        # Create a mock OpenAI rate limit error
        class MockRateLimitError(Exception):
            def __init__(self, message):
                super().__init__(message)
                self.__class__.__name__ = "RateLimitError"

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = MockRateLimitError("Rate limit exceeded")
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider("gpt-4", max_retries=2)
        provider._initialized = True

        with pytest.raises(RateLimitError) as exc_info:
            provider.generate("Test prompt")

        assert mock_client.chat.completions.create.call_count == 3  # Initial + 2 retries
        assert mock_sleep.call_count == 2  # Sleep between retries
        assert exc_info.value.retry_after == 4  # 1 * (2 ** 2)

    @patch("llm_providers.openai.OpenAI")
    def test_generate_timeout_error(self, mock_openai_class, mock_env):
        """Test timeout error handling."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("Request timeout")
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider("gpt-4")
        provider._initialized = True

        with pytest.raises(ProviderTimeoutError) as exc_info:
            provider.generate("Test prompt")

        # Check that it's a timeout error
        assert exc_info.value.provider_name == "openai"

    @patch("llm_providers.openai.OpenAI")
    def test_generate_model_not_found(self, mock_openai_class, mock_env):
        """Test model not found error."""

        # Create mock error that looks like OpenAI model not found
        class MockModelNotFoundError(Exception):
            def __init__(self, message):
                super().__init__(message)
                self.__class__.__name__ = "ModelNotFoundError"

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = MockModelNotFoundError(
            "Model 'gpt-5' does not exist"
        )
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider("gpt-4")  # Initialize with valid model
        provider.model_name = "gpt-5"  # Change to invalid model
        provider._initialized = True

        with pytest.raises(ModelNotSupportedError) as exc_info:
            provider.generate("Test prompt")

        assert "gpt-5" in str(exc_info.value)

    @patch("llm_providers.openai.OpenAI")
    def test_generate_no_response_content(self, mock_openai_class, mock_env):
        """Test handling of empty response content."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=None))]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider("gpt-4")
        provider._initialized = True

        with pytest.raises(ProviderResponseError) as exc_info:
            provider.generate("Test prompt")

        assert "No text in model response" in str(exc_info.value)

    def test_get_model_info_gpt4(self, mock_env):
        """Test model info for GPT-4."""
        provider = OpenAIProvider("gpt-4")
        info = provider.get_model_info()

        assert info["model_name"] == "gpt-4"
        assert info["provider"] == "openai"
        assert info["max_tokens"] == 8192
        assert info["context_window"] == 8192
        assert info["training_cutoff"] == "September 2021"
        assert "text-generation" in info["capabilities"]
        assert "chat" in info["capabilities"]
        assert info["supports_streaming"] is True
        assert info["supports_functions"] is True

    def test_get_model_info_gpt4_turbo(self, mock_env):
        """Test model info for GPT-4 Turbo."""
        provider = OpenAIProvider("gpt-4-turbo")
        info = provider.get_model_info()

        assert info["model_name"] == "gpt-4-turbo"
        assert info["max_tokens"] == 4096
        assert info["context_window"] == 128000
        assert info["training_cutoff"] == "April 2023"

    def test_get_model_info_gpt35(self, mock_env):
        """Test model info for GPT-3.5 Turbo."""
        provider = OpenAIProvider("gpt-3.5-turbo")
        info = provider.get_model_info()

        assert info["model_name"] == "gpt-3.5-turbo"
        assert info["max_tokens"] == 4096
        assert info["context_window"] == 16385
        assert info["training_cutoff"] == "September 2021"

    def test_get_model_info_unknown_model(self, mock_env):
        """Test model info for unknown model (uses defaults)."""
        # Create provider with a model that's in SUPPORTED_MODELS but not in model_info
        provider = OpenAIProvider("gpt-4")
        provider.model_name = "gpt-future"  # Simulate unknown model
        info = provider.get_model_info()

        assert info["model_name"] == "gpt-future"
        assert info["max_tokens"] == 4096  # Default
        assert info["context_window"] == 8192  # Default
        assert info["training_cutoff"] == "Unknown"  # Default

    @patch("llm_providers.openai.OpenAI")
    def test_full_initialization_flow(self, mock_openai_class, mock_env):
        """Test the full initialization and generation flow."""
        # Mock successful validation
        mock_client = Mock()
        mock_client.models.list.return_value = Mock()

        # Mock successful generation
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test successful"))]
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai_class.return_value = mock_client

        # Create provider and use it
        provider = OpenAIProvider("gpt-4o-mini")
        provider.initialize()  # This should validate credentials

        response = provider.generate("Test prompt")
        assert response == "Test successful"

        # Verify initialization was called
        assert provider._initialized is True
        assert provider._client is not None


class TestOpenAIProviderIntegration:
    """Integration tests for OpenAI provider (only run with real API key)."""

    @pytest.mark.skipif(
        not os.getenv("TEST_OPENAI_INTEGRATION"),
        reason="Set TEST_OPENAI_INTEGRATION=1 to run integration tests",
    )
    def test_real_api_call(self):
        """Test actual API call to OpenAI."""
        # This test requires a real API key
        if not os.getenv("OPENAI_API_KEY", "").startswith("sk-"):
            pytest.skip("Valid OPENAI_API_KEY required for integration test")

        provider = OpenAIProvider("gpt-3.5-turbo")
        provider.initialize()

        response = provider.generate(
            "Say 'Hello, integration test!' and nothing else.", temperature=0.0, max_tokens=20
        )

        assert response
        assert "integration test" in response.lower()
