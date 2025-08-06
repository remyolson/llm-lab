"""Tests for provider exception classes."""

import pytest

from src.providers.exceptions import (
    InvalidCredentialsError,
    ModelNotSupportedError,
    ProviderConfigurationError,
    ProviderError,
    ProviderNotFoundError,
    ProviderResponseError,
    ProviderTimeoutError,
    RateLimitError,
    map_provider_exception,
)


class TestProviderError:
    """Test the base ProviderError class."""

    def test_basic_error(self):
        """Test basic ProviderError creation."""
        error = ProviderError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.provider_name is None
        assert error.error_code is None
        assert error.is_retryable is False

    def test_error_with_all_attributes(self):
        """Test ProviderError with all attributes."""
        error = ProviderError(
            "Test error", provider_name="test-provider", error_code="TEST_ERROR", is_retryable=True
        )
        assert str(error) == "[test-provider] (TEST_ERROR) Test error"
        assert error.provider_name == "test-provider"
        assert error.error_code == "TEST_ERROR"
        assert error.is_retryable is True

    def test_error_with_provider_only(self):
        """Test ProviderError with provider name only."""
        error = ProviderError("Test error", provider_name="test-provider")
        assert str(error) == "[test-provider] Test error"

    def test_error_with_code_only(self):
        """Test ProviderError with error code only."""
        error = ProviderError("Test error", error_code="TEST_ERROR")
        assert str(error) == "(TEST_ERROR) Test error"


class TestProviderNotFoundError:
    """Test ProviderNotFoundError class."""

    def test_basic_not_found_error(self):
        """Test basic ProviderNotFoundError."""
        error = ProviderNotFoundError("unknown-model")
        assert str(error) == "No provider found for model 'unknown-model'"
        assert error.is_retryable is False

    def test_not_found_with_available_models(self):
        """Test ProviderNotFoundError with available models list."""
        error = ProviderNotFoundError(
            "unknown-model", available_models=["model1", "model2", "model3"]
        )
        expected = (
            "No provider found for model 'unknown-model'. Available models: model1, model2, model3"
        )
        assert str(error) == expected


class TestInvalidCredentialsError:
    """Test InvalidCredentialsError class."""

    def test_basic_credentials_error(self):
        """Test basic InvalidCredentialsError."""
        error = InvalidCredentialsError("openai")
        assert (
            str(error) == "[openai] (INVALID_CREDENTIALS) Invalid or missing credentials for openai"
        )
        assert error.provider_name == "openai"
        assert error.error_code == "INVALID_CREDENTIALS"
        assert error.is_retryable is False

    def test_credentials_error_with_details(self):
        """Test InvalidCredentialsError with details."""
        error = InvalidCredentialsError("anthropic", "API key format is invalid")
        expected = "[anthropic] (INVALID_CREDENTIALS) Invalid or missing credentials for anthropic: API key format is invalid"
        assert str(error) == expected


class TestModelNotSupportedError:
    """Test ModelNotSupportedError class."""

    def test_basic_model_not_supported(self):
        """Test basic ModelNotSupportedError."""
        error = ModelNotSupportedError("gpt-5", "openai")
        assert (
            str(error) == "[openai] (MODEL_NOT_SUPPORTED) Model 'gpt-5' is not supported by openai"
        )
        assert error.provider_name == "openai"
        assert error.error_code == "MODEL_NOT_SUPPORTED"
        assert error.is_retryable is False

    def test_model_not_supported_with_list(self):
        """Test ModelNotSupportedError with supported models list."""
        error = ModelNotSupportedError(
            "gpt-5", "openai", supported_models=["gpt-4", "gpt-3.5-turbo"]
        )
        expected = "[openai] (MODEL_NOT_SUPPORTED) Model 'gpt-5' is not supported by openai. Supported models: gpt-4, gpt-3.5-turbo"
        assert str(error) == expected


class TestProviderTimeoutError:
    """Test ProviderTimeoutError class."""

    def test_basic_timeout_error(self):
        """Test basic ProviderTimeoutError."""
        error = ProviderTimeoutError("anthropic")
        assert str(error) == "[anthropic] (TIMEOUT) Request to anthropic timed out"
        assert error.provider_name == "anthropic"
        assert error.error_code == "TIMEOUT"
        assert error.is_retryable is True

    def test_timeout_with_seconds(self):
        """Test ProviderTimeoutError with timeout seconds."""
        error = ProviderTimeoutError("openai", timeout_seconds=30.5)
        assert str(error) == "[openai] (TIMEOUT) Request to openai timed out after 30.5s"

    def test_timeout_with_operation(self):
        """Test ProviderTimeoutError with operation name."""
        error = ProviderTimeoutError("google", operation="Generate")
        assert str(error) == "[google] (TIMEOUT) Generate operation on google timed out"

    def test_timeout_with_all_params(self):
        """Test ProviderTimeoutError with all parameters."""
        error = ProviderTimeoutError("anthropic", timeout_seconds=60, operation="Text generation")
        assert (
            str(error)
            == "[anthropic] (TIMEOUT) Text generation operation on anthropic timed out after 60s"
        )


class TestRateLimitError:
    """Test RateLimitError class."""

    def test_basic_rate_limit_error(self):
        """Test basic RateLimitError."""
        error = RateLimitError("openai")
        assert str(error) == "[openai] (RATE_LIMIT) Rate limit exceeded for openai"
        assert error.provider_name == "openai"
        assert error.error_code == "RATE_LIMIT"
        assert error.is_retryable is True
        assert error.retry_after is None

    def test_rate_limit_with_retry_after(self):
        """Test RateLimitError with retry_after."""
        error = RateLimitError("anthropic", retry_after=60)
        assert (
            str(error)
            == "[anthropic] (RATE_LIMIT) Rate limit exceeded for anthropic. Retry after 60 seconds"
        )
        assert error.retry_after == 60

    def test_rate_limit_with_limit_type(self):
        """Test RateLimitError with limit type."""
        error = RateLimitError("openai", limit_type="requests per minute")
        assert (
            str(error)
            == "[openai] (RATE_LIMIT) Rate limit exceeded for openai (requests per minute)"
        )

    def test_rate_limit_with_all_params(self):
        """Test RateLimitError with all parameters."""
        error = RateLimitError("google", retry_after=120, limit_type="tokens per hour")
        expected = "[google] (RATE_LIMIT) Rate limit exceeded for google (tokens per hour). Retry after 120 seconds"
        assert str(error) == expected


class TestProviderConfigurationError:
    """Test ProviderConfigurationError class."""

    def test_basic_config_error(self):
        """Test basic ProviderConfigurationError."""
        error = ProviderConfigurationError("openai", "Missing API endpoint")
        assert (
            str(error)
            == "[openai] (CONFIG_ERROR) Configuration error for openai: Missing API endpoint"
        )
        assert error.provider_name == "openai"
        assert error.error_code == "CONFIG_ERROR"
        assert error.is_retryable is False

    def test_config_error_with_field(self):
        """Test ProviderConfigurationError with field name."""
        error = ProviderConfigurationError("anthropic", "Invalid value", field_name="temperature")
        expected = "[anthropic] (CONFIG_ERROR) Configuration error for anthropic: Invalid value (field: temperature)"
        assert str(error) == expected


class TestProviderResponseError:
    """Test ProviderResponseError class."""

    def test_basic_response_error(self):
        """Test basic ProviderResponseError."""
        error = ProviderResponseError("openai", "Empty response received")
        assert (
            str(error)
            == "[openai] (INVALID_RESPONSE) Invalid response from openai: Empty response received"
        )
        assert error.provider_name == "openai"
        assert error.error_code == "INVALID_RESPONSE"
        assert error.is_retryable is True
        assert error.response_data is None

    def test_response_error_with_data(self):
        """Test ProviderResponseError with response data."""
        response_data = {"status": "error", "message": "Internal error"}
        error = ProviderResponseError("google", "Unexpected format", response_data=response_data)
        assert (
            str(error)
            == "[google] (INVALID_RESPONSE) Invalid response from google: Unexpected format"
        )
        assert error.response_data == response_data


class TestMapProviderException:
    """Test the map_provider_exception utility function."""

    def test_map_rate_limit_exception(self):
        """Test mapping rate limit exceptions."""
        # Test with "rate limit" in message
        original = Exception("Rate limit exceeded")
        mapped = map_provider_exception("openai", original)
        assert isinstance(mapped, RateLimitError)
        assert mapped.provider_name == "openai"

        # Test with "quota" in message
        original = Exception("Quota exceeded for the day")
        mapped = map_provider_exception("anthropic", original)
        assert isinstance(mapped, RateLimitError)

    def test_map_authentication_exception(self):
        """Test mapping authentication exceptions."""
        # Test with "unauthorized"
        original = Exception("Unauthorized access")
        mapped = map_provider_exception("openai", original)
        assert isinstance(mapped, InvalidCredentialsError)
        assert mapped.provider_name == "openai"

        # Test with "authentication"
        original = Exception("Authentication failed")
        mapped = map_provider_exception("google", original)
        assert isinstance(mapped, InvalidCredentialsError)

    def test_map_timeout_exception(self):
        """Test mapping timeout exceptions."""
        # Test with "timeout"
        original = Exception("Request timeout")
        mapped = map_provider_exception("anthropic", original)
        assert isinstance(mapped, ProviderTimeoutError)

        # Test with "timed out"
        original = Exception("Operation timed out after 30s")
        mapped = map_provider_exception("openai", original)
        assert isinstance(mapped, ProviderTimeoutError)

    def test_map_not_found_exception(self):
        """Test mapping not found exceptions."""
        # Test with "not found"
        original = Exception("Model not found")
        mapped = map_provider_exception("openai", original)
        assert isinstance(mapped, ModelNotSupportedError)

        # Test with "404"
        original = Exception("404 Error: Resource not found")
        mapped = map_provider_exception("google", original)
        assert isinstance(mapped, ModelNotSupportedError)

    def test_map_generic_exception(self):
        """Test mapping generic exceptions."""
        original = Exception("Something went wrong")
        mapped = map_provider_exception("test-provider", original)
        assert isinstance(mapped, ProviderError)
        assert mapped.provider_name == "test-provider"
        assert mapped.message == "Something went wrong"
        assert mapped.error_code == "Exception"
        assert mapped.is_retryable is True

    def test_map_exception_preserves_type_name(self):
        """Test that exception type name is preserved as error code."""

        class CustomException(Exception):
            pass

        original = CustomException("Custom error")
        mapped = map_provider_exception("provider", original)
        assert isinstance(mapped, ProviderError)
        assert mapped.error_code == "CustomException"
