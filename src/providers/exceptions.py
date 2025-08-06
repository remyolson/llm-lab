"""
Provider Exceptions

This module defines custom exceptions for LLM provider operations.
It provides a hierarchy of exceptions that allow for precise error handling
and better debugging of provider-related issues.
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

from typing import Optional


class ProviderError(Exception):
    """
    Base exception for all provider-related errors.

    Attributes:
        message: The error message
        provider_name: The name of the provider that raised the error
        error_code: Optional error code for programmatic handling
        is_retryable: Whether the operation can be retried
    """

    def __init__(
        self,
        message: str,
        provider_name: str | None = None,
        error_code: str | None = None,
        is_retryable: bool = False,
    ):
        super().__init__(message)
        self.message = message
        self.provider_name = provider_name
        self.error_code = error_code
        self.is_retryable = is_retryable

    def __str__(self):
        parts = []
        if self.provider_name:
            parts.append(f"[{self.provider_name}]")
        if self.error_code:
            parts.append(f"({self.error_code})")
        parts.append(self.message)
        return " ".join(parts)


class ProviderNotFoundError(ProviderError):
    """Raised when a requested provider cannot be found."""

    def __init__(self, model_name: str, available_models: list | None = None):
        message = f"No provider found for model '{model_name}'"
        if available_models:
            message += f". Available models: {', '.join(available_models)}"
        super().__init__(message, is_retryable=False)


class InvalidCredentialsError(ProviderError):
    """Raised when provider credentials are missing or invalid."""

    def __init__(self, provider_name: str, details: str | None = None):
        message = f"Invalid or missing credentials for {provider_name}"
        if details:
            message += f": {details}"

        # Add helpful troubleshooting suggestions
        troubleshooting = self._get_troubleshooting_tips(provider_name)
        if troubleshooting:
            message += f"\n\nTroubleshooting:\n{troubleshooting}"

        super().__init__(
            message,
            provider_name=provider_name,
            error_code="INVALID_CREDENTIALS",
            is_retryable=False,
        )

    def _get_troubleshooting_tips(self, provider_name: str) -> str:
        """Get provider-specific troubleshooting tips."""
        tips = {
            "OpenAI": (
                "1. Check that OPENAI_API_KEY is set in your environment or .env file\n"
                "2. Verify your API key at https://platform.openai.com/api-keys\n"
                "3. Ensure your API key starts with 'sk-'\n"
                "4. Check if your account has available credits"
            ),
            "Anthropic": (
                "1. Check that ANTHROPIC_API_KEY is set in your environment or .env file\n"
                "2. Verify your API key at https://console.anthropic.com/settings/keys\n"
                "3. Ensure your API key is valid and not expired\n"
                "4. Check your usage limits at https://console.anthropic.com/settings/limits"
            ),
            "Google": (
                "1. Check that GOOGLE_API_KEY is set in your environment or .env file\n"
                "2. Verify your API key in Google Cloud Console\n"
                "3. Ensure the Generative AI API is enabled for your project\n"
                "4. Check API quotas at https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com"
            ),
        }
        return tips.get(
            provider_name,
            f"1. Check that the API key for {provider_name} is correctly set\n"
            f"2. Verify the API key is valid and not expired\n"
            f"3. Check the provider's documentation for setup instructions",
        )


class ModelNotSupportedError(ProviderError):
    """Raised when a model is not supported by the provider."""

    def __init__(self, model_name: str, provider_name: str, supported_models: list | None = None):
        message = f"Model '{model_name}' is not supported by {provider_name}"
        if supported_models:
            message += f". Supported models: {', '.join(supported_models)}"
        super().__init__(
            message,
            provider_name=provider_name,
            error_code="MODEL_NOT_SUPPORTED",
            is_retryable=False,
        )


class ProviderTimeoutError(ProviderError):
    """Raised when a provider operation times out."""

    def __init__(
        self,
        provider_name: str,
        timeout_seconds: float | None = None,
        operation: str | None = None,
    ):
        message = f"Request to {provider_name} timed out"
        if operation:
            message = f"{operation} operation on {provider_name} timed out"
        if timeout_seconds:
            message += f" after {timeout_seconds}s"
        super().__init__(
            message, provider_name=provider_name, error_code="TIMEOUT", is_retryable=True
        )


class RateLimitError(ProviderError):
    """Raised when provider rate limits are exceeded."""

    def __init__(
        self,
        provider_name: str,
        retry_after: int | None = None,
        limit_type: str | None = None,
    ):
        message = f"Rate limit exceeded for {provider_name}"
        if limit_type:
            message += f" ({limit_type})"
        if retry_after:
            message += f". Retry after {retry_after} seconds"

        # Add suggestions for handling rate limits
        message += "\n\nSuggestions:"
        message += "\n1. Implement exponential backoff with retry logic"
        message += "\n2. Consider using a different model with higher rate limits"
        message += "\n3. Batch requests to reduce API calls"

        if provider_name == "OpenAI":
            message += "\n4. Check your rate limits at https://platform.openai.com/account/limits"
        elif provider_name == "Anthropic":
            message += "\n4. Review rate limits at https://console.anthropic.com/settings/limits"

        super().__init__(
            message, provider_name=provider_name, error_code="RATE_LIMIT", is_retryable=True
        )
        self.retry_after = retry_after


class ProviderConfigurationError(ProviderError):
    """Raised when provider configuration is invalid."""

    def __init__(self, provider_name: str, config_issue: str, field_name: str | None = None):
        message = f"Configuration error for {provider_name}: {config_issue}"
        if field_name:
            message += f" (field: {field_name})"
        super().__init__(
            message, provider_name=provider_name, error_code="CONFIG_ERROR", is_retryable=False
        )


class ProviderResponseError(ProviderError):
    """Raised when provider returns an unexpected or malformed response."""

    def __init__(self, provider_name: str, details: str, response_data: dict | None = None):
        message = f"Invalid response from {provider_name}: {details}"
        super().__init__(
            message, provider_name=provider_name, error_code="INVALID_RESPONSE", is_retryable=True
        )
        self.response_data = response_data


# Exception mapping utilities
def map_provider_exception(provider_name: str, original_exception: Exception) -> ProviderError:
    """
    Map provider-specific exceptions to standardized ProviderError exceptions.

    This function helps convert provider SDK exceptions into our standardized
    exception hierarchy for consistent error handling.

    Args:
        provider_name: Name of the provider
        original_exception: The original exception from the provider SDK

    Returns:
        A standardized ProviderError instance
    """
    exception_str = str(original_exception).lower()
    exception_type = type(original_exception).__name__

    # Check for common patterns with more specific error detection
    if any(term in exception_str for term in ["rate limit", "quota", "too many requests", "429"]):
        # Try to extract retry-after information
        retry_after = None
        if "retry after" in exception_str:
            import re

            match = re.search(r"retry after (\d+)", exception_str)
            if match:
                retry_after = int(match.group(1))
        return RateLimitError(provider_name, retry_after=retry_after)

    if any(
        term in exception_str
        for term in ["unauthorized", "authentication", "401", "invalid api key", "invalid_api_key"]
    ):
        return InvalidCredentialsError(provider_name, str(original_exception))

    if any(term in exception_str for term in ["timeout", "timed out", "deadline exceeded"]):
        return ProviderTimeoutError(provider_name)

    if any(term in exception_str for term in ["not found", "404", "model not available"]):
        # Try to extract model name from error
        model_name = "unknown"
        if "model" in exception_str:
            import re

            match = re.search(r"model['\"]?\s*:\s*['\"]?([^'\"]+)", exception_str)
            if match:
                model_name = match.group(1)
        return ModelNotSupportedError(model_name=model_name, provider_name=provider_name)

    if any(term in exception_str for term in ["invalid request", "400", "bad request"]):
        return ProviderConfigurationError(
            provider_name=provider_name,
            config_issue=f"Invalid request parameters: {str(original_exception)}",
        )

    if any(
        term in exception_str
        for term in ["500", "502", "503", "internal server error", "service unavailable"]
    ):
        return ProviderError(
            message=f"Provider service temporarily unavailable: {str(original_exception)}",
            provider_name=provider_name,
            error_code="SERVICE_ERROR",
            is_retryable=True,
        )

    # Default to generic provider error with the original message
    return ProviderError(
        message=f"Unexpected error from {provider_name}: {str(original_exception)}",
        provider_name=provider_name,
        error_code=exception_type,
        is_retryable=False,  # Conservative default for unknown errors
    )
