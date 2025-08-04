"""
Provider Exceptions

This module defines custom exceptions for LLM provider operations.
It provides a hierarchy of exceptions that allow for precise error handling
and better debugging of provider-related issues.
"""

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
        provider_name: Optional[str] = None,
        error_code: Optional[str] = None,
        is_retryable: bool = False
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
    
    def __init__(self, model_name: str, available_models: Optional[list] = None):
        message = f"No provider found for model '{model_name}'"
        if available_models:
            message += f". Available models: {', '.join(available_models)}"
        super().__init__(message, is_retryable=False)


class InvalidCredentialsError(ProviderError):
    """Raised when provider credentials are missing or invalid."""
    
    def __init__(self, provider_name: str, details: Optional[str] = None):
        message = f"Invalid or missing credentials for {provider_name}"
        if details:
            message += f": {details}"
        super().__init__(
            message, 
            provider_name=provider_name,
            error_code="INVALID_CREDENTIALS",
            is_retryable=False
        )


class ModelNotSupportedError(ProviderError):
    """Raised when a model is not supported by the provider."""
    
    def __init__(
        self, 
        model_name: str, 
        provider_name: str, 
        supported_models: Optional[list] = None
    ):
        message = f"Model '{model_name}' is not supported by {provider_name}"
        if supported_models:
            message += f". Supported models: {', '.join(supported_models)}"
        super().__init__(
            message,
            provider_name=provider_name,
            error_code="MODEL_NOT_SUPPORTED",
            is_retryable=False
        )


class ProviderTimeoutError(ProviderError):
    """Raised when a provider operation times out."""
    
    def __init__(
        self, 
        provider_name: str, 
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None
    ):
        message = f"Request to {provider_name} timed out"
        if operation:
            message = f"{operation} operation on {provider_name} timed out"
        if timeout_seconds:
            message += f" after {timeout_seconds}s"
        super().__init__(
            message,
            provider_name=provider_name,
            error_code="TIMEOUT",
            is_retryable=True
        )


class RateLimitError(ProviderError):
    """Raised when provider rate limits are exceeded."""
    
    def __init__(
        self, 
        provider_name: str,
        retry_after: Optional[int] = None,
        limit_type: Optional[str] = None
    ):
        message = f"Rate limit exceeded for {provider_name}"
        if limit_type:
            message += f" ({limit_type})"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        
        super().__init__(
            message,
            provider_name=provider_name,
            error_code="RATE_LIMIT",
            is_retryable=True
        )
        self.retry_after = retry_after


class ProviderConfigurationError(ProviderError):
    """Raised when provider configuration is invalid."""
    
    def __init__(
        self, 
        provider_name: str,
        config_issue: str,
        field_name: Optional[str] = None
    ):
        message = f"Configuration error for {provider_name}: {config_issue}"
        if field_name:
            message += f" (field: {field_name})"
        super().__init__(
            message,
            provider_name=provider_name,
            error_code="CONFIG_ERROR",
            is_retryable=False
        )


class ProviderResponseError(ProviderError):
    """Raised when provider returns an unexpected or malformed response."""
    
    def __init__(
        self,
        provider_name: str,
        details: str,
        response_data: Optional[dict] = None
    ):
        message = f"Invalid response from {provider_name}: {details}"
        super().__init__(
            message,
            provider_name=provider_name,
            error_code="INVALID_RESPONSE",
            is_retryable=True
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
    
    # Check for common patterns
    if 'rate limit' in exception_str or 'quota' in exception_str:
        return RateLimitError(provider_name)
    
    if 'unauthorized' in exception_str or 'authentication' in exception_str:
        return InvalidCredentialsError(provider_name, str(original_exception))
    
    if 'timeout' in exception_str or 'timed out' in exception_str:
        return ProviderTimeoutError(provider_name)
    
    if 'not found' in exception_str or '404' in exception_str:
        return ModelNotSupportedError(
            model_name="unknown",
            provider_name=provider_name
        )
    
    # Default to generic provider error
    return ProviderError(
        message=str(original_exception),
        provider_name=provider_name,
        error_code=exception_type,
        is_retryable=True  # Optimistic default
    )