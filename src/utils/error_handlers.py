"""
Error Handling Utilities

This module provides common error handling patterns and decorators used throughout the LLM Lab.
It centralizes error mapping, retry logic, and standardized error responses.
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import functools
import logging
import time
from typing import Any, Callable, Dict, Optional, Type

logger = logging.getLogger(__name__)


def with_error_handling(
    error_mapping: Dict[str, Type[Exception | None]] = None,
    default_error: Type[Exception] = Exception,
    log_errors: bool = True,
):
    """
    Decorator to handle exceptions with standardized error mapping and logging.

    Args:
        error_mapping: Dictionary mapping error keywords to exception types
        default_error: Default exception type to raise for unmapped errors
        log_errors: Whether to log errors before re-raising
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {e}", exc_info=True)

                if error_mapping:
                    error_msg = str(e).lower()
                    for keyword, exception_type in error_mapping.items():
                        if keyword.lower() in error_msg:
                            raise exception_type(str(e)) from e

                # Raise default error if no mapping found
                raise default_error(f"Error in {func.__name__}: {e}") from e

        return wrapper

    return decorator


def with_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Callable | None = None,
):
    """
    Decorator to add retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply delay by for each retry
        exceptions: Tuple of exception types to catch and retry on
        on_retry: Optional callback function called on each retry
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                        raise

                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {current_delay}s delay: {e}"
                    )

                    if on_retry:
                        on_retry(attempt + 1, e)

                    time.sleep(current_delay)
                    current_delay *= backoff_factor

            # This should never be reached due to the raise in the loop
            raise last_exception

        return wrapper

    return decorator


def safe_execute(
    func: Callable, default_return: Any = None, log_errors: bool = True, reraise: bool = False
) -> Any:
    """
    Execute a function safely, returning a default value on error.

    Args:
        func: Function to execute
        default_return: Value to return if function raises an exception
        log_errors: Whether to log errors
        reraise: Whether to re-raise the exception after logging

    Returns:
        Function result or default_return on error
    """
    try:
        return func()
    except Exception as e:
        if log_errors:
            logger.error(f"Error in safe_execute: {e}", exc_info=True)

        if reraise:
            raise

        return default_return


class ErrorContext:
    """Context manager for standardized error handling."""

    def __init__(
        self,
        operation_name: str,
        error_mapping: Dict[str, Type[Exception | None]] = None,
        cleanup_func: Callable | None = None,
    ):
        self.operation_name = operation_name
        self.error_mapping = error_mapping or {}
        self.cleanup_func = cleanup_func

    def __enter__(self):
        logger.debug(f"Starting operation: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Error in {self.operation_name}: {exc_val}", exc_info=True)

            # Map to specific exception if configured
            if self.error_mapping:
                error_msg = str(exc_val).lower()
                for keyword, exception_type in self.error_mapping.items():
                    if keyword.lower() in error_msg:
                        raise exception_type(str(exc_val)) from exc_val

        # Always run cleanup if provided
        if self.cleanup_func:
            try:
                self.cleanup_func()
            except Exception as cleanup_error:
                logger.error(f"Error in cleanup for {self.operation_name}: {cleanup_error}")

        logger.debug(f"Completed operation: {self.operation_name}")

        # Don't suppress the exception
        return False


def map_provider_errors(provider_name: str, error: Exception) -> Exception:
    """
    Common error mapping logic for provider exceptions.

    Args:
        provider_name: Name of the provider
        error: Original exception

    Returns:
        Mapped exception with provider context
    """
    from ..providers.exceptions import (
        InvalidCredentialsError,
        ProviderError,
        ProviderTimeoutError,
        RateLimitError,
    )

    error_msg = str(error).lower()
    error_type = type(error).__name__.lower()

    # Rate limiting errors
    if any(
        keyword in error_msg or keyword in error_type
        for keyword in ["rate limit", "quota", "throttle", "too many requests"]
    ):
        return RateLimitError(provider_name=provider_name, limit_type="API requests")

    # Authentication errors
    if any(
        keyword in error_msg or keyword in error_type
        for keyword in ["authentication", "unauthorized", "invalid key", "api key"]
    ):
        return InvalidCredentialsError(provider_name=provider_name, details=str(error))

    # Timeout errors
    if any(
        keyword in error_msg or keyword in error_type
        for keyword in ["timeout", "deadline", "connection"]
    ):
        return ProviderTimeoutError(provider_name=provider_name, operation="API call")

    # Default to generic provider error
    return ProviderError(provider_name, str(error))
