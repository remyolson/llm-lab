"""
Service Interface Definitions

This module defines the protocol interfaces (contracts) for all injectable
services in the LLM Lab framework. These protocols enable loose coupling
between services and their implementations, making the code more testable
and maintainable.

Key Design Principles:
- Protocol-based interfaces for structural typing
- Clear separation between interface and implementation
- Comprehensive method signatures with return types
- Runtime checkable protocols for testing
- Backward compatibility with existing code patterns
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class ILogger(Protocol):
    """Protocol for logger services."""

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message."""
        ...

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an info message."""
        ...

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message."""
        ...

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message."""
        ...

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a critical message."""
        ...

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an exception with stack trace."""
        ...


@runtime_checkable
class ILoggerFactory(Protocol):
    """Protocol for logger factory services."""

    def get_logger(self, name: str) -> ILogger:
        """
        Get a logger instance for the specified name.

        Args:
            name: Logger name (typically __name__)

        Returns:
            Logger instance
        """
        ...

    def configure_logging(
        self,
        level: str = "INFO",
        format_string: Optional[str] = None,
        handlers: Optional[List[logging.Handler]] = None,
    ) -> None:
        """
        Configure global logging settings.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format_string: Custom log format string
            handlers: List of logging handlers to add
        """
        ...


@runtime_checkable
class IConfigurationService(Protocol):
    """Protocol for configuration services."""

    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration setting by key.

        Args:
            key: Configuration key (supports dot notation like 'providers.openai.api_key')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        ...

    def get_environment_variable(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get an environment variable with optional default.

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            Environment variable value or default
        """
        ...

    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific provider.

        Args:
            provider_name: Name of the provider (e.g., 'openai', 'anthropic')

        Returns:
            Provider configuration dictionary
        """
        ...

    def get_model_parameters(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get model parameters configuration.

        Args:
            model_name: Specific model name, or None for defaults

        Returns:
            Model parameters dictionary
        """
        ...

    def get_network_config(self) -> Dict[str, Any]:
        """
        Get network-related configuration.

        Returns:
            Network configuration dictionary
        """
        ...

    def reload_configuration(self) -> None:
        """Reload configuration from all sources."""
        ...


@runtime_checkable
class IHttpClient(Protocol):
    """Protocol for HTTP client services."""

    def get(self, url: str, **kwargs: Any) -> Any:
        """Make a GET request."""
        ...

    def post(self, url: str, **kwargs: Any) -> Any:
        """Make a POST request."""
        ...

    def put(self, url: str, **kwargs: Any) -> Any:
        """Make a PUT request."""
        ...

    def delete(self, url: str, **kwargs: Any) -> Any:
        """Make a DELETE request."""
        ...

    def close(self) -> None:
        """Close the HTTP client and clean up resources."""
        ...


@runtime_checkable
class IHttpClientFactory(Protocol):
    """Protocol for HTTP client factory services."""

    def create_client(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> IHttpClient:
        """
        Create an HTTP client with specified configuration.

        Args:
            base_url: Base URL for the client
            timeout: Request timeout in seconds
            headers: Default headers for requests
            **kwargs: Additional client configuration

        Returns:
            Configured HTTP client instance
        """
        ...

    def create_provider_client(self, provider_name: str) -> IHttpClient:
        """
        Create an HTTP client configured for a specific provider.

        Args:
            provider_name: Name of the provider (e.g., 'openai', 'anthropic')

        Returns:
            HTTP client configured for the provider
        """
        ...


@runtime_checkable
class ILLMProvider(Protocol):
    """Protocol for LLM provider services."""

    @property
    def model_name(self) -> str:
        """Get the model name."""
        ...

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        ...

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: Input prompt
            **kwargs: Generation parameters

        Returns:
            Generated response text
        """
        ...

    def batch_generate(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of input prompts
            **kwargs: Generation parameters

        Returns:
            List of generated responses
        """
        ...

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Model information dictionary
        """
        ...

    def validate_credentials(self) -> bool:
        """
        Validate provider credentials.

        Returns:
            True if credentials are valid
        """
        ...


@runtime_checkable
class IProviderFactory(Protocol):
    """Protocol for LLM provider factory services."""

    def create_provider(self, provider_name: str, model_name: str, **config: Any) -> ILLMProvider:
        """
        Create an LLM provider instance.

        Args:
            provider_name: Name of the provider (e.g., 'openai', 'anthropic')
            model_name: Name of the model to use
            **config: Additional provider configuration

        Returns:
            Configured provider instance
        """
        ...

    def get_available_providers(self) -> List[str]:
        """
        Get list of available provider names.

        Returns:
            List of provider names
        """
        ...

    def get_provider_models(self, provider_name: str) -> List[str]:
        """
        Get list of models supported by a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            List of supported model names
        """
        ...


@runtime_checkable
class IEvaluationService(Protocol):
    """Protocol for evaluation services."""

    def evaluate_response(
        self, response: str, expected: str, evaluation_type: str = "keyword_match", **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Evaluate a response against expected results.

        Args:
            response: Generated response to evaluate
            expected: Expected response or keywords
            evaluation_type: Type of evaluation to perform
            **kwargs: Additional evaluation parameters

        Returns:
            Evaluation results dictionary
        """
        ...

    def batch_evaluate(
        self,
        responses: List[str],
        expected_list: List[str],
        evaluation_type: str = "keyword_match",
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple responses in batch.

        Args:
            responses: List of generated responses
            expected_list: List of expected responses
            evaluation_type: Type of evaluation to perform
            **kwargs: Additional evaluation parameters

        Returns:
            List of evaluation results
        """
        ...

    def get_available_evaluators(self) -> List[str]:
        """
        Get list of available evaluation types.

        Returns:
            List of evaluation type names
        """
        ...


@runtime_checkable
class IResourceMonitor(Protocol):
    """Protocol for resource monitoring services."""

    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        ...

    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        ...

    def get_current_usage(self) -> Dict[str, Any]:
        """
        Get current resource usage.

        Returns:
            Dictionary with CPU, memory, and other resource metrics
        """
        ...

    def get_usage_history(self, duration_minutes: int = 10) -> List[Dict[str, Any]]:
        """
        Get resource usage history.

        Args:
            duration_minutes: How many minutes of history to return

        Returns:
            List of resource usage snapshots
        """
        ...


@runtime_checkable
class IFileService(Protocol):
    """Protocol for file system services."""

    def read_file(self, path: str | Path, encoding: str = "utf-8") -> str:
        """
        Read file contents.

        Args:
            path: File path
            encoding: File encoding

        Returns:
            File contents as string
        """
        ...

    def write_file(self, path: str | Path, content: str, encoding: str = "utf-8") -> None:
        """
        Write content to file.

        Args:
            path: File path
            content: Content to write
            encoding: File encoding
        """
        ...

    def file_exists(self, path: str | Path) -> bool:
        """
        Check if file exists.

        Args:
            path: File path

        Returns:
            True if file exists
        """
        ...

    def create_directory(self, path: str | Path, exist_ok: bool = True) -> None:
        """
        Create directory.

        Args:
            path: Directory path
            exist_ok: Don't raise error if directory exists
        """
        ...


@runtime_checkable
class ICacheService(Protocol):
    """Protocol for caching services."""

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for no expiration)
        """
        ...

    def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False if not found
        """
        ...

    def clear(self) -> None:
        """Clear all cached values."""
        ...


# Type aliases for common service combinations
ServiceProvider = ILLMProvider | IProviderFactory
ConfigurationProvider = IConfigurationService | IFileService
LoggingProvider = ILogger | ILoggerFactory
NetworkProvider = IHttpClient | IHttpClientFactory
