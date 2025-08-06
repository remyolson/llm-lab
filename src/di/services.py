"""
Injectable Service Implementations

This module provides concrete implementations of the service protocols
defined in interfaces.py. These services are designed to be injected
via the DI container and provide loose coupling from existing code.

Key Features:
- Protocol-compliant implementations
- Integration with existing systems (Pydantic Settings, Python logging)
- Backward compatibility with current patterns
- Enhanced testability through interface abstraction
- Centralized dependency management
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..types.generics import GenericCache
from .interfaces import (
    ICacheService,
    IConfigurationService,
    IEvaluationService,
    IFileService,
    IHttpClient,
    IHttpClientFactory,
    ILLMProvider,
    ILogger,
    ILoggerFactory,
    IProviderFactory,
    IResourceMonitor,
)


class LoggerAdapter(ILogger):
    """Adapter that wraps Python's standard logger to implement ILogger protocol."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message."""
        self._logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an info message."""
        self._logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message."""
        self._logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message."""
        self._logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a critical message."""
        self._logger.critical(message, *args, **kwargs)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an exception with stack trace."""
        self._logger.exception(message, *args, **kwargs)


class LoggerFactory(ILoggerFactory):
    """Factory for creating logger instances with centralized configuration."""

    def __init__(self, config_service: Optional[IConfigurationService] = None):
        self._config_service = config_service
        self._configured = False

    def get_logger(self, name: str) -> ILogger:
        """
        Get a logger instance for the specified name.

        Args:
            name: Logger name (typically __name__)

        Returns:
            Logger instance wrapped in LoggerAdapter
        """
        if not self._configured:
            self._configure_default_logging()

        python_logger = logging.getLogger(name)
        return LoggerAdapter(python_logger)

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
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper()))

        # Clear existing handlers
        root_logger.handlers.clear()

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(format_string))
        root_logger.addHandler(console_handler)

        # Add custom handlers
        if handlers:
            for handler in handlers:
                root_logger.addHandler(handler)

        self._configured = True

    def _configure_default_logging(self) -> None:
        """Configure default logging settings."""
        level = "INFO"

        # Try to get level from configuration service
        if self._config_service:
            try:
                level = self._config_service.get_setting("logging.level", "INFO")
            except Exception:
                pass  # Use default if config service fails

        self.configure_logging(level=level)


class ConfigurationService(IConfigurationService):
    """Configuration service that integrates with the existing Pydantic Settings system."""

    def __init__(self):
        self._settings = None
        self._logger = logging.getLogger(__name__)

    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration setting by key with dot notation support.

        Args:
            key: Configuration key (supports dot notation like 'providers.openai.api_key')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        try:
            settings = self._get_settings()

            # Handle dot notation
            value = settings
            for part in key.split("."):
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default

            return value
        except Exception as e:
            self._logger.debug(f"Failed to get setting '{key}': {e}")
            return default

    def get_environment_variable(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get an environment variable with optional default.

        This centralizes environment variable access, making it easier to mock for testing.

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            Environment variable value or default
        """
        return os.getenv(key, default)

    def get_provider_config(self, provider_name: str) -> Dict[str | Any]:
        """
        Get configuration for a specific provider.

        Args:
            provider_name: Name of the provider (e.g., 'openai', 'anthropic')

        Returns:
            Provider configuration dictionary
        """
        try:
            settings = self._get_settings()

            # Try to get provider-specific config
            provider_config = getattr(settings, "providers", {}).get(provider_name, {})

            # Add API key from environment if not in config
            api_key_env = f"{provider_name.upper()}_API_KEY"
            if "api_key" not in provider_config:
                api_key = self.get_environment_variable(api_key_env)
                if api_key:
                    provider_config = {**provider_config, "api_key": api_key}

            return provider_config
        except Exception as e:
            self._logger.debug(f"Failed to get provider config for '{provider_name}': {e}")
            return {}

    def get_model_parameters(self, model_name: Optional[str] = None) -> Dict[str | Any]:
        """
        Get model parameters configuration.

        Args:
            model_name: Specific model name, or None for defaults

        Returns:
            Model parameters dictionary
        """
        try:
            settings = self._get_settings()

            # Get default model parameters
            if hasattr(settings, "model_parameters"):
                return settings.model_parameters.dict()
            else:
                # Fallback defaults
                return {
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "top_p": 1.0,
                }
        except Exception as e:
            self._logger.debug(f"Failed to get model parameters: {e}")
            return {
                "temperature": 0.7,
                "max_tokens": 1000,
                "top_p": 1.0,
            }

    def get_network_config(self) -> Dict[str | Any]:
        """
        Get network-related configuration.

        Returns:
            Network configuration dictionary
        """
        try:
            settings = self._get_settings()

            if hasattr(settings, "network"):
                return settings.network.dict()
            else:
                # Fallback defaults
                return {
                    "default_timeout": 30,
                    "max_retries": 3,
                    "retry_delay": 1.0,
                }
        except Exception as e:
            self._logger.debug(f"Failed to get network config: {e}")
            return {
                "default_timeout": 30,
                "max_retries": 3,
                "retry_delay": 1.0,
            }

    def reload_configuration(self) -> None:
        """Reload configuration from all sources."""
        self._settings = None
        self._get_settings()  # Force reload

    def _get_settings(self):
        """Get settings instance, loading if necessary."""
        if self._settings is None:
            try:
                # Import here to avoid circular imports
                from ..config.settings import get_settings

                self._settings = get_settings()
            except ImportError:
                self._logger.warning("Could not import settings, using fallback configuration")
                self._settings = {}  # Fallback to empty dict

        return self._settings


class RequestsHttpClient(IHttpClient):
    """HTTP client implementation using the requests library."""

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        timeout: Optional[int] = None,
        retries: int = 3,
    ):
        self._session = session or requests.Session()
        self._timeout = timeout

        # Configure retry strategy
        retry_strategy = Retry(
            total=retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        """Make a GET request."""
        kwargs.setdefault("timeout", self._timeout)
        return self._session.get(url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> requests.Response:
        """Make a POST request."""
        kwargs.setdefault("timeout", self._timeout)
        return self._session.post(url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> requests.Response:
        """Make a PUT request."""
        kwargs.setdefault("timeout", self._timeout)
        return self._session.put(url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> requests.Response:
        """Make a DELETE request."""
        kwargs.setdefault("timeout", self._timeout)
        return self._session.delete(url, **kwargs)

    def close(self) -> None:
        """Close the HTTP client and clean up resources."""
        self._session.close()


class HttpClientFactory(IHttpClientFactory):
    """Factory for creating HTTP clients with provider-specific configurations."""

    def __init__(self, config_service: Optional[IConfigurationService] = None):
        self._config_service = config_service
        self._logger = logging.getLogger(__name__)

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
        session = requests.Session()

        if base_url:
            # Note: requests doesn't have a built-in base_url, but we can add it to session
            session.mount(base_url, HTTPAdapter())

        if headers:
            session.headers.update(headers)

        # Get default timeout from config if not provided
        if timeout is None and self._config_service:
            network_config = self._config_service.get_network_config()
            timeout = network_config.get("default_timeout", 30)

        return RequestsHttpClient(session=session, timeout=timeout)

    def create_provider_client(self, provider_name: str) -> IHttpClient:
        """
        Create an HTTP client configured for a specific provider.

        Args:
            provider_name: Name of the provider (e.g., 'openai', 'anthropic')

        Returns:
            HTTP client configured for the provider
        """
        config = {}
        timeout = 30
        headers = {}

        # Get provider-specific configuration
        if self._config_service:
            provider_config = self._config_service.get_provider_config(provider_name)
            timeout = provider_config.get("timeout", 30)

            # Add authentication headers
            api_key = provider_config.get("api_key")
            if api_key:
                if provider_name == "openai":
                    headers["Authorization"] = f"Bearer {api_key}"
                elif provider_name == "anthropic":
                    headers["x-api-key"] = api_key
                    headers["anthropic-version"] = "2023-06-01"
                elif provider_name == "google":
                    headers["Authorization"] = f"Bearer {api_key}"

        return self.create_client(timeout=timeout, headers=headers)


class ProviderFactory(IProviderFactory):
    """Factory for creating LLM provider instances with dependency injection."""

    def __init__(
        self,
        config_service: IConfigurationService,
        http_client_factory: IHttpClientFactory,
        logger_factory: ILoggerFactory,
    ):
        self._config_service = config_service
        self._http_client_factory = http_client_factory
        self._logger_factory = logger_factory
        self._logger = self._logger_factory.get_logger(__name__)

        # Cache of available providers
        self._providers_cache: Optional[Dict[str, Any]] = None

    def create_provider(self, provider_name: str, model_name: str, **config: Any) -> ILLMProvider:
        """
        Create an LLM provider instance with dependency injection.

        Args:
            provider_name: Name of the provider (e.g., 'openai', 'anthropic')
            model_name: Name of the model to use
            **config: Additional provider configuration

        Returns:
            Configured provider instance
        """
        try:
            # Get provider class
            provider_class = self._get_provider_class(provider_name)
            if provider_class is None:
                raise ValueError(f"Provider '{provider_name}' is not available")

            # Get configuration
            provider_config = self._config_service.get_provider_config(provider_name)
            provider_config.update(config)  # Override with provided config

            # Create HTTP client for the provider
            http_client = self._http_client_factory.create_provider_client(provider_name)

            # Create logger for the provider
            logger = self._logger_factory.get_logger(f"providers.{provider_name}")

            # Inject dependencies into provider
            # Note: This is a simplified implementation - in practice, you'd need
            # to modify the provider classes to accept these dependencies
            provider_config["_http_client"] = http_client
            provider_config["_logger"] = logger

            # Create provider instance
            provider = provider_class(model_name, **provider_config)

            self._logger.info(f"Created {provider_name} provider for model {model_name}")
            return provider

        except Exception as e:
            self._logger.error(f"Failed to create {provider_name} provider: {e}")
            raise

    def get_available_providers(self) -> List[str]:
        """
        Get list of available provider names.

        Returns:
            List of provider names
        """
        providers = self._get_providers()
        return list(providers.keys())

    def get_provider_models(self, provider_name: str) -> List[str]:
        """
        Get list of models supported by a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            List of supported model names
        """
        providers = self._get_providers()
        provider_class = providers.get(provider_name)

        if provider_class and hasattr(provider_class, "SUPPORTED_MODELS"):
            return provider_class.SUPPORTED_MODELS
        else:
            return []

    def _get_providers(self) -> Dict[str | Any]:
        """Get dictionary of available providers."""
        if self._providers_cache is None:
            try:
                # Import provider registry
                from ..providers.registry import registry

                self._providers_cache = registry._providers
            except ImportError:
                self._logger.warning("Could not import provider registry")
                self._providers_cache = {}

        return self._providers_cache

    def _get_provider_class(self, provider_name: str) -> Optional[type]:
        """Get provider class by name."""
        providers = self._get_providers()
        return providers.get(provider_name)


class SimpleFileService(IFileService):
    """Simple file service implementation using standard Python file operations."""

    def __init__(self, logger_factory: Optional[ILoggerFactory] = None):
        if logger_factory:
            self._logger = logger_factory.get_logger(__name__)
        else:
            self._logger = LoggerAdapter(logging.getLogger(__name__))

    def read_file(self, path: str | Path, encoding: str = "utf-8") -> str:
        """Read file contents."""
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except Exception as e:
            self._logger.error(f"Failed to read file {path}: {e}")
            raise

    def write_file(self, path: str | Path, content: str, encoding: str = "utf-8") -> None:
        """Write content to file."""
        try:
            # Create parent directories if they don't exist
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding=encoding) as f:
                f.write(content)
        except Exception as e:
            self._logger.error(f"Failed to write file {path}: {e}")
            raise

    def file_exists(self, path: str | Path) -> bool:
        """Check if file exists."""
        return Path(path).exists()

    def create_directory(self, path: str | Path, exist_ok: bool = True) -> None:
        """Create directory."""
        try:
            Path(path).mkdir(parents=True, exist_ok=exist_ok)
        except Exception as e:
            self._logger.error(f"Failed to create directory {path}: {e}")
            raise


class InMemoryCacheService(ICacheService):
    """Simple in-memory cache implementation using generic cache base."""

    def __init__(self, max_size: Optional[int] = None):
        self._generic_cache: GenericCache[str, Any] = GenericCache(max_size)
        self._logger = logging.getLogger(__name__)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self._generic_cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache. TTL is ignored in this simple implementation."""
        self._generic_cache.set(key, value)
        self._logger.debug(f"Cached value for key: {key}")

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        result = self._generic_cache.delete(key)
        if result:
            self._logger.debug(f"Deleted cache key: {key}")
        return result

    def clear(self) -> None:
        """Clear all cached values."""
        self._generic_cache.clear()
        self._logger.debug("Cache cleared")

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self._generic_cache.get(key) is not None

    def keys(self) -> List[str]:
        """Get all cache keys."""
        return self._generic_cache.keys()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._generic_cache.keys())
