"""
Integration with Existing LLM Lab Systems

This module provides integration points between the new dependency injection
system and the existing LLM Lab infrastructure, enabling gradual migration
and backward compatibility.
"""

import inspect
import logging
import os
from functools import wraps
from typing import Any, Dict, Optional, Type, TypeVar

from .container import DIContainer
from .interfaces import (
    IConfigurationService,
    IHttpClientFactory,
    ILLMProvider,
    ILogger,
    ILoggerFactory,
    IProviderFactory,
)
from .services import ConfigurationService, HttpClientFactory, LoggerFactory

T = TypeVar("T")
logger = logging.getLogger(__name__)


def with_dependency_injection(func: callable) -> callable:
    """
    Decorator to enable dependency injection for existing functions.

    This decorator can be applied to existing functions to automatically
    inject dependencies without changing the function signature.

    Usage:
        @with_dependency_injection
        def some_function(prompt: str, provider_name: str = "openai"):
            # This function can now access injected services
            config = get_injected_service(IConfigurationService)
            logger = get_injected_service(ILogger)
            # ... rest of function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        from . import get_container

        # Store container in thread-local storage for access within the function
        container = get_container()
        _current_container.value = container

        try:
            return func(*args, **kwargs)
        finally:
            _current_container.value = None

    return wrapper


# Thread-local storage for current container
import threading

_current_container = threading.local()


def get_injected_service(interface: Type[T]) -> T:
    """
    Get an injected service within a function decorated with @with_dependency_injection.

    Args:
        interface: The service interface to retrieve

    Returns:
        Instance of the requested service

    Raises:
        RuntimeError: If called outside a DI-enabled function
    """
    container = getattr(_current_container, "value", None)
    if container is None:
        raise RuntimeError(
            "get_injected_service() can only be called within a function decorated with @with_dependency_injection"
        )

    return container.get(interface)


class ProviderAdapter:
    """
    Adapter that makes existing LLM providers work with the DI system.

    This allows existing provider classes to be used with dependency injection
    without requiring immediate refactoring.
    """

    def __init__(
        self,
        provider_class: Type,
        config_service: IConfigurationService,
        logger_factory: ILoggerFactory,
        http_client_factory: IHttpClientFactory,
    ):
        self._provider_class = provider_class
        self._config_service = config_service
        self._logger_factory = logger_factory
        self._http_client_factory = http_client_factory

    def create_provider(self, model_name: str, **kwargs: Any) -> ILLMProvider:
        """
        Create a provider instance with injected dependencies.

        Args:
            model_name: The model name to use
            **kwargs: Additional configuration parameters

        Returns:
            Provider instance with dependencies injected
        """
        # Get provider name from class
        provider_name = self._provider_class.__name__.replace("Provider", "").lower()

        # Get configuration from config service
        provider_config = self._config_service.get_provider_config(provider_name)
        provider_config.update(kwargs)  # Override with provided args

        # Create logger for this provider
        logger = self._logger_factory.get_logger(f"providers.{provider_name}")

        # Create HTTP client
        http_client = self._http_client_factory.create_provider_client(provider_name)

        # Inject dependencies into the provider config
        provider_config["_injected_logger"] = logger
        provider_config["_injected_http_client"] = http_client
        provider_config["_injected_config_service"] = self._config_service

        # Create the provider instance
        try:
            provider = self._provider_class(model_name, **provider_config)

            # If the provider has methods to set injected dependencies, call them
            if hasattr(provider, "set_logger"):
                provider.set_logger(logger)
            if hasattr(provider, "set_http_client"):
                provider.set_http_client(http_client)
            if hasattr(provider, "set_config_service"):
                provider.set_config_service(self._config_service)

            logger.info(f"Created {provider_name} provider with dependency injection")
            return provider

        except Exception as e:
            logger.error(f"Failed to create {provider_name} provider: {e}")
            raise


class BackwardCompatibilityService:
    """
    Service that provides backward compatibility with existing code patterns.

    This service allows existing code to continue working while gradually
    migrating to the new dependency injection system.
    """

    def __init__(self, container: DIContainer):
        self._container = container
        self._logger = self._get_logger()

    def get_legacy_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using legacy patterns.

        This method tries multiple approaches to maintain compatibility:
        1. DI configuration service
        2. Environment variables
        3. Legacy configuration files
        4. Fallback defaults
        """
        try:
            # Try DI configuration service first
            config_service = self._container.get(IConfigurationService)
            value = config_service.get_setting(key, default)
            if value != default:
                return value
        except Exception:
            pass

        # Try environment variable
        env_key = key.upper().replace(".", "_")
        env_value = os.getenv(env_key)
        if env_value is not None:
            return env_value

        # Try legacy configuration system
        try:
            from ..config.settings import get_settings

            settings = get_settings()

            # Navigate through dot notation
            value = settings
            for part in key.split("."):
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        except Exception:
            pass

        return default

    def get_legacy_logger(self, name: str) -> logging.Logger:
        """
        Get a logger using legacy patterns, with DI fallback.

        Args:
            name: Logger name

        Returns:
            Logger instance (standard Python logger for compatibility)
        """
        try:
            # Try to get logger from DI system
            logger_factory = self._container.get(ILoggerFactory)
            di_logger = logger_factory.get_logger(name)

            # If it's a LoggerAdapter, return the underlying logger for compatibility
            if hasattr(di_logger, "_logger"):
                return di_logger._logger

        except Exception:
            pass

        # Fallback to standard logging
        return logging.getLogger(name)

    def create_legacy_provider(self, provider_name: str, model_name: str, **kwargs: Any) -> Any:
        """
        Create a provider using legacy patterns with DI enhancement.

        Args:
            provider_name: Name of the provider
            model_name: Model name to use
            **kwargs: Additional configuration

        Returns:
            Provider instance
        """
        try:
            # Try to use DI provider factory
            provider_factory = self._container.get(IProviderFactory)
            return provider_factory.create_provider(provider_name, model_name, **kwargs)
        except Exception as e:
            self._logger.debug(f"DI provider creation failed, falling back to legacy: {e}")

            # Fallback to legacy provider creation
            return self._create_legacy_provider_fallback(provider_name, model_name, **kwargs)

    def _create_legacy_provider_fallback(
        self, provider_name: str, model_name: str, **kwargs: Any
    ) -> Any:
        """Fallback provider creation using legacy patterns."""
        try:
            # Import provider classes directly
            if provider_name.lower() == "anthropic":
                from ..providers.anthropic import AnthropicProvider

                return AnthropicProvider(model_name, **kwargs)
            elif provider_name.lower() == "openai":
                from ..providers.openai import OpenAIProvider

                return OpenAIProvider(model_name, **kwargs)
            elif provider_name.lower() == "google":
                from ..providers.google import GoogleProvider

                return GoogleProvider(model_name, **kwargs)
            else:
                raise ValueError(f"Unknown provider: {provider_name}")
        except Exception as e:
            self._logger.error(f"Failed to create legacy provider {provider_name}: {e}")
            raise

    def _get_logger(self) -> logging.Logger:
        """Get a logger for this service."""
        try:
            logger_factory = self._container.get(ILoggerFactory)
            di_logger = logger_factory.get_logger(__name__)
            if hasattr(di_logger, "_logger"):
                return di_logger._logger
        except Exception:
            pass

        return logging.getLogger(__name__)


def create_compatibility_layer(
    container: Optional[DIContainer] = None,
) -> BackwardCompatibilityService:
    """
    Create a backward compatibility service.

    Args:
        container: DI container to use (if None, uses global container)

    Returns:
        BackwardCompatibilityService instance
    """
    if container is None:
        from . import get_container

        container = get_container()

    return BackwardCompatibilityService(container)


class MigrationHelper:
    """
    Helper class for migrating existing code to use dependency injection.

    This provides utilities and patterns for gradually converting existing
    code to use the DI system.
    """

    @staticmethod
    def wrap_existing_function(func: callable, inject_params: Dict[str, Type]) -> callable:
        """
        Wrap an existing function to inject specified dependencies.

        Args:
            func: The function to wrap
            inject_params: Dictionary mapping parameter names to service interfaces

        Returns:
            Wrapped function with dependency injection

        Usage:
            def old_function(config_value, logger):
                pass

            new_function = MigrationHelper.wrap_existing_function(
                old_function,
                {
                    'config_value': lambda: get_injected_service(IConfigurationService).get_setting('key'),
                    'logger': lambda: get_injected_service(ILogger)
                }
            )
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            from . import get_container

            container = get_container()

            # Inject dependencies for missing parameters
            signature = inspect.signature(func)
            bound_args = signature.bind_partial(*args, **kwargs)

            for param_name, service_type in inject_params.items():
                if param_name not in bound_args.arguments:
                    try:
                        if callable(service_type):
                            # If it's a callable, call it to get the dependency
                            dependency = service_type()
                        else:
                            # Otherwise, treat it as a service interface
                            dependency = container.get(service_type)
                        bound_args.arguments[param_name] = dependency
                    except Exception as e:
                        logger.warning(f"Failed to inject {param_name}: {e}")

            bound_args.apply_defaults()
            return func(*bound_args.args, **bound_args.kwargs)

        return wrapper

    @staticmethod
    def convert_global_access_to_injection(module, global_mappings: Dict[str, Type]) -> None:
        """
        Convert global variable access to dependency injection.

        Args:
            module: The module to modify
            global_mappings: Dictionary mapping global variable names to service interfaces

        Usage:
            # Convert globals like 'config' and 'logger' to DI
            MigrationHelper.convert_global_access_to_injection(
                my_module,
                {
                    'config': IConfigurationService,
                    'logger': ILogger
                }
            )
        """
        from . import get_container

        container = get_container()

        for global_name, service_type in global_mappings.items():
            try:
                service_instance = container.get(service_type)
                setattr(module, global_name, service_instance)
                logger.info(f"Replaced global '{global_name}' with DI service")
            except Exception as e:
                logger.warning(f"Failed to replace global '{global_name}': {e}")


def setup_integration_layer(container: Optional[DIContainer] = None) -> None:
    """
    Set up the integration layer to connect DI with existing systems.

    This function should be called during application startup to ensure
    that the DI system is properly integrated with existing code.

    Args:
        container: DI container to configure (if None, uses global container)
    """
    if container is None:
        from . import get_container

        container = get_container()

    logger.info("Setting up DI integration layer")

    # Ensure core services are registered
    if not container.is_registered(IConfigurationService):
        container.register_singleton(IConfigurationService, ConfigurationService)

    if not container.is_registered(ILoggerFactory):
        container.register_singleton(ILoggerFactory, LoggerFactory)

    if not container.is_registered(IHttpClientFactory):
        container.register_singleton(IHttpClientFactory, HttpClientFactory)

    # Set up provider adapters for existing providers
    _setup_provider_adapters(container)

    logger.info("DI integration layer setup complete")


def _setup_provider_adapters(container: DIContainer) -> None:
    """Set up adapters for existing provider classes."""
    try:
        config_service = container.get(IConfigurationService)
        logger_factory = container.get(ILoggerFactory)
        http_client_factory = container.get(IHttpClientFactory)

        # Create adapters for existing providers
        provider_classes = []

        try:
            from ..providers.anthropic import AnthropicProvider

            provider_classes.append(AnthropicProvider)
        except ImportError:
            pass

        try:
            from ..providers.google import GoogleProvider

            provider_classes.append(GoogleProvider)
        except ImportError:
            pass

        # Register provider adapters
        for provider_class in provider_classes:
            adapter = ProviderAdapter(
                provider_class, config_service, logger_factory, http_client_factory
            )
            # You could register these adapters in the container if needed

        logger.info(f"Set up adapters for {len(provider_classes)} provider classes")

    except Exception as e:
        logger.warning(f"Failed to set up provider adapters: {e}")


# Global compatibility service instance
_compatibility_service: Optional[BackwardCompatibilityService] = None


def get_compatibility_service() -> BackwardCompatibilityService:
    """Get the global compatibility service instance."""
    global _compatibility_service
    if _compatibility_service is None:
        _compatibility_service = create_compatibility_layer()
    return _compatibility_service


# Convenience functions for backward compatibility
def get_config_value(key: str, default: Any = None) -> Any:
    """Get a configuration value with backward compatibility."""
    return get_compatibility_service().get_legacy_config_value(key, default)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with backward compatibility."""
    return get_compatibility_service().get_legacy_logger(name)


def create_provider(provider_name: str, model_name: str, **kwargs: Any) -> Any:
    """Create a provider with backward compatibility."""
    return get_compatibility_service().create_legacy_provider(provider_name, model_name, **kwargs)
