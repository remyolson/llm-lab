"""
Dependency Injection System for LLM Lab

This package provides a lightweight, type-safe dependency injection framework
designed specifically for the LLM Lab codebase. It enables loose coupling,
improved testability, and better separation of concerns.

Key Features:
- Protocol-based service abstractions
- Constructor injection with type hint resolution
- Singleton and transient service lifetimes
- Factory methods for complex service creation
- Backward compatibility with existing code
- Enhanced testing capabilities

Usage:
    from src.di import get_container, IConfigurationService

    # Get services from container
    container = get_container()
    config_service = container.get(IConfigurationService)

    # Or use decorator for automatic injection
    @inject
    def my_function(config: IConfigurationService):
        # config is automatically injected
        pass

The DI system is designed to be adopted gradually, allowing existing code
to continue working while new code takes advantage of dependency injection.
"""

from .container import DIContainer, ServiceLifetime, inject
from .interfaces import (
    IConfigurationService,
    IEvaluationService,
    IHttpClientFactory,
    ILLMProvider,
    ILogger,
    ILoggerFactory,
    IProviderFactory,
)
from .services import (
    ConfigurationService,
    HttpClientFactory,
    LoggerFactory,
    ProviderFactory,
)

# Global container instance - initialized lazily
_container: DIContainer = None


def get_container() -> DIContainer:
    """
    Get the global DI container instance.

    Returns:
        The global DIContainer instance
    """
    global _container
    if _container is None:
        _container = create_default_container()
    return _container


def create_default_container() -> DIContainer:
    """
    Create a DI container with default service registrations.

    This sets up the standard services needed by the LLM Lab framework,
    including configuration, logging, HTTP clients, and providers.

    Returns:
        Configured DIContainer instance
    """
    container = DIContainer()

    # Register core services
    container.register_singleton(IConfigurationService, ConfigurationService)
    container.register_singleton(ILoggerFactory, LoggerFactory)
    container.register_singleton(IHttpClientFactory, HttpClientFactory)
    container.register_transient(IProviderFactory, ProviderFactory)

    return container


def reset_container() -> None:
    """
    Reset the global container (primarily for testing).

    This clears the global container and forces recreation on next access.
    Useful for test isolation.
    """
    global _container
    _container = None


__all__ = [
    # Core DI infrastructure
    "DIContainer",
    "ServiceLifetime",
    "inject",
    "get_container",
    "create_default_container",
    "reset_container",
    # Service interfaces
    "IConfigurationService",
    "IHttpClientFactory",
    "ILogger",
    "ILoggerFactory",
    "ILLMProvider",
    "IEvaluationService",
    "IProviderFactory",
    # Service implementations
    "ConfigurationService",
    "HttpClientFactory",
    "LoggerFactory",
    "ProviderFactory",
]
