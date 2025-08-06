"""
Testing Utilities for Dependency Injection

This module provides utilities for testing code that uses dependency injection,
including mock implementations and test container setup.
"""

import logging
from typing import Any, Callable, Dict, Optional, Type, TypeVar
from unittest.mock import MagicMock, Mock

from .container import DIContainer, ServiceLifetime
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

T = TypeVar("T")


class MockLogger(ILogger):
    """Mock logger implementation that captures log messages for testing."""

    def __init__(self):
        self.debug_messages = []
        self.info_messages = []
        self.warning_messages = []
        self.error_messages = []
        self.critical_messages = []
        self.exception_messages = []

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Capture debug messages."""
        formatted_message = message % args if args else str(message)
        self.debug_messages.append(formatted_message)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Capture info messages."""
        formatted_message = message % args if args else str(message)
        self.info_messages.append(formatted_message)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Capture warning messages."""
        formatted_message = message % args if args else str(message)
        self.warning_messages.append(formatted_message)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Capture error messages."""
        formatted_message = message % args if args else str(message)
        self.error_messages.append(formatted_message)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Capture critical messages."""
        formatted_message = message % args if args else str(message)
        self.critical_messages.append(formatted_message)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Capture exception messages."""
        formatted_message = message % args if args else str(message)
        self.exception_messages.append(formatted_message)

    def get_all_messages(self) -> Dict[str, list]:
        """Get all captured messages."""
        return {
            "debug": self.debug_messages,
            "info": self.info_messages,
            "warning": self.warning_messages,
            "error": self.error_messages,
            "critical": self.critical_messages,
            "exception": self.exception_messages,
        }

    def clear(self) -> None:
        """Clear all captured messages."""
        self.debug_messages.clear()
        self.info_messages.clear()
        self.warning_messages.clear()
        self.error_messages.clear()
        self.critical_messages.clear()
        self.exception_messages.clear()


class MockLoggerFactory(ILoggerFactory):
    """Mock logger factory that creates MockLogger instances."""

    def __init__(self):
        self._loggers: Dict[str, MockLogger] = {}

    def get_logger(self, name: str) -> ILogger:
        """Get or create a mock logger for the specified name."""
        if name not in self._loggers:
            self._loggers[name] = MockLogger()
        return self._loggers[name]

    def configure_logging(
        self,
        level: str = "INFO",
        format_string: Optional[str] = None,
        handlers: Optional[list] = None,
    ) -> None:
        """Mock implementation - does nothing."""
        pass

    def get_all_loggers(self) -> Dict[str, MockLogger]:
        """Get all created loggers for testing."""
        return self._loggers.copy()


class MockConfigurationService(IConfigurationService):
    """Mock configuration service for testing."""

    def __init__(self, initial_config: Optional[Dict[str, Any]] = None):
        self._config = initial_config or {}
        self._env_vars = {}

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a mock configuration setting."""
        # Handle dot notation
        value = self._config
        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def get_environment_variable(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a mock environment variable."""
        return self._env_vars.get(key, default)

    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """Get mock provider configuration."""
        return self._config.get("providers", {}).get(provider_name, {})

    def get_model_parameters(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get mock model parameters."""
        return self._config.get(
            "model_parameters",
            {
                "temperature": 0.7,
                "max_tokens": 1000,
                "top_p": 1.0,
            },
        )

    def get_network_config(self) -> Dict[str, Any]:
        """Get mock network configuration."""
        return self._config.get(
            "network",
            {
                "default_timeout": 30,
                "max_retries": 3,
                "retry_delay": 1.0,
            },
        )

    def reload_configuration(self) -> None:
        """Mock implementation - does nothing."""
        pass

    def set_config(self, key: str, value: Any) -> None:
        """Set a configuration value for testing."""
        # Handle dot notation
        parts = key.split(".")
        config = self._config
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]
        config[parts[-1]] = value

    def set_env_var(self, key: str, value: str) -> None:
        """Set an environment variable for testing."""
        self._env_vars[key] = value


class MockHttpClient(IHttpClient):
    """Mock HTTP client for testing."""

    def __init__(self):
        self.requests = []
        self.responses = {}
        self.default_response = Mock()

    def get(self, url: str, **kwargs: Any) -> Any:
        """Mock GET request."""
        self.requests.append(("GET", url, kwargs))
        return self.responses.get(("GET", url), self.default_response)

    def post(self, url: str, **kwargs: Any) -> Any:
        """Mock POST request."""
        self.requests.append(("POST", url, kwargs))
        return self.responses.get(("POST", url), self.default_response)

    def put(self, url: str, **kwargs: Any) -> Any:
        """Mock PUT request."""
        self.requests.append(("PUT", url, kwargs))
        return self.responses.get(("PUT", url), self.default_response)

    def delete(self, url: str, **kwargs: Any) -> Any:
        """Mock DELETE request."""
        self.requests.append(("DELETE", url, kwargs))
        return self.responses.get(("DELETE", url), self.default_response)

    def close(self) -> None:
        """Mock close - does nothing."""
        pass

    def set_response(self, method: str, url: str, response: Any) -> None:
        """Set a mock response for a specific request."""
        self.responses[(method, url)] = response


class MockProvider(ILLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, model_name: str = "mock-model", provider_name: str = "mock"):
        self._model_name = model_name
        self._provider_name = provider_name
        self.generation_calls = []
        self.responses = ["Mock response"]
        self._response_index = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Mock generation."""
        self.generation_calls.append((prompt, kwargs))
        response = self.responses[self._response_index % len(self.responses)]
        self._response_index += 1
        return response

    def batch_generate(self, prompts: list, **kwargs: Any) -> list:
        """Mock batch generation."""
        return [self.generate(prompt, **kwargs) for prompt in prompts]

    def get_model_info(self) -> Dict[str, Any]:
        """Mock model info."""
        return {
            "model_name": self._model_name,
            "provider": self._provider_name,
            "max_tokens": 1000,
        }

    def validate_credentials(self) -> bool:
        """Mock credential validation."""
        return True

    def set_responses(self, responses: list) -> None:
        """Set mock responses for testing."""
        self.responses = responses
        self._response_index = 0


def create_test_container() -> DIContainer:
    """
    Create a DI container configured for testing.

    This container includes mock implementations of all major services,
    making it easy to write isolated unit tests.

    Returns:
        DIContainer configured for testing
    """
    container = DIContainer()

    # Register mock services
    container.register_singleton(IConfigurationService, MockConfigurationService())
    container.register_singleton(ILoggerFactory, MockLoggerFactory())

    # Create mock HTTP client factory
    mock_http_factory = Mock(spec=IHttpClientFactory)
    mock_http_factory.create_client.return_value = MockHttpClient()
    mock_http_factory.create_provider_client.return_value = MockHttpClient()
    container.register_instance(IHttpClientFactory, mock_http_factory)

    # Create mock provider factory
    mock_provider_factory = Mock(spec=IProviderFactory)
    mock_provider_factory.get_available_providers.return_value = ["mock"]
    mock_provider_factory.get_provider_models.return_value = ["mock-model"]
    mock_provider_factory.create_provider.return_value = MockProvider()
    container.register_instance(IProviderFactory, mock_provider_factory)

    return container


def create_mock_service(interface: Type[T], **kwargs: Any) -> T:
    """
    Create a mock implementation of a service interface.

    Args:
        interface: The interface type to mock
        **kwargs: Additional attributes to set on the mock

    Returns:
        Mock instance implementing the interface
    """
    mock = MagicMock(spec=interface)

    # Set any provided attributes
    for key, value in kwargs.items():
        setattr(mock, key, value)

    return mock


class ContainerTestContext:
    """
    Context manager for testing with a DI container.

    This automatically sets up and tears down a test container,
    making it easy to write tests that use dependency injection.

    Usage:
        with ContainerTestContext() as container:
            # Test code here
            service = container.get(IConfigurationService)
    """

    def __init__(self, container: Optional[DIContainer] = None):
        self._container = container or create_test_container()
        self._original_container = None

    def __enter__(self) -> DIContainer:
        # Save the original global container and replace with test container
        from . import _container, reset_container

        global _container
        self._original_container = _container
        _container = self._container
        return self._container

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the original container
        from . import _container

        global _container
        _container = self._original_container


def inject_mock(container: DIContainer, interface: Type[T], mock_instance: T) -> None:
    """
    Inject a mock instance into a container for testing.

    Args:
        container: The DI container
        interface: The interface type
        mock_instance: The mock instance to inject
    """
    container.register_instance(interface, mock_instance)


def assert_service_called(mock_service: Mock, method_name: str, *args, **kwargs) -> None:
    """
    Assert that a method was called on a mock service.

    Args:
        mock_service: The mock service instance
        method_name: Name of the method that should have been called
        *args: Expected positional arguments
        **kwargs: Expected keyword arguments
    """
    method_mock = getattr(mock_service, method_name)
    if args and kwargs:
        method_mock.assert_called_with(*args, **kwargs)
    elif args:
        method_mock.assert_called_with(*args)
    elif kwargs:
        method_mock.assert_called_with(**kwargs)
    else:
        method_mock.assert_called()


def capture_logs(logger_factory: MockLoggerFactory, logger_name: str) -> MockLogger:
    """
    Capture logs from a specific logger for testing.

    Args:
        logger_factory: Mock logger factory instance
        logger_name: Name of the logger to capture

    Returns:
        MockLogger instance for assertions
    """
    return logger_factory.get_logger(logger_name)
