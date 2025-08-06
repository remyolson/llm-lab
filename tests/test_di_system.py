"""
Comprehensive Test Suite for Dependency Injection System

This module provides thorough testing of all DI system components including
container functionality, service registration, dependency resolution, and
integration with existing systems.
"""

from typing import Protocol, runtime_checkable
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import DI system components
from src.di import (
    DIContainer,
    ServiceLifetime,
    create_default_container,
    get_container,
    inject,
    reset_container,
)
from src.di.container import (
    CircularDependencyError,
    ServiceCreationError,
    ServiceNotRegisteredError,
)
from src.di.factories import (
    ServiceRegistry,
    create_service_builder,
    factory,
    injectable,
    lazy,
    service,
)
from src.di.integration import (
    get_injected_service,
    setup_integration_layer,
    with_dependency_injection,
)
from src.di.interfaces import (
    IConfigurationService,
    IHttpClientFactory,
    ILogger,
    ILoggerFactory,
    IProviderFactory,
)
from src.di.services import (
    ConfigurationService,
    HttpClientFactory,
    LoggerFactory,
    ProviderFactory,
)
from src.di.testing import (
    MockConfigurationService,
    MockLoggerFactory,
    TestContainerContext,
    create_test_container,
    inject_mock,
)


class TestDIContainer:
    """Test the core DI container functionality."""

    def test_container_creation(self):
        """Test basic container creation."""
        container = DIContainer()
        assert container is not None
        assert isinstance(container, DIContainer)

    def test_singleton_registration_and_resolution(self):
        """Test singleton service registration and resolution."""
        container = DIContainer()

        # Register service
        container.register_singleton(IConfigurationService, ConfigurationService)

        # Resolve service twice
        service1 = container.get(IConfigurationService)
        service2 = container.get(IConfigurationService)

        # Should be the same instance
        assert service1 is service2
        assert isinstance(service1, ConfigurationService)

    def test_transient_registration_and_resolution(self):
        """Test transient service registration and resolution."""
        container = DIContainer()

        # Create a simple service class for testing
        class TestService:
            def __init__(self):
                self.id = id(self)

        container.register_transient(TestService, TestService)

        # Resolve service twice
        service1 = container.get(TestService)
        service2 = container.get(TestService)

        # Should be different instances
        assert service1 is not service2
        assert service1.id != service2.id

    def test_instance_registration(self):
        """Test registering specific instances."""
        container = DIContainer()

        # Create a test instance
        test_instance = Mock()
        container.register_instance(Mock, test_instance)

        # Resolve service
        resolved = container.get(Mock)

        # Should be the same instance
        assert resolved is test_instance

    def test_factory_registration(self):
        """Test factory function registration."""
        container = DIContainer()

        # Create factory function
        def create_service():
            service = Mock()
            service.created_by_factory = True
            return service

        container.register_factory(Mock, create_service, ServiceLifetime.TRANSIENT)

        # Resolve service
        service = container.get(Mock)

        assert service.created_by_factory is True

    def test_service_not_registered_error(self):
        """Test error when resolving unregistered service."""
        container = DIContainer()

        with pytest.raises(ServiceNotRegisteredError):
            container.get(IConfigurationService)

    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        container = DIContainer()

        # Create classes with circular dependencies
        class ServiceA:
            def __init__(self, service_b: "ServiceB"):
                self.service_b = service_b

        class ServiceB:
            def __init__(self, service_a: ServiceA):
                self.service_a = service_a

        container.register_transient(ServiceA, ServiceA)
        container.register_transient(ServiceB, ServiceB)

        with pytest.raises(CircularDependencyError):
            container.get(ServiceA)

    def test_constructor_dependency_injection(self):
        """Test automatic constructor dependency injection."""
        container = DIContainer()

        # Set up dependencies
        mock_config = Mock(spec=IConfigurationService)
        container.register_instance(IConfigurationService, mock_config)

        # Create service that depends on config
        class ServiceWithDependency:
            def __init__(self, config: IConfigurationService):
                self.config = config

        container.register_transient(ServiceWithDependency, ServiceWithDependency)

        # Resolve service
        service = container.get(ServiceWithDependency)

        assert service.config is mock_config

    def test_is_registered(self):
        """Test service registration checking."""
        container = DIContainer()

        assert not container.is_registered(IConfigurationService)

        container.register_singleton(IConfigurationService, ConfigurationService)

        assert container.is_registered(IConfigurationService)

    def test_try_get(self):
        """Test optional service resolution."""
        container = DIContainer()

        # Should return None for unregistered service
        result = container.try_get(IConfigurationService)
        assert result is None

        # Should return service for registered service
        container.register_singleton(IConfigurationService, ConfigurationService)
        result = container.try_get(IConfigurationService)
        assert isinstance(result, ConfigurationService)


class TestInjectDecorator:
    """Test the @inject decorator functionality."""

    def test_inject_decorator_basic(self):
        """Test basic @inject decorator functionality."""
        container = create_test_container()

        # Replace global container for test
        with patch("src.di.get_container", return_value=container):

            @inject
            def test_function(
                value: str, config: IConfigurationService, logger_factory: ILoggerFactory
            ):
                return f"{value}_{type(config).__name__}_{type(logger_factory).__name__}"

            result = test_function("test")

            assert "test" in result
            assert "MockConfigurationService" in result
            assert "MockLoggerFactory" in result

    def test_inject_decorator_with_provided_args(self):
        """Test @inject decorator when some args are provided."""
        container = create_test_container()

        with patch("src.di.get_container", return_value=container):

            @inject
            def test_function(
                value: str, config: IConfigurationService, optional_param: str = "default"
            ):
                return f"{value}_{type(config).__name__}_{optional_param}"

            # Provide some arguments
            result = test_function("test", optional_param="custom")

            assert "test_MockConfigurationService_custom" == result

    def test_inject_decorator_missing_service(self):
        """Test @inject decorator with missing service."""
        container = DIContainer()  # Empty container

        with patch("src.di.get_container", return_value=container):

            @inject
            def test_function(config: IConfigurationService):
                return "should not reach here"

            # Should still work but skip unresolvable dependencies
            test_function()  # Should not raise exception


class TestServiceImplementations:
    """Test the concrete service implementations."""

    def test_configuration_service(self):
        """Test ConfigurationService functionality."""
        config_service = ConfigurationService()

        # Test environment variable access
        with patch.dict("os.environ", {"TEST_VAR": "test_value"}):
            result = config_service.get_environment_variable("TEST_VAR")
            assert result == "test_value"

        # Test with default
        result = config_service.get_environment_variable("NONEXISTENT", "default")
        assert result == "default"

    def test_logger_factory(self):
        """Test LoggerFactory functionality."""
        logger_factory = LoggerFactory()

        # Get logger
        logger = logger_factory.get_logger("test_logger")

        assert logger is not None
        # Logger should implement ILogger protocol
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")

    def test_http_client_factory(self):
        """Test HttpClientFactory functionality."""
        config_service = MockConfigurationService()
        http_factory = HttpClientFactory(config_service)

        # Create client
        client = http_factory.create_client(timeout=30)

        assert client is not None
        assert hasattr(client, "get")
        assert hasattr(client, "post")


class TestTestingUtilities:
    """Test the DI testing utilities."""

    def test_test_container_context(self):
        """Test TestContainerContext."""
        with TestContainerContext() as container:
            # Should have mock services registered
            config = container.get(IConfigurationService)
            assert isinstance(config, MockConfigurationService)

            logger_factory = container.get(ILoggerFactory)
            assert isinstance(logger_factory, MockLoggerFactory)

    def test_mock_configuration_service(self):
        """Test MockConfigurationService."""
        mock_config = MockConfigurationService()

        # Set configuration
        mock_config.set_config("test.key", "test_value")

        # Get configuration
        result = mock_config.get_setting("test.key")
        assert result == "test_value"

        # Test environment variables
        mock_config.set_env_var("TEST_ENV", "env_value")
        result = mock_config.get_environment_variable("TEST_ENV")
        assert result == "env_value"

    def test_mock_logger_factory(self):
        """Test MockLoggerFactory."""
        mock_factory = MockLoggerFactory()

        # Get logger
        logger = mock_factory.get_logger("test")

        # Log messages
        logger.info("test info message")
        logger.error("test error message")

        # Verify messages were captured
        assert "test info message" in logger.info_messages
        assert "test error message" in logger.error_messages

    def test_inject_mock(self):
        """Test inject_mock utility."""
        container = DIContainer()

        # Create custom mock
        custom_mock = Mock()
        custom_mock.test_method.return_value = "mocked"

        # Inject mock
        inject_mock(container, Mock, custom_mock)

        # Resolve mock
        resolved = container.get(Mock)
        assert resolved is custom_mock
        assert resolved.test_method() == "mocked"


class TestFactoriesAndDecorators:
    """Test factory decorators and utilities."""

    def test_service_decorator(self):
        """Test @service decorator."""

        @service(IConfigurationService, ServiceLifetime.SINGLETON)
        class TestConfigService:
            def get_setting(self, key, default=None):
                return f"setting_{key}"

        # Check metadata
        assert hasattr(TestConfigService, "_di_interface")
        assert TestConfigService._di_interface == IConfigurationService
        assert TestConfigService._di_lifetime == ServiceLifetime.SINGLETON

    def test_factory_decorator(self):
        """Test @factory decorator."""

        @factory(IConfigurationService, ServiceLifetime.TRANSIENT)
        def create_config_service():
            mock = Mock(spec=IConfigurationService)
            mock.created_by_factory = True
            return mock

        # Check metadata
        assert hasattr(create_config_service, "_di_interface")
        assert create_config_service._di_interface == IConfigurationService
        assert create_config_service._di_lifetime == ServiceLifetime.TRANSIENT
        assert create_config_service._di_is_factory is True

    def test_injectable_decorator(self):
        """Test @injectable decorator."""
        container = create_test_container()

        with patch("src.di.get_container", return_value=container):

            @injectable
            class InjectableService:
                def __init__(self, config: IConfigurationService):
                    self.config = config
                    self.initialized = True

            # Create instance - dependencies should be injected
            service = InjectableService()

            assert hasattr(service, "config")
            assert isinstance(service.config, MockConfigurationService)
            assert service.initialized is True

    def test_service_registry(self):
        """Test ServiceRegistry functionality."""
        registry = ServiceRegistry()

        # Register service
        class TestService:
            pass

        registry.register_service(TestService, TestService, ServiceLifetime.SINGLETON)

        # Apply to container
        container = DIContainer()
        registry.apply_to_container(container)

        # Should be able to resolve
        service = container.get(TestService)
        assert isinstance(service, TestService)

    def test_service_builder(self):
        """Test ServiceBuilder fluent API."""
        container = DIContainer()
        builder = create_service_builder(container)

        # Configure services
        test_instance = Mock()

        builder.configure(Mock).implemented_by(lambda: test_instance).as_singleton()
        builder.build()

        # Resolve service
        resolved = container.get(Mock)
        assert resolved is test_instance

    def test_lazy_service(self):
        """Test lazy service resolution."""
        container = create_test_container()

        with patch("src.di.get_container", return_value=container):
            lazy_config = lazy(IConfigurationService)

            # Service not resolved yet
            assert not hasattr(lazy_config, "_instance") or lazy_config._instance is None

            # Resolve service
            config = lazy_config.get()

            assert isinstance(config, MockConfigurationService)

            # Second call should return same instance
            config2 = lazy_config()
            assert config is config2


class TestIntegration:
    """Test integration with existing systems."""

    def test_with_dependency_injection_decorator(self):
        """Test @with_dependency_injection decorator."""
        container = create_test_container()

        @with_dependency_injection
        def test_function():
            config = get_injected_service(IConfigurationService)
            return type(config).__name__

        with patch("src.di.get_container", return_value=container):
            result = test_function()
            assert result == "MockConfigurationService"

    def test_get_injected_service_outside_context(self):
        """Test get_injected_service outside DI context."""
        with pytest.raises(RuntimeError):
            get_injected_service(IConfigurationService)

    def test_setup_integration_layer(self):
        """Test integration layer setup."""
        container = DIContainer()

        # Should not raise exception
        setup_integration_layer(container)

        # Core services should be registered
        assert container.is_registered(IConfigurationService)
        assert container.is_registered(ILoggerFactory)
        assert container.is_registered(IHttpClientFactory)


class TestGlobalContainer:
    """Test global container functionality."""

    def test_get_container_singleton(self):
        """Test global container is singleton."""
        container1 = get_container()
        container2 = get_container()

        assert container1 is container2

    def test_reset_container(self):
        """Test container reset functionality."""
        container1 = get_container()
        reset_container()
        container2 = get_container()

        # Should be different instances after reset
        assert container1 is not container2

    def test_create_default_container(self):
        """Test default container creation."""
        container = create_default_container()

        # Should have core services registered
        assert container.is_registered(IConfigurationService)
        assert container.is_registered(ILoggerFactory)
        assert container.is_registered(IHttpClientFactory)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_service_creation_error(self):
        """Test service creation error handling."""
        container = DIContainer()

        # Service that throws in constructor
        class FailingService:
            def __init__(self):
                raise ValueError("Construction failed")

        container.register_transient(FailingService, FailingService)

        with pytest.raises(ServiceCreationError):
            container.get(FailingService)

    def test_invalid_service_registration(self):
        """Test invalid service registration handling."""
        container = DIContainer()

        # This should work without error
        container.register_singleton(str, str)

        # Resolve should work
        result = container.get(str)
        assert isinstance(result, str)

    def test_complex_dependency_graph(self):
        """Test complex dependency graph resolution."""
        container = DIContainer()

        # Create complex dependency graph
        class ServiceC:
            def __init__(self):
                self.name = "C"

        class ServiceB:
            def __init__(self, service_c: ServiceC):
                self.service_c = service_c
                self.name = "B"

        class ServiceA:
            def __init__(self, service_b: ServiceB, service_c: ServiceC):
                self.service_b = service_b
                self.service_c = service_c
                self.name = "A"

        # Register services
        container.register_singleton(ServiceC, ServiceC)
        container.register_transient(ServiceB, ServiceB)
        container.register_transient(ServiceA, ServiceA)

        # Resolve top-level service
        service_a = container.get(ServiceA)

        assert service_a.name == "A"
        assert service_a.service_b.name == "B"
        assert service_a.service_c.name == "C"

        # ServiceC should be the same instance (singleton)
        assert service_a.service_c is service_a.service_b.service_c


class TestPerformance:
    """Test performance characteristics of the DI system."""

    def test_resolution_performance(self):
        """Test service resolution performance."""
        container = DIContainer()

        # Register multiple services
        for i in range(100):
            service_class = type(f"Service{i}", (), {})
            container.register_singleton(service_class, service_class)

        # Resolution should be fast
        import time

        start_time = time.time()

        for i in range(100):
            service_class = type(f"Service{i}", (), {})
            container.get(service_class)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete in reasonable time (< 0.1 seconds)
        assert duration < 0.1

    def test_singleton_thread_safety(self):
        """Test singleton thread safety (basic test)."""
        import threading

        container = DIContainer()
        container.register_singleton(str, str)

        results = []

        def resolve_service():
            service = container.get(str)
            results.append(id(service))

        threads = []
        for _ in range(10):
            thread = threading.Thread(target=resolve_service)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All results should be the same (same instance)
        assert len(set(results)) == 1


@pytest.fixture(autouse=True)
def reset_di_container():
    """Reset DI container after each test."""
    yield
    reset_container()
