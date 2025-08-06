"""
Advanced Factory Methods and Decorators for Dependency Injection

This module provides enhanced factory methods, decorators, and utilities
that make dependency injection more powerful and convenient to use.
"""

import inspect
import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

from .container import DIContainer, ServiceLifetime
from .interfaces import IConfigurationService, ILogger, ILoggerFactory

T = TypeVar("T")
logger = logging.getLogger(__name__)


def service(
    interface: Optional[Type[T]] = None,
    lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
    name: Optional[str] = None,
) -> Callable:
    """
    Decorator to mark a class as a service for automatic registration.

    Args:
        interface: The interface this service implements (if None, uses the class itself)
        lifetime: Service lifetime (singleton, transient, scoped)
        name: Optional service name for disambiguation

    Usage:
        @service(IMyService, ServiceLifetime.SINGLETON)
        class MyService:
            pass
    """

    def decorator(cls: Type[T]) -> Type[T]:
        # Store service metadata on the class
        cls._di_interface = interface or cls
        cls._di_lifetime = lifetime
        cls._di_name = name
        return cls

    return decorator


def factory(interface: Type[T], lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT) -> Callable:
    """
    Decorator to mark a function as a factory for creating service instances.

    Args:
        interface: The interface this factory creates
        lifetime: Service lifetime for instances created by this factory

    Usage:
        @factory(IMyService, ServiceLifetime.SINGLETON)
        def create_my_service(config: IConfigurationService) -> IMyService:
            return MyService(config.get_setting('my_service.config'))
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        func._di_interface = interface
        func._di_lifetime = lifetime
        func._di_is_factory = True
        return func

    return decorator


def injectable(cls: Type[T]) -> Type[T]:
    """
    Decorator to make a class constructor injectable.

    This decorator modifies the class to automatically resolve dependencies
    from the DI container when instantiated.

    Usage:
        @injectable
        class MyClass:
            def __init__(self, config: IConfigurationService, logger: ILogger):
                self.config = config
                self.logger = logger
    """
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        from . import get_container  # Import here to avoid circular imports

        container = get_container()

        # Get constructor signature and type hints
        signature = inspect.signature(original_init)
        type_hints = getattr(original_init, "__annotations__", {})

        # Inject dependencies for parameters not provided
        bound_args = signature.bind_partial(self, *args, **kwargs)

        for param_name, param in signature.parameters.items():
            if param_name in ["self"] or param_name in bound_args.arguments:
                continue

            param_type = type_hints.get(param_name)
            if param_type is None:
                continue

            try:
                dependency = container.get(param_type)
                bound_args.arguments[param_name] = dependency
            except Exception:
                if param.default is inspect.Parameter.empty:
                    logger.warning(f"Cannot inject dependency '{param_name}' for {cls.__name__}")
                continue

        bound_args.apply_defaults()
        original_init(*bound_args.args, **bound_args.kwargs)

    cls.__init__ = new_init
    return cls


class ServiceRegistry:
    """
    Registry for collecting and managing service registrations.

    This allows for automated discovery and registration of services
    decorated with @service or @factory.
    """

    def __init__(self):
        self._services: Dict[Type, Dict[str, Any]] = {}
        self._factories: Dict[Type, Callable] = {}

    def register_service(
        self,
        service_class: Type[T],
        interface: Type[T],
        lifetime: ServiceLifetime,
        name: Optional[str] = None,
    ) -> None:
        """Register a service class."""
        key = (interface, name) if name else interface
        self._services[key] = {
            "class": service_class,
            "interface": interface,
            "lifetime": lifetime,
            "name": name,
        }

    def register_factory(
        self, factory_func: Callable[..., T], interface: Type[T], lifetime: ServiceLifetime
    ) -> None:
        """Register a factory function."""
        self._factories[interface] = {
            "factory": factory_func,
            "interface": interface,
            "lifetime": lifetime,
        }

    def apply_to_container(self, container: DIContainer) -> None:
        """Apply all registrations to a DI container."""
        # Register services
        for key, service_info in self._services.items():
            interface = service_info["interface"]
            service_class = service_info["class"]
            lifetime = service_info["lifetime"]

            if lifetime == ServiceLifetime.SINGLETON:
                container.register_singleton(interface, service_class)
            else:
                container.register_transient(interface, service_class)

        # Register factories
        for interface, factory_info in self._factories.items():
            factory_func = factory_info["factory"]
            lifetime = factory_info["lifetime"]

            container.register_factory(interface, factory_func, lifetime)

    def scan_module(self, module) -> None:
        """
        Scan a module for services and factories.

        Args:
            module: The module to scan for decorated classes and functions
        """
        for name in dir(module):
            obj = getattr(module, name)

            # Check for service classes
            if inspect.isclass(obj) and hasattr(obj, "_di_interface"):
                self.register_service(
                    obj, obj._di_interface, obj._di_lifetime, getattr(obj, "_di_name", None)
                )

            # Check for factory functions
            elif inspect.isfunction(obj) and hasattr(obj, "_di_is_factory"):
                self.register_factory(obj, obj._di_interface, obj._di_lifetime)


# Global service registry
_registry = ServiceRegistry()


def get_service_registry() -> ServiceRegistry:
    """Get the global service registry."""
    return _registry


def auto_register_services(*modules) -> None:
    """
    Automatically register services from the specified modules.

    Args:
        *modules: Modules to scan for service registrations
    """
    registry = get_service_registry()
    for module in modules:
        registry.scan_module(module)


class ServiceBuilder:
    """
    Builder for creating complex service configurations.

    This provides a fluent interface for configuring services with
    complex dependencies and initialization requirements.
    """

    def __init__(self, container: DIContainer):
        self._container = container
        self._configurations: Dict[Type, Dict[str, Any]] = {}

    def configure(self, interface: Type[T]) -> "ServiceConfiguration[T]":
        """
        Start configuring a service.

        Args:
            interface: The service interface to configure

        Returns:
            ServiceConfiguration instance for fluent API
        """
        return ServiceConfiguration(self, interface)

    def build(self) -> None:
        """Apply all configurations to the container."""
        for interface, config in self._configurations.items():
            implementation = config["implementation"]
            lifetime = config.get("lifetime", ServiceLifetime.TRANSIENT)

            if config.get("is_factory", False):
                self._container.register_factory(interface, implementation, lifetime)
            elif lifetime == ServiceLifetime.SINGLETON:
                self._container.register_singleton(interface, implementation)
            else:
                self._container.register_transient(interface, implementation)

    def _add_configuration(self, interface: Type[T], config: Dict[str, Any]) -> None:
        """Add a service configuration."""
        self._configurations[interface] = config


class ServiceConfiguration:
    """
    Configuration builder for a specific service.

    Provides a fluent API for configuring service implementations,
    lifetimes, and initialization parameters.
    """

    def __init__(self, builder: ServiceBuilder, interface: Type[T]):
        self._builder = builder
        self._interface = interface
        self._config: Dict[str, Any] = {}

    def implemented_by(
        self, implementation: Union[Type[T], Callable[[], T]]
    ) -> "ServiceConfiguration[T]":
        """
        Specify the implementation for this service.

        Args:
            implementation: The implementation class or factory function

        Returns:
            Self for chaining
        """
        self._config["implementation"] = implementation
        return self

    def as_singleton(self) -> "ServiceConfiguration[T]":
        """
        Configure this service as a singleton.

        Returns:
            Self for chaining
        """
        self._config["lifetime"] = ServiceLifetime.SINGLETON
        return self

    def as_transient(self) -> "ServiceConfiguration[T]":
        """
        Configure this service as transient.

        Returns:
            Self for chaining
        """
        self._config["lifetime"] = ServiceLifetime.TRANSIENT
        return self

    def using_factory(self, factory: Callable[[], T]) -> "ServiceConfiguration[T]":
        """
        Use a factory function to create instances.

        Args:
            factory: Factory function that creates instances

        Returns:
            Self for chaining
        """
        self._config["implementation"] = factory
        self._config["is_factory"] = True
        return self

    def with_parameters(self, **parameters: Any) -> "ServiceConfiguration[T]":
        """
        Specify initialization parameters.

        Args:
            **parameters: Parameters to pass to the implementation

        Returns:
            Self for chaining
        """
        self._config["parameters"] = parameters
        return self

    def build(self) -> ServiceBuilder:
        """
        Complete configuration and return to builder.

        Returns:
            The ServiceBuilder for further configuration
        """
        self._builder._add_configuration(self._interface, self._config)
        return self._builder


def create_service_builder(container: Optional[DIContainer] = None) -> ServiceBuilder:
    """
    Create a service builder for fluent service configuration.

    Args:
        container: DI container to configure (if None, uses global container)

    Returns:
        ServiceBuilder instance

    Usage:
        builder = create_service_builder()
        builder.configure(IMyService).implemented_by(MyService).as_singleton()
        builder.configure(IOtherService).using_factory(create_other_service)
        builder.build()
    """
    if container is None:
        from . import get_container

        container = get_container()

    return ServiceBuilder(container)


class LazyService:
    """
    Wrapper for lazy service resolution.

    This allows services to be resolved only when first accessed,
    which can help with circular dependencies and performance.
    """

    def __init__(self, interface: Type[T], container: Optional[DIContainer] = None):
        self._interface = interface
        self._container = container
        self._instance: Optional[T] = None
        self._resolved = False

    def get(self) -> T:
        """
        Get the service instance, resolving it if necessary.

        Returns:
            The service instance
        """
        if not self._resolved:
            if self._container is None:
                from . import get_container

                self._container = get_container()

            self._instance = self._container.get(self._interface)
            self._resolved = True

        return self._instance

    def __call__(self) -> T:
        """Allow calling the lazy service like a function."""
        return self.get()


def lazy(interface: Type[T], container: Optional[DIContainer] = None) -> "LazyService[T]":
    """
    Create a lazy service wrapper.

    Args:
        interface: The service interface to resolve lazily
        container: Optional container to use (if None, uses global container)

    Returns:
        LazyService wrapper

    Usage:
        lazy_config = lazy(IConfigurationService)
        # Service is not resolved until first access
        config = lazy_config.get()
    """
    return LazyService(interface, container)
