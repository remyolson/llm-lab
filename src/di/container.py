"""
Core Dependency Injection Container

This module implements the main DI container with service registration,
resolution, and lifetime management capabilities.
"""

import inspect
import logging
from enum import Enum
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union, get_type_hints

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceLifetime(Enum):
    """Service lifetime management options."""

    SINGLETON = "singleton"  # Single instance shared across the application
    TRANSIENT = "transient"  # New instance created each time requested
    SCOPED = "scoped"  # Single instance per scope (future enhancement)


class ServiceRegistration:
    """Represents a service registration in the DI container."""

    def __init__(
        self,
        service_type: Type[T],
        implementation: Union[Type[T], Callable[[], T], T],
        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
    ):
        self.service_type = service_type
        self.implementation = implementation
        self.lifetime = lifetime
        self.instance: Optional[T] = None
        self._lock = Lock()

    def get_instance(self, container: "DIContainer") -> T:
        """Get an instance of the service based on its lifetime."""
        if self.lifetime == ServiceLifetime.SINGLETON:
            return self._get_singleton(container)
        elif self.lifetime == ServiceLifetime.TRANSIENT:
            return self._create_instance(container)
        else:
            raise ValueError(f"Unsupported service lifetime: {self.lifetime}")

    def _get_singleton(self, container: "DIContainer") -> T:
        """Get or create singleton instance with thread safety."""
        if self.instance is None:
            with self._lock:
                if self.instance is None:  # Double-checked locking
                    self.instance = self._create_instance(container)
        return self.instance

    def _create_instance(self, container: "DIContainer") -> T:
        """Create a new instance of the service."""
        # If it's already an instance (not a class), return it
        if not inspect.isclass(self.implementation):
            # Check if it's a factory function - functions have __code__ attribute
            if callable(self.implementation) and hasattr(self.implementation, "__code__"):
                return self.implementation()
            # Otherwise it's an instance, return as-is
            return self.implementation

        # If it's a class, instantiate with dependency injection
        return container._create_instance(self.implementation)


class DIContainer:
    """
    Lightweight dependency injection container.

    Provides service registration, resolution, and automatic constructor injection
    based on type hints. Supports singleton and transient lifetimes.
    """

    def __init__(self):
        self._services: Dict[Type, ServiceRegistration] = {}
        self._lock = Lock()
        self._resolution_stack: list = []  # For circular dependency detection

    def register_singleton(
        self, service_type: Type[T], implementation: Union[Type[T], Callable[[], T], T]
    ) -> "DIContainer":
        """
        Register a service with singleton lifetime.

        Args:
            service_type: The service interface or type to register
            implementation: The implementation class, factory function, or instance

        Returns:
            Self for chaining registrations
        """
        return self._register(service_type, implementation, ServiceLifetime.SINGLETON)

    def register_transient(
        self, service_type: Type[T], implementation: Union[Type[T], Callable[[], T]]
    ) -> "DIContainer":
        """
        Register a service with transient lifetime.

        Args:
            service_type: The service interface or type to register
            implementation: The implementation class or factory function

        Returns:
            Self for chaining registrations
        """
        return self._register(service_type, implementation, ServiceLifetime.TRANSIENT)

    def register_instance(self, service_type: Type[T], instance: T) -> "DIContainer":
        """
        Register a specific instance as a singleton.

        Args:
            service_type: The service interface or type to register
            instance: The pre-created instance to register

        Returns:
            Self for chaining registrations
        """
        return self._register(service_type, instance, ServiceLifetime.SINGLETON)

    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[[], T],
        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
    ) -> "DIContainer":
        """
        Register a factory function for creating service instances.

        Args:
            service_type: The service interface or type to register
            factory: Factory function that creates instances
            lifetime: Service lifetime (singleton or transient)

        Returns:
            Self for chaining registrations
        """
        return self._register(service_type, factory, lifetime)

    def get(self, service_type: Type[T]) -> T:
        """
        Get an instance of the requested service type.

        Args:
            service_type: The service type to resolve

        Returns:
            Instance of the requested service

        Raises:
            ServiceNotRegisteredError: If the service is not registered
            CircularDependencyError: If circular dependencies are detected
        """
        registration = self._services.get(service_type)
        if registration is None:
            raise ServiceNotRegisteredError(f"Service {service_type.__name__} is not registered")

        # Check for circular dependencies
        if service_type in self._resolution_stack:
            cycle = " -> ".join(
                t.__name__
                for t in self._resolution_stack[self._resolution_stack.index(service_type) :]
            )
            raise CircularDependencyError(
                f"Circular dependency detected: {cycle} -> {service_type.__name__}"
            )

        try:
            self._resolution_stack.append(service_type)
            return registration.get_instance(self)
        finally:
            self._resolution_stack.pop()

    def try_get(self, service_type: Type[T]) -> Optional[T]:
        """
        Try to get an instance of the requested service type.

        Args:
            service_type: The service type to resolve

        Returns:
            Instance of the requested service, or None if not registered
        """
        try:
            return self.get(service_type)
        except ServiceNotRegisteredError:
            return None

    def is_registered(self, service_type: Type[T]) -> bool:
        """
        Check if a service type is registered.

        Args:
            service_type: The service type to check

        Returns:
            True if the service is registered, False otherwise
        """
        return service_type in self._services

    def clear(self) -> None:
        """Clear all service registrations (primarily for testing)."""
        with self._lock:
            self._services.clear()

    def _register(
        self,
        service_type: Type[T],
        implementation: Union[Type[T], Callable[[], T], T],
        lifetime: ServiceLifetime,
    ) -> "DIContainer":
        """Internal method to register a service."""
        with self._lock:
            registration = ServiceRegistration(service_type, implementation, lifetime)
            self._services[service_type] = registration
            logger.debug(f"Registered {service_type.__name__} with {lifetime.value} lifetime")
        return self

    def _create_instance(self, cls: Type[T]) -> T:
        """
        Create an instance of a class with dependency injection.

        This method analyzes the constructor signature and automatically
        injects dependencies based on type hints.
        """
        try:
            # Get constructor signature and type hints
            signature = inspect.signature(cls.__init__)
            try:
                type_hints = get_type_hints(cls.__init__)
            except NameError:
                # Handle forward references that can't be resolved
                # Try with the class's module namespace
                try:
                    type_hints = get_type_hints(
                        cls.__init__,
                        globalns=cls.__module__.__dict__ if hasattr(cls, "__module__") else {},
                    )
                except:
                    # Last resort: use raw annotations and resolve manually
                    type_hints = {}
                    raw_annotations = getattr(cls.__init__, "__annotations__", {})
                    for name, annotation in raw_annotations.items():
                        if isinstance(annotation, str):
                            # Try to resolve the string annotation to a registered type
                            for registered_type in self._services.keys():
                                if registered_type.__name__ == annotation:
                                    type_hints[name] = registered_type
                                    break
                            else:
                                # If we can't resolve it, skip this parameter
                                continue
                        else:
                            type_hints[name] = annotation

            # Prepare constructor arguments
            kwargs = {}
            for param_name, param in signature.parameters.items():
                if param_name == "self":
                    continue

                # Get the parameter type from type hints
                param_type = type_hints.get(param_name)
                if param_type is None:
                    # If no type hint, check if it has a default value
                    if param.default is not inspect.Parameter.empty:
                        continue  # Skip parameters with defaults
                    else:
                        logger.warning(
                            f"No type hint for parameter '{param_name}' in {cls.__name__}"
                        )
                        continue

                # Try to resolve the dependency
                try:
                    dependency = self.get(param_type)
                    kwargs[param_name] = dependency
                except CircularDependencyError:
                    # Let circular dependency errors bubble up
                    raise
                except ServiceNotRegisteredError:
                    # If dependency not registered, check for default value
                    if param.default is not inspect.Parameter.empty:
                        continue  # Use default value
                    else:
                        logger.warning(
                            f"Cannot resolve dependency '{param_name}: {param_type.__name__}' for {cls.__name__}"
                        )
                        continue  # Skip unresolvable dependencies

            # Create and return the instance
            instance = cls(**kwargs)
            logger.debug(
                f"Created instance of {cls.__name__} with dependencies: {list(kwargs.keys())}"
            )
            return instance

        except CircularDependencyError:
            # Let circular dependency errors bubble up unchanged
            raise
        except Exception as e:
            logger.error(f"Failed to create instance of {cls.__name__}: {e}")
            raise ServiceCreationError(f"Failed to create instance of {cls.__name__}: {e}")


def inject(func: Callable) -> Callable:
    """
    Decorator for automatic dependency injection in functions.

    This decorator analyzes the function signature and automatically
    injects dependencies based on type hints.

    Usage:
        @inject
        def my_function(config: IConfigurationService, logger: ILogger):
            # Dependencies are automatically injected
            pass
    """
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        from . import get_container  # Import here to avoid circular imports

        container = get_container()

        # Inject dependencies for parameters not provided
        bound_args = signature.bind_partial(*args, **kwargs)

        missing_required_deps = []

        for param_name, param in signature.parameters.items():
            if param_name in bound_args.arguments:
                continue  # Already provided

            param_type = type_hints.get(param_name)
            if param_type is None:
                continue  # No type hint

            try:
                dependency = container.get(param_type)
                bound_args.arguments[param_name] = dependency
            except ServiceNotRegisteredError:
                if param.default is inspect.Parameter.empty:
                    missing_required_deps.append(param_name)
                    logger.warning(
                        f"Cannot inject dependency '{param_name}: {param_type.__name__}' for {func.__name__}"
                    )
                continue

        # If there are missing required dependencies, don't call the function
        if missing_required_deps:
            logger.error(
                f"Cannot call {func.__name__} due to missing required dependencies: {missing_required_deps}"
            )
            return None

        bound_args.apply_defaults()
        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper


# Exception classes for DI container
class DIError(Exception):
    """Base exception for dependency injection errors."""

    pass


class ServiceNotRegisteredError(DIError):
    """Raised when trying to resolve an unregistered service."""

    pass


class CircularDependencyError(DIError):
    """Raised when circular dependencies are detected."""

    pass


class ServiceCreationError(DIError):
    """Raised when service instance creation fails."""

    pass
