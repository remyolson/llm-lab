# Dependency Injection System Guide

This document provides a comprehensive guide to the LLM Lab dependency injection (DI) system, including usage examples, migration strategies, and best practices.

## Overview

The LLM Lab dependency injection system is designed to:
- **Reduce coupling** between components
- **Improve testability** through mock injection
- **Centralize configuration** management
- **Enable gradual migration** from existing patterns
- **Maintain backward compatibility** with current code

## Key Components

### Core Infrastructure
- **DIContainer**: Main container for service registration and resolution
- **Service Interfaces**: Protocol-based contracts for all services
- **Service Implementations**: Concrete implementations of service interfaces
- **Integration Layer**: Backward compatibility with existing systems

### Service Interfaces

```python
from src.di import (
    IConfigurationService,    # Configuration management
    ILoggerFactory,          # Logger creation and management
    IHttpClientFactory,      # HTTP client creation
    IProviderFactory,        # LLM provider creation
    IEvaluationService,      # Response evaluation
    IFileService,           # File system operations
    ICacheService,          # Caching services
)
```

## Basic Usage

### Getting Services from Container

```python
from src.di import get_container

# Get the global container
container = get_container()

# Resolve services
config_service = container.get(IConfigurationService)
logger_factory = container.get(ILoggerFactory)
provider_factory = container.get(IProviderFactory)

# Use the services
config_value = config_service.get_setting('providers.openai.api_key')
logger = logger_factory.get_logger(__name__)
provider = provider_factory.create_provider('openai', 'gpt-4')
```

### Using the @inject Decorator

```python
from src.di import inject, IConfigurationService, ILogger

@inject
def process_requests(
    requests: list,
    config: IConfigurationService,
    logger: ILogger
):
    """Function with automatic dependency injection."""
    api_key = config.get_environment_variable('OPENAI_API_KEY')
    logger.info(f"Processing {len(requests)} requests")
    # Dependencies are automatically injected when called

# Call the function normally - dependencies are injected automatically
process_requests(['request1', 'request2'])
```

### Constructor Injection

```python
from src.di import injectable, IConfigurationService, ILogger

@injectable
class MyService:
    def __init__(
        self,
        config: IConfigurationService,
        logger: ILogger,
        custom_param: str = "default"
    ):
        self.config = config
        self.logger = logger
        self.custom_param = custom_param

    def do_work(self):
        setting = self.config.get_setting('my_service.setting')
        self.logger.info(f"Working with setting: {setting}")

# Dependencies are automatically injected
service = MyService(custom_param="custom_value")
```

## Service Registration

### Basic Registration

```python
from src.di import get_container, ServiceLifetime

container = get_container()

# Register singleton (shared instance)
container.register_singleton(IMyService, MyServiceImpl)

# Register transient (new instance each time)
container.register_transient(IMyService, MyServiceImpl)

# Register specific instance
container.register_instance(IMyService, my_service_instance)

# Register with factory function
def create_my_service():
    return MyServiceImpl(custom_config="value")

container.register_factory(IMyService, create_my_service, ServiceLifetime.SINGLETON)
```

### Advanced Registration with Decorators

```python
from src.di.factories import service, factory, ServiceLifetime

# Mark a class as a service
@service(IMyService, ServiceLifetime.SINGLETON)
class MyServiceImpl:
    def __init__(self, config: IConfigurationService):
        self.config = config

# Create a factory function
@factory(IComplexService, ServiceLifetime.TRANSIENT)
def create_complex_service(
    config: IConfigurationService,
    logger: ILogger
) -> IComplexService:
    service_config = config.get_setting('complex_service')
    return ComplexServiceImpl(service_config, logger)

# Auto-register services from modules
from src.di.factories import auto_register_services
import my_services_module

auto_register_services(my_services_module)
```

## Testing with DI

### Basic Testing Setup

```python
import pytest
from src.di.testing import (
    create_test_container,
    MockConfigurationService,
    MockLoggerFactory,
    TestContainerContext
)

def test_my_service():
    # Create test container with mocks
    with TestContainerContext() as container:
        # Get mock services for configuration
        mock_config = container.get(IConfigurationService)
        mock_config.set_config('my_service.setting', 'test_value')

        # Test your service
        service = MyService()
        result = service.do_work()

        # Verify behavior
        assert result == expected_result

        # Check mock interactions
        mock_logger_factory = container.get(ILoggerFactory)
        logger = mock_logger_factory.get_logger('my_service')
        assert 'Working with setting: test_value' in logger.info_messages
```

### Advanced Testing with Custom Mocks

```python
from unittest.mock import Mock
from src.di.testing import inject_mock

def test_with_custom_mock():
    with TestContainerContext() as container:
        # Create custom mock
        mock_provider = Mock(spec=ILLMProvider)
        mock_provider.generate.return_value = "mock response"

        # Inject custom mock
        inject_mock(container, ILLMProvider, mock_provider)

        # Test code that uses the provider
        service = MyProviderService()
        result = service.process("test prompt")

        # Verify mock was called
        mock_provider.generate.assert_called_once_with("test prompt")
        assert result == "mock response"
```

## Migration Strategies

### Strategy 1: Gradual Function Migration

**Before (existing code):**
```python
import os
import logging

def process_data(data):
    api_key = os.getenv('OPENAI_API_KEY')
    logger = logging.getLogger(__name__)

    logger.info("Processing data")
    # ... rest of function
```

**After (with DI):**
```python
from src.di import inject, IConfigurationService, ILogger

@inject
def process_data(
    data,
    config: IConfigurationService,
    logger: ILogger
):
    api_key = config.get_environment_variable('OPENAI_API_KEY')
    logger.info("Processing data")
    # ... rest of function
```

### Strategy 2: Class Migration

**Before:**
```python
class DataProcessor:
    def __init__(self):
        self.config = load_config()
        self.logger = logging.getLogger(__name__)

    def process(self, data):
        # ... processing logic
```

**After:**
```python
from src.di import injectable, IConfigurationService, ILogger

@injectable
class DataProcessor:
    def __init__(
        self,
        config: IConfigurationService,
        logger: ILogger
    ):
        self.config = config
        self.logger = logger

    def process(self, data):
        # ... same logic, now testable
```

### Strategy 3: Provider Migration

**Before:**
```python
# Direct provider instantiation
from src.providers.openai import OpenAIProvider

provider = OpenAIProvider('gpt-4', api_key=os.getenv('OPENAI_API_KEY'))
```

**After:**
```python
# Provider factory with DI
from src.di import get_container, IProviderFactory

container = get_container()
provider_factory = container.get(IProviderFactory)
provider = provider_factory.create_provider('openai', 'gpt-4')
```

### Strategy 4: Backward Compatibility

For code that can't be immediately migrated:

```python
from src.di.integration import get_config_value, get_logger, create_provider

# These functions provide DI benefits while maintaining legacy interfaces
config_value = get_config_value('providers.openai.api_key', 'default')
logger = get_logger(__name__)  # Returns standard Python logger
provider = create_provider('openai', 'gpt-4')  # Uses DI internally
```

## Integration with Existing Systems

### Provider System Integration

The DI system integrates seamlessly with the existing provider system:

```python
from src.di import get_container, IProviderFactory

# Get all available providers
container = get_container()
provider_factory = container.get(IProviderFactory)

available_providers = provider_factory.get_available_providers()
# ['openai', 'anthropic', 'google', 'local']

# Create providers with centralized configuration
openai_provider = provider_factory.create_provider('openai', 'gpt-4')
anthropic_provider = provider_factory.create_provider('anthropic', 'claude-3-opus')
```

### Configuration System Integration

The DI system leverages the existing Pydantic Settings configuration:

```python
from src.di import get_container, IConfigurationService

container = get_container()
config = container.get(IConfigurationService)

# Access all existing configuration
model_params = config.get_model_parameters()
network_config = config.get_network_config()
provider_config = config.get_provider_config('openai')

# Environment variables are centralized
api_key = config.get_environment_variable('OPENAI_API_KEY')
```

## Best Practices

### 1. Interface Design

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class IMyService(Protocol):
    def do_work(self, input_data: str) -> str:
        """Clear method signature with types."""
        ...

    def get_status(self) -> dict:
        """Descriptive method names and return types."""
        ...
```

### 2. Service Lifetimes

- **Singleton**: Configuration services, loggers, factories
- **Transient**: Providers, evaluators, request handlers
- **Scoped**: Database connections (future enhancement)

```python
# Configuration services should be singletons
container.register_singleton(IConfigurationService, ConfigurationService)

# Providers should be transient for isolation
container.register_transient(ILLMProvider, OpenAIProvider)
```

### 3. Dependency Management

```python
# Good: Clear, minimal dependencies
class EvaluationService:
    def __init__(
        self,
        config: IConfigurationService,
        logger: ILogger
    ):
        self.config = config
        self.logger = logger

# Avoid: Too many dependencies (consider aggregation)
class OverComplexService:
    def __init__(
        self,
        config: IConfigurationService,
        logger: ILogger,
        http_client: IHttpClient,
        cache: ICacheService,
        provider: ILLMProvider,
        evaluator: IEvaluationService
    ):
        # Consider creating a service aggregator instead
        pass
```

### 4. Testing

```python
# Always test with mocks
def test_my_service():
    with TestContainerContext() as container:
        # Configure mocks
        mock_config = container.get(IConfigurationService)
        mock_config.set_config('key', 'test_value')

        # Test behavior, not implementation
        service = MyService()
        result = service.do_work('input')

        assert result == expected_output
        # Verify important interactions only
```

## Migration Checklist

### Phase 1: Setup (Week 1)
- [ ] Initialize DI container in application startup
- [ ] Set up integration layer for backward compatibility
- [ ] Configure core services (config, logging, HTTP clients)
- [ ] Update development documentation

### Phase 2: Core Services (Weeks 2-3)
- [ ] Migrate configuration access to IConfigurationService
- [ ] Convert logging to ILoggerFactory pattern
- [ ] Update provider creation to use IProviderFactory
- [ ] Migrate evaluation services

### Phase 3: Application Services (Weeks 4-6)
- [ ] Convert major service classes to use constructor injection
- [ ] Update CLI tools and scripts
- [ ] Migrate benchmarking and testing code
- [ ] Convert monitoring and dashboard services

### Phase 4: Testing & Cleanup (Weeks 7-8)
- [ ] Replace all direct os.getenv calls with IConfigurationService
- [ ] Update all tests to use DI mocks
- [ ] Remove legacy service creation patterns
- [ ] Add comprehensive integration tests

## Common Patterns

### Service Factory Pattern

```python
from src.di.factories import factory, ServiceLifetime

@factory(IComplexService, ServiceLifetime.SINGLETON)
def create_complex_service(
    config: IConfigurationService,
    logger_factory: ILoggerFactory,
    http_client_factory: IHttpClientFactory
) -> IComplexService:
    # Complex initialization logic
    service_config = config.get_setting('complex_service')
    logger = logger_factory.get_logger('complex_service')
    http_client = http_client_factory.create_client(
        timeout=service_config.get('timeout', 30)
    )

    return ComplexServiceImpl(service_config, logger, http_client)
```

### Lazy Service Resolution

```python
from src.di.factories import lazy

class MyService:
    def __init__(self):
        # Lazy resolution - service created only when first accessed
        self._config = lazy(IConfigurationService)
        self._logger = lazy(ILoggerFactory)

    def do_work(self):
        config = self._config.get()  # Resolved here
        logger = self._logger.get().get_logger(__name__)
        # ... work with resolved services
```

### Service Composition

```python
@injectable
class CompositeService:
    def __init__(
        self,
        evaluation_service: IEvaluationService,
        provider_factory: IProviderFactory,
        logger: ILogger
    ):
        self.evaluation = evaluation_service
        self.provider_factory = provider_factory
        self.logger = logger

    def evaluate_multiple_providers(self, prompt: str, providers: list):
        results = {}
        for provider_name in providers:
            provider = self.provider_factory.create_provider(provider_name, 'default')
            response = provider.generate(prompt)
            score = self.evaluation.evaluate_response(response, prompt)
            results[provider_name] = score

        return results
```

## Troubleshooting

### Common Issues

**1. Service Not Registered Error**
```python
# Error: ServiceNotRegisteredError: Service IMyService is not registered

# Solution: Register the service
container.register_singleton(IMyService, MyServiceImpl)
```

**2. Circular Dependencies**
```python
# Error: CircularDependencyError: A -> B -> A

# Solution: Use lazy resolution or refactor dependencies
class ServiceA:
    def __init__(self, service_b_factory: Callable[[], IServiceB]):
        self._service_b_factory = service_b_factory

    def use_service_b(self):
        service_b = self._service_b_factory()
        return service_b.do_work()
```

**3. Testing Issues**
```python
# Issue: Tests are not isolated

# Solution: Use TestContainerContext
def test_isolated():
    with TestContainerContext() as container:
        # Each test gets a fresh container
        service = container.get(IMyService)
        # ... test code
```

## Performance Considerations

- **Container Resolution**: O(1) for registered services
- **Singleton Creation**: Thread-safe with double-checked locking
- **Memory Overhead**: Minimal - protocols have no runtime cost
- **Startup Time**: Negligible impact on application startup

## Migration Timeline

| Week | Phase | Activities |
|------|-------|------------|
| 1 | Setup | Initialize DI, configure core services |
| 2-3 | Core Migration | Config, logging, providers |
| 4-6 | Application Services | Major components, CLI tools |
| 7-8 | Testing & Cleanup | Test migration, cleanup |

The dependency injection system is designed to be adopted gradually, providing immediate benefits while maintaining full backward compatibility with existing code.
