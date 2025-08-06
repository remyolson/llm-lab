#!/usr/bin/env python3
"""
Dependency Injection System Demonstration

This script demonstrates the key features and benefits of the LLM Lab
dependency injection system, showing before/after comparisons and
practical usage examples.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.di import (
    DIContainer,
    IConfigurationService,
    ILogger,
    ILoggerFactory,
    IProviderFactory,
    ServiceLifetime,
    get_container,
    inject,
)
from src.di.factories import factory, injectable, service
from src.di.integration import get_injected_service, with_dependency_injection
from src.di.testing import MockConfigurationService, TestContainerContext


def demo_basic_dependency_injection():
    """Demonstrate basic DI container usage."""
    print("🔧 Basic Dependency Injection Demo")
    print("=" * 50)

    # Get the global container
    container = get_container()

    # Resolve services
    config_service = container.get(IConfigurationService)
    logger_factory = container.get(ILoggerFactory)

    print(f"✅ Configuration Service: {type(config_service).__name__}")
    print(f"✅ Logger Factory: {type(logger_factory).__name__}")

    # Use the services
    logger = logger_factory.get_logger("demo")
    logger.info("DI system working correctly!")

    # Get configuration (will fallback gracefully)
    api_key = config_service.get_environment_variable("DEMO_API_KEY", "not_set")
    print(f"📋 Demo API Key: {api_key}")

    print()


def demo_inject_decorator():
    """Demonstrate the @inject decorator."""
    print("🎯 @inject Decorator Demo")
    print("=" * 50)

    @inject
    def process_data(
        data: str, config: IConfigurationService, logger_factory: ILoggerFactory
    ) -> str:
        """Function with automatic dependency injection."""
        logger = logger_factory.get_logger("processor")

        # Get some configuration
        setting = config.get_setting("processor.mode", "default")

        logger.info(f"Processing data in {setting} mode")
        return f"Processed: {data} (mode: {setting})"

    # Call function - dependencies are automatically injected
    result = process_data("sample data")
    print(f"📊 Result: {result}")
    print()


def demo_injectable_class():
    """Demonstrate the @injectable decorator for classes."""
    print("🏭 @injectable Class Demo")
    print("=" * 50)

    @injectable
    class DataProcessor:
        """Class with constructor dependency injection."""

        def __init__(
            self,
            config: IConfigurationService,
            logger_factory: ILoggerFactory,
            name: str = "DefaultProcessor",
        ):
            self.config = config
            self.logger = logger_factory.get_logger(f"processor.{name.lower()}")
            self.name = name

        def process(self, data: str) -> str:
            """Process data with injected dependencies."""
            batch_size = self.config.get_setting("processor.batch_size", 10)

            self.logger.info(f"{self.name} processing data (batch_size: {batch_size})")
            return f"{self.name}: {data}"

    # Create instance - dependencies are automatically injected
    processor = DataProcessor(name="DemoProcessor")
    result = processor.process("test data")
    print(f"📊 Processed: {result}")
    print()


def demo_service_registration():
    """Demonstrate custom service registration."""
    print("📝 Custom Service Registration Demo")
    print("=" * 50)

    # Define a custom service interface
    from typing import Protocol, runtime_checkable

    @runtime_checkable
    class ICalculatorService(Protocol):
        def add(self, a: int, b: int) -> int: ...
        def multiply(self, a: int, b: int) -> int: ...

    # Implement the service
    class CalculatorService:
        def __init__(self, logger_factory: ILoggerFactory):
            self.logger = logger_factory.get_logger("calculator")

        def add(self, a: int, b: int) -> int:
            result = a + b
            self.logger.info(f"Addition: {a} + {b} = {result}")
            return result

        def multiply(self, a: int, b: int) -> int:
            result = a * b
            self.logger.info(f"Multiplication: {a} * {b} = {result}")
            return result

    # Register the service
    container = get_container()
    container.register_singleton(ICalculatorService, CalculatorService)

    # Use the service
    calculator = container.get(ICalculatorService)

    sum_result = calculator.add(5, 3)
    product_result = calculator.multiply(4, 7)

    print(f"🧮 Calculator results: {sum_result}, {product_result}")
    print()


def demo_testing_with_mocks():
    """Demonstrate testing with dependency injection."""
    print("🧪 Testing with Mocks Demo")
    print("=" * 50)

    @inject
    def send_notification(
        message: str, config: IConfigurationService, logger_factory: ILoggerFactory
    ) -> bool:
        """Function that sends notifications."""
        logger = logger_factory.get_logger("notifications")

        # Get notification settings
        enabled = config.get_setting("notifications.enabled", True)
        service_url = config.get_setting("notifications.service_url", "http://localhost")

        if not enabled:
            logger.info("Notifications disabled")
            return False

        logger.info(f"Sending notification to {service_url}: {message}")
        return True

    # Test with real dependencies
    print("📤 Real dependencies:")
    result = send_notification("Hello from DI system!")
    print(f"   Result: {result}")

    # Test with mocks
    print("🎭 Mock dependencies:")
    with TestContainerContext() as test_container:
        # Configure mock
        mock_config = test_container.get(IConfigurationService)
        mock_config.set_config("notifications.enabled", False)
        mock_config.set_config("notifications.service_url", "http://mock-service")

        # Test with mocked dependencies
        result = send_notification("Test message")
        print(f"   Result with mocks: {result}")

        # Verify mock logger captured messages
        mock_logger_factory = test_container.get(ILoggerFactory)
        logger = mock_logger_factory.get_logger("notifications")
        print(f"   Captured log: {logger.info_messages}")

    print()


def demo_before_after_comparison():
    """Show before/after comparison of DI benefits."""
    print("⚖️  Before/After DI Comparison")
    print("=" * 50)

    print("❌ BEFORE (tightly coupled):")
    print("""
    import os
    import logging

    def process_user_data(user_id):
        # Direct dependencies - hard to test!
        api_key = os.getenv('API_KEY')  # Hard to mock
        logger = logging.getLogger(__name__)  # Global state

        if not api_key:
            logger.error("Missing API key")
            return None

        logger.info(f"Processing user {user_id}")
        return f"processed_{user_id}"
    """)

    print("✅ AFTER (with dependency injection):")
    print("""
    from src.di import inject, IConfigurationService, ILoggerFactory

    @inject
    def process_user_data(
        user_id: str,
        config: IConfigurationService,
        logger_factory: ILoggerFactory
    ):
        # Injected dependencies - easy to test!
        api_key = config.get_environment_variable('API_KEY')
        logger = logger_factory.get_logger(__name__)

        if not api_key:
            logger.error("Missing API key")
            return None

        logger.info(f"Processing user {user_id}")
        return f"processed_{user_id}"
    """)

    print("🎯 Benefits:")
    print("   • Easy to test with mocks")
    print("   • No global state access")
    print("   • Clear dependencies in function signature")
    print("   • Centralized configuration management")
    print("   • Better error handling")
    print()


def demo_integration_with_existing_code():
    """Demonstrate integration with existing code patterns."""
    print("🔗 Integration with Existing Code")
    print("=" * 50)

    # Existing function that can be enhanced with DI
    @with_dependency_injection
    def legacy_function(data: str) -> str:
        """Existing function enhanced with DI."""
        # Can now access DI services without changing signature
        config = get_injected_service(IConfigurationService)
        logger_factory = get_injected_service(ILoggerFactory)

        logger = logger_factory.get_logger("legacy")
        prefix = config.get_setting("legacy.prefix", "LEGACY")

        logger.info(f"Legacy function processing: {data}")
        return f"{prefix}: {data}"

    result = legacy_function("test data")
    print(f"📊 Legacy function result: {result}")
    print()


def demo_performance_characteristics():
    """Demonstrate performance characteristics."""
    print("⚡ Performance Characteristics")
    print("=" * 50)

    import time

    container = get_container()

    # Measure service resolution time
    start_time = time.time()

    for _ in range(1000):
        config = container.get(IConfigurationService)
        logger_factory = container.get(ILoggerFactory)

    end_time = time.time()
    duration = (end_time - start_time) * 1000  # Convert to milliseconds

    print(f"⏱️  1000 service resolutions: {duration:.2f}ms")
    print(f"📊 Average per resolution: {duration / 1000:.4f}ms")
    print("✅ Performance impact: Negligible")
    print()


def main():
    """Run all demonstrations."""
    print("🚀 LLM Lab Dependency Injection System Demo")
    print("=" * 60)
    print()

    try:
        demo_basic_dependency_injection()
        demo_inject_decorator()
        demo_injectable_class()
        demo_service_registration()
        demo_testing_with_mocks()
        demo_before_after_comparison()
        demo_integration_with_existing_code()
        demo_performance_characteristics()

        print("🎉 All demonstrations completed successfully!")
        print()
        print("📚 Next Steps:")
        print("   1. Read docs/development/DEPENDENCY_INJECTION.md")
        print("   2. Run migration analysis: python scripts/migration_tools.py scan src/")
        print("   3. Start migrating high-priority files")
        print("   4. Update tests to use DI mocks")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
