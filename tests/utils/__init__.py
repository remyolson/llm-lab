"""
Test Utilities Package

Comprehensive test utilities for LLM Lab, providing factories, mocks, fixtures,
assertions, and database utilities for testing.

This package is organized into the following modules:
- factories: Data factory classes for generating test data
- mocks: Mock implementations of providers and services
- fixtures: Pytest fixtures for common test scenarios
- assertions: Custom assertion helpers for domain-specific testing
- database: Database utilities for test isolation and cleanup
- plugins: Custom pytest plugins for enhanced testing capabilities
- builders: Builder patterns for complex test object construction
- helpers: General test helper functions and utilities
"""

from .assertions import (
    assert_approximately_equal,
    assert_config_valid,
    assert_error_handled,
    assert_evaluation_result,
    assert_metric_in_range,
    assert_provider_response,
    assert_response_format,
)
from .builders import (
    ConfigBuilder,
    EvaluationBuilder,
    ProviderBuilder,
    ResponseBuilder,
    TestScenarioBuilder,
)
from .database import (
    DatabaseFixture,
    TestDatabase,
    cleanup_test_db,
    create_test_db,
    populate_test_data,
    reset_database,
)
from .factories import (
    BenchmarkDataFactory,
    ConfigFactory,
    EvaluationResultFactory,
    MetricFactory,
    ModelParametersFactory,
    ProviderFactory,
    ResponseFactory,
)
from .fixtures import (
    api_key_fixture,
    benchmark_data_fixture,
    config_fixture,
    database_fixture,
    mock_responses_fixture,
    provider_fixture,
    temp_dir_fixture,
)
from .helpers import (
    capture_logs,
    create_temp_config,
    generate_test_prompt,
    generate_test_response,
    measure_performance,
    mock_api_call,
    retry_on_failure,
    wait_for_condition,
)
from .mocks import (
    MockAnthropicProvider,
    MockCache,
    MockEvaluator,
    MockGoogleProvider,
    MockLocalProvider,
    MockLogger,
    MockOpenAIProvider,
    create_mock_provider,
)
from .plugins import (
    CostTrackingPlugin,
    LLMTestPlugin,
    PerformancePlugin,
    ResponseComparisonPlugin,
)

__all__ = [
    # Factories
    "ProviderFactory",
    "ConfigFactory",
    "ResponseFactory",
    "ModelParametersFactory",
    "EvaluationResultFactory",
    "MetricFactory",
    "BenchmarkDataFactory",
    # Mocks
    "MockOpenAIProvider",
    "MockAnthropicProvider",
    "MockGoogleProvider",
    "MockLocalProvider",
    "MockEvaluator",
    "MockLogger",
    "MockCache",
    "create_mock_provider",
    # Fixtures
    "provider_fixture",
    "config_fixture",
    "database_fixture",
    "temp_dir_fixture",
    "api_key_fixture",
    "mock_responses_fixture",
    "benchmark_data_fixture",
    # Assertions
    "assert_provider_response",
    "assert_evaluation_result",
    "assert_metric_in_range",
    "assert_config_valid",
    "assert_approximately_equal",
    "assert_response_format",
    "assert_error_handled",
    # Database utilities
    "TestDatabase",
    "create_test_db",
    "cleanup_test_db",
    "populate_test_data",
    "reset_database",
    "DatabaseFixture",
    # Plugins
    "LLMTestPlugin",
    "CostTrackingPlugin",
    "PerformancePlugin",
    "ResponseComparisonPlugin",
    # Builders
    "ProviderBuilder",
    "ConfigBuilder",
    "ResponseBuilder",
    "EvaluationBuilder",
    "TestScenarioBuilder",
    # Helpers
    "generate_test_prompt",
    "generate_test_response",
    "create_temp_config",
    "mock_api_call",
    "wait_for_condition",
    "retry_on_failure",
    "capture_logs",
    "measure_performance",
]
