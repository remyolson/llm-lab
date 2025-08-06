# LLM Lab Test Suite Documentation

## Overview

The LLM Lab test suite is organized into distinct categories to ensure comprehensive coverage and easy maintenance. Tests are designed to be run independently or as a complete suite.

## Test Structure

```
tests/
├── unit/              # Fast, isolated unit tests
├── integration/       # Tests requiring external services
├── e2e/              # End-to-end workflow tests
├── benchmarks/       # Performance benchmarks
├── fixtures/         # Shared test data and utilities
├── conftest.py       # Pytest configuration and fixtures
└── README.md         # This file
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Runtime**: < 1 second per test
- **Dependencies**: None (uses mocks)
- **Run with**: `pytest tests/unit/`

### Integration Tests (`tests/integration/`)
- **Purpose**: Test integration with external services
- **Runtime**: 1-10 seconds per test
- **Dependencies**: May require API keys
- **Run with**: `pytest tests/integration/`

### End-to-End Tests (`tests/e2e/`)
- **Purpose**: Test complete user workflows
- **Runtime**: 5-30 seconds per test
- **Dependencies**: Full system setup
- **Run with**: `pytest tests/e2e/`

### Benchmark Tests (`tests/benchmarks/`)
- **Purpose**: Measure and track performance
- **Runtime**: Variable (can be long)
- **Dependencies**: Performance tracking tools
- **Run with**: `pytest tests/benchmarks/`

## Running Tests

### Quick Test Run
```bash
# Run only fast unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html
```

### Selective Test Running
```bash
# Run tests by marker
pytest -m unit              # Only unit tests
pytest -m integration        # Only integration tests
pytest -m "not slow"        # Skip slow tests
pytest -m "not expensive"   # Skip tests that use API quota

# Run tests for specific provider
pytest -m openai            # Only OpenAI tests
pytest -m anthropic         # Only Anthropic tests
pytest -m google            # Only Google tests
```

### Full Test Suite
```bash
# Run all tests
pytest

# Run all tests with detailed output
pytest -v --tb=short

# Run all tests including expensive ones
pytest --run-expensive

# Quick mode (skip slow tests)
pytest --quick
```

### Performance Benchmarks
```bash
# Run benchmarks and save results
pytest tests/benchmarks/ --benchmark-save=results.json

# Compare with baseline
pytest tests/benchmarks/ --benchmark-compare=baseline.json
```

## Test Markers

### Category Markers
- `@pytest.mark.unit` - Unit test
- `@pytest.mark.integration` - Integration test
- `@pytest.mark.e2e` - End-to-end test
- `@pytest.mark.benchmark` - Performance benchmark

### Provider Markers
- `@pytest.mark.openai` - Requires OpenAI API
- `@pytest.mark.anthropic` - Requires Anthropic API
- `@pytest.mark.google` - Requires Google API

### Resource Markers
- `@pytest.mark.slow` - Test takes > 5 seconds
- `@pytest.mark.expensive` - Consumes significant API quota
- `@pytest.mark.requires_gpu` - Requires GPU resources
- `@pytest.mark.requires_network` - Requires network access

### Environment Markers
- `@pytest.mark.ci_only` - Only run in CI
- `@pytest.mark.local_only` - Only run locally

## Mock Providers

The test suite includes mock providers for testing without API keys:

```python
def test_with_mock(mock_openai_provider):
    """Test using mock OpenAI provider."""
    response = mock_openai_provider.generate("Test prompt")
    assert isinstance(response, str)
    assert mock_openai_provider.call_count == 1
```

Available mock providers:
- `mock_openai_provider`
- `mock_anthropic_provider`
- `mock_google_provider`
- `mock_all_providers` (dictionary of all mocks)

## Performance Tracking

Use the `performance_tracker` fixture to measure performance:

```python
def test_performance(performance_tracker):
    """Test with performance tracking."""
    with performance_tracker.measure("operation_name"):
        # Code to measure
        result = expensive_operation()

    metrics = performance_tracker.get_summary()
    assert metrics["mean"] < 1.0  # Average time < 1 second
```

Save benchmark results:
```bash
pytest tests/benchmarks/ --benchmark-save=results.json
```

## Test Data

Shared test data is available in `tests/fixtures/test_data.py`:

- `SAMPLE_PROMPTS` - Categorized test prompts
- `EXPECTED_RESPONSES` - Expected responses for evaluation
- `BENCHMARK_DATASETS` - Pre-configured datasets
- `CONFIG_TEMPLATES` - Configuration templates
- `ERROR_SCENARIOS` - Error testing scenarios
- `PERFORMANCE_BASELINES` - Performance regression baselines

## Writing New Tests

### Unit Test Template
```python
import pytest
from unittest.mock import Mock, patch

@pytest.mark.unit
class TestMyComponent:
    """Test suite for MyComponent."""

    def test_basic_functionality(self):
        """Test basic component functionality."""
        # Arrange
        component = MyComponent()

        # Act
        result = component.process("input")

        # Assert
        assert result == expected_value

    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            MyComponent().process(None)
```

### Integration Test Template
```python
import pytest
import os

@pytest.mark.integration
@pytest.mark.requires_network
class TestProviderIntegration:
    """Test provider integration."""

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="API key not available"
    )
    def test_real_api_call(self):
        """Test real API call."""
        # Test with actual API
        pass
```

### Benchmark Test Template
```python
import pytest

@pytest.mark.benchmark
def test_performance(benchmark_results):
    """Test performance metrics."""
    with benchmark_results.measure("operation"):
        # Code to benchmark
        result = perform_operation()

    metrics = benchmark_results.get_summary()
    assert metrics["mean"] < threshold
```

## CI/CD Integration

The test suite is integrated with GitHub Actions:

1. **Unit tests** run on every push
2. **Integration tests** run on PR and main branch
3. **E2E tests** run before releases
4. **Benchmarks** run nightly to detect regressions

## Troubleshooting

### Common Issues

1. **Missing API Keys**
   ```bash
   export OPENAI_API_KEY="your-key"
   export ANTHROPIC_API_KEY="your-key"
   export GOOGLE_API_KEY="your-key"
   ```

2. **Slow Tests**
   ```bash
   # Skip slow tests
   pytest -m "not slow"

   # Or use quick mode
   pytest --quick
   ```

3. **Memory Issues**
   ```bash
   # Run tests in parallel with limited workers
   pytest -n 2

   # Or run categories separately
   pytest tests/unit/
   pytest tests/integration/
   ```

4. **Flaky Tests**
   ```bash
   # Rerun failures
   pytest --reruns 3

   # Or increase timeouts
   pytest --timeout=60
   ```

## Best Practices

1. **Keep unit tests fast** - Use mocks and avoid I/O
2. **Mark tests appropriately** - Use correct markers for categorization
3. **Clean up resources** - Use fixtures for setup/teardown
4. **Test error cases** - Don't just test happy paths
5. **Use meaningful assertions** - Include helpful error messages
6. **Track performance** - Use benchmarks to detect regressions
7. **Document complex tests** - Add docstrings explaining the test
8. **Keep tests independent** - Tests shouldn't depend on each other

## Contributing

When adding new tests:

1. Place tests in the appropriate category directory
2. Use appropriate markers
3. Follow the naming convention: `test_*.py`
4. Include docstrings for test classes and methods
5. Update this README if adding new markers or fixtures
6. Ensure tests pass locally before submitting PR

## Coverage Goals

- **Unit Tests**: > 80% coverage
- **Integration Tests**: All external APIs tested
- **E2E Tests**: All major user workflows covered
- **Benchmarks**: All performance-critical paths measured

Current coverage can be viewed by running:
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```
