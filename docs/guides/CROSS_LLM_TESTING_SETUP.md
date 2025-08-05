# Cross-LLM Testing Examples Setup Guide

This guide provides comprehensive instructions for setting up and running the cross-LLM testing examples developed for Use Case 3 & 4. These examples demonstrate advanced testing patterns including unit testing with pytest fixtures, regression monitoring, performance benchmarking, and automated CI/CD integration.

## Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [API Keys and Secrets Management](#api-keys-and-secrets-management)
- [Local Development Setup](#local-development-setup)
- [GitHub Actions Configuration](#github-actions-configuration)
- [Running Tests](#running-tests)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## Overview

The cross-LLM testing suite includes three main components:

1. **Unit Tests with Cross-Provider Fixtures** (`cross_llm_testing_examples.py`)
   - Parametrized fixtures for testing across OpenAI, Anthropic, and Google providers
   - Mock-based testing for rapid development
   - Integration testing capabilities with real APIs

2. **Regression Testing Suite** (`regression_testing_suite.py`)
   - Performance baseline establishment and monitoring
   - Statistical drift detection
   - Historical trend analysis with SQLite database

3. **Performance Benchmark Suite** (`performance_benchmark_suite.py`)
   - Latency, throughput, and cost analysis
   - Resource utilization monitoring
   - Comparative performance analysis with visualizations

4. **GitHub Actions Workflow** (`.github/workflows/cross-llm-testing.yml`)
   - Automated testing on PR and scheduled runs
   - Multi-provider integration testing
   - Performance monitoring and reporting

## Environment Setup

### System Requirements

- **Python**: 3.9+ (recommended: 3.11)
- **Operating System**: Linux, macOS, or Windows with WSL
- **Memory**: 4GB+ available RAM
- **Storage**: 2GB+ free space for test artifacts and databases

### Core Dependencies

Install the required dependencies:

```bash
# Core requirements
pip install -r requirements.txt
pip install -r requirements-test.txt

# Additional testing dependencies
pip install pytest-html pytest-json-report pytest-cov
pip install scipy matplotlib seaborn pandas psutil numpy
```

### Development Dependencies

For local development and testing:

```bash
pip install pytest-xdist  # Parallel test execution
pip install pytest-benchmark  # Performance benchmarking
pip install pytest-mock  # Enhanced mocking capabilities
pip install jupyter  # For analysis notebooks
```

## API Keys and Secrets Management

### Required API Keys

To run integration tests with real providers, you'll need API keys for:

- **OpenAI**: `OPENAI_API_KEY`
- **Anthropic**: `ANTHROPIC_API_KEY`
- **Google**: `GOOGLE_API_KEY`

### Local Development

Create a `.env` file in the project root:

```bash
# .env file (DO NOT commit to version control)
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
GOOGLE_API_KEY=your-google-api-key-here

# Test configuration
INTEGRATION_TESTS=true
PERFORMANCE_DURATION=2
```

Load environment variables:

```bash
# Using python-dotenv
pip install python-dotenv

# Or manually export
export OPENAI_API_KEY="sk-your-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
export GOOGLE_API_KEY="your-key-here"
```

### GitHub Secrets Configuration

Configure repository secrets for automated testing:

1. Go to your GitHub repository
2. Navigate to Settings → Secrets and variables → Actions
3. Add the following secrets:

```
OPENAI_API_KEY: sk-your-openai-key-here
ANTHROPIC_API_KEY: sk-ant-your-anthropic-key-here
GOOGLE_API_KEY: your-google-api-key-here
```

### Security Best Practices

- **Never commit API keys** to version control
- Use **separate API keys** for testing vs. production
- Set **usage limits** and monitoring on API keys
- **Rotate keys** regularly
- Use **organization-level secrets** for team projects

## Local Development Setup

### 1. Clone and Setup

```bash
git clone <repository-url>
cd llm-lab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt
pip install scipy matplotlib seaborn pandas psutil numpy
```

### 2. Verify Installation

```bash
# Test basic imports
python -c "
import pytest
import numpy as np
import matplotlib.pyplot as plt
from src.providers.openai import OpenAIProvider
print('✅ All imports successful')
"

# Validate test files
python -m py_compile examples/use_cases/cross_llm_testing_examples.py
python -m py_compile examples/use_cases/regression_testing_suite.py
python -m py_compile examples/use_cases/performance_benchmark_suite.py
echo "✅ All test files are valid"
```

### 3. Configure Test Environment

```bash
# Create test directories
mkdir -p test_results regression_data benchmark_results

# Set permissions
chmod 755 test_results regression_data benchmark_results

# Create test configuration
cat > pytest.ini << EOF
[tool:pytest]
testpaths = examples/use_cases tests
python_files = test_*.py *_test.py *_tests.py
python_functions = test_*
python_classes = Test*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
markers =
    integration: marks tests as integration tests (deselect with '-m "not integration"')
    performance: marks tests as performance tests
    slow: marks tests as slow running
EOF
```

## GitHub Actions Configuration

### Workflow Overview

The GitHub Actions workflow (`cross-llm-testing.yml`) provides:

- **Multi-Python version testing** (3.9, 3.10, 3.11, 3.12)
- **Cross-provider integration testing**
- **Scheduled regression monitoring**
- **Performance benchmarking**
- **Comprehensive reporting**

### Workflow Triggers

1. **Push/PR triggers**: Run on main/develop branch changes
2. **Scheduled runs**: Daily at 6 AM UTC for regression monitoring
3. **Manual dispatch**: With customizable parameters

### Manual Workflow Execution

Navigate to Actions → Cross-LLM Testing and run with parameters:

```yaml
test_suite: "all"  # Options: all, unit_tests, regression_tests, performance_tests
provider_filter: "openai,anthropic,google"
enable_real_api_tests: false
performance_duration: "2"
```

### Workflow Configuration

The workflow includes several jobs:

1. **setup-and-lint**: Code quality and environment setup
2. **unit-tests**: Cross-provider unit testing
3. **integration-tests**: Real API integration testing
4. **regression-tests**: Performance regression monitoring
5. **performance-tests**: Benchmark analysis
6. **generate-report**: Comprehensive test reporting

## Running Tests

### Unit Tests Only (Fast)

```bash
# Run all unit tests with mocks
pytest examples/use_cases/cross_llm_testing_examples.py::TestChatbotCrossProvider -v

# Run with coverage
pytest examples/use_cases/cross_llm_testing_examples.py::TestChatbotCrossProvider \
  --cov=examples/use_cases --cov-report=html -v

# Parallel execution
pytest examples/use_cases/cross_llm_testing_examples.py::TestChatbotCrossProvider \
  -n auto -v
```

### Integration Tests (Requires API Keys)

```bash
# Set environment variable
export INTEGRATION_TESTS=true

# Run integration tests
pytest examples/use_cases/cross_llm_testing_examples.py::TestChatbotIntegration \
  -m integration -v

# Run specific provider integration tests
pytest tests/integration/ -k "openai" -v
```

### Regression Tests

```bash
# Run regression test suite
pytest examples/use_cases/regression_testing_suite.py::TestRegressionSuite -v

# Run with real providers (uses API quotas)
INTEGRATION_TESTS=true pytest examples/use_cases/regression_testing_suite.py \
  -m integration -v
```

### Performance Tests

```bash
# Run performance benchmark tests
pytest examples/use_cases/performance_benchmark_suite.py::TestPerformanceBenchmark -v

# Run with real providers for accurate benchmarks
INTEGRATION_TESTS=true pytest examples/use_cases/performance_benchmark_suite.py \
  -m integration -v -s

# Set custom performance test duration
PERFORMANCE_DURATION=5 pytest examples/use_cases/performance_benchmark_suite.py -v
```

### Complete Test Suite

```bash
# Run all tests with comprehensive reporting
pytest examples/use_cases/ \
  --cov=examples/use_cases \
  --cov-report=html \
  --cov-report=xml \
  --html=reports/all-tests.html \
  --json-report --json-report-file=reports/all-tests.json \
  -v

# Run excluding integration tests
pytest examples/use_cases/ \
  -m "not integration" \
  --cov=examples/use_cases \
  --cov-report=html \
  -v
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError` when running tests

**Solution**:
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use pip install in development mode
pip install -e .
```

#### 2. API Key Issues

**Problem**: `Authentication failed` or `Invalid API key`

**Solutions**:
```bash
# Check if API keys are set
echo $OPENAI_API_KEY | head -c 10
echo $ANTHROPIC_API_KEY | head -c 10
echo $GOOGLE_API_KEY | head -c 10

# Test API key validity
python -c "
import os
from src.providers.openai import OpenAIProvider
provider = OpenAIProvider(model='gpt-4o-mini')
provider.initialize()
print('✅ OpenAI API key valid')
"
```

#### 3. Permission Errors

**Problem**: `Permission denied` when creating test files

**Solution**:
```bash
# Fix directory permissions
chmod 755 test_results regression_data benchmark_results

# Fix file permissions
find . -name "*.py" -exec chmod 644 {} \;
```

#### 4. Database Lock Errors

**Problem**: `database is locked` in regression tests

**Solution**:
```bash
# Remove existing database files
rm -f regression_results.db* demo_regression.db*

# Run tests with isolated databases
pytest examples/use_cases/regression_testing_suite.py -v --tb=short
```

#### 5. Memory Issues

**Problem**: `MemoryError` during performance tests

**Solutions**:
```bash
# Reduce performance test duration
export PERFORMANCE_DURATION=1

# Run tests sequentially instead of parallel
pytest examples/use_cases/performance_benchmark_suite.py -v --maxfail=1

# Monitor memory usage
htop  # or Activity Monitor on macOS
```

### Debug Mode

Enable verbose debugging:

```bash
# Enable pytest debugging
pytest examples/use_cases/ -v -s --tb=long --capture=no

# Enable Python debugging
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -u examples/use_cases/cross_llm_testing_examples.py

# Enable provider debugging
export DEBUG_PROVIDERS=true
pytest examples/use_cases/ -v -s
```

### Performance Debugging

Monitor resource usage during tests:

```bash
# Install monitoring tools
pip install psutil memory_profiler

# Run with memory profiling
python -m memory_profiler examples/use_cases/performance_benchmark_suite.py

# Monitor system resources
python -c "
import psutil
import time
while True:
    print(f'CPU: {psutil.cpu_percent():.1f}% | Memory: {psutil.virtual_memory().percent:.1f}%')
    time.sleep(1)
"
```

### GitHub Actions Debugging

#### Check Workflow Status

```bash
# Using GitHub CLI
gh run list --workflow="cross-llm-testing.yml"
gh run view <run-id> --log
```

#### Debug Failed Jobs

1. Check the Actions tab in your GitHub repository
2. Click on the failed workflow run
3. Expand the failed job to see detailed logs
4. Look for specific error messages and stack traces

#### Common GitHub Actions Issues

1. **Secrets not available**: Ensure secrets are configured in repository settings
2. **Timeout errors**: Increase timeout values in workflow file
3. **Rate limiting**: Reduce concurrent API calls or add delays
4. **Artifact upload failures**: Check artifact size limits

## Advanced Configuration

### Custom Test Configuration

Create a custom pytest configuration:

```python
# conftest.py
import pytest
import os

def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment."""
    os.makedirs("test_results", exist_ok=True)
    os.makedirs("regression_data", exist_ok=True)
    os.makedirs("benchmark_results", exist_ok=True)

@pytest.fixture
def api_keys():
    """Provide API keys for testing."""
    return {
        'openai': os.getenv('OPENAI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'google': os.getenv('GOOGLE_API_KEY')
    }
```

### Performance Tuning

Optimize test execution:

```bash
# Parallel execution with custom worker count
pytest examples/use_cases/ -n 4 -v

# Distributed testing across multiple machines
pytest examples/use_cases/ --dist=loadscope -v

# Skip slow tests during development
pytest examples/use_cases/ -m "not slow" -v

# Run only failed tests from last run
pytest examples/use_cases/ --lf -v
```

### Custom Reporting

Generate custom test reports:

```python
# custom_report.py
import json
from datetime import datetime

def generate_custom_report(results_file):
    """Generate custom test report."""
    with open(results_file) as f:
        data = json.load(f)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': data['summary'],
        'failed_tests': [
            test for test in data['tests'] 
            if test['outcome'] == 'failed'
        ],
        'performance_metrics': {
            'total_duration': data['duration'],
            'avg_test_duration': data['duration'] / data['summary']['total']
        }
    }
    
    with open('custom_report.json', 'w') as f:
        json.dump(report, f, indent=2)

# Usage
# pytest examples/use_cases/ --json-report --json-report-file=results.json
# python custom_report.py
```

### Continuous Integration Best Practices

1. **Use matrix builds** for multiple Python versions and OS combinations
2. **Cache dependencies** to speed up workflow execution
3. **Run tests in parallel** where possible
4. **Set appropriate timeouts** for long-running tests
5. **Use conditional execution** for expensive integration tests
6. **Implement proper error handling** and retry logic
7. **Generate comprehensive reports** with artifacts
8. **Monitor resource usage** and optimize accordingly

### Monitoring and Alerting

Set up monitoring for regression detection:

```python
# monitoring.py
import sqlite3
from datetime import datetime, timedelta

def check_performance_regression():
    """Check for performance regressions."""
    conn = sqlite3.connect('regression_results.db')
    
    # Get recent results
    recent_query = """
    SELECT AVG(overall_score) as avg_score
    FROM test_results 
    WHERE timestamp >= datetime('now', '-7 days')
    """
    
    baseline_query = """
    SELECT baseline_value 
    FROM baselines 
    WHERE metric_name = 'overall_score'
    """
    
    recent_score = conn.execute(recent_query).fetchone()[0]
    baseline_score = conn.execute(baseline_query).fetchone()[0]
    
    if recent_score < baseline_score * 0.9:  # 10% degradation
        print(f"⚠️ Performance regression detected!")
        print(f"Recent: {recent_score:.3f}, Baseline: {baseline_score:.3f}")
        return False
    
    print(f"✅ Performance within acceptable range")
    return True

if __name__ == "__main__":
    check_performance_regression()
```

## Support and Contributing

### Getting Help

1. **Check this documentation** for common issues and solutions
2. **Review test logs** for specific error messages
3. **Check GitHub Issues** for known problems
4. **Run tests in debug mode** with `-v -s --tb=long`

### Contributing

When contributing new tests or improvements:

1. Follow the existing code style and patterns
2. Add comprehensive docstrings and comments
3. Include both unit tests and integration tests
4. Update documentation as needed
5. Test across multiple providers when applicable
6. Follow security best practices for API key handling

### Testing Your Changes

Before submitting changes:

```bash
# Run full test suite
pytest examples/use_cases/ -v

# Check code quality
ruff check examples/use_cases/
ruff format --check examples/use_cases/

# Test with real providers (if you have API keys)
INTEGRATION_TESTS=true pytest examples/use_cases/ -m integration -v

# Generate coverage report
pytest examples/use_cases/ --cov=examples/use_cases --cov-report=html
```

This setup guide provides everything needed to successfully implement and run the cross-LLM testing examples. The combination of comprehensive unit testing, regression monitoring, performance benchmarking, and automated CI/CD creates a robust framework for ensuring LLM application quality across multiple providers.