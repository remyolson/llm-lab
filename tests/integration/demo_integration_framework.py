#!/usr/bin/env python3
"""
Integration Test Framework Demo

This script demonstrates how the integration test framework works using
mocked providers, so it can run without real API keys.

This shows the complete integration test workflow:
1. Setting up test environment
2. Creating providers
3. Running test suites  
4. Generating reports
5. Saving results

Run with: python demo_integration_framework.py
"""

import os
import sys
import logging
from unittest.mock import Mock, patch
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tests.integration.integration_runner import (
    IntegrationTestRunner,
    IntegrationTestResult,
    IntegrationTestSuite,
    STANDARD_INTEGRATION_TESTS
)
from tests.integration.test_config import IntegrationTestConfig


def create_mock_provider(name: str, model: str, should_fail: bool = False):
    """Create a mock provider for demonstration."""
    mock_provider = Mock()
    mock_provider.__class__.__name__ = f"{name.title()}Provider"
    mock_provider.model_name = model
    mock_provider.validate_credentials.return_value = True
    
    # Mock generate method
    def mock_generate(prompt, **kwargs):
        if should_fail and "error" in prompt.lower():
            raise Exception("Simulated API error")
        
        # Generate realistic responses based on prompt
        if "2 + 2" in prompt:
            return "The answer is 4."
        elif "count to 5" in prompt.lower():
            return "1, 2, 3, 4, 5"
        elif "favorite color" in prompt.lower():
            return "My favorite color is blue."
        elif "renewable energy" in prompt.lower():
            return "Renewable energy sources like solar and wind power offer many benefits including reduced carbon emissions, sustainable power generation, and decreased reliance on fossil fuels."
        else:
            return f"This is a response from {name} to: {prompt[:50]}..."
    
    mock_provider.generate = mock_generate
    return mock_provider


def demo_basic_integration_test():
    """Demonstrate basic integration test functionality."""
    print("🔬 Demo 1: Basic Integration Test")
    print("-" * 50)
    
    # Create a mock provider
    provider = create_mock_provider("OpenAI", "gpt-3.5-turbo")
    
    # Test basic generation
    from tests.integration.integration_runner import test_basic_generation
    
    try:
        result = test_basic_generation(provider, "What is 2 + 2?")
        print(f"✓ Basic generation test passed: {result}")
    except Exception as e:
        print(f"✗ Basic generation test failed: {e}")
    
    print()


def demo_test_runner():
    """Demonstrate the integration test runner."""
    print("🚀 Demo 2: Integration Test Runner")
    print("-" * 50)
    
    # Create mock providers
    providers = [
        create_mock_provider("OpenAI", "gpt-3.5-turbo"),
        create_mock_provider("Anthropic", "claude-3-haiku", should_fail=False),
        create_mock_provider("Google", "gemini-1.5-flash")
    ]
    
    # Create test runner
    runner = IntegrationTestRunner(
        max_workers=2,
        rate_limit_delay=0.1,  # Fast for demo
        max_retries=1,
        timeout=5.0
    )
    
    # Mock environment to enable all providers
    env_patch = patch.dict('os.environ', {
        'TEST_ALL_PROVIDERS_INTEGRATION': 'true'
    })
    
    with env_patch:
        # Run standard test suite
        suite = runner.run_test_suite(
            STANDARD_INTEGRATION_TESTS,
            providers,
            "Demo Integration Test Suite"
        )
    
    # Display results
    print(f"✓ Test suite completed:")
    print(f"  - Total tests: {len(suite.results)}")
    print(f"  - Success rate: {suite.success_rate:.1f}%")
    print(f"  - Duration: {suite.duration:.2f}s")
    
    # Show results by provider
    by_provider = suite.get_results_by_provider()
    for provider_name, results in by_provider.items():
        successful = sum(1 for r in results if r.success)
        print(f"  - {provider_name}: {successful}/{len(results)} tests passed")
    
    print()
    return suite


def demo_custom_tests():
    """Demonstrate custom integration tests."""
    print("🎨 Demo 3: Custom Integration Tests")
    print("-" * 50)
    
    # Define custom test functions
    def test_math_problem(provider):
        """Custom test for math problems."""
        response = provider.generate("If I have 10 apples and eat 3, how many are left?")
        if "7" not in response:
            raise ValueError("Math problem answered incorrectly")
        return response
    
    def test_creativity(provider):
        """Custom test for creative writing."""
        response = provider.generate("Write a haiku about programming.")
        if len(response.split()) < 5:
            raise ValueError("Response too short for creative test")
        return response
    
    def test_error_handling(provider):
        """Custom test that should fail."""
        # This will trigger our mock failure condition
        response = provider.generate("Generate an error please")
        return response
    
    custom_tests = {
        "math_problem": test_math_problem,
        "creativity": test_creativity,
        "error_handling": test_error_handling
    }
    
    # Create providers (one that fails on error)
    providers = [
        create_mock_provider("OpenAI", "gpt-3.5-turbo"),
        create_mock_provider("Anthropic", "claude-3-haiku", should_fail=True)  # Will fail on error
    ]
    
    # Run custom tests
    runner = IntegrationTestRunner(rate_limit_delay=0.05)
    
    with patch.dict('os.environ', {'TEST_ALL_PROVIDERS_INTEGRATION': 'true'}):
        suite = runner.run_test_suite(custom_tests, providers, "Custom Test Demo")
    
    print(f"✓ Custom tests completed:")
    print(f"  - Success rate: {suite.success_rate:.1f}%")
    
    # Show specific failures
    failures = [r for r in suite.results if not r.success]
    if failures:
        print(f"  - Failures ({len(failures)}):")
        for failure in failures:
            print(f"    • {failure.test_name} ({failure.provider}): {failure.error}")
    
    print()
    return suite


def demo_report_generation(suite):
    """Demonstrate report generation."""
    print("📊 Demo 4: Report Generation")
    print("-" * 50)
    
    runner = IntegrationTestRunner()
    
    # Generate text report
    report = runner.generate_report(suite)
    print("Generated integration test report:")
    print(report)
    
    # Save to file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(report)
        print(f"\n✓ Report saved to: {f.name}")
    
    # Generate JSON report
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        runner.save_results_json(suite, f.name)
        print(f"✓ JSON results saved to: {f.name}")
    
    print()


def demo_configuration():
    """Demonstrate configuration options."""
    print("⚙️  Demo 5: Configuration Options")
    print("-" * 50)
    
    # Show current configuration
    print("Integration Test Configuration:")
    
    config = IntegrationTestConfig()
    
    # Provider configs
    print("\nProvider Configurations:")
    for provider, settings in config.PROVIDER_CONFIGS.items():
        print(f"  {provider}:")
        print(f"    - Default model: {settings['default_model']}")
        print(f"    - Rate limit: {settings['rate_limit']} RPM")
        print(f"    - Rate delay: {settings['rate_limit_delay']}s")
    
    # Test prompts
    print(f"\nTest Prompt Categories: {list(config.TEST_PROMPTS.keys())}")
    print(f"Simple prompts: {len(config.TEST_PROMPTS['simple'])}")
    
    # Environment checks
    print("\nEnvironment Status:")
    providers = ['openai', 'anthropic', 'google']
    for provider in providers:
        enabled = config.is_provider_enabled(provider)
        has_key = config.has_valid_api_key(provider)
        model = config.get_test_model(provider)
        print(f"  {provider}: enabled={enabled}, has_key={has_key}, model={model}")
    
    print()


def main():
    """Run all integration test framework demos."""
    print("🎭 Integration Test Framework Demo")
    print("=" * 60)
    print("This demo shows how the integration test framework works")
    print("using mocked providers (no real API keys required).\n")
    
    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise for demo
        format='%(levelname)s: %(message)s'
    )
    
    # Run demos
    demo_basic_integration_test()
    suite = demo_test_runner()
    demo_custom_tests()
    demo_report_generation(suite)
    demo_configuration()
    
    print("🎉 Demo completed!")
    print("\nTo run real integration tests:")
    print("1. Set up API keys: export OPENAI_API_KEY=your_key")
    print("2. Enable tests: export TEST_OPENAI_INTEGRATION=true")
    print("3. Run: python tests/integration/run_integration_tests.py")
    print("\nOr run the comprehensive test runner:")
    print("TEST_ALL_PROVIDERS_INTEGRATION=true python tests/integration/run_integration_tests.py")


if __name__ == "__main__":
    main()