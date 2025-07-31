"""
Test the integration test runner itself

This module tests the integration test framework to ensure it works correctly.
"""

import pytest
import os
import time
from unittest.mock import Mock, patch, MagicMock

from .integration_runner import (
    IntegrationTestRunner, 
    IntegrationTestResult, 
    IntegrationTestSuite,
    test_basic_generation,
    test_parameter_handling,
    STANDARD_INTEGRATION_TESTS
)
from .test_config import IntegrationTestConfig
from llm_providers import OpenAIProvider
from llm_providers.exceptions import ProviderError, RateLimitError


class TestIntegrationTestRunner:
    """Test the integration test runner functionality."""
    
    def test_runner_initialization(self):
        """Test runner initialization with default and custom settings."""
        # Default initialization
        runner = IntegrationTestRunner()
        assert runner.max_workers == 3
        assert runner.rate_limit_delay == 1.0
        assert runner.max_retries == 2
        assert runner.timeout == 30.0
        
        # Custom initialization
        runner = IntegrationTestRunner(
            max_workers=5,
            rate_limit_delay=0.5,
            max_retries=1,
            timeout=15.0
        )
        assert runner.max_workers == 5
        assert runner.rate_limit_delay == 0.5
        assert runner.max_retries == 1
        assert runner.timeout == 15.0
    
    def test_provider_enabled_check(self):
        """Test provider enabled checking."""
        runner = IntegrationTestRunner()
        
        # Test with environment variables
        with patch.dict('os.environ', {'TEST_OPENAI_INTEGRATION': 'true'}):
            assert runner.is_provider_enabled('openai') == True
        
        with patch.dict('os.environ', {'TEST_ALL_PROVIDERS_INTEGRATION': 'true'}):
            assert runner.is_provider_enabled('openai') == True
            assert runner.is_provider_enabled('anthropic') == True
        
        with patch.dict('os.environ', {}, clear=True):
            assert runner.is_provider_enabled('openai') == False
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        runner = IntegrationTestRunner(rate_limit_delay=0.1)
        
        # First call should not delay
        start_time = time.time()
        runner._rate_limit()
        first_call_time = time.time() - start_time
        assert first_call_time < 0.05  # Should be very fast
        
        # Second call should delay
        start_time = time.time()
        runner._rate_limit()
        second_call_time = time.time() - start_time
        assert second_call_time >= 0.09  # Should wait ~0.1 seconds
    
    def test_api_key_validation(self):
        """Test API key validation."""
        runner = IntegrationTestRunner()
        
        # Mock providers
        mock_provider1 = Mock()
        mock_provider1.__class__.__name__ = "OpenAIProvider"
        mock_provider1.validate_credentials.return_value = True
        
        mock_provider2 = Mock()
        mock_provider2.__class__.__name__ = "AnthropicProvider"
        mock_provider2.validate_credentials.return_value = False
        
        providers = [mock_provider1, mock_provider2]
        results = runner.validate_api_keys(providers)
        
        assert results['openai'] == True
        assert results['anthropic'] == False
    
    def test_single_test_execution_success(self):
        """Test successful single test execution."""
        runner = IntegrationTestRunner(rate_limit_delay=0.01)
        
        # Mock provider
        mock_provider = Mock()
        mock_provider.__class__.__name__ = "OpenAIProvider"
        mock_provider.model_name = "gpt-3.5-turbo"
        
        # Mock test function
        def mock_test_func(provider):
            return "Test response"
        
        result = runner._run_single_test(
            mock_test_func,
            mock_provider,
            "test_name"
        )
        
        assert isinstance(result, IntegrationTestResult)
        assert result.success == True
        assert result.response == "Test response"
        assert result.provider == "openai"
        assert result.model == "gpt-3.5-turbo"
        assert result.response_time > 0
    
    def test_single_test_execution_failure(self):
        """Test failed single test execution."""
        runner = IntegrationTestRunner(rate_limit_delay=0.01, max_retries=1)
        
        # Mock provider
        mock_provider = Mock()
        mock_provider.__class__.__name__ = "AnthropicProvider"
        mock_provider.model_name = "claude-3-haiku"
        
        # Mock test function that raises an error
        def mock_test_func(provider):
            raise ValueError("Test error")
        
        result = runner._run_single_test(
            mock_test_func,
            mock_provider,
            "failing_test"
        )
        
        assert isinstance(result, IntegrationTestResult)
        assert result.success == False
        assert "Test error" in result.error
        assert result.provider == "anthropic"
        assert result.metadata['attempts'] == 2  # Original + 1 retry
    
    def test_single_test_rate_limit_retry(self):
        """Test rate limit retry logic."""
        runner = IntegrationTestRunner(rate_limit_delay=0.01, max_retries=2)
        
        mock_provider = Mock()
        mock_provider.__class__.__name__ = "GoogleProvider"
        mock_provider.model_name = "gemini-1.5-flash"
        
        # Mock test function that raises rate limit error then succeeds
        call_count = 0
        def mock_test_func(provider):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError("Rate limited", retry_after=0.01)
            return "Success after retry"
        
        result = runner._run_single_test(
            mock_test_func,
            mock_provider,
            "rate_limit_test"
        )
        
        assert result.success == True
        assert result.response == "Success after retry"
        assert result.metadata['attempt'] == 2  # Succeeded on second attempt
    
    def test_test_suite_execution(self):
        """Test full test suite execution."""
        runner = IntegrationTestRunner(rate_limit_delay=0.01, max_workers=2)
        
        # Mock providers
        mock_provider1 = Mock()
        mock_provider1.__class__.__name__ = "OpenAIProvider"
        mock_provider1.model_name = "gpt-3.5-turbo"
        mock_provider1.validate_credentials.return_value = True
        
        mock_provider2 = Mock()
        mock_provider2.__class__.__name__ = "AnthropicProvider" 
        mock_provider2.model_name = "claude-3-haiku"
        mock_provider2.validate_credentials.return_value = True
        
        providers = [mock_provider1, mock_provider2]
        
        # Mock test functions
        def test1(provider):
            return f"Response from {provider.__class__.__name__}"
        
        def test2(provider):
            return f"Another response from {provider.__class__.__name__}"
        
        test_functions = {
            "test_one": test1,
            "test_two": test2
        }
        
        # Mock environment to enable all providers
        with patch.dict('os.environ', {'TEST_ALL_PROVIDERS_INTEGRATION': 'true'}):
            suite = runner.run_test_suite(test_functions, providers, "Mock Test Suite")
        
        assert isinstance(suite, IntegrationTestSuite)
        assert suite.suite_name == "Mock Test Suite"
        assert len(suite.results) == 4  # 2 tests × 2 providers
        assert suite.success_rate == 100.0  # All should succeed
        assert suite.duration > 0
        
        # Check results by provider
        by_provider = suite.get_results_by_provider()
        assert "openai" in by_provider
        assert "anthropic" in by_provider
        assert len(by_provider["openai"]) == 2
        assert len(by_provider["anthropic"]) == 2
    
    def test_report_generation(self):
        """Test report generation."""
        runner = IntegrationTestRunner()
        
        # Create mock test suite with results
        suite = IntegrationTestSuite("Test Report Suite")
        
        # Add some mock results
        suite.results = [
            IntegrationTestResult(
                test_name="test1",
                provider="openai",
                model="gpt-3.5-turbo",
                success=True,
                response_time=1.5,
                response="Success"
            ),
            IntegrationTestResult(
                test_name="test2",
                provider="openai",
                model="gpt-3.5-turbo",
                success=False,
                response_time=0.5,
                error="Failed test"
            ),
            IntegrationTestResult(
                test_name="test1",
                provider="anthropic",
                model="claude-3-haiku",
                success=True,
                response_time=2.0,
                response="Success"
            )
        ]
        
        # Generate report
        report = runner.generate_report(suite)
        
        assert "Test Report Suite" in report
        assert "Success Rate: 66.7%" in report
        assert "OPENAI Provider:" in report
        assert "ANTHROPIC Provider:" in report
        assert "Failed test" in report
    
    def test_json_results_export(self, tmp_path):
        """Test JSON results export."""
        runner = IntegrationTestRunner()
        
        # Create mock test suite
        suite = IntegrationTestSuite("JSON Export Test")
        suite.results = [
            IntegrationTestResult(
                test_name="json_test",
                provider="openai",
                model="gpt-3.5-turbo",
                success=True,
                response_time=1.0,
                response="JSON response"
            )
        ]
        suite.end_time = suite.start_time  # Set end time
        
        # Export to JSON
        json_file = tmp_path / "test_results.json"
        runner.save_results_json(suite, str(json_file))
        
        # Verify file was created and contains expected data
        assert json_file.exists()
        
        import json
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        assert data['suite_name'] == "JSON Export Test"
        assert data['success_rate'] == 100.0
        assert len(data['results']) == 1
        assert data['results'][0]['test_name'] == "json_test"
        assert data['results'][0]['provider'] == "openai"
        assert data['results'][0]['success'] == True


class TestStandardTestFunctions:
    """Test the standard integration test functions."""
    
    def test_basic_generation_function(self):
        """Test the basic generation test function."""
        # Mock provider
        mock_provider = Mock()
        mock_provider.generate.return_value = "4"
        
        result = test_basic_generation(mock_provider)
        
        assert result == "4"
        mock_provider.generate.assert_called_once_with("What is 2 + 2?", max_tokens=50)
    
    def test_basic_generation_empty_response(self):
        """Test basic generation with empty response."""
        mock_provider = Mock()
        mock_provider.generate.return_value = ""
        
        with pytest.raises(ValueError, match="Empty response"):
            test_basic_generation(mock_provider)
    
    def test_parameter_handling_function(self):
        """Test the parameter handling test function."""
        mock_provider = Mock()
        mock_provider.generate.return_value = "1, 2, 3, 4, 5"
        
        result = test_parameter_handling(mock_provider)
        
        assert result == "1, 2, 3, 4, 5"
        mock_provider.generate.assert_called_once_with(
            "Count to 5",
            temperature=0.5,
            max_tokens=25
        )
    
    def test_standard_test_suite_completeness(self):
        """Test that the standard test suite contains expected tests."""
        expected_tests = [
            "basic_generation",
            "parameter_handling", 
            "system_prompt",
            "longer_generation"
        ]
        
        for test_name in expected_tests:
            assert test_name in STANDARD_INTEGRATION_TESTS
            assert callable(STANDARD_INTEGRATION_TESTS[test_name])


class TestIntegrationTestResult:
    """Test the IntegrationTestResult class."""
    
    def test_result_creation(self):
        """Test creating test results."""
        result = IntegrationTestResult(
            test_name="test_creation",
            provider="openai",
            model="gpt-3.5-turbo",
            success=True,
            response_time=1.5,
            response="Test response"
        )
        
        assert result.test_name == "test_creation"
        assert result.provider == "openai"
        assert result.model == "gpt-3.5-turbo"
        assert result.success == True
        assert result.response_time == 1.5
        assert result.response == "Test response"
        assert result.error is None
        assert result.timestamp is not None


class TestIntegrationTestSuite:
    """Test the IntegrationTestSuite class."""
    
    def test_suite_creation(self):
        """Test creating test suites."""
        suite = IntegrationTestSuite("Test Suite")
        
        assert suite.suite_name == "Test Suite"
        assert len(suite.results) == 0
        assert suite.start_time is not None
        assert suite.end_time is None
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        suite = IntegrationTestSuite("Success Rate Test")
        
        # Empty suite
        assert suite.success_rate == 0.0
        
        # Add results
        suite.results = [
            IntegrationTestResult("test1", "provider1", "model1", True, 1.0),
            IntegrationTestResult("test2", "provider1", "model1", True, 1.0),
            IntegrationTestResult("test3", "provider1", "model1", False, 1.0)
        ]
        
        assert suite.success_rate == 66.7  # 2/3 * 100, rounded
    
    def test_results_by_provider_grouping(self):
        """Test grouping results by provider."""
        suite = IntegrationTestSuite("Grouping Test")
        
        suite.results = [
            IntegrationTestResult("test1", "openai", "gpt-3.5", True, 1.0),
            IntegrationTestResult("test2", "openai", "gpt-3.5", True, 1.0),
            IntegrationTestResult("test1", "anthropic", "claude", True, 1.0)
        ]
        
        by_provider = suite.get_results_by_provider()
        
        assert "openai" in by_provider
        assert "anthropic" in by_provider
        assert len(by_provider["openai"]) == 2
        assert len(by_provider["anthropic"]) == 1
    
    def test_duration_calculation(self):
        """Test duration calculation."""
        suite = IntegrationTestSuite("Duration Test")
        
        # Duration while running (end_time is None)
        duration1 = suite.duration
        assert duration1 >= 0
        
        # Set end time and check duration
        import datetime
        suite.end_time = suite.start_time + datetime.timedelta(seconds=5)
        assert suite.duration == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])