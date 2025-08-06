"""
Integration test runner for LLM providers

This module provides a framework for running integration tests against real APIs
with proper rate limiting, error handling, and result reporting.
"""

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type

from llm_providers.base import LLMProvider
from llm_providers.exceptions import InvalidCredentialsError, ProviderError, RateLimitError

logger = logging.getLogger(__name__)


@dataclass
class IntegrationTestResult:
    """Result from a single integration test."""

    test_name: str
    provider: str
    model: str
    success: bool
    response_time: float
    response: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationTestSuite:
    """Collection of integration test results."""

    suite_name: str
    results: List[IntegrationTestResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    @property
    def duration(self) -> float:
        """Get total test suite duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def success_rate(self) -> float:
        """Get success rate as a percentage."""
        if not self.results:
            return 0.0
        successful = sum(1 for r in self.results if r.success)
        return (successful / len(self.results)) * 100

    def get_results_by_provider(self) -> Dict[str, List[IntegrationTestResult]]:
        """Group results by provider."""
        by_provider = {}
        for result in self.results:
            if result.provider not in by_provider:
                by_provider[result.provider] = []
            by_provider[result.provider].append(result)
        return by_provider


class IntegrationTestRunner:
    """
    Runner for integration tests with real API calls.

    Features:
    - Rate limiting and backoff
    - Parallel test execution
    - Detailed result reporting
    - API key validation
    - Test skipping based on environment
    """

    def __init__(
        self,
        max_workers: int = 3,
        rate_limit_delay: float = 1.0,
        max_retries: int = 2,
        timeout: float = 30.0,
    ):
        """
        Initialize the integration test runner.

        Args:
            max_workers: Maximum concurrent API calls
            rate_limit_delay: Delay between API calls (seconds)
            max_retries: Maximum retries for failed tests
            timeout: Timeout for individual API calls
        """
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.last_api_call = 0

    def is_provider_enabled(self, provider: str) -> bool:
        """Check if integration tests are enabled for a provider."""
        env_vars = {
            "openai": "TEST_OPENAI_INTEGRATION",
            "anthropic": "TEST_ANTHROPIC_INTEGRATION",
            "google": "TEST_GOOGLE_INTEGRATION",
            "all": "TEST_ALL_PROVIDERS_INTEGRATION",
        }

        # Check provider-specific flag
        if provider.lower() in env_vars:
            if os.getenv(env_vars[provider.lower()], "").lower() == "true":
                return True

        # Check global flag
        if os.getenv(env_vars["all"], "").lower() == "true":
            return True

        return False

    def validate_api_keys(self, providers: List[LLMProvider]) -> Dict[str, bool]:
        """Validate API keys for providers."""
        validation_results = {}

        for provider in providers:
            provider_name = provider.__class__.__name__.lower().replace("provider", "")

            try:
                # Try to validate credentials
                is_valid = provider.validate_credentials()
                validation_results[provider_name] = is_valid

                if is_valid:
                    logger.info(f"✓ {provider_name} API key is valid")
                else:
                    logger.warning(f"✗ {provider_name} API key validation failed")

            except InvalidCredentialsError as e:
                logger.error(f"✗ {provider_name} credentials invalid: {e}")
                validation_results[provider_name] = False
            except Exception as e:
                logger.error(f"✗ {provider_name} validation error: {e}")
                validation_results[provider_name] = False

        return validation_results

    def _rate_limit(self):
        """Apply rate limiting between API calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_api_call

        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)

        self.last_api_call = time.time()

    def _run_single_test(
        self, test_func: Callable, provider: LLMProvider, test_name: str, **kwargs
    ) -> IntegrationTestResult:
        """Run a single integration test with retries and error handling."""
        provider_name = provider.__class__.__name__.lower().replace("provider", "")
        model = provider.model_name

        for attempt in range(self.max_retries + 1):
            try:
                # Apply rate limiting
                self._rate_limit()

                # Run the test
                start_time = time.time()
                result = test_func(provider, **kwargs)
                end_time = time.time()

                response_time = end_time - start_time

                return IntegrationTestResult(
                    test_name=test_name,
                    provider=provider_name,
                    model=model,
                    success=True,
                    response_time=response_time,
                    response=str(result) if result else None,
                    metadata={"attempt": attempt + 1},
                )

            except RateLimitError as e:
                if attempt < self.max_retries:
                    retry_delay = getattr(e, "retry_after", 5.0)
                    logger.warning(
                        f"Rate limited, retrying in {retry_delay}s (attempt {attempt + 1})"
                    )
                    time.sleep(retry_delay)
                    continue
                else:
                    return IntegrationTestResult(
                        test_name=test_name,
                        provider=provider_name,
                        model=model,
                        success=False,
                        response_time=0,
                        error=f"Rate limit exceeded after {self.max_retries} retries: {e!s}",
                        metadata={"attempts": attempt + 1},
                    )

            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(f"Test failed, retrying (attempt {attempt + 1}): {e!s}")
                    time.sleep(1.0)  # Brief delay before retry
                    continue
                else:
                    return IntegrationTestResult(
                        test_name=test_name,
                        provider=provider_name,
                        model=model,
                        success=False,
                        response_time=0,
                        error=str(e),
                        metadata={"attempts": attempt + 1},
                    )

        # This should never be reached, but just in case
        return IntegrationTestResult(
            test_name=test_name,
            provider=provider_name,
            model=model,
            success=False,
            response_time=0,
            error="Unknown error - max retries exceeded",
        )

    def run_test_suite(
        self,
        test_functions: Dict[str, Callable],
        providers: List[LLMProvider],
        suite_name: str = "Integration Tests",
    ) -> IntegrationTestSuite:
        """
        Run a suite of integration tests across multiple providers.

        Args:
            test_functions: Dictionary mapping test names to test functions
            providers: List of provider instances to test
            suite_name: Name for this test suite

        Returns:
            IntegrationTestSuite with all results
        """
        suite = IntegrationTestSuite(suite_name=suite_name)

        # Validate API keys first
        logger.info("Validating API keys...")
        validation_results = self.validate_api_keys(providers)

        # Filter providers to only those with valid keys and enabled tests
        valid_providers = []
        for provider in providers:
            provider_name = provider.__class__.__name__.lower().replace("provider", "")

            if not self.is_provider_enabled(provider_name):
                logger.info(f"Skipping {provider_name} - integration tests not enabled")
                continue

            if not validation_results.get(provider_name, False):
                logger.warning(f"Skipping {provider_name} - invalid or missing API key")
                continue

            valid_providers.append(provider)

        if not valid_providers:
            logger.warning("No valid providers found for integration testing")
            suite.end_time = datetime.now()
            return suite

        logger.info(
            f"Running integration tests with providers: {[p.__class__.__name__ for p in valid_providers]}"
        )

        # Create test tasks
        test_tasks = []
        for test_name, test_func in test_functions.items():
            for provider in valid_providers:
                test_tasks.append((test_name, test_func, provider))

        # Run tests with thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_test = {}
            for test_name, test_func, provider in test_tasks:
                future = executor.submit(self._run_single_test, test_func, provider, test_name)
                future_to_test[future] = (test_name, provider)

            # Collect results as they complete
            for future in as_completed(future_to_test):
                test_name, provider = future_to_test[future]
                try:
                    result = future.result(timeout=self.timeout)
                    suite.results.append(result)

                    status = "✓" if result.success else "✗"
                    logger.info(
                        f"{status} {test_name} ({provider.__class__.__name__}): "
                        f"{result.response_time:.2f}s"
                    )

                except Exception as e:
                    # Create error result for completely failed test
                    provider_name = provider.__class__.__name__.lower().replace("provider", "")
                    error_result = IntegrationTestResult(
                        test_name=test_name,
                        provider=provider_name,
                        model=provider.model_name,
                        success=False,
                        response_time=0,
                        error=f"Test execution failed: {e!s}",
                    )
                    suite.results.append(error_result)
                    logger.error(f"✗ {test_name} ({provider.__class__.__name__}) failed: {e}")

        suite.end_time = datetime.now()
        return suite

    def generate_report(
        self, suite: IntegrationTestSuite, output_file: Optional[str] = None
    ) -> str:
        """Generate a detailed test report."""
        report_lines = []
        report_lines.append(f"Integration Test Report: {suite.suite_name}")
        report_lines.append("=" * 60)
        report_lines.append(f"Start Time: {suite.start_time}")
        report_lines.append(f"End Time: {suite.end_time}")
        report_lines.append(f"Duration: {suite.duration:.2f}s")
        report_lines.append(f"Total Tests: {len(suite.results)}")
        report_lines.append(f"Success Rate: {suite.success_rate:.1f}%")
        report_lines.append("")

        # Results by provider
        by_provider = suite.get_results_by_provider()
        for provider, results in by_provider.items():
            successful = sum(1 for r in results if r.success)
            total = len(results)
            avg_time = sum(r.response_time for r in results if r.success) / max(successful, 1)

            report_lines.append(f"{provider.upper()} Provider:")
            report_lines.append(
                f"  Tests: {successful}/{total} successful ({successful / total * 100:.1f}%)"
            )
            report_lines.append(f"  Avg Response Time: {avg_time:.2f}s")

            # Show failures
            failures = [r for r in results if not r.success]
            if failures:
                report_lines.append("  Failures:")
                for failure in failures:
                    report_lines.append(f"    - {failure.test_name}: {failure.error}")
            report_lines.append("")

        # Detailed results
        report_lines.append("Detailed Results:")
        report_lines.append("-" * 40)
        for result in sorted(suite.results, key=lambda x: (x.provider, x.test_name)):
            status = "PASS" if result.success else "FAIL"
            report_lines.append(
                f"{status:4} | {result.provider:10} | {result.test_name:25} | {result.response_time:.2f}s"
            )
            if not result.success and result.error:
                report_lines.append(f"     | Error: {result.error}")

        report = "\n".join(report_lines)

        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {output_file}")

        return report

    def save_results_json(self, suite: IntegrationTestSuite, output_file: str):
        """Save test results as JSON for further analysis."""
        data = {
            "suite_name": suite.suite_name,
            "start_time": suite.start_time.isoformat(),
            "end_time": suite.end_time.isoformat() if suite.end_time else None,
            "duration": suite.duration,
            "success_rate": suite.success_rate,
            "results": [],
        }

        for result in suite.results:
            data["results"].append(
                {
                    "test_name": result.test_name,
                    "provider": result.provider,
                    "model": result.model,
                    "success": result.success,
                    "response_time": result.response_time,
                    "response": result.response,
                    "error": result.error,
                    "timestamp": result.timestamp.isoformat(),
                    "metadata": result.metadata,
                }
            )

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to {output_file}")


# Common integration test functions
def test_basic_generation(provider: LLMProvider, prompt: str = "What is 2 + 2?") -> str:
    """Test basic text generation."""
    response = provider.generate(prompt, max_tokens=50)

    if not response or len(response.strip()) == 0:
        raise ValueError("Empty response from provider")

    return response


def test_parameter_handling(provider: LLMProvider) -> str:
    """Test that provider handles parameters correctly."""
    response = provider.generate("Count to 5", temperature=0.5, max_tokens=25)

    if not response:
        raise ValueError("No response with parameters")

    return response


def test_system_prompt(provider: LLMProvider) -> str:
    """Test system prompt functionality."""
    response = provider.generate(
        "What is your favorite color?",
        system_prompt="You are a helpful assistant who always answers questions about colors with 'blue'.",
        max_tokens=30,
    )

    if not response:
        raise ValueError("No response with system prompt")

    # Check if the system prompt had an effect
    if "blue" not in response.lower():
        logger.warning("System prompt may not have been properly applied")

    return response


def test_longer_generation(provider: LLMProvider) -> str:
    """Test longer text generation."""
    response = provider.generate(
        "Write a short paragraph about the benefits of renewable energy.", max_tokens=150
    )

    if not response or len(response.split()) < 10:
        raise ValueError("Response too short for longer generation test")

    return response


# Standard test suite
STANDARD_INTEGRATION_TESTS = {
    "basic_generation": test_basic_generation,
    "parameter_handling": test_parameter_handling,
    "system_prompt": test_system_prompt,
    "longer_generation": test_longer_generation,
}
