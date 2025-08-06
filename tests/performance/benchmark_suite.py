"""
Core benchmark suite implementation

This module provides the main BenchmarkSuite class and supporting data structures
for running comprehensive performance benchmarks on LLM providers.
"""

import gc
import logging
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Type

import psutil

from llm_providers.base import LLMProvider
from llm_providers.exceptions import ProviderError, RateLimitError

from .benchmark_config import BenchmarkConfig, BenchmarkMode

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Core performance metrics for a benchmark run."""

    response_time: float  # seconds
    tokens_per_second: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model_used: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark test."""

    test_name: str
    provider: str
    model: str
    category: str
    metrics: List[BenchmarkMetrics] = field(default_factory=list)

    @property
    def avg_response_time(self) -> float:
        """Average response time across all successful metrics."""
        successful = [m for m in self.metrics if m.success]
        if not successful:
            return 0.0
        return statistics.mean(m.response_time for m in successful)

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if not self.metrics:
            return 0.0
        successful = sum(1 for m in self.metrics if m.success)
        return (successful / len(self.metrics)) * 100

    @property
    def avg_tokens_per_second(self) -> Optional[float]:
        """Average tokens per second across successful requests."""
        successful = [m for m in self.metrics if m.success and m.tokens_per_second]
        if not successful:
            return None
        return statistics.mean(m.tokens_per_second for m in successful)

    @property
    def total_tokens_processed(self) -> int:
        """Total tokens processed across all requests."""
        return sum(m.total_tokens or 0 for m in self.metrics if m.success)


class MemoryMonitor:
    """Monitor memory usage during benchmark execution."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.measurements = []
        self.running = False
        self.thread = None
        self.process = psutil.Process()

    def start(self):
        """Start memory monitoring."""
        self.running = True
        self.measurements.clear()
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()

    def stop(self) -> List[float]:
        """Stop monitoring and return measurements."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return self.measurements.copy()

    def _monitor(self):
        """Internal monitoring loop."""
        while self.running:
            try:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                self.measurements.append(memory_mb)
                time.sleep(self.interval)
            except Exception as e:
                logger.warning(f"Memory monitoring error: {e}")
                break

    def get_peak_usage(self) -> float:
        """Get peak memory usage in MB."""
        return max(self.measurements) if self.measurements else 0.0

    def get_average_usage(self) -> float:
        """Get average memory usage in MB."""
        return statistics.mean(self.measurements) if self.measurements else 0.0


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for LLM providers.

    Features:
    - Multiple test categories (response time, throughput, etc.)
    - Memory usage monitoring
    - Statistical analysis
    - Concurrent testing
    - Rate limit handling
    """

    def __init__(self, mode: BenchmarkMode = BenchmarkMode.STANDARD):
        """
        Initialize benchmark suite.

        Args:
            mode: Benchmark execution mode
        """
        self.mode = mode
        self.config = BenchmarkConfig.get_mode_config(mode)
        self.providers: List[LLMProvider] = []
        self.results: Dict[str, BenchmarkResult] = {}
        self.memory_monitor = MemoryMonitor()

    def add_provider(self, provider: LLMProvider):
        """Add a provider to benchmark."""
        self.providers.append(provider)
        logger.info(f"Added provider: {provider.__class__.__name__} ({provider.model_name})")

    def run_response_time_benchmark(self) -> Dict[str, BenchmarkResult]:
        """Run response time benchmarks across all providers."""
        logger.info("Running response time benchmarks...")
        results = {}

        categories = ["short", "medium", "long"] if self.mode != BenchmarkMode.QUICK else ["short"]

        for category in categories:
            prompts = BenchmarkConfig.get_prompts(
                category, 3 if self.mode == BenchmarkMode.QUICK else None
            )
            token_limits = BenchmarkConfig.get_token_limits(category)

            for provider in self.providers:
                provider_name = f"{provider.__class__.__name__}_{provider.model_name}"
                test_name = f"response_time_{category}"
                result_key = f"{provider_name}_{test_name}"

                logger.info(f"Testing {provider_name} - {category} prompts")

                result = BenchmarkResult(
                    test_name=test_name,
                    provider=provider.__class__.__name__,
                    model=provider.model_name,
                    category=category,
                )

                # Run tests for each prompt
                for prompt in prompts:
                    for _ in range(self.config["num_requests"] // len(prompts)):
                        metrics = self._run_single_request(
                            provider, prompt, max_tokens=token_limits["max"]
                        )
                        result.metrics.append(metrics)

                        # Rate limiting
                        if metrics.success:
                            delay = BenchmarkConfig.get_rate_limit_delay(
                                provider.__class__.__name__.lower().replace("provider", "")
                            )
                            time.sleep(delay)

                results[result_key] = result
                logger.info(
                    f"Completed {test_name} for {provider_name}: "
                    f"{result.success_rate:.1f}% success, "
                    f"{result.avg_response_time:.2f}s avg response time"
                )

        return results

    def run_throughput_benchmark(self) -> Dict[str, BenchmarkResult]:
        """Run throughput benchmarks (requests per second)."""
        logger.info("Running throughput benchmarks...")
        results = {}

        prompt = "What is artificial intelligence in one sentence?"

        for provider in self.providers:
            provider_name = f"{provider.__class__.__name__}_{provider.model_name}"
            test_name = "throughput"
            result_key = f"{provider_name}_{test_name}"

            logger.info(f"Testing throughput for {provider_name}")

            result = BenchmarkResult(
                test_name=test_name,
                provider=provider.__class__.__name__,
                model=provider.model_name,
                category="throughput",
            )

            # Measure throughput over time
            start_time = time.time()
            requests_completed = 0
            max_concurrent = BenchmarkConfig.get_max_concurrent_requests(
                provider.__class__.__name__.lower().replace("provider", "")
            )

            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                futures = []

                # Submit requests
                for i in range(self.config["num_requests"]):
                    future = executor.submit(
                        self._run_single_request,
                        provider,
                        f"{prompt} (Request {i + 1})",
                        max_tokens=50,
                    )
                    futures.append(future)

                # Collect results
                for future in as_completed(futures):
                    try:
                        metrics = future.result()
                        result.metrics.append(metrics)

                        if metrics.success:
                            requests_completed += 1

                    except Exception as e:
                        error_metrics = BenchmarkMetrics(
                            response_time=0.0, success=False, error=str(e)
                        )
                        result.metrics.append(error_metrics)

            total_time = time.time() - start_time
            throughput = requests_completed / total_time if total_time > 0 else 0

            logger.info(f"Throughput for {provider_name}: {throughput:.2f} requests/second")

            results[result_key] = result

        return results

    def run_token_efficiency_benchmark(self) -> Dict[str, BenchmarkResult]:
        """Run token efficiency benchmarks (tokens per second)."""
        logger.info("Running token efficiency benchmarks...")
        results = {}

        # Use prompts that generate substantial responses
        prompts = [
            "Write a detailed explanation of machine learning algorithms.",
            "Describe the process of software development from planning to deployment.",
            "Explain the principles of sustainable energy and their applications.",
        ]

        for provider in self.providers:
            provider_name = f"{provider.__class__.__name__}_{provider.model_name}"
            test_name = "token_efficiency"
            result_key = f"{provider_name}_{test_name}"

            logger.info(f"Testing token efficiency for {provider_name}")

            result = BenchmarkResult(
                test_name=test_name,
                provider=provider.__class__.__name__,
                model=provider.model_name,
                category="token_efficiency",
            )

            for prompt in prompts:
                metrics = self._run_single_request(provider, prompt, max_tokens=300)
                result.metrics.append(metrics)

                if metrics.success:
                    time.sleep(1.5)  # Longer delay for token-heavy requests

            results[result_key] = result

            if result.avg_tokens_per_second:
                logger.info(
                    f"Token efficiency for {provider_name}: "
                    f"{result.avg_tokens_per_second:.1f} tokens/second"
                )

        return results

    def run_memory_usage_benchmark(self) -> Dict[str, BenchmarkResult]:
        """Run memory usage benchmarks."""
        logger.info("Running memory usage benchmarks...")
        results = {}

        # Force garbage collection before starting
        gc.collect()

        for provider in self.providers:
            provider_name = f"{provider.__class__.__name__}_{provider.model_name}"
            test_name = "memory_usage"
            result_key = f"{provider_name}_{test_name}"

            logger.info(f"Testing memory usage for {provider_name}")

            result = BenchmarkResult(
                test_name=test_name,
                provider=provider.__class__.__name__,
                model=provider.model_name,
                category="memory",
            )

            # Start memory monitoring
            self.memory_monitor.start()

            # Run a series of requests
            prompt = "Generate a comprehensive analysis of renewable energy technologies."

            for i in range(min(10, self.config["num_requests"])):
                metrics = self._run_single_request(provider, prompt, max_tokens=200)

                # Add memory information to metrics
                current_memory = self.memory_monitor.get_average_usage()
                metrics.memory_usage_mb = current_memory

                result.metrics.append(metrics)

                if metrics.success:
                    time.sleep(0.5)

            # Stop monitoring and get peak usage
            memory_measurements = self.memory_monitor.stop()
            peak_memory = max(memory_measurements) if memory_measurements else 0

            logger.info(f"Memory usage for {provider_name}: {peak_memory:.1f} MB peak")

            results[result_key] = result

            # Cleanup
            gc.collect()

        return results

    def run_stress_test(self) -> Dict[str, BenchmarkResult]:
        """Run stress tests with high load."""
        if not BenchmarkConfig.should_run_stress_tests():
            logger.info("Stress tests disabled, skipping...")
            return {}

        logger.info("Running stress tests...")
        results = {}

        stress_config = BenchmarkConfig.get_mode_config(BenchmarkMode.STRESS)

        for provider in self.providers:
            provider_name = f"{provider.__class__.__name__}_{provider.model_name}"
            test_name = "stress_test"
            result_key = f"{provider_name}_{test_name}"

            logger.info(f"Running stress test for {provider_name}")

            result = BenchmarkResult(
                test_name=test_name,
                provider=provider.__class__.__name__,
                model=provider.model_name,
                category="stress",
            )

            # High-load test with many concurrent requests
            prompt = "Quickly respond with a brief explanation of AI."

            start_time = time.time()

            with ThreadPoolExecutor(max_workers=stress_config["concurrent_workers"]) as executor:
                futures = []

                for i in range(stress_config["num_requests"]):
                    future = executor.submit(
                        self._run_single_request, provider, f"{prompt} #{i + 1}", max_tokens=30
                    )
                    futures.append(future)

                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        metrics = future.result(timeout=stress_config["timeout"])
                        result.metrics.append(metrics)
                    except Exception as e:
                        error_metrics = BenchmarkMetrics(
                            response_time=0.0, success=False, error=str(e)
                        )
                        result.metrics.append(error_metrics)

            total_time = time.time() - start_time

            logger.info(
                f"Stress test for {provider_name}: "
                f"{result.success_rate:.1f}% success rate, "
                f"{total_time:.1f}s total time"
            )

            results[result_key] = result

        return results

    def run_all_benchmarks(self) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks and return combined results."""
        logger.info(f"Starting comprehensive benchmark suite in {self.mode.value} mode")

        all_results = {}

        # Warmup phase
        logger.info("Running warmup requests...")
        self._run_warmup()

        # Run different benchmark categories
        benchmark_functions = [
            self.run_response_time_benchmark,
            self.run_throughput_benchmark,
            self.run_token_efficiency_benchmark,
            self.run_memory_usage_benchmark,
        ]

        # Add stress test for comprehensive mode
        if self.mode in [BenchmarkMode.COMPREHENSIVE, BenchmarkMode.STRESS]:
            benchmark_functions.append(self.run_stress_test)

        for benchmark_func in benchmark_functions:
            try:
                results = benchmark_func()
                all_results.update(results)
            except Exception as e:
                logger.error(f"Benchmark function {benchmark_func.__name__} failed: {e}")

        self.results = all_results

        logger.info(f"Benchmark suite completed. Total results: {len(all_results)}")
        return all_results

    def _run_warmup(self):
        """Run warmup requests to prepare providers."""
        warmup_requests = self.config.get("warmup_requests", 1)

        for provider in self.providers:
            for _ in range(warmup_requests):
                try:
                    self._run_single_request(provider, "Hello", max_tokens=10)
                    time.sleep(0.5)
                except Exception as e:
                    logger.warning(f"Warmup request failed for {provider.__class__.__name__}: {e}")

    def _run_single_request(self, provider: LLMProvider, prompt: str, **kwargs) -> BenchmarkMetrics:
        """Run a single request and collect metrics."""
        start_time = time.time()

        try:
            response = provider.generate(prompt, **kwargs)
            end_time = time.time()

            response_time = end_time - start_time

            # Calculate tokens per second if possible
            tokens_per_second = None
            if response and response_time > 0:
                # Rough estimation: ~4 characters per token
                estimated_tokens = len(response) / 4
                tokens_per_second = estimated_tokens / response_time

            return BenchmarkMetrics(
                response_time=response_time,
                tokens_per_second=tokens_per_second,
                success=True,
                model_used=provider.model_name,
                # Note: Token counts would need provider-specific implementation
                completion_tokens=len(response.split()) if response else 0,
                total_tokens=len(prompt.split()) + (len(response.split()) if response else 0),
            )

        except RateLimitError as e:
            end_time = time.time()
            return BenchmarkMetrics(
                response_time=end_time - start_time,
                success=False,
                error=f"Rate limited: {e!s}",
                model_used=provider.model_name,
            )

        except Exception as e:
            end_time = time.time()
            return BenchmarkMetrics(
                response_time=end_time - start_time,
                success=False,
                error=str(e),
                model_used=provider.model_name,
            )

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for all results."""
        if not self.results:
            return {}

        summary = {
            "total_tests": len(self.results),
            "providers_tested": len(set(r.provider for r in self.results.values())),
            "overall_success_rate": 0.0,
            "avg_response_time": 0.0,
            "provider_summary": {},
        }

        # Calculate overall metrics
        all_metrics = []
        for result in self.results.values():
            all_metrics.extend(result.metrics)

        if all_metrics:
            successful_metrics = [m for m in all_metrics if m.success]
            summary["overall_success_rate"] = (len(successful_metrics) / len(all_metrics)) * 100

            if successful_metrics:
                summary["avg_response_time"] = statistics.mean(
                    m.response_time for m in successful_metrics
                )

        # Provider-specific summary
        by_provider = {}
        for result in self.results.values():
            provider = result.provider
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(result)

        for provider, results in by_provider.items():
            all_provider_metrics = []
            for result in results:
                all_provider_metrics.extend(result.metrics)

            successful = [m for m in all_provider_metrics if m.success]

            provider_summary = {
                "total_requests": len(all_provider_metrics),
                "successful_requests": len(successful),
                "success_rate": (len(successful) / len(all_provider_metrics)) * 100
                if all_provider_metrics
                else 0,
                "avg_response_time": statistics.mean(m.response_time for m in successful)
                if successful
                else 0,
                "min_response_time": min(m.response_time for m in successful) if successful else 0,
                "max_response_time": max(m.response_time for m in successful) if successful else 0,
            }

            summary["provider_summary"][provider] = provider_summary

        return summary
