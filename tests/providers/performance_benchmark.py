"""
Performance benchmarking for LLM providers

This module provides tools to benchmark and compare performance across providers.
"""

import json
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from llm_providers.base import LLMProvider

from .test_config import TestConfig


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    provider: str
    model: str
    prompt: str
    response_time: float
    success: bool
    error: Optional[str] = None
    response_length: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ProviderBenchmarkSummary:
    """Summary statistics for a provider's benchmark results."""

    provider: str
    model: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    avg_response_length: float
    total_time: float

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def requests_per_second(self) -> float:
        """Calculate average requests per second."""
        if self.total_time == 0:
            return 0.0
        return self.total_requests / self.total_time


class PerformanceBenchmark:
    """
    Performance benchmarking suite for LLM providers.

    This class provides comprehensive performance testing including
    latency measurements, throughput testing, and reliability metrics.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the benchmark suite.

        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = output_dir or TestConfig.get_test_output_dir()
        self.results: List[BenchmarkResult] = []

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def benchmark_provider(
        self, provider: LLMProvider, prompts: List[str], num_runs: int = 1, parallel: bool = False
    ) -> ProviderBenchmarkSummary:
        """
        Benchmark a single provider.

        Args:
            provider: Provider instance to benchmark
            prompts: List of prompts to test
            num_runs: Number of times to run each prompt
            parallel: Whether to run requests in parallel

        Returns:
            Summary of benchmark results
        """
        provider_results = []
        start_time = time.time()

        for run in range(num_runs):
            for prompt in prompts:
                result = self._benchmark_single_request(provider, prompt)
                provider_results.append(result)
                self.results.append(result)

        total_time = time.time() - start_time

        return self._calculate_summary(provider_results, total_time)

    def benchmark_providers(
        self, providers: List[LLMProvider], prompts: Optional[List[str]] = None, num_runs: int = 1
    ) -> Dict[str, ProviderBenchmarkSummary]:
        """
        Benchmark multiple providers.

        Args:
            providers: List of provider instances to benchmark
            prompts: List of prompts (uses default if None)
            num_runs: Number of times to run each prompt

        Returns:
            Dictionary mapping provider names to their summaries
        """
        if prompts is None:
            prompts = TestConfig.TEST_PROMPTS["simple"]

        summaries = {}

        for provider in providers:
            provider_name = f"{provider.__class__.__name__}_{provider.model}"
            print(f"\nBenchmarking {provider_name}...")

            summary = self.benchmark_provider(provider, prompts, num_runs)
            summaries[provider_name] = summary

            # Print immediate results
            self._print_summary(summary)

        return summaries

    def _benchmark_single_request(self, provider: LLMProvider, prompt: str) -> BenchmarkResult:
        """Benchmark a single request."""
        start_time = time.time()

        try:
            response = provider.generate(prompt, max_tokens=50)
            response_time = time.time() - start_time

            return BenchmarkResult(
                provider=provider.__class__.__name__,
                model=provider.model,
                prompt=prompt[:50] + "..." if len(prompt) > 50 else prompt,
                response_time=response_time,
                success=True,
                response_length=len(response),
            )
        except Exception as e:
            response_time = time.time() - start_time

            return BenchmarkResult(
                provider=provider.__class__.__name__,
                model=provider.model,
                prompt=prompt[:50] + "..." if len(prompt) > 50 else prompt,
                response_time=response_time,
                success=False,
                error=str(e),
            )

    def _calculate_summary(
        self, results: List[BenchmarkResult], total_time: float
    ) -> ProviderBenchmarkSummary:
        """Calculate summary statistics from results."""
        if not results:
            raise ValueError("No results to summarize")

        successful_results = [r for r in results if r.success]
        response_times = [r.response_time for r in successful_results]

        if not response_times:
            # All requests failed
            return ProviderBenchmarkSummary(
                provider=results[0].provider,
                model=results[0].model,
                total_requests=len(results),
                successful_requests=0,
                failed_requests=len(results),
                avg_response_time=0,
                min_response_time=0,
                max_response_time=0,
                median_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                avg_response_length=0,
                total_time=total_time,
            )

        response_times.sort()
        response_lengths = [r.response_length for r in successful_results if r.response_length]

        return ProviderBenchmarkSummary(
            provider=results[0].provider,
            model=results[0].model,
            total_requests=len(results),
            successful_requests=len(successful_results),
            failed_requests=len(results) - len(successful_results),
            avg_response_time=statistics.mean(response_times),
            min_response_time=min(response_times),
            max_response_time=max(response_times),
            median_response_time=statistics.median(response_times),
            p95_response_time=self._percentile(response_times, 95),
            p99_response_time=self._percentile(response_times, 99),
            avg_response_length=statistics.mean(response_lengths) if response_lengths else 0,
            total_time=total_time,
        )

    def _percentile(self, sorted_list: List[float], percentile: int) -> float:
        """Calculate percentile from a sorted list."""
        if not sorted_list:
            return 0.0

        index = int(len(sorted_list) * percentile / 100)
        if index >= len(sorted_list):
            index = len(sorted_list) - 1

        return sorted_list[index]

    def _print_summary(self, summary: ProviderBenchmarkSummary):
        """Print a summary of results."""
        print(f"\n{summary.provider} ({summary.model}) Results:")
        print(f"  Total Requests: {summary.total_requests}")
        print(f"  Success Rate: {summary.success_rate:.1f}%")
        print(f"  Avg Response Time: {summary.avg_response_time:.3f}s")
        print(f"  Min Response Time: {summary.min_response_time:.3f}s")
        print(f"  Max Response Time: {summary.max_response_time:.3f}s")
        print(f"  P95 Response Time: {summary.p95_response_time:.3f}s")
        print(f"  P99 Response Time: {summary.p99_response_time:.3f}s")
        print(f"  Requests/Second: {summary.requests_per_second:.2f}")

    def save_results(self, filename: Optional[str] = None):
        """Save benchmark results to a JSON file."""
        if filename is None:
            filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = os.path.join(self.output_dir, filename)

        # Convert results to dictionaries
        results_data = []
        for result in self.results:
            results_data.append(
                {
                    "provider": result.provider,
                    "model": result.model,
                    "prompt": result.prompt,
                    "response_time": result.response_time,
                    "success": result.success,
                    "error": result.error,
                    "response_length": result.response_length,
                    "timestamp": result.timestamp.isoformat(),
                }
            )

        with open(filepath, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"\nResults saved to: {filepath}")

    def compare_providers(self, summaries: Dict[str, ProviderBenchmarkSummary]):
        """Generate a comparison report of multiple providers."""
        print("\n" + "=" * 80)
        print("PROVIDER COMPARISON")
        print("=" * 80)

        # Sort by average response time
        sorted_providers = sorted(summaries.items(), key=lambda x: x[1].avg_response_time)

        # Print header
        print(f"{'Provider':<30} {'Success%':<10} {'Avg RT':<10} {'P95 RT':<10} {'RPS':<10}")
        print("-" * 70)

        # Print each provider
        for name, summary in sorted_providers:
            print(
                f"{name:<30} {summary.success_rate:<10.1f} "
                f"{summary.avg_response_time:<10.3f} "
                f"{summary.p95_response_time:<10.3f} "
                f"{summary.requests_per_second:<10.2f}"
            )

        print("\n" + "=" * 80)

        # Find best/worst
        if sorted_providers:
            print(
                f"\nFastest Provider: {sorted_providers[0][0]} "
                f"({sorted_providers[0][1].avg_response_time:.3f}s avg)"
            )

            if len(sorted_providers) > 1:
                print(
                    f"Slowest Provider: {sorted_providers[-1][0]} "
                    f"({sorted_providers[-1][1].avg_response_time:.3f}s avg)"
                )

            # Find most reliable
            most_reliable = max(summaries.items(), key=lambda x: x[1].success_rate)
            print(
                f"Most Reliable: {most_reliable[0]} ({most_reliable[1].success_rate:.1f}% success)"
            )


def run_performance_comparison(provider_classes: List[Type[LLMProvider]], num_requests: int = 10):
    """
    Run a performance comparison across multiple providers.

    Args:
        provider_classes: List of provider classes to test
        num_requests: Number of requests per provider
    """
    benchmark = PerformanceBenchmark()
    providers = []

    # Initialize providers
    for provider_class in provider_classes:
        if hasattr(provider_class, "SUPPORTED_MODELS"):
            model = TestConfig.get_test_model(
                provider_class.__name__.lower().replace("provider", "")
            )
            try:
                provider = provider_class(model=model)
                provider.initialize()
                providers.append(provider)
            except Exception as e:
                print(f"Failed to initialize {provider_class.__name__}: {e}")

    if not providers:
        print("No providers could be initialized for testing")
        return

    # Run benchmarks
    prompts = TestConfig.TEST_PROMPTS["simple"][:num_requests]
    summaries = benchmark.benchmark_providers(providers, prompts)

    # Compare results
    benchmark.compare_providers(summaries)

    # Save results
    benchmark.save_results()


if __name__ == "__main__":
    # Example usage
    from llm_providers import AnthropicProvider, GoogleProvider, OpenAIProvider

    print("Running performance benchmark comparison...")
    run_performance_comparison([OpenAIProvider, AnthropicProvider, GoogleProvider], num_requests=5)
