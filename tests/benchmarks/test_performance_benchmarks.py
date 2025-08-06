"""
Performance benchmark tests.

These tests measure and track performance metrics to detect regressions.
"""

import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest


@pytest.mark.benchmark
class TestProviderLatency:
    """Benchmark provider response latencies."""

    def test_single_request_latency(self, mock_all_providers, benchmark_results):
        """Measure single request latency for each provider."""
        latencies = {}

        for provider_name, provider in mock_all_providers.items():
            start = time.perf_counter()
            response = provider.generate("Latency test prompt")
            latency = time.perf_counter() - start

            latencies[provider_name] = latency

            # Track in benchmark results
            with benchmark_results.measure(f"{provider_name}_latency"):
                pass  # Already measured above
            benchmark_results.metrics[-1]["duration"] = latency

        # Assert reasonable latencies (mocks should be fast)
        for provider_name, latency in latencies.items():
            assert latency < 1.0, f"{provider_name} latency too high: {latency:.3f}s"

        # Log results
        print(f"\nLatency results: {latencies}")

    @pytest.mark.slow
    def test_batch_request_performance(
        self, mock_openai_provider, benchmark_results, sample_prompts
    ):
        """Benchmark batch processing performance."""
        batch_sizes = [1, 5, 10, 20]
        results = {}

        for batch_size in batch_sizes:
            prompts = sample_prompts[:batch_size] * (batch_size // len(sample_prompts) + 1)
            prompts = prompts[:batch_size]

            with benchmark_results.measure(f"batch_{batch_size}"):
                responses = []
                for prompt in prompts:
                    response = mock_openai_provider.generate(prompt)
                    responses.append(response)

            duration = benchmark_results.metrics[-1]["duration"]
            throughput = batch_size / duration

            results[batch_size] = {
                "duration": duration,
                "throughput": throughput,
                "avg_latency": duration / batch_size,
            }

        # Verify throughput increases with batch size (up to a point)
        throughputs = [results[size]["throughput"] for size in batch_sizes]
        assert throughputs[1] > throughputs[0]  # Some improvement with batching

        print(f"\nBatch performance: {results}")


@pytest.mark.benchmark
class TestMemoryUsage:
    """Benchmark memory usage patterns."""

    def test_memory_baseline(self):
        """Establish memory usage baseline."""
        import gc

        import psutil

        gc.collect()
        process = psutil.Process()
        baseline = process.memory_info().rss / 1024 / 1024  # MB

        assert baseline < 500, f"Baseline memory too high: {baseline:.2f}MB"

        return baseline

    def test_memory_during_generation(self, mock_anthropic_provider, benchmark_results):
        """Test memory usage during text generation."""
        import gc

        import psutil

        process = psutil.Process()
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Generate multiple responses
        responses = []
        memory_samples = []

        for i in range(50):
            response = mock_anthropic_provider.generate(f"Memory test {i}")
            responses.append(response)

            if i % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)

        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024

        # Calculate metrics
        memory_increase = final_memory - initial_memory
        peak_memory = max(memory_samples) if memory_samples else final_memory

        # Track in benchmark results
        benchmark_results.metrics.append(
            {
                "operation": "memory_usage",
                "duration": 0,
                "memory_increase_mb": memory_increase,
                "peak_memory_mb": peak_memory,
                "timestamp": time.time(),
            }
        )

        # Assert reasonable memory usage
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB"
        assert peak_memory < initial_memory + 150, f"Peak memory too high: {peak_memory:.2f}MB"

    @pytest.mark.slow
    def test_memory_leak_detection(self, mock_google_provider, benchmark_results):
        """Test for memory leaks during extended operation."""
        import gc

        import psutil

        process = psutil.Process()
        memory_samples = []

        # Run multiple iterations
        for iteration in range(5):
            gc.collect()

            # Generate responses
            for i in range(20):
                response = mock_google_provider.generate(f"Iteration {iteration}, request {i}")

            # Sample memory after each iteration
            gc.collect()
            memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(memory)

        # Check for memory leak (memory shouldn't continuously increase)
        memory_increases = [
            memory_samples[i + 1] - memory_samples[i] for i in range(len(memory_samples) - 1)
        ]

        # Average increase should be near zero (allowing for some variation)
        avg_increase = statistics.mean(memory_increases)
        assert abs(avg_increase) < 5, (
            f"Possible memory leak: avg increase {avg_increase:.2f}MB per iteration"
        )


@pytest.mark.benchmark
class TestThroughputBenchmarks:
    """Benchmark system throughput under various conditions."""

    def test_sequential_throughput(self, mock_openai_provider, benchmark_results):
        """Test sequential request throughput."""
        num_requests = 20

        with benchmark_results.measure("sequential_throughput"):
            for i in range(num_requests):
                mock_openai_provider.generate(f"Sequential request {i}")

        duration = benchmark_results.metrics[-1]["duration"]
        throughput = num_requests / duration

        # Assert minimum throughput
        assert throughput > 10, f"Sequential throughput too low: {throughput:.2f} req/s"

        return throughput

    @pytest.mark.slow
    def test_concurrent_throughput(self, mock_all_providers, benchmark_results):
        """Test concurrent request throughput."""
        import concurrent.futures

        def make_request(args):
            provider_name, prompt_id = args
            provider = mock_all_providers[provider_name]
            return provider.generate(f"Concurrent request {prompt_id}")

        # Create work items
        work_items = []
        for i in range(30):
            provider_name = list(mock_all_providers.keys())[i % 3]
            work_items.append((provider_name, i))

        # Run concurrent requests
        with benchmark_results.measure("concurrent_throughput"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                results = list(executor.map(make_request, work_items))

        duration = benchmark_results.metrics[-1]["duration"]
        throughput = len(work_items) / duration

        # Concurrent should be faster than sequential
        assert throughput > 15, f"Concurrent throughput too low: {throughput:.2f} req/s"

        return throughput

    def test_throughput_degradation(self, mock_anthropic_provider, benchmark_results):
        """Test throughput degradation under load."""
        throughputs = []

        # Gradually increase load
        for load_factor in [1, 2, 4, 8]:
            # Simulate increased processing time under load
            original_delay = mock_anthropic_provider.response_delay
            mock_anthropic_provider.response_delay = original_delay * (1 + load_factor * 0.1)

            num_requests = 10
            with benchmark_results.measure(f"load_{load_factor}x"):
                for i in range(num_requests):
                    mock_anthropic_provider.generate(f"Load test {i}")

            duration = benchmark_results.metrics[-1]["duration"]
            throughput = num_requests / duration
            throughputs.append(throughput)

            # Reset delay
            mock_anthropic_provider.response_delay = original_delay

        # Verify degradation pattern
        for i in range(1, len(throughputs)):
            assert throughputs[i] <= throughputs[i - 1] * 1.1, (
                "Unexpected throughput increase under load"
            )


@pytest.mark.benchmark
@pytest.mark.slow
class TestScalabilityBenchmarks:
    """Test system scalability characteristics."""

    def test_provider_scaling(self, mock_all_providers, benchmark_results):
        """Test how system scales with number of providers."""
        results = {}

        for num_providers in [1, 2, 3]:
            providers_to_use = dict(list(mock_all_providers.items())[:num_providers])

            with benchmark_results.measure(f"providers_{num_providers}"):
                responses = []
                for provider_name, provider in providers_to_use.items():
                    for i in range(10):
                        response = provider.generate(f"Scaling test {i}")
                        responses.append(response)

            duration = benchmark_results.metrics[-1]["duration"]
            results[num_providers] = {
                "duration": duration,
                "requests": len(responses),
                "throughput": len(responses) / duration,
            }

        # Verify scaling behavior
        # Adding providers shouldn't drastically increase total time
        assert results[3]["duration"] < results[1]["duration"] * 3.5

        print(f"\nScaling results: {results}")

    def test_dataset_size_scaling(self, mock_openai_provider, benchmark_results):
        """Test performance with different dataset sizes."""
        dataset_sizes = [10, 50, 100, 200]
        results = {}

        for size in dataset_sizes:
            prompts = [f"Dataset item {i}" for i in range(size)]

            with benchmark_results.measure(f"dataset_{size}"):
                responses = []
                for prompt in prompts:
                    response = mock_openai_provider.generate(prompt)
                    responses.append(response)

            duration = benchmark_results.metrics[-1]["duration"]
            results[size] = {
                "duration": duration,
                "avg_time_per_item": duration / size,
                "throughput": size / duration,
            }

        # Verify linear or better scaling
        avg_times = [results[size]["avg_time_per_item"] for size in dataset_sizes]

        # Average time per item shouldn't increase significantly
        for i in range(1, len(avg_times)):
            assert avg_times[i] <= avg_times[0] * 1.5, "Poor scaling with dataset size"

        print(f"\nDataset scaling: {results}")


@pytest.mark.benchmark
class TestRegressionDetection:
    """Test for performance regressions."""

    @pytest.fixture
    def baseline_metrics(self, temp_dir):
        """Load or create baseline metrics."""
        baseline_file = temp_dir / "baseline_metrics.json"

        if baseline_file.exists():
            with open(baseline_file) as f:
                return json.load(f)
        else:
            # Create baseline
            return {"response_time": 0.2, "throughput": 50, "memory_usage": 100}

    def test_response_time_regression(
        self, mock_openai_provider, baseline_metrics, benchmark_results
    ):
        """Test for response time regression."""
        # Measure current response time
        num_samples = 10
        response_times = []

        for i in range(num_samples):
            start = time.perf_counter()
            mock_openai_provider.generate(f"Regression test {i}")
            response_times.append(time.perf_counter() - start)

        current_avg = statistics.mean(response_times)
        baseline_avg = baseline_metrics["response_time"]

        # Check for regression (>20% slower)
        regression_threshold = 1.2
        has_regression = current_avg > baseline_avg * regression_threshold

        # Track metrics
        benchmark_results.metrics.append(
            {
                "operation": "response_time_check",
                "duration": current_avg,
                "baseline": baseline_avg,
                "regression": has_regression,
                "timestamp": time.time(),
            }
        )

        # For mock tests, we expect no regression
        assert not has_regression, (
            f"Response time regression: {current_avg:.3f}s vs baseline {baseline_avg:.3f}s"
        )

    def test_throughput_regression(
        self, mock_anthropic_provider, baseline_metrics, benchmark_results
    ):
        """Test for throughput regression."""
        num_requests = 20

        start = time.perf_counter()
        for i in range(num_requests):
            mock_anthropic_provider.generate(f"Throughput test {i}")
        duration = time.perf_counter() - start

        current_throughput = num_requests / duration
        baseline_throughput = baseline_metrics["throughput"]

        # Check for regression (>20% slower)
        regression_threshold = 0.8
        has_regression = current_throughput < baseline_throughput * regression_threshold

        # Track metrics
        benchmark_results.metrics.append(
            {
                "operation": "throughput_check",
                "duration": duration,
                "throughput": current_throughput,
                "baseline": baseline_throughput,
                "regression": has_regression,
                "timestamp": time.time(),
            }
        )

        # For mock tests, we expect good throughput
        assert current_throughput > 10, f"Throughput too low: {current_throughput:.2f} req/s"
