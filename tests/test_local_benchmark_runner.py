"""
Tests for Local Model Benchmark Runner

This test suite verifies the functionality of the local model benchmark runner,
including resource monitoring, performance metrics, and integration with the
unified local provider.
"""

import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

from benchmarks.integrated_runner import (
    is_local_model,
    run_integrated_benchmark,
    run_local_model_benchmark,
)
from benchmarks.local_model_runner import (
    LocalModelBenchmarkRunner,
    ResourceMonitor,
    ResourceSnapshot,
    create_local_model_runner,
)
from evaluation.local_model_metrics import (
    LocalModelBenchmarkResult,
    LocalModelPerformanceMetrics,
)
from src.providers.local import ResourceManager, UnifiedLocalProvider


class TestResourceMonitor(unittest.TestCase):
    """Test resource monitoring functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_resource_manager = Mock(spec=ResourceManager)
        self.monitor = ResourceMonitor(self.mock_resource_manager, sampling_interval=0.1)

    def test_resource_monitor_initialization(self):
        """Test resource monitor initialization."""
        self.assertEqual(self.monitor.resource_manager, self.mock_resource_manager)
        self.assertEqual(self.monitor.sampling_interval, 0.1)
        self.assertFalse(self.monitor.monitoring)
        self.assertEqual(len(self.monitor.snapshots), 0)

    def test_resource_snapshot_collection(self):
        """Test resource snapshot data collection."""
        # Mock system resources
        mock_resources = Mock()
        mock_resources.used_ram_mb = 1024.0
        mock_resources.total_ram_mb = 8192.0
        mock_resources.used_vram_mb = 512.0
        mock_resources.total_vram_mb = 4096.0
        mock_resources.cpu_percent = 25.5

        self.mock_resource_manager.get_system_resources.return_value = mock_resources

        # Start and stop monitoring quickly
        self.monitor.start_monitoring()
        time.sleep(0.2)  # Allow at least one sample
        snapshots = self.monitor.stop_monitoring()

        # Verify snapshots were collected
        self.assertGreater(len(snapshots), 0)
        snapshot = snapshots[0]
        self.assertIsInstance(snapshot, ResourceSnapshot)
        self.assertEqual(snapshot.memory_used_mb, 1024.0)
        self.assertEqual(snapshot.memory_total_mb, 8192.0)

    def test_peak_usage_calculation(self):
        """Test peak usage calculation from snapshots."""
        # Create mock snapshots
        snapshots = [
            ResourceSnapshot(
                timestamp=datetime.now(),
                memory_used_mb=1000.0,
                memory_total_mb=8192.0,
                gpu_memory_used_mb=500.0,
                gpu_memory_total_mb=4096.0,
                gpu_utilization_percent=50.0,
                cpu_utilization_percent=25.0,
            ),
            ResourceSnapshot(
                timestamp=datetime.now(),
                memory_used_mb=1500.0,  # Peak
                memory_total_mb=8192.0,
                gpu_memory_used_mb=750.0,  # Peak
                gpu_memory_total_mb=4096.0,
                gpu_utilization_percent=75.0,  # Peak
                cpu_utilization_percent=45.0,  # Peak
            ),
        ]

        self.monitor.snapshots = snapshots
        peak_usage = self.monitor.get_peak_usage()

        self.assertEqual(peak_usage["peak_memory_mb"], 1500.0)
        self.assertEqual(peak_usage["peak_gpu_memory_mb"], 750.0)
        self.assertEqual(peak_usage["peak_gpu_utilization"], 75.0)
        self.assertEqual(peak_usage["peak_cpu_utilization"], 45.0)
        self.assertEqual(peak_usage["avg_memory_mb"], 1250.0)


class TestLocalModelBenchmarkRunner(unittest.TestCase):
    """Test local model benchmark runner functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = LocalModelBenchmarkRunner(
            enable_resource_monitoring=True, monitoring_interval=0.1
        )

        # Mock provider
        self.mock_provider = Mock(spec=UnifiedLocalProvider)
        self.mock_provider.model_name = "local:test-model"
        self.mock_provider.generate.return_value = "Test response"
        self.mock_provider.get_model_info.return_value = {
            "model_name": "test-model",
            "model_size_mb": 150.0,
        }

    @patch("src.benchmarks.local_model_runner.time.perf_counter")
    def test_single_evaluation(self, mock_time):
        """Test single evaluation execution."""
        # Mock timing
        mock_time.side_effect = [0.0, 0.1, 0.2]  # start, first_token, end

        # Mock resource manager
        mock_resources = Mock()
        mock_resources.used_ram_mb = 1024.0
        mock_resources.total_ram_mb = 8192.0
        mock_resources.used_vram_mb = 512.0
        mock_resources.cpu_percent = 25.0

        with patch.object(
            self.runner.resource_manager, "get_system_resources", return_value=mock_resources
        ):
            with patch.object(
                self.runner.resource_monitor,
                "get_peak_usage",
                return_value={
                    "avg_memory_mb": 1100.0,
                    "peak_memory_mb": 1200.0,
                    "avg_gpu_memory_mb": 600.0,
                    "avg_gpu_utilization": 80.0,
                    "avg_cpu_utilization": 30.0,
                },
            ):
                result = self.runner.run_single_evaluation(
                    provider=self.mock_provider,
                    prompt="Test prompt",
                    expected_answer="Expected answer",
                    dataset_name="test_dataset",
                )

        # Verify result
        self.assertIsInstance(result, LocalModelBenchmarkResult)
        self.assertEqual(result.model_name, "local:test-model")
        self.assertEqual(result.dataset_name, "test_dataset")
        self.assertEqual(result.response, "Test response")
        self.assertEqual(result.prompt, "Test prompt")
        self.assertEqual(result.expected_answer, "Expected answer")

        # Verify performance metrics
        metrics = result.performance_metrics
        self.assertIsInstance(metrics, LocalModelPerformanceMetrics)
        self.assertEqual(metrics.inference_time_ms, 200.0)  # 0.2 * 1000
        self.assertEqual(metrics.memory_used_mb, 1100.0)
        self.assertEqual(metrics.memory_peak_mb, 1200.0)

    def test_batch_evaluation(self):
        """Test batch evaluation execution."""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        expected_answers = ["Answer 1", "Answer 2", "Answer 3"]

        # Mock single evaluation results
        with patch.object(self.runner, "run_single_evaluation") as mock_single:
            mock_single.side_effect = [
                LocalModelBenchmarkResult(
                    model_name="local:test-model",
                    dataset_name="test_dataset",
                    prompt=prompt,
                    response=f"Response to {prompt}",
                    expected_answer=expected,
                    evaluation_score=0.8,
                    evaluation_method="test",
                    performance_metrics=LocalModelPerformanceMetrics(
                        inference_time_ms=100.0,
                        tokens_per_second=10.0,
                        memory_used_mb=1000.0,
                        memory_peak_mb=1100.0,
                    ),
                    backend_used="test_backend",
                    model_format="test_format",
                )
                for prompt, expected in zip(prompts, expected_answers)
            ]

            results = self.runner.run_batch_evaluation(
                provider=self.mock_provider,
                prompts=prompts,
                expected_answers=expected_answers,
                dataset_name="test_dataset",
            )

        # Verify results
        self.assertEqual(len(results), 3)
        self.assertEqual(mock_single.call_count, 3)

        for i, result in enumerate(results):
            self.assertEqual(result.prompt, prompts[i])
            self.assertEqual(result.expected_answer, expected_answers[i])

    def test_progress_callback(self):
        """Test progress callback functionality."""
        prompts = ["Prompt 1", "Prompt 2"]
        expected_answers = ["Answer 1", "Answer 2"]

        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        with patch.object(self.runner, "run_single_evaluation") as mock_single:
            mock_single.return_value = LocalModelBenchmarkResult(
                model_name="test",
                dataset_name="test",
                prompt="test",
                response="test",
                expected_answer="test",
                evaluation_score=0.8,
                evaluation_method="test",
                performance_metrics=LocalModelPerformanceMetrics(
                    inference_time_ms=100.0,
                    tokens_per_second=10.0,
                    memory_used_mb=1000.0,
                    memory_peak_mb=1100.0,
                ),
                backend_used="test",
                model_format="test",
            )

            self.runner.run_batch_evaluation(
                provider=self.mock_provider,
                prompts=prompts,
                expected_answers=expected_answers,
                dataset_name="test_dataset",
                progress_callback=progress_callback,
            )

        # Verify progress calls
        self.assertEqual(progress_calls, [(1, 2), (2, 2)])


class TestIntegratedRunner(unittest.TestCase):
    """Test integrated benchmark runner functionality."""

    def test_local_model_detection(self):
        """Test local model detection logic."""
        # Test explicit local prefixes
        self.assertTrue(is_local_model("local:pythia-70m"))
        self.assertTrue(is_local_model("transformers:gpt2"))
        self.assertTrue(is_local_model("llamacpp:llama-7b"))
        self.assertTrue(is_local_model("ollama:mistral"))

        # Test known local model names
        self.assertTrue(is_local_model("pythia-70m"))
        self.assertTrue(is_local_model("smollm-135m"))
        self.assertTrue(is_local_model("qwen-0.5b"))

        # Test cloud models
        self.assertFalse(is_local_model("gpt-4"))
        self.assertFalse(is_local_model("claude-3-sonnet"))
        self.assertFalse(is_local_model("gemini-1.5-flash"))

    @patch("src.benchmarks.integrated_runner.run_local_model_benchmark")
    @patch("src.benchmarks.integrated_runner.run_cloud_model_benchmark")
    def test_integrated_benchmark_routing(self, mock_cloud, mock_local):
        """Test that integrated benchmark routes to correct runner."""
        prompts = ["Test prompt"]
        expected_answers = ["Test answer"]

        # Test local model routing
        mock_local.return_value = {"benchmark_type": "local_model"}
        result = run_integrated_benchmark(
            model_name="local:pythia-70m",
            dataset_name="test",
            prompts=prompts,
            expected_answers=expected_answers,
        )

        mock_local.assert_called_once()
        mock_cloud.assert_not_called()
        self.assertEqual(result["benchmark_type"], "local_model")

        # Reset mocks
        mock_local.reset_mock()
        mock_cloud.reset_mock()

        # Test cloud model routing
        mock_cloud.return_value = {"benchmark_type": "cloud_model"}
        result = run_integrated_benchmark(
            model_name="gpt-4",
            dataset_name="test",
            prompts=prompts,
            expected_answers=expected_answers,
        )

        mock_cloud.assert_called_once()
        mock_local.assert_not_called()
        self.assertEqual(result["benchmark_type"], "cloud_model")


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions and utilities."""

    def test_create_local_model_runner(self):
        """Test local model runner factory."""
        runner = create_local_model_runner(enable_monitoring=True, monitoring_interval=0.5)

        self.assertIsInstance(runner, LocalModelBenchmarkRunner)
        self.assertTrue(runner.enable_resource_monitoring)
        self.assertEqual(runner.monitoring_interval, 0.5)

    def test_create_local_model_runner_disabled_monitoring(self):
        """Test local model runner factory with disabled monitoring."""
        runner = create_local_model_runner(enable_monitoring=False, monitoring_interval=1.0)

        self.assertIsInstance(runner, LocalModelBenchmarkRunner)
        self.assertFalse(runner.enable_resource_monitoring)
        self.assertEqual(runner.monitoring_interval, 1.0)


if __name__ == "__main__":
    unittest.main()
