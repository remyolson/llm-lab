"""
Local Model Benchmark Runner Extension

This module extends the benchmark runner specifically for local models with:
- Resource monitoring during inference
- Performance metrics collection
- Local model-specific result handling
- Integration with the UnifiedLocalProvider
"""

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..evaluation.local_model_metrics import (
    LocalModelBenchmarkResult,
    LocalModelPerformanceMetrics,
    evaluate_local_model_response,
)
from ..providers.local import ResourceManager, UnifiedLocalProvider

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time."""

    timestamp: datetime
    memory_used_mb: float
    memory_total_mb: float
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    gpu_utilization_percent: float
    cpu_utilization_percent: float


class ResourceMonitor:
    """Monitor system resources during benchmark execution."""

    def __init__(self, resource_manager: ResourceManager, sampling_interval: float = 0.5):
        """
        Initialize resource monitor.

        Args:
            resource_manager: Resource manager instance
            sampling_interval: How often to sample resources (seconds)
        """
        self.resource_manager = resource_manager
        self.sampling_interval = sampling_interval
        self.snapshots: List[ResourceSnapshot] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start_monitoring(self):
        """Start resource monitoring in a background thread."""
        if self.monitoring:
            return

        self.monitoring = True
        self._stop_event.clear()
        self.snapshots = []

        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self) -> List[ResourceSnapshot]:
        """Stop resource monitoring and return collected snapshots."""
        if not self.monitoring:
            return self.snapshots

        self.monitoring = False
        self._stop_event.set()

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)

        return self.snapshots.copy()

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while not self._stop_event.wait(self.sampling_interval):
            try:
                resources = self.resource_manager.get_system_resources()

                snapshot = ResourceSnapshot(
                    timestamp=datetime.now(),
                    memory_used_mb=resources.used_ram_mb,
                    memory_total_mb=resources.total_ram_mb,
                    gpu_memory_used_mb=resources.used_vram_mb,
                    gpu_memory_total_mb=resources.total_vram_mb,
                    gpu_utilization_percent=resources.gpu_utilization_percent
                    if hasattr(resources, "gpu_utilization_percent")
                    else 0.0,
                    cpu_utilization_percent=resources.cpu_percent,
                )

                self.snapshots.append(snapshot)

            except Exception as e:
                logger.warning(f"Error collecting resource snapshot: {e}")

    def get_peak_usage(self) -> Dict[str | float]:
        """Get peak resource usage from collected snapshots."""
        if not self.snapshots:
            return {}

        return {
            "peak_memory_mb": max(s.memory_used_mb for s in self.snapshots),
            "peak_gpu_memory_mb": max(s.gpu_memory_used_mb for s in self.snapshots),
            "peak_gpu_utilization": max(s.gpu_utilization_percent for s in self.snapshots),
            "peak_cpu_utilization": max(s.cpu_utilization_percent for s in self.snapshots),
            "avg_memory_mb": sum(s.memory_used_mb for s in self.snapshots) / len(self.snapshots),
            "avg_gpu_memory_mb": sum(s.gpu_memory_used_mb for s in self.snapshots)
            / len(self.snapshots),
            "avg_gpu_utilization": sum(s.gpu_utilization_percent for s in self.snapshots)
            / len(self.snapshots),
            "avg_cpu_utilization": sum(s.cpu_utilization_percent for s in self.snapshots)
            / len(self.snapshots),
        }


class LocalModelBenchmarkRunner:
    """Specialized benchmark runner for local models."""

    def __init__(self, enable_resource_monitoring: bool = True, monitoring_interval: float = 0.5):
        """
        Initialize local model benchmark runner.

        Args:
            enable_resource_monitoring: Whether to monitor resources during inference
            monitoring_interval: Resource monitoring sampling interval in seconds
        """
        self.enable_resource_monitoring = enable_resource_monitoring
        self.monitoring_interval = monitoring_interval
        self.resource_manager = ResourceManager()
        self.resource_monitor = ResourceMonitor(self.resource_manager, monitoring_interval)
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def monitored_inference(self):
        """Context manager for monitored inference."""
        if self.enable_resource_monitoring:
            self.resource_monitor.start_monitoring()

        start_time = time.time()
        start_memory = self.resource_manager.get_system_resources()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.resource_manager.get_system_resources()

            if self.enable_resource_monitoring:
                self.resource_monitor.stop_monitoring()

    def run_single_evaluation(
        self,
        provider: UnifiedLocalProvider,
        prompt: str,
        expected_answer: str,
        dataset_name: str,
        evaluation_method: str = "local_model_eval",
        **generation_kwargs,
    ) -> LocalModelBenchmarkResult:
        """
        Run a single evaluation with local model monitoring.

        Args:
            provider: Unified local provider instance
            prompt: Input prompt
            expected_answer: Expected answer for evaluation
            dataset_name: Name of the benchmark dataset
            evaluation_method: Evaluation method to use
            **generation_kwargs: Additional generation parameters

        Returns:
            LocalModelBenchmarkResult with comprehensive metrics
        """
        # Pre-inference resource state
        pre_inference_resources = self.resource_manager.get_system_resources()

        # Track token counts
        prompt_tokens = len(prompt.split())  # Simplified token counting

        # Monitored inference
        start_time = time.perf_counter()
        first_token_time = None

        with self.monitored_inference():
            try:
                # Generate response
                response = provider.generate(prompt, **generation_kwargs)

                # Estimate first token latency (simplified)
                first_token_time = time.perf_counter()

            except Exception as e:
                logger.error(f"Error during inference: {e}")
                response = f"[ERROR: {str(e)}]"

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        if first_token_time:
            first_token_latency_ms = (first_token_time - start_time) * 1000
        else:
            first_token_latency_ms = None

        # Post-inference resource state
        post_inference_resources = self.resource_manager.get_system_resources()

        # Get resource monitoring data
        peak_usage = self.resource_monitor.get_peak_usage()

        # Calculate completion tokens
        completion_tokens = len(response.split()) if response else 0

        # Get model information
        model_info = provider.get_model_info()
        backend_name = (
            provider._get_backend_name() if hasattr(provider, "_get_backend_name") else "unknown"
        )

        # Create performance metrics
        performance_metrics = LocalModelPerformanceMetrics(
            inference_time_ms=inference_time_ms,
            tokens_per_second=(completion_tokens * 1000) / inference_time_ms
            if inference_time_ms > 0
            else 0.0,
            first_token_latency_ms=first_token_latency_ms,
            memory_used_mb=peak_usage.get("avg_memory_mb", post_inference_resources.used_ram_mb),
            memory_peak_mb=peak_usage.get("peak_memory_mb", post_inference_resources.used_ram_mb),
            gpu_memory_used_mb=peak_usage.get(
                "avg_gpu_memory_mb", post_inference_resources.used_vram_mb
            ),
            gpu_utilization_percent=peak_usage.get("avg_gpu_utilization", 0.0),
            cpu_utilization_percent=peak_usage.get(
                "avg_cpu_utilization", post_inference_resources.cpu_percent
            ),
            model_size_mb=model_info.get("model_size_mb", 0.0),
            context_length_used=prompt_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            response_length=len(response),
        )

        # Determine model format and quantization
        model_format = "unknown"
        quantization_level = None
        gpu_accelerated = peak_usage.get("peak_gpu_utilization", 0.0) > 10.0

        if hasattr(provider, "_get_current_backend"):
            backend = provider._get_current_backend()
            if hasattr(backend, "get_model_format"):
                model_format = backend.get_model_format()
            if hasattr(backend, "get_quantization_level"):
                quantization_level = backend.get_quantization_level()

        # Perform comprehensive evaluation
        result = evaluate_local_model_response(
            model_name=provider.model_name,
            dataset_name=dataset_name,
            prompt=prompt,
            response=response,
            expected_answer=expected_answer,
            performance_metrics=performance_metrics,
            backend_used=backend_name,
            model_format=model_format,
            evaluation_method=evaluation_method,
            quantization_level=quantization_level,
            gpu_accelerated=gpu_accelerated,
        )

        return result

    def run_batch_evaluation(
        self,
        provider: UnifiedLocalProvider,
        prompts: List[str],
        expected_answers: List[str],
        dataset_name: str,
        evaluation_method: str = "local_model_eval",
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **generation_kwargs,
    ) -> List[LocalModelBenchmarkResult]:
        """
        Run batch evaluation with resource monitoring.

        Args:
            provider: Unified local provider instance
            prompts: List of input prompts
            expected_answers: List of expected answers
            dataset_name: Name of the benchmark dataset
            evaluation_method: Evaluation method to use
            progress_callback: Optional progress callback function
            **generation_kwargs: Additional generation parameters

        Returns:
            List of LocalModelBenchmarkResult objects
        """
        if len(prompts) != len(expected_answers):
            raise ValueError("Number of prompts must match number of expected answers")

        results = []
        total_items = len(prompts)

        self.logger.info(
            f"Starting batch evaluation with {total_items} items using {provider.model_name}"
        )

        for i, (prompt, expected_answer) in enumerate(zip(prompts, expected_answers)):
            try:
                result = self.run_single_evaluation(
                    provider=provider,
                    prompt=prompt,
                    expected_answer=expected_answer,
                    dataset_name=dataset_name,
                    evaluation_method=evaluation_method,
                    **generation_kwargs,
                )
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, total_items)

                self.logger.debug(f"Completed evaluation {i + 1}/{total_items}")

            except Exception as e:
                self.logger.error(f"Error in evaluation {i + 1}/{total_items}: {e}")
                # Create error result
                error_result = LocalModelBenchmarkResult(
                    model_name=provider.model_name,
                    dataset_name=dataset_name,
                    prompt=prompt,
                    response=f"[ERROR: {str(e)}]",
                    expected_answer=expected_answer,
                    evaluation_score=0.0,
                    evaluation_method=evaluation_method,
                    performance_metrics=LocalModelPerformanceMetrics(
                        inference_time_ms=0.0,
                        tokens_per_second=0.0,
                        memory_used_mb=0.0,
                        memory_peak_mb=0.0,
                    ),
                    backend_used="error",
                    model_format="error",
                )
                results.append(error_result)

        self.logger.info(f"Completed batch evaluation: {len(results)} results")
        return results


def create_local_model_runner(
    enable_monitoring: bool = True, monitoring_interval: float = 0.5
) -> LocalModelBenchmarkRunner:
    """
    Factory function to create a local model benchmark runner.

    Args:
        enable_monitoring: Whether to enable resource monitoring
        monitoring_interval: Resource monitoring sampling interval

    Returns:
        LocalModelBenchmarkRunner instance
    """
    return LocalModelBenchmarkRunner(
        enable_resource_monitoring=enable_monitoring, monitoring_interval=monitoring_interval
    )
