"""
Multi-Model Benchmark Execution Engine

This module provides the core execution engine for running benchmarks across
multiple LLM models with support for sequential and parallel execution,
error isolation, timeout handling, and result aggregation.
"""

import logging
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum
import threading

from src.providers import get_provider_for_model
from src.providers.exceptions import ProviderError


logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for multi-model benchmarking."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


@dataclass
class ModelBenchmarkResult:
    """Result from benchmarking a single model."""
    model_name: str
    provider_name: str
    dataset_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    total_prompts: int = 0
    successful_evaluations: int = 0
    failed_evaluations: int = 0
    overall_score: float = 0.0
    evaluations: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    error_type: Optional[str] = None
    timed_out: bool = False
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if the benchmark completed successfully."""
        return self.error is None and not self.timed_out
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time from evaluations."""
        if not self.evaluations:
            return 0.0
        
        response_times = [
            eval_data.get('response_time_seconds', 0)
            for eval_data in self.evaluations
            if 'response_time_seconds' in eval_data
        ]
        
        if response_times:
            return sum(response_times) / len(response_times)
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with existing code."""
        return {
            'model': self.model_name,
            'provider': self.provider_name,
            'dataset': self.dataset_name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_duration_seconds': self.duration_seconds,
            'total_prompts': self.total_prompts,
            'successful_evaluations': self.successful_evaluations,
            'failed_evaluations': self.failed_evaluations,
            'overall_score': self.overall_score,
            'evaluations': self.evaluations,
            'error': self.error,
            'error_type': self.error_type,
            'timed_out': self.timed_out,
            'model_config': self.model_config,
            'average_response_time_seconds': self.average_response_time
        }


@dataclass
class BenchmarkResult:
    """Aggregated results from multi-model benchmarking."""
    dataset_name: str
    models: List[str]
    execution_mode: ExecutionMode
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_seconds: float = 0.0
    model_results: List[ModelBenchmarkResult] = field(default_factory=list)
    
    @property
    def successful_models(self) -> List[ModelBenchmarkResult]:
        """Get list of successfully benchmarked models."""
        return [r for r in self.model_results if r.success]
    
    @property
    def failed_models(self) -> List[ModelBenchmarkResult]:
        """Get list of failed model benchmarks."""
        return [r for r in self.model_results if not r.success]
    
    @property
    def summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        successful = self.successful_models
        
        if not successful:
            return {
                'total_models': len(self.models),
                'successful_models': 0,
                'failed_models': len(self.failed_models),
                'average_score': 0.0,
                'best_model': None,
                'worst_model': None
            }
        
        scores = [r.overall_score for r in successful]
        best_model = max(successful, key=lambda r: r.overall_score)
        worst_model = min(successful, key=lambda r: r.overall_score)
        
        return {
            'total_models': len(self.models),
            'successful_models': len(successful),
            'failed_models': len(self.failed_models),
            'average_score': sum(scores) / len(scores),
            'best_model': best_model.model_name,
            'best_score': best_model.overall_score,
            'worst_model': worst_model.model_name,
            'worst_score': worst_model.overall_score
        }


class ProgressTracker:
    """Thread-safe progress tracking for multi-model benchmarks."""
    
    def __init__(self, total_models: int, callback: Optional[Callable[[str, int, int], None]] = None):
        """
        Initialize progress tracker.
        
        Args:
            total_models: Total number of models to benchmark
            callback: Optional callback function(model_name, current, total)
        """
        self.total_models = total_models
        self.completed_models = 0
        self.current_model = None
        self.callback = callback
        self._lock = threading.Lock()
    
    def start_model(self, model_name: str):
        """Mark a model as started."""
        with self._lock:
            self.current_model = model_name
            if self.callback:
                self.callback(model_name, self.completed_models, self.total_models)
    
    def complete_model(self, model_name: str):
        """Mark a model as completed."""
        with self._lock:
            self.completed_models += 1
            if self.callback:
                self.callback(model_name, self.completed_models, self.total_models)
    
    @property
    def progress_percentage(self) -> float:
        """Get current progress percentage."""
        with self._lock:
            return (self.completed_models / self.total_models) * 100 if self.total_models > 0 else 0


class MultiModelBenchmarkRunner:
    """
    Execution engine for running benchmarks across multiple models.
    
    This class provides both sequential and parallel execution modes with
    proper error isolation, timeout handling, and progress tracking.
    """
    
    def __init__(
        self,
        benchmark_function: Callable[[str, str], Dict[str, Any]],
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        max_workers: int = 4,
        timeout_per_model: Optional[int] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ):
        """
        Initialize the multi-model benchmark runner.
        
        Args:
            benchmark_function: Function that takes (model_name, dataset_name) and returns results
            execution_mode: Sequential or parallel execution
            max_workers: Maximum number of parallel workers
            timeout_per_model: Optional timeout in seconds per model
            progress_callback: Optional callback for progress updates
        """
        self.benchmark_function = benchmark_function
        self.execution_mode = execution_mode
        self.max_workers = max_workers
        self.timeout_per_model = timeout_per_model
        self.progress_callback = progress_callback
    
    def run(
        self,
        models: List[str],
        dataset_name: str,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run benchmarks for multiple models.
        
        Args:
            models: List of model names to benchmark
            dataset_name: Name of the dataset to use
            **kwargs: Additional arguments passed to benchmark function
            
        Returns:
            Aggregated benchmark results
        """
        start_time = datetime.now()
        progress_tracker = ProgressTracker(len(models), self.progress_callback)
        
        # Initialize result container
        result = BenchmarkResult(
            dataset_name=dataset_name,
            models=models,
            execution_mode=self.execution_mode,
            start_time=start_time
        )
        
        # Validate all models exist before starting
        logger.info(f"Validating {len(models)} models...")
        for model in models:
            try:
                provider_class = get_provider_for_model(model)
                logger.debug(f"Model {model} -> {provider_class.__name__}")
            except ValueError as e:
                # Create error result for invalid model
                model_result = ModelBenchmarkResult(
                    model_name=model,
                    provider_name="unknown",
                    dataset_name=dataset_name,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    error=str(e),
                    error_type="ModelNotFoundError"
                )
                result.model_results.append(model_result)
                logger.error(f"Model validation failed for {model}: {e}")
        
        # Filter out invalid models
        valid_models = [m for m in models if not any(
            r.model_name == m and r.error for r in result.model_results
        )]
        
        if not valid_models:
            logger.error("No valid models to benchmark")
            result.end_time = datetime.now()
            result.total_duration_seconds = (result.end_time - start_time).total_seconds()
            return result
        
        # Execute benchmarks
        logger.info(f"Running benchmarks in {self.execution_mode.value} mode for {len(valid_models)} models")
        
        if self.execution_mode == ExecutionMode.SEQUENTIAL:
            model_results = self._run_sequential(valid_models, dataset_name, progress_tracker, **kwargs)
        else:
            model_results = self._run_parallel(valid_models, dataset_name, progress_tracker, **kwargs)
        
        # Add results
        result.model_results.extend(model_results)
        result.end_time = datetime.now()
        result.total_duration_seconds = (result.end_time - start_time).total_seconds()
        
        # Log summary
        stats = result.summary_stats
        logger.info(
            f"Benchmark completed: {stats['successful_models']}/{stats['total_models']} models successful, "
            f"average score: {stats['average_score']:.2%}"
        )
        
        return result
    
    def _run_sequential(
        self,
        models: List[str],
        dataset_name: str,
        progress_tracker: ProgressTracker,
        **kwargs
    ) -> List[ModelBenchmarkResult]:
        """Run benchmarks sequentially."""
        results = []
        
        for model in models:
            progress_tracker.start_model(model)
            logger.info(f"Starting benchmark for model: {model}")
            
            result = self._run_single_model(model, dataset_name, **kwargs)
            results.append(result)
            
            progress_tracker.complete_model(model)
            
            if result.success:
                logger.info(f"Completed {model}: score={result.overall_score:.2%}")
            else:
                logger.error(f"Failed {model}: {result.error}")
        
        return results
    
    def _run_parallel(
        self,
        models: List[str],
        dataset_name: str,
        progress_tracker: ProgressTracker,
        **kwargs
    ) -> List[ModelBenchmarkResult]:
        """Run benchmarks in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=min(len(models), self.max_workers)) as executor:
            # Submit all tasks
            future_to_model = {
                executor.submit(
                    self._run_single_model_with_progress,
                    model,
                    dataset_name,
                    progress_tracker,
                    **kwargs
                ): model
                for model in models
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_model):
                model = future_to_model[future]
                
                try:
                    result = future.result(timeout=1)  # Short timeout for getting result
                    results.append(result)
                    
                    if result.success:
                        logger.info(f"Completed {model}: score={result.overall_score:.2%}")
                    else:
                        logger.error(f"Failed {model}: {result.error}")
                        
                except Exception as e:
                    # This should rarely happen as errors are caught in _run_single_model
                    logger.exception(f"Unexpected error getting result for {model}")
                    result = ModelBenchmarkResult(
                        model_name=model,
                        provider_name="unknown",
                        dataset_name=dataset_name,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error=f"Unexpected error: {str(e)}",
                        error_type=type(e).__name__
                    )
                    results.append(result)
        
        return results
    
    def _run_single_model_with_progress(
        self,
        model: str,
        dataset_name: str,
        progress_tracker: ProgressTracker,
        **kwargs
    ) -> ModelBenchmarkResult:
        """Run single model with progress tracking."""
        progress_tracker.start_model(model)
        result = self._run_single_model(model, dataset_name, **kwargs)
        progress_tracker.complete_model(model)
        return result
    
    def _run_single_model(
        self,
        model: str,
        dataset_name: str,
        **kwargs
    ) -> ModelBenchmarkResult:
        """
        Run benchmark for a single model with error isolation.
        
        Args:
            model: Model name
            dataset_name: Dataset name
            **kwargs: Additional arguments for benchmark function
            
        Returns:
            Model benchmark result
        """
        start_time = datetime.now()
        
        # Get provider name
        try:
            provider_class = get_provider_for_model(model)
            provider_name = provider_class.__name__.replace('Provider', '').lower()
        except Exception:
            provider_name = "unknown"
        
        # Create result object
        result = ModelBenchmarkResult(
            model_name=model,
            provider_name=provider_name,
            dataset_name=dataset_name,
            start_time=start_time
        )
        
        try:
            # Run benchmark with timeout if specified
            if self.timeout_per_model:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.benchmark_function, model, dataset_name, **kwargs)
                    try:
                        benchmark_data = future.result(timeout=self.timeout_per_model)
                    except TimeoutError:
                        result.timed_out = True
                        result.error = f"Benchmark timed out after {self.timeout_per_model} seconds"
                        result.error_type = "TimeoutError"
                        raise
            else:
                # Run without timeout
                benchmark_data = self.benchmark_function(model, dataset_name, **kwargs)
            
            # Extract results
            result.total_prompts = benchmark_data.get('total_prompts', 0)
            result.successful_evaluations = benchmark_data.get('successful_evaluations', 0)
            result.failed_evaluations = benchmark_data.get('failed_evaluations', 0)
            result.overall_score = benchmark_data.get('overall_score', 0.0)
            result.evaluations = benchmark_data.get('evaluations', [])
            result.model_config = benchmark_data.get('model_config', {})
            
            # Check for errors in benchmark data
            if 'error' in benchmark_data:
                result.error = benchmark_data['error']
                result.error_type = benchmark_data.get('error_type', 'BenchmarkError')
            
        except TimeoutError:
            # Already handled above
            logger.warning(f"Model {model} timed out after {self.timeout_per_model}s")
            
        except ProviderError as e:
            result.error = str(e)
            result.error_type = type(e).__name__
            logger.error(f"Provider error for {model}: {e}")
            
        except Exception as e:
            result.error = str(e)
            result.error_type = type(e).__name__
            logger.exception(f"Unexpected error benchmarking {model}")
        
        finally:
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - start_time).total_seconds()
        
        return result