"""
Tests for the Multi-Model Benchmark Execution Engine
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, call
import concurrent.futures
import threading
import time

from benchmark import MultiModelBenchmarkRunner, BenchmarkResult, ModelBenchmarkResult, ExecutionMode
from benchmark.multi_runner import ProgressTracker
from llm_providers.exceptions import ProviderError, RateLimitError


class TestModelBenchmarkResult:
    """Test the ModelBenchmarkResult dataclass."""
    
    def test_success_property(self):
        """Test the success property logic."""
        # Successful result
        result = ModelBenchmarkResult(
            model_name="test-model",
            provider_name="test",
            dataset_name="test-dataset",
            start_time=datetime.now(),
            end_time=datetime.now(),
            overall_score=0.85
        )
        assert result.success is True
        
        # Failed with error
        result.error = "API error"
        assert result.success is False
        
        # Failed with timeout
        result.error = None
        result.timed_out = True
        assert result.success is False
    
    def test_average_response_time(self):
        """Test average response time calculation."""
        result = ModelBenchmarkResult(
            model_name="test-model",
            provider_name="test",
            dataset_name="test-dataset",
            start_time=datetime.now()
        )
        
        # No evaluations
        assert result.average_response_time == 0.0
        
        # With evaluations
        result.evaluations = [
            {'response_time_seconds': 1.0},
            {'response_time_seconds': 2.0},
            {'response_time_seconds': 3.0},
            {'no_time': True}  # Should be ignored
        ]
        assert result.average_response_time == 2.0
    
    def test_to_dict(self):
        """Test conversion to dictionary format."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=10)
        
        result = ModelBenchmarkResult(
            model_name="gpt-4",
            provider_name="openai",
            dataset_name="truthfulness",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=10.0,
            total_prompts=5,
            successful_evaluations=4,
            failed_evaluations=1,
            overall_score=0.8,
            evaluations=[{'test': 'data'}],
            model_config={'temperature': 0.7}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['model'] == "gpt-4"
        assert result_dict['provider'] == "openai"
        assert result_dict['dataset'] == "truthfulness"
        assert result_dict['total_prompts'] == 5
        assert result_dict['successful_evaluations'] == 4
        assert result_dict['failed_evaluations'] == 1
        assert result_dict['overall_score'] == 0.8
        assert result_dict['evaluations'] == [{'test': 'data'}]
        assert result_dict['model_config'] == {'temperature': 0.7}
        assert result_dict['average_response_time_seconds'] == 0.0
        assert result_dict['error'] is None
        assert result_dict['timed_out'] is False


class TestBenchmarkResult:
    """Test the BenchmarkResult aggregation class."""
    
    def test_successful_and_failed_models(self):
        """Test categorization of successful and failed models."""
        result = BenchmarkResult(
            dataset_name="test",
            models=["model1", "model2", "model3"],
            execution_mode=ExecutionMode.SEQUENTIAL,
            start_time=datetime.now()
        )
        
        # Add results
        result.model_results = [
            ModelBenchmarkResult(
                model_name="model1",
                provider_name="provider1",
                dataset_name="test",
                start_time=datetime.now(),
                overall_score=0.9
            ),
            ModelBenchmarkResult(
                model_name="model2",
                provider_name="provider2",
                dataset_name="test",
                start_time=datetime.now(),
                error="Failed to connect"
            ),
            ModelBenchmarkResult(
                model_name="model3",
                provider_name="provider3",
                dataset_name="test",
                start_time=datetime.now(),
                timed_out=True
            )
        ]
        
        assert len(result.successful_models) == 1
        assert result.successful_models[0].model_name == "model1"
        
        assert len(result.failed_models) == 2
        assert set(r.model_name for r in result.failed_models) == {"model2", "model3"}
    
    def test_summary_stats(self):
        """Test summary statistics generation."""
        result = BenchmarkResult(
            dataset_name="test",
            models=["model1", "model2", "model3"],
            execution_mode=ExecutionMode.PARALLEL,
            start_time=datetime.now()
        )
        
        # No results
        stats = result.summary_stats
        assert stats['total_models'] == 3
        assert stats['successful_models'] == 0
        assert stats['failed_models'] == 0
        assert stats['average_score'] == 0.0
        assert stats['best_model'] is None
        
        # With results
        result.model_results = [
            ModelBenchmarkResult(
                model_name="model1",
                provider_name="provider1",
                dataset_name="test",
                start_time=datetime.now(),
                overall_score=0.9
            ),
            ModelBenchmarkResult(
                model_name="model2",
                provider_name="provider2",
                dataset_name="test",
                start_time=datetime.now(),
                overall_score=0.7
            ),
            ModelBenchmarkResult(
                model_name="model3",
                provider_name="provider3",
                dataset_name="test",
                start_time=datetime.now(),
                error="Failed"
            )
        ]
        
        stats = result.summary_stats
        assert stats['total_models'] == 3
        assert stats['successful_models'] == 2
        assert stats['failed_models'] == 1
        assert stats['average_score'] == 0.8
        assert stats['best_model'] == "model1"
        assert stats['best_score'] == 0.9
        assert stats['worst_model'] == "model2"
        assert stats['worst_score'] == 0.7


class TestProgressTracker:
    """Test the thread-safe progress tracker."""
    
    def test_progress_tracking(self):
        """Test basic progress tracking functionality."""
        callback_calls = []
        
        def callback(model, current, total):
            callback_calls.append((model, current, total))
        
        tracker = ProgressTracker(total_models=3, callback=callback)
        
        # Start and complete models
        tracker.start_model("model1")
        assert callback_calls[-1] == ("model1", 0, 3)
        
        tracker.complete_model("model1")
        assert callback_calls[-1] == ("model1", 1, 3)
        assert tracker.progress_percentage == pytest.approx(33.33, rel=0.01)
        
        tracker.start_model("model2")
        tracker.complete_model("model2")
        assert tracker.progress_percentage == pytest.approx(66.67, rel=0.01)
        
        tracker.start_model("model3")
        tracker.complete_model("model3")
        assert tracker.progress_percentage == 100.0
    
    def test_thread_safety(self):
        """Test thread-safe operations."""
        tracker = ProgressTracker(total_models=100)
        errors = []
        
        def complete_models():
            try:
                for i in range(10):
                    tracker.start_model(f"model-{i}")
                    tracker.complete_model(f"model-{i}")
            except Exception as e:
                errors.append(e)
        
        # Run from multiple threads
        threads = [threading.Thread(target=complete_models) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have no errors and correct count
        assert len(errors) == 0
        assert tracker.completed_models == 100
        assert tracker.progress_percentage == 100.0


class TestMultiModelBenchmarkRunner:
    """Test the main benchmark runner class."""
    
    @pytest.fixture
    def mock_benchmark_function(self):
        """Create a mock benchmark function."""
        def benchmark(model_name, dataset_name, **kwargs):
            # Simulate different behaviors based on model name
            if "error" in model_name:
                raise RuntimeError(f"Simulated error for {model_name}")
            elif "timeout" in model_name:
                time.sleep(10)  # Will timeout if timeout is set
            else:
                return {
                    'total_prompts': 10,
                    'successful_evaluations': 8,
                    'failed_evaluations': 2,
                    'overall_score': 0.8,
                    'evaluations': [{'id': i} for i in range(10)],
                    'model_config': {'temperature': 0.7}
                }
        
        return Mock(side_effect=benchmark)
    
    @patch('benchmark.multi_runner.get_provider_for_model')
    def test_sequential_execution(self, mock_get_provider, mock_benchmark_function):
        """Test sequential execution mode."""
        # Setup mock provider
        mock_provider_class = Mock()
        mock_provider_class.__name__ = "TestProvider"
        mock_get_provider.return_value = mock_provider_class
        
        runner = MultiModelBenchmarkRunner(
            benchmark_function=mock_benchmark_function,
            execution_mode=ExecutionMode.SEQUENTIAL
        )
        
        models = ["model1", "model2", "model3"]
        result = runner.run(models, "test_dataset")
        
        # Verify execution
        assert len(result.model_results) == 3
        assert all(r.model_name in models for r in result.model_results)
        assert result.execution_mode == ExecutionMode.SEQUENTIAL
        
        # Verify sequential calls
        assert mock_benchmark_function.call_count == 3
        calls = [call("model1", "test_dataset"), 
                 call("model2", "test_dataset"),
                 call("model3", "test_dataset")]
        mock_benchmark_function.assert_has_calls(calls, any_order=False)
        
        # Check results
        successful = result.successful_models
        assert len(successful) == 3
        assert all(r.overall_score == 0.8 for r in successful)
    
    @patch('benchmark.multi_runner.get_provider_for_model')
    def test_parallel_execution(self, mock_get_provider, mock_benchmark_function):
        """Test parallel execution mode."""
        mock_provider_class = Mock()
        mock_provider_class.__name__ = "TestProvider"
        mock_get_provider.return_value = mock_provider_class
        
        runner = MultiModelBenchmarkRunner(
            benchmark_function=mock_benchmark_function,
            execution_mode=ExecutionMode.PARALLEL,
            max_workers=2
        )
        
        models = ["model1", "model2", "model3", "model4"]
        result = runner.run(models, "test_dataset")
        
        # Verify execution
        assert len(result.model_results) == 4
        assert result.execution_mode == ExecutionMode.PARALLEL
        assert mock_benchmark_function.call_count == 4
        
        # All should be successful
        assert len(result.successful_models) == 4
    
    @patch('benchmark.multi_runner.get_provider_for_model')
    def test_error_handling(self, mock_get_provider, mock_benchmark_function):
        """Test error handling and isolation."""
        mock_provider_class = Mock()
        mock_provider_class.__name__ = "TestProvider"
        mock_get_provider.return_value = mock_provider_class
        
        runner = MultiModelBenchmarkRunner(
            benchmark_function=mock_benchmark_function,
            execution_mode=ExecutionMode.SEQUENTIAL
        )
        
        models = ["model1", "error-model", "model3"]
        result = runner.run(models, "test_dataset")
        
        # Should have results for all models
        assert len(result.model_results) == 3
        
        # Check error isolation
        successful = result.successful_models
        failed = result.failed_models
        
        assert len(successful) == 2
        assert len(failed) == 1
        assert failed[0].model_name == "error-model"
        assert "Simulated error" in failed[0].error
        assert failed[0].error_type == "RuntimeError"
    
    @patch('benchmark.multi_runner.get_provider_for_model')
    def test_timeout_handling(self, mock_get_provider):
        """Test timeout handling."""
        mock_provider_class = Mock()
        mock_provider_class.__name__ = "TestProvider"
        mock_get_provider.return_value = mock_provider_class
        
        # Create a benchmark function that takes too long
        def slow_benchmark(model_name, dataset_name, **kwargs):
            time.sleep(2)
            return {'overall_score': 1.0}
        
        runner = MultiModelBenchmarkRunner(
            benchmark_function=slow_benchmark,
            execution_mode=ExecutionMode.SEQUENTIAL,
            timeout_per_model=1  # 1 second timeout
        )
        
        models = ["slow-model"]
        result = runner.run(models, "test_dataset")
        
        # Should timeout
        assert len(result.model_results) == 1
        assert result.model_results[0].timed_out is True
        assert "timed out after 1 seconds" in result.model_results[0].error
    
    @patch('benchmark.multi_runner.get_provider_for_model')
    def test_invalid_model_handling(self, mock_get_provider):
        """Test handling of invalid models."""
        # Make first model invalid
        mock_get_provider.side_effect = [
            ValueError("Model not found: invalid-model"),
            Mock(__name__="ValidProvider"),
            Mock(__name__="ValidProvider")
        ]
        
        runner = MultiModelBenchmarkRunner(
            benchmark_function=Mock(return_value={'overall_score': 0.5}),
            execution_mode=ExecutionMode.SEQUENTIAL
        )
        
        models = ["invalid-model", "valid-model1", "valid-model2"]
        result = runner.run(models, "test_dataset")
        
        # Should have results for all models
        assert len(result.model_results) == 3
        
        # First should be error, others successful
        assert result.model_results[0].error == "Model not found: invalid-model"
        assert result.model_results[0].error_type == "ModelNotFoundError"
        assert len(result.successful_models) == 2
        assert len(result.failed_models) == 1
    
    @patch('benchmark.multi_runner.get_provider_for_model')
    def test_progress_callback(self, mock_get_provider, mock_benchmark_function):
        """Test progress callback functionality."""
        mock_provider_class = Mock()
        mock_provider_class.__name__ = "TestProvider"
        mock_get_provider.return_value = mock_provider_class
        
        progress_calls = []
        
        def progress_callback(model, current, total):
            progress_calls.append((model, current, total))
        
        runner = MultiModelBenchmarkRunner(
            benchmark_function=mock_benchmark_function,
            execution_mode=ExecutionMode.SEQUENTIAL,
            progress_callback=progress_callback
        )
        
        models = ["model1", "model2"]
        runner.run(models, "test_dataset")
        
        # Should have start and complete calls for each model
        expected_calls = [
            ("model1", 0, 2),  # start model1
            ("model1", 1, 2),  # complete model1
            ("model2", 1, 2),  # start model2
            ("model2", 2, 2),  # complete model2
        ]
        
        assert progress_calls == expected_calls
    
    @patch('benchmark.multi_runner.get_provider_for_model')
    def test_provider_error_handling(self, mock_get_provider):
        """Test handling of specific provider errors."""
        mock_provider_class = Mock()
        mock_provider_class.__name__ = "TestProvider"
        mock_get_provider.return_value = mock_provider_class
        
        def benchmark_with_provider_error(model_name, dataset_name, **kwargs):
            if "rate-limit" in model_name:
                raise RateLimitError("Rate limit exceeded", retry_after=60)
            elif "provider-error" in model_name:
                raise ProviderError("Generic provider error")
            else:
                return {'overall_score': 0.9}
        
        runner = MultiModelBenchmarkRunner(
            benchmark_function=benchmark_with_provider_error,
            execution_mode=ExecutionMode.SEQUENTIAL
        )
        
        models = ["good-model", "rate-limit-model", "provider-error-model"]
        result = runner.run(models, "test_dataset")
        
        # Check each result
        assert result.model_results[0].success is True
        assert result.model_results[0].overall_score == 0.9
        
        assert result.model_results[1].success is False
        assert result.model_results[1].error_type == "RateLimitError"
        assert "Rate limit exceeded" in result.model_results[1].error
        
        assert result.model_results[2].success is False
        assert result.model_results[2].error_type == "ProviderError"
        assert "Generic provider error" in result.model_results[2].error
    
    @patch('benchmark.multi_runner.get_provider_for_model')
    def test_empty_model_list(self, mock_get_provider):
        """Test handling of empty model list."""
        runner = MultiModelBenchmarkRunner(
            benchmark_function=Mock(),
            execution_mode=ExecutionMode.SEQUENTIAL
        )
        
        result = runner.run([], "test_dataset")
        
        assert len(result.model_results) == 0
        assert result.summary_stats['total_models'] == 0
        assert result.summary_stats['successful_models'] == 0
    
    @patch('benchmark.multi_runner.get_provider_for_model')
    def test_all_models_fail(self, mock_get_provider):
        """Test when all models fail validation."""
        mock_get_provider.side_effect = ValueError("Model not found")
        
        runner = MultiModelBenchmarkRunner(
            benchmark_function=Mock(),
            execution_mode=ExecutionMode.SEQUENTIAL
        )
        
        models = ["invalid1", "invalid2", "invalid3"]
        result = runner.run(models, "test_dataset")
        
        # Should have error results for all
        assert len(result.model_results) == 3
        assert all(r.error is not None for r in result.model_results)
        assert result.summary_stats['successful_models'] == 0
        assert result.summary_stats['failed_models'] == 3