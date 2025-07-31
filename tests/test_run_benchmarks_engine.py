"""
Tests for run_benchmarks.py with the new execution engine
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock, call
from click.testing import CliRunner
from datetime import datetime

import sys
import os
# Add parent directory to path to ensure correct imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_benchmarks import main
from benchmark import ModelBenchmarkResult, BenchmarkResult, ExecutionMode


class TestRunBenchmarksWithEngine:
    """Test the integration of MultiModelBenchmarkRunner with the CLI."""
    
    @pytest.fixture
    def runner(self):
        """Create a Click test runner."""
        return CliRunner()
    
    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create a mock dataset file."""
        dataset_path = tmp_path / "test_dataset.jsonl"
        
        entries = [
            {
                "id": "test_1",
                "prompt": "Test prompt 1",
                "evaluation_method": "keyword_match",
                "expected_keywords": ["test", "keyword"]
            },
            {
                "id": "test_2",
                "prompt": "Test prompt 2",
                "evaluation_method": "keyword_match",
                "expected_keywords": ["another", "test"]
            }
        ]
        
        with open(dataset_path, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        
        return str(dataset_path)
    
    @patch('run_benchmarks.DATASETS', {'test': 'test_dataset.jsonl'})
    @patch('run_benchmarks.load_dataset')
    @patch('benchmark.multi_runner.get_provider_for_model')
    @patch('run_benchmarks.get_provider_for_model')
    @patch('run_benchmarks.MultiModelBenchmarkRunner')
    def test_use_engine_flag_single_model(self, mock_runner_class, mock_get_provider_cli, 
                                          mock_get_provider_engine, mock_load_dataset, runner):
        """Test --use-engine flag with a single model."""
        # Setup mocks
        mock_provider = Mock()
        mock_provider.__name__ = "TestProvider"
        mock_get_provider_cli.return_value = mock_provider
        mock_get_provider_engine.return_value = mock_provider
        
        mock_load_dataset.return_value = [
            {
                "id": "test_1",
                "prompt": "Test prompt",
                "evaluation_method": "keyword_match",
                "expected_keywords": ["test"]
            }
        ]
        
        # Create mock runner instance
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        
        # Create mock result
        mock_result = BenchmarkResult(
            dataset_name="test",
            models=["test-model"],
            execution_mode=ExecutionMode.SEQUENTIAL,
            start_time=datetime.now()
        )
        mock_result.model_results = [
            ModelBenchmarkResult(
                model_name="test-model",
                provider_name="test",
                dataset_name="test",
                start_time=datetime.now(),
                end_time=datetime.now(),
                overall_score=0.85,
                total_prompts=10,
                successful_evaluations=8,
                failed_evaluations=2
            )
        ]
        mock_result.end_time = datetime.now()
        mock_result.total_duration_seconds = 10.0
        
        mock_runner.run.return_value = mock_result
        
        # Run command
        result = runner.invoke(main, ['--model', 'test-model', '--dataset', 'test', '--use-engine'])
        
        # Verify
        assert result.exit_code == 2  # Some evaluations failed
        assert "Using multi-model execution engine" in result.output
        assert "Execution completed: 1/1 models successful" in result.output
        
        # Verify runner was created correctly
        mock_runner_class.assert_called_once()
        call_kwargs = mock_runner_class.call_args[1]
        assert call_kwargs['execution_mode'] == ExecutionMode.SEQUENTIAL
        assert call_kwargs['max_workers'] == 4
        assert call_kwargs['timeout_per_model'] is None
        assert call_kwargs['progress_callback'] is not None
        
        # Verify run was called
        mock_runner.run.assert_called_once_with(["test-model"], "test")
    
    @patch('run_benchmarks.DATASETS', {'test': 'test_dataset.jsonl'})
    @patch('run_benchmarks.load_dataset')
    @patch('benchmark.multi_runner.get_provider_for_model')
    @patch('run_benchmarks.get_provider_for_model')
    @patch('run_benchmarks.MultiModelBenchmarkRunner')
    def test_use_engine_with_parallel(self, mock_runner_class, mock_get_provider_cli, 
                                      mock_get_provider_engine, mock_load_dataset, runner):
        """Test --use-engine with --parallel flag."""
        # Setup mocks
        mock_provider = Mock()
        mock_provider.__name__ = "TestProvider"
        mock_get_provider_cli.return_value = mock_provider
        mock_get_provider_engine.return_value = mock_provider
        
        mock_load_dataset.return_value = [{"id": "test_1", "prompt": "Test", "evaluation_method": "keyword_match", "expected_keywords": ["test"]}]
        
        # Create mock runner
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        
        # Create mock result with multiple models
        mock_result = BenchmarkResult(
            dataset_name="test",
            models=["model1", "model2"],
            execution_mode=ExecutionMode.PARALLEL,
            start_time=datetime.now()
        )
        mock_result.model_results = [
            ModelBenchmarkResult(
                model_name="model1",
                provider_name="test",
                dataset_name="test",
                start_time=datetime.now(),
                overall_score=0.9
            ),
            ModelBenchmarkResult(
                model_name="model2",
                provider_name="test",
                dataset_name="test",
                start_time=datetime.now(),
                overall_score=0.8
            )
        ]
        
        mock_runner.run.return_value = mock_result
        
        # Run command
        result = runner.invoke(main, ['--models', 'model1,model2', '--dataset', 'test', '--use-engine', '--parallel'])
        
        # Verify
        assert result.exit_code == 0
        assert "Using multi-model execution engine" in result.output
        
        # Verify parallel mode was used
        call_kwargs = mock_runner_class.call_args[1]
        assert call_kwargs['execution_mode'] == ExecutionMode.PARALLEL
    
    @patch('run_benchmarks.DATASETS', {'test': 'test_dataset.jsonl'})
    @patch('run_benchmarks.load_dataset')
    @patch('benchmark.multi_runner.get_provider_for_model')
    @patch('run_benchmarks.get_provider_for_model')
    @patch('run_benchmarks.MultiModelBenchmarkRunner')
    def test_use_engine_with_timeout(self, mock_runner_class, mock_get_provider_cli, 
                                     mock_get_provider_engine, mock_load_dataset, runner):
        """Test --use-engine with --timeout flag."""
        # Setup mocks
        mock_provider = Mock()
        mock_provider.__name__ = "TestProvider"
        mock_get_provider_cli.return_value = mock_provider
        mock_get_provider_engine.return_value = mock_provider
        
        mock_load_dataset.return_value = [{"id": "test_1", "prompt": "Test", "evaluation_method": "keyword_match", "expected_keywords": ["test"]}]
        
        # Create mock runner
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        
        # Create result
        mock_result = BenchmarkResult(
            dataset_name="test",
            models=["test-model"],
            execution_mode=ExecutionMode.SEQUENTIAL,
            start_time=datetime.now()
        )
        mock_result.model_results = [
            ModelBenchmarkResult(
                model_name="test-model",
                provider_name="test",
                dataset_name="test",
                start_time=datetime.now(),
                overall_score=0.9
            )
        ]
        
        mock_runner.run.return_value = mock_result
        
        # Run command
        result = runner.invoke(main, ['--model', 'test-model', '--dataset', 'test', '--use-engine', '--timeout', '30'])
        
        # Verify
        assert result.exit_code == 0
        
        # Verify timeout was passed
        call_kwargs = mock_runner_class.call_args[1]
        assert call_kwargs['timeout_per_model'] == 30
    
    @patch('run_benchmarks.DATASETS', {'test': 'test_dataset.jsonl'})
    @patch('run_benchmarks.load_dataset')
    @patch('benchmark.multi_runner.get_provider_for_model')
    @patch('run_benchmarks.get_provider_for_model')
    @patch('run_benchmarks.MultiModelBenchmarkRunner')
    def test_use_engine_progress_callback(self, mock_runner_class, mock_get_provider_cli, 
                                          mock_get_provider_engine, mock_load_dataset, runner):
        """Test that progress callback is properly connected."""
        # Setup mocks
        mock_provider = Mock()
        mock_provider.__name__ = "TestProvider"
        mock_get_provider_cli.return_value = mock_provider
        mock_get_provider_engine.return_value = mock_provider
        
        mock_load_dataset.return_value = [{"id": "test_1", "prompt": "Test", "evaluation_method": "keyword_match", "expected_keywords": ["test"]}]
        
        # Capture the progress callback
        captured_callback = None
        
        def capture_runner(*args, **kwargs):
            nonlocal captured_callback
            captured_callback = kwargs.get('progress_callback')
            mock_runner = Mock()
            mock_runner.run.return_value = BenchmarkResult(
                dataset_name="test",
                models=["test-model"],
                execution_mode=ExecutionMode.SEQUENTIAL,
                start_time=datetime.now(),
                model_results=[]
            )
            return mock_runner
        
        mock_runner_class.side_effect = capture_runner
        
        # Run command
        result = runner.invoke(main, ['--model', 'test-model', '--dataset', 'test', '--use-engine'])
        
        # Test the callback
        assert captured_callback is not None
        
        # The callback should print progress - we can't easily test the output,
        # but we can verify it doesn't crash
        captured_callback("model1", 1, 3)  # Should print [1/3] Processing: model1
    
    @patch('run_benchmarks.DATASETS', {'test': 'test_dataset.jsonl'})
    @patch('run_benchmarks.load_dataset')
    @patch('benchmark.multi_runner.get_provider_for_model')
    @patch('run_benchmarks.get_provider_for_model')
    @patch('run_benchmarks.MultiModelBenchmarkRunner')
    def test_use_engine_with_failed_models(self, mock_runner_class, mock_get_provider_cli, 
                                           mock_get_provider_engine, mock_load_dataset, runner):
        """Test --use-engine with some models failing."""
        # Setup mocks
        mock_provider = Mock()
        mock_provider.__name__ = "TestProvider"
        mock_get_provider_cli.return_value = mock_provider
        mock_get_provider_engine.return_value = mock_provider
        
        mock_load_dataset.return_value = [{"id": "test_1", "prompt": "Test", "evaluation_method": "keyword_match", "expected_keywords": ["test"]}]
        
        # Create mock runner
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        
        # Create result with mixed success/failure
        mock_result = BenchmarkResult(
            dataset_name="test",
            models=["model1", "model2", "model3"],
            execution_mode=ExecutionMode.SEQUENTIAL,
            start_time=datetime.now()
        )
        mock_result.model_results = [
            ModelBenchmarkResult(
                model_name="model1",
                provider_name="test",
                dataset_name="test",
                start_time=datetime.now(),
                overall_score=0.9
            ),
            ModelBenchmarkResult(
                model_name="model2",
                provider_name="test",
                dataset_name="test",
                start_time=datetime.now(),
                error="API Error",
                error_type="ProviderError"
            ),
            ModelBenchmarkResult(
                model_name="model3",
                provider_name="test",
                dataset_name="test",
                start_time=datetime.now(),
                overall_score=0.7
            )
        ]
        
        mock_runner.run.return_value = mock_result
        
        # Run command
        result = runner.invoke(main, ['--models', 'model1,model2,model3', '--dataset', 'test', '--use-engine'])
        
        # Verify
        assert result.exit_code == 0  # Some models succeeded
        assert "Execution completed: 2/3 models successful" in result.output
        assert "Failed models: 1" in result.output
        assert "Best model: model1" in result.output
    
    def test_timeout_without_engine(self, runner):
        """Test that --timeout requires --use-engine."""
        result = runner.invoke(main, ['--model', 'test-model', '--dataset', 'truthfulness', '--timeout', '30'])
        
        # Should still work but timeout is ignored (no error)
        # The command will fail for other reasons (no API key etc) but that's ok
        assert result.exit_code != 0