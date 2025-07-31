"""Tests for run_benchmarks module."""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from run_benchmarks import (
    load_dataset,
    main,
    retry_with_backoff,
    run_benchmark,
)


class TestRetryWithBackoff:
    """Test retry_with_backoff function."""

    def test_retry_success_first_attempt(self):
        """Test successful execution on first attempt."""
        mock_func = Mock(return_value="success")
        result = retry_with_backoff(mock_func, max_retries=3)
        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_success_after_failures(self):
        """Test successful execution after failures."""
        mock_func = Mock(side_effect=[Exception("fail"), Exception("fail"), "success"])
        result = retry_with_backoff(mock_func, max_retries=3)
        assert result == "success"
        assert mock_func.call_count == 3

    def test_retry_all_attempts_fail(self):
        """Test all retry attempts fail."""
        mock_func = Mock(side_effect=Exception("persistent failure"))
        with pytest.raises(Exception) as exc_info:
            retry_with_backoff(mock_func, max_retries=3)
        assert str(exc_info.value) == "persistent failure"
        assert mock_func.call_count == 3

    @patch('time.sleep')
    def test_retry_exponential_backoff(self, mock_sleep):
        """Test exponential backoff delays."""
        mock_func = Mock(side_effect=[Exception("fail"), Exception("fail"), "success"])
        result = retry_with_backoff(mock_func, max_retries=3, base_delay=1.0)
        assert result == "success"
        # Check sleep was called with exponential delays
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)  # First retry: 1.0 * 2^0
        mock_sleep.assert_any_call(2.0)  # Second retry: 1.0 * 2^1


class TestLoadDataset:
    """Test load_dataset function."""

    def test_load_dataset_success(self):
        """Test successful dataset loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"id": "test1", "prompt": "Test prompt 1", "evaluation_method": "keyword_match", "expected_keywords": ["test"]}\n')
            f.write('{"id": "test2", "prompt": "Test prompt 2", "evaluation_method": "keyword_match", "expected_keywords": ["test"]}\n')
            f.flush()

            try:
                entries = load_dataset(f.name)
                assert len(entries) == 2
                assert entries[0]['id'] == 'test1'
                assert entries[1]['id'] == 'test2'
            finally:
                os.unlink(f.name)

    def test_load_dataset_file_not_found(self):
        """Test loading non-existent dataset."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_dataset('/nonexistent/path.jsonl')
        assert "Dataset file not found" in str(exc_info.value)

    def test_load_dataset_invalid_json(self):
        """Test dataset with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"valid": "json"}\n')
            f.write('invalid json\n')
            f.write('{"id": "test", "prompt": "Test", "evaluation_method": "keyword_match", "expected_keywords": ["test"]}\n')
            f.flush()

            try:
                entries = load_dataset(f.name)
                assert len(entries) == 1  # Only valid entry
                assert entries[0]['id'] == 'test'
            finally:
                os.unlink(f.name)

    def test_load_dataset_missing_fields(self):
        """Test dataset with missing required fields."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"id": "test1", "prompt": "Test"}\n')  # Missing evaluation_method
            f.write('{"prompt": "Test", "evaluation_method": "keyword_match"}\n')  # Missing id
            f.write('{"id": "test3", "prompt": "Test", "evaluation_method": "keyword_match", "expected_keywords": ["test"]}\n')
            f.flush()

            try:
                entries = load_dataset(f.name)
                assert len(entries) == 1  # Only valid entry
                assert entries[0]['id'] == 'test3'
            finally:
                os.unlink(f.name)

    def test_load_dataset_empty_file(self):
        """Test loading empty dataset file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.flush()

            try:
                with pytest.raises(ValueError) as exc_info:
                    load_dataset(f.name)
                assert "No valid entries found" in str(exc_info.value)
            finally:
                os.unlink(f.name)

    def test_load_dataset_keyword_match_validation(self):
        """Test keyword_match method validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Missing expected_keywords
            f.write('{"id": "test1", "prompt": "Test", "evaluation_method": "keyword_match"}\n')
            # expected_keywords not a list
            f.write('{"id": "test2", "prompt": "Test", "evaluation_method": "keyword_match", "expected_keywords": "not a list"}\n')
            # Valid entry
            f.write('{"id": "test3", "prompt": "Test", "evaluation_method": "keyword_match", "expected_keywords": ["test"]}\n')
            f.flush()

            try:
                entries = load_dataset(f.name)
                assert len(entries) == 1
                assert entries[0]['id'] == 'test3'
            finally:
                os.unlink(f.name)


class TestRunBenchmark:
    """Test run_benchmark function."""

    @patch('run_benchmarks.get_api_key')
    @patch('run_benchmarks.get_model_config')
    @patch('run_benchmarks.load_dataset')
    @patch('run_benchmarks.CSVResultLogger')
    def test_run_benchmark_success(self, mock_logger, mock_load_dataset, mock_get_config, mock_get_key):
        """Test successful benchmark run."""
        # Setup mocks
        mock_get_key.return_value = 'test-api-key'
        mock_get_config.return_value = {
            'temperature': 0.7,
            'max_tokens': 1000,
            'timeout_seconds': 30,
            'max_retries': 3,
            'retry_delay': 1.0
        }
        mock_load_dataset.return_value = [
            {
                'id': 'test1',
                'prompt': 'Test prompt',
                'evaluation_method': 'keyword_match',
                'expected_keywords': ['test']
            }
        ]

        # Mock provider
        mock_provider = Mock()
        mock_provider.generate.return_value = 'Test response'

        with patch('run_benchmarks.GeminiProvider', return_value=mock_provider):
            with patch('run_benchmarks.keyword_match', return_value={
                'success': True,
                'score': 1.0,
                'matched_keywords': ['test']
            }):
                with patch('benchmarks.truthfulness.validate_dataset'):
                    results = run_benchmark('google', 'truthfulness')

        assert results['provider'] == 'google'
        assert results['dataset'] == 'truthfulness'
        assert results['total_prompts'] == 1
        assert results['successful_evaluations'] == 1
        assert results['failed_evaluations'] == 0
        assert results['overall_score'] == 1.0

    @patch('run_benchmarks.get_api_key')
    def test_run_benchmark_missing_api_key(self, mock_get_key):
        """Test benchmark with missing API key."""
        from config import ConfigurationError
        mock_get_key.side_effect = ConfigurationError("API key not found")

        results = run_benchmark('google', 'truthfulness')

        assert 'error' in results
        assert "API key not found" in results['error']
        assert results['total_prompts'] == 0

    @patch('run_benchmarks.get_api_key')
    @patch('run_benchmarks.get_model_config')
    def test_run_benchmark_unknown_provider(self, mock_get_config, mock_get_key):
        """Test benchmark with unknown provider."""
        mock_get_key.return_value = 'test-key'
        mock_get_config.return_value = {}

        results = run_benchmark('unknown_provider', 'truthfulness')

        assert 'error' in results
        assert "Unknown provider" in results['error']

    @patch('run_benchmarks.get_api_key')
    @patch('run_benchmarks.get_model_config')
    def test_run_benchmark_unknown_dataset(self, mock_get_config, mock_get_key):
        """Test benchmark with unknown dataset."""
        mock_get_key.return_value = 'test-key'
        mock_get_config.return_value = {}

        with patch('run_benchmarks.GeminiProvider'):
            results = run_benchmark('google', 'unknown_dataset')

        assert 'error' in results
        assert "Unknown dataset" in results['error']

    @patch('run_benchmarks.get_api_key')
    @patch('run_benchmarks.get_model_config')
    def test_run_benchmark_provider_init_error(self, mock_get_config, mock_get_key):
        """Test benchmark with provider initialization error."""
        mock_get_key.return_value = 'test-key'
        mock_get_config.return_value = {}

        with patch.dict('run_benchmarks.PROVIDERS', {'google': Mock(side_effect=Exception("Provider init failed"))}):
            results = run_benchmark('google', 'truthfulness')

        assert 'error' in results
        assert "Provider initialization failed" in results['error']

    @patch('run_benchmarks.get_api_key')
    @patch('run_benchmarks.get_model_config')
    @patch('run_benchmarks.load_dataset')
    def test_run_benchmark_generation_error(self, mock_load_dataset, mock_get_config, mock_get_key):
        """Test benchmark with generation error."""
        mock_get_key.return_value = 'test-key'
        mock_get_config.return_value = {'max_retries': 1, 'retry_delay': 0.1}
        mock_load_dataset.return_value = [
            {
                'id': 'test1',
                'prompt': 'Test prompt',
                'evaluation_method': 'keyword_match',
                'expected_keywords': ['test']
            }
        ]

        mock_provider_class = Mock()
        mock_provider_instance = Mock()
        mock_provider_instance.generate.side_effect = Exception("Generation failed")
        mock_provider_class.return_value = mock_provider_instance

        with patch.dict('run_benchmarks.PROVIDERS', {'google': mock_provider_class}):
            with patch('benchmarks.truthfulness.validate_dataset'):
                results = run_benchmark('google', 'truthfulness')

        assert results['failed_evaluations'] == 1
        assert results['successful_evaluations'] == 0
        assert len(results['evaluations']) == 1
        # The error is stored in the evaluation result
        assert 'error' in results['evaluations'][0]
        assert "Generation failed" in results['evaluations'][0]['error']


class TestCLI:
    """Test CLI interface."""

    def test_cli_help(self):
        """Test CLI help output."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert '--provider' in result.output
        assert '--dataset' in result.output

    @patch('run_benchmarks.run_benchmark')
    def test_cli_basic_run(self, mock_run_benchmark):
        """Test basic CLI run."""
        mock_run_benchmark.return_value = {
            'provider': 'google',
            'dataset': 'truthfulness',
            'total_prompts': 10,
            'successful_evaluations': 8,
            'failed_evaluations': 2,
            'overall_score': 0.8,
            'total_duration_seconds': 5.0,
            'average_response_time_seconds': 0.5,
            'evaluations': []
        }

        runner = CliRunner()
        result = runner.invoke(main, ['--provider', 'google', '--dataset', 'truthfulness'])

        assert result.exit_code == 2  # Some evaluations failed
        assert 'LLM Lab Benchmark Runner' in result.output
        assert 'Overall Score: 80.00%' in result.output

    @patch('run_benchmarks.run_benchmark')
    def test_cli_with_csv_output(self, mock_run_benchmark):
        """Test CLI with CSV output."""
        mock_run_benchmark.return_value = {
            'provider': 'google',
            'dataset': 'truthfulness',
            'total_prompts': 1,
            'successful_evaluations': 1,
            'failed_evaluations': 0,
            'overall_score': 1.0,
            'evaluations': [{'test': 'data'}]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            result = runner.invoke(main, [
                '--provider', 'google',
                '--dataset', 'truthfulness',
                '--output-dir', tmpdir
            ])

            assert result.exit_code == 0
            assert 'Saving results to CSV' in result.output

    @patch('run_benchmarks.run_benchmark')
    def test_cli_no_csv_flag(self, mock_run_benchmark):
        """Test CLI with --no-csv flag."""
        mock_run_benchmark.return_value = {
            'provider': 'google',
            'dataset': 'truthfulness',
            'total_prompts': 1,
            'successful_evaluations': 1,
            'failed_evaluations': 0,
            'overall_score': 1.0,
            'evaluations': [{'test': 'data'}]
        }

        runner = CliRunner()
        result = runner.invoke(main, [
            '--provider', 'google',
            '--dataset', 'truthfulness',
            '--no-csv'
        ])

        assert result.exit_code == 0
        assert 'Saving results to CSV' not in result.output

    @patch('run_benchmarks.run_benchmark')
    def test_cli_keyboard_interrupt(self, mock_run_benchmark):
        """Test CLI handling keyboard interrupt."""
        mock_run_benchmark.side_effect = KeyboardInterrupt()

        runner = CliRunner()
        result = runner.invoke(main, ['--provider', 'google', '--dataset', 'truthfulness'])

        assert result.exit_code == 130
        assert 'interrupted by user' in result.output

