"""
Test suite for enhanced CLI features in run_benchmarks.py

This module tests the new multi-model support, argument parsing,
and backward compatibility features added to the benchmark runner.
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from click.testing import CliRunner
from run_benchmarks import main, run_benchmark

from llm_providers import registry


class TestCLIMultiModelSupport:
    """Test the enhanced CLI with multi-model support."""

    @pytest.fixture
    def runner(self):
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_registry(self):
        """Mock the provider registry."""
        with patch("run_benchmarks.registry") as mock_reg:
            # Mock available models
            mock_reg.list_all_models.return_value = [
                "gemini-1.5-flash",
                "gpt-4",
                "gpt-3.5-turbo",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
            ]
            yield mock_reg

    @pytest.fixture
    def mock_provider_lookup(self):
        """Mock the get_provider_for_model function."""
        with patch("run_benchmarks.get_provider_for_model") as mock_get:
            # Create mock provider classes
            mock_google = Mock()
            mock_google.__name__ = "GoogleProvider"

            mock_openai = Mock()
            mock_openai.__name__ = "OpenAIProvider"

            mock_anthropic = Mock()
            mock_anthropic.__name__ = "AnthropicProvider"

            # Map models to providers
            def get_provider(model_name):
                if "gemini" in model_name:
                    return mock_google
                elif "gpt" in model_name:
                    return mock_openai
                elif "claude" in model_name:
                    return mock_anthropic
                else:
                    raise ValueError(f"No provider found for model '{model_name}'")

            mock_get.side_effect = get_provider
            yield mock_get

    @pytest.fixture
    def mock_run_benchmark(self):
        """Mock the run_benchmark function."""
        with patch("run_benchmarks.run_benchmark") as mock_run:
            # Return successful results by default
            def create_result(model_name, dataset_name):
                return {
                    "model": model_name,
                    "dataset": dataset_name,
                    "total_prompts": 10,
                    "successful_evaluations": 8,
                    "failed_evaluations": 2,
                    "overall_score": 0.8,
                    "evaluations": [],
                }

            mock_run.side_effect = create_result
            yield mock_run

    def test_single_model_argument(
        self, runner, mock_registry, mock_provider_lookup, mock_run_benchmark
    ):
        """Test running with a single model using --model."""
        result = runner.invoke(main, ["--model", "gemini-1.5-flash", "--dataset", "truthfulness"])

        assert result.exit_code == 2  # Some evaluations failed
        assert "✓ gemini-1.5-flash -> GoogleProvider" in result.output
        assert "Model: gemini-1.5-flash" in result.output
        assert "Overall Score: 80.00%" in result.output

        # Verify run_benchmark was called once with correct model
        mock_run_benchmark.assert_called_once_with("gemini-1.5-flash", "truthfulness")

    def test_multiple_models_argument(
        self, runner, mock_registry, mock_provider_lookup, mock_run_benchmark
    ):
        """Test running with multiple models using --models."""
        # Mock different results for each model
        results = [
            {
                "model": "gpt-4",
                "provider": "openai",
                "dataset": "truthfulness",
                "total_prompts": 10,
                "successful_evaluations": 9,
                "failed_evaluations": 1,
                "overall_score": 0.9,
                "total_duration_seconds": 30.5,
                "evaluations": [],
            },
            {
                "model": "claude-3-opus-20240229",
                "provider": "anthropic",
                "dataset": "truthfulness",
                "total_prompts": 10,
                "successful_evaluations": 8,
                "failed_evaluations": 2,
                "overall_score": 0.8,
                "total_duration_seconds": 25.3,
                "evaluations": [],
            },
        ]
        mock_run_benchmark.side_effect = results

        result = runner.invoke(
            main, ["--models", "gpt-4,claude-3-opus-20240229", "--dataset", "truthfulness"]
        )

        assert result.exit_code == 2  # Some evaluations failed
        assert "✓ gpt-4 -> OpenAIProvider" in result.output
        assert "✓ claude-3-opus-20240229 -> AnthropicProvider" in result.output
        assert "Model Comparison:" in result.output
        assert "gpt-4" in result.output
        assert "90.00%" in result.output
        assert "claude-3-opus-20240229" in result.output
        assert "80.00%" in result.output

        # Verify run_benchmark was called for each model
        assert mock_run_benchmark.call_count == 2

    def test_all_models_flag(self, runner, mock_registry, mock_provider_lookup, mock_run_benchmark):
        """Test running with --all-models flag."""

        # Mock successful results
        def create_perfect_result(model_name, dataset_name):
            return {
                "model": model_name,
                "dataset": dataset_name,
                "total_prompts": 10,
                "successful_evaluations": 10,
                "failed_evaluations": 0,
                "overall_score": 1.0,
                "evaluations": [],
            }

        mock_run_benchmark.side_effect = create_perfect_result

        result = runner.invoke(main, ["--all-models", "--dataset", "truthfulness"])

        assert result.exit_code == 0  # All evaluations successful
        assert "Models Tested: 5" in result.output

        # Verify all models were validated
        assert "✓ gemini-1.5-flash -> GoogleProvider" in result.output
        assert "✓ gpt-4 -> OpenAIProvider" in result.output
        assert "✓ claude-3-opus-20240229 -> AnthropicProvider" in result.output

        # Verify run_benchmark was called for each model
        assert mock_run_benchmark.call_count == 5

    def test_parallel_execution(
        self, runner, mock_registry, mock_provider_lookup, mock_run_benchmark
    ):
        """Test parallel execution with --parallel flag."""

        # Mock results that include timing
        def create_perfect_result(model_name, dataset_name):
            return {
                "model": model_name,
                "dataset": dataset_name,
                "total_prompts": 10,
                "successful_evaluations": 10,
                "failed_evaluations": 0,
                "overall_score": 1.0,
                "total_duration_seconds": 10.0,
                "evaluations": [],
            }

        mock_run_benchmark.side_effect = create_perfect_result

        result = runner.invoke(
            main,
            ["--models", "gpt-4,claude-3-opus-20240229", "--dataset", "truthfulness", "--parallel"],
        )

        assert result.exit_code == 0
        assert "Running benchmarks in parallel" in result.output
        assert mock_run_benchmark.call_count == 2

    def test_backward_compatibility_provider(
        self, runner, mock_registry, mock_provider_lookup, mock_run_benchmark
    ):
        """Test backward compatibility with --provider argument."""
        result = runner.invoke(main, ["--provider", "google", "--dataset", "truthfulness"])

        assert result.exit_code == 2  # Some evaluations failed (default mock)
        assert "⚠️  Warning: --provider is deprecated" in result.output
        assert "Model: gemini-1.5-flash" in result.output

        # Verify it mapped to the correct model
        mock_run_benchmark.assert_called_once_with("gemini-1.5-flash", "truthfulness")

    def test_invalid_model_error(self, runner, mock_registry, mock_provider_lookup):
        """Test error handling for invalid model names."""
        result = runner.invoke(main, ["--model", "invalid-model-name", "--dataset", "truthfulness"])

        assert result.exit_code == 1
        assert "✗ invalid-model-name -> No provider found" in result.output
        assert "Invalid models: invalid-model-name" in result.output
        assert "Available models:" in result.output

    def test_no_model_specified_error(self, runner):
        """Test error when no model is specified."""
        result = runner.invoke(main, ["--dataset", "truthfulness"])

        assert result.exit_code == 1
        assert "❌ You must specify at least one model" in result.output

    def test_mixed_valid_invalid_models(
        self, runner, mock_registry, mock_provider_lookup, mock_run_benchmark
    ):
        """Test handling of mixed valid and invalid models."""
        result = runner.invoke(
            main,
            ["--models", "gpt-4,invalid-model,claude-3-opus-20240229", "--dataset", "truthfulness"],
        )

        assert result.exit_code == 1
        assert "✓ gpt-4 -> OpenAIProvider" in result.output
        assert "✗ invalid-model -> No provider found" in result.output
        assert "✓ claude-3-opus-20240229 -> AnthropicProvider" in result.output
        assert "Invalid models: invalid-model" in result.output

    def test_csv_output_multiple_models(
        self, runner, mock_registry, mock_provider_lookup, mock_run_benchmark
    ):
        """Test CSV output with multiple models."""

        # Add evaluations to the mock results
        def create_result_with_evaluations(model_name, dataset_name):
            return {
                "model": model_name,
                "dataset": dataset_name,
                "total_prompts": 10,
                "successful_evaluations": 10,
                "failed_evaluations": 0,
                "overall_score": 1.0,
                "evaluations": [{"prompt_id": "1", "success": True, "score": 1.0}],
            }

        mock_run_benchmark.side_effect = create_result_with_evaluations

        with patch("run_benchmarks.CSVResultLogger") as mock_logger_class:
            mock_logger = Mock()
            mock_logger.write_results.return_value = "/results/test.csv"
            mock_logger_class.return_value = mock_logger

            result = runner.invoke(
                main, ["--models", "gpt-4,claude-3-opus-20240229", "--dataset", "truthfulness"]
            )

            assert result.exit_code == 0
            assert "💾 Saving results to CSV" in result.output
            assert "Saved results for 2 model(s)" in result.output

            # Verify write_results was called for each model
            assert mock_logger.write_results.call_count == 2

    def test_model_comparison_sorting(
        self, runner, mock_registry, mock_provider_lookup, mock_run_benchmark
    ):
        """Test that model comparison table is sorted by score."""
        # Mock different scores for sorting test
        results = [
            {
                "model": "model-1",
                "provider": "provider-1",
                "overall_score": 0.5,
                "successful_evaluations": 5,
                "failed_evaluations": 5,
                "total_duration_seconds": 10.0,
                "dataset": "truthfulness",
                "total_prompts": 10,
                "evaluations": [],
            },
            {
                "model": "model-2",
                "provider": "provider-2",
                "overall_score": 0.9,
                "successful_evaluations": 9,
                "failed_evaluations": 1,
                "total_duration_seconds": 12.0,
                "dataset": "truthfulness",
                "total_prompts": 10,
                "evaluations": [],
            },
            {
                "model": "model-3",
                "provider": "provider-3",
                "overall_score": 0.7,
                "successful_evaluations": 7,
                "failed_evaluations": 3,
                "total_duration_seconds": 11.0,
                "dataset": "truthfulness",
                "total_prompts": 10,
                "evaluations": [],
            },
        ]
        mock_run_benchmark.side_effect = results

        result = runner.invoke(
            main, ["--models", "model-1,model-2,model-3", "--dataset", "truthfulness"]
        )

        # Extract the comparison table
        output_lines = result.output.split("\n")
        table_start = None
        for i, line in enumerate(output_lines):
            if "Model Comparison:" in line:
                table_start = i
                break

        assert table_start is not None

        if table_start is not None:
            # Check that models appear in descending score order
            table_content = "\n".join(output_lines[table_start:])
            model_2_pos = table_content.find("model-2")
            model_3_pos = table_content.find("model-3")
            model_1_pos = table_content.find("model-1")

            assert model_2_pos < model_3_pos < model_1_pos  # Sorted by score: 0.9, 0.7, 0.5
        else:
            # If parallel output, just check all models completed
            assert "model-1" in result.output
            assert "model-2" in result.output
            assert "model-3" in result.output

    def test_error_handling_in_parallel(
        self, runner, mock_registry, mock_provider_lookup, mock_run_benchmark
    ):
        """Test error handling during parallel execution."""

        # Mock one success and one failure
        def mock_benchmark(model_name, dataset):
            if model_name == "gpt-4":
                return {
                    "model": "gpt-4",
                    "dataset": dataset,
                    "error": "API key invalid",
                    "overall_score": 0.0,
                }
            else:
                return {
                    "model": model_name,
                    "dataset": dataset,
                    "total_prompts": 10,
                    "successful_evaluations": 10,
                    "failed_evaluations": 0,
                    "overall_score": 1.0,
                    "evaluations": [],
                }

        mock_run_benchmark.side_effect = mock_benchmark

        result = runner.invoke(
            main,
            ["--models", "gpt-4,claude-3-opus-20240229", "--dataset", "truthfulness", "--parallel"],
        )

        # Check for error handling
        if "✗ Failed: gpt-4" in result.output:
            assert result.exit_code == 2  # Some failures
            assert "✓ Completed: claude-3-opus-20240229" in result.output
        else:
            # Sequential execution shows errors differently
            assert "error" in result.output.lower() or "API key invalid" in result.output
