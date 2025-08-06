"""Tests for the ResultsComparator class."""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from analysis.comparator import ComparisonResult, ModelResult, ResultsComparator


class TestModelResult:
    """Test the ModelResult dataclass."""

    def test_from_json(self, tmp_path):
        """Test loading ModelResult from JSON file."""
        # Create test JSON data
        test_data = {
            "model": "gpt-4",
            "provider": "openai",
            "dataset": "truthfulness",
            "start_time": "2024-01-01T12:00:00",
            "total_prompts": 10,
            "successful_evaluations": 8,
            "failed_evaluations": 2,
            "overall_score": 0.8,
            "average_response_time_seconds": 1.5,
            "evaluations": [{"prompt_id": "test_1", "success": True, "response_time_seconds": 1.2}],
            "model_config": {"temperature": 0.7},
        }

        # Write to file
        json_path = tmp_path / "test_result.json"
        with open(json_path, "w") as f:
            json.dump(test_data, f)

        # Load and verify
        result = ModelResult.from_json(json_path)
        assert result.model_name == "gpt-4"
        assert result.provider == "openai"
        assert result.dataset == "truthfulness"
        assert result.total_prompts == 10
        assert result.successful_evaluations == 8
        assert result.overall_score == 0.8
        assert len(result.evaluations) == 1

    def test_from_csv(self, tmp_path):
        """Test loading ModelResult from CSV file."""
        import pandas as pd

        # Create test CSV data
        df_data = {
            "model_name": ["gpt-4", "gpt-4"],
            "provider": ["openai", "openai"],
            "benchmark_name": ["truthfulness", "truthfulness"],
            "timestamp": ["2024-01-01T12:00:00", "2024-01-01T12:00:01"],
            "success": [True, False],
            "response_time_seconds": [1.2, 2.3],
            "prompt_id": ["test_1", "test_2"],
        }
        df = pd.DataFrame(df_data)

        # Write to file
        csv_path = tmp_path / "test_result.csv"
        df.to_csv(csv_path, index=False)

        # Load and verify
        result = ModelResult.from_csv(csv_path)
        assert result.model_name == "gpt-4"
        assert result.provider == "openai"
        assert result.dataset == "truthfulness"
        assert result.total_prompts == 2
        assert result.successful_evaluations == 1
        assert result.failed_evaluations == 1
        assert result.overall_score == 0.5
        assert result.average_response_time == pytest.approx(1.75, 0.01)


class TestResultsComparator:
    """Test the ResultsComparator class."""

    @pytest.fixture
    def temp_results_dir(self):
        """Create a temporary results directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_results(self, temp_results_dir):
        """Create sample result files."""
        # Model 1 results
        model1_data = {
            "model": "gpt-4",
            "provider": "openai",
            "dataset": "truthfulness",
            "start_time": "2024-01-01T12:00:00",
            "total_prompts": 5,
            "successful_evaluations": 4,
            "failed_evaluations": 1,
            "overall_score": 0.8,
            "average_response_time_seconds": 1.5,
            "evaluations": [
                {
                    "prompt_id": f"q{i}",
                    "success": i != 4,
                    "response_time_seconds": 1.5,
                    "matched_keywords": ["yes"] if i != 4 else [],
                    "prompt": f"Question {i}",
                }
                for i in range(5)
            ],
        }

        # Model 2 results
        model2_data = {
            "model": "claude-3",
            "provider": "anthropic",
            "dataset": "truthfulness",
            "start_time": "2024-01-01T12:10:00",
            "total_prompts": 5,
            "successful_evaluations": 3,
            "failed_evaluations": 2,
            "overall_score": 0.6,
            "average_response_time_seconds": 2.0,
            "evaluations": [
                {
                    "prompt_id": f"q{i}",
                    "success": i < 3,
                    "response_time_seconds": 2.0,
                    "matched_keywords": ["yes"] if i < 3 else [],
                    "prompt": f"Question {i}",
                }
                for i in range(5)
            ],
        }

        # Write files
        with open(Path(temp_results_dir) / "gpt4_results.json", "w") as f:
            json.dump(model1_data, f)

        with open(Path(temp_results_dir) / "claude3_results.json", "w") as f:
            json.dump(model2_data, f)

        return temp_results_dir

    def test_load_results(self, sample_results):
        """Test loading results from files."""
        comparator = ResultsComparator(sample_results)

        # Load all results
        results = comparator.load_results()
        assert len(results) == 2
        assert "gpt-4" in results
        assert "claude-3" in results

        # Load specific model
        comparator2 = ResultsComparator(sample_results)
        results2 = comparator2.load_results(model_names=["gpt-4"])
        assert len(results2) == 1
        assert "gpt-4" in results2

    def test_align_results_by_prompt(self, sample_results):
        """Test aligning results by prompt ID."""
        comparator = ResultsComparator(sample_results)
        comparator.load_results()

        aligned = comparator.align_results_by_prompt()

        # Should have 5 aligned prompts (q0-q4)
        assert len(aligned) == 5

        # Each prompt should have results from both models
        for prompt_id, model_evals in aligned.items():
            assert len(model_evals) == 2
            assert "gpt-4" in model_evals
            assert "claude-3" in model_evals

    def test_calculate_metrics(self, sample_results):
        """Test metric calculation."""
        comparator = ResultsComparator(sample_results)
        comparator.load_results()

        metrics = comparator.calculate_metrics()

        # Check metric structure
        assert "overall_accuracy" in metrics
        assert "average_response_time" in metrics
        assert "consistency_score" in metrics

        # Verify values
        assert metrics["overall_accuracy"]["gpt-4"] == 0.8
        assert metrics["overall_accuracy"]["claude-3"] == 0.6
        assert metrics["average_response_time"]["gpt-4"] == 1.5
        assert metrics["average_response_time"]["claude-3"] == 2.0

    def test_rank_models(self, sample_results):
        """Test model ranking."""
        comparator = ResultsComparator(sample_results)
        comparator.load_results()
        metrics = comparator.calculate_metrics()

        rankings = comparator.rank_models(metrics)

        # GPT-4 should rank first in accuracy
        assert rankings["overall_accuracy"][0] == "gpt-4"
        assert rankings["overall_accuracy"][1] == "claude-3"

        # GPT-4 should also be faster
        assert rankings["average_response_time"][0] == "gpt-4"

    def test_statistical_significance(self, sample_results):
        """Test statistical significance calculation."""
        comparator = ResultsComparator(sample_results)
        comparator.load_results()

        # Test chi-square for success rates
        p_val, is_sig = comparator.calculate_statistical_significance(
            "gpt-4", "claude-3", "success"
        )

        assert isinstance(p_val, float)
        assert isinstance(is_sig, bool)
        assert 0 <= p_val <= 1

    def test_confidence_intervals(self, sample_results):
        """Test confidence interval calculation."""
        comparator = ResultsComparator(sample_results)
        comparator.load_results()

        # Test accuracy CI
        lower, upper = comparator.calculate_confidence_intervals("gpt-4", "overall_accuracy")

        assert lower < 0.8 < upper  # Point estimate should be within CI
        assert 0 <= lower <= upper <= 1

        # Test response time CI
        lower_t, upper_t = comparator.calculate_confidence_intervals(
            "gpt-4", "average_response_time"
        )

        assert lower_t <= 1.5 <= upper_t

    def test_compare_full_workflow(self, sample_results):
        """Test the complete comparison workflow."""
        comparator = ResultsComparator(sample_results)

        result = comparator.compare()

        assert isinstance(result, ComparisonResult)
        assert len(result.models) == 2
        assert result.dataset == "truthfulness"
        assert len(result.prompt_alignment) == 5
        assert "overall_accuracy" in result.metrics
        assert len(result.rankings) > 0
        assert len(result.statistical_tests) > 0

    def test_generate_comparison_csv(self, sample_results, tmp_path):
        """Test CSV generation."""
        comparator = ResultsComparator(sample_results)
        comparator.compare()

        csv_path = tmp_path / "comparison.csv"
        comparator.generate_comparison_csv(csv_path)

        assert csv_path.exists()

        # Load and verify CSV
        import pandas as pd

        df = pd.read_csv(csv_path)

        assert len(df) == 5  # 5 prompts
        assert "prompt_id" in df.columns
        assert "gpt-4_success" in df.columns
        assert "claude-3_success" in df.columns

    def test_generate_metrics_csv(self, sample_results, tmp_path):
        """Test metrics CSV generation."""
        comparator = ResultsComparator(sample_results)
        comparator.compare()

        csv_path = tmp_path / "metrics.csv"
        comparator.generate_metrics_csv(csv_path)

        assert csv_path.exists()

        # Load and verify
        import pandas as pd

        df = pd.read_csv(csv_path)

        assert len(df) == 2  # 2 models
        assert "model" in df.columns
        assert "overall_accuracy" in df.columns
        assert "average_response_time" in df.columns

    def test_generate_markdown_report(self, sample_results, tmp_path):
        """Test markdown report generation."""
        comparator = ResultsComparator(sample_results)
        comparator.compare()

        report_path = tmp_path / "report.md"
        comparator.generate_markdown_report(report_path)

        assert report_path.exists()

        # Verify content
        content = report_path.read_text()
        assert "# LLM Benchmark Comparison Report" in content
        assert "gpt-4" in content
        assert "claude-3" in content
        assert "## Executive Summary" in content
        assert "## Model Performance Overview" in content

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_generate_visualizations(self, mock_close, mock_savefig, sample_results, tmp_path):
        """Test visualization generation."""
        comparator = ResultsComparator(sample_results)
        comparator.compare()

        viz_dir = tmp_path / "viz"
        files = comparator.generate_visualizations(viz_dir)

        # Should generate multiple charts
        assert len(files) > 0
        assert mock_savefig.called
        assert mock_close.called

    def test_empty_results_handling(self, temp_results_dir):
        """Test handling of empty results directory."""
        comparator = ResultsComparator(temp_results_dir)
        results = comparator.load_results()

        assert len(results) == 0

        # Compare should raise error with no results
        with pytest.raises(ValueError):
            comparator.compare()

    def test_single_model_error(self, temp_results_dir):
        """Test that comparison requires at least 2 models."""
        # Create single model result
        model_data = {"model": "gpt-4", "provider": "openai", "dataset": "test", "evaluations": []}

        with open(Path(temp_results_dir) / "single.json", "w") as f:
            json.dump(model_data, f)

        comparator = ResultsComparator(temp_results_dir)
        comparator.load_results()

        with pytest.raises(ValueError, match="At least 2 models"):
            comparator.compare()

    def test_cohens_h_interpretation(self, sample_results):
        """Test Cohen's h effect size interpretation."""
        comparator = ResultsComparator(sample_results)

        assert comparator._interpret_cohens_h(0.1) == "small"
        assert comparator._interpret_cohens_h(0.3) == "medium"
        assert comparator._interpret_cohens_h(0.6) == "large"
        assert comparator._interpret_cohens_h(0.9) == "very large"
