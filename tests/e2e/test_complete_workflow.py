"""
End-to-end tests for complete workflows.

These tests verify full user workflows from start to finish.
"""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.mark.e2e
class TestBenchmarkWorkflow:
    """Test complete benchmark workflow from configuration to results."""

    def test_single_provider_benchmark(
        self, mock_openai_provider, temp_dataset_file, temp_dir, performance_tracker
    ):
        """Test running a complete benchmark for a single provider."""
        # Step 1: Configure benchmark
        benchmark_config = {
            "provider": "openai",
            "dataset": str(temp_dataset_file),
            "output_dir": str(temp_dir),
            "timeout": 30,
            "retries": 3,
        }

        # Step 2: Load dataset
        with open(temp_dataset_file) as f:
            dataset = json.load(f)

        # Step 3: Run benchmark
        results = []
        with performance_tracker.measure("benchmark_execution"):
            for prompt_data in dataset["prompts"]:
                response = mock_openai_provider.generate(prompt_data["prompt"])

                # Simple evaluation
                success = prompt_data["expected"].lower() in response.lower()

                results.append(
                    {
                        "prompt_id": prompt_data["id"],
                        "success": success,
                        "response": response,
                        "expected": prompt_data["expected"],
                    }
                )

        # Step 4: Save results
        results_file = temp_dir / "benchmark_results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "config": benchmark_config,
                    "results": results,
                    "metrics": performance_tracker.get_summary(),
                },
                f,
                indent=2,
            )

        # Verify workflow completion
        assert results_file.exists()
        assert len(results) == len(dataset["prompts"])
        assert all("success" in r for r in results)

        # Verify performance
        summary = performance_tracker.get_summary()
        assert summary["count"] == 1
        assert summary["total"] < 5  # Should complete quickly with mocks

    @pytest.mark.slow
    def test_multi_provider_comparison(self, mock_all_providers, sample_prompts, temp_dir):
        """Test comparing multiple providers on the same dataset."""
        comparison_results = {}

        for provider_name, provider in mock_all_providers.items():
            provider_results = []

            for prompt in sample_prompts[:3]:  # Use subset for speed
                response = provider.generate(prompt)
                provider_results.append(
                    {"prompt": prompt, "response": response, "length": len(response)}
                )

            comparison_results[provider_name] = {
                "results": provider_results,
                "avg_length": sum(r["length"] for r in provider_results) / len(provider_results),
                "total_calls": provider.call_count,
            }

        # Save comparison
        comparison_file = temp_dir / "provider_comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison_results, f, indent=2)

        # Verify all providers were tested
        assert len(comparison_results) == 3
        assert all(provider in comparison_results for provider in ["openai", "anthropic", "google"])

        # Verify each provider processed all prompts
        for provider_name, results in comparison_results.items():
            assert len(results["results"]) == 3
            assert results["total_calls"] == 3


@pytest.mark.e2e
class TestConfigurationWorkflow:
    """Test configuration management workflow."""

    def test_config_load_validate_use(self, temp_config_file, temp_dir):
        """Test loading, validating, and using configuration."""
        # Step 1: Load configuration
        import yaml

        with open(temp_config_file) as f:
            config = yaml.safe_load(f)

        # Step 2: Validate configuration
        assert "providers" in config
        assert "benchmarks" in config

        for provider_name, provider_config in config["providers"].items():
            assert "api_key" in provider_config
            assert "model" in provider_config

        # Step 3: Use configuration to initialize providers
        from tests.conftest import MockLLMProvider

        providers = {}
        for provider_name, provider_config in config["providers"].items():
            providers[provider_name] = MockLLMProvider(
                name=provider_name, model=provider_config["model"]
            )

        # Step 4: Run a simple test with configured providers
        test_prompt = "Config test"
        results = {}

        for name, provider in providers.items():
            results[name] = provider.generate(test_prompt)

        # Verify workflow
        assert len(providers) == 2  # OpenAI and Anthropic in sample config
        assert all(isinstance(r, str) for r in results.values())

    def test_environment_override(self, temp_config_file, mock_env):
        """Test that environment variables override config file."""
        import yaml

        # Load base config
        with open(temp_config_file) as f:
            config = yaml.safe_load(f)

        # Environment should override
        original_key = config["providers"]["openai"]["api_key"]
        env_key = mock_env.get("OPENAI_API_KEY")

        # In real implementation, env would override config
        final_key = env_key if env_key else original_key

        assert final_key == "mock-openai-key"  # From mock_env
        assert final_key != original_key  # Overridden


@pytest.mark.e2e
@pytest.mark.slow
class TestPerformanceRegressionWorkflow:
    """Test performance regression detection workflow."""

    def test_performance_regression_detection(
        self, mock_openai_provider, performance_tracker, temp_dir
    ):
        """Test detecting performance regressions between runs."""
        # Run 1: Baseline performance
        baseline_times = []
        for i in range(5):
            with performance_tracker.measure(f"baseline_{i}"):
                mock_openai_provider.generate(f"Test {i}")
            baseline_times.append(performance_tracker.metrics[-1]["duration"])

        baseline_avg = sum(baseline_times) / len(baseline_times)

        # Simulate performance degradation
        mock_openai_provider.response_delay = 0.5  # Make slower

        # Run 2: Current performance
        current_times = []
        for i in range(5):
            with performance_tracker.measure(f"current_{i}"):
                mock_openai_provider.generate(f"Test {i}")
            current_times.append(performance_tracker.metrics[-1]["duration"])

        current_avg = sum(current_times) / len(current_times)

        # Detect regression
        regression_threshold = 1.5  # 50% slower is a regression
        has_regression = current_avg > baseline_avg * regression_threshold

        # Save regression report
        report = {
            "baseline": {"times": baseline_times, "average": baseline_avg},
            "current": {"times": current_times, "average": current_avg},
            "regression_detected": has_regression,
            "slowdown_factor": current_avg / baseline_avg,
        }

        report_file = temp_dir / "regression_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Verify regression was detected
        assert has_regression
        assert report["slowdown_factor"] > regression_threshold
        assert report_file.exists()


@pytest.mark.e2e
class TestErrorRecoveryWorkflow:
    """Test error handling and recovery workflows."""

    def test_provider_failure_recovery(
        self, mock_openai_provider, mock_anthropic_provider, sample_prompts
    ):
        """Test recovery from provider failures."""
        results = []
        errors = []

        for i, prompt in enumerate(sample_prompts[:5]):
            try:
                # Simulate intermittent failures
                if i == 2:
                    raise Exception("Provider temporarily unavailable")

                response = mock_openai_provider.generate(prompt)
                results.append({"prompt": prompt, "response": response, "provider": "openai"})

            except Exception as e:
                # Fallback to alternative provider
                errors.append({"prompt": prompt, "error": str(e), "provider": "openai"})

                try:
                    response = mock_anthropic_provider.generate(prompt)
                    results.append(
                        {"prompt": prompt, "response": response, "provider": "anthropic"}
                    )
                except Exception as fallback_error:
                    errors.append(
                        {"prompt": prompt, "error": str(fallback_error), "provider": "anthropic"}
                    )

        # Verify recovery
        assert len(results) == 5  # All prompts processed
        assert len(errors) == 1  # One error recorded
        assert sum(1 for r in results if r["provider"] == "anthropic") == 1  # One fallback

    def test_data_validation_workflow(self, temp_dataset_file):
        """Test data validation and error reporting."""
        # Load dataset
        with open(temp_dataset_file) as f:
            dataset = json.load(f)

        # Add invalid entries
        dataset["prompts"].append(
            {
                "id": "invalid_001",
                # Missing required fields
            }
        )
        dataset["prompts"].append(
            {
                "id": "invalid_002",
                "prompt": "",  # Empty prompt
                "expected": "result",
            }
        )

        # Validate dataset
        validation_errors = []
        valid_prompts = []

        for prompt_data in dataset["prompts"]:
            errors = []

            # Check required fields
            if "prompt" not in prompt_data:
                errors.append("Missing 'prompt' field")
            elif not prompt_data["prompt"]:
                errors.append("Empty prompt")

            if "expected" not in prompt_data:
                errors.append("Missing 'expected' field")

            if errors:
                validation_errors.append({"id": prompt_data.get("id", "unknown"), "errors": errors})
            else:
                valid_prompts.append(prompt_data)

        # Verify validation
        assert len(validation_errors) == 2
        assert len(valid_prompts) == 3  # Original 3 valid prompts
        assert validation_errors[0]["id"] == "invalid_001"
        assert "Missing 'prompt' field" in validation_errors[0]["errors"]
