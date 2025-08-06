#!/usr/bin/env python3
"""
Integration Test for Local Model Benchmark System

This script tests the complete local model benchmark system end-to-end,
including provider integration, resource monitoring, and evaluation metrics.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def test_local_model_benchmark_integration():
    """Test the complete local model benchmark integration."""
    logger.info("Starting local model benchmark integration test...")

    try:
        # Import the integrated benchmark runner
        from benchmarks.integrated_runner import is_local_model, run_integrated_benchmark

        # Test local model detection
        logger.info("Testing local model detection...")

        test_cases = [
            ("local:pythia-70m", True),
            ("transformers:gpt2", True),
            ("llamacpp:llama-7b", True),
            ("ollama:mistral", True),
            ("pythia-70m", True),
            ("smollm-135m", True),
            ("gpt-4", False),
            ("claude-3-sonnet", False),
            ("gemini-1.5-flash", False),
        ]

        for model_name, expected in test_cases:
            result = is_local_model(model_name)
            status = "✓" if result == expected else "✗"
            logger.info(f"{status} {model_name}: {result} (expected {expected})")

            if result != expected:
                logger.error(f"Local model detection failed for {model_name}")
                return False

        logger.info("✓ Local model detection tests passed")

        # Test configuration system
        logger.info("Testing local model configuration...")

        from config import get_local_model_config

        config = get_local_model_config()
        required_keys = [
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "gpu_layers",
            "context_length",
            "batch_size",
            "threads",
            "memory_threshold",
            "vram_threshold",
        ]

        for key in required_keys:
            if key not in config:
                logger.error(f"Missing configuration key: {key}")
                return False

        logger.info(f"✓ Local model configuration loaded with {len(config)} parameters")

        # Test evaluation metrics system
        logger.info("Testing local model evaluation metrics...")

        from evaluation.local_model_metrics import (
            LocalModelEvaluator,
            LocalModelPerformanceMetrics,
            evaluate_local_model_response,
        )

        evaluator = LocalModelEvaluator()

        # Test quality evaluation
        test_response = "This is a test response that demonstrates coherent text generation."
        test_prompt = "Generate a test response."
        test_expected = "test response"

        quality_metrics = evaluator.evaluate_response_quality(
            response=test_response, expected_answer=test_expected, prompt=test_prompt
        )

        expected_quality_keys = [
            "response_length",
            "response_word_count",
            "repetition_score",
            "coherence_score",
            "prompt_leakage_score",
            "token_efficiency",
        ]

        for key in expected_quality_keys:
            if key not in quality_metrics:
                logger.error(f"Missing quality metric: {key}")
                return False

        logger.info(f"✓ Quality evaluation produced {len(quality_metrics)} metrics")

        # Test performance metrics creation
        perf_metrics = LocalModelPerformanceMetrics(
            inference_time_ms=500.0,
            tokens_per_second=20.0,
            memory_used_mb=1024.0,
            memory_peak_mb=1200.0,
            gpu_memory_used_mb=512.0,
            gpu_utilization_percent=75.0,
            cpu_utilization_percent=30.0,
            model_size_mb=150.0,
            prompt_tokens=10,
            completion_tokens=15,
        )

        logger.info("✓ Performance metrics created successfully")

        # Test comprehensive evaluation
        result = evaluate_local_model_response(
            model_name="test-model",
            dataset_name="test-dataset",
            prompt=test_prompt,
            response=test_response,
            expected_answer=test_expected,
            performance_metrics=perf_metrics,
            backend_used="test-backend",
            model_format="test-format",
        )

        if not hasattr(result, "model_name") or result.model_name != "test-model":
            logger.error("Evaluation result missing or incorrect model_name")
            return False

        if not hasattr(result, "performance_metrics"):
            logger.error("Evaluation result missing performance_metrics")
            return False

        logger.info("✓ Comprehensive evaluation completed successfully")

        # Test resource monitoring components
        logger.info("Testing resource monitoring components...")

        from benchmarks.local_model_runner import ResourceMonitor
        from providers.local.resource_manager import ResourceManager

        resource_manager = ResourceManager()
        monitor = ResourceMonitor(resource_manager, sampling_interval=0.1)

        # Test basic functionality (without actual monitoring)
        if not hasattr(monitor, "start_monitoring"):
            logger.error("ResourceMonitor missing start_monitoring method")
            return False

        if not hasattr(monitor, "stop_monitoring"):
            logger.error("ResourceMonitor missing stop_monitoring method")
            return False

        logger.info("✓ Resource monitoring components available")

        # Test benchmark runner factory
        logger.info("Testing benchmark runner factory...")

        from benchmarks import create_local_model_runner

        runner = create_local_model_runner(enable_monitoring=True, monitoring_interval=0.5)

        if not hasattr(runner, "run_single_evaluation"):
            logger.error("LocalModelBenchmarkRunner missing run_single_evaluation method")
            return False

        if not hasattr(runner, "run_batch_evaluation"):
            logger.error("LocalModelBenchmarkRunner missing run_batch_evaluation method")
            return False

        logger.info("✓ Benchmark runner factory works correctly")

        logger.info("🎉 All integration tests passed! Local model benchmark system is ready.")
        return True

    except ImportError as e:
        logger.error(f"Import error during integration test: {e}")
        logger.info("This might be due to missing dependencies or import issues.")
        return False

    except Exception as e:
        logger.error(f"Integration test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run integration test."""
    logger.info("=" * 60)
    logger.info("Local Model Benchmark Integration Test")
    logger.info("=" * 60)

    success = test_local_model_benchmark_integration()

    logger.info("\n" + "=" * 60)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("=" * 60)

    if success:
        logger.info("✅ Integration test PASSED")
        logger.info("\nThe local model benchmark system is fully integrated and ready to use!")
        logger.info("\nNext steps:")
        logger.info("1. Test with actual local models")
        logger.info("2. Run benchmarks with different datasets")
        logger.info("3. Monitor performance and resource usage")
        return True
    else:
        logger.error("❌ Integration test FAILED")
        logger.info("\nSome components are not working correctly.")
        logger.info("Check the error messages above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
