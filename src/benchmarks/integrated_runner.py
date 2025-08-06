"""
Integrated Benchmark Runner

This module provides an enhanced benchmark runner that automatically detects
local models and uses appropriate monitoring and evaluation methods.
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..evaluation.local_model_metrics import generate_local_model_report
from ..providers import get_provider_for_model
from ..providers.local import UnifiedLocalProvider
from .local_model_runner import LocalModelBenchmarkRunner

logger = logging.getLogger(__name__)


def is_local_model(model_name: str) -> bool:
    """
    Check if a model name indicates a local model.

    Args:
        model_name: Name of the model to check

    Returns:
        True if it's a local model, False otherwise
    """
    local_prefixes = [
        "local:",
        "transformers:",
        "llamacpp:",
        "ollama:",
    ]

    # Check for explicit local prefixes
    if any(model_name.startswith(prefix) for prefix in local_prefixes):
        return True

    # Check for known local model names
    local_model_names = [
        "pythia-70m",
        "pythia-160m",
        "smollm-135m",
        "smollm-360m",
        "qwen-0.5b",
        "llama3.2:1b",
        "llama3.2:3b",
    ]

    if any(name in model_name.lower() for name in local_model_names):
        return True

    return False


def run_integrated_benchmark(
    model_name: str,
    dataset_name: str,
    prompts: List[str],
    expected_answers: List[str],
    evaluation_method: str = "multi_method",
    enable_resource_monitoring: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    **generation_kwargs,
) -> Dict[str, Any]:
    """
    Run benchmark with automatic local model detection and appropriate handling.

    Args:
        model_name: Name of the model to benchmark
        dataset_name: Name of the benchmark dataset
        prompts: List of prompts to evaluate
        expected_answers: List of expected answers
        evaluation_method: Evaluation method to use
        enable_resource_monitoring: Enable resource monitoring for local models
        progress_callback: Optional progress callback
        **generation_kwargs: Additional generation parameters

    Returns:
        Dictionary with benchmark results and metadata
    """
    logger.info(f"Starting integrated benchmark for {model_name} on {dataset_name}")

    # Detect if this is a local model
    is_local = is_local_model(model_name)

    if is_local:
        logger.info(f"Detected local model: {model_name}")
        return run_local_model_benchmark(
            model_name=model_name,
            dataset_name=dataset_name,
            prompts=prompts,
            expected_answers=expected_answers,
            evaluation_method=evaluation_method,
            enable_resource_monitoring=enable_resource_monitoring,
            progress_callback=progress_callback,
            **generation_kwargs,
        )
    else:
        logger.info(f"Detected cloud model: {model_name}")
        return run_cloud_model_benchmark(
            model_name=model_name,
            dataset_name=dataset_name,
            prompts=prompts,
            expected_answers=expected_answers,
            evaluation_method=evaluation_method,
            progress_callback=progress_callback,
            **generation_kwargs,
        )


def run_local_model_benchmark(
    model_name: str,
    dataset_name: str,
    prompts: List[str],
    expected_answers: List[str],
    evaluation_method: str = "local_model_eval",
    enable_resource_monitoring: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    **generation_kwargs,
) -> Dict[str, Any]:
    """
    Run benchmark specifically for local models with full monitoring.

    Args:
        model_name: Local model name
        dataset_name: Benchmark dataset name
        prompts: List of prompts
        expected_answers: List of expected answers
        evaluation_method: Evaluation method
        enable_resource_monitoring: Enable resource monitoring
        progress_callback: Progress callback
        **generation_kwargs: Generation parameters

    Returns:
        Comprehensive benchmark results with local model metrics
    """
    start_time = datetime.now()

    try:
        # Get provider for the local model
        provider_class = get_provider_for_model(model_name)
        provider = provider_class(model_name)

        # Ensure it's a UnifiedLocalProvider
        if not isinstance(provider, UnifiedLocalProvider):
            raise ValueError(
                f"Expected UnifiedLocalProvider for {model_name}, got {type(provider)}"
            )

        # Initialize the provider
        provider.initialize()

        # Create local model runner
        runner = LocalModelBenchmarkRunner(
            enable_resource_monitoring=enable_resource_monitoring, monitoring_interval=0.5
        )

        # Run batch evaluation
        results = runner.run_batch_evaluation(
            provider=provider,
            prompts=prompts,
            expected_answers=expected_answers,
            dataset_name=dataset_name,
            evaluation_method=evaluation_method,
            progress_callback=progress_callback,
            **generation_kwargs,
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Generate comprehensive report
        report = generate_local_model_report(results)

        return {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "benchmark_type": "local_model",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "total_evaluations": len(results),
            "successful_evaluations": len([r for r in results if r.evaluation_score > 0]),
            "failed_evaluations": len([r for r in results if r.evaluation_score == 0]),
            "results": [
                {
                    "prompt": r.prompt,
                    "response": r.response,
                    "expected_answer": r.expected_answer,
                    "evaluation_score": r.evaluation_score,
                    "performance_metrics": {
                        "inference_time_ms": r.performance_metrics.inference_time_ms,
                        "tokens_per_second": r.performance_metrics.tokens_per_second,
                        "memory_used_mb": r.performance_metrics.memory_used_mb,
                        "gpu_utilization_percent": r.performance_metrics.gpu_utilization_percent,
                        "response_length": r.performance_metrics.response_length,
                    },
                    "backend_used": r.backend_used,
                    "model_format": r.model_format,
                    "gpu_accelerated": r.gpu_accelerated,
                }
                for r in results
            ],
            "report": report,
        }

    except Exception as e:
        logger.error(f"Error running local model benchmark: {e}")
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        return {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "benchmark_type": "local_model",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "error": str(e),
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": len(prompts),
        }


def run_cloud_model_benchmark(
    model_name: str,
    dataset_name: str,
    prompts: List[str],
    expected_answers: List[str],
    evaluation_method: str = "multi_method",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    **generation_kwargs,
) -> Dict[str, Any]:
    """
    Run benchmark for cloud models using existing evaluation methods.

    Args:
        model_name: Cloud model name
        dataset_name: Benchmark dataset name
        prompts: List of prompts
        expected_answers: List of expected answers
        evaluation_method: Evaluation method
        progress_callback: Progress callback
        **generation_kwargs: Generation parameters

    Returns:
        Standard benchmark results for cloud models
    """
    from ..evaluation.improved_evaluation import multi_method_evaluation

    start_time = datetime.now()

    try:
        # Get provider for the cloud model
        provider_class = get_provider_for_model(model_name)
        provider = provider_class(model_name)
        provider.initialize()

        results = []
        total_items = len(prompts)

        for i, (prompt, expected_answer) in enumerate(zip(prompts, expected_answers)):
            try:
                # Generate response
                response = provider.generate(prompt, **generation_kwargs)

                # Evaluate using standard methods
                evaluation_result = multi_method_evaluation(
                    response=response,
                    expected_keywords=[expected_answer] if expected_answer else [],
                    expected_answer=expected_answer,
                )

                results.append(
                    {
                        "prompt": prompt,
                        "response": response,
                        "expected_answer": expected_answer,
                        "evaluation_score": evaluation_result.get("overall_score", 0.0),
                        "evaluation_details": evaluation_result,
                    }
                )

                if progress_callback:
                    progress_callback(i + 1, total_items)

            except Exception as e:
                logger.error(f"Error in evaluation {i + 1}/{total_items}: {e}")
                results.append(
                    {
                        "prompt": prompt,
                        "response": f"[ERROR: {str(e)}]",
                        "expected_answer": expected_answer,
                        "evaluation_score": 0.0,
                        "error": str(e),
                    }
                )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        successful_count = len([r for r in results if r["evaluation_score"] > 0])

        return {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "benchmark_type": "cloud_model",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "total_evaluations": len(results),
            "successful_evaluations": successful_count,
            "failed_evaluations": len(results) - successful_count,
            "average_score": sum(r["evaluation_score"] for r in results) / len(results)
            if results
            else 0.0,
            "results": results,
        }

    except Exception as e:
        logger.error(f"Error running cloud model benchmark: {e}")
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        return {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "benchmark_type": "cloud_model",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "error": str(e),
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": len(prompts),
        }
