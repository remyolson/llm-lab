#!/usr/bin/env python3
"""
Demo script showing how to use the Multi-Model Benchmark Execution Engine

This demonstrates both programmatic usage and various execution modes.
"""

# Import paths fixed - sys.path manipulation removed
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import ExecutionMode, MultiModelBenchmarkRunner


def mock_benchmark_function(model_name: str, dataset_name: str) -> dict:
    """
    Mock benchmark function for demonstration.
    In real usage, this would call actual LLM providers.
    """
    import random
    import time

    print(f"  → Starting benchmark for {model_name} on {dataset_name}")

    # Simulate processing time
    time.sleep(random.uniform(1, 3))

    # Simulate different scenarios
    if "error" in model_name.lower():
        raise RuntimeError(f"Simulated error for {model_name}")

    if "slow" in model_name.lower():
        time.sleep(5)  # Will timeout if timeout is set

    # Generate mock results
    total = 100
    success_rate = random.uniform(0.7, 0.95)
    successful = int(total * success_rate)

    return {
        "total_prompts": total,
        "successful_evaluations": successful,
        "failed_evaluations": total - successful,
        "overall_score": success_rate,
        "evaluations": [
            {
                "prompt_id": f"test_{i}",
                "success": i < successful,
                "score": 1.0 if i < successful else 0.0,
            }
            for i in range(10)  # Just first 10 for demo
        ],
        "model_config": {"temperature": 0.7, "max_tokens": 1000},
    }


def progress_callback(model_name: str, current: int, total: int):
    """Progress callback for visual feedback."""
    percentage = (current / total * 100) if total > 0 else 0
    print(f"[{current}/{total}] {percentage:5.1f}% - Processing: {model_name}")


def demo_sequential_execution():
    """Demonstrate sequential execution mode."""
    print("\n" + "=" * 60)
    print("DEMO: Sequential Execution")
    print("=" * 60)

    runner = MultiModelBenchmarkRunner(
        benchmark_function=mock_benchmark_function,
        execution_mode=ExecutionMode.SEQUENTIAL,
        progress_callback=progress_callback,
    )

    models = ["gpt-4", "claude-3-opus-20240229", "gemini-1.5-flash"]
    result = runner.run(models, "demo_dataset")

    print(f"\nExecution completed in {result.total_duration_seconds:.2f} seconds")
    print(f"Successful models: {len(result.successful_models)}/{len(models)}")

    for model_result in result.model_results:
        print(f"  - {model_result.model_name}: Score = {model_result.overall_score:.2%}")


def demo_parallel_execution():
    """Demonstrate parallel execution mode."""
    print("\n" + "=" * 60)
    print("DEMO: Parallel Execution")
    print("=" * 60)

    runner = MultiModelBenchmarkRunner(
        benchmark_function=mock_benchmark_function,
        execution_mode=ExecutionMode.PARALLEL,
        max_workers=3,
        progress_callback=progress_callback,
    )

    models = [
        "gpt-4",
        "claude-3-opus-20240229",
        "gemini-1.5-flash",
        "gpt-3.5-turbo",
        "claude-3-haiku-20240307",
    ]
    result = runner.run(models, "demo_dataset")

    print(f"\nExecution completed in {result.total_duration_seconds:.2f} seconds")
    print(f"Successful models: {len(result.successful_models)}/{len(models)}")

    # Show summary statistics
    stats = result.summary_stats
    print("\nSummary Statistics:")
    print(f"  Average score: {stats['average_score']:.2%}")
    print(f"  Best model: {stats['best_model']} ({stats['best_score']:.2%})")
    print(f"  Worst model: {stats['worst_model']} ({stats['worst_score']:.2%})")


def demo_error_handling():
    """Demonstrate error handling and isolation."""
    print("\n" + "=" * 60)
    print("DEMO: Error Handling")
    print("=" * 60)

    runner = MultiModelBenchmarkRunner(
        benchmark_function=mock_benchmark_function,
        execution_mode=ExecutionMode.SEQUENTIAL,
        progress_callback=progress_callback,
    )

    # Mix of good and bad models
    models = ["gpt-4", "error-model", "claude-3-opus-20240229", "another-error", "gemini-1.5-flash"]
    result = runner.run(models, "demo_dataset")

    print("\nExecution completed with errors")
    print(f"Successful: {len(result.successful_models)}")
    print(f"Failed: {len(result.failed_models)}")

    if result.failed_models:
        print("\nFailed models:")
        for failed in result.failed_models:
            print(f"  - {failed.model_name}: {failed.error}")


def demo_timeout_handling():
    """Demonstrate timeout functionality."""
    print("\n" + "=" * 60)
    print("DEMO: Timeout Handling")
    print("=" * 60)

    runner = MultiModelBenchmarkRunner(
        benchmark_function=mock_benchmark_function,
        execution_mode=ExecutionMode.SEQUENTIAL,
        timeout_per_model=2,  # 2 second timeout
        progress_callback=progress_callback,
    )

    models = ["gpt-4", "slow-model", "claude-3-opus-20240229"]
    result = runner.run(models, "demo_dataset")

    print("\nExecution completed with timeouts")

    for model_result in result.model_results:
        if model_result.timed_out:
            print(f"  - {model_result.model_name}: TIMED OUT")
        elif model_result.success:
            print(f"  - {model_result.model_name}: Success ({model_result.overall_score:.2%})")
        else:
            print(f"  - {model_result.model_name}: Error - {model_result.error}")


def main():
    """Run all demonstrations."""
    print("Multi-Model Benchmark Execution Engine Demo")
    print("==========================================")

    # Run demonstrations
    demo_sequential_execution()
    demo_parallel_execution()
    demo_error_handling()
    demo_timeout_handling()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
