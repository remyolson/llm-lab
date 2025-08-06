#!/usr/bin/env python3
"""
LLM Lab Quick Start Example

This example demonstrates the most common use case: comparing multiple LLMs
on a benchmark dataset and generating a report.

Prerequisites:
    - At least one API key configured in .env or environment variables
    - Python 3.9+
    - Required packages installed (pip install -r requirements.txt)

Usage:
    python examples/quick_start.py

    # Use specific providers
    python examples/quick_start.py --providers openai,anthropic

    # Use specific dataset
    python examples/quick_start.py --dataset gsm8k

    # Generate HTML report
    python examples/quick_start.py --report html
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from logging import ResultsLogger

from analysis import ResultsComparator
from config import ConfigManager
from evaluation import TruthfulnessEvaluator
from providers import ProviderRegistry


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def quick_benchmark(
    providers: List[str],
    dataset: str = "truthfulqa",
    sample_size: int = 10,
    report_format: str = "console",
) -> Dict[str, Any]:
    """
    Run a quick benchmark across specified providers.

    Args:
        providers: List of provider names to test
        dataset: Dataset to use for benchmarking
        sample_size: Number of samples to test
        report_format: Output format (console, json, html)

    Returns:
        Dictionary containing results and analysis
    """
    print_header("LLM Lab Quick Start Benchmark")

    # Initialize components
    config = ConfigManager()
    registry = ProviderRegistry()
    evaluator = TruthfulnessEvaluator()
    comparator = ResultsComparator()
    logger = ResultsLogger()

    # Step 1: Load providers
    print("1. Initializing providers...")
    active_providers = {}

    for provider_name in providers:
        try:
            # Get default model for provider
            models = {
                "openai": "gpt-3.5-turbo",
                "anthropic": "claude-3-haiku-20240307",
                "google": "gemini-1.5-flash",
            }

            model = models.get(provider_name, "default")
            provider = registry.create_provider(provider_name, model=model)
            active_providers[provider_name] = provider
            print(f"   ✓ {provider_name} ({model})")
        except Exception as e:
            print(f"   ✗ {provider_name}: {e!s}")

    if not active_providers:
        print("\n❌ No providers could be initialized. Please check your API keys.")
        return {}

    # Step 2: Load dataset
    print(f"\n2. Loading dataset: {dataset}")
    from evaluation.datasets import DatasetLoader

    loader = DatasetLoader()

    try:
        test_data = loader.load(dataset, split="test", sample_size=sample_size)
        print(f"   ✓ Loaded {len(test_data)} samples")
    except Exception as e:
        print(f"   ✗ Failed to load dataset: {e}")
        return {}

    # Step 3: Run evaluations
    print("\n3. Running evaluations...")
    results = {}

    for provider_name, provider in active_providers.items():
        print(f"\n   Testing {provider_name}:")
        provider_results = []

        for i, sample in enumerate(test_data):
            try:
                # Generate response
                response = provider.generate(
                    prompt=sample["prompt"],
                    max_tokens=200,
                    temperature=0,  # Deterministic for benchmarking
                )

                # Evaluate response
                eval_result = evaluator.evaluate(
                    response=response["text"],
                    expected=sample.get("expected", sample.get("answer", "")),
                )

                provider_results.append(
                    {
                        "prompt": sample["prompt"],
                        "response": response["text"],
                        "score": eval_result["accuracy"],
                        "latency": response.get("latency", 0),
                        "tokens": response.get("usage", {}).get("total_tokens", 0),
                    }
                )

                print(f"      Sample {i + 1}/{len(test_data)}: Score={eval_result['accuracy']:.2f}")

            except Exception as e:
                print(f"      Sample {i + 1}/{len(test_data)}: Error - {e!s}")
                provider_results.append(
                    {"prompt": sample["prompt"], "response": "", "score": 0.0, "error": str(e)}
                )

        results[provider_name] = provider_results

    # Step 4: Analyze results
    print("\n4. Analyzing results...")
    analysis = comparator.compare(results)

    # Step 5: Generate report
    print("\n5. Generating report...")

    if report_format == "console":
        print_console_report(results, analysis)
    elif report_format == "json":
        save_json_report(results, analysis)
    elif report_format == "html":
        save_html_report(results, analysis)

    # Step 6: Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/quick_start_{dataset}_{timestamp}.json"

    os.makedirs("results", exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(
            {
                "metadata": {
                    "dataset": dataset,
                    "sample_size": sample_size,
                    "providers": list(active_providers.keys()),
                    "timestamp": timestamp,
                },
                "results": results,
                "analysis": analysis,
            },
            f,
            indent=2,
        )

    print(f"\n   ✓ Results saved to: {results_file}")

    return {"results": results, "analysis": analysis}


def print_console_report(results: Dict, analysis: Dict) -> None:
    """Print a formatted console report."""
    print_header("Benchmark Results")

    # Summary statistics
    for provider, provider_results in results.items():
        scores = [r["score"] for r in provider_results if "error" not in r]
        latencies = [r["latency"] for r in provider_results if "latency" in r]

        if scores:
            avg_score = sum(scores) / len(scores)
            avg_latency = sum(latencies) / len(latencies) if latencies else 0

            print(f"{provider}:")
            print(f"  Average Score: {avg_score:.2%}")
            print(f"  Average Latency: {avg_latency:.2f}s")
            print(f"  Success Rate: {len(scores)}/{len(provider_results)}")
            print()


def save_json_report(results: Dict, analysis: Dict) -> None:
    """Save results as JSON report."""
    report_file = "results/quick_start_report.json"
    with open(report_file, "w") as f:
        json.dump({"results": results, "analysis": analysis}, f, indent=2)
    print(f"   ✓ JSON report saved to: {report_file}")


def save_html_report(results: Dict, analysis: Dict) -> None:
    """Save results as HTML report."""
    # Simple HTML report generation
    html = """
    <html>
    <head>
        <title>LLM Lab Quick Start Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            .score { font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>LLM Lab Benchmark Results</h1>
        <h2>Provider Comparison</h2>
        <table>
            <tr>
                <th>Provider</th>
                <th>Average Score</th>
                <th>Average Latency</th>
                <th>Success Rate</th>
            </tr>
    """

    for provider, provider_results in results.items():
        scores = [r["score"] for r in provider_results if "error" not in r]
        latencies = [r["latency"] for r in provider_results if "latency" in r]

        if scores:
            avg_score = sum(scores) / len(scores)
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            success_rate = f"{len(scores)}/{len(provider_results)}"

            html += f"""
            <tr>
                <td>{provider}</td>
                <td class="score">{avg_score:.2%}</td>
                <td>{avg_latency:.2f}s</td>
                <td>{success_rate}</td>
            </tr>
            """

    html += """
        </table>
    </body>
    </html>
    """

    report_file = "results/quick_start_report.html"
    with open(report_file, "w") as f:
        f.write(html)
    print(f"   ✓ HTML report saved to: {report_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LLM Lab Quick Start - Compare LLMs on benchmarks")
    parser.add_argument(
        "--providers",
        type=str,
        default="openai,anthropic,google",
        help="Comma-separated list of providers to test",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="truthfulqa",
        choices=["truthfulqa", "gsm8k", "mmlu", "humaneval"],
        help="Dataset to use for benchmarking",
    )
    parser.add_argument(
        "--samples", type=int, default=10, help="Number of samples to test (default: 10)"
    )
    parser.add_argument(
        "--report",
        type=str,
        default="console",
        choices=["console", "json", "html"],
        help="Report format (default: console)",
    )

    args = parser.parse_args()

    # Check for API keys
    providers_to_test = [p.strip() for p in args.providers.split(",")]
    available_providers = []

    for provider in providers_to_test:
        env_key = f"{provider.upper()}_API_KEY"
        if os.getenv(env_key):
            available_providers.append(provider)

    if not available_providers:
        print("❌ No API keys found in environment variables.")
        print("\nPlease set at least one of:")
        print("  - OPENAI_API_KEY")
        print("  - ANTHROPIC_API_KEY")
        print("  - GOOGLE_API_KEY")
        print("\nExample:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)

    # Run benchmark
    try:
        quick_benchmark(
            providers=available_providers,
            dataset=args.dataset,
            sample_size=args.samples,
            report_format=args.report,
        )
        print("\n✅ Quick start benchmark completed!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
