#!/usr/bin/env python3
"""
Cost-Per-Quality Metrics Calculator

This module provides formulas and functions to calculate cost-per-quality metrics
across different LLM providers, helping users make data-driven decisions based on
both performance and cost considerations.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class BenchmarkResult:
    """Represents benchmark results for a model."""

    provider: str
    model: str
    accuracy: float  # 0-100
    latency_ms: float  # milliseconds
    tokens_per_second: float
    task_completion_rate: float  # 0-1
    user_satisfaction_score: float  # 1-5


@dataclass
class CostData:
    """Cost information for a model."""

    provider: str
    model: str
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float
    avg_tokens_per_request: int


@dataclass
class QualityMetrics:
    """Calculated quality metrics with costs."""

    provider: str
    model: str
    cost_per_request: float
    cost_per_accuracy_point: float
    cost_per_satisfaction_point: float
    cost_per_successful_completion: float
    latency_adjusted_cost: float
    roi_score: float
    weighted_efficiency_score: float


class CostQualityAnalyzer:
    """Analyzes cost vs quality trade-offs for LLM providers."""

    def __init__(self):
        # Default weights for different metrics
        self.default_weights = {
            "accuracy": 0.30,
            "latency": 0.20,
            "satisfaction": 0.25,
            "completion": 0.25,
        }

        # Pricing data (per 1K tokens)
        self.pricing_data = {
            "google/gemini-1.5-flash": {"input": 0.00015, "output": 0.0006},
            "google/gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
            "openai/gpt-4o": {"input": 0.005, "output": 0.015},
            "openai/gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "openai/gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
            "anthropic/claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
            "anthropic/claude-3-5-haiku-20241022": {"input": 0.00025, "output": 0.00125},
        }

    def calculate_cost_per_request(
        self, cost_data: CostData, input_tokens: int = 500, output_tokens: int = 500
    ) -> float:
        """Calculate the cost per request based on token usage."""
        input_cost = (input_tokens / 1000) * cost_data.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000) * cost_data.cost_per_1k_output_tokens
        return input_cost + output_cost

    def calculate_cost_per_accuracy_point(self, cost_per_request: float, accuracy: float) -> float:
        """Calculate cost per accuracy percentage point."""
        if accuracy <= 0:
            return float("inf")
        return cost_per_request / accuracy

    def calculate_cost_per_satisfaction_point(
        self, cost_per_request: float, satisfaction_score: float
    ) -> float:
        """Calculate cost per user satisfaction point (1-5 scale)."""
        if satisfaction_score <= 0:
            return float("inf")
        return cost_per_request / satisfaction_score

    def calculate_cost_per_successful_completion(
        self, cost_per_request: float, completion_rate: float
    ) -> float:
        """Calculate the effective cost considering task completion rate."""
        if completion_rate <= 0:
            return float("inf")
        # If completion rate is 0.8, we need 1.25 attempts on average
        avg_attempts = 1 / completion_rate
        return cost_per_request * avg_attempts

    def calculate_latency_adjusted_cost(
        self, cost_per_request: float, latency_ms: float, target_latency_ms: float = 1000
    ) -> float:
        """Calculate cost with latency penalty/bonus."""
        # Penalty/bonus factor based on latency vs target
        latency_factor = latency_ms / target_latency_ms

        if latency_factor <= 1:
            # Bonus for faster than target (up to 20% discount)
            adjustment = 1 - (0.2 * (1 - latency_factor))
        else:
            # Penalty for slower than target (up to 50% increase)
            adjustment = 1 + (0.5 * (latency_factor - 1))

        return cost_per_request * adjustment

    def calculate_roi_score(
        self, benchmark: BenchmarkResult, cost_per_request: float, revenue_per_request: float = 0.10
    ) -> float:
        """Calculate ROI score based on assumed revenue per successful request."""
        # Expected revenue considering completion rate
        expected_revenue = revenue_per_request * benchmark.task_completion_rate

        # ROI = (Revenue - Cost) / Cost
        if cost_per_request <= 0:
            return 0

        roi = (expected_revenue - cost_per_request) / cost_per_request
        return max(0, roi)  # Return 0 for negative ROI

    def calculate_weighted_efficiency_score(
        self,
        benchmark: BenchmarkResult,
        metrics: QualityMetrics,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Calculate a weighted efficiency score combining multiple factors."""
        if weights is None:
            weights = self.default_weights

        # Normalize metrics to 0-1 scale
        normalized_accuracy = benchmark.accuracy / 100
        normalized_latency = min(1, 1000 / benchmark.latency_ms)  # Inverse, capped at 1
        normalized_satisfaction = benchmark.user_satisfaction_score / 5
        normalized_completion = benchmark.task_completion_rate

        # Calculate weighted score
        score = (
            weights["accuracy"] * normalized_accuracy
            + weights["latency"] * normalized_latency
            + weights["satisfaction"] * normalized_satisfaction
            + weights["completion"] * normalized_completion
        )

        # Divide by cost to get efficiency (higher is better)
        if metrics.cost_per_request > 0:
            efficiency = score / metrics.cost_per_request
        else:
            efficiency = float("inf")

        return efficiency

    def analyze_model(
        self,
        benchmark: BenchmarkResult,
        cost_data: CostData,
        weights: Optional[Dict[str, float]] = None,
    ) -> QualityMetrics:
        """Perform complete cost-quality analysis for a model."""
        # Calculate base cost
        cost_per_request = self.calculate_cost_per_request(cost_data)

        # Calculate all metrics
        metrics = QualityMetrics(
            provider=benchmark.provider,
            model=benchmark.model,
            cost_per_request=cost_per_request,
            cost_per_accuracy_point=self.calculate_cost_per_accuracy_point(
                cost_per_request, benchmark.accuracy
            ),
            cost_per_satisfaction_point=self.calculate_cost_per_satisfaction_point(
                cost_per_request, benchmark.user_satisfaction_score
            ),
            cost_per_successful_completion=self.calculate_cost_per_successful_completion(
                cost_per_request, benchmark.task_completion_rate
            ),
            latency_adjusted_cost=self.calculate_latency_adjusted_cost(
                cost_per_request, benchmark.latency_ms
            ),
            roi_score=self.calculate_roi_score(benchmark, cost_per_request),
            weighted_efficiency_score=0,  # Calculated below
        )

        # Calculate efficiency score
        metrics.weighted_efficiency_score = self.calculate_weighted_efficiency_score(
            benchmark, metrics, weights
        )

        return metrics

    def compare_models(
        self,
        models_data: List[Tuple[BenchmarkResult, CostData]],
        weights: Optional[Dict[str, float]] = None,
    ) -> List[QualityMetrics]:
        """Compare multiple models and rank by efficiency."""
        results = []

        for benchmark, cost_data in models_data:
            metrics = self.analyze_model(benchmark, cost_data, weights)
            results.append(metrics)

        # Sort by weighted efficiency score (descending)
        results.sort(key=lambda x: x.weighted_efficiency_score, reverse=True)

        return results


def create_sample_benchmarks() -> List[Tuple[BenchmarkResult, CostData]]:
    """Create sample benchmark data for demonstration."""
    samples = [
        # High performance, high cost
        (
            BenchmarkResult(
                provider="openai",
                model="gpt-4o",
                accuracy=95.5,
                latency_ms=1200,
                tokens_per_second=50,
                task_completion_rate=0.98,
                user_satisfaction_score=4.8,
            ),
            CostData(
                provider="openai",
                model="gpt-4o",
                cost_per_1k_input_tokens=5.0,
                cost_per_1k_output_tokens=15.0,
                avg_tokens_per_request=1000,
            ),
        ),
        # Balanced performance and cost
        (
            BenchmarkResult(
                provider="google",
                model="gemini-1.5-pro",
                accuracy=92.0,
                latency_ms=800,
                tokens_per_second=80,
                task_completion_rate=0.95,
                user_satisfaction_score=4.5,
            ),
            CostData(
                provider="google",
                model="gemini-1.5-pro",
                cost_per_1k_input_tokens=1.25,
                cost_per_1k_output_tokens=5.0,
                avg_tokens_per_request=1000,
            ),
        ),
        # Low cost, good performance
        (
            BenchmarkResult(
                provider="google",
                model="gemini-1.5-flash",
                accuracy=88.0,
                latency_ms=400,
                tokens_per_second=120,
                task_completion_rate=0.92,
                user_satisfaction_score=4.2,
            ),
            CostData(
                provider="google",
                model="gemini-1.5-flash",
                cost_per_1k_input_tokens=0.15,
                cost_per_1k_output_tokens=0.60,
                avg_tokens_per_request=1000,
            ),
        ),
        # Claude balanced option
        (
            BenchmarkResult(
                provider="anthropic",
                model="claude-3-5-haiku-20241022",
                accuracy=90.0,
                latency_ms=600,
                tokens_per_second=100,
                task_completion_rate=0.94,
                user_satisfaction_score=4.4,
            ),
            CostData(
                provider="anthropic",
                model="claude-3-5-haiku-20241022",
                cost_per_1k_input_tokens=0.25,
                cost_per_1k_output_tokens=1.25,
                avg_tokens_per_request=1000,
            ),
        ),
    ]

    return samples


def demonstrate_weighted_analysis():
    """Demonstrate analysis with different weight configurations."""
    analyzer = CostQualityAnalyzer()
    models_data = create_sample_benchmarks()

    # Different weight scenarios
    weight_scenarios = {
        "Balanced": {"accuracy": 0.25, "latency": 0.25, "satisfaction": 0.25, "completion": 0.25},
        "Quality-Focused": {
            "accuracy": 0.40,
            "latency": 0.10,
            "satisfaction": 0.35,
            "completion": 0.15,
        },
        "Speed-Focused": {
            "accuracy": 0.20,
            "latency": 0.50,
            "satisfaction": 0.15,
            "completion": 0.15,
        },
        "Cost-Focused": {
            "accuracy": 0.20,
            "latency": 0.20,
            "satisfaction": 0.30,
            "completion": 0.30,
        },
    }

    results_by_scenario = {}

    for scenario_name, weights in weight_scenarios.items():
        results = analyzer.compare_models(models_data, weights)
        results_by_scenario[scenario_name] = results

    return results_by_scenario


def format_metrics_table(metrics_list: List[QualityMetrics]) -> str:
    """Format metrics as a readable table."""
    lines = []

    # Header
    lines.append("Model | Cost/Req | Cost/Acc | Cost/Sat | Efficiency Score")
    lines.append("-" * 65)

    # Rows
    for m in metrics_list:
        model_name = f"{m.provider}/{m.model}"[:25].ljust(25)
        lines.append(
            f"{model_name} | ${m.cost_per_request:.4f} | "
            f"${m.cost_per_accuracy_point:.4f} | "
            f"${m.cost_per_satisfaction_point:.4f} | "
            f"{m.weighted_efficiency_score:.2f}"
        )

    return "\n".join(lines)


def save_analysis_results(results_by_scenario: Dict, output_dir: str = "examples/results"):
    """Save detailed analysis results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare data for JSON
    json_data = {}
    for scenario, metrics_list in results_by_scenario.items():
        json_data[scenario] = [
            {
                "model": f"{m.provider}/{m.model}",
                "cost_per_request": round(m.cost_per_request, 4),
                "cost_per_accuracy_point": round(m.cost_per_accuracy_point, 4),
                "cost_per_satisfaction_point": round(m.cost_per_satisfaction_point, 4),
                "cost_per_successful_completion": round(m.cost_per_successful_completion, 4),
                "latency_adjusted_cost": round(m.latency_adjusted_cost, 4),
                "roi_score": round(m.roi_score, 2),
                "weighted_efficiency_score": round(m.weighted_efficiency_score, 2),
            }
            for m in metrics_list
        ]

    # Save JSON
    json_path = Path(output_dir) / "cost_quality_analysis.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # Save readable report
    report_path = Path(output_dir) / "cost_quality_report.md"
    with open(report_path, "w") as f:
        f.write("# Cost-Per-Quality Metrics Analysis\n\n")

        for scenario, metrics_list in results_by_scenario.items():
            f.write(f"## {scenario} Scenario\n\n")
            f.write("```\n")
            f.write(format_metrics_table(metrics_list))
            f.write("\n```\n\n")

            # Winner for this scenario
            winner = metrics_list[0]
            f.write(f"**Best Choice**: {winner.provider}/{winner.model} ")
            f.write(f"(Efficiency Score: {winner.weighted_efficiency_score:.2f})\n\n")

    print("✅ Analysis saved to:")
    print(f"   - {json_path}")
    print(f"   - {report_path}")


def main():
    """Run cost-quality analysis demonstration."""
    print("💰 Cost-Per-Quality Metrics Analysis")
    print("=" * 50)

    # Run analysis with different weight scenarios
    results = demonstrate_weighted_analysis()

    # Display results
    for scenario_name, metrics_list in results.items():
        print(f"\n📊 {scenario_name} Scenario")
        print("-" * 40)

        for i, metrics in enumerate(metrics_list[:3], 1):  # Top 3
            print(f"{i}. {metrics.provider}/{metrics.model}")
            print(f"   Cost per request: ${metrics.cost_per_request:.4f}")
            print(f"   Efficiency score: {metrics.weighted_efficiency_score:.2f}")
            print(f"   ROI score: {metrics.roi_score:.2f}")

    # Save results
    print("\n💾 Saving analysis results...")
    save_analysis_results(results)

    print("\n🎯 Key Insights:")
    print("- Different use cases require different weight configurations")
    print("- Low-cost models can be highly efficient for many tasks")
    print("- Premium models justify their cost for quality-critical applications")
    print("- Latency requirements significantly impact the optimal choice")


if __name__ == "__main__":
    main()
