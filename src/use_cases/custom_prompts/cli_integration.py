"""CLI integration utilities for custom prompts with evaluation metrics.

This module provides enhanced functionality for the CLI to use evaluation metrics
with custom prompts.
"""

from datetime import datetime
from typing import Any, Dict, List


# Use dynamic imports to avoid circular dependencies
def get_evaluation_metrics():
    """Dynamically import evaluation metrics to avoid circular imports."""
    from . import evaluation_metrics

    return evaluation_metrics


def enhance_response_with_metrics(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance a custom prompt response with comprehensive evaluation metrics.

    Args:
        response_data: Dictionary containing response information with at least:
            - response: The model's response text
            - model: Model name
            - prompt: Original prompt

    Returns:
        Enhanced response data with metrics added
    """
    metrics_module = get_evaluation_metrics()

    # Skip if response was not successful
    if not response_data.get("success", False) or not response_data.get("response"):
        return response_data

    # Get the response text
    response_text = response_data["response"]

    # Create metric suite with all default metrics
    metric_suite = metrics_module.MetricSuite()

    # Calculate metrics
    try:
        metrics_results = metric_suite.evaluate(response_text)

        # Extract key metrics for CLI display
        enhanced_metrics = {
            "response_length": metrics_results["response_length"]["value"],
            "sentiment": metrics_results["sentiment"]["value"],
            "coherence": metrics_results["coherence"]["value"],
            "evaluation_timestamp": datetime.now().isoformat(),
        }

        # Update the response data
        response_data["metrics"] = enhanced_metrics
        response_data["metrics_version"] = "1.0"

    except Exception as e:
        # If metrics calculation fails, add error info but don't fail the whole response
        response_data["metrics_error"] = {"error": str(e), "error_type": type(e).__name__}

    return response_data


def calculate_diversity_metrics(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate diversity metrics across multiple responses.

    Args:
        responses: List of response dictionaries

    Returns:
        Dictionary containing diversity metrics
    """
    metrics_module = get_evaluation_metrics()

    # Extract successful response texts
    response_texts = [
        r["response"] for r in responses if r.get("success", False) and r.get("response")
    ]

    if len(response_texts) < 2:
        return {
            "error": "Need at least 2 successful responses for diversity calculation",
            "response_count": len(response_texts),
        }

    # Calculate diversity
    diversity_metric = metrics_module.ResponseDiversityMetric()
    diversity_results = diversity_metric.calculate_batch(response_texts)

    return diversity_results[0].value if diversity_results else {}


def format_metrics_for_cli(metrics: Dict[str, Any]) -> List[str]:
    """Format metrics dictionary for CLI display.

    Args:
        metrics: Dictionary of metrics

    Returns:
        List of formatted strings for display
    """
    lines = []

    # Response length metrics
    if "response_length" in metrics:
        length = metrics["response_length"]
        lines.append(f"📏 Length: {length['words']} words, {length['sentences']} sentences")

    # Sentiment metrics
    if "sentiment" in metrics:
        sentiment = metrics["sentiment"]
        emoji = {"positive": "😊", "negative": "😟", "neutral": "😐"}.get(sentiment["label"], "")
        lines.append(f"💭 Sentiment: {sentiment['label']} {emoji} (score: {sentiment['score']})")

    # Coherence metrics
    if "coherence" in metrics:
        coherence = metrics["coherence"]
        score = coherence["score"]
        quality = "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
        lines.append(f"🔗 Coherence: {quality} (score: {score})")

    return lines


def create_metrics_summary(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a summary of metrics across multiple responses.

    Args:
        responses: List of response dictionaries with metrics

    Returns:
        Summary dictionary with aggregated metrics
    """
    metrics_module = get_evaluation_metrics()

    # Filter responses with metrics
    responses_with_metrics = [r for r in responses if r.get("success", False) and "metrics" in r]

    if not responses_with_metrics:
        return {"error": "No successful responses with metrics to summarize"}

    # Aggregate metrics
    summary = {
        "total_responses": len(responses),
        "successful_responses": len(responses_with_metrics),
        "metrics": {},
    }

    # Calculate averages for numeric metrics
    metric_keys = ["response_length", "sentiment", "coherence"]
    for key in metric_keys:
        values = []
        for response in responses_with_metrics:
            if key in response.get("metrics", {}):
                metric_data = response["metrics"][key]
                if key == "response_length":
                    values.append(metric_data["words"])
                elif key == "sentiment":
                    values.append(metric_data["score"])
                elif key == "coherence":
                    values.append(metric_data["score"])

        if values:
            import statistics

            summary["metrics"][key] = {
                "mean": round(statistics.mean(values), 3),
                "std": round(statistics.stdev(values), 3) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
            }

    # Add diversity metrics if multiple responses
    if len(responses_with_metrics) > 1:
        diversity = calculate_diversity_metrics(responses_with_metrics)
        summary["metrics"]["diversity"] = diversity

    return summary


def should_calculate_metrics(cli_args: Dict[str, Any]) -> bool:
    """Determine if metrics should be calculated based on CLI arguments.

    Args:
        cli_args: Dictionary of CLI arguments

    Returns:
        True if metrics should be calculated
    """
    # Check if metrics are explicitly disabled
    if cli_args.get("no_metrics", False):
        return False

    # Check if metrics are explicitly enabled
    if cli_args.get("metrics", False):
        return True

    # Default: calculate metrics for custom prompts
    return cli_args.get("custom_prompt") or cli_args.get("prompt_file")


# Example usage in run_benchmarks.py:
"""
# In run_custom_prompt function, after getting response:
if calculate_metrics:
    from src.use_cases.custom_prompts.cli_integration import enhance_response_with_metrics
    results = enhance_response_with_metrics(results)

    # Display metrics in CLI
    if results.get("metrics"):
        from src.use_cases.custom_prompts.cli_integration import format_metrics_for_cli
        metric_lines = format_metrics_for_cli(results["metrics"])
        for line in metric_lines:
            click.echo(f"   {line}")

# After running multiple models:
if calculate_metrics and len(all_results) > 1:
    from src.use_cases.custom_prompts.cli_integration import create_metrics_summary
    summary = create_metrics_summary(all_results)
    click.echo(f"\\n📊 Metrics Summary:")
    click.echo(f"   Successful responses: {summary['successful_responses']}/{summary['total_responses']}")
    for metric, stats in summary['metrics'].items():
        if isinstance(stats, dict) and 'mean' in stats:
            click.echo(f"   {metric}: mean={stats['mean']}, std={stats['std']}")
"""
