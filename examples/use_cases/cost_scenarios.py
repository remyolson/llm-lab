#!/usr/bin/env python3
"""
Real-World Cost Analysis Scenarios

This module demonstrates practical cost vs performance trade-offs for different
LLM use cases, helping users make informed decisions based on their specific needs.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Scenario:
    """Represents a real-world use case scenario."""

    name: str
    description: str
    monthly_requests: int
    avg_input_tokens: int
    avg_output_tokens: int
    quality_requirements: str
    latency_tolerance: str
    features_needed: List[str]
    recommended_models: Dict[str, str]  # provider -> model

    def calculate_monthly_cost(self, pricing_data: Dict) -> Dict[str, float]:
        """Calculate monthly costs for each provider."""
        costs = {}
        total_input_tokens = self.monthly_requests * self.avg_input_tokens
        total_output_tokens = self.monthly_requests * self.avg_output_tokens

        for provider, model in self.recommended_models.items():
            if provider in pricing_data and model in pricing_data[provider]:
                prices = pricing_data[provider][model]
                input_cost = (total_input_tokens / 1_000_000) * prices["input"]
                output_cost = (total_output_tokens / 1_000_000) * prices["output"]
                costs[f"{provider}/{model}"] = input_cost + output_cost

        return costs


# Define real-world scenarios
SCENARIOS = [
    Scenario(
        name="Customer Service Chatbot",
        description="High-volume customer support automation handling FAQs and basic troubleshooting",
        monthly_requests=500_000,
        avg_input_tokens=150,  # Customer query + context
        avg_output_tokens=200,  # Support response
        quality_requirements="Medium - Must understand queries and provide helpful responses",
        latency_tolerance="Low - Sub-second response required",
        features_needed=["Multi-turn conversation", "Context retention", "Polite tone"],
        recommended_models={
            "google": "gemini-1.5-flash",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-haiku-20241022",
        },
    ),
    Scenario(
        name="Code Generation Assistant",
        description="Developer tool for generating code snippets, debugging, and refactoring",
        monthly_requests=50_000,
        avg_input_tokens=500,  # Code context + instructions
        avg_output_tokens=800,  # Generated code + explanations
        quality_requirements="High - Must generate correct, efficient code",
        latency_tolerance="Medium - 2-5 seconds acceptable",
        features_needed=["Code understanding", "Multiple languages", "Best practices"],
        recommended_models={
            "google": "gemini-1.5-pro",
            "openai": "gpt-4o",
            "anthropic": "claude-3-5-sonnet-20241022",
        },
    ),
    Scenario(
        name="Content Creation Tool",
        description="Marketing content generator for blog posts, social media, and ad copy",
        monthly_requests=10_000,
        avg_input_tokens=300,  # Brief + guidelines
        avg_output_tokens=1500,  # Full article/content
        quality_requirements="High - Creative, engaging, brand-aligned content",
        latency_tolerance="High - Minutes acceptable for quality content",
        features_needed=["Creativity", "Style adaptation", "SEO awareness"],
        recommended_models={
            "google": "gemini-1.5-pro",
            "openai": "gpt-4o",
            "anthropic": "claude-3-5-sonnet-20241022",
        },
    ),
    Scenario(
        name="Data Analysis Assistant",
        description="Batch processing of reports, data summaries, and insights generation",
        monthly_requests=5_000,
        avg_input_tokens=2000,  # Large datasets/reports
        avg_output_tokens=1000,  # Analysis and insights
        quality_requirements="High - Accurate analysis and insights",
        latency_tolerance="Very High - Batch processing overnight",
        features_needed=["Data comprehension", "Statistical analysis", "Visualization suggestions"],
        recommended_models={
            "google": "gemini-1.5-pro",
            "openai": "gpt-4-turbo",
            "anthropic": "claude-3-opus-20240229",
        },
    ),
    Scenario(
        name="Real-time Translation Service",
        description="Live translation for chat, documents, and customer communications",
        monthly_requests=200_000,
        avg_input_tokens=100,  # Short messages
        avg_output_tokens=120,  # Translated text
        quality_requirements="Medium-High - Accurate translations with context",
        latency_tolerance="Very Low - Near real-time required",
        features_needed=["Multi-language support", "Context awareness", "Cultural nuance"],
        recommended_models={
            "google": "gemini-1.5-flash",
            "openai": "gpt-3.5-turbo",
            "anthropic": "claude-3-haiku-20240307",
        },
    ),
]

# Current pricing data (same as in cost_analysis.py)
PRICING_DATA = {
    "google": {
        "gemini-1.5-flash": {"input": 0.15, "output": 0.60},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.0-pro": {"input": 0.50, "output": 1.50},
    },
    "openai": {
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 1.00, "output": 2.00},
    },
    "anthropic": {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.25, "output": 1.25},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    },
}


def analyze_scenario(scenario: Scenario) -> Dict:
    """Analyze costs and provide recommendations for a scenario."""
    costs = scenario.calculate_monthly_cost(PRICING_DATA)

    # Sort by cost
    sorted_costs = sorted(costs.items(), key=lambda x: x[1])

    # Calculate cost range
    if sorted_costs:
        min_cost = sorted_costs[0][1]
        max_cost = sorted_costs[-1][1]
        cost_variance = (max_cost - min_cost) / min_cost if min_cost > 0 else 0
    else:
        min_cost = max_cost = cost_variance = 0

    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "monthly_volume": {
            "requests": scenario.monthly_requests,
            "input_tokens": scenario.monthly_requests * scenario.avg_input_tokens,
            "output_tokens": scenario.monthly_requests * scenario.avg_output_tokens,
            "total_tokens": scenario.monthly_requests
            * (scenario.avg_input_tokens + scenario.avg_output_tokens),
        },
        "requirements": {
            "quality": scenario.quality_requirements,
            "latency": scenario.latency_tolerance,
            "features": scenario.features_needed,
        },
        "cost_analysis": {
            "options": sorted_costs,
            "cheapest": sorted_costs[0] if sorted_costs else None,
            "most_expensive": sorted_costs[-1] if sorted_costs else None,
            "cost_range": f"${min_cost:.2f} - ${max_cost:.2f}",
            "variance_percentage": f"{cost_variance:.1%}",
        },
        "recommendations": generate_recommendations(scenario, sorted_costs),
    }


def generate_recommendations(
    scenario: Scenario, sorted_costs: List[Tuple[str, float]]
) -> List[str]:
    """Generate specific recommendations based on scenario analysis."""
    recommendations = []

    if not sorted_costs:
        return ["No pricing data available for recommended models"]

    cheapest_option = sorted_costs[0][0]
    cheapest_cost = sorted_costs[0][1]

    # Cost-based recommendations
    if cheapest_cost > 1000:
        recommendations.append(
            f"⚠️ High monthly cost (>${cheapest_cost:.0f}). Consider optimizing prompts or caching responses."
        )
    elif cheapest_cost > 500:
        recommendations.append("💰 Moderate monthly cost. Monitor usage to stay within budget.")
    else:
        recommendations.append(f"✅ Low monthly cost (${cheapest_cost:.2f}). Good for scaling.")

    # Latency-based recommendations
    if "Very Low" in scenario.latency_tolerance or "Low" in scenario.latency_tolerance:
        recommendations.append(
            "🚀 Use dedicated instances or cached responses for consistent low latency."
        )
        if "gemini-1.5-flash" in cheapest_option or "gpt-3.5-turbo" in cheapest_option:
            recommendations.append("✅ Fast models selected - good for real-time applications.")

    # Quality-based recommendations
    if "High" in scenario.quality_requirements:
        if any(
            model in cheapest_option for model in ["gpt-4", "claude-3-5-sonnet", "gemini-1.5-pro"]
        ):
            recommendations.append("✅ High-quality models selected for accuracy.")
        else:
            recommendations.append("⚠️ Consider upgrading to premium models for better quality.")

    # Volume-based recommendations
    if scenario.monthly_requests > 100_000:
        recommendations.append(
            "📊 High volume usage - negotiate enterprise pricing or consider fine-tuned models."
        )

    # Feature-specific recommendations
    if "Code understanding" in scenario.features_needed:
        recommendations.append("💻 For code tasks, Claude and GPT-4 models typically perform best.")
    if "Multi-language support" in scenario.features_needed:
        recommendations.append("🌍 Google and OpenAI models have strong multilingual capabilities.")

    return recommendations


def create_comparison_matrix() -> str:
    """Create a comparison matrix of all scenarios."""
    headers = [
        "Scenario",
        "Monthly Requests",
        "Cheapest Option",
        "Cost",
        "Best Performance",
        "Cost",
    ]
    rows = []

    for scenario in SCENARIOS:
        analysis = analyze_scenario(scenario)
        costs = analysis["cost_analysis"]["options"]

        if costs:
            cheapest = costs[0]
            # Assume highest cost = best performance for this example
            best_perf = costs[-1]

            rows.append(
                [
                    scenario.name,
                    f"{scenario.monthly_requests:,}",
                    cheapest[0].split("/")[1],
                    f"${cheapest[1]:.2f}",
                    best_perf[0].split("/")[1],
                    f"${best_perf[1]:.2f}",
                ]
            )

    # Format as table
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]

    table = []
    # Header
    table.append(" | ".join(h.ljust(w) for h, w in zip(headers, col_widths)))
    table.append("-|-".join("-" * w for w in col_widths))
    # Rows
    for row in rows:
        table.append(" | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))

    return "\n".join(table)


def save_scenario_analysis(output_dir: str = "examples/results"):
    """Save detailed scenario analysis to file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    full_analysis = {
        "scenarios": [analyze_scenario(scenario) for scenario in SCENARIOS],
        "comparison_matrix": create_comparison_matrix(),
        "general_recommendations": [
            "Start with the cheapest option and upgrade based on quality needs",
            "Implement caching for repeated queries to reduce costs",
            "Monitor actual token usage vs estimates and adjust",
            "Consider fine-tuning for high-volume, domain-specific tasks",
            "Use different models for different parts of your pipeline",
        ],
    }

    # Save JSON
    json_path = Path(output_dir) / "cost_scenarios_analysis.json"
    with open(json_path, "w") as f:
        json.dump(full_analysis, f, indent=2)

    # Save readable report
    report_path = Path(output_dir) / "cost_scenarios_report.md"
    with open(report_path, "w") as f:
        f.write("# Real-World LLM Cost Analysis Scenarios\n\n")

        for analysis in full_analysis["scenarios"]:
            f.write(f"## {analysis['scenario']}\n\n")
            f.write(f"**Description**: {analysis['description']}\n\n")
            f.write("**Monthly Volume**:\n")
            f.write(f"- Requests: {analysis['monthly_volume']['requests']:,}\n")
            f.write(f"- Total tokens: {analysis['monthly_volume']['total_tokens']:,}\n\n")
            f.write("**Requirements**:\n")
            f.write(f"- Quality: {analysis['requirements']['quality']}\n")
            f.write(f"- Latency: {analysis['requirements']['latency']}\n\n")
            f.write(f"**Cost Range**: {analysis['cost_analysis']['cost_range']} ")
            f.write(f"({analysis['cost_analysis']['variance_percentage']} variance)\n\n")
            f.write("**Options**:\n")
            for option, cost in analysis["cost_analysis"]["options"]:
                f.write(f"- {option}: ${cost:.2f}/month\n")
            f.write("\n**Recommendations**:\n")
            for rec in analysis["recommendations"]:
                f.write(f"- {rec}\n")
            f.write("\n---\n\n")

        f.write("## Cost Comparison Matrix\n\n")
        f.write("```\n")
        f.write(full_analysis["comparison_matrix"])
        f.write("\n```\n")

    print("✅ Scenario analysis saved to:")
    print(f"   - {json_path}")
    print(f"   - {report_path}")


def main():
    """Run scenario analysis and generate reports."""
    print("🎯 Real-World Cost Analysis Scenarios")
    print("=" * 50)

    for i, scenario in enumerate(SCENARIOS, 1):
        print(f"\n{i}. {scenario.name}")
        analysis = analyze_scenario(scenario)

        print(f"   Volume: {scenario.monthly_requests:,} requests/month")
        print(f"   Cost range: {analysis['cost_analysis']['cost_range']}")

        if analysis["cost_analysis"]["cheapest"]:
            model, cost = analysis["cost_analysis"]["cheapest"]
            print(f"   Cheapest: {model} at ${cost:.2f}/month")

        print("   Key recommendations:")
        for rec in analysis["recommendations"][:2]:  # Show first 2
            print(f"   {rec}")

    print("\n" + "=" * 50)
    print("💾 Saving detailed analysis...")
    save_scenario_analysis()

    print("\n📊 Comparison Matrix:")
    print(create_comparison_matrix())


if __name__ == "__main__":
    main()
