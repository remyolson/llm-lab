#!/usr/bin/env python3
"""
Cost Analysis Visualizations

This module provides comprehensive visualization examples for LLM cost vs performance
analysis using matplotlib and plotly.
"""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

# Set style for better-looking plots
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class CostVisualizer:
    """Creates various visualizations for cost analysis data."""

    def __init__(self, output_dir: str = "examples/results/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Sample data for demonstrations
        self.providers = ["OpenAI", "Anthropic", "Google"]
        self.models = {
            "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            "Anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            "Google": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
        }

    def generate_sample_data(self) -> pd.DataFrame:
        """Generate sample cost and performance data."""
        data = []

        # Cost and performance characteristics
        model_specs = {
            "gpt-4o": {"cost": 10.0, "accuracy": 95, "latency": 1200, "tokens_per_sec": 50},
            "gpt-4o-mini": {"cost": 0.375, "accuracy": 89, "latency": 800, "tokens_per_sec": 80},
            "gpt-3.5-turbo": {"cost": 1.5, "accuracy": 85, "latency": 600, "tokens_per_sec": 100},
            "claude-3-opus": {"cost": 45.0, "accuracy": 96, "latency": 1500, "tokens_per_sec": 40},
            "claude-3-sonnet": {"cost": 9.0, "accuracy": 93, "latency": 1000, "tokens_per_sec": 60},
            "claude-3-haiku": {"cost": 0.75, "accuracy": 88, "latency": 500, "tokens_per_sec": 120},
            "gemini-1.5-pro": {"cost": 3.125, "accuracy": 94, "latency": 900, "tokens_per_sec": 70},
            "gemini-1.5-flash": {
                "cost": 0.375,
                "accuracy": 87,
                "latency": 400,
                "tokens_per_sec": 150,
            },
            "gemini-1.0-pro": {"cost": 1.0, "accuracy": 86, "latency": 700, "tokens_per_sec": 90},
        }

        for provider, models in self.models.items():
            for model in models:
                specs = model_specs[model]
                data.append(
                    {
                        "provider": provider,
                        "model": model,
                        "cost_per_1k_tokens": specs["cost"],
                        "accuracy": specs["accuracy"],
                        "latency_ms": specs["latency"],
                        "tokens_per_second": specs["tokens_per_sec"],
                        "cost_per_accuracy": specs["cost"] / specs["accuracy"],
                        "efficiency_score": specs["accuracy"] / specs["cost"],
                    }
                )

        return pd.DataFrame(data)

    def create_scatter_plot(self, df: pd.DataFrame) -> Tuple[plt.Figure, str]:
        """Create scatter plot of cost vs accuracy."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot each provider with different colors
        for provider in self.providers:
            provider_data = df[df["provider"] == provider]
            ax.scatter(
                provider_data["cost_per_1k_tokens"],
                provider_data["accuracy"],
                label=provider,
                s=100,
                alpha=0.7,
            )

            # Add model labels
            for _, row in provider_data.iterrows():
                ax.annotate(
                    row["model"].split("-")[-1],
                    (row["cost_per_1k_tokens"], row["accuracy"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        ax.set_xlabel("Cost per 1K Tokens ($)", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("LLM Cost vs Accuracy Analysis", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add efficient frontier line
        frontier_models = df.nlargest(3, "efficiency_score")
        ax.plot(
            frontier_models["cost_per_1k_tokens"],
            frontier_models["accuracy"],
            "r--",
            alpha=0.5,
            label="Efficiency Frontier",
        )

        plt.tight_layout()

        # Save
        output_path = self.output_dir / "cost_vs_accuracy_scatter.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

        return fig, str(output_path)

    def create_bar_chart_comparison(self, df: pd.DataFrame) -> Tuple[plt.Figure, str]:
        """Create grouped bar chart comparing providers."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Prepare data
        provider_avg = (
            df.groupby("provider")
            .agg({"cost_per_1k_tokens": "mean", "accuracy": "mean", "latency_ms": "mean"})
            .round(2)
        )

        # Cost comparison
        x = np.arange(len(provider_avg))
        width = 0.35

        ax1.bar(x, provider_avg["cost_per_1k_tokens"], width, label="Avg Cost", color="coral")
        ax1.set_ylabel("Average Cost per 1K Tokens ($)", fontsize=12)
        ax1.set_title("Provider Cost Comparison", fontsize=14, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(provider_avg.index)

        # Add value labels
        for i, v in enumerate(provider_avg["cost_per_1k_tokens"]):
            ax1.text(i, v + 0.5, f"${v:.2f}", ha="center", va="bottom")

        # Performance comparison
        ax2_twin = ax2.twinx()

        bar1 = ax2.bar(
            x - width / 2, provider_avg["accuracy"], width, label="Accuracy", color="skyblue"
        )
        bar2 = ax2_twin.bar(
            x + width / 2, provider_avg["latency_ms"], width, label="Latency", color="lightgreen"
        )

        ax2.set_ylabel("Average Accuracy (%)", fontsize=12)
        ax2_twin.set_ylabel("Average Latency (ms)", fontsize=12)
        ax2.set_xlabel("Provider", fontsize=12)
        ax2.set_title("Provider Performance Comparison", fontsize=14, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(provider_avg.index)

        # Add legends
        ax2.legend(loc="upper left")
        ax2_twin.legend(loc="upper right")

        plt.tight_layout()

        # Save
        output_path = self.output_dir / "provider_comparison_bars.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

        return fig, str(output_path)

    def create_line_graph_volume_cost(self) -> Tuple[plt.Figure, str]:
        """Create line graph showing cost over different request volumes."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Generate volume data
        volumes = np.array([1000, 5000, 10000, 50000, 100000, 500000, 1000000])

        # Cost data for different models
        models_to_plot = ["gpt-4o", "claude-3-haiku", "gemini-1.5-flash"]
        costs_per_1k = {"gpt-4o": 10.0, "claude-3-haiku": 0.75, "gemini-1.5-flash": 0.375}

        for model in models_to_plot:
            costs = (volumes / 1000) * costs_per_1k[model]
            ax.plot(volumes, costs, marker="o", label=model, linewidth=2)

        ax.set_xlabel("Monthly Request Volume", fontsize=12)
        ax.set_ylabel("Monthly Cost ($)", fontsize=12)
        ax.set_title("Cost Scaling with Request Volume", fontsize=14, fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend()

        # Add annotations for key thresholds
        ax.axhline(y=100, color="red", linestyle="--", alpha=0.5)
        ax.text(2000, 120, "$100/month threshold", fontsize=10, color="red")

        plt.tight_layout()

        # Save
        output_path = self.output_dir / "volume_cost_scaling.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

        return fig, str(output_path)

    def create_heatmap_performance_cost(self, df: pd.DataFrame) -> Tuple[plt.Figure, str]:
        """Create heatmap of model performance vs cost matrix."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Prepare pivot tables
        cost_pivot = df.pivot_table(values="cost_per_1k_tokens", index="model", columns="provider")
        efficiency_pivot = df.pivot_table(
            values="efficiency_score", index="model", columns="provider"
        )

        # Cost heatmap
        sns.heatmap(
            cost_pivot,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            ax=ax1,
            cbar_kws={"label": "Cost per 1K tokens ($)"},
        )
        ax1.set_title("Cost Heatmap by Model and Provider", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Provider")
        ax1.set_ylabel("Model")

        # Efficiency heatmap
        sns.heatmap(
            efficiency_pivot,
            annot=True,
            fmt=".1f",
            cmap="YlGn",
            ax=ax2,
            cbar_kws={"label": "Efficiency Score"},
        )
        ax2.set_title("Efficiency Score Heatmap", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Provider")
        ax2.set_ylabel("Model")

        plt.tight_layout()

        # Save
        output_path = self.output_dir / "performance_cost_heatmap.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

        return fig, str(output_path)

    def create_interactive_dashboard(self, df: pd.DataFrame) -> str:
        """Create interactive Plotly dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Cost vs Accuracy",
                "Provider Comparison",
                "Cost Distribution",
                "Performance Radar",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "box"}, {"type": "scatterpolar"}],
            ],
        )

        # 1. Interactive scatter plot
        for provider in self.providers:
            provider_data = df[df["provider"] == provider]
            fig.add_trace(
                go.Scatter(
                    x=provider_data["cost_per_1k_tokens"],
                    y=provider_data["accuracy"],
                    mode="markers+text",
                    name=provider,
                    text=provider_data["model"].str.split("-").str[-1],
                    textposition="top center",
                    marker=dict(size=12),
                    hovertemplate="Model: %{text}<br>Cost: $%{x:.2f}<br>Accuracy: %{y}%",
                ),
                row=1,
                col=1,
            )

        # 2. Bar chart
        provider_avg = df.groupby("provider")["cost_per_1k_tokens"].mean()
        fig.add_trace(
            go.Bar(
                x=provider_avg.index,
                y=provider_avg.values,
                name="Avg Cost",
                text=[f"${v:.2f}" for v in provider_avg.values],
                textposition="outside",
            ),
            row=1,
            col=2,
        )

        # 3. Box plot for cost distribution
        for provider in self.providers:
            provider_data = df[df["provider"] == provider]
            fig.add_trace(
                go.Box(
                    y=provider_data["cost_per_1k_tokens"],
                    name=provider,
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                ),
                row=2,
                col=1,
            )

        # 4. Radar chart for top models
        top_models = df.nsmallest(3, "cost_per_accuracy")

        categories = ["Cost Efficiency", "Accuracy", "Speed", "Overall"]

        for _, model in top_models.iterrows():
            # Normalize metrics to 0-100 scale
            cost_eff = (1 - model["cost_per_1k_tokens"] / df["cost_per_1k_tokens"].max()) * 100
            accuracy = model["accuracy"]
            speed = (model["tokens_per_second"] / df["tokens_per_second"].max()) * 100
            overall = (cost_eff + accuracy + speed) / 3

            fig.add_trace(
                go.Scatterpolar(
                    r=[cost_eff, accuracy, speed, overall],
                    theta=categories,
                    fill="toself",
                    name=model["model"],
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(
            title_text="LLM Cost Analysis Interactive Dashboard",
            showlegend=True,
            height=900,
            width=1400,
        )

        # Update axes
        fig.update_xaxes(title_text="Cost per 1K Tokens ($)", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
        fig.update_xaxes(title_text="Provider", row=1, col=2)
        fig.update_yaxes(title_text="Average Cost ($)", row=1, col=2)
        fig.update_yaxes(title_text="Cost per 1K Tokens ($)", row=2, col=1)

        # Save interactive HTML
        output_path = self.output_dir / "interactive_dashboard.html"
        fig.write_html(str(output_path))

        return str(output_path)

    def create_combined_metrics_plot(self, df: pd.DataFrame) -> Tuple[plt.Figure, str]:
        """Create a combined visualization showing multiple metrics."""
        fig = plt.figure(figsize=(16, 10))

        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Main scatter plot with size based on efficiency
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        scatter = ax1.scatter(
            df["cost_per_1k_tokens"],
            df["accuracy"],
            s=df["efficiency_score"] * 50,
            c=df["latency_ms"],
            cmap="viridis",
            alpha=0.6,
            edgecolors="black",
            linewidth=1,
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label("Latency (ms)", fontsize=10)

        ax1.set_xlabel("Cost per 1K Tokens ($)", fontsize=12)
        ax1.set_ylabel("Accuracy (%)", fontsize=12)
        ax1.set_title(
            "Multi-Dimensional Cost Analysis\n(Size = Efficiency Score)",
            fontsize=14,
            fontweight="bold",
        )

        # Add model labels
        for _, row in df.iterrows():
            ax1.annotate(
                row["model"].split("-")[-1],
                (row["cost_per_1k_tokens"], row["accuracy"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )

        # 2. Cost per accuracy ranking
        ax2 = fig.add_subplot(gs[0, 2])
        sorted_df = df.sort_values("cost_per_accuracy")
        colors = ["green" if i < 3 else "orange" if i < 6 else "red" for i in range(len(sorted_df))]
        ax2.barh(range(len(sorted_df)), sorted_df["cost_per_accuracy"], color=colors)
        ax2.set_yticks(range(len(sorted_df)))
        ax2.set_yticklabels(sorted_df["model"], fontsize=8)
        ax2.set_xlabel("Cost per Accuracy Point", fontsize=10)
        ax2.set_title("Cost Efficiency Ranking", fontsize=12)
        ax2.invert_yaxis()

        # 3. Speed comparison
        ax3 = fig.add_subplot(gs[1, 2])
        sorted_speed = df.sort_values("tokens_per_second", ascending=False)
        ax3.barh(range(len(sorted_speed)), sorted_speed["tokens_per_second"])
        ax3.set_yticks(range(len(sorted_speed)))
        ax3.set_yticklabels(sorted_speed["model"], fontsize=8)
        ax3.set_xlabel("Tokens per Second", fontsize=10)
        ax3.set_title("Speed Ranking", fontsize=12)
        ax3.invert_yaxis()

        # 4. Provider summary pie chart
        ax4 = fig.add_subplot(gs[2, :])
        provider_counts = df["provider"].value_counts()
        colors_pie = plt.cm.Set3(range(len(provider_counts)))
        wedges, texts, autotexts = ax4.pie(
            provider_counts.values,
            labels=provider_counts.index,
            autopct="%1.1f%%",
            colors=colors_pie,
            startangle=90,
        )
        ax4.set_title("Model Distribution by Provider", fontsize=12)

        plt.suptitle("Comprehensive LLM Cost-Performance Analysis", fontsize=16, fontweight="bold")

        # Save
        output_path = self.output_dir / "combined_metrics_visualization.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

        return fig, str(output_path)

    def export_all_formats(self, fig: plt.Figure, base_name: str):
        """Export visualization in multiple formats."""
        formats = ["png", "pdf", "svg"]
        paths = []

        for fmt in formats:
            output_path = self.output_dir / f"{base_name}.{fmt}"
            fig.savefig(output_path, format=fmt, dpi=300, bbox_inches="tight")
            paths.append(str(output_path))

        return paths


def main():
    """Generate all visualization examples."""
    print("📊 Generating Cost Analysis Visualizations")
    print("=" * 50)

    # Initialize visualizer
    viz = CostVisualizer()

    # Generate sample data
    df = viz.generate_sample_data()

    # Create all visualizations
    visualizations = []

    print("\n1. Creating scatter plot...")
    fig1, path1 = viz.create_scatter_plot(df)
    visualizations.append(("Scatter Plot", path1))
    plt.close(fig1)

    print("2. Creating bar chart comparison...")
    fig2, path2 = viz.create_bar_chart_comparison(df)
    visualizations.append(("Bar Chart", path2))
    plt.close(fig2)

    print("3. Creating volume-cost scaling graph...")
    fig3, path3 = viz.create_line_graph_volume_cost()
    visualizations.append(("Line Graph", path3))
    plt.close(fig3)

    print("4. Creating performance-cost heatmap...")
    fig4, path4 = viz.create_heatmap_performance_cost(df)
    visualizations.append(("Heatmap", path4))
    plt.close(fig4)

    print("5. Creating interactive dashboard...")
    path5 = viz.create_interactive_dashboard(df)
    visualizations.append(("Interactive Dashboard", path5))

    print("6. Creating combined metrics visualization...")
    fig6, path6 = viz.create_combined_metrics_plot(df)
    visualizations.append(("Combined Metrics", path6))

    # Export in multiple formats
    print("\n7. Exporting in multiple formats...")
    multi_format_paths = viz.export_all_formats(fig6, "combined_metrics_multi_format")
    plt.close(fig6)

    # Summary
    print("\n✅ Visualizations created successfully!")
    print("\nGenerated files:")
    for name, path in visualizations:
        print(f"  - {name}: {path}")

    print("\nMulti-format exports:")
    for path in multi_format_paths:
        print(f"  - {path}")

    print("\n💡 Usage tips:")
    print("  - PNG files are best for reports and presentations")
    print("  - PDF files are ideal for high-quality printing")
    print("  - SVG files can be edited in vector graphics software")
    print("  - HTML dashboard is interactive and can be shared via web")


if __name__ == "__main__":
    main()
