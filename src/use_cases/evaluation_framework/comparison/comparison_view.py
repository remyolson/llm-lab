"""
Side-by-Side Comparison Interface

This module provides an interactive comparison interface using Streamlit or Gradio
for visualizing model performance differences before and after fine-tuning,
including split-screen views, metric visualization, diff highlighting, and
statistical significance indicators.

Example:
    # Streamlit app
    app = create_streamlit_app()
    app.run()
    
    # Gradio interface
    interface = create_gradio_interface()
    interface.launch()
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import comparison results
from ..benchmark_runner import ComparisonResult, BenchmarkResult

# Import UI frameworks
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ComparisonConfig:
    """Configuration for comparison view."""
    title: str = "Model Fine-Tuning Comparison"
    theme: str = "plotly_white"
    show_statistical_significance: bool = True
    significance_threshold: float = 0.05
    highlight_improvements: bool = True
    highlight_regressions: bool = True
    improvement_color: str = "#28a745"
    regression_color: str = "#dc3545"
    neutral_color: str = "#6c757d"
    chart_height: int = 400
    enable_export: bool = True


class ComparisonView:
    """Interactive comparison view for model evaluation results."""
    
    def __init__(
        self,
        comparison_result: ComparisonResult,
        config: Optional[ComparisonConfig] = None
    ):
        """Initialize comparison view.
        
        Args:
            comparison_result: Comparison results to visualize
            config: View configuration
        """
        self.comparison_result = comparison_result
        self.config = config or ComparisonConfig()
        
        # Prepare data
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for visualization."""
        # Extract benchmark scores
        self.benchmark_data = []
        
        base_benchmarks = {
            b.name: b for b in self.comparison_result.base_result.evaluation_results.benchmarks
        }
        ft_benchmarks = {
            b.name: b for b in self.comparison_result.fine_tuned_result.evaluation_results.benchmarks
        }
        
        for name in base_benchmarks:
            if name in ft_benchmarks:
                base_score = base_benchmarks[name].overall_score
                ft_score = ft_benchmarks[name].overall_score
                improvement = ft_score - base_score
                improvement_pct = (improvement / base_score * 100) if base_score > 0 else 0
                
                self.benchmark_data.append({
                    "benchmark": name,
                    "base_score": base_score,
                    "ft_score": ft_score,
                    "improvement": improvement,
                    "improvement_pct": improvement_pct,
                    "is_significant": self._is_significant(name)
                })
        
        self.benchmark_df = pd.DataFrame(self.benchmark_data)
        
        # Extract metrics over time if available
        self._extract_time_series_metrics()
    
    def _is_significant(self, benchmark: str) -> bool:
        """Check if improvement is statistically significant.
        
        Args:
            benchmark: Benchmark name
            
        Returns:
            True if significant
        """
        if not self.config.show_statistical_significance:
            return True
        
        significance = self.comparison_result.statistical_analysis.get("significance", {})
        benchmark_sig = significance.get(benchmark, {})
        
        return benchmark_sig.get("is_significant", False)
    
    def _extract_time_series_metrics(self):
        """Extract time series metrics if available."""
        # This would extract training metrics over time
        # For now, we'll create sample data
        self.time_series_data = None
    
    def create_overview_chart(self) -> go.Figure:
        """Create overview comparison chart.
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add base model bars
        fig.add_trace(go.Bar(
            name='Base Model',
            x=self.benchmark_df['benchmark'],
            y=self.benchmark_df['base_score'],
            marker_color='lightblue',
            text=self.benchmark_df['base_score'].round(4),
            textposition='auto'
        ))
        
        # Add fine-tuned model bars
        fig.add_trace(go.Bar(
            name='Fine-Tuned Model',
            x=self.benchmark_df['benchmark'],
            y=self.benchmark_df['ft_score'],
            marker_color='darkblue',
            text=self.benchmark_df['ft_score'].round(4),
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Benchmark Performance Comparison",
            xaxis_title="Benchmark",
            yaxis_title="Score",
            barmode='group',
            height=self.config.chart_height,
            template=self.config.theme,
            showlegend=True
        )
        
        return fig
    
    def create_improvement_chart(self) -> go.Figure:
        """Create improvement visualization.
        
        Returns:
            Plotly figure
        """
        # Sort by improvement percentage
        sorted_df = self.benchmark_df.sort_values('improvement_pct', ascending=True)
        
        # Assign colors based on improvement
        colors = []
        for _, row in sorted_df.iterrows():
            if row['improvement_pct'] > 0:
                color = self.config.improvement_color
            elif row['improvement_pct'] < 0:
                color = self.config.regression_color
            else:
                color = self.config.neutral_color
            
            # Make non-significant improvements lighter
            if not row['is_significant'] and self.config.show_statistical_significance:
                color += '80'  # Add transparency
            
            colors.append(color)
        
        fig = go.Figure(go.Bar(
            x=sorted_df['improvement_pct'],
            y=sorted_df['benchmark'],
            orientation='h',
            marker_color=colors,
            text=sorted_df['improvement_pct'].round(2).astype(str) + '%',
            textposition='auto'
        ))
        
        # Add vertical line at 0
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title="Performance Improvements (%)",
            xaxis_title="Improvement (%)",
            yaxis_title="Benchmark",
            height=self.config.chart_height,
            template=self.config.theme
        )
        
        return fig
    
    def create_scatter_comparison(self) -> go.Figure:
        """Create scatter plot comparing base vs fine-tuned scores.
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=self.benchmark_df['base_score'],
            y=self.benchmark_df['ft_score'],
            mode='markers+text',
            text=self.benchmark_df['benchmark'],
            textposition="top center",
            marker=dict(
                size=12,
                color=self.benchmark_df['improvement_pct'],
                colorscale='RdYlGn',
                colorbar=dict(title="Improvement %"),
                showscale=True
            )
        ))
        
        # Add diagonal line (y=x)
        max_val = max(
            self.benchmark_df['base_score'].max(),
            self.benchmark_df['ft_score'].max()
        )
        fig.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            showlegend=False
        ))
        
        fig.update_layout(
            title="Base vs Fine-Tuned Performance",
            xaxis_title="Base Model Score",
            yaxis_title="Fine-Tuned Model Score",
            height=self.config.chart_height,
            template=self.config.theme
        )
        
        return fig
    
    def create_metric_table(self) -> pd.DataFrame:
        """Create detailed metrics table.
        
        Returns:
            DataFrame with formatted metrics
        """
        # Create formatted table
        table_data = []
        
        for _, row in self.benchmark_df.iterrows():
            # Format improvement with color
            if row['improvement_pct'] > 0:
                improvement_str = f"↑ {row['improvement_pct']:.2f}%"
                if self.config.highlight_improvements:
                    improvement_str = f"<span style='color: {self.config.improvement_color}'>{improvement_str}</span>"
            elif row['improvement_pct'] < 0:
                improvement_str = f"↓ {abs(row['improvement_pct']):.2f}%"
                if self.config.highlight_regressions:
                    improvement_str = f"<span style='color: {self.config.regression_color}'>{improvement_str}</span>"
            else:
                improvement_str = "→ 0.00%"
            
            # Add significance indicator
            if self.config.show_statistical_significance:
                if row['is_significant']:
                    significance = "✓"
                else:
                    significance = "○"
            else:
                significance = "-"
            
            table_data.append({
                "Benchmark": row['benchmark'],
                "Base Score": f"{row['base_score']:.4f}",
                "Fine-Tuned Score": f"{row['ft_score']:.4f}",
                "Improvement": improvement_str,
                "Significant": significance
            })
        
        return pd.DataFrame(table_data)
    
    def create_statistical_summary(self) -> Dict[str, Any]:
        """Create statistical summary of improvements.
        
        Returns:
            Summary statistics
        """
        summary = self.comparison_result.statistical_analysis.get("summary", {})
        
        # Add additional statistics
        improvements = self.benchmark_df['improvement_pct'].values
        
        summary.update({
            "total_benchmarks": len(self.benchmark_df),
            "improved_count": (improvements > 0).sum(),
            "regressed_count": (improvements < 0).sum(),
            "unchanged_count": (improvements == 0).sum(),
            "significant_improvements": sum(
                1 for _, row in self.benchmark_df.iterrows()
                if row['improvement_pct'] > 0 and row['is_significant']
            )
        })
        
        return summary
    
    def create_cost_benefit_summary(self) -> Dict[str, Any]:
        """Create cost/benefit summary.
        
        Returns:
            Cost/benefit metrics
        """
        # Calculate training duration
        base_duration = self.comparison_result.base_result.duration_seconds
        ft_duration = self.comparison_result.fine_tuned_result.duration_seconds
        total_duration = base_duration + ft_duration
        
        # Calculate average improvement
        avg_improvement = self.benchmark_df['improvement_pct'].mean()
        
        # Estimate cost (simplified)
        cost_per_hour = 2.0  # Example GPU cost
        training_cost = (total_duration / 3600) * cost_per_hour
        
        return {
            "total_evaluation_time_hours": total_duration / 3600,
            "estimated_training_cost": training_cost,
            "average_improvement": avg_improvement,
            "cost_per_percent_improvement": training_cost / avg_improvement if avg_improvement > 0 else float('inf'),
            "benchmarks_improved": (self.benchmark_df['improvement_pct'] > 0).sum(),
            "benchmarks_regressed": (self.benchmark_df['improvement_pct'] < 0).sum()
        }


def create_streamlit_app(comparison_result: Optional[ComparisonResult] = None) -> None:
    """Create Streamlit comparison app.
    
    Args:
        comparison_result: Optional comparison result to display
    """
    if not STREAMLIT_AVAILABLE:
        raise ImportError("Streamlit is not installed")
    
    st.set_page_config(
        page_title="Model Comparison Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🔍 Model Fine-Tuning Comparison Dashboard")
    
    # Load comparison result if not provided
    if comparison_result is None:
        st.sidebar.header("Load Comparison")
        
        uploaded_file = st.sidebar.file_uploader(
            "Upload comparison result JSON",
            type=['json']
        )
        
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                # Reconstruct ComparisonResult from JSON
                # (Implementation would depend on proper deserialization)
                st.success("Loaded comparison result")
            except Exception as e:
                st.error(f"Failed to load file: {e}")
                return
        else:
            st.info("Please upload a comparison result JSON file")
            return
    
    # Create comparison view
    config = ComparisonConfig()
    view = ComparisonView(comparison_result, config)
    
    # Sidebar configuration
    st.sidebar.header("View Configuration")
    config.show_statistical_significance = st.sidebar.checkbox(
        "Show Statistical Significance",
        value=True
    )
    config.highlight_improvements = st.sidebar.checkbox(
        "Highlight Improvements",
        value=True
    )
    config.highlight_regressions = st.sidebar.checkbox(
        "Highlight Regressions",
        value=True
    )
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    # Summary metrics
    summary = view.create_statistical_summary()
    
    with col1:
        st.metric(
            "Total Benchmarks",
            summary['total_benchmarks'],
            delta=None
        )
    
    with col2:
        st.metric(
            "Improvements",
            summary['improved_count'],
            delta=f"+{summary['improved_count']}"
        )
    
    with col3:
        st.metric(
            "Regressions",
            summary['regressed_count'],
            delta=f"-{summary['regressed_count']}" if summary['regressed_count'] > 0 else "0"
        )
    
    # Charts
    st.header("Performance Comparison")
    
    tab1, tab2, tab3 = st.tabs(["Overview", "Improvements", "Scatter Plot"])
    
    with tab1:
        st.plotly_chart(
            view.create_overview_chart(),
            use_container_width=True
        )
    
    with tab2:
        st.plotly_chart(
            view.create_improvement_chart(),
            use_container_width=True
        )
    
    with tab3:
        st.plotly_chart(
            view.create_scatter_comparison(),
            use_container_width=True
        )
    
    # Detailed metrics table
    st.header("Detailed Metrics")
    
    metrics_df = view.create_metric_table()
    st.write(metrics_df.to_html(escape=False), unsafe_allow_html=True)
    
    # Cost/Benefit Analysis
    st.header("Cost/Benefit Analysis")
    
    cost_benefit = view.create_cost_benefit_summary()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Average Improvement",
            f"{cost_benefit['average_improvement']:.2f}%"
        )
        st.metric(
            "Evaluation Time",
            f"{cost_benefit['total_evaluation_time_hours']:.2f} hours"
        )
    
    with col2:
        st.metric(
            "Estimated Cost",
            f"${cost_benefit['estimated_training_cost']:.2f}"
        )
        st.metric(
            "Cost per % Improvement",
            f"${cost_benefit['cost_per_percent_improvement']:.2f}"
            if cost_benefit['cost_per_percent_improvement'] != float('inf')
            else "N/A"
        )
    
    # Statistical Summary
    if config.show_statistical_significance:
        st.header("Statistical Analysis")
        
        stat_summary = comparison_result.statistical_analysis.get("summary", {})
        
        if stat_summary:
            stat_df = pd.DataFrame([stat_summary])
            st.dataframe(stat_df)
    
    # Export options
    if config.enable_export:
        st.header("Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export as JSON"):
                json_str = json.dumps(comparison_result.to_dict(), indent=2)
                st.download_button(
                    "Download JSON",
                    json_str,
                    "comparison_result.json",
                    "application/json"
                )
        
        with col2:
            if st.button("Export as CSV"):
                csv = metrics_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "comparison_metrics.csv",
                    "text/csv"
                )
        
        with col3:
            if st.button("Generate Report"):
                st.info("Report generation will be implemented in Task 12.5")


def create_gradio_interface(comparison_result: Optional[ComparisonResult] = None):
    """Create Gradio comparison interface.
    
    Args:
        comparison_result: Optional comparison result to display
        
    Returns:
        Gradio interface
    """
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio is not installed")
    
    def process_comparison(file_path: str) -> Tuple[Any, Any, Any, str]:
        """Process comparison file and return visualizations."""
        try:
            # Load comparison result
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Create view (would need proper deserialization)
            # For now, return placeholder
            return (
                None,  # Overview chart
                None,  # Improvement chart
                None,  # Metrics table
                "Loaded successfully"
            )
        except Exception as e:
            return None, None, None, f"Error: {str(e)}"
    
    # Create interface
    with gr.Blocks(title="Model Comparison") as interface:
        gr.Markdown("# Model Fine-Tuning Comparison")
        
        with gr.Row():
            file_input = gr.File(
                label="Upload Comparison Result",
                file_types=[".json"]
            )
            status_output = gr.Textbox(label="Status")
        
        with gr.Tab("Overview"):
            overview_plot = gr.Plot(label="Benchmark Comparison")
        
        with gr.Tab("Improvements"):
            improvement_plot = gr.Plot(label="Performance Improvements")
        
        with gr.Tab("Detailed Metrics"):
            metrics_table = gr.Dataframe(label="Metrics Table")
        
        # Connect processing
        file_input.change(
            fn=process_comparison,
            inputs=[file_input],
            outputs=[overview_plot, improvement_plot, metrics_table, status_output]
        )
    
    return interface


# Example usage
if __name__ == "__main__":
    # Example: Create dummy comparison result for testing
    from ..benchmark_runner import ModelVersion, EvaluationResult, ModelVersionType
    from ..fine_tuning.evaluation.suite import BenchmarkResult as EvalBenchmarkResult
    
    # Create dummy data
    base_eval = EvaluationResult(
        model_name="gpt2",
        benchmarks=[
            EvalBenchmarkResult(
                name="hellaswag",
                overall_score=0.45,
                task_scores={"accuracy": 0.45},
                runtime_seconds=120,
                samples_evaluated=1000
            ),
            EvalBenchmarkResult(
                name="mmlu",
                overall_score=0.35,
                task_scores={"accuracy": 0.35},
                runtime_seconds=150,
                samples_evaluated=1000
            )
        ]
    )
    
    ft_eval = EvaluationResult(
        model_name="gpt2-finetuned",
        benchmarks=[
            EvalBenchmarkResult(
                name="hellaswag",
                overall_score=0.52,
                task_scores={"accuracy": 0.52},
                runtime_seconds=125,
                samples_evaluated=1000
            ),
            EvalBenchmarkResult(
                name="mmlu",
                overall_score=0.38,
                task_scores={"accuracy": 0.38},
                runtime_seconds=155,
                samples_evaluated=1000
            )
        ]
    )
    
    # Create benchmark results
    base_result = BenchmarkResult(
        model_version=ModelVersion(
            version_id="base_001",
            model_path="gpt2",
            version_type=ModelVersionType.BASE,
            created_at=datetime.now()
        ),
        evaluation_results=base_eval,
        start_time=datetime.now(),
        end_time=datetime.now(),
        duration_seconds=300
    )
    
    ft_result = BenchmarkResult(
        model_version=ModelVersion(
            version_id="ft_001",
            model_path="gpt2-finetuned",
            version_type=ModelVersionType.FINE_TUNED,
            created_at=datetime.now()
        ),
        evaluation_results=ft_eval,
        start_time=datetime.now(),
        end_time=datetime.now(),
        duration_seconds=310
    )
    
    # Create comparison
    comparison = ComparisonResult(
        base_result=base_result,
        fine_tuned_result=ft_result,
        improvements={
            "hellaswag": {
                "base_score": 0.45,
                "ft_score": 0.52,
                "improvement": 0.07,
                "improvement_pct": 15.56
            },
            "mmlu": {
                "base_score": 0.35,
                "ft_score": 0.38,
                "improvement": 0.03,
                "improvement_pct": 8.57
            }
        },
        statistical_analysis={
            "summary": {
                "mean_improvement_pct": 12.07,
                "median_improvement_pct": 12.07,
                "std_improvement_pct": 3.49
            },
            "significance": {
                "hellaswag": {"is_significant": True, "confidence": 0.95},
                "mmlu": {"is_significant": True, "confidence": 0.95}
            }
        }
    )
    
    # Test visualization
    view = ComparisonView(comparison)
    
    # Create charts
    overview = view.create_overview_chart()
    improvements = view.create_improvement_chart()
    scatter = view.create_scatter_comparison()
    
    print("Charts created successfully")
    print(f"Statistical summary: {view.create_statistical_summary()}")
    print(f"Cost/benefit summary: {view.create_cost_benefit_summary()}")
    
    # Test Streamlit app (would need to run separately)
    # create_streamlit_app(comparison)