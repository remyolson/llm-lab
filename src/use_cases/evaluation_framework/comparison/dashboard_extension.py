"""
Performance Metrics Dashboard Extension

This module extends the existing monitoring dashboard from Task 9 with
evaluation-specific views including fine-tuning timeline view, performance
delta charts, regression analysis, multi-model comparison matrices,
drill-down capabilities, and real-time metric streaming.

Example:
    dashboard = EvaluationDashboard()
    dashboard.add_comparison_results(comparison_results)
    dashboard.launch()
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import asyncio
import json
import logging
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px

# Import plotly for visualizations
import plotly.graph_objects as go
import websockets
from plotly.subplots import make_subplots

# Import existing dashboard components
try:
    from ...monitoring.dashboard.app import app as base_app
    from ...monitoring.dashboard.data_service import DataService
    from ...monitoring.dashboard.models import MetricData

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    base_app = None

# Import evaluation components
from ..benchmark_runner import BenchmarkResult, ComparisonResult

# Import UI frameworks
try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import dash
    import dash_bootstrap_components as dbc
    from dash import Input, Output, State, dcc, html

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

logger = logging.getLogger(__name__)


class EvaluationDashboard:
    """Extended dashboard for evaluation metrics and comparisons."""

    def __init__(
        self, data_dir: str = "./evaluation_data", update_interval: int = 5, port: int = 8050
    ):
        """Initialize evaluation dashboard.

        Args:
            data_dir: Directory for storing evaluation data
            update_interval: Update interval in seconds
            port: Port for dashboard server
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.update_interval = update_interval
        self.port = port

        # Data storage
        self.comparison_results = []
        self.timeline_data = defaultdict(list)
        self.model_registry = {}
        self.live_metrics = deque(maxlen=1000)

        # Initialize base dashboard if available
        if MONITORING_AVAILABLE:
            self.data_service = DataService()
        else:
            self.data_service = None

        # Initialize Dash app if available
        if DASH_AVAILABLE:
            self._init_dash_app()

    def _init_dash_app(self):
        """Initialize Dash application."""
        self.app = dash.Dash(
            __name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True
        )

        # Define layout
        self.app.layout = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1(
                                    "Evaluation Performance Dashboard", className="text-center mb-4"
                                ),
                                html.Hr(),
                            ]
                        )
                    ]
                ),
                # Navigation tabs
                dbc.Tabs(
                    [
                        dbc.Tab(label="Timeline View", tab_id="timeline"),
                        dbc.Tab(label="Model Comparison", tab_id="comparison"),
                        dbc.Tab(label="Performance Delta", tab_id="delta"),
                        dbc.Tab(label="Regression Analysis", tab_id="regression"),
                        dbc.Tab(label="Live Metrics", tab_id="live"),
                    ],
                    id="tabs",
                    active_tab="timeline",
                ),
                # Content area
                html.Div(id="tab-content", className="mt-4"),
                # Update interval
                dcc.Interval(
                    id="interval-component", interval=self.update_interval * 1000, n_intervals=0
                ),
            ],
            fluid=True,
        )

        # Define callbacks
        self._setup_callbacks()

    def _setup_callbacks(self):
        """Setup Dash callbacks."""

        @self.app.callback(
            Output("tab-content", "children"),
            Input("tabs", "active_tab"),
            Input("interval-component", "n_intervals"),
        )
        def render_tab_content(active_tab, n_intervals):
            """Render content based on active tab."""
            if active_tab == "timeline":
                return self._render_timeline_view()
            elif active_tab == "comparison":
                return self._render_comparison_view()
            elif active_tab == "delta":
                return self._render_delta_view()
            elif active_tab == "regression":
                return self._render_regression_view()
            elif active_tab == "live":
                return self._render_live_metrics()
            else:
                return html.Div("Select a tab")

    def add_comparison_results(self, comparison: ComparisonResult):
        """Add comparison results to dashboard.

        Args:
            comparison: Comparison results to add
        """
        self.comparison_results.append(comparison)

        # Update timeline data
        timestamp = datetime.now()

        # Extract key metrics
        for benchmark, improvement in comparison.improvements.items():
            if isinstance(improvement, dict):
                self.timeline_data[benchmark].append(
                    {
                        "timestamp": timestamp,
                        "base_score": improvement.get("base_score", 0),
                        "ft_score": improvement.get("ft_score", 0),
                        "improvement_pct": improvement.get("improvement_pct", 0),
                    }
                )

        # Update model registry
        base_version = comparison.base_result.model_version
        ft_version = comparison.fine_tuned_result.model_version

        self.model_registry[base_version.version_id] = base_version
        self.model_registry[ft_version.version_id] = ft_version

        # Save to disk
        self._save_comparison(comparison)

    def _save_comparison(self, comparison: ComparisonResult):
        """Save comparison results to disk.

        Args:
            comparison: Comparison to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.data_dir / f"comparison_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(comparison.to_dict(), f, indent=2)

    def _render_timeline_view(self) -> html.Div:
        """Render fine-tuning timeline view."""
        if not self.timeline_data:
            return html.Div("No timeline data available")

        # Create timeline chart
        fig = make_subplots(
            rows=len(self.timeline_data),
            cols=1,
            subplot_titles=list(self.timeline_data.keys()),
            shared_xaxes=True,
        )

        for idx, (benchmark, data) in enumerate(self.timeline_data.items(), 1):
            df = pd.DataFrame(data)

            # Add base score line
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["base_score"],
                    mode="lines+markers",
                    name=f"{benchmark} - Base",
                    line=dict(dash="dash"),
                ),
                row=idx,
                col=1,
            )

            # Add fine-tuned score line
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["ft_score"],
                    mode="lines+markers",
                    name=f"{benchmark} - Fine-tuned",
                ),
                row=idx,
                col=1,
            )

        fig.update_layout(
            height=300 * len(self.timeline_data),
            title="Fine-Tuning Performance Timeline",
            showlegend=True,
        )

        return dcc.Graph(figure=fig)

    def _render_comparison_view(self) -> html.Div:
        """Render multi-model comparison matrix."""
        if not self.comparison_results:
            return html.Div("No comparison data available")

        # Create comparison matrix
        models = list(self.model_registry.keys())
        benchmarks = set()

        # Collect all benchmarks
        for comp in self.comparison_results:
            benchmarks.update(comp.improvements.keys())

        benchmarks = sorted(list(benchmarks))

        # Create matrix data
        matrix_data = []
        for model_id in models:
            row = {"Model": model_id}

            # Find results for this model
            for comp in self.comparison_results:
                if comp.fine_tuned_result.model_version.version_id == model_id:
                    for benchmark in benchmarks:
                        if benchmark in comp.improvements:
                            imp = comp.improvements[benchmark]
                            if isinstance(imp, dict):
                                row[benchmark] = imp.get("ft_score", 0)
                    break

            matrix_data.append(row)

        # Create heatmap
        df = pd.DataFrame(matrix_data)
        df = df.set_index("Model")

        fig = go.Figure(
            data=go.Heatmap(
                z=df.values,
                x=df.columns,
                y=df.index,
                colorscale="RdYlGn",
                text=df.values.round(3),
                texttemplate="%{text}",
                textfont={"size": 10},
            )
        )

        fig.update_layout(
            title="Model Performance Comparison Matrix",
            xaxis_title="Benchmarks",
            yaxis_title="Models",
            height=400 + len(models) * 30,
        )

        return dcc.Graph(figure=fig)

    def _render_delta_view(self) -> html.Div:
        """Render performance delta charts."""
        if not self.comparison_results:
            return html.Div("No comparison data available")

        # Get latest comparison
        latest_comp = self.comparison_results[-1]

        # Create waterfall chart for performance deltas
        benchmarks = []
        deltas = []

        for benchmark, imp in latest_comp.improvements.items():
            if isinstance(imp, dict):
                benchmarks.append(benchmark)
                deltas.append(imp.get("improvement", 0))

        # Sort by delta
        sorted_indices = np.argsort(deltas)[::-1]
        benchmarks = [benchmarks[i] for i in sorted_indices]
        deltas = [deltas[i] for i in sorted_indices]

        # Create waterfall chart
        fig = go.Figure(
            go.Waterfall(
                x=benchmarks,
                y=deltas,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                text=[f"{d:.4f}" for d in deltas],
                textposition="outside",
            )
        )

        fig.update_layout(
            title="Performance Delta Waterfall",
            xaxis_title="Benchmarks",
            yaxis_title="Score Delta",
            showlegend=False,
            height=500,
        )

        return dcc.Graph(figure=fig)

    def _render_regression_view(self) -> html.Div:
        """Render regression analysis view."""
        if not self.comparison_results:
            return html.Div("No comparison data available")

        # Collect all regressions
        all_regressions = []

        for comp in self.comparison_results:
            for reg in comp.regressions:
                all_regressions.append(
                    {
                        "Model": comp.fine_tuned_result.model_version.version_id,
                        "Benchmark": reg["benchmark"],
                        "Regression %": reg["regression_pct"],
                        "Base Score": reg["base_score"],
                        "FT Score": reg["ft_score"],
                    }
                )

        if not all_regressions:
            return html.Div(dbc.Alert("No regressions detected! 🎉", color="success"))

        # Create regression table
        df = pd.DataFrame(all_regressions)

        table = dbc.Table.from_dataframe(
            df, striped=True, bordered=True, hover=True, responsive=True, className="mt-3"
        )

        # Create regression scatter plot
        fig = px.scatter(
            df,
            x="Base Score",
            y="FT Score",
            color="Regression %",
            size=df["Regression %"].abs(),
            hover_data=["Model", "Benchmark"],
            color_continuous_scale="Reds_r",
            title="Regression Analysis",
        )

        # Add y=x line
        max_val = max(df["Base Score"].max(), df["FT Score"].max())
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                showlegend=False,
            )
        )

        return html.Div([html.H3("Detected Regressions"), table, dcc.Graph(figure=fig)])

    def _render_live_metrics(self) -> html.Div:
        """Render live metrics streaming view."""
        # Create live updating chart
        fig = go.Figure()

        if self.live_metrics:
            df = pd.DataFrame(list(self.live_metrics))

            for column in df.columns:
                if column != "timestamp":
                    fig.add_trace(
                        go.Scatter(x=df["timestamp"], y=df[column], mode="lines", name=column)
                    )

        fig.update_layout(
            title="Live Evaluation Metrics", xaxis_title="Time", yaxis_title="Value", height=500
        )

        return html.Div(
            [
                dcc.Graph(id="live-graph", figure=fig),
                html.Div(id="live-update-text", className="mt-2"),
            ]
        )

    def add_live_metric(self, metric_name: str, value: float, metadata: Optional[Dict] = None):
        """Add live metric data point.

        Args:
            metric_name: Name of metric
            value: Metric value
            metadata: Optional metadata
        """
        self.live_metrics.append(
            {"timestamp": datetime.now(), metric_name: value, **(metadata or {})}
        )

    def create_drill_down_view(self, model_id: str, benchmark: str) -> Dict[str | Any]:
        """Create detailed drill-down view for specific model/benchmark.

        Args:
            model_id: Model version ID
            benchmark: Benchmark name

        Returns:
            Drill-down data and visualizations
        """
        drill_down_data = {
            "model_id": model_id,
            "benchmark": benchmark,
            "detailed_metrics": {},
            "visualizations": [],
        }

        # Find relevant comparison
        for comp in self.comparison_results:
            if (
                comp.fine_tuned_result.model_version.version_id == model_id
                or comp.base_result.model_version.version_id == model_id
            ):
                # Extract detailed metrics
                for bench_result in comp.fine_tuned_result.evaluation_results.benchmarks:
                    if bench_result.name == benchmark:
                        drill_down_data["detailed_metrics"] = {
                            "overall_score": bench_result.overall_score,
                            "task_scores": bench_result.task_scores,
                            "runtime": bench_result.runtime_seconds,
                            "samples": bench_result.samples_evaluated,
                        }

                        # Create task score breakdown chart
                        if bench_result.task_scores:
                            fig = px.bar(
                                x=list(bench_result.task_scores.keys()),
                                y=list(bench_result.task_scores.values()),
                                title=f"{benchmark} Task Score Breakdown",
                            )
                            drill_down_data["visualizations"].append(fig)

                        break
                break

        return drill_down_data

    def export_dashboard_data(self, format: str = "json") -> str | bytes:
        """Export dashboard data in specified format.

        Args:
            format: Export format (json, csv, excel)

        Returns:
            Exported data
        """
        # Prepare export data
        export_data = {
            "comparison_results": [comp.to_dict() for comp in self.comparison_results],
            "timeline_data": dict(self.timeline_data),
            "model_registry": {k: v.to_dict() for k, v in self.model_registry.items()},
            "export_timestamp": datetime.now().isoformat(),
        }

        if format == "json":
            return json.dumps(export_data, indent=2)

        elif format == "csv":
            # Convert to DataFrame
            rows = []
            for comp in self.comparison_results:
                for benchmark, imp in comp.improvements.items():
                    if isinstance(imp, dict):
                        rows.append(
                            {
                                "base_model": comp.base_result.model_version.version_id,
                                "ft_model": comp.fine_tuned_result.model_version.version_id,
                                "benchmark": benchmark,
                                **imp,
                            }
                        )

            df = pd.DataFrame(rows)
            return df.to_csv(index=False)

        elif format == "excel":
            # Create Excel file with multiple sheets
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                # Comparison summary
                summary_df = pd.DataFrame(
                    [
                        {
                            "base_model": comp.base_result.model_version.version_id,
                            "ft_model": comp.fine_tuned_result.model_version.version_id,
                            "num_benchmarks": len(comp.improvements),
                            "avg_improvement": comp.statistical_analysis.get("summary", {}).get(
                                "mean_improvement_pct", 0
                            ),
                        }
                        for comp in self.comparison_results
                    ]
                )
                summary_df.to_excel(writer, sheet_name="Summary", index=False)

                # Detailed results
                detail_rows = []
                for comp in self.comparison_results:
                    for benchmark, imp in comp.improvements.items():
                        if isinstance(imp, dict):
                            detail_rows.append(
                                {
                                    "base_model": comp.base_result.model_version.version_id,
                                    "ft_model": comp.fine_tuned_result.model_version.version_id,
                                    "benchmark": benchmark,
                                    **imp,
                                }
                            )

                detail_df = pd.DataFrame(detail_rows)
                detail_df.to_excel(writer, sheet_name="Detailed Results", index=False)

            return output.getvalue()

        else:
            raise ValueError(f"Unsupported format: {format}")

    def launch(self):
        """Launch the dashboard server."""
        if DASH_AVAILABLE:
            logger.info(f"Launching evaluation dashboard on port {self.port}")
            self.app.run_server(host="0.0.0.0", port=self.port, debug=False)
        else:
            logger.error("Dash is not installed. Cannot launch dashboard.")

    async def start_websocket_server(self, host: str = "localhost", port: int = 8765):
        """Start WebSocket server for real-time updates.

        Args:
            host: WebSocket host
            port: WebSocket port
        """

        async def handle_client(websocket, path):
            """Handle WebSocket client connections."""
            logger.info(f"Client connected: {websocket.remote_address}")

            try:
                while True:
                    # Send latest metrics
                    if self.live_metrics:
                        latest = self.live_metrics[-1]
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "metric_update",
                                    "data": {k: v for k, v in latest.items() if k != "timestamp"},
                                    "timestamp": latest["timestamp"].isoformat(),
                                }
                            )
                        )

                    await asyncio.sleep(self.update_interval)

            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Client disconnected: {websocket.remote_address}")

        logger.info(f"Starting WebSocket server on {host}:{port}")
        await websockets.serve(handle_client, host, port)


def create_streamlit_extension(dashboard: EvaluationDashboard):
    """Create Streamlit extension for evaluation dashboard.

    Args:
        dashboard: EvaluationDashboard instance
    """
    if not STREAMLIT_AVAILABLE:
        raise ImportError("Streamlit is not installed")

    st.set_page_config(page_title="Evaluation Performance Dashboard", page_icon="📈", layout="wide")

    st.title("🚀 Fine-Tuning Evaluation Performance Dashboard")

    # Sidebar navigation
    view_mode = st.sidebar.selectbox(
        "Select View",
        [
            "Timeline",
            "Model Comparison",
            "Performance Delta",
            "Regression Analysis",
            "Live Metrics",
        ],
    )

    # Main content based on view mode
    if view_mode == "Timeline":
        st.header("Fine-Tuning Timeline View")

        if dashboard.timeline_data:
            # Create timeline charts for each benchmark
            for benchmark, data in dashboard.timeline_data.items():
                st.subheader(f"{benchmark} Performance Over Time")

                df = pd.DataFrame(data)

                # Create line chart
                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df["base_score"],
                        mode="lines+markers",
                        name="Base Model",
                        line=dict(dash="dash"),
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df["ft_score"],
                        mode="lines+markers",
                        name="Fine-Tuned Model",
                    )
                )

                fig.update_layout(xaxis_title="Time", yaxis_title="Score", height=400)

                st.plotly_chart(fig, use_container_width=True)

                # Show improvement trend
                st.metric(
                    "Latest Improvement",
                    f"{df['improvement_pct'].iloc[-1]:.2f}%",
                    delta=f"{df['improvement_pct'].iloc[-1] - df['improvement_pct'].iloc[0]:.2f}%",
                )
        else:
            st.info("No timeline data available yet")

    elif view_mode == "Model Comparison":
        st.header("Multi-Model Comparison Matrix")

        if dashboard.comparison_results:
            # Model selection
            model_ids = list(dashboard.model_registry.keys())
            selected_models = st.multiselect(
                "Select models to compare",
                model_ids,
                default=model_ids[:5],  # Show first 5 by default
            )

            # Benchmark selection
            all_benchmarks = set()
            for comp in dashboard.comparison_results:
                all_benchmarks.update(comp.improvements.keys())

            selected_benchmarks = st.multiselect(
                "Select benchmarks",
                sorted(list(all_benchmarks)),
                default=sorted(list(all_benchmarks)),
            )

            # Create comparison matrix
            if selected_models and selected_benchmarks:
                matrix_data = []

                for model_id in selected_models:
                    row = {"Model": model_id}

                    for comp in dashboard.comparison_results:
                        if comp.fine_tuned_result.model_version.version_id == model_id:
                            for benchmark in selected_benchmarks:
                                if benchmark in comp.improvements:
                                    imp = comp.improvements[benchmark]
                                    if isinstance(imp, dict):
                                        row[benchmark] = imp.get("ft_score", 0)
                            break

                    matrix_data.append(row)

                df = pd.DataFrame(matrix_data)

                # Display as heatmap
                if len(df) > 0:
                    df_display = df.set_index("Model")

                    fig = px.imshow(
                        df_display,
                        labels=dict(x="Benchmark", y="Model", color="Score"),
                        x=df_display.columns,
                        y=df_display.index,
                        color_continuous_scale="RdYlGn",
                        aspect="auto",
                    )

                    fig.update_traces(text=df_display.values.round(3), texttemplate="%{text}")
                    st.plotly_chart(fig, use_container_width=True)

                    # Also show as table
                    st.dataframe(df_display.style.background_gradient(axis=None))
        else:
            st.info("No comparison data available")

    elif view_mode == "Performance Delta":
        st.header("Performance Delta Analysis")

        if dashboard.comparison_results:
            # Select comparison
            comparison_idx = st.selectbox(
                "Select comparison",
                range(len(dashboard.comparison_results)),
                format_func=lambda x: f"Comparison {x + 1} - {dashboard.comparison_results[x].fine_tuned_result.model_version.version_id}",
            )

            comp = dashboard.comparison_results[comparison_idx]

            # Create delta visualization
            benchmarks = []
            deltas = []
            colors = []

            for benchmark, imp in comp.improvements.items():
                if isinstance(imp, dict):
                    benchmarks.append(benchmark)
                    delta = imp.get("improvement_pct", 0)
                    deltas.append(delta)
                    colors.append("green" if delta > 0 else "red")

            # Sort by delta
            sorted_indices = sorted(range(len(deltas)), key=lambda i: deltas[i], reverse=True)
            benchmarks = [benchmarks[i] for i in sorted_indices]
            deltas = [deltas[i] for i in sorted_indices]
            colors = [colors[i] for i in sorted_indices]

            # Create bar chart
            fig = go.Figure(
                go.Bar(
                    x=deltas,
                    y=benchmarks,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{d:.2f}%" for d in deltas],
                    textposition="auto",
                )
            )

            fig.add_vline(x=0, line_dash="dash", line_color="gray")

            fig.update_layout(
                title="Performance Delta by Benchmark",
                xaxis_title="Improvement (%)",
                yaxis_title="Benchmark",
                height=400 + len(benchmarks) * 30,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Average Improvement", f"{np.mean(deltas):.2f}%")
            with col2:
                st.metric("Best Improvement", f"{max(deltas):.2f}%")
            with col3:
                st.metric("Worst Change", f"{min(deltas):.2f}%")
            with col4:
                st.metric("Improved Benchmarks", f"{sum(1 for d in deltas if d > 0)}/{len(deltas)}")
        else:
            st.info("No comparison data available")

    elif view_mode == "Regression Analysis":
        st.header("Regression Detection and Analysis")

        all_regressions = []

        for comp in dashboard.comparison_results:
            for reg in comp.regressions:
                all_regressions.append(
                    {
                        "Model": comp.fine_tuned_result.model_version.version_id,
                        "Benchmark": reg["benchmark"],
                        "Regression %": reg["regression_pct"],
                        "Base Score": reg["base_score"],
                        "FT Score": reg["ft_score"],
                    }
                )

        if all_regressions:
            df = pd.DataFrame(all_regressions)

            # Show regression table
            st.error(f"⚠️ {len(all_regressions)} regressions detected")
            st.dataframe(df.style.highlight_min(subset=["Regression %"], color="lightcoral"))

            # Regression visualization
            fig = px.scatter(
                df,
                x="Base Score",
                y="FT Score",
                color="Regression %",
                size=df["Regression %"].abs(),
                hover_data=["Model", "Benchmark"],
                color_continuous_scale="Reds_r",
            )

            # Add y=x line
            max_val = max(df["Base Score"].max(), df["FT Score"].max())
            fig.add_trace(
                go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode="lines",
                    line=dict(dash="dash", color="gray"),
                    showlegend=False,
                )
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No regressions detected!")

    elif view_mode == "Live Metrics":
        st.header("Live Evaluation Metrics")

        # Create placeholder for live updates
        metric_placeholder = st.empty()
        chart_placeholder = st.empty()

        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh", value=True)

        if auto_refresh:
            # Display latest metrics
            if dashboard.live_metrics:
                latest_metrics = list(dashboard.live_metrics)[-50:]  # Last 50 points
                df = pd.DataFrame(latest_metrics)

                # Create line chart
                fig = go.Figure()

                for column in df.columns:
                    if column != "timestamp":
                        fig.add_trace(
                            go.Scatter(x=df["timestamp"], y=df[column], mode="lines", name=column)
                        )

                fig.update_layout(
                    title="Real-time Metrics", xaxis_title="Time", yaxis_title="Value", height=500
                )

                chart_placeholder.plotly_chart(fig, use_container_width=True)

                # Show latest values
                if len(dashboard.live_metrics) > 0:
                    latest = dashboard.live_metrics[-1]
                    cols = st.columns(len(latest) - 1)  # Exclude timestamp

                    for idx, (key, value) in enumerate(latest.items()):
                        if key != "timestamp":
                            with cols[idx % len(cols)]:
                                st.metric(key, f"{value:.4f}")
            else:
                st.info("No live metrics available")

            # Add refresh button
            if st.button("Refresh Now"):
                st.experimental_rerun()

    # Export options in sidebar
    st.sidebar.header("Export Options")

    export_format = st.sidebar.selectbox("Export Format", ["JSON", "CSV", "Excel"])

    if st.sidebar.button("Export Dashboard Data"):
        try:
            data = dashboard.export_dashboard_data(export_format.lower())

            if export_format == "JSON":
                st.sidebar.download_button(
                    "Download JSON", data, "evaluation_dashboard.json", "application/json"
                )
            elif export_format == "CSV":
                st.sidebar.download_button(
                    "Download CSV", data, "evaluation_dashboard.csv", "text/csv"
                )
            elif export_format == "Excel":
                st.sidebar.download_button(
                    "Download Excel", data, "evaluation_dashboard.xlsx", "application/vnd.ms-excel"
                )
        except Exception as e:
            st.sidebar.error(f"Export failed: {e}")


# Example usage
if __name__ == "__main__":
    from io import BytesIO

    # Create dashboard
    dashboard = EvaluationDashboard()

    # Add sample data
    from ..benchmark_runner import BenchmarkResult, ComparisonResult, ModelVersion, ModelVersionType
    from ..fine_tuning.evaluation.suite import (
        BenchmarkResult as EvalBenchmarkResult,
        EvaluationResult,
    )

    # Create sample comparison
    base_eval = EvaluationResult(
        model_name="gpt2",
        benchmarks=[
            EvalBenchmarkResult(
                name="hellaswag",
                overall_score=0.45,
                task_scores={"accuracy": 0.45},
                runtime_seconds=120,
                samples_evaluated=1000,
            )
        ],
    )

    ft_eval = EvaluationResult(
        model_name="gpt2-finetuned",
        benchmarks=[
            EvalBenchmarkResult(
                name="hellaswag",
                overall_score=0.52,
                task_scores={"accuracy": 0.52},
                runtime_seconds=125,
                samples_evaluated=1000,
            )
        ],
    )

    comparison = ComparisonResult(
        base_result=BenchmarkResult(
            model_version=ModelVersion(
                version_id="base_001",
                model_path="gpt2",
                version_type=ModelVersionType.BASE,
                created_at=datetime.now(),
            ),
            evaluation_results=base_eval,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=300,
        ),
        fine_tuned_result=BenchmarkResult(
            model_version=ModelVersion(
                version_id="ft_001",
                model_path="gpt2-finetuned",
                version_type=ModelVersionType.FINE_TUNED,
                created_at=datetime.now(),
            ),
            evaluation_results=ft_eval,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=310,
        ),
        improvements={
            "hellaswag": {
                "base_score": 0.45,
                "ft_score": 0.52,
                "improvement": 0.07,
                "improvement_pct": 15.56,
            }
        },
    )

    dashboard.add_comparison_results(comparison)

    # Add live metrics
    for i in range(10):
        dashboard.add_live_metric("loss", 2.5 - i * 0.1)
        dashboard.add_live_metric("accuracy", 0.7 + i * 0.02)

    print("Dashboard initialized with sample data")

    # Launch dashboard
    # dashboard.launch()
