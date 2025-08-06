"""
Metrics Dashboard for Real-Time Training Visualization

This module provides a comprehensive dashboard for visualizing training metrics
in real-time, including loss curves, learning rate schedules, gradient norms,
and validation metrics.

Example:
    dashboard = MetricsDashboard()
    dashboard.add_training_run("gpt2_finetune", model_path)
    dashboard.update_metrics(run_id, step=100, metrics={...})
    dashboard.launch()
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Visualization libraries
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import dash
    import dash_bootstrap_components as dbc
    from dash import Input, Output, State, callback_context, dcc, html
    from dash.exceptions import PreventUpdate

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    step: int
    timestamp: datetime
    loss: float
    learning_rate: float
    gradient_norm: Optional[float] = None
    validation_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None
    tokens_per_second: Optional[float] = None
    memory_usage_gb: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "timestamp": self.timestamp.isoformat(),
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "gradient_norm": self.gradient_norm,
            "validation_loss": self.validation_loss,
            "validation_accuracy": self.validation_accuracy,
            "tokens_per_second": self.tokens_per_second,
            "memory_usage_gb": self.memory_usage_gb,
            **self.custom_metrics,
        }


@dataclass
class TrainingRun:
    """Information about a training run."""

    run_id: str
    model_name: str
    start_time: datetime
    config: Dict[str, Any]
    metrics_history: List[TrainingMetrics] = field(default_factory=list)
    status: str = "running"
    best_validation_loss: Optional[float] = None
    best_checkpoint_step: Optional[int] = None

    def get_latest_metrics(self) -> Optional[TrainingMetrics]:
        """Get the most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None

    def get_metrics_df(self) -> pd.DataFrame:
        """Convert metrics history to DataFrame."""
        if not self.metrics_history:
            return pd.DataFrame()

        data = [m.to_dict() for m in self.metrics_history]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df


class MetricsDashboard:
    """Real-time metrics dashboard for training visualization."""

    def __init__(
        self, update_interval_seconds: int = 5, max_history_length: int = 10000, port: int = 8050
    ):
        """Initialize metrics dashboard.

        Args:
            update_interval_seconds: Update interval for dashboard
            max_history_length: Maximum metrics history to keep
            port: Port for dashboard server
        """
        self.update_interval = update_interval_seconds
        self.max_history_length = max_history_length
        self.port = port

        # Training runs storage
        self.training_runs: Dict[str, TrainingRun] = {}
        self.active_run_id: Optional[str] = None

        # Metrics buffer for real-time updates
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=100))

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
                                    "🚀 Training Metrics Dashboard", className="text-center mb-4"
                                ),
                                html.Hr(),
                            ]
                        )
                    ]
                ),
                # Run selector
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Select Training Run:"),
                                dcc.Dropdown(
                                    id="run-selector",
                                    options=[],
                                    value=None,
                                    placeholder="Select a training run...",
                                ),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [dbc.Label("Status:"), html.Div(id="run-status", className="h4")],
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),
                # Metrics cards
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4("Current Loss", className="card-title"),
                                                html.H2(id="current-loss", children="--"),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4("Learning Rate", className="card-title"),
                                                html.H2(id="current-lr", children="--"),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4("Validation Loss", className="card-title"),
                                                html.H2(id="current-val-loss", children="--"),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4("Steps", className="card-title"),
                                                html.H2(id="current-step", children="--"),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=3,
                        ),
                    ],
                    className="mb-4",
                ),
                # Main charts
                dbc.Row(
                    [
                        dbc.Col(
                            [dcc.Graph(id="loss-chart", config={"displayModeBar": False})], width=6
                        ),
                        dbc.Col(
                            [dcc.Graph(id="lr-chart", config={"displayModeBar": False})], width=6
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [dcc.Graph(id="gradient-chart", config={"displayModeBar": False})],
                            width=6,
                        ),
                        dbc.Col(
                            [dcc.Graph(id="validation-chart", config={"displayModeBar": False})],
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),
                # Performance metrics
                dbc.Row(
                    [
                        dbc.Col(
                            [dcc.Graph(id="performance-chart", config={"displayModeBar": False})],
                            width=12,
                        )
                    ]
                ),
                # Auto-refresh interval
                dcc.Interval(
                    id="interval-component", interval=self.update_interval * 1000, n_intervals=0
                ),
                # Hidden div to store current data
                html.Div(id="hidden-data", style={"display": "none"}),
            ],
            fluid=True,
        )

        # Register callbacks
        self._register_callbacks()

    def _register_callbacks(self):
        """Register Dash callbacks."""

        @self.app.callback(
            [Output("run-selector", "options"), Output("run-selector", "value")],
            [Input("interval-component", "n_intervals")],
        )
        def update_run_selector(n):
            """Update run selector dropdown."""
            options = [
                {"label": f"{run_id} - {run.model_name}", "value": run_id}
                for run_id, run in self.training_runs.items()
            ]

            # Keep current selection or select the active run
            current_value = (
                callback_context.triggered[0]["prop_id"].split(".")[0]
                if callback_context.triggered
                else None
            )
            if current_value == "run-selector":
                raise PreventUpdate

            value = (
                self.active_run_id
                if self.active_run_id
                else (options[0]["value"] if options else None)
            )

            return options, value

        @self.app.callback(
            [
                Output("current-loss", "children"),
                Output("current-lr", "children"),
                Output("current-val-loss", "children"),
                Output("current-step", "children"),
                Output("run-status", "children"),
                Output("hidden-data", "children"),
            ],
            [Input("interval-component", "n_intervals"), Input("run-selector", "value")],
        )
        def update_metrics_cards(n, run_id):
            """Update metric cards with latest values."""
            if not run_id or run_id not in self.training_runs:
                return "--", "--", "--", "--", "No run selected", None

            run = self.training_runs[run_id]
            latest = run.get_latest_metrics()

            if not latest:
                return "--", "--", "--", "--", f"Status: {run.status}", None

            loss = f"{latest.loss:.4f}" if latest.loss is not None else "--"
            lr = f"{latest.learning_rate:.2e}" if latest.learning_rate is not None else "--"
            val_loss = (
                f"{latest.validation_loss:.4f}" if latest.validation_loss is not None else "--"
            )
            step = f"{latest.step:,}" if latest.step is not None else "--"

            # Status badge
            status_color = (
                "success"
                if run.status == "completed"
                else "warning"
                if run.status == "running"
                else "danger"
            )
            status = dbc.Badge(run.status.upper(), color=status_color, className="ml-2")

            # Store data for charts
            df = run.get_metrics_df()
            hidden_data = df.to_json(date_format="iso", orient="records") if not df.empty else None

            return loss, lr, val_loss, step, status, hidden_data

        @self.app.callback(
            [
                Output("loss-chart", "figure"),
                Output("lr-chart", "figure"),
                Output("gradient-chart", "figure"),
                Output("validation-chart", "figure"),
                Output("performance-chart", "figure"),
            ],
            [Input("hidden-data", "children")],
        )
        def update_charts(json_data):
            """Update all charts."""
            if not json_data:
                empty_fig = go.Figure()
                empty_fig.update_layout(title="No data available")
                return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

            df = pd.read_json(json_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Loss chart
            loss_fig = go.Figure()
            loss_fig.add_trace(
                go.Scatter(
                    x=df["step"],
                    y=df["loss"],
                    mode="lines",
                    name="Training Loss",
                    line=dict(color="blue", width=2),
                )
            )

            if "validation_loss" in df.columns and df["validation_loss"].notna().any():
                loss_fig.add_trace(
                    go.Scatter(
                        x=df[df["validation_loss"].notna()]["step"],
                        y=df[df["validation_loss"].notna()]["validation_loss"],
                        mode="lines+markers",
                        name="Validation Loss",
                        line=dict(color="red", width=2),
                    )
                )

            loss_fig.update_layout(
                title="Loss Curves",
                xaxis_title="Step",
                yaxis_title="Loss",
                hovermode="x unified",
                showlegend=True,
            )

            # Learning rate chart
            lr_fig = go.Figure()
            lr_fig.add_trace(
                go.Scatter(
                    x=df["step"],
                    y=df["learning_rate"],
                    mode="lines",
                    name="Learning Rate",
                    line=dict(color="green", width=2),
                )
            )
            lr_fig.update_layout(
                title="Learning Rate Schedule",
                xaxis_title="Step",
                yaxis_title="Learning Rate",
                yaxis_type="log",
            )

            # Gradient norm chart
            grad_fig = go.Figure()
            if "gradient_norm" in df.columns and df["gradient_norm"].notna().any():
                grad_fig.add_trace(
                    go.Scatter(
                        x=df[df["gradient_norm"].notna()]["step"],
                        y=df[df["gradient_norm"].notna()]["gradient_norm"],
                        mode="lines",
                        name="Gradient Norm",
                        line=dict(color="purple", width=2),
                    )
                )
                grad_fig.update_layout(
                    title="Gradient Norm", xaxis_title="Step", yaxis_title="Norm"
                )
            else:
                grad_fig.update_layout(title="Gradient Norm (No data)")

            # Validation metrics chart
            val_fig = go.Figure()
            if "validation_accuracy" in df.columns and df["validation_accuracy"].notna().any():
                val_fig.add_trace(
                    go.Scatter(
                        x=df[df["validation_accuracy"].notna()]["step"],
                        y=df[df["validation_accuracy"].notna()]["validation_accuracy"] * 100,
                        mode="lines+markers",
                        name="Validation Accuracy",
                        line=dict(color="orange", width=2),
                    )
                )
                val_fig.update_layout(
                    title="Validation Metrics", xaxis_title="Step", yaxis_title="Accuracy (%)"
                )
            else:
                val_fig.update_layout(title="Validation Metrics (No data)")

            # Performance chart
            perf_fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=("Tokens per Second", "Memory Usage"),
                shared_xaxes=True,
            )

            if "tokens_per_second" in df.columns and df["tokens_per_second"].notna().any():
                perf_fig.add_trace(
                    go.Scatter(
                        x=df[df["tokens_per_second"].notna()]["step"],
                        y=df[df["tokens_per_second"].notna()]["tokens_per_second"],
                        mode="lines",
                        name="Tokens/sec",
                        line=dict(color="cyan", width=2),
                    ),
                    row=1,
                    col=1,
                )

            if "memory_usage_gb" in df.columns and df["memory_usage_gb"].notna().any():
                perf_fig.add_trace(
                    go.Scatter(
                        x=df[df["memory_usage_gb"].notna()]["step"],
                        y=df[df["memory_usage_gb"].notna()]["memory_usage_gb"],
                        mode="lines",
                        name="Memory (GB)",
                        line=dict(color="magenta", width=2),
                    ),
                    row=2,
                    col=1,
                )

            perf_fig.update_layout(title="Performance Metrics", height=500, showlegend=True)
            perf_fig.update_xaxes(title_text="Step", row=2, col=1)

            return loss_fig, lr_fig, grad_fig, val_fig, perf_fig

    def add_training_run(self, run_id: str, model_name: str, config: Dict[str, Any]) -> TrainingRun:
        """Add a new training run.

        Args:
            run_id: Unique identifier for the run
            model_name: Name of the model being trained
            config: Training configuration

        Returns:
            TrainingRun object
        """
        run = TrainingRun(
            run_id=run_id, model_name=model_name, start_time=datetime.now(), config=config
        )

        self.training_runs[run_id] = run
        self.active_run_id = run_id

        logger.info(f"Added training run: {run_id} for model {model_name}")

        return run

    def update_metrics(self, run_id: str, step: int, loss: float, learning_rate: float, **kwargs):
        """Update metrics for a training run.

        Args:
            run_id: Run identifier
            step: Training step
            loss: Training loss
            learning_rate: Current learning rate
            **kwargs: Additional metrics
        """
        if run_id not in self.training_runs:
            logger.warning(f"Unknown run_id: {run_id}")
            return

        run = self.training_runs[run_id]

        # Create metrics object
        metrics = TrainingMetrics(
            step=step,
            timestamp=datetime.now(),
            loss=loss,
            learning_rate=learning_rate,
            gradient_norm=kwargs.get("gradient_norm"),
            validation_loss=kwargs.get("validation_loss"),
            validation_accuracy=kwargs.get("validation_accuracy"),
            tokens_per_second=kwargs.get("tokens_per_second"),
            memory_usage_gb=kwargs.get("memory_usage_gb"),
            custom_metrics={
                k: v
                for k, v in kwargs.items()
                if k
                not in [
                    "gradient_norm",
                    "validation_loss",
                    "validation_accuracy",
                    "tokens_per_second",
                    "memory_usage_gb",
                ]
            },
        )

        # Add to history
        run.metrics_history.append(metrics)

        # Trim history if needed
        if len(run.metrics_history) > self.max_history_length:
            run.metrics_history = run.metrics_history[-self.max_history_length :]

        # Update best validation loss
        if metrics.validation_loss is not None:
            if (
                run.best_validation_loss is None
                or metrics.validation_loss < run.best_validation_loss
            ):
                run.best_validation_loss = metrics.validation_loss
                run.best_checkpoint_step = step

        # Add to buffer for real-time updates
        self.metrics_buffer[run_id].append(metrics)

    def complete_run(self, run_id: str):
        """Mark a run as completed.

        Args:
            run_id: Run identifier
        """
        if run_id in self.training_runs:
            self.training_runs[run_id].status = "completed"
            logger.info(f"Completed training run: {run_id}")

    def export_metrics(self, run_id: str, format: str = "csv") -> str | bytes:
        """Export metrics for a run.

        Args:
            run_id: Run identifier
            format: Export format ('csv', 'json', 'parquet')

        Returns:
            Exported data
        """
        if run_id not in self.training_runs:
            raise ValueError(f"Unknown run_id: {run_id}")

        run = self.training_runs[run_id]
        df = run.get_metrics_df()

        if format == "csv":
            return df.to_csv(index=False)
        elif format == "json":
            return df.to_json(orient="records", date_format="iso")
        elif format == "parquet":
            import io

            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            return buffer.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def launch(self):
        """Launch the dashboard server."""
        if DASH_AVAILABLE:
            logger.info(f"Launching metrics dashboard on port {self.port}")
            self.app.run_server(host="0.0.0.0", port=self.port, debug=False)
        else:
            logger.error("Dash is not installed. Cannot launch dashboard.")

    def create_streamlit_app(self):
        """Create Streamlit version of the dashboard."""
        if not STREAMLIT_AVAILABLE:
            raise ImportError("Streamlit is not installed")

        st.set_page_config(page_title="Training Metrics Dashboard", page_icon="🚀", layout="wide")

        st.title("🚀 Training Metrics Dashboard")

        # Run selector
        col1, col2 = st.columns(2)

        with col1:
            run_ids = list(self.training_runs.keys())
            selected_run = st.selectbox(
                "Select Training Run",
                run_ids,
                index=run_ids.index(self.active_run_id) if self.active_run_id in run_ids else 0,
            )

        with col2:
            if selected_run:
                run = self.training_runs[selected_run]
                status_color = (
                    "🟢" if run.status == "completed" else "🟡" if run.status == "running" else "🔴"
                )
                st.metric("Status", f"{status_color} {run.status.upper()}")

        if selected_run:
            run = self.training_runs[selected_run]
            latest = run.get_latest_metrics()

            if latest:
                # Metrics cards
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Current Loss", f"{latest.loss:.4f}" if latest.loss else "--", delta=None
                    )

                with col2:
                    st.metric(
                        "Learning Rate",
                        f"{latest.learning_rate:.2e}" if latest.learning_rate else "--",
                    )

                with col3:
                    st.metric(
                        "Validation Loss",
                        f"{latest.validation_loss:.4f}" if latest.validation_loss else "--",
                    )

                with col4:
                    st.metric("Steps", f"{latest.step:,}" if latest.step else "--")

                # Get DataFrame
                df = run.get_metrics_df()

                if not df.empty:
                    # Loss curves
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Loss Curves")
                        loss_data = pd.DataFrame({"Step": df["step"], "Training Loss": df["loss"]})

                        if "validation_loss" in df.columns:
                            loss_data["Validation Loss"] = df["validation_loss"]

                        st.line_chart(loss_data.set_index("Step"))

                    with col2:
                        st.subheader("Learning Rate Schedule")
                        lr_data = pd.DataFrame(
                            {"Step": df["step"], "Learning Rate": df["learning_rate"]}
                        )
                        st.line_chart(lr_data.set_index("Step"))

                    # Additional metrics
                    if "gradient_norm" in df.columns:
                        st.subheader("Gradient Norm")
                        grad_data = pd.DataFrame(
                            {"Step": df["step"], "Gradient Norm": df["gradient_norm"]}
                        )
                        st.line_chart(grad_data.set_index("Step"))

                    # Export button
                    st.subheader("Export Metrics")
                    col1, col2 = st.columns(2)

                    with col1:
                        format = st.selectbox("Format", ["csv", "json", "parquet"])

                    with col2:
                        if st.button("Export"):
                            data = self.export_metrics(selected_run, format)

                            if format == "csv":
                                st.download_button(
                                    "Download CSV", data, f"{selected_run}_metrics.csv", "text/csv"
                                )
                            elif format == "json":
                                st.download_button(
                                    "Download JSON",
                                    data,
                                    f"{selected_run}_metrics.json",
                                    "application/json",
                                )
                            elif format == "parquet":
                                st.download_button(
                                    "Download Parquet",
                                    data,
                                    f"{selected_run}_metrics.parquet",
                                    "application/octet-stream",
                                )

        # Auto-refresh
        if st.checkbox("Auto-refresh", value=True):
            time.sleep(self.update_interval)
            st.rerun()


# Example usage
if __name__ == "__main__":
    # Create dashboard
    dashboard = MetricsDashboard(update_interval_seconds=2)

    # Add sample training run
    run_id = dashboard.add_training_run(
        run_id="gpt2_finetune_001",
        model_name="gpt2",
        config={"batch_size": 32, "learning_rate": 5e-5, "num_epochs": 3, "warmup_steps": 500},
    ).run_id

    # Simulate training updates
    import random

    for step in range(0, 1000, 10):
        dashboard.update_metrics(
            run_id=run_id,
            step=step,
            loss=3.5 * np.exp(-step / 500) + random.uniform(-0.1, 0.1),
            learning_rate=5e-5 * (1 - step / 1000),
            gradient_norm=random.uniform(0.5, 2.0),
            validation_loss=3.2 * np.exp(-step / 600) + random.uniform(-0.1, 0.1)
            if step % 50 == 0
            else None,
            validation_accuracy=min(0.95, step / 1000 + random.uniform(-0.05, 0.05))
            if step % 50 == 0
            else None,
            tokens_per_second=random.uniform(1000, 1500),
            memory_usage_gb=random.uniform(10, 12),
        )

    dashboard.complete_run(run_id)

    # Launch dashboard
    if DASH_AVAILABLE:
        dashboard.launch()
    elif STREAMLIT_AVAILABLE:
        dashboard.create_streamlit_app()
    else:
        print("No dashboard framework available. Install dash or streamlit.")
