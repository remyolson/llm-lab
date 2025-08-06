"""Interactive dashboard for model interpretability visualization."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from dash import Input, Output, State, dcc, html

logger = logging.getLogger(__name__)


class DashboardManager:
    """Manages interactive dashboard for interpretability visualizations."""

    def __init__(self, port: int = 8050, host: str = "127.0.0.1", theme: str = "bootstrap"):
        """
        Initialize the dashboard manager.

        Args:
            port: Port to run dashboard on
            host: Host address
            theme: Dashboard theme
        """
        self.port = port
        self.host = host
        self.theme = theme
        self.app = None
        self.data_store = {}
        self._init_app()

    def _init_app(self):
        """Initialize Dash application."""
        # Use bootstrap theme
        external_stylesheets = [dbc.themes.BOOTSTRAP]

        self.app = dash.Dash(
            __name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True
        )

        # Set up layout
        self._setup_layout()

        # Register callbacks
        self._register_callbacks()

    def _setup_layout(self):
        """Set up dashboard layout."""
        self.app.layout = dbc.Container(
            [
                # Header
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1(
                                    "LLM Interpretability Dashboard", className="text-center mb-4"
                                ),
                                html.Hr(),
                            ]
                        )
                    ]
                ),
                # Navigation tabs
                dbc.Tabs(
                    id="main-tabs",
                    children=[
                        dbc.Tab(label="Attention Analysis", tab_id="attention-tab"),
                        dbc.Tab(label="Gradient Analysis", tab_id="gradient-tab"),
                        dbc.Tab(label="Activation Analysis", tab_id="activation-tab"),
                        dbc.Tab(label="Model Overview", tab_id="overview-tab"),
                    ],
                    active_tab="attention-tab",
                ),
                # Tab content
                html.Div(id="tab-content", className="mt-4"),
                # Hidden stores for data
                dcc.Store(id="attention-data-store"),
                dcc.Store(id="gradient-data-store"),
                dcc.Store(id="activation-data-store"),
                # Interval for auto-refresh
                dcc.Interval(id="interval-component", interval=5000, n_intervals=0),
            ],
            fluid=True,
        )

    def _register_callbacks(self):
        """Register dashboard callbacks."""

        @self.app.callback(Output("tab-content", "children"), Input("main-tabs", "active_tab"))
        def render_tab_content(active_tab):
            """Render content based on active tab."""
            if active_tab == "attention-tab":
                return self._create_attention_tab()
            elif active_tab == "gradient-tab":
                return self._create_gradient_tab()
            elif active_tab == "activation-tab":
                return self._create_activation_tab()
            elif active_tab == "overview-tab":
                return self._create_overview_tab()
            return html.Div()

        @self.app.callback(
            Output("attention-heatmap", "figure"),
            [Input("layer-dropdown", "value"), Input("head-dropdown", "value")],
            prevent_initial_call=True,
        )
        def update_attention_heatmap(layer_name, head_idx):
            """Update attention heatmap based on selection."""
            if layer_name in self.data_store.get("attention_weights", {}):
                weights = self.data_store["attention_weights"][layer_name]

                if isinstance(weights, torch.Tensor):
                    weights = weights.cpu().numpy()

                # Select specific head if specified
                if head_idx is not None and len(weights.shape) > 2:
                    weights = weights[head_idx]
                elif len(weights.shape) > 2:
                    weights = weights.mean(axis=0)

                fig = go.Figure(data=go.Heatmap(z=weights, colorscale="Blues", hoverongaps=False))

                fig.update_layout(
                    title=f"Attention Weights - {layer_name}",
                    xaxis_title="To Token",
                    yaxis_title="From Token",
                    height=500,
                )

                return fig

            return go.Figure()

        @self.app.callback(
            Output("gradient-flow-chart", "figure"), Input("interval-component", "n_intervals")
        )
        def update_gradient_flow(n):
            """Update gradient flow visualization."""
            if "gradient_norms" in self.data_store:
                layers = list(self.data_store["gradient_norms"].keys())
                norms = list(self.data_store["gradient_norms"].values())

                fig = go.Figure(data=go.Bar(x=layers, y=norms, marker_color="rgb(55, 83, 109)"))

                fig.update_layout(
                    title="Gradient Norms Across Layers",
                    xaxis_title="Layer",
                    yaxis_title="Gradient Norm",
                    yaxis_type="log",
                    height=400,
                )

                return fig

            return go.Figure()

    def _create_attention_tab(self) -> html.Div:
        """Create attention analysis tab content."""
        return html.Div(
            [
                dbc.Row([dbc.Col([html.H3("Attention Pattern Analysis"), html.Hr()], width=12)]),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Select Layer:"),
                                dcc.Dropdown(
                                    id="layer-dropdown",
                                    options=[
                                        {"label": layer, "value": layer}
                                        for layer in self.data_store.get(
                                            "attention_weights", {}
                                        ).keys()
                                    ],
                                    value=None,
                                    placeholder="Select a layer...",
                                ),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.Label("Select Head:"),
                                dcc.Dropdown(
                                    id="head-dropdown",
                                    options=[],
                                    value=None,
                                    placeholder="Select a head (optional)...",
                                ),
                            ],
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col([dcc.Graph(id="attention-heatmap")], width=8),
                        dbc.Col(
                            [
                                html.H5("Statistics"),
                                html.Div(
                                    id="attention-stats", children=[self._create_stats_card()]
                                ),
                            ],
                            width=4,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Head Importance Ranking"),
                                dcc.Graph(id="head-importance-chart"),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [html.H5("Token Importance"), dcc.Graph(id="token-importance-chart")],
                            width=6,
                        ),
                    ],
                    className="mt-4",
                ),
            ]
        )

    def _create_gradient_tab(self) -> html.Div:
        """Create gradient analysis tab content."""
        return html.Div(
            [
                dbc.Row([dbc.Col([html.H3("Gradient Analysis"), html.Hr()], width=12)]),
                dbc.Row(
                    [
                        dbc.Col(
                            [html.H5("Gradient Flow"), dcc.Graph(id="gradient-flow-chart")],
                            width=12,
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col([html.H5("Saliency Map"), dcc.Graph(id="saliency-map")], width=6),
                        dbc.Col(
                            [
                                html.H5("Integrated Gradients"),
                                dcc.Graph(id="integrated-gradients-chart"),
                            ],
                            width=6,
                        ),
                    ],
                    className="mt-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [html.H5("Gradient Statistics"), html.Div(id="gradient-stats-table")],
                            width=12,
                        )
                    ],
                    className="mt-4",
                ),
            ]
        )

    def _create_activation_tab(self) -> html.Div:
        """Create activation analysis tab content."""
        return html.Div(
            [
                dbc.Row([dbc.Col([html.H3("Activation Analysis"), html.Hr()], width=12)]),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Activation Distribution"),
                                dcc.Graph(id="activation-distribution"),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.H5("Layer Activations Comparison"),
                                dcc.Graph(id="activation-comparison"),
                            ],
                            width=6,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [html.H5("Dead Neurons Detection"), html.Div(id="dead-neurons-list")],
                            width=6,
                        ),
                        dbc.Col(
                            [html.H5("Saturation Analysis"), html.Div(id="saturation-analysis")],
                            width=6,
                        ),
                    ],
                    className="mt-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [html.H5("Activation Heatmap"), dcc.Graph(id="activation-heatmap")],
                            width=12,
                        )
                    ],
                    className="mt-4",
                ),
            ]
        )

    def _create_overview_tab(self) -> html.Div:
        """Create model overview tab content."""
        return html.Div(
            [
                dbc.Row([dbc.Col([html.H3("Model Overview"), html.Hr()], width=12)]),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    "Model Architecture", className="card-title"
                                                ),
                                                html.Div(id="model-architecture-info"),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4("Analysis Summary", className="card-title"),
                                                html.Div(id="analysis-summary"),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=6,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Layer-wise Metrics"),
                                dcc.Graph(id="layerwise-metrics-chart"),
                            ],
                            width=12,
                        )
                    ],
                    className="mt-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Export Options"),
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button(
                                            "Export Report", id="export-report-btn", color="primary"
                                        ),
                                        dbc.Button(
                                            "Export Data", id="export-data-btn", color="secondary"
                                        ),
                                        dbc.Button(
                                            "Export Visualizations",
                                            id="export-viz-btn",
                                            color="info",
                                        ),
                                    ]
                                ),
                            ],
                            width=12,
                        )
                    ],
                    className="mt-4",
                ),
            ]
        )

    def _create_stats_card(self) -> dbc.Card:
        """Create statistics card component."""
        stats = self.data_store.get("attention_stats", {})

        return dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.P(f"Total Layers: {stats.get('total_layers', 0)}"),
                        html.P(f"Total Heads: {stats.get('total_heads', 0)}"),
                        html.P(f"Sequence Length: {stats.get('seq_length', 0)}"),
                        html.P(f"Average Attention: {stats.get('avg_attention', 0):.3f}"),
                        html.P(f"Max Attention: {stats.get('max_attention', 0):.3f}"),
                    ]
                )
            ]
        )

    def update_data(self, data_type: str, data: Any) -> None:
        """
        Update dashboard data.

        Args:
            data_type: Type of data (attention, gradient, activation)
            data: Data to store
        """
        if data_type == "attention":
            self.data_store["attention_weights"] = data.get("weights", {})
            self.data_store["attention_stats"] = data.get("stats", {})
            self.data_store["tokens"] = data.get("tokens", [])
        elif data_type == "gradient":
            self.data_store["gradient_norms"] = data.get("norms", {})
            self.data_store["gradient_stats"] = data.get("stats", {})
            self.data_store["saliency_maps"] = data.get("saliency", {})
        elif data_type == "activation":
            self.data_store["activation_stats"] = data.get("stats", {})
            self.data_store["activation_distributions"] = data.get("distributions", {})
            self.data_store["dead_neurons"] = data.get("dead_neurons", {})

    def add_custom_visualization(
        self, viz_id: str, viz_function: callable, tab_name: str = "custom"
    ) -> None:
        """
        Add custom visualization to dashboard.

        Args:
            viz_id: Unique ID for visualization
            viz_function: Function that returns plotly figure
            tab_name: Tab to add visualization to
        """
        # Store custom visualization function
        if "custom_viz" not in self.data_store:
            self.data_store["custom_viz"] = {}
        self.data_store["custom_viz"][viz_id] = viz_function

        logger.info(f"Added custom visualization: {viz_id}")

    def generate_report(self, output_path: str, include_plots: bool = True) -> None:
        """
        Generate interpretability report.

        Args:
            output_path: Path to save report
            include_plots: Whether to include plots in report
        """
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Create HTML report
        html_content = f"""
        <html>
        <head>
            <title>Interpretability Report - {datetime.now().strftime("%Y-%m-%d %H:%M")}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
                .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .chart {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Model Interpretability Report</h1>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

            <h2>Attention Analysis</h2>
            <div class="metric">
                <strong>Total Layers Analyzed:</strong> {len(self.data_store.get("attention_weights", {}))}
            </div>

            <h2>Gradient Analysis</h2>
            <div class="metric">
                <strong>Gradient Norms Computed:</strong> {len(self.data_store.get("gradient_norms", {}))}
            </div>

            <h2>Activation Analysis</h2>
            <div class="metric">
                <strong>Activation Statistics:</strong> {len(self.data_store.get("activation_stats", {}))} layers
            </div>
        </body>
        </html>
        """

        with open(report_path, "w") as f:
            f.write(html_content)

        logger.info(f"Generated report at {output_path}")

    def run(self, debug: bool = False) -> None:
        """
        Run the dashboard server.

        Args:
            debug: Whether to run in debug mode
        """
        logger.info(f"Starting dashboard at http://{self.host}:{self.port}")
        self.app.run_server(host=self.host, port=self.port, debug=debug)

    def stop(self) -> None:
        """Stop the dashboard server."""
        if self.app:
            # Dash doesn't provide a clean way to stop, typically handled by KeyboardInterrupt
            logger.info("Dashboard stop requested")
