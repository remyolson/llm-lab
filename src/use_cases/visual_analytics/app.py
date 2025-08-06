"""
Unified Visual Analytics Dashboard

This module provides the main application interface that combines all visual
analytics components into a unified dashboard with WebSocket support and
Redis caching.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import redis
import websockets

from .analysis.behavior_analyzer import ModelBehaviorAnalyzer
from .comparison.response_analyzer import ResponseComparisonView

# Import all components
from .dashboard.metrics_dashboard import MetricsDashboard
from .evaluation.task_evaluator import TaskEvaluationPanel, TaskType
from .monitoring.regression_detector import RegressionDetector

# Web frameworks
try:
    import dash
    import dash_bootstrap_components as dbc
    from dash import Input, Output, State, dcc, html

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

logger = logging.getLogger(__name__)


class VisualAnalyticsApp:
    """Main visual analytics application."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        websocket_port: int = 8765,
        dashboard_port: int = 8050,
    ):
        """Initialize visual analytics app.

        Args:
            redis_host: Redis host
            redis_port: Redis port
            websocket_port: WebSocket server port
            dashboard_port: Dashboard server port
        """
        # Components
        self.metrics_dashboard = MetricsDashboard()
        self.response_analyzer = ResponseComparisonView()
        self.task_evaluator = TaskEvaluationPanel()
        self.behavior_analyzer = ModelBehaviorAnalyzer()
        self.regression_detector = RegressionDetector(alert_callback=self._handle_regression_alert)

        # Redis cache
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis_client.ping()
            self.redis_enabled = True
            logger.info("Redis cache connected")
        except:
            self.redis_client = None
            self.redis_enabled = False
            logger.warning("Redis not available - caching disabled")

        # WebSocket connections
        self.websocket_clients = set()
        self.websocket_port = websocket_port

        # Dashboard
        self.dashboard_port = dashboard_port

        if DASH_AVAILABLE:
            self._init_dash_app()

    def _init_dash_app(self):
        """Initialize Dash application."""
        self.app = dash.Dash(
            __name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True
        )

        # Main layout
        self.app.layout = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1(
                                    "🎯 Visual Performance Analytics", className="text-center mb-4"
                                ),
                                html.Hr(),
                            ]
                        )
                    ]
                ),
                # Navigation
                dbc.Nav(
                    [
                        dbc.NavItem(dbc.NavLink("Metrics", href="/metrics", id="metrics-link")),
                        dbc.NavItem(
                            dbc.NavLink("Comparison", href="/comparison", id="comparison-link")
                        ),
                        dbc.NavItem(
                            dbc.NavLink("Evaluation", href="/evaluation", id="evaluation-link")
                        ),
                        dbc.NavItem(dbc.NavLink("Behavior", href="/behavior", id="behavior-link")),
                        dbc.NavItem(
                            dbc.NavLink("Monitoring", href="/monitoring", id="monitoring-link")
                        ),
                    ],
                    pills=True,
                    className="mb-4",
                ),
                # Content area
                html.Div(id="page-content"),
                # WebSocket status
                dbc.Alert(
                    id="websocket-status",
                    children="WebSocket: Connected",
                    color="success",
                    className="mt-4",
                ),
                # Auto-refresh
                dcc.Interval(id="interval-component", interval=5000, n_intervals=0),
            ],
            fluid=True,
        )

        # Callbacks
        self._register_callbacks()

    def _register_callbacks(self):
        """Register Dash callbacks."""

        @self.app.callback(
            Output("page-content", "children"),
            [
                Input("metrics-link", "n_clicks"),
                Input("comparison-link", "n_clicks"),
                Input("evaluation-link", "n_clicks"),
                Input("behavior-link", "n_clicks"),
                Input("monitoring-link", "n_clicks"),
            ],
        )
        def display_page(*args):
            """Display selected page."""
            # Simplified page routing
            return html.Div(
                [
                    html.H2("Visual Analytics Dashboard"),
                    html.P("Select a component from the navigation menu above."),
                    html.Hr(),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Active Models"),
                                            dbc.CardBody(
                                                [
                                                    html.H4(
                                                        len(self.metrics_dashboard.training_runs)
                                                    ),
                                                    html.P("Training runs"),
                                                ]
                                            ),
                                        ]
                                    )
                                ],
                                width=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Comparisons"),
                                            dbc.CardBody(
                                                [
                                                    html.H4(
                                                        len(
                                                            self.response_analyzer.comparison_history
                                                        )
                                                    ),
                                                    html.P("Response comparisons"),
                                                ]
                                            ),
                                        ]
                                    )
                                ],
                                width=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Evaluations"),
                                            dbc.CardBody(
                                                [
                                                    html.H4(len(self.task_evaluator.results_cache)),
                                                    html.P("Task evaluations"),
                                                ]
                                            ),
                                        ]
                                    )
                                ],
                                width=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Alerts"),
                                            dbc.CardBody(
                                                [
                                                    html.H4(len(self.regression_detector.alerts)),
                                                    html.P("Regression alerts"),
                                                ]
                                            ),
                                        ]
                                    )
                                ],
                                width=3,
                            ),
                        ]
                    ),
                ]
            )

    def _handle_regression_alert(self, alert):
        """Handle regression alert.

        Args:
            alert: RegressionAlert object
        """
        logger.warning(
            f"Regression detected: {alert.metric_name} - {alert.regression_percent:.1f}%"
        )

        # Broadcast to WebSocket clients
        asyncio.create_task(self._broadcast_alert(alert))

        # Cache alert
        if self.redis_enabled:
            self._cache_alert(alert)

    async def _broadcast_alert(self, alert):
        """Broadcast alert to WebSocket clients.

        Args:
            alert: Alert to broadcast
        """
        if self.websocket_clients:
            message = json.dumps(
                {
                    "type": "regression_alert",
                    "data": {
                        "metric": alert.metric_name,
                        "severity": alert.severity,
                        "regression": alert.regression_percent,
                        "timestamp": alert.timestamp.isoformat(),
                    },
                }
            )

            disconnected = set()
            for client in self.websocket_clients:
                try:
                    await client.send(message)
                except:
                    disconnected.add(client)

            # Remove disconnected clients
            self.websocket_clients -= disconnected

    def _cache_alert(self, alert):
        """Cache alert in Redis.

        Args:
            alert: Alert to cache
        """
        if not self.redis_enabled:
            return

        key = f"alert:{alert.metric_name}:{alert.timestamp.timestamp()}"
        value = json.dumps(
            {
                "metric": alert.metric_name,
                "baseline": alert.baseline_value,
                "current": alert.current_value,
                "regression": alert.regression_percent,
                "severity": alert.severity,
            }
        )

        self.redis_client.setex(key, 86400, value)  # 24 hour TTL

    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections.

        Args:
            websocket: WebSocket connection
            path: Request path
        """
        # Register client
        self.websocket_clients.add(websocket)
        logger.info(f"WebSocket client connected: {websocket.remote_address}")

        try:
            # Send initial data
            await websocket.send(
                json.dumps(
                    {
                        "type": "connection",
                        "status": "connected",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            )

            # Handle messages
            async for message in websocket:
                data = json.loads(message)

                if data.get("type") == "subscribe":
                    # Handle subscription requests
                    pass
                elif data.get("type") == "metric_update":
                    # Handle metric updates
                    metric_name = data.get("metric")
                    value = data.get("value")

                    if metric_name and value is not None:
                        self.regression_detector.update_metric(metric_name, value)

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            # Unregister client
            self.websocket_clients.remove(websocket)
            logger.info(f"WebSocket client disconnected: {websocket.remote_address}")

    async def start_websocket_server(self):
        """Start WebSocket server."""
        logger.info(f"Starting WebSocket server on port {self.websocket_port}")
        await websockets.serve(self.websocket_handler, "localhost", self.websocket_port)

    def launch(self):
        """Launch the unified dashboard."""
        if DASH_AVAILABLE:
            # Start WebSocket server in background
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.create_task(self.start_websocket_server())

            # Launch Dash app
            logger.info(f"Launching visual analytics dashboard on port {self.dashboard_port}")
            self.app.run_server(host="0.0.0.0", port=self.dashboard_port, debug=False)
        else:
            logger.error("Dash is not installed. Cannot launch dashboard.")

    def get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data from Redis.

        Args:
            key: Cache key

        Returns:
            Cached data or None
        """
        if not self.redis_enabled:
            return None

        try:
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        except:
            return None

    def set_cached_data(self, key: str, value: Any, ttl: int = 3600):
        """Set cached data in Redis.

        Args:
            key: Cache key
            value: Data to cache
            ttl: Time-to-live in seconds
        """
        if not self.redis_enabled:
            return

        try:
            self.redis_client.setex(key, ttl, json.dumps(value))
        except:
            pass


# Example usage
if __name__ == "__main__":
    # Create and launch app
    app = VisualAnalyticsApp()

    # Add sample data
    app.metrics_dashboard.add_training_run("test_run", "gpt2", {"batch_size": 32})

    # Launch dashboard
    app.launch()
