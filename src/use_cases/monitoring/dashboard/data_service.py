"""
Data Integration Service

Handles data collection from monitoring sources and real-time updates.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .models import Alert, DatabaseManager, MetricSnapshot


@dataclass
class DataCollector:
    """
    Collects data from various monitoring sources.
    """

    db_manager: DatabaseManager
    collection_interval: int = 60  # seconds
    running: bool = False
    _thread: Optional[threading.Thread] = None
    _stop_event: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self):
        """Initialize the data collector."""
        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start the data collection thread."""
        if self.running:
            self.logger.warning("Data collector already running")
            return

        self.running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()
        self.logger.info("Data collector started")

    def stop(self):
        """Stop the data collection thread."""
        if not self.running:
            return

        self.running = False
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        self.logger.info("Data collector stopped")

    def _collection_loop(self):
        """Main collection loop."""
        while not self._stop_event.wait(self.collection_interval):
            try:
                self._collect_metrics()
                self._collect_alerts()
                self._cleanup_old_data()
            except Exception as e:
                self.logger.error(f"Error in collection loop: {e}")

    def _collect_metrics(self):
        """Collect metrics from monitoring sources."""
        try:
            # Check for existing monitoring database
            monitoring_db_path = Path("monitoring.db")
            if monitoring_db_path.exists():
                self._collect_from_monitoring_db(str(monitoring_db_path))
            else:
                # Generate sample metrics for demonstration
                self._generate_sample_metrics()

        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")

    def _collect_from_monitoring_db(self, db_path: str):
        """Collect metrics from existing monitoring database."""
        try:
            import sqlite3

            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()

                # Try to get recent metrics from various possible table structures
                tables_to_check = [
                    "metrics",
                    "metric_snapshots",
                    "performance_metrics",
                    "monitoring_data",
                    "llm_metrics",
                ]

                for table in tables_to_check:
                    try:
                        cursor.execute(
                            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
                        )
                        if cursor.fetchone():
                            self._extract_metrics_from_table(cursor, table)
                            break
                    except sqlite3.Error:
                        continue

        except Exception as e:
            self.logger.error(f"Error reading monitoring database: {e}")

    def _extract_metrics_from_table(self, cursor, table_name: str):
        """Extract metrics from a specific table."""
        try:
            # Get table structure
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]

            # Build query based on available columns
            time_col = self._find_time_column(columns)
            if not time_col:
                return

            # Get recent data
            query = f"SELECT * FROM {table_name} WHERE {time_col} >= datetime('now', '-24 hours') ORDER BY {time_col} DESC LIMIT 100"
            cursor.execute(query)

            rows = cursor.fetchall()
            if not rows:
                return

            # Convert to metric snapshots
            for row in rows:
                row_dict = dict(zip(columns, row))
                self._create_metric_snapshot_from_row(row_dict, time_col)

        except Exception as e:
            self.logger.error(f"Error extracting from table {table_name}: {e}")

    def _find_time_column(self, columns: List[str]) -> Optional[str]:
        """Find the timestamp column in the table."""
        time_candidates = ["timestamp", "created_at", "time", "datetime", "date"]
        for candidate in time_candidates:
            if candidate in columns:
                return candidate
        return None

    def _create_metric_snapshot_from_row(self, row_dict: Dict[str, Any], time_col: str):
        """Create a metric snapshot from database row."""
        try:
            # Parse timestamp
            timestamp_str = row_dict.get(time_col)
            if isinstance(timestamp_str, str):
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                timestamp = datetime.utcnow()

            # Extract common fields with fallbacks
            snapshot_data = {
                "timestamp": timestamp,
                "provider": str(row_dict.get("provider", "unknown")),
                "model": str(row_dict.get("model", "unknown")),
                "requests_count": int(
                    row_dict.get("requests_count", row_dict.get("request_count", 0))
                ),
                "avg_latency": float(row_dict.get("avg_latency", row_dict.get("latency", 0.0))),
                "success_rate": float(row_dict.get("success_rate", 100.0)),
                "error_count": int(row_dict.get("error_count", 0)),
                "total_cost": float(row_dict.get("total_cost", row_dict.get("cost", 0.0))),
                "cost_per_request": float(row_dict.get("cost_per_request", 0.0)),
                "token_count": int(row_dict.get("token_count", row_dict.get("tokens", 0))),
                "avg_quality_score": float(
                    row_dict.get("avg_quality_score", row_dict.get("quality_score", 0.0))
                ),
                "accuracy_score": float(row_dict.get("accuracy_score", 0.0)),
                "coherence_score": float(row_dict.get("coherence_score", 0.0)),
                "cpu_usage": float(row_dict.get("cpu_usage", 0.0)),
                "memory_usage": float(row_dict.get("memory_usage", 0.0)),
                "metadata": {k: v for k, v in row_dict.items() if k not in snapshot_data},
            }

            # Check if this snapshot already exists
            session = self.db_manager.get_session()
            try:
                existing = (
                    session.query(MetricSnapshot)
                    .filter(
                        MetricSnapshot.timestamp == timestamp,
                        MetricSnapshot.provider == snapshot_data["provider"],
                        MetricSnapshot.model == snapshot_data["model"],
                    )
                    .first()
                )

                if not existing:
                    snapshot = MetricSnapshot(**snapshot_data)
                    session.add(snapshot)
                    session.commit()

            finally:
                session.close()

        except Exception as e:
            self.logger.error(f"Error creating metric snapshot: {e}")

    def _generate_sample_metrics(self):
        """Generate sample metrics for demonstration."""
        import random

        providers = [
            ("OpenAI", "gpt-4o-mini"),
            ("Anthropic", "claude-3-haiku-20240307"),
            ("Google", "gemini-1.5-flash"),
        ]

        for provider, model in providers:
            snapshot_data = {
                "timestamp": datetime.utcnow(),
                "provider": provider,
                "model": model,
                "requests_count": random.randint(50, 200),
                "avg_latency": random.uniform(0.2, 2.0),
                "success_rate": random.uniform(95.0, 100.0),
                "error_count": random.randint(0, 5),
                "total_cost": random.uniform(0.1, 5.0),
                "cost_per_request": random.uniform(0.001, 0.01),
                "token_count": random.randint(1000, 10000),
                "avg_quality_score": random.uniform(0.7, 0.95),
                "accuracy_score": random.uniform(0.8, 0.98),
                "coherence_score": random.uniform(0.75, 0.95),
                "cpu_usage": random.uniform(10.0, 80.0),
                "memory_usage": random.uniform(100.0, 1000.0),
                "metadata": {
                    "source": "sample_generator",
                    "generation_time": datetime.utcnow().isoformat(),
                },
            }

            try:
                session = self.db_manager.get_session()
                try:
                    snapshot = MetricSnapshot(**snapshot_data)
                    session.add(snapshot)
                    session.commit()
                finally:
                    session.close()
            except Exception as e:
                self.logger.error(f"Error saving sample metrics: {e}")

    def _collect_alerts(self):
        """Collect alerts from monitoring sources."""
        try:
            # Check for existing alerts in monitoring system
            alerts_db_path = Path("alerts.db")
            if alerts_db_path.exists():
                self._collect_from_alerts_db(str(alerts_db_path))
            else:
                # Generate sample alerts occasionally
                import random

                if random.random() < 0.1:  # 10% chance
                    self._generate_sample_alert()

        except Exception as e:
            self.logger.error(f"Error collecting alerts: {e}")

    def _collect_from_alerts_db(self, db_path: str):
        """Collect alerts from existing alerts database."""
        try:
            import sqlite3

            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()

                # Look for alerts table
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='alerts'"
                )
                if cursor.fetchone():
                    cursor.execute("""
                        SELECT * FROM alerts
                        WHERE created_at >= datetime('now', '-24 hours')
                        ORDER BY created_at DESC LIMIT 50
                    """)

                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]

                    for row in rows:
                        row_dict = dict(zip(columns, row))
                        self._create_alert_from_row(row_dict)

        except Exception as e:
            self.logger.error(f"Error reading alerts database: {e}")

    def _create_alert_from_row(self, row_dict: Dict[str, Any]):
        """Create an alert from database row."""
        try:
            alert_data = {
                "timestamp": datetime.fromisoformat(
                    row_dict.get("created_at", datetime.utcnow().isoformat())
                ),
                "alert_type": row_dict.get("alert_type", "unknown"),
                "severity": row_dict.get("severity", "info"),
                "title": row_dict.get("title", "Alert"),
                "message": row_dict.get("message", "No message"),
                "provider": row_dict.get("provider"),
                "model": row_dict.get("model"),
                "metric_name": row_dict.get("metric_name"),
                "current_value": row_dict.get("current_value"),
                "threshold_value": row_dict.get("threshold_value"),
                "threshold_operator": row_dict.get("threshold_operator"),
                "status": row_dict.get("status", "active"),
                "details": {k: v for k, v in row_dict.items() if k not in alert_data},
            }

            # Check if alert already exists
            session = self.db_manager.get_session()
            try:
                existing = (
                    session.query(Alert)
                    .filter(
                        Alert.timestamp == alert_data["timestamp"],
                        Alert.title == alert_data["title"],
                        Alert.provider == alert_data["provider"],
                    )
                    .first()
                )

                if not existing:
                    alert = Alert(**alert_data)
                    session.add(alert)
                    session.commit()

            finally:
                session.close()

        except Exception as e:
            self.logger.error(f"Error creating alert: {e}")

    def _generate_sample_alert(self):
        """Generate a sample alert."""
        import random

        alert_types = ["performance", "cost", "error", "quality"]
        severities = ["warning", "critical"]
        providers = ["OpenAI", "Anthropic", "Google"]

        alert_type = random.choice(alert_types)
        severity = random.choice(severities)
        provider = random.choice(providers)

        messages = {
            "performance": f"High latency detected for {provider}",
            "cost": f"Cost threshold exceeded for {provider}",
            "error": f"Error rate spike detected for {provider}",
            "quality": f"Quality score below threshold for {provider}",
        }

        alert_data = {
            "timestamp": datetime.utcnow(),
            "alert_type": alert_type,
            "severity": severity,
            "title": f"{alert_type.title()} Alert - {provider}",
            "message": messages[alert_type],
            "provider": provider,
            "model": "sample-model",
            "metric_name": f"avg_{alert_type}",
            "current_value": random.uniform(0.5, 2.0),
            "threshold_value": 1.0,
            "threshold_operator": ">",
            "status": "active",
            "details": {"source": "sample_generator", "auto_generated": True},
        }

        try:
            session = self.db_manager.get_session()
            try:
                alert = Alert(**alert_data)
                session.add(alert)
                session.commit()
                self.logger.info(f"Generated sample alert: {alert.title}")
            finally:
                session.close()
        except Exception as e:
            self.logger.error(f"Error generating sample alert: {e}")

    def _cleanup_old_data(self):
        """Clean up old data periodically."""
        try:
            # Only cleanup once per hour
            if hasattr(self, "_last_cleanup"):
                if (datetime.utcnow() - self._last_cleanup).seconds < 3600:
                    return

            result = self.db_manager.cleanup_old_data(days=90)
            self._last_cleanup = datetime.utcnow()

            if any(result.values()):
                self.logger.info(f"Cleaned up old data: {result}")

        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")


@dataclass
class RealTimeUpdater:
    """
    Handles real-time updates via WebSocket.
    """

    db_manager: DatabaseManager
    update_interval: int = 30  # seconds
    subscribers: Dict[str, List[Callable]] = field(default_factory=dict)
    running: bool = False
    _thread: Optional[threading.Thread] = None
    _stop_event: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self):
        """Initialize the real-time updater."""
        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start the real-time update thread."""
        if self.running:
            return

        self.running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        self.logger.info("Real-time updater started")

    def stop(self):
        """Stop the real-time update thread."""
        if not self.running:
            return

        self.running = False
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        self.logger.info("Real-time updater stopped")

    def subscribe(self, channel: str, callback: Callable):
        """Subscribe to real-time updates."""
        if channel not in self.subscribers:
            self.subscribers[channel] = []
        self.subscribers[channel].append(callback)
        self.logger.info(f"Subscriber added to channel: {channel}")

    def unsubscribe(self, channel: str, callback: Callable):
        """Unsubscribe from real-time updates."""
        if channel in self.subscribers:
            try:
                self.subscribers[channel].remove(callback)
                if not self.subscribers[channel]:
                    del self.subscribers[channel]
                self.logger.info(f"Subscriber removed from channel: {channel}")
            except ValueError:
                pass

    def _update_loop(self):
        """Main update loop."""
        while not self._stop_event.wait(self.update_interval):
            try:
                self._send_updates()
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")

    def _send_updates(self):
        """Send updates to subscribers."""
        try:
            # Get latest metrics
            metrics = self.db_manager.get_metrics_summary()
            self._notify_subscribers(
                "metrics",
                {"type": "overview", "data": metrics, "timestamp": datetime.utcnow().isoformat()},
            )

            # Get latest alerts
            alerts = self.db_manager.get_active_alerts(limit=10)
            if alerts:
                self._notify_subscribers(
                    "alerts",
                    {
                        "type": "alert_update",
                        "data": alerts,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )

        except Exception as e:
            self.logger.error(f"Error sending updates: {e}")

    def _notify_subscribers(self, channel: str, data: Dict[str, Any]):
        """Notify all subscribers in a channel."""
        if channel not in self.subscribers:
            return

        for callback in self.subscribers[channel][
            :
        ]:  # Copy list to avoid modification during iteration
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Error calling subscriber callback: {e}")
                # Remove failed callback
                try:
                    self.subscribers[channel].remove(callback)
                except ValueError:
                    pass


class DataService:
    """
    Main data service that coordinates collection and real-time updates.
    """

    def __init__(self, db_manager: DatabaseManager):
        """Initialize the data service."""
        self.db_manager = db_manager
        self.collector = DataCollector(db_manager)
        self.updater = RealTimeUpdater(db_manager)
        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start all data services."""
        self.logger.info("Starting data services...")
        self.collector.start()
        self.updater.start()
        self.logger.info("All data services started")

    def stop(self):
        """Stop all data services."""
        self.logger.info("Stopping data services...")
        self.collector.stop()
        self.updater.stop()
        self.logger.info("All data services stopped")

    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary metrics."""
        return self.db_manager.get_metrics_summary(hours)

    def get_performance_data(
        self, hours: int = 24, provider: str = None, model: str = None
    ) -> Dict[str, Any]:
        """Get performance data."""
        return self.db_manager.get_performance_data(hours, provider, model)

    def get_cost_breakdown(self, hours: int = 24) -> Dict[str, Any]:
        """Get cost breakdown."""
        return self.db_manager.get_cost_breakdown(hours)

    def get_active_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get active alerts."""
        return self.db_manager.get_active_alerts(limit)

    def subscribe_to_updates(self, channel: str, callback: Callable):
        """Subscribe to real-time updates."""
        self.updater.subscribe(channel, callback)

    def unsubscribe_from_updates(self, channel: str, callback: Callable):
        """Unsubscribe from real-time updates."""
        self.updater.unsubscribe(channel, callback)


# Global data service instance
data_service: Optional[DataService] = None


def get_data_service() -> DataService:
    """Get the global data service instance."""
    global data_service
    if data_service is None:
        raise RuntimeError("Data service not initialized. Call init_data_service() first.")
    return data_service


def init_data_service(db_manager: DatabaseManager) -> DataService:
    """Initialize the global data service."""
    global data_service
    data_service = DataService(db_manager)
    return data_service
