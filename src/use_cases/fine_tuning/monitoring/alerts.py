"""
Alerting System for Fine-Tuning Monitoring

This module provides an alerting system for monitoring fine-tuning jobs,
including anomaly detection, threshold alerts, and notification mechanisms.

Example:
    alert_system = AlertSystem()

    # Add threshold alert
    alert_system.add_threshold_alert(
        metric="loss",
        threshold=2.0,
        condition="greater_than"
    )

    # Add anomaly detection
    alert_system.add_anomaly_detector(
        metric="gpu_memory",
        sensitivity=2.5
    )

    # Check metrics
    alert_system.check_metrics({
        "loss": 2.5,
        "gpu_memory": 15000
    })
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import json
import logging
import smtplib
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCondition(Enum):
    """Alert condition types."""

    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    BETWEEN = "between"
    OUTSIDE = "outside"


@dataclass
class AlertConfig:
    """Alert configuration."""

    name: str
    metric: str
    severity: AlertSeverity = AlertSeverity.WARNING
    enabled: bool = True
    cooldown_minutes: int = 5
    max_alerts_per_hour: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertEvent:
    """Alert event record."""

    alert_name: str
    timestamp: datetime
    severity: AlertSeverity
    metric_name: str
    metric_value: float
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_name": self.alert_name,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "message": self.message,
            "context": self.context,
        }


class BaseAlert(ABC):
    """Base class for alert types."""

    def __init__(self, config: AlertConfig):
        """Initialize alert.

        Args:
            config: Alert configuration
        """
        self.config = config
        self.last_alert_time = None
        self.alert_count_window = deque(maxlen=60)  # Last hour

    @abstractmethod
    def check(self, value: float, context: Dict[str, Any]) -> Optional[AlertEvent]:
        """Check if alert should trigger.

        Args:
            value: Metric value
            context: Additional context

        Returns:
            AlertEvent if triggered, None otherwise
        """
        pass

    def should_alert(self) -> bool:
        """Check if alert should be sent based on cooldown and rate limits."""
        current_time = datetime.now()

        # Check cooldown
        if self.last_alert_time:
            cooldown_delta = timedelta(minutes=self.config.cooldown_minutes)
            if current_time - self.last_alert_time < cooldown_delta:
                return False

        # Check rate limit
        hour_ago = current_time - timedelta(hours=1)
        recent_alerts = sum(1 for t in self.alert_count_window if t > hour_ago)
        if recent_alerts >= self.config.max_alerts_per_hour:
            return False

        return True

    def record_alert(self):
        """Record that an alert was triggered."""
        current_time = datetime.now()
        self.last_alert_time = current_time
        self.alert_count_window.append(current_time)


class ThresholdAlert(BaseAlert):
    """Threshold-based alert."""

    def __init__(
        self,
        config: AlertConfig,
        threshold: float,
        condition: AlertCondition = AlertCondition.GREATER_THAN,
        threshold2: Optional[float] = None,
    ):
        """Initialize threshold alert.

        Args:
            config: Alert configuration
            threshold: Threshold value
            condition: Alert condition
            threshold2: Second threshold for between/outside conditions
        """
        super().__init__(config)
        self.threshold = threshold
        self.condition = condition
        self.threshold2 = threshold2

    def check(self, value: float, context: Dict[str, Any]) -> Optional[AlertEvent]:
        """Check threshold condition."""
        if not self.config.enabled or not self.should_alert():
            return None

        triggered = False

        if self.condition == AlertCondition.GREATER_THAN:
            triggered = value > self.threshold
        elif self.condition == AlertCondition.LESS_THAN:
            triggered = value < self.threshold
        elif self.condition == AlertCondition.EQUALS:
            triggered = abs(value - self.threshold) < 1e-6
        elif self.condition == AlertCondition.NOT_EQUALS:
            triggered = abs(value - self.threshold) >= 1e-6
        elif self.condition == AlertCondition.BETWEEN and self.threshold2:
            triggered = self.threshold <= value <= self.threshold2
        elif self.condition == AlertCondition.OUTSIDE and self.threshold2:
            triggered = value < self.threshold or value > self.threshold2

        if triggered:
            self.record_alert()

            message = f"{self.config.metric} ({value:.4f}) {self.condition.value} {self.threshold}"
            if self.threshold2:
                message += f" and {self.threshold2}"

            return AlertEvent(
                alert_name=self.config.name,
                timestamp=datetime.now(),
                severity=self.config.severity,
                metric_name=self.config.metric,
                metric_value=value,
                message=message,
                context=context,
            )

        return None


class AnomalyAlert(BaseAlert):
    """Anomaly detection alert using statistical methods."""

    def __init__(
        self,
        config: AlertConfig,
        window_size: int = 50,
        sensitivity: float = 3.0,
        min_samples: int = 20,
    ):
        """Initialize anomaly alert.

        Args:
            config: Alert configuration
            window_size: Window size for statistics
            sensitivity: Number of standard deviations for anomaly
            min_samples: Minimum samples before alerting
        """
        super().__init__(config)
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.min_samples = min_samples
        self.value_history = deque(maxlen=window_size)

        # Moving statistics
        self.mean = 0
        self.std = 0
        self.update_count = 0

    def update_statistics(self, value: float):
        """Update running statistics."""
        self.value_history.append(value)
        self.update_count += 1

        if len(self.value_history) >= self.min_samples:
            values = np.array(self.value_history)
            self.mean = np.mean(values)
            self.std = np.std(values)

    def check(self, value: float, context: Dict[str, Any]) -> Optional[AlertEvent]:
        """Check for anomalies."""
        self.update_statistics(value)

        if not self.config.enabled or not self.should_alert():
            return None

        if len(self.value_history) < self.min_samples:
            return None

        if self.std == 0:
            return None

        # Calculate z-score
        z_score = abs(value - self.mean) / self.std

        if z_score > self.sensitivity:
            self.record_alert()

            direction = "above" if value > self.mean else "below"

            return AlertEvent(
                alert_name=self.config.name,
                timestamp=datetime.now(),
                severity=self.config.severity,
                metric_name=self.config.metric,
                metric_value=value,
                message=f"Anomaly detected: {self.config.metric} is {z_score:.2f} std devs {direction} mean",
                context={
                    **context,
                    "z_score": z_score,
                    "mean": self.mean,
                    "std": self.std,
                    "window_size": len(self.value_history),
                },
            )

        return None


class TrendAlert(BaseAlert):
    """Alert based on metric trends."""

    def __init__(
        self,
        config: AlertConfig,
        window_size: int = 20,
        trend_threshold: float = 0.1,
        direction: str = "increasing",
    ):
        """Initialize trend alert.

        Args:
            config: Alert configuration
            window_size: Window size for trend calculation
            trend_threshold: Minimum trend slope
            direction: "increasing" or "decreasing"
        """
        super().__init__(config)
        self.window_size = window_size
        self.trend_threshold = trend_threshold
        self.direction = direction
        self.value_history = deque(maxlen=window_size)
        self.time_history = deque(maxlen=window_size)

    def check(self, value: float, context: Dict[str, Any]) -> Optional[AlertEvent]:
        """Check for trend anomalies."""
        current_time = time.time()
        self.value_history.append(value)
        self.time_history.append(current_time)

        if not self.config.enabled or not self.should_alert():
            return None

        if len(self.value_history) < self.window_size:
            return None

        # Calculate trend using linear regression
        times = np.array(self.time_history) - self.time_history[0]
        values = np.array(self.value_history)

        # Normalize time to avoid numerical issues
        times = times / times[-1] if times[-1] > 0 else times

        # Calculate slope
        slope = np.polyfit(times, values, 1)[0]

        # Check trend direction
        triggered = False
        if self.direction == "increasing" and slope > self.trend_threshold:
            triggered = True
        elif self.direction == "decreasing" and slope < -self.trend_threshold:
            triggered = True

        if triggered:
            self.record_alert()

            return AlertEvent(
                alert_name=self.config.name,
                timestamp=datetime.now(),
                severity=self.config.severity,
                metric_name=self.config.metric,
                metric_value=value,
                message=f"Trend alert: {self.config.metric} is {self.direction} (slope: {slope:.4f})",
                context={
                    **context,
                    "slope": slope,
                    "window_size": len(self.value_history),
                    "direction": self.direction,
                },
            )

        return None


class BaseNotifier(ABC):
    """Base class for notification methods."""

    @abstractmethod
    def send(self, alert_event: AlertEvent) -> bool:
        """Send notification.

        Args:
            alert_event: Alert event to notify about

        Returns:
            True if successful
        """
        pass


class ConsoleNotifier(BaseNotifier):
    """Console notification output."""

    def send(self, alert_event: AlertEvent) -> bool:
        """Print alert to console."""
        severity_colors = {
            AlertSeverity.INFO: "\033[94m",  # Blue
            AlertSeverity.WARNING: "\033[93m",  # Yellow
            AlertSeverity.ERROR: "\033[91m",  # Red
            AlertSeverity.CRITICAL: "\033[95m",  # Magenta
        }

        color = severity_colors.get(alert_event.severity, "")
        reset = "\033[0m"

        print(
            f"{color}[ALERT - {alert_event.severity.value.upper()}] "
            f"{alert_event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}{reset}"
        )
        print(f"  Alert: {alert_event.alert_name}")
        print(f"  Message: {alert_event.message}")
        print(f"  Metric: {alert_event.metric_name} = {alert_event.metric_value:.4f}")

        if alert_event.context:
            print("  Context:")
            for key, value in alert_event.context.items():
                print(f"    {key}: {value}")

        print()
        return True


class EmailNotifier(BaseNotifier):
    """Email notification sender."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_emails: List[str],
        use_tls: bool = True,
    ):
        """Initialize email notifier.

        Args:
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_email: Sender email
            to_emails: List of recipient emails
            use_tls: Use TLS encryption
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
        self.use_tls = use_tls

    def send(self, alert_event: AlertEvent) -> bool:
        """Send email notification."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.to_emails)
            msg["Subject"] = (
                f"[{alert_event.severity.value.upper()}] Fine-Tuning Alert: {alert_event.alert_name}"
            )

            # Create body
            body = f"""
Fine-Tuning Alert Notification

Alert: {alert_event.alert_name}
Severity: {alert_event.severity.value.upper()}
Time: {alert_event.timestamp.strftime("%Y-%m-%d %H:%M:%S")}

Message: {alert_event.message}

Metric: {alert_event.metric_name}
Value: {alert_event.metric_value:.4f}

Context:
{json.dumps(alert_event.context, indent=2)}
            """

            msg.attach(MIMEText(body, "plain"))

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            return True

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False


class SlackNotifier(BaseNotifier):
    """Slack notification sender."""

    def __init__(self, webhook_url: str):
        """Initialize Slack notifier.

        Args:
            webhook_url: Slack webhook URL
        """
        self.webhook_url = webhook_url

    def send(self, alert_event: AlertEvent) -> bool:
        """Send Slack notification."""
        try:
            # Severity colors for Slack
            severity_colors = {
                AlertSeverity.INFO: "#36a64f",  # Green
                AlertSeverity.WARNING: "#ff9900",  # Orange
                AlertSeverity.ERROR: "#ff0000",  # Red
                AlertSeverity.CRITICAL: "#9b59b6",  # Purple
            }

            # Create Slack message
            payload = {
                "attachments": [
                    {
                        "color": severity_colors.get(alert_event.severity, "#808080"),
                        "title": f"{alert_event.severity.value.upper()}: {alert_event.alert_name}",
                        "text": alert_event.message,
                        "fields": [
                            {
                                "title": "Metric",
                                "value": f"{alert_event.metric_name}: {alert_event.metric_value:.4f}",
                                "short": True,
                            },
                            {
                                "title": "Time",
                                "value": alert_event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True,
                            },
                        ],
                        "footer": "Fine-Tuning Alert System",
                        "ts": int(alert_event.timestamp.timestamp()),
                    }
                ]
            }

            # Add context fields
            if alert_event.context:
                for key, value in list(alert_event.context.items())[:4]:  # Limit to 4 fields
                    payload["attachments"][0]["fields"].append(
                        {"title": key, "value": str(value), "short": True}
                    )

            # Send to Slack
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()

            return True

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False


class AlertSystem:
    """Main alert system for monitoring fine-tuning."""

    def __init__(self):
        """Initialize alert system."""
        self.alerts: Dict[str, BaseAlert] = {}
        self.notifiers: List[BaseNotifier] = []
        self.alert_history: List[AlertEvent] = []
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1000))

        # Add default console notifier
        self.add_notifier(ConsoleNotifier())

    def add_threshold_alert(
        self,
        name: str,
        metric: str,
        threshold: float,
        condition: str | AlertCondition = AlertCondition.GREATER_THAN,
        severity: str | AlertSeverity = AlertSeverity.WARNING,
        **kwargs,
    ):
        """Add threshold alert.

        Args:
            name: Alert name
            metric: Metric to monitor
            threshold: Threshold value
            condition: Alert condition
            severity: Alert severity
            **kwargs: Additional alert configuration
        """
        if isinstance(condition, str):
            condition = AlertCondition(condition)
        if isinstance(severity, str):
            severity = AlertSeverity(severity)

        config = AlertConfig(name=name, metric=metric, severity=severity, **kwargs)

        alert = ThresholdAlert(config, threshold, condition)
        self.alerts[name] = alert

        logger.info(f"Added threshold alert: {name} for {metric}")

    def add_anomaly_detector(
        self,
        name: str,
        metric: str,
        sensitivity: float = 3.0,
        window_size: int = 50,
        severity: str | AlertSeverity = AlertSeverity.WARNING,
        **kwargs,
    ):
        """Add anomaly detection alert.

        Args:
            name: Alert name
            metric: Metric to monitor
            sensitivity: Sensitivity (number of std devs)
            window_size: Window size for statistics
            severity: Alert severity
            **kwargs: Additional alert configuration
        """
        if isinstance(severity, str):
            severity = AlertSeverity(severity)

        config = AlertConfig(name=name, metric=metric, severity=severity, **kwargs)

        alert = AnomalyAlert(config, window_size, sensitivity)
        self.alerts[name] = alert

        logger.info(f"Added anomaly detector: {name} for {metric}")

    def add_trend_alert(
        self,
        name: str,
        metric: str,
        trend_threshold: float = 0.1,
        direction: str = "increasing",
        window_size: int = 20,
        severity: str | AlertSeverity = AlertSeverity.WARNING,
        **kwargs,
    ):
        """Add trend detection alert.

        Args:
            name: Alert name
            metric: Metric to monitor
            trend_threshold: Minimum trend slope
            direction: Trend direction ("increasing" or "decreasing")
            window_size: Window size for trend calculation
            severity: Alert severity
            **kwargs: Additional alert configuration
        """
        if isinstance(severity, str):
            severity = AlertSeverity(severity)

        config = AlertConfig(name=name, metric=metric, severity=severity, **kwargs)

        alert = TrendAlert(config, window_size, trend_threshold, direction)
        self.alerts[name] = alert

        logger.info(f"Added trend alert: {name} for {metric}")

    def add_notifier(self, notifier: BaseNotifier):
        """Add notification method.

        Args:
            notifier: Notifier instance
        """
        self.notifiers.append(notifier)
        logger.info(f"Added notifier: {type(notifier).__name__}")

    def check_metrics(self, metrics: Dict[str, float], context: Optional[Dict[str, Any]] = None):
        """Check metrics against all alerts.

        Args:
            metrics: Current metrics
            context: Additional context
        """
        context = context or {}

        # Update metrics buffer
        for metric_name, value in metrics.items():
            self.metrics_buffer[metric_name].append(value)

        # Check each alert
        for alert_name, alert in self.alerts.items():
            if alert.config.metric in metrics:
                value = metrics[alert.config.metric]

                alert_event = alert.check(value, context)
                if alert_event:
                    self._handle_alert(alert_event)

    def _handle_alert(self, alert_event: AlertEvent):
        """Handle triggered alert.

        Args:
            alert_event: Alert event
        """
        # Add to history
        self.alert_history.append(alert_event)

        # Send notifications
        for notifier in self.notifiers:
            try:
                notifier.send(alert_event)
            except Exception as e:
                logger.error(f"Notifier {type(notifier).__name__} failed: {e}")

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert activity."""
        if not self.alert_history:
            return {"total_alerts": 0, "by_severity": {}, "by_alert": {}, "recent_alerts": []}

        # Count by severity
        by_severity = defaultdict(int)
        by_alert = defaultdict(int)

        for event in self.alert_history:
            by_severity[event.severity.value] += 1
            by_alert[event.alert_name] += 1

        # Get recent alerts
        recent_alerts = [event.to_dict() for event in self.alert_history[-10:]]

        return {
            "total_alerts": len(self.alert_history),
            "by_severity": dict(by_severity),
            "by_alert": dict(by_alert),
            "recent_alerts": recent_alerts,
        }

    def disable_alert(self, name: str):
        """Disable an alert.

        Args:
            name: Alert name
        """
        if name in self.alerts:
            self.alerts[name].config.enabled = False
            logger.info(f"Disabled alert: {name}")

    def enable_alert(self, name: str):
        """Enable an alert.

        Args:
            name: Alert name
        """
        if name in self.alerts:
            self.alerts[name].config.enabled = True
            logger.info(f"Enabled alert: {name}")

    def remove_alert(self, name: str):
        """Remove an alert.

        Args:
            name: Alert name
        """
        if name in self.alerts:
            del self.alerts[name]
            logger.info(f"Removed alert: {name}")


# Example usage
if __name__ == "__main__":
    # Create alert system
    alert_system = AlertSystem()

    # Add alerts
    alert_system.add_threshold_alert(
        name="high_loss", metric="loss", threshold=2.0, condition="greater_than", severity="warning"
    )

    alert_system.add_anomaly_detector(
        name="loss_anomaly", metric="loss", sensitivity=2.5, window_size=30
    )

    alert_system.add_trend_alert(
        name="increasing_memory",
        metric="gpu_memory",
        trend_threshold=100,  # MB per step
        direction="increasing",
        severity="error",
    )

    # Add Slack notifier (example)
    # slack_webhook = os.environ.get("SLACK_WEBHOOK_URL")
    # if slack_webhook:
    #     alert_system.add_notifier(SlackNotifier(slack_webhook))

    # Simulate metrics
    import random

    for step in range(100):
        metrics = {
            "loss": 2.5 - step * 0.02 + random.gauss(0, 0.1),
            "gpu_memory": 10000 + step * 50 + random.randint(-100, 100),
            "learning_rate": 1e-4 * (0.95 ** (step // 20)),
        }

        # Inject anomaly
        if step == 50:
            metrics["loss"] = 5.0

        # Check metrics
        alert_system.check_metrics(metrics, {"step": step})

        time.sleep(0.1)

    # Print summary
    print("\nAlert Summary:")
    print(json.dumps(alert_system.get_alert_summary(), indent=2))
