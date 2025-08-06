"""
Alerting system for the monitoring infrastructure.

This module provides comprehensive alerting capabilities including alert rules,
notification channels, alert grouping, and escalation policies for performance
monitoring and regression detection.
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import logging
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp

from .database import DatabaseManager
from .regression_detector import RegressionResult

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertType(Enum):
    """Types of alerts that can be generated."""

    PERFORMANCE_REGRESSION = "performance_regression"
    BENCHMARK_FAILURE = "benchmark_failure"
    MODEL_UNAVAILABLE = "model_unavailable"
    HIGH_ERROR_RATE = "high_error_rate"
    COST_ANOMALY = "cost_anomaly"
    LATENCY_SPIKE = "latency_spike"
    SYSTEM_ERROR = "system_error"


class NotificationChannel(Enum):
    """Available notification channels."""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    CONSOLE = "console"
    DATABASE = "database"


@dataclass
class AlertRule:
    """Configuration for an alert rule."""

    rule_id: str
    name: str
    description: str
    alert_type: AlertType
    severity: AlertSeverity
    enabled: bool = True
    conditions: Dict[str, Any] = None
    notification_channels: List[NotificationChannel] = None
    cooldown_minutes: int = 60  # Minimum time between same alerts
    auto_resolve_minutes: Optional[int] = None
    escalation_rules: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}
        if self.notification_channels is None:
            self.notification_channels = [NotificationChannel.DATABASE]


@dataclass
class Alert:
    """Represents an active alert."""

    alert_id: str
    rule_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    model_id: Optional[int] = None
    run_id: Optional[int] = None
    metric_type: Optional[str] = None
    trigger_value: Optional[float] = None
    threshold_value: Optional[float] = None
    baseline_value: Optional[float] = None
    timestamp: datetime = None
    status: str = "active"
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class NotificationHandler(ABC):
    """Abstract base class for notification handlers."""

    @abstractmethod
    async def send_notification(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send notification for an alert."""
        pass


class EmailNotificationHandler(NotificationHandler):
    """Email notification handler."""

    async def send_notification(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send email notification."""
        try:
            smtp_server = config.get("smtp_server", "localhost")
            smtp_port = config.get("smtp_port", 587)
            username = config.get("username")
            password = config.get("password")
            from_email = config.get("from_email", "monitoring@llmlab.com")
            to_emails = config.get("to_emails", [])

            if not to_emails:
                logger.warning("No email recipients configured")
                return False

            # Create message
            msg = MIMEMultipart()
            msg["From"] = from_email
            msg["To"] = ", ".join(to_emails)
            msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.title}"

            # Create email body
            body = self._create_email_body(alert)
            msg.attach(MIMEText(body, "html"))

            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if username and password:
                    server.starttls()
                    server.login(username, password)

                server.send_message(msg)

            logger.info(f"Email notification sent for alert {alert.alert_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False

    def _create_email_body(self, alert: Alert) -> str:
        """Create HTML email body."""
        severity_colors = {"critical": "#dc3545", "warning": "#ffc107", "info": "#17a2b8"}

        color = severity_colors.get(alert.severity.value, "#6c757d")

        return f"""
        <html>
        <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px;">
                <div style="background-color: {color}; color: white; padding: 20px; border-radius: 5px 5px 0 0;">
                    <h2 style="margin: 0;">{alert.title}</h2>
                    <p style="margin: 5px 0 0 0; font-size: 14px;">
                        Severity: {alert.severity.value.upper()} | Type: {alert.alert_type.value}
                    </p>
                </div>

                <div style="border: 1px solid #ddd; border-top: none; padding: 20px; border-radius: 0 0 5px 5px;">
                    <p><strong>Message:</strong></p>
                    <p>{alert.message}</p>

                    <table style="width: 100%; border-collapse: collapse; margin-top: 20px;">
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Alert ID:</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #eee;">{alert.alert_id}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Timestamp:</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #eee;">{alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}</td>
                        </tr>
                        {f'<tr><td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Model ID:</strong></td><td style="padding: 8px; border-bottom: 1px solid #eee;">{alert.model_id}</td></tr>' if alert.model_id else ""}
                        {f'<tr><td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Metric:</strong></td><td style="padding: 8px; border-bottom: 1px solid #eee;">{alert.metric_type}</td></tr>' if alert.metric_type else ""}
                        {f'<tr><td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Trigger Value:</strong></td><td style="padding: 8px; border-bottom: 1px solid #eee;">{alert.trigger_value}</td></tr>' if alert.trigger_value is not None else ""}
                        {f'<tr><td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Threshold:</strong></td><td style="padding: 8px; border-bottom: 1px solid #eee;">{alert.threshold_value}</td></tr>' if alert.threshold_value is not None else ""}
                    </table>

                    <p style="margin-top: 20px; font-size: 12px; color: #666;">
                        This alert was generated by LLM Lab Monitoring System at {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}
                    </p>
                </div>
            </div>
        </body>
        </html>
        """


class SlackNotificationHandler(NotificationHandler):
    """Slack notification handler."""

    async def send_notification(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send Slack notification."""
        try:
            webhook_url = config.get("webhook_url")
            if not webhook_url:
                logger.error("Slack webhook URL not configured")
                return False

            # Create Slack message
            color_map = {"critical": "danger", "warning": "warning", "info": "good"}

            attachment = {
                "color": color_map.get(alert.severity.value, "good"),
                "title": alert.title,
                "text": alert.message,
                "fields": [
                    {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                    {"title": "Type", "value": alert.alert_type.value, "short": True},
                    {"title": "Alert ID", "value": alert.alert_id, "short": True},
                    {
                        "title": "Timestamp",
                        "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "short": True,
                    },
                ],
                "footer": "LLM Lab Monitoring",
                "ts": int(alert.timestamp.timestamp()),
            }

            # Add model-specific fields
            if alert.model_id:
                attachment["fields"].append(
                    {"title": "Model ID", "value": str(alert.model_id), "short": True}
                )
            if alert.metric_type:
                attachment["fields"].append(
                    {"title": "Metric", "value": alert.metric_type, "short": True}
                )
            if alert.trigger_value is not None:
                attachment["fields"].append(
                    {"title": "Trigger Value", "value": f"{alert.trigger_value:.4f}", "short": True}
                )

            payload = {
                "username": "LLM Lab Monitor",
                "icon_emoji": ":warning:",
                "attachments": [attachment],
            }

            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Slack notification sent for alert {alert.alert_id}")
                        return True
                    else:
                        logger.error(f"Slack notification failed: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False


class WebhookNotificationHandler(NotificationHandler):
    """Generic webhook notification handler."""

    async def send_notification(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send webhook notification."""
        try:
            webhook_url = config.get("url")
            if not webhook_url:
                logger.error("Webhook URL not configured")
                return False

            # Prepare payload
            payload = {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "alert_type": alert.alert_type.value,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "status": alert.status,
            }

            # Add optional fields
            if alert.model_id:
                payload["model_id"] = alert.model_id
            if alert.run_id:
                payload["run_id"] = alert.run_id
            if alert.metric_type:
                payload["metric_type"] = alert.metric_type
            if alert.trigger_value is not None:
                payload["trigger_value"] = alert.trigger_value
            if alert.threshold_value is not None:
                payload["threshold_value"] = alert.threshold_value
            if alert.baseline_value is not None:
                payload["baseline_value"] = alert.baseline_value
            if alert.metadata:
                payload["metadata"] = alert.metadata

            headers = config.get("headers", {"Content-Type": "application/json"})
            timeout = config.get("timeout", 30)

            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    if 200 <= response.status < 300:
                        logger.info(f"Webhook notification sent for alert {alert.alert_id}")
                        return True
                    else:
                        logger.error(f"Webhook notification failed: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False


class ConsoleNotificationHandler(NotificationHandler):
    """Console/logging notification handler."""

    async def send_notification(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Log alert to console."""
        try:
            log_level = config.get("log_level", "INFO").upper()

            message = f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.message}"

            if log_level == "ERROR" or alert.severity == AlertSeverity.CRITICAL:
                logger.error(message)
            elif log_level == "WARNING" or alert.severity == AlertSeverity.WARNING:
                logger.warning(message)
            else:
                logger.info(message)

            return True

        except Exception as e:
            logger.error(f"Failed to log alert to console: {e}")
            return False


class AlertManager:
    """Manages alerts, rules, and notifications."""

    def __init__(self, database_manager: DatabaseManager):
        """
        Initialize alert manager.

        Args:
            database_manager: Database manager for storing alerts
        """
        self.db_manager = database_manager
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.notification_handlers = {
            NotificationChannel.EMAIL: EmailNotificationHandler(),
            NotificationChannel.SLACK: SlackNotificationHandler(),
            NotificationChannel.WEBHOOK: WebhookNotificationHandler(),
            NotificationChannel.CONSOLE: ConsoleNotificationHandler(),
        }
        self.notification_configs: Dict[NotificationChannel, Dict[str, Any]] = {}

        # Default alert rules
        self._setup_default_rules()

        logger.info("Alert manager initialized")

    def _setup_default_rules(self):
        """Set up default alert rules."""
        default_rules = [
            AlertRule(
                rule_id="performance_regression_critical",
                name="Critical Performance Regression",
                description="Detect critical performance regressions",
                alert_type=AlertType.PERFORMANCE_REGRESSION,
                severity=AlertSeverity.CRITICAL,
                conditions={
                    "change_percent_threshold": 0.1,  # 10% change
                    "confidence_threshold": 0.8,
                },
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                cooldown_minutes=30,
            ),
            AlertRule(
                rule_id="benchmark_failure",
                name="Benchmark Execution Failure",
                description="Alert when benchmark execution fails",
                alert_type=AlertType.BENCHMARK_FAILURE,
                severity=AlertSeverity.WARNING,
                conditions={},
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.CONSOLE],
                cooldown_minutes=15,
            ),
            AlertRule(
                rule_id="high_error_rate",
                name="High Error Rate",
                description="Alert when error rate exceeds threshold",
                alert_type=AlertType.HIGH_ERROR_RATE,
                severity=AlertSeverity.WARNING,
                conditions={
                    "error_rate_threshold": 0.05  # 5% error rate
                },
                notification_channels=[NotificationChannel.CONSOLE],
                cooldown_minutes=60,
            ),
        ]

        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule

    def add_alert_rule(self, rule: AlertRule) -> bool:
        """Add or update an alert rule."""
        try:
            self.alert_rules[rule.rule_id] = rule
            logger.info(f"Added alert rule: {rule.rule_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add alert rule {rule.rule_id}: {e}")
            return False

    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False

    def configure_notification_channel(
        self, channel: NotificationChannel, config: Dict[str, Any]
    ) -> None:
        """Configure notification channel settings."""
        self.notification_configs[channel] = config
        logger.info(f"Configured notification channel: {channel.value}")

    async def process_regression_results(
        self, regression_results: List[RegressionResult]
    ) -> List[Alert]:
        """Process regression detection results and generate alerts."""
        alerts_generated = []

        for result in regression_results:
            if not result.regression_detected:
                continue

            # Find applicable alert rules
            applicable_rules = [
                rule
                for rule in self.alert_rules.values()
                if (
                    rule.alert_type == AlertType.PERFORMANCE_REGRESSION
                    and rule.enabled
                    and self._rule_conditions_met(rule, result)
                )
            ]

            for rule in applicable_rules:
                # Check cooldown period
                if self._is_in_cooldown(rule.rule_id, result.model_id):
                    continue

                # Create alert
                alert = self._create_regression_alert(rule, result)

                # Send notifications
                await self._send_alert_notifications(alert, rule)

                # Store in database
                self._store_alert_in_database(alert)

                # Track active alert
                self.active_alerts[alert.alert_id] = alert

                alerts_generated.append(alert)

        return alerts_generated

    async def process_benchmark_failure(
        self, run_id: int, model_id: int, error_message: str
    ) -> Optional[Alert]:
        """Process benchmark failure and generate alert if needed."""
        # Find applicable alert rules
        applicable_rules = [
            rule
            for rule in self.alert_rules.values()
            if (rule.alert_type == AlertType.BENCHMARK_FAILURE and rule.enabled)
        ]

        if not applicable_rules:
            return None

        rule = applicable_rules[0]  # Use first applicable rule

        # Check cooldown
        if self._is_in_cooldown(rule.rule_id, model_id):
            return None

        # Create alert
        alert_id = f"benchmark_failure_{run_id}_{int(datetime.utcnow().timestamp())}"
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            alert_type=AlertType.BENCHMARK_FAILURE,
            severity=rule.severity,
            title="Benchmark Execution Failed",
            message=f"Benchmark run {run_id} failed: {error_message}",
            model_id=model_id,
            run_id=run_id,
        )

        # Send notifications
        await self._send_alert_notifications(alert, rule)

        # Store in database
        self._store_alert_in_database(alert)

        # Track active alert
        self.active_alerts[alert.alert_id] = alert

        return alert

    def _rule_conditions_met(self, rule: AlertRule, result: RegressionResult) -> bool:
        """Check if alert rule conditions are met."""
        conditions = rule.conditions

        # Check change percentage threshold
        change_threshold = conditions.get("change_percent_threshold", 0.05)
        if abs(result.change_percent) < change_threshold:
            return False

        # Check confidence threshold
        confidence_threshold = conditions.get("confidence_threshold", 0.5)
        if result.confidence_score < confidence_threshold:
            return False

        # Check severity matching
        if rule.severity == AlertSeverity.CRITICAL and result.severity != "critical":
            return False

        return True

    def _is_in_cooldown(self, rule_id: str, model_id: int) -> bool:
        """Check if alert rule is in cooldown period for a model."""
        rule = self.alert_rules.get(rule_id)
        if not rule or rule.cooldown_minutes <= 0:
            return False

        cooldown_cutoff = datetime.utcnow() - timedelta(minutes=rule.cooldown_minutes)

        # Check database for recent alerts
        recent_alerts = self.db_manager.get_active_alerts(model_id=model_id)

        for alert_record in recent_alerts:
            if (
                alert_record.timestamp > cooldown_cutoff
                and alert_record.alert_type == rule.alert_type.value
            ):
                return True

        return False

    def _create_regression_alert(self, rule: AlertRule, result: RegressionResult) -> Alert:
        """Create alert from regression result."""
        alert_id = f"regression_{result.model_id}_{result.metric_type}_{result.metric_name}_{int(datetime.utcnow().timestamp())}"

        # Create descriptive title and message
        direction = "decreased" if result.change_percent < 0 else "increased"
        title = f"Performance Regression Detected: {result.metric_name}"

        message = (
            f"Model {result.model_id} {result.metric_name} has {direction} by "
            f"{abs(result.change_percent) * 100:.1f}% "
            f"(from {result.baseline_value:.4f} to {result.current_value:.4f}). "
            f"Detection confidence: {result.confidence_score:.2f}"
        )

        return Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            alert_type=rule.alert_type,
            severity=AlertSeverity(result.severity),
            title=title,
            message=message,
            model_id=result.model_id,
            metric_type=result.metric_type,
            trigger_value=result.current_value,
            threshold_value=result.baseline_value
            * (1 + rule.conditions.get("change_percent_threshold", 0.05)),
            baseline_value=result.baseline_value,
            metadata={
                "detection_method": result.detection_method.value,
                "confidence_score": result.confidence_score,
                "p_value": result.p_value,
                "statistical_details": result.statistical_details,
            },
        )

    async def _send_alert_notifications(self, alert: Alert, rule: AlertRule) -> None:
        """Send notifications for an alert."""
        notification_status = {}

        for channel in rule.notification_channels:
            try:
                handler = self.notification_handlers.get(channel)
                if not handler:
                    logger.warning(f"No handler available for channel: {channel.value}")
                    notification_status[channel.value] = {"status": "failed", "error": "No handler"}
                    continue

                config = self.notification_configs.get(channel, {})
                success = await handler.send_notification(alert, config)

                notification_status[channel.value] = {
                    "status": "success" if success else "failed",
                    "timestamp": datetime.utcnow().isoformat(),
                }

            except Exception as e:
                logger.error(f"Failed to send notification via {channel.value}: {e}")
                notification_status[channel.value] = {"status": "failed", "error": str(e)}

        # Update alert with notification status
        if alert.metadata is None:
            alert.metadata = {}
        alert.metadata["notification_status"] = notification_status

    def _store_alert_in_database(self, alert: Alert) -> None:
        """Store alert in database."""
        try:
            alert_data = {
                "timestamp": alert.timestamp,
                "alert_type": alert.alert_type.value,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "model_id": alert.model_id,
                "run_id": alert.run_id,
                "metric_type": alert.metric_type,
                "trigger_value": alert.trigger_value,
                "threshold_value": alert.threshold_value,
                "baseline_value": alert.baseline_value,
                "status": alert.status,
                "notification_channels": [
                    ch.value for ch in self.alert_rules[alert.rule_id].notification_channels
                ],
                "notification_status": alert.metadata.get("notification_status", {})
                if alert.metadata
                else {},
            }

            self.db_manager.create_alert(alert_data)
            logger.info(f"Alert {alert.alert_id} stored in database")

        except Exception as e:
            logger.error(f"Failed to store alert in database: {e}")

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        try:
            # Update database
            alert_record = None
            with self.db_manager.get_session() as session:
                from .models import AlertHistory

                alert_record = (
                    session.query(AlertHistory)
                    .filter_by(id=int(alert_id) if alert_id.isdigit() else None)
                    .first()
                )

            if alert_record:
                self.db_manager.acknowledge_alert(alert_record.id, acknowledged_by)

            # Update active alerts
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].status = "acknowledged"

            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True

        except Exception as e:
            logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
            return False

    def resolve_alert(
        self, alert_id: str, resolved_by: str, resolution_notes: Optional[str] = None
    ) -> bool:
        """Resolve an alert."""
        try:
            # Update database
            alert_record = None
            with self.db_manager.get_session() as session:
                from .models import AlertHistory

                alert_record = (
                    session.query(AlertHistory)
                    .filter_by(id=int(alert_id) if alert_id.isdigit() else None)
                    .first()
                )

            if alert_record:
                self.db_manager.resolve_alert(alert_record.id, resolved_by, resolution_notes)

            # Remove from active alerts
            if alert_id in self.active_alerts:
                del self.active_alerts[alert_id]

            logger.info(f"Alert {alert_id} resolved by {resolved_by}")
            return True

        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return False

    def get_active_alerts(self, model_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get active alerts."""
        alerts = self.db_manager.get_active_alerts(model_id=model_id)
        return [alert.to_dict() for alert in alerts]

    def get_alert_statistics(self, days_back: int = 7) -> Dict[str, Any]:
        """Get alert statistics for the specified period."""
        start_date = datetime.utcnow() - timedelta(days=days_back)

        with self.db_manager.get_session() as session:
            from sqlalchemy import and_, func

            from .models import AlertHistory

            # Total alerts by severity
            alerts_by_severity = (
                session.query(AlertHistory.severity, func.count(AlertHistory.id))
                .filter(AlertHistory.timestamp >= start_date)
                .group_by(AlertHistory.severity)
                .all()
            )

            # Total alerts by type
            alerts_by_type = (
                session.query(AlertHistory.alert_type, func.count(AlertHistory.id))
                .filter(AlertHistory.timestamp >= start_date)
                .group_by(AlertHistory.alert_type)
                .all()
            )

            # Resolution statistics
            total_alerts = (
                session.query(AlertHistory).filter(AlertHistory.timestamp >= start_date).count()
            )

            resolved_alerts = (
                session.query(AlertHistory)
                .filter(
                    and_(AlertHistory.timestamp >= start_date, AlertHistory.status == "resolved")
                )
                .count()
            )

            active_alerts = (
                session.query(AlertHistory).filter(AlertHistory.status == "active").count()
            )

        return {
            "period_days": days_back,
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "resolved_alerts": resolved_alerts,
            "resolution_rate": resolved_alerts / total_alerts if total_alerts > 0 else 0,
            "alerts_by_severity": dict(alerts_by_severity),
            "alerts_by_type": dict(alerts_by_type),
        }
