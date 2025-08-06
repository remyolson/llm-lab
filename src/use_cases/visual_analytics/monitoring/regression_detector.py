"""
Regression Detector with Alerting

This module provides automatic performance regression detection with alerting
capabilities and root cause analysis.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegressionAlert:
    """Alert for detected regression."""

    timestamp: datetime
    metric_name: str
    baseline_value: float
    current_value: float
    regression_percent: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    root_causes: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class RegressionDetector:
    """Detector for performance regressions."""

    def __init__(
        self,
        threshold_percent: float = 5.0,
        window_size: int = 100,
        alert_callback: Optional[callable] = None,
    ):
        """Initialize regression detector.

        Args:
            threshold_percent: Regression threshold percentage
            window_size: Window size for trend analysis
            alert_callback: Callback for alerts
        """
        self.threshold_percent = threshold_percent
        self.window_size = window_size
        self.alert_callback = alert_callback

        # Metrics history
        self.metrics_history = {}
        self.baselines = {}
        self.alerts = []

    def update_metric(
        self, metric_name: str, value: float, timestamp: Optional[datetime] = None
    ) -> Optional[RegressionAlert]:
        """Update metric and check for regression.

        Args:
            metric_name: Name of metric
            value: Current value
            timestamp: Timestamp

        Returns:
            RegressionAlert if regression detected
        """
        timestamp = timestamp or datetime.now()

        # Initialize history if needed
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = deque(maxlen=self.window_size)
            self.baselines[metric_name] = value

        # Add to history
        self.metrics_history[metric_name].append((timestamp, value))

        # Check for regression
        alert = self._check_regression(metric_name, value, timestamp)

        if alert:
            self.alerts.append(alert)
            if self.alert_callback:
                self.alert_callback(alert)

        return alert

    def _check_regression(
        self, metric_name: str, current_value: float, timestamp: datetime
    ) -> Optional[RegressionAlert]:
        """Check if current value represents a regression.

        Args:
            metric_name: Metric name
            current_value: Current value
            timestamp: Timestamp

        Returns:
            RegressionAlert if regression detected
        """
        baseline = self.baselines.get(metric_name)

        if baseline is None:
            return None

        # Calculate regression
        if baseline != 0:
            regression_pct = ((baseline - current_value) / baseline) * 100
        else:
            regression_pct = 0

        # Check threshold
        if abs(regression_pct) < self.threshold_percent:
            return None

        # Determine severity
        if abs(regression_pct) >= 20:
            severity = "critical"
        elif abs(regression_pct) >= 15:
            severity = "high"
        elif abs(regression_pct) >= 10:
            severity = "medium"
        else:
            severity = "low"

        # Analyze root causes
        root_causes = self._analyze_root_causes(metric_name, current_value)

        # Generate recommendations
        recommendations = self._generate_recommendations(metric_name, regression_pct)

        return RegressionAlert(
            timestamp=timestamp,
            metric_name=metric_name,
            baseline_value=baseline,
            current_value=current_value,
            regression_percent=regression_pct,
            severity=severity,
            root_causes=root_causes,
            recommendations=recommendations,
        )

    def _analyze_root_causes(self, metric_name: str, current_value: float) -> List[str]:
        """Analyze potential root causes.

        Args:
            metric_name: Metric name
            current_value: Current value

        Returns:
            List of potential causes
        """
        causes = []

        # Analyze trend
        history = list(self.metrics_history.get(metric_name, []))
        if len(history) > 2:
            recent_values = [v for _, v in history[-5:]]
            if all(v < current_value for v in recent_values[:-1]):
                causes.append("Sudden spike detected")
            elif np.std(recent_values) > np.mean(recent_values) * 0.2:
                causes.append("High variance in recent metrics")

        # Check for specific metric patterns
        if "loss" in metric_name.lower() and current_value > 1.0:
            causes.append("Loss divergence detected")
        elif "accuracy" in metric_name.lower() and current_value < 0.5:
            causes.append("Accuracy below random baseline")

        return causes if causes else ["Unknown cause - investigate model and data"]

    def _generate_recommendations(self, metric_name: str, regression_pct: float) -> List[str]:
        """Generate recommendations.

        Args:
            metric_name: Metric name
            regression_pct: Regression percentage

        Returns:
            List of recommendations
        """
        recommendations = []

        if abs(regression_pct) >= 15:
            recommendations.append("Revert to previous model version")
            recommendations.append("Review recent changes")

        if "loss" in metric_name.lower():
            recommendations.append("Check learning rate schedule")
            recommendations.append("Verify data preprocessing")
        elif "accuracy" in metric_name.lower():
            recommendations.append("Increase training epochs")
            recommendations.append("Review data quality")

        recommendations.append("Run diagnostic tests")

        return recommendations

    def get_trend_analysis(self, metric_name: str, lookback_hours: int = 24) -> Dict[str, Any]:
        """Analyze metric trends.

        Args:
            metric_name: Metric name
            lookback_hours: Hours to look back

        Returns:
            Trend analysis results
        """
        if metric_name not in self.metrics_history:
            return {}

        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        recent_data = [(t, v) for t, v in self.metrics_history[metric_name] if t >= cutoff_time]

        if not recent_data:
            return {}

        values = [v for _, v in recent_data]

        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "trend": "increasing" if values[-1] > values[0] else "decreasing",
            "volatility": np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,
        }
