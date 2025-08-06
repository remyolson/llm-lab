"""
Analytics engine for red team attack campaigns and vulnerability trends.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.models import CampaignResult
from ..workflows.scoring import SessionScore

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Direction of trend."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class MetricType(Enum):
    """Types of metrics to track."""

    SUCCESS_RATE = "success_rate"
    VULNERABILITY_COUNT = "vulnerability_count"
    SEVERITY_SCORE = "severity_score"
    RESPONSE_TIME = "response_time"
    DETECTION_RATE = "detection_rate"
    EVASION_SUCCESS = "evasion_success"


@dataclass
class TrendAnalysis:
    """Analysis of a metric trend."""

    metric: MetricType
    direction: TrendDirection
    change_percentage: float
    current_value: float
    previous_value: float
    trend_strength: float  # 0.0 to 1.0
    data_points: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)

    def get_summary(self) -> str:
        """Get trend summary."""
        direction_symbol = (
            "↑"
            if self.direction == TrendDirection.INCREASING
            else "↓"
            if self.direction == TrendDirection.DECREASING
            else "→"
            if self.direction == TrendDirection.STABLE
            else "↕"
        )

        return (
            f"{self.metric.value} {direction_symbol} "
            f"{self.change_percentage:+.1f}% "
            f"({self.previous_value:.2f} → {self.current_value:.2f})"
        )


@dataclass
class ModelWeakness:
    """Identified weakness in a model."""

    model_name: str
    weakness_type: str
    severity: str
    frequency: int
    first_detected: datetime
    last_detected: datetime
    affected_scenarios: List[str] = field(default_factory=list)
    exploitation_rate: float = 0.0
    remediation_suggested: Optional[str] = None

    def get_risk_score(self) -> float:
        """Calculate risk score for the weakness."""
        severity_scores = {"critical": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25}

        base_score = severity_scores.get(self.severity, 0.5)
        frequency_factor = min(1.0, self.frequency / 10)
        exploitation_factor = self.exploitation_rate

        return base_score * (0.5 + 0.3 * frequency_factor + 0.2 * exploitation_factor)


@dataclass
class AttackPattern:
    """Pattern identified in attack success."""

    pattern_id: str
    name: str
    description: str
    attack_types: List[str]
    success_rate: float
    models_affected: List[str]
    techniques_used: List[str]
    discovered_date: datetime
    occurrences: int = 0

    def is_universal(self, total_models: int) -> bool:
        """Check if pattern affects all models."""
        return len(self.models_affected) == total_models


class AnalyticsEngine:
    """
    Core analytics engine for red team data analysis.
    """

    def __init__(self):
        """Initialize analytics engine."""
        # Data storage
        self._campaign_data: List[CampaignResult] = []
        self._session_data: List[SessionScore] = []
        self._time_series_data: Dict[MetricType, List[Tuple[datetime, float]]] = defaultdict(list)

        # Cached analytics
        self._weakness_cache: Dict[str, List[ModelWeakness]] = {}
        self._pattern_cache: List[AttackPattern] = []
        self._trend_cache: Dict[MetricType, TrendAnalysis] = {}

        logger.info("AnalyticsEngine initialized")

    def ingest_campaign_result(self, result: CampaignResult):
        """Ingest campaign result for analysis."""
        self._campaign_data.append(result)

        # Extract metrics
        self._extract_time_series_metrics(result)

        # Update caches
        self._update_weakness_analysis(result)
        self._update_pattern_detection(result)

        logger.debug("Ingested campaign result: %s", result.campaign_id)

    def ingest_session_score(self, score: SessionScore):
        """Ingest session score for analysis."""
        self._session_data.append(score)

        # Extract session metrics
        if score.end_time:
            self._time_series_data[MetricType.SUCCESS_RATE].append(
                (score.end_time, score.success_rate)
            )
            self._time_series_data[MetricType.SEVERITY_SCORE].append(
                (score.end_time, score.average_severity)
            )

    def _extract_time_series_metrics(self, result: CampaignResult):
        """Extract time series metrics from campaign result."""
        if result.end_time:
            # Success rate
            self._time_series_data[MetricType.SUCCESS_RATE].append(
                (result.end_time, result.success_rate)
            )

            # Vulnerability count
            total_vulns = result.total_vulnerabilities
            self._time_series_data[MetricType.VULNERABILITY_COUNT].append(
                (result.end_time, total_vulns)
            )

    def _update_weakness_analysis(self, result: CampaignResult):
        """Update weakness analysis from campaign result."""
        model_name = result.model_name

        if model_name not in self._weakness_cache:
            self._weakness_cache[model_name] = []

        # Analyze sessions for weaknesses
        for session in result.sessions:
            for vuln in session.vulnerability_findings:
                weakness_type = vuln.get("type", "unknown")

                # Find or create weakness entry
                existing = None
                for weakness in self._weakness_cache[model_name]:
                    if weakness.weakness_type == weakness_type:
                        existing = weakness
                        break

                if existing:
                    # Update existing
                    existing.frequency += 1
                    existing.last_detected = datetime.now()
                    existing.affected_scenarios.append(result.scenario_name)
                else:
                    # Create new
                    weakness = ModelWeakness(
                        model_name=model_name,
                        weakness_type=weakness_type,
                        severity=vuln.get("severity", "medium"),
                        frequency=1,
                        first_detected=datetime.now(),
                        last_detected=datetime.now(),
                        affected_scenarios=[result.scenario_name],
                        exploitation_rate=session.success_rate,
                    )
                    self._weakness_cache[model_name].append(weakness)

    def _update_pattern_detection(self, result: CampaignResult):
        """Update attack pattern detection."""
        # Analyze successful attack chains
        for session in result.sessions:
            if session.overall_success_rate > 0.5:
                # Extract pattern
                techniques = []
                attack_types = []

                for step in session.executed_steps:
                    if step.get("status") == "success":
                        attack_types.append(step.get("attack_type", "unknown"))
                        techniques.extend(step.get("evasion_techniques", []))

                if attack_types:
                    # Create or update pattern
                    pattern_id = f"pattern_{len(attack_types)}_{hash(tuple(attack_types)) % 10000}"

                    existing = None
                    for pattern in self._pattern_cache:
                        if pattern.pattern_id == pattern_id:
                            existing = pattern
                            break

                    if existing:
                        existing.occurrences += 1
                        existing.success_rate = (
                            existing.success_rate * (existing.occurrences - 1)
                            + session.overall_success_rate
                        ) / existing.occurrences

                        if result.model_name not in existing.models_affected:
                            existing.models_affected.append(result.model_name)
                    else:
                        pattern = AttackPattern(
                            pattern_id=pattern_id,
                            name=f"Attack Chain {len(self._pattern_cache) + 1}",
                            description=f"Successful attack chain using {len(attack_types)} steps",
                            attack_types=attack_types,
                            success_rate=session.overall_success_rate,
                            models_affected=[result.model_name],
                            techniques_used=list(set(techniques)),
                            discovered_date=datetime.now(),
                            occurrences=1,
                        )
                        self._pattern_cache.append(pattern)

    def analyze_trends(self, metric: MetricType, window_hours: int = 24) -> TrendAnalysis:
        """
        Analyze trends for a specific metric.

        Args:
            metric: Metric to analyze
            window_hours: Time window for analysis

        Returns:
            TrendAnalysis object
        """
        # Check cache
        if metric in self._trend_cache:
            cached = self._trend_cache[metric]
            if (datetime.now() - cached.timestamps[-1]).total_seconds() < 3600:
                return cached

        # Get data points
        data_points = self._time_series_data.get(metric, [])
        if not data_points:
            return TrendAnalysis(
                metric=metric,
                direction=TrendDirection.STABLE,
                change_percentage=0.0,
                current_value=0.0,
                previous_value=0.0,
                trend_strength=0.0,
            )

        # Filter by time window
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_data = [(ts, value) for ts, value in data_points if ts >= cutoff_time]

        if len(recent_data) < 2:
            return TrendAnalysis(
                metric=metric,
                direction=TrendDirection.STABLE,
                change_percentage=0.0,
                current_value=recent_data[0][1] if recent_data else 0.0,
                previous_value=recent_data[0][1] if recent_data else 0.0,
                trend_strength=0.0,
            )

        # Calculate trend
        timestamps = [ts for ts, _ in recent_data]
        values = [value for _, value in recent_data]

        # Simple linear regression for trend
        x = np.arange(len(values))
        y = np.array(values)

        # Calculate slope
        slope = np.polyfit(x, y, 1)[0] if len(values) > 1 else 0

        # Determine direction
        if abs(slope) < 0.01:
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING

        # Check volatility
        std_dev = np.std(values)
        mean_val = np.mean(values)
        cv = std_dev / mean_val if mean_val != 0 else 0

        if cv > 0.5:
            direction = TrendDirection.VOLATILE

        # Calculate change percentage
        current_value = values[-1]
        previous_value = values[0]
        change_percentage = (
            (current_value - previous_value) / previous_value * 100 if previous_value != 0 else 0
        )

        # Calculate trend strength
        trend_strength = min(1.0, abs(slope) / (mean_val if mean_val != 0 else 1))

        trend = TrendAnalysis(
            metric=metric,
            direction=direction,
            change_percentage=change_percentage,
            current_value=current_value,
            previous_value=previous_value,
            trend_strength=trend_strength,
            data_points=values,
            timestamps=timestamps,
        )

        # Cache result
        self._trend_cache[metric] = trend

        return trend

    def get_model_weaknesses(
        self, model_name: str, min_severity: Optional[str] = None
    ) -> List[ModelWeakness]:
        """
        Get identified weaknesses for a model.

        Args:
            model_name: Model to analyze
            min_severity: Minimum severity level

        Returns:
            List of model weaknesses
        """
        weaknesses = self._weakness_cache.get(model_name, [])

        if min_severity:
            severity_order = ["low", "medium", "high", "critical"]
            min_index = severity_order.index(min_severity)

            weaknesses = [w for w in weaknesses if severity_order.index(w.severity) >= min_index]

        # Sort by risk score
        weaknesses.sort(key=lambda w: w.get_risk_score(), reverse=True)

        return weaknesses

    def get_attack_patterns(
        self, min_success_rate: float = 0.5, universal_only: bool = False
    ) -> List[AttackPattern]:
        """
        Get identified attack patterns.

        Args:
            min_success_rate: Minimum success rate
            universal_only: Only return universal patterns

        Returns:
            List of attack patterns
        """
        patterns = [p for p in self._pattern_cache if p.success_rate >= min_success_rate]

        if universal_only:
            total_models = len(set(r.model_name for r in self._campaign_data))
            patterns = [p for p in patterns if p.is_universal(total_models)]

        # Sort by success rate and occurrences
        patterns.sort(key=lambda p: (p.success_rate, p.occurrences), reverse=True)

        return patterns

    def get_comparative_analysis(self, models: List[str]) -> Dict[str, Any]:
        """
        Get comparative analysis across models.

        Args:
            models: List of model names to compare

        Returns:
            Comparative analysis dictionary
        """
        comparison = {"models": models, "metrics": {}, "weaknesses": {}, "rankings": {}}

        # Calculate metrics for each model
        for model in models:
            model_campaigns = [c for c in self._campaign_data if c.model_name == model]

            if model_campaigns:
                # Average success rate
                avg_success = np.mean([c.success_rate for c in model_campaigns])

                # Total vulnerabilities
                total_vulns = sum(c.total_vulnerabilities for c in model_campaigns)

                # Average severity
                all_severities = []
                for campaign in model_campaigns:
                    for session in campaign.sessions:
                        if hasattr(session, "max_severity"):
                            all_severities.append(session.max_severity)

                avg_severity = np.mean(all_severities) if all_severities else 0

                comparison["metrics"][model] = {
                    "average_success_rate": avg_success,
                    "total_vulnerabilities": total_vulns,
                    "average_severity": avg_severity,
                    "campaigns_run": len(model_campaigns),
                }

                # Weaknesses
                comparison["weaknesses"][model] = len(self._weakness_cache.get(model, []))

        # Create rankings
        if comparison["metrics"]:
            # Rank by security (lower success rate is better)
            security_ranking = sorted(
                comparison["metrics"].items(), key=lambda x: x[1]["average_success_rate"]
            )
            comparison["rankings"]["security"] = [m[0] for m in security_ranking]

            # Rank by vulnerabilities (fewer is better)
            vuln_ranking = sorted(
                comparison["metrics"].items(), key=lambda x: x[1]["total_vulnerabilities"]
            )
            comparison["rankings"]["vulnerabilities"] = [m[0] for m in vuln_ranking]

        return comparison

    def generate_insights(self) -> List[str]:
        """Generate actionable insights from analytics."""
        insights = []

        # Trend insights
        for metric_type in MetricType:
            trend = self.analyze_trends(metric_type, window_hours=24)

            if trend.direction == TrendDirection.INCREASING:
                if metric_type in [MetricType.VULNERABILITY_COUNT, MetricType.SUCCESS_RATE]:
                    insights.append(
                        f"⚠️ {metric_type.value} is increasing by {trend.change_percentage:.1f}% - "
                        "security posture may be degrading"
                    )
            elif trend.direction == TrendDirection.DECREASING:
                if metric_type == MetricType.SUCCESS_RATE:
                    insights.append(
                        f"✅ Attack success rate decreasing by {abs(trend.change_percentage):.1f}% - "
                        "defenses are improving"
                    )

        # Weakness insights
        all_weaknesses = []
        for weaknesses in self._weakness_cache.values():
            all_weaknesses.extend(weaknesses)

        critical_weaknesses = [w for w in all_weaknesses if w.severity == "critical"]
        if critical_weaknesses:
            insights.append(
                f"🔴 {len(critical_weaknesses)} critical weaknesses identified - "
                "immediate remediation required"
            )

        # Pattern insights
        universal_patterns = [p for p in self._pattern_cache if len(p.models_affected) > 1]
        if universal_patterns:
            insights.append(
                f"🔍 {len(universal_patterns)} attack patterns work across multiple models - "
                "consider implementing universal defenses"
            )

        # Success rate insights
        recent_campaigns = (
            self._campaign_data[-10:] if len(self._campaign_data) >= 10 else self._campaign_data
        )
        if recent_campaigns:
            avg_success = np.mean([c.success_rate for c in recent_campaigns])
            if avg_success > 0.5:
                insights.append(
                    f"⚠️ Recent average attack success rate is {avg_success:.1%} - "
                    "significant security improvements needed"
                )

        return insights

    def export_analytics(self) -> Dict[str, Any]:
        """Export all analytics data."""
        return {
            "summary": {
                "total_campaigns": len(self._campaign_data),
                "total_sessions": len(self._session_data),
                "models_tested": len(self._weakness_cache),
                "patterns_identified": len(self._pattern_cache),
            },
            "trends": {
                metric.value: self.analyze_trends(metric).get_summary() for metric in MetricType
            },
            "top_weaknesses": [
                {
                    "model": model,
                    "weakness": w.weakness_type,
                    "severity": w.severity,
                    "frequency": w.frequency,
                    "risk_score": w.get_risk_score(),
                }
                for model, weaknesses in self._weakness_cache.items()
                for w in sorted(weaknesses, key=lambda x: x.get_risk_score(), reverse=True)[:3]
            ],
            "top_patterns": [
                {
                    "name": p.name,
                    "success_rate": p.success_rate,
                    "models_affected": len(p.models_affected),
                    "occurrences": p.occurrences,
                }
                for p in self.get_attack_patterns()[:5]
            ],
            "insights": self.generate_insights(),
            "export_time": datetime.now().isoformat(),
        }


class TrendAnalyzer:
    """
    Specialized analyzer for vulnerability and attack trends.
    """

    def __init__(self, analytics_engine: AnalyticsEngine):
        """Initialize trend analyzer."""
        self.analytics = analytics_engine

    def analyze_vulnerability_trends(
        self,
        time_windows: List[int] = [24, 168, 720],  # 1 day, 1 week, 1 month
    ) -> Dict[str, List[TrendAnalysis]]:
        """
        Analyze vulnerability trends across multiple time windows.

        Args:
            time_windows: List of time windows in hours

        Returns:
            Dictionary of trend analyses by window
        """
        trends = {}

        for window in time_windows:
            window_name = self._get_window_name(window)
            trends[window_name] = []

            # Analyze each metric
            for metric in [MetricType.VULNERABILITY_COUNT, MetricType.SEVERITY_SCORE]:
                trend = self.analytics.analyze_trends(metric, window)
                trends[window_name].append(trend)

        return trends

    def _get_window_name(self, hours: int) -> str:
        """Get human-readable window name."""
        if hours <= 24:
            return f"{hours}h"
        elif hours <= 168:
            return f"{hours // 24}d"
        else:
            return f"{hours // 168}w"

    def predict_future_risk(self, horizon_hours: int = 24) -> Dict[str, Any]:
        """
        Predict future risk based on current trends.

        Args:
            horizon_hours: Prediction horizon in hours

        Returns:
            Risk prediction
        """
        # Get current trends
        success_trend = self.analytics.analyze_trends(MetricType.SUCCESS_RATE, 24)
        vuln_trend = self.analytics.analyze_trends(MetricType.VULNERABILITY_COUNT, 24)

        # Simple linear projection
        if success_trend.direction == TrendDirection.INCREASING:
            projected_success = min(
                1.0, success_trend.current_value * (1 + success_trend.change_percentage / 100)
            )
        else:
            projected_success = max(
                0.0, success_trend.current_value * (1 + success_trend.change_percentage / 100)
            )

        # Risk calculation
        risk_score = projected_success * 100

        risk_level = (
            "CRITICAL"
            if risk_score > 70
            else "HIGH"
            if risk_score > 50
            else "MEDIUM"
            if risk_score > 30
            else "LOW"
        )

        return {
            "horizon_hours": horizon_hours,
            "projected_success_rate": projected_success,
            "projected_risk_score": risk_score,
            "risk_level": risk_level,
            "confidence": success_trend.trend_strength,
            "recommendation": self._get_risk_recommendation(risk_level),
        }

    def _get_risk_recommendation(self, risk_level: str) -> str:
        """Get recommendation based on risk level."""
        recommendations = {
            "CRITICAL": "Immediate action required - implement emergency security measures",
            "HIGH": "Urgent review needed - strengthen defenses and monitoring",
            "MEDIUM": "Schedule security review - consider additional controls",
            "LOW": "Continue monitoring - maintain current security posture",
        }
        return recommendations.get(risk_level, "Monitor situation")


class AnalyticsDashboard:
    """
    Dashboard interface for analytics visualization.
    """

    def __init__(self, analytics_engine: AnalyticsEngine, trend_analyzer: TrendAnalyzer):
        """Initialize analytics dashboard."""
        self.analytics = analytics_engine
        self.trends = trend_analyzer

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data."""
        return {
            "overview": self._get_overview(),
            "trends": self._get_trend_charts(),
            "weaknesses": self._get_weakness_matrix(),
            "patterns": self._get_pattern_analysis(),
            "models": self._get_model_comparison(),
            "insights": self.analytics.generate_insights(),
            "risk_forecast": self.trends.predict_future_risk(),
            "timestamp": datetime.now().isoformat(),
        }

    def _get_overview(self) -> Dict[str, Any]:
        """Get overview metrics."""
        recent_campaigns = self.analytics._campaign_data[-10:]

        if recent_campaigns:
            avg_success = np.mean([c.success_rate for c in recent_campaigns])
            total_vulns = sum(c.total_vulnerabilities for c in recent_campaigns)
        else:
            avg_success = 0
            total_vulns = 0

        return {
            "total_campaigns": len(self.analytics._campaign_data),
            "recent_success_rate": avg_success,
            "total_vulnerabilities": total_vulns,
            "models_tested": len(self.analytics._weakness_cache),
            "patterns_detected": len(self.analytics._pattern_cache),
        }

    def _get_trend_charts(self) -> Dict[str, Any]:
        """Get trend chart data."""
        charts = {}

        for metric in MetricType:
            trend = self.analytics.analyze_trends(metric, 24)
            charts[metric.value] = {
                "current": trend.current_value,
                "change": trend.change_percentage,
                "direction": trend.direction.value,
                "data_points": trend.data_points[-20:],  # Last 20 points
                "timestamps": [ts.isoformat() for ts in trend.timestamps[-20:]],
            }

        return charts

    def _get_weakness_matrix(self) -> List[Dict[str, Any]]:
        """Get weakness matrix data."""
        matrix = []

        for model, weaknesses in self.analytics._weakness_cache.items():
            for weakness in weaknesses:
                matrix.append(
                    {
                        "model": model,
                        "weakness": weakness.weakness_type,
                        "severity": weakness.severity,
                        "frequency": weakness.frequency,
                        "risk_score": weakness.get_risk_score(),
                        "last_seen": weakness.last_detected.isoformat(),
                    }
                )

        # Sort by risk score
        matrix.sort(key=lambda x: x["risk_score"], reverse=True)

        return matrix[:20]  # Top 20 weaknesses

    def _get_pattern_analysis(self) -> List[Dict[str, Any]]:
        """Get pattern analysis data."""
        patterns = self.analytics.get_attack_patterns()

        return [
            {
                "pattern_id": p.pattern_id,
                "name": p.name,
                "success_rate": p.success_rate,
                "occurrences": p.occurrences,
                "models_affected": len(p.models_affected),
                "techniques": p.techniques_used[:3],  # Top 3 techniques
            }
            for p in patterns[:10]  # Top 10 patterns
        ]

    def _get_model_comparison(self) -> Dict[str, Any]:
        """Get model comparison data."""
        all_models = list(self.analytics._weakness_cache.keys())

        if not all_models:
            return {}

        return self.analytics.get_comparative_analysis(all_models)
