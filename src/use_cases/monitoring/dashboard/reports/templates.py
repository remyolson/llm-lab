"""
Report Templates

Predefined report templates for common monitoring scenarios including
daily summaries, weekly performance reports, and monthly analysis.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

from .generator import ReportTemplate


class DailySummaryTemplate(ReportTemplate):
    """Daily summary report template."""

    def __init__(self):
        super().__init__(
            name="daily_summary", description="Daily monitoring summary with key metrics and alerts"
        )

    def get_context(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Get context for daily summary template."""
        context = super().get_context(data, **kwargs)

        # Calculate daily metrics
        metrics = data.get("metrics_summary", {})
        performance = data.get("performance_data", {})
        costs = data.get("cost_breakdown", {})
        alerts = data.get("active_alerts", [])

        # Daily highlights
        context["highlights"] = {
            "total_requests": metrics.get("total_requests", 0),
            "avg_response_time": f"{metrics.get('avg_latency', 0) * 1000:.1f}ms",
            "success_rate": f"{self._calculate_avg_success_rate(performance):.1f}%",
            "total_cost": costs.get("total_cost", 0),
            "active_alerts": len([a for a in alerts if a.get("status") != "resolved"]),
            "critical_alerts": len([a for a in alerts if a.get("severity") == "critical"]),
        }

        # Top performing models
        context["top_models"] = self._get_top_models(performance, limit=5)

        # Cost by provider
        context["provider_costs"] = costs.get("provider_breakdown", {})

        # Recent alerts (last 24 hours)
        context["recent_alerts"] = self._filter_recent_alerts(alerts, hours=24)

        # Performance trends
        context["trends"] = self._calculate_trends(performance)

        return context

    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate data for daily summary."""
        required_keys = ["metrics_summary", "performance_data"]
        return all(key in data for key in required_keys)

    def _calculate_avg_success_rate(self, performance_data: Dict[str, Any]) -> float:
        """Calculate average success rate."""
        time_series = performance_data.get("time_series", [])
        if not time_series:
            return 0.0

        total_rate = sum(point.get("success_rate", 0) for point in time_series)
        return total_rate / len(time_series)

    def _get_top_models(self, performance_data: Dict[str, Any], limit: int = 5) -> List[Dict]:
        """Get top performing models."""
        models = performance_data.get("models", [])
        if not models:
            return []

        # Sort by success rate and response time
        sorted_models = sorted(
            models, key=lambda m: (m.get("success_rate", 0), -m.get("avg_latency", 1)), reverse=True
        )

        return sorted_models[:limit]

    def _filter_recent_alerts(self, alerts: List[Dict], hours: int = 24) -> List[Dict]:
        """Filter alerts from the last N hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_alerts = []

        for alert in alerts:
            try:
                alert_time = datetime.fromisoformat(alert["created_at"].replace("Z", "+00:00"))
                if alert_time >= cutoff_time:
                    recent_alerts.append(alert)
            except (KeyError, ValueError):
                continue

        # Sort by severity and time
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        recent_alerts.sort(
            key=lambda a: (
                severity_order.get(a.get("severity", "info"), 3),
                a.get("created_at", ""),
            )
        )

        return recent_alerts

    def _calculate_trends(self, performance_data: Dict[str, Any]) -> Dict[str, str]:
        """Calculate performance trends."""
        time_series = performance_data.get("time_series", [])
        if len(time_series) < 2:
            return {"latency": "stable", "success_rate": "stable", "requests": "stable"}

        # Compare recent vs earlier periods
        mid_point = len(time_series) // 2
        recent = time_series[mid_point:]
        earlier = time_series[:mid_point]

        def get_trend(recent_avg: float, earlier_avg: float, threshold: float = 0.05) -> str:
            diff = (recent_avg - earlier_avg) / earlier_avg if earlier_avg > 0 else 0
            if abs(diff) < threshold:
                return "stable"
            return "improving" if diff < 0 else "degrading"  # Lower latency is better

        recent_latency = sum(p.get("avg_latency", 0) for p in recent) / len(recent)
        earlier_latency = sum(p.get("avg_latency", 0) for p in earlier) / len(earlier)

        recent_success = sum(p.get("success_rate", 0) for p in recent) / len(recent)
        earlier_success = sum(p.get("success_rate", 0) for p in earlier) / len(earlier)

        recent_requests = sum(p.get("requests_count", 0) for p in recent) / len(recent)
        earlier_requests = sum(p.get("requests_count", 0) for p in earlier) / len(earlier)

        return {
            "latency": get_trend(recent_latency, earlier_latency),
            "success_rate": get_trend(
                -recent_success, -earlier_success
            ),  # Higher success is better
            "requests": get_trend(recent_requests, earlier_requests),
        }


class WeeklyPerformanceTemplate(ReportTemplate):
    """Weekly performance analysis report template."""

    def __init__(self):
        super().__init__(
            name="weekly_performance",
            description="Weekly performance analysis with trends and comparisons",
        )

    def get_context(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Get context for weekly performance template."""
        context = super().get_context(data, **kwargs)

        performance = data.get("performance_data", {})
        metrics = data.get("metrics_summary", {})

        # Weekly statistics
        context["weekly_stats"] = self._calculate_weekly_stats(performance)

        # Model comparison
        context["model_comparison"] = self._compare_models(performance)

        # Performance by day of week
        context["daily_breakdown"] = self._get_daily_breakdown(performance)

        # SLA compliance
        context["sla_compliance"] = self._calculate_sla_compliance(performance)

        # Recommendations
        context["recommendations"] = self._generate_recommendations(performance, metrics)

        return context

    def _calculate_weekly_stats(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate weekly performance statistics."""
        time_series = performance_data.get("time_series", [])
        if not time_series:
            return {}

        latencies = [p.get("avg_latency", 0) for p in time_series]
        success_rates = [p.get("success_rate", 0) for p in time_series]
        request_counts = [p.get("requests_count", 0) for p in time_series]

        return {
            "avg_latency": sum(latencies) / len(latencies),
            "p50_latency": sorted(latencies)[len(latencies) // 2],
            "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)],
            "p99_latency": sorted(latencies)[int(len(latencies) * 0.99)],
            "min_success_rate": min(success_rates) if success_rates else 0,
            "avg_success_rate": sum(success_rates) / len(success_rates) if success_rates else 0,
            "max_success_rate": max(success_rates) if success_rates else 0,
            "total_requests": sum(request_counts),
            "peak_requests": max(request_counts) if request_counts else 0,
            "uptime_percentage": len([s for s in success_rates if s > 0])
            / len(success_rates)
            * 100,
        }

    def _compare_models(self, performance_data: Dict[str, Any]) -> List[Dict]:
        """Compare model performance."""
        models = performance_data.get("models", [])
        if not models:
            return []

        model_stats = []
        for model in models:
            model_stats.append(
                {
                    "name": model.get("name", "Unknown"),
                    "avg_latency": model.get("avg_latency", 0),
                    "success_rate": model.get("success_rate", 0),
                    "total_requests": model.get("total_requests", 0),
                    "cost_per_request": model.get("cost_per_request", 0),
                    "performance_score": self._calculate_performance_score(model),
                }
            )

        return sorted(model_stats, key=lambda m: m["performance_score"], reverse=True)

    def _calculate_performance_score(self, model: Dict[str, Any]) -> float:
        """Calculate overall performance score."""
        latency_score = max(0, 100 - model.get("avg_latency", 1) * 100)  # Lower is better
        success_score = model.get("success_rate", 0)  # Higher is better
        cost_score = max(0, 100 - model.get("cost_per_request", 0.01) * 1000)  # Lower is better

        # Weighted average
        return latency_score * 0.4 + success_score * 0.4 + cost_score * 0.2

    def _get_daily_breakdown(self, performance_data: Dict[str, Any]) -> List[Dict]:
        """Get performance breakdown by day of week."""
        time_series = performance_data.get("time_series", [])
        daily_data = {}

        for point in time_series:
            try:
                timestamp = datetime.fromisoformat(point["timestamp"])
                day_name = timestamp.strftime("%A")

                if day_name not in daily_data:
                    daily_data[day_name] = {
                        "latencies": [],
                        "success_rates": [],
                        "request_counts": [],
                    }

                daily_data[day_name]["latencies"].append(point.get("avg_latency", 0))
                daily_data[day_name]["success_rates"].append(point.get("success_rate", 0))
                daily_data[day_name]["request_counts"].append(point.get("requests_count", 0))

            except (KeyError, ValueError):
                continue

        # Calculate averages for each day
        breakdown = []
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
            if day in daily_data:
                data = daily_data[day]
                breakdown.append(
                    {
                        "day": day,
                        "avg_latency": sum(data["latencies"]) / len(data["latencies"]),
                        "avg_success_rate": sum(data["success_rates"]) / len(data["success_rates"]),
                        "total_requests": sum(data["request_counts"]),
                    }
                )

        return breakdown

    def _calculate_sla_compliance(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate SLA compliance metrics."""
        time_series = performance_data.get("time_series", [])
        if not time_series:
            return {}

        # Define SLA thresholds
        sla_latency = 1.0  # 1 second
        sla_success_rate = 95.0  # 95%

        latency_violations = [p for p in time_series if p.get("avg_latency", 0) > sla_latency]
        success_violations = [p for p in time_series if p.get("success_rate", 0) < sla_success_rate]

        return {
            "latency_compliance": (len(time_series) - len(latency_violations))
            / len(time_series)
            * 100,
            "success_rate_compliance": (len(time_series) - len(success_violations))
            / len(time_series)
            * 100,
            "latency_violations": len(latency_violations),
            "success_rate_violations": len(success_violations),
            "overall_compliance": min(
                (len(time_series) - len(latency_violations)) / len(time_series) * 100,
                (len(time_series) - len(success_violations)) / len(time_series) * 100,
            ),
        }

    def _generate_recommendations(
        self, performance_data: Dict[str, Any], metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []

        time_series = performance_data.get("time_series", [])
        if not time_series:
            return recommendations

        avg_latency = sum(p.get("avg_latency", 0) for p in time_series) / len(time_series)
        avg_success_rate = sum(p.get("success_rate", 0) for p in time_series) / len(time_series)

        if avg_latency > 0.5:
            recommendations.append(
                "Consider optimizing model inference or implementing caching to reduce average latency"
            )

        if avg_success_rate < 95:
            recommendations.append(
                "Investigate causes of request failures and implement retry mechanisms"
            )

        if metrics.get("active_alerts", 0) > 5:
            recommendations.append("Review and tune alert thresholds to reduce alert fatigue")

        return recommendations


class MonthlyAnalysisTemplate(ReportTemplate):
    """Monthly analysis report template."""

    def __init__(self):
        super().__init__(
            name="monthly_analysis",
            description="Comprehensive monthly analysis with cost optimization and trends",
        )

    def get_context(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Get context for monthly analysis template."""
        context = super().get_context(data, **kwargs)

        costs = data.get("cost_breakdown", {})
        performance = data.get("performance_data", {})
        metrics = data.get("metrics_summary", {})

        # Monthly cost analysis
        context["cost_analysis"] = self._analyze_monthly_costs(costs)

        # ROI analysis
        context["roi_analysis"] = self._calculate_roi(costs, performance)

        # Usage patterns
        context["usage_patterns"] = self._analyze_usage_patterns(performance)

        # Cost optimization opportunities
        context["cost_optimization"] = self._find_cost_optimization(costs, performance)

        # Executive summary
        context["executive_summary"] = self._create_executive_summary(data)

        return context

    def _analyze_monthly_costs(self, cost_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze monthly cost data."""
        daily_costs = cost_data.get("daily_costs", [])
        provider_breakdown = cost_data.get("provider_breakdown", {})

        if not daily_costs:
            return {}

        costs = [item["cost"] for item in daily_costs]

        return {
            "total_cost": sum(costs),
            "avg_daily_cost": sum(costs) / len(costs),
            "min_daily_cost": min(costs),
            "max_daily_cost": max(costs),
            "cost_variance": max(costs) - min(costs),
            "provider_breakdown": provider_breakdown,
            "cost_trend": "increasing" if costs[-1] > costs[0] else "decreasing",
        }

    def _calculate_roi(
        self, cost_data: Dict[str, Any], performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate return on investment metrics."""
        total_cost = cost_data.get("total_cost", 0)
        time_series = performance_data.get("time_series", [])

        if not time_series or total_cost == 0:
            return {}

        total_requests = sum(p.get("requests_count", 0) for p in time_series)
        successful_requests = sum(
            p.get("requests_count", 0) * p.get("success_rate", 0) / 100 for p in time_series
        )

        return {
            "cost_per_request": total_cost / total_requests if total_requests > 0 else 0,
            "cost_per_successful_request": total_cost / successful_requests
            if successful_requests > 0
            else 0,
            "efficiency_score": successful_requests / total_cost if total_cost > 0 else 0,
            "total_requests": total_requests,
            "successful_requests": int(successful_requests),
        }

    def _analyze_usage_patterns(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze usage patterns."""
        time_series = performance_data.get("time_series", [])
        if not time_series:
            return {}

        # Analyze by hour of day
        hourly_usage = {}
        for point in time_series:
            try:
                timestamp = datetime.fromisoformat(point["timestamp"])
                hour = timestamp.hour
                requests = point.get("requests_count", 0)

                if hour not in hourly_usage:
                    hourly_usage[hour] = []
                hourly_usage[hour].append(requests)
            except (KeyError, ValueError):
                continue

        # Find peak and low usage hours
        avg_hourly_usage = {
            hour: sum(requests) / len(requests) for hour, requests in hourly_usage.items()
        }

        peak_hour = (
            max(avg_hourly_usage.keys(), key=lambda h: avg_hourly_usage[h])
            if avg_hourly_usage
            else 0
        )
        low_hour = (
            min(avg_hourly_usage.keys(), key=lambda h: avg_hourly_usage[h])
            if avg_hourly_usage
            else 0
        )

        return {
            "peak_hour": peak_hour,
            "low_hour": low_hour,
            "peak_usage": avg_hourly_usage.get(peak_hour, 0),
            "low_usage": avg_hourly_usage.get(low_hour, 0),
            "usage_variance": avg_hourly_usage.get(peak_hour, 0)
            - avg_hourly_usage.get(low_hour, 0),
        }

    def _find_cost_optimization(
        self, cost_data: Dict[str, Any], performance_data: Dict[str, Any]
    ) -> List[Dict]:
        """Find cost optimization opportunities."""
        opportunities = []

        provider_breakdown = cost_data.get("provider_breakdown", {})
        models = performance_data.get("models", [])

        # Expensive providers
        if provider_breakdown:
            max_cost_provider = max(provider_breakdown.keys(), key=lambda p: provider_breakdown[p])
            max_cost = provider_breakdown[max_cost_provider]

            if max_cost > sum(provider_breakdown.values()) * 0.5:
                opportunities.append(
                    {
                        "type": "provider_consolidation",
                        "description": f"{max_cost_provider} represents {max_cost / sum(provider_breakdown.values()) * 100:.1f}% of costs",
                        "recommendation": "Consider diversifying across providers or negotiating better rates",
                        "potential_savings": max_cost * 0.1,  # Assume 10% savings
                    }
                )

        # Underperforming models
        for model in models:
            if model.get("success_rate", 100) < 90 and model.get("cost_per_request", 0) > 0.01:
                opportunities.append(
                    {
                        "type": "model_optimization",
                        "description": f"Model {model.get('name', 'Unknown')} has low success rate but high cost",
                        "recommendation": "Consider replacing with more reliable model or improving error handling",
                        "potential_savings": model.get("cost_per_request", 0)
                        * model.get("total_requests", 0)
                        * 0.2,
                    }
                )

        return opportunities

    def _create_executive_summary(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Create executive summary."""
        metrics = data.get("metrics_summary", {})
        costs = data.get("cost_breakdown", {})
        alerts = data.get("active_alerts", [])

        total_cost = costs.get("total_cost", 0)
        total_requests = metrics.get("total_requests", 0)
        active_alerts = len([a for a in alerts if a.get("status") != "resolved"])

        summary = {
            "overview": f"Processed {total_requests:,} requests with total cost of ${total_cost:.2f}",
            "performance": f"Average latency: {metrics.get('avg_latency', 0) * 1000:.1f}ms",
            "reliability": f"System uptime: {metrics.get('uptime', 0):.1f}%",
            "alerts": f"Currently {active_alerts} active alerts requiring attention",
        }

        return summary


class CustomReportTemplate(ReportTemplate):
    """Customizable report template."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        super().__init__(
            name=config.get("name", "custom_report"),
            description=config.get("description", "Custom report template"),
        )

    def get_context(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Get context for custom template."""
        context = super().get_context(data, **kwargs)

        # Apply custom filters and transformations based on config
        sections = self.config.get("sections", [])

        for section in sections:
            section_type = section.get("type")
            section_config = section.get("config", {})

            if section_type == "metrics":
                context[section["name"]] = self._extract_metrics(data, section_config)
            elif section_type == "charts":
                context[section["name"]] = self._prepare_chart_data(data, section_config)
            elif section_type == "tables":
                context[section["name"]] = self._create_table_data(data, section_config)

        return context

    def _extract_metrics(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract specific metrics based on configuration."""
        metrics = {}
        fields = config.get("fields", [])

        for field in fields:
            path = field.split(".")
            value = data

            try:
                for key in path:
                    value = value[key]
                metrics[field.replace(".", "_")] = value
            except (KeyError, TypeError):
                metrics[field.replace(".", "_")] = None

        return metrics

    def _prepare_chart_data(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for charts."""
        chart_type = config.get("chart_type", "line")
        data_source = config.get("data_source", "performance_data.time_series")

        # Extract data based on source path
        path = data_source.split(".")
        chart_data = data

        try:
            for key in path:
                chart_data = chart_data[key]
        except (KeyError, TypeError):
            chart_data = []

        return {"type": chart_type, "data": chart_data, "config": config}

    def _create_table_data(self, data: Dict[str, Any], config: Dict[str, Any]) -> List[Dict]:
        """Create table data."""
        data_source = config.get("data_source", "performance_data.models")
        columns = config.get("columns", [])

        # Extract data
        path = data_source.split(".")
        table_data = data

        try:
            for key in path:
                table_data = table_data[key]
        except (KeyError, TypeError):
            table_data = []

        # Filter and format columns
        if isinstance(table_data, list) and columns:
            formatted_data = []
            for row in table_data:
                formatted_row = {}
                for col in columns:
                    formatted_row[col] = row.get(col, "")
                formatted_data.append(formatted_row)
            return formatted_data

        return table_data if isinstance(table_data, list) else []
