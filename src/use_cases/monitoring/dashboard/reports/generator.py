"""
Report Generator

Core report generation engine that creates PDF and HTML reports from monitoring data
using customizable templates with embedded charts and data visualizations.
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import base64
import io
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from weasyprint import CSS, HTML
    from weasyprint.text.fonts import FontConfiguration

    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    logging.warning("WeasyPrint not available. PDF generation will be limited.")

try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-GUI backend
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available. Chart generation will be limited.")


from jinja2 import Environment, FileSystemLoader, Template


class ReportTemplate:
    """Base class for report templates."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.template_dir = Path(__file__).parent / "templates"
        self.template_dir.mkdir(exist_ok=True)

    def get_template_path(self) -> Path:
        """Get the template file path."""
        return self.template_dir / f"{self.name}.html"

    def get_context(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Get template context from data."""
        return {
            "report_title": self.name.replace("_", " ").title(),
            "generated_at": datetime.utcnow().isoformat(),
            "data": data,
            **kwargs,
        }

    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate input data for this template."""
        return True


class ReportGenerator:
    """Main report generation engine."""

    def __init__(self, data_service=None, output_dir: Optional[str] = None):
        self.data_service = data_service
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "reports"
        self.output_dir.mkdir(exist_ok=True)

        # Initialize Jinja2 environment
        template_dirs = [
            Path(__file__).parent / "templates",
            Path(__file__).parent.parent / "templates",
        ]
        self.jinja_env = Environment(
            loader=FileSystemLoader([str(d) for d in template_dirs if d.exists()]), autoescape=True
        )

        # Add custom filters
        self.jinja_env.filters["datetime"] = self._format_datetime
        self.jinja_env.filters["currency"] = self._format_currency
        self.jinja_env.filters["duration"] = self._format_duration
        self.jinja_env.filters["percentage"] = self._format_percentage

        self.logger = logging.getLogger(__name__)

    def generate_report(
        self,
        template: ReportTemplate | str,
        output_format: str = "html",
        date_range: Optional[tuple] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a report using the specified template.

        Args:
            template: ReportTemplate instance or template name
            output_format: 'html' or 'pdf'
            date_range: (start_date, end_date) tuple
            filters: Additional data filters
            **kwargs: Additional template variables

        Returns:
            Dict with report metadata and file path
        """
        try:
            # Handle template parameter
            if isinstance(template, str):
                template_name = template
                template_obj = ReportTemplate(template_name)
            else:
                template_name = template.name
                template_obj = template

            # Get report data
            report_data = self._gather_report_data(date_range, filters)

            # Validate data
            if not template_obj.validate_data(report_data):
                raise ValueError(f"Invalid data for template {template_name}")

            # Generate charts if needed
            charts = self._generate_charts(report_data, template_name)

            # Get template context
            context = template_obj.get_context(
                report_data, charts=charts, date_range=date_range, filters=filters or {}, **kwargs
            )

            # Generate report content
            content = self._render_template(template_name, context)

            # Save report
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{template_name}_{timestamp}.{output_format}"
            output_path = self.output_dir / filename

            if output_format.lower() == "pdf":
                self._save_as_pdf(content, output_path)
            else:
                self._save_as_html(content, output_path)

            report_metadata = {
                "id": f"{template_name}_{timestamp}",
                "template": template_name,
                "format": output_format,
                "generated_at": datetime.utcnow().isoformat(),
                "file_path": str(output_path),
                "file_size": output_path.stat().st_size,
                "date_range": date_range,
                "filters": filters,
                "status": "completed",
            }

            self.logger.info(f"Report generated: {output_path}")
            return report_metadata

        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "generated_at": datetime.utcnow().isoformat(),
            }

    def _gather_report_data(
        self, date_range: Optional[tuple] = None, filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Gather data for report generation."""
        if not self.data_service:
            # Return mock data for testing
            return self._get_mock_data(date_range)

        try:
            # Calculate date range
            if date_range:
                start_date, end_date = date_range
                hours = int((end_date - start_date).total_seconds() / 3600)
            else:
                hours = 24
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(hours=hours)

            # Apply filters
            provider = filters.get("provider") if filters else None
            model = filters.get("model") if filters else None

            # Gather all necessary data
            data = {
                "metrics_summary": self.data_service.get_metrics_summary(),
                "performance_data": self.data_service.get_performance_data(hours, provider, model),
                "cost_breakdown": self.data_service.get_cost_breakdown(hours),
                "active_alerts": self.data_service.get_active_alerts(50),
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "hours": hours,
                },
                "filters": filters or {},
            }

            return data

        except Exception as e:
            self.logger.error(f"Data gathering failed: {e}")
            return self._get_mock_data(date_range)

    def _get_mock_data(self, date_range: Optional[tuple] = None) -> Dict[str, Any]:
        """Generate mock data for testing."""
        if date_range:
            start_date, end_date = date_range
            hours = int((end_date - start_date).total_seconds() / 3600)
        else:
            hours = 24
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(hours=hours)

        # Generate mock time series data
        time_points = []
        current_time = start_date
        while current_time <= end_date:
            time_points.append(
                {
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "avg_latency": 0.5 + (hash(str(current_time)) % 100) / 1000,
                    "success_rate": 95 + (hash(str(current_time)) % 10),
                    "requests_count": 100 + (hash(str(current_time)) % 50),
                    "cost": 0.01 + (hash(str(current_time)) % 10) / 1000,
                }
            )
            current_time += timedelta(hours=1)

        return {
            "metrics_summary": {
                "total_models": 5,
                "total_requests": sum(point["requests_count"] for point in time_points),
                "avg_latency": sum(point["avg_latency"] for point in time_points)
                / len(time_points),
                "total_cost": sum(point["cost"] for point in time_points),
                "active_alerts": 3,
                "uptime": 99.5,
                "last_updated": datetime.utcnow().isoformat(),
            },
            "performance_data": {
                "time_series": time_points,
                "providers": ["OpenAI", "Anthropic", "Google"],
                "models": ["gpt-4o-mini", "claude-3-haiku", "gemini-pro"],
            },
            "cost_breakdown": {
                "daily_costs": [
                    {
                        "date": (start_date + timedelta(days=i)).strftime("%Y-%m-%d"),
                        "cost": 10 + i * 2,
                    }
                    for i in range(7)
                ],
                "provider_breakdown": {"OpenAI": 45.6, "Anthropic": 32.4, "Google": 22.0},
                "total_cost": 100.0,
            },
            "active_alerts": [
                {
                    "id": 1,
                    "severity": "warning",
                    "title": "High latency detected",
                    "message": "Average latency exceeded 1s",
                    "created_at": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                    "provider": "OpenAI",
                    "model": "gpt-4o-mini",
                },
                {
                    "id": 2,
                    "severity": "critical",
                    "title": "Cost threshold exceeded",
                    "message": "Daily cost exceeded $50",
                    "created_at": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                    "provider": "Anthropic",
                    "model": "claude-3-opus",
                },
            ],
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "hours": hours,
            },
            "filters": {},
        }

    def _generate_charts(self, data: Dict[str, Any], template_name: str) -> Dict[str, str]:
        """Generate charts for the report."""
        if not MATPLOTLIB_AVAILABLE:
            return {}

        charts = {}

        try:
            # Performance over time chart
            if "performance_data" in data and data["performance_data"]["time_series"]:
                charts["performance_chart"] = self._create_performance_chart(
                    data["performance_data"]["time_series"]
                )

            # Cost breakdown chart
            if "cost_breakdown" in data:
                charts["cost_chart"] = self._create_cost_chart(data["cost_breakdown"])

            # Alert severity chart
            if "active_alerts" in data:
                charts["alert_chart"] = self._create_alert_chart(data["active_alerts"])

        except Exception as e:
            self.logger.error(f"Chart generation failed: {e}")

        return charts

    def _create_performance_chart(self, time_series: List[Dict]) -> str:
        """Create performance over time chart."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        timestamps = [datetime.fromisoformat(point["timestamp"]) for point in time_series]
        latencies = [point["avg_latency"] * 1000 for point in time_series]  # Convert to ms
        success_rates = [point["success_rate"] for point in time_series]

        # Latency plot
        ax1.plot(timestamps, latencies, "b-", linewidth=2, label="Latency (ms)")
        ax1.set_ylabel("Latency (ms)")
        ax1.set_title("Performance Metrics Over Time")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Success rate plot
        ax2.plot(timestamps, success_rates, "g-", linewidth=2, label="Success Rate (%)")
        ax2.set_ylabel("Success Rate (%)")
        ax2.set_xlabel("Time")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        plt.xticks(rotation=45)

        plt.tight_layout()

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return f"data:image/png;base64,{chart_data}"

    def _create_cost_chart(self, cost_data: Dict[str, Any]) -> str:
        """Create cost breakdown chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Daily costs
        if "daily_costs" in cost_data:
            dates = [item["date"] for item in cost_data["daily_costs"]]
            costs = [item["cost"] for item in cost_data["daily_costs"]]

            ax1.bar(dates, costs, color="skyblue", alpha=0.7)
            ax1.set_title("Daily Costs")
            ax1.set_ylabel("Cost ($)")
            ax1.tick_params(axis="x", rotation=45)

        # Provider breakdown
        if "provider_breakdown" in cost_data:
            providers = list(cost_data["provider_breakdown"].keys())
            amounts = list(cost_data["provider_breakdown"].values())

            colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
            ax2.pie(
                amounts,
                labels=providers,
                autopct="%1.1f%%",
                colors=colors[: len(providers)],
                startangle=90,
            )
            ax2.set_title("Cost by Provider")

        plt.tight_layout()

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return f"data:image/png;base64,{chart_data}"

    def _create_alert_chart(self, alerts: List[Dict]) -> str:
        """Create alert severity chart."""
        severity_counts = {"critical": 0, "warning": 0, "info": 0}

        for alert in alerts:
            severity = alert.get("severity", "info")
            if severity in severity_counts:
                severity_counts[severity] += 1

        fig, ax = plt.subplots(figsize=(8, 6))

        severities = list(severity_counts.keys())
        counts = list(severity_counts.values())
        colors = ["#FF4444", "#FFA500", "#4444FF"]

        bars = ax.bar(severities, counts, color=colors)
        ax.set_title("Alerts by Severity")
        ax.set_ylabel("Count")

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    str(count),
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return f"data:image/png;base64,{chart_data}"

    def _render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render template with context."""
        try:
            template = self.jinja_env.get_template(f"{template_name}.html")
            return template.render(**context)
        except Exception:
            # Fallback to basic template
            basic_template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>{{ report_title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }
        .section { margin-bottom: 30px; }
        .chart { text-align: center; margin: 20px 0; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
        .metric-card { border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
        .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
        .metric-label { color: #666; }
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ report_title }}</h1>
        <p>Generated: {{ generated_at }}</p>
    </div>

    <div class="section">
        <h2>Summary</h2>
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{{ data.metrics_summary.total_models }}</div>
                <div class="metric-label">Active Models</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ data.metrics_summary.total_requests }}</div>
                <div class="metric-label">Total Requests</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${{ "%.2f"|format(data.metrics_summary.total_cost) }}</div>
                <div class="metric-label">Total Cost</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ data.metrics_summary.active_alerts }}</div>
                <div class="metric-label">Active Alerts</div>
            </div>
        </div>
    </div>

    {% if charts.performance_chart %}
    <div class="section">
        <h2>Performance Trends</h2>
        <div class="chart">
            <img src="{{ charts.performance_chart }}" alt="Performance Chart" style="max-width: 100%;">
        </div>
    </div>
    {% endif %}

    {% if charts.cost_chart %}
    <div class="section">
        <h2>Cost Analysis</h2>
        <div class="chart">
            <img src="{{ charts.cost_chart }}" alt="Cost Chart" style="max-width: 100%;">
        </div>
    </div>
    {% endif %}
</body>
</html>
            """)
            return basic_template.render(**context)

    def _save_as_html(self, content: str, output_path: Path) -> None:
        """Save content as HTML file."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _save_as_pdf(self, content: str, output_path: Path) -> None:
        """Save content as PDF file."""
        if not WEASYPRINT_AVAILABLE:
            # Fallback: save as HTML with PDF extension
            self.logger.warning("WeasyPrint not available. Saving as HTML instead.")
            html_path = output_path.with_suffix(".html")
            self._save_as_html(content, html_path)
            return

        try:
            # Create CSS for better PDF styling
            css_content = """
            @page {
                size: A4;
                margin: 2cm;
            }
            body {
                font-family: Arial, sans-serif;
                font-size: 12pt;
                line-height: 1.5;
            }
            .chart img {
                max-width: 100%;
                height: auto;
            }
            .page-break {
                page-break-before: always;
            }
            """

            # Generate PDF
            html_doc = HTML(string=content)
            css_doc = CSS(string=css_content)
            html_doc.write_pdf(str(output_path), stylesheets=[css_doc])

        except Exception as e:
            self.logger.error(f"PDF generation failed: {e}")
            # Fallback to HTML
            html_path = output_path.with_suffix(".html")
            self._save_as_html(content, html_path)

    # Template filters
    def _format_datetime(self, value: str) -> str:
        """Format datetime string."""
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except:
            return str(value)

    def _format_currency(self, value: float) -> str:
        """Format as currency."""
        return f"${value:.2f}"

    def _format_duration(self, value: float) -> str:
        """Format duration in seconds."""
        if value < 1:
            return f"{value * 1000:.0f}ms"
        elif value < 60:
            return f"{value:.1f}s"
        elif value < 3600:
            return f"{value / 60:.1f}m"
        else:
            return f"{value / 3600:.1f}h"

    def _format_percentage(self, value: float) -> str:
        """Format as percentage."""
        return f"{value:.1f}%"


def create_report_generator(data_service=None, output_dir: Optional[str] = None) -> ReportGenerator:
    """Factory function to create a report generator."""
    return ReportGenerator(data_service, output_dir)
