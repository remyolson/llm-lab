"""
Automated Report Generator for Evaluation Framework

This module provides comprehensive report generation with PDF/HTML output,
executive summaries, metric visualizations, export functionality, and
template-based customization.

Example:
    generator = ReportGenerator()

    # Generate comprehensive report
    report = generator.generate_report(
        comparison_result=comparison,
        cost_analysis=cost_analysis,
        format=ReportFormat.PDF
    )

    # Save report
    generator.save_report(report, "evaluation_report.pdf")
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import base64
import json
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px

# Import visualization libraries
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import report generation libraries
try:
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.shapes import Drawing
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    from reportlab.platypus import (
        Image,
        KeepTogether,
        ListFlowable,
        ListItem,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    from jinja2 import Environment, FileSystemLoader, Template

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

try:
    import markdown

    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

# Import evaluation components
from ..analysis.cost_benefit import CostBenefitAnalyzer
from ..benchmark_runner import BenchmarkResult, ComparisonResult
from ..comparison.comparison_view import ComparisonView

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Report output formats."""

    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    EXCEL = "excel"


class ReportSection(Enum):
    """Report sections."""

    EXECUTIVE_SUMMARY = "executive_summary"
    PERFORMANCE_METRICS = "performance_metrics"
    COST_ANALYSIS = "cost_analysis"
    DETAILED_BENCHMARKS = "detailed_benchmarks"
    VISUALIZATIONS = "visualizations"
    RECOMMENDATIONS = "recommendations"
    TECHNICAL_DETAILS = "technical_details"
    APPENDIX = "appendix"


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    title: str = "Model Fine-Tuning Evaluation Report"
    author: str = "LLM Lab Evaluation Framework"
    company: str = ""
    include_sections: List[ReportSection] = field(default_factory=lambda: list(ReportSection))
    format: ReportFormat = ReportFormat.PDF
    page_size: str = "letter"  # letter or A4
    include_visualizations: bool = True
    include_raw_data: bool = False
    executive_summary_length: int = 500  # words
    color_scheme: str = "default"
    template_path: Optional[str] = None
    logo_path: Optional[str] = None
    confidentiality_level: str = "Internal"


@dataclass
class ReportContent:
    """Content for report generation."""

    comparison_result: ComparisonResult
    cost_analysis: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    custom_sections: Dict[str, Any] = field(default_factory=dict)
    visualizations: Dict[str, Any] = field(default_factory=dict)


class ReportGenerator:
    """Automated report generator for evaluation results."""

    def __init__(self, config: Optional[ReportConfig] = None, template_dir: Optional[str] = None):
        """Initialize report generator.

        Args:
            config: Report configuration
            template_dir: Directory containing report templates
        """
        self.config = config or ReportConfig()
        self.template_dir = (
            Path(template_dir) if template_dir else Path(__file__).parent / "templates"
        )

        # Initialize styles
        self._init_styles()

        # Initialize templates
        if JINJA2_AVAILABLE:
            self.jinja_env = Environment(loader=FileSystemLoader(self.template_dir))

    def _init_styles(self):
        """Initialize report styles."""
        if REPORTLAB_AVAILABLE:
            self.styles = getSampleStyleSheet()

            # Custom styles
            self.styles.add(
                ParagraphStyle(
                    name="CustomTitle",
                    parent=self.styles["Title"],
                    fontSize=24,
                    textColor=colors.HexColor("#1a1a1a"),
                    spaceAfter=30,
                )
            )

            self.styles.add(
                ParagraphStyle(
                    name="SectionHeading",
                    parent=self.styles["Heading1"],
                    fontSize=16,
                    textColor=colors.HexColor("#2c3e50"),
                    spaceAfter=12,
                    spaceBefore=12,
                )
            )

            self.styles.add(
                ParagraphStyle(
                    name="SubHeading",
                    parent=self.styles["Heading2"],
                    fontSize=14,
                    textColor=colors.HexColor("#34495e"),
                    spaceAfter=6,
                )
            )

            self.styles.add(
                ParagraphStyle(
                    name="MetricHighlight",
                    parent=self.styles["Normal"],
                    fontSize=12,
                    textColor=colors.HexColor("#27ae60"),
                    alignment=TA_CENTER,
                )
            )

    def generate_report(
        self, content: ReportContent, format: Optional[ReportFormat] = None
    ) -> bytes | str:
        """Generate comprehensive evaluation report.

        Args:
            content: Report content
            format: Output format (overrides config)

        Returns:
            Generated report in specified format
        """
        format = format or self.config.format

        # Generate sections
        sections = self._generate_sections(content)

        # Generate visualizations
        if self.config.include_visualizations:
            visualizations = self._generate_visualizations(content)
            sections["visualizations"] = visualizations

        # Generate report based on format
        if format == ReportFormat.PDF:
            return self._generate_pdf_report(sections, content)
        elif format == ReportFormat.HTML:
            return self._generate_html_report(sections, content)
        elif format == ReportFormat.MARKDOWN:
            return self._generate_markdown_report(sections, content)
        elif format == ReportFormat.JSON:
            return self._generate_json_report(sections, content)
        elif format == ReportFormat.EXCEL:
            return self._generate_excel_report(sections, content)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_sections(self, content: ReportContent) -> Dict[str, Any]:
        """Generate report sections.

        Args:
            content: Report content

        Returns:
            Dictionary of generated sections
        """
        sections = {}

        # Executive Summary
        if ReportSection.EXECUTIVE_SUMMARY in self.config.include_sections:
            sections["executive_summary"] = self._generate_executive_summary(content)

        # Performance Metrics
        if ReportSection.PERFORMANCE_METRICS in self.config.include_sections:
            sections["performance_metrics"] = self._generate_performance_metrics(content)

        # Cost Analysis
        if ReportSection.COST_ANALYSIS in self.config.include_sections and content.cost_analysis:
            sections["cost_analysis"] = self._generate_cost_analysis(content)

        # Detailed Benchmarks
        if ReportSection.DETAILED_BENCHMARKS in self.config.include_sections:
            sections["detailed_benchmarks"] = self._generate_detailed_benchmarks(content)

        # Recommendations
        if ReportSection.RECOMMENDATIONS in self.config.include_sections:
            sections["recommendations"] = self._generate_recommendations(content)

        # Technical Details
        if ReportSection.TECHNICAL_DETAILS in self.config.include_sections:
            sections["technical_details"] = self._generate_technical_details(content)

        # Appendix
        if ReportSection.APPENDIX in self.config.include_sections:
            sections["appendix"] = self._generate_appendix(content)

        # Add custom sections
        sections.update(content.custom_sections)

        return sections

    def _generate_executive_summary(self, content: ReportContent) -> Dict[str, Any]:
        """Generate executive summary.

        Args:
            content: Report content

        Returns:
            Executive summary data
        """
        comparison = content.comparison_result

        # Calculate key metrics
        improvements = []
        for benchmark, imp in comparison.improvements.items():
            if isinstance(imp, dict) and "improvement_pct" in imp:
                improvements.append(imp["improvement_pct"])

        avg_improvement = np.mean(improvements) if improvements else 0
        max_improvement = max(improvements) if improvements else 0

        # Count regressions
        num_regressions = len(comparison.regressions)

        # Generate summary text
        summary_text = f"""
        This report presents the evaluation results comparing the base model
        ({comparison.base_result.model_version.model_path}) with the fine-tuned version
        ({comparison.fine_tuned_result.model_version.model_path}).

        Key findings:
        - Average performance improvement: {avg_improvement:.1f}%
        - Maximum improvement: {max_improvement:.1f}%
        - Number of benchmarks evaluated: {len(improvements)}
        - Regressions detected: {num_regressions}
        """

        # Add cost analysis if available
        if content.cost_analysis:
            roi = content.cost_analysis.get("roi_analysis", {})
            if roi:
                summary_text += f"""

                Financial Analysis:
                - Initial investment: ${roi.get("initial_investment", 0):.2f}
                - Break-even period: {roi.get("break_even_months", 0):.1f} months
                - 1-year ROI: {roi.get("one_year_roi", 0):.1f}%
                """

        return {
            "text": summary_text.strip(),
            "key_metrics": {
                "avg_improvement": avg_improvement,
                "max_improvement": max_improvement,
                "num_benchmarks": len(improvements),
                "num_regressions": num_regressions,
            },
            "recommendation": self._get_overall_recommendation(avg_improvement, num_regressions),
        }

    def _generate_performance_metrics(self, content: ReportContent) -> Dict[str, Any]:
        """Generate performance metrics section.

        Args:
            content: Report content

        Returns:
            Performance metrics data
        """
        comparison = content.comparison_result

        # Prepare benchmark data
        benchmark_data = []

        for benchmark, imp in comparison.improvements.items():
            if isinstance(imp, dict):
                benchmark_data.append(
                    {
                        "name": benchmark,
                        "base_score": imp.get("base_score", 0),
                        "ft_score": imp.get("ft_score", 0),
                        "improvement": imp.get("improvement", 0),
                        "improvement_pct": imp.get("improvement_pct", 0),
                        "is_regression": imp.get("improvement_pct", 0) < -5,
                    }
                )

        # Statistical summary
        stats = comparison.statistical_analysis.get("summary", {})

        return {
            "benchmarks": benchmark_data,
            "statistics": stats,
            "performance_summary": {
                "total_benchmarks": len(benchmark_data),
                "improved": len([b for b in benchmark_data if b["improvement_pct"] > 0]),
                "regressed": len([b for b in benchmark_data if b["is_regression"]]),
                "unchanged": len([b for b in benchmark_data if abs(b["improvement_pct"]) < 1]),
            },
        }

    def _generate_cost_analysis(self, content: ReportContent) -> Dict[str, Any]:
        """Generate cost analysis section.

        Args:
            content: Report content

        Returns:
            Cost analysis data
        """
        cost_analysis = content.cost_analysis

        if not cost_analysis:
            return {}

        return {
            "cost_breakdown": cost_analysis.get("cost_breakdown"),
            "roi_analysis": cost_analysis.get("roi_analysis"),
            "projections": cost_analysis.get("projections"),
            "decision_matrix": cost_analysis.get("decision_matrix"),
            "recommendation": cost_analysis.get("recommendation"),
        }

    def _generate_detailed_benchmarks(self, content: ReportContent) -> Dict[str, Any]:
        """Generate detailed benchmark analysis.

        Args:
            content: Report content

        Returns:
            Detailed benchmark data
        """
        comparison = content.comparison_result
        detailed_results = []

        # Get benchmark results
        base_benchmarks = {b.name: b for b in comparison.base_result.evaluation_results.benchmarks}
        ft_benchmarks = {
            b.name: b for b in comparison.fine_tuned_result.evaluation_results.benchmarks
        }

        for benchmark_name in base_benchmarks:
            if benchmark_name in ft_benchmarks:
                base = base_benchmarks[benchmark_name]
                ft = ft_benchmarks[benchmark_name]

                detailed_results.append(
                    {
                        "name": benchmark_name,
                        "base": {
                            "overall_score": base.overall_score,
                            "task_scores": base.task_scores,
                            "runtime": base.runtime_seconds,
                            "samples": base.samples_evaluated,
                        },
                        "fine_tuned": {
                            "overall_score": ft.overall_score,
                            "task_scores": ft.task_scores,
                            "runtime": ft.runtime_seconds,
                            "samples": ft.samples_evaluated,
                        },
                        "comparison": {
                            "score_delta": ft.overall_score - base.overall_score,
                            "runtime_delta": ft.runtime_seconds - base.runtime_seconds,
                            "efficiency_gain": (
                                (base.runtime_seconds - ft.runtime_seconds)
                                / base.runtime_seconds
                                * 100
                            )
                            if base.runtime_seconds > 0
                            else 0,
                        },
                    }
                )

        return {"detailed_results": detailed_results}

    def _generate_recommendations(self, content: ReportContent) -> List[Dict[str, str]]:
        """Generate actionable recommendations.

        Args:
            content: Report content

        Returns:
            List of recommendations
        """
        recommendations = []
        comparison = content.comparison_result

        # Performance-based recommendations
        improvements = []
        for benchmark, imp in comparison.improvements.items():
            if isinstance(imp, dict) and "improvement_pct" in imp:
                improvements.append(imp["improvement_pct"])

        avg_improvement = np.mean(improvements) if improvements else 0

        if avg_improvement > 10:
            recommendations.append(
                {
                    "priority": "High",
                    "category": "Deployment",
                    "recommendation": "Strong performance improvements justify immediate deployment consideration.",
                    "action": "Proceed with production deployment planning and A/B testing.",
                }
            )
        elif avg_improvement > 5:
            recommendations.append(
                {
                    "priority": "Medium",
                    "category": "Further Testing",
                    "recommendation": "Moderate improvements warrant additional testing.",
                    "action": "Conduct extended evaluation on production-like data.",
                }
            )
        else:
            recommendations.append(
                {
                    "priority": "Low",
                    "category": "Optimization",
                    "recommendation": "Limited improvements suggest need for further optimization.",
                    "action": "Review training data and hyperparameters for improvement opportunities.",
                }
            )

        # Regression handling
        if comparison.regressions:
            recommendations.append(
                {
                    "priority": "High",
                    "category": "Regression Analysis",
                    "recommendation": f"Address {len(comparison.regressions)} detected regressions.",
                    "action": "Investigate regression causes and consider targeted fine-tuning.",
                }
            )

        # Cost-based recommendations
        if content.cost_analysis:
            roi = content.cost_analysis.get("roi_analysis", {})
            if roi.get("break_even_months", float("inf")) < 6:
                recommendations.append(
                    {
                        "priority": "High",
                        "category": "Cost Efficiency",
                        "recommendation": "Excellent ROI with quick payback period.",
                        "action": "Prioritize deployment to maximize cost savings.",
                    }
                )

        return recommendations

    def _generate_technical_details(self, content: ReportContent) -> Dict[str, Any]:
        """Generate technical details section.

        Args:
            content: Report content

        Returns:
            Technical details data
        """
        comparison = content.comparison_result

        return {
            "base_model": {
                "path": comparison.base_result.model_version.model_path,
                "version_id": comparison.base_result.model_version.version_id,
                "created_at": comparison.base_result.model_version.created_at.isoformat(),
                "metadata": comparison.base_result.model_version.metadata,
            },
            "fine_tuned_model": {
                "path": comparison.fine_tuned_result.model_version.model_path,
                "version_id": comparison.fine_tuned_result.model_version.version_id,
                "created_at": comparison.fine_tuned_result.model_version.created_at.isoformat(),
                "metadata": comparison.fine_tuned_result.model_version.metadata,
            },
            "evaluation": {
                "base_duration": comparison.base_result.duration_seconds,
                "ft_duration": comparison.fine_tuned_result.duration_seconds,
                "system_info": comparison.base_result.system_info,
            },
        }

    def _generate_appendix(self, content: ReportContent) -> Dict[str, Any]:
        """Generate appendix with raw data.

        Args:
            content: Report content

        Returns:
            Appendix data
        """
        appendix = {
            "generation_time": datetime.now().isoformat(),
            "report_version": "1.0.0",
            "framework_version": "1.0.0",
        }

        if self.config.include_raw_data:
            appendix["raw_comparison"] = content.comparison_result.to_dict()
            if content.cost_analysis:
                appendix["raw_cost_analysis"] = content.cost_analysis

        return appendix

    def _generate_visualizations(self, content: ReportContent) -> Dict[str, Any]:
        """Generate visualizations for report.

        Args:
            content: Report content

        Returns:
            Dictionary of visualizations
        """
        visualizations = {}

        # Create comparison view
        comparison_view = ComparisonView(content.comparison_result)

        # Generate charts
        visualizations["overview_chart"] = comparison_view.create_overview_chart()
        visualizations["improvement_chart"] = comparison_view.create_improvement_chart()
        visualizations["scatter_comparison"] = comparison_view.create_scatter_comparison()

        # Add cost analysis visualizations if available
        if content.cost_analysis:
            analyzer = CostBenefitAnalyzer()
            cost_visualizations = analyzer.create_visualizations(content.cost_analysis)
            visualizations.update(cost_visualizations)

        return visualizations

    def _generate_pdf_report(self, sections: Dict[str, Any], content: ReportContent) -> bytes:
        """Generate PDF report.

        Args:
            sections: Report sections
            content: Report content

        Returns:
            PDF bytes
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF generation")

        # Create temporary file
        output = BytesIO()

        # Create document
        doc = SimpleDocTemplate(
            output,
            pagesize=letter if self.config.page_size == "letter" else A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )

        # Build story
        story = []

        # Title page
        story.append(Paragraph(self.config.title, self.styles["CustomTitle"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Author: {self.config.author}", self.styles["Normal"]))
        story.append(
            Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", self.styles["Normal"])
        )
        if self.config.company:
            story.append(Paragraph(f"Company: {self.config.company}", self.styles["Normal"]))
        story.append(
            Paragraph(
                f"Confidentiality: {self.config.confidentiality_level}", self.styles["Normal"]
            )
        )
        story.append(PageBreak())

        # Executive Summary
        if "executive_summary" in sections:
            story.append(Paragraph("Executive Summary", self.styles["SectionHeading"]))
            story.append(Paragraph(sections["executive_summary"]["text"], self.styles["Normal"]))
            story.append(Spacer(1, 12))

            # Key metrics table
            metrics = sections["executive_summary"]["key_metrics"]
            data = [
                ["Metric", "Value"],
                ["Average Improvement", f"{metrics['avg_improvement']:.1f}%"],
                ["Maximum Improvement", f"{metrics['max_improvement']:.1f}%"],
                ["Benchmarks Evaluated", str(metrics["num_benchmarks"])],
                ["Regressions Detected", str(metrics["num_regressions"])],
            ]

            table = Table(data)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 14),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            story.append(table)
            story.append(PageBreak())

        # Performance Metrics
        if "performance_metrics" in sections:
            story.append(Paragraph("Performance Metrics", self.styles["SectionHeading"]))

            # Benchmark results table
            data = [["Benchmark", "Base Score", "Fine-Tuned Score", "Improvement"]]
            for benchmark in sections["performance_metrics"]["benchmarks"]:
                improvement_str = f"{benchmark['improvement_pct']:.1f}%"
                if benchmark["is_regression"]:
                    improvement_str = f"<font color='red'>{improvement_str}</font>"
                else:
                    improvement_str = f"<font color='green'>{improvement_str}</font>"

                data.append(
                    [
                        benchmark["name"],
                        f"{benchmark['base_score']:.4f}",
                        f"{benchmark['ft_score']:.4f}",
                        Paragraph(improvement_str, self.styles["Normal"]),
                    ]
                )

            table = Table(data, colWidths=[2 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 12),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            story.append(table)
            story.append(PageBreak())

        # Cost Analysis
        if "cost_analysis" in sections and sections["cost_analysis"]:
            story.append(Paragraph("Cost Analysis", self.styles["SectionHeading"]))

            roi = sections["cost_analysis"].get("roi_analysis", {})
            if roi:
                story.append(
                    Paragraph(
                        f"Initial Investment: ${roi.get('initial_investment', 0):.2f}",
                        self.styles["Normal"],
                    )
                )
                story.append(
                    Paragraph(
                        f"Monthly Savings: ${roi.get('monthly_savings', 0):.2f}",
                        self.styles["Normal"],
                    )
                )
                story.append(
                    Paragraph(
                        f"Break-even Period: {roi.get('break_even_months', 0):.1f} months",
                        self.styles["Normal"],
                    )
                )
                story.append(
                    Paragraph(
                        f"1-Year ROI: {roi.get('one_year_roi', 0):.1f}%", self.styles["Normal"]
                    )
                )
                story.append(Spacer(1, 12))

            recommendation = sections["cost_analysis"].get("recommendation", "")
            if recommendation:
                story.append(
                    Paragraph(f"Recommendation: {recommendation}", self.styles["MetricHighlight"])
                )
            story.append(PageBreak())

        # Recommendations
        if "recommendations" in sections:
            story.append(Paragraph("Recommendations", self.styles["SectionHeading"]))

            for rec in sections["recommendations"]:
                story.append(
                    Paragraph(f"• [{rec['priority']}] {rec['category']}", self.styles["SubHeading"])
                )
                story.append(Paragraph(f"  {rec['recommendation']}", self.styles["Normal"]))
                story.append(Paragraph(f"  Action: {rec['action']}", self.styles["Normal"]))
                story.append(Spacer(1, 6))

        # Build PDF
        doc.build(story)

        # Return bytes
        output.seek(0)
        return output.read()

    def _generate_html_report(self, sections: Dict[str, Any], content: ReportContent) -> str:
        """Generate HTML report.

        Args:
            sections: Report sections
            content: Report content

        Returns:
            HTML string
        """
        if not JINJA2_AVAILABLE:
            # Generate basic HTML without templates
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{self.config.title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #34495e; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .metric-positive {{ color: #27ae60; }}
                    .metric-negative {{ color: #e74c3c; }}
                </style>
            </head>
            <body>
                <h1>{self.config.title}</h1>
                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            """

            # Add sections
            for section_name, section_data in sections.items():
                if section_name == "executive_summary":
                    html += f"""
                    <h2>Executive Summary</h2>
                    <p>{section_data.get("text", "")}</p>
                    """
                elif section_name == "performance_metrics":
                    html += "<h2>Performance Metrics</h2><table>"
                    html += "<tr><th>Benchmark</th><th>Base Score</th><th>Fine-Tuned Score</th><th>Improvement</th></tr>"
                    for benchmark in section_data.get("benchmarks", []):
                        class_name = (
                            "metric-negative" if benchmark["is_regression"] else "metric-positive"
                        )
                        html += f"""
                        <tr>
                            <td>{benchmark["name"]}</td>
                            <td>{benchmark["base_score"]:.4f}</td>
                            <td>{benchmark["ft_score"]:.4f}</td>
                            <td class='{class_name}'>{benchmark["improvement_pct"]:.1f}%</td>
                        </tr>
                        """
                    html += "</table>"

            html += "</body></html>"
            return html

        # Use Jinja2 template if available
        template = self.jinja_env.get_template("report_template.html")
        return template.render(
            config=self.config, sections=sections, content=content, generation_time=datetime.now()
        )

    def _generate_markdown_report(self, sections: Dict[str, Any], content: ReportContent) -> str:
        """Generate Markdown report.

        Args:
            sections: Report sections
            content: Report content

        Returns:
            Markdown string
        """
        md = f"# {self.config.title}\n\n"
        md += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        md += f"**Author:** {self.config.author}\n"
        if self.config.company:
            md += f"**Company:** {self.config.company}\n"
        md += f"**Confidentiality:** {self.config.confidentiality_level}\n\n"

        # Executive Summary
        if "executive_summary" in sections:
            md += "## Executive Summary\n\n"
            md += sections["executive_summary"]["text"] + "\n\n"

            # Key metrics
            metrics = sections["executive_summary"]["key_metrics"]
            md += "### Key Metrics\n\n"
            md += f"- **Average Improvement:** {metrics['avg_improvement']:.1f}%\n"
            md += f"- **Maximum Improvement:** {metrics['max_improvement']:.1f}%\n"
            md += f"- **Benchmarks Evaluated:** {metrics['num_benchmarks']}\n"
            md += f"- **Regressions Detected:** {metrics['num_regressions']}\n\n"

        # Performance Metrics
        if "performance_metrics" in sections:
            md += "## Performance Metrics\n\n"
            md += "| Benchmark | Base Score | Fine-Tuned Score | Improvement |\n"
            md += "|-----------|------------|------------------|-------------|\n"

            for benchmark in sections["performance_metrics"]["benchmarks"]:
                improvement = f"{benchmark['improvement_pct']:.1f}%"
                if benchmark["is_regression"]:
                    improvement = f"⚠️ {improvement}"
                else:
                    improvement = f"✅ {improvement}"

                md += f"| {benchmark['name']} | {benchmark['base_score']:.4f} | "
                md += f"{benchmark['ft_score']:.4f} | {improvement} |\n"
            md += "\n"

        # Cost Analysis
        if "cost_analysis" in sections and sections["cost_analysis"]:
            md += "## Cost Analysis\n\n"
            roi = sections["cost_analysis"].get("roi_analysis", {})
            if roi:
                md += f"- **Initial Investment:** ${roi.get('initial_investment', 0):.2f}\n"
                md += f"- **Monthly Savings:** ${roi.get('monthly_savings', 0):.2f}\n"
                md += f"- **Break-even Period:** {roi.get('break_even_months', 0):.1f} months\n"
                md += f"- **1-Year ROI:** {roi.get('one_year_roi', 0):.1f}%\n\n"

            recommendation = sections["cost_analysis"].get("recommendation", "")
            if recommendation:
                md += f"**Recommendation:** {recommendation}\n\n"

        # Recommendations
        if "recommendations" in sections:
            md += "## Recommendations\n\n"
            for rec in sections["recommendations"]:
                md += f"### [{rec['priority']}] {rec['category']}\n\n"
                md += f"{rec['recommendation']}\n\n"
                md += f"**Action:** {rec['action']}\n\n"

        return md

    def _generate_json_report(self, sections: Dict[str, Any], content: ReportContent) -> str:
        """Generate JSON report.

        Args:
            sections: Report sections
            content: Report content

        Returns:
            JSON string
        """
        report_data = {
            "metadata": {
                "title": self.config.title,
                "author": self.config.author,
                "company": self.config.company,
                "generated_at": datetime.now().isoformat(),
                "confidentiality": self.config.confidentiality_level,
            },
            "sections": sections,
            "raw_data": {
                "comparison_result": content.comparison_result.to_dict()
                if self.config.include_raw_data
                else None,
                "cost_analysis": content.cost_analysis if self.config.include_raw_data else None,
            },
        }

        return json.dumps(report_data, indent=2, default=str)

    def _generate_excel_report(self, sections: Dict[str, Any], content: ReportContent) -> bytes:
        """Generate Excel report.

        Args:
            sections: Report sections
            content: Report content

        Returns:
            Excel file bytes
        """
        output = BytesIO()

        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            # Summary sheet
            summary_data = {
                "Metric": ["Average Improvement", "Max Improvement", "Benchmarks", "Regressions"],
                "Value": [
                    sections["executive_summary"]["key_metrics"]["avg_improvement"],
                    sections["executive_summary"]["key_metrics"]["max_improvement"],
                    sections["executive_summary"]["key_metrics"]["num_benchmarks"],
                    sections["executive_summary"]["key_metrics"]["num_regressions"],
                ],
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

            # Performance metrics sheet
            if "performance_metrics" in sections:
                perf_df = pd.DataFrame(sections["performance_metrics"]["benchmarks"])
                perf_df.to_excel(writer, sheet_name="Performance", index=False)

            # Cost analysis sheet
            if "cost_analysis" in sections and sections["cost_analysis"]:
                roi = sections["cost_analysis"].get("roi_analysis", {})
                if roi:
                    cost_data = {"Metric": list(roi.keys()), "Value": list(roi.values())}
                    pd.DataFrame(cost_data).to_excel(
                        writer, sheet_name="Cost Analysis", index=False
                    )

            # Recommendations sheet
            if "recommendations" in sections:
                rec_df = pd.DataFrame(sections["recommendations"])
                rec_df.to_excel(writer, sheet_name="Recommendations", index=False)

        output.seek(0)
        return output.read()

    def _get_overall_recommendation(self, avg_improvement: float, num_regressions: int) -> str:
        """Get overall recommendation based on metrics.

        Args:
            avg_improvement: Average improvement percentage
            num_regressions: Number of regressions

        Returns:
            Recommendation text
        """
        if avg_improvement > 15 and num_regressions == 0:
            return "Strongly Recommended - Excellent improvements with no regressions"
        elif avg_improvement > 10 and num_regressions <= 1:
            return "Recommended - Good improvements with minimal regressions"
        elif avg_improvement > 5:
            return "Conditionally Recommended - Moderate improvements, review regressions"
        elif avg_improvement > 0:
            return "Further Optimization Needed - Limited improvements"
        else:
            return "Not Recommended - No clear improvements detected"

    def save_report(self, report_data: bytes | str, filepath: str):
        """Save report to file.

        Args:
            report_data: Report data (bytes or string)
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(report_data, bytes):
            with open(filepath, "wb") as f:
                f.write(report_data)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report_data)

        logger.info(f"Report saved to {filepath}")


# Example usage
if __name__ == "__main__":
    from ...fine_tuning.evaluation.suite import (
        BenchmarkResult as EvalBenchmarkResult,
        EvaluationResult,
    )
    from ..benchmark_runner import BenchmarkResult, ComparisonResult, ModelVersion, ModelVersionType

    # Create sample data
    base_eval = EvaluationResult(
        model_name="gpt2",
        benchmarks=[
            EvalBenchmarkResult(
                name="hellaswag",
                overall_score=0.45,
                task_scores={"accuracy": 0.45},
                runtime_seconds=120,
                samples_evaluated=1000,
            ),
            EvalBenchmarkResult(
                name="mmlu",
                overall_score=0.35,
                task_scores={"accuracy": 0.35},
                runtime_seconds=150,
                samples_evaluated=1000,
            ),
        ],
    )

    ft_eval = EvaluationResult(
        model_name="gpt2-finetuned",
        benchmarks=[
            EvalBenchmarkResult(
                name="hellaswag",
                overall_score=0.52,
                task_scores={"accuracy": 0.52},
                runtime_seconds=125,
                samples_evaluated=1000,
            ),
            EvalBenchmarkResult(
                name="mmlu",
                overall_score=0.38,
                task_scores={"accuracy": 0.38},
                runtime_seconds=155,
                samples_evaluated=1000,
            ),
        ],
    )

    # Create comparison
    comparison = ComparisonResult(
        base_result=BenchmarkResult(
            model_version=ModelVersion(
                version_id="base_001",
                model_path="gpt2",
                version_type=ModelVersionType.BASE,
                created_at=datetime.now(),
            ),
            evaluation_results=base_eval,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=300,
        ),
        fine_tuned_result=BenchmarkResult(
            model_version=ModelVersion(
                version_id="ft_001",
                model_path="gpt2-finetuned",
                version_type=ModelVersionType.FINE_TUNED,
                created_at=datetime.now(),
            ),
            evaluation_results=ft_eval,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=310,
        ),
        improvements={
            "hellaswag": {
                "base_score": 0.45,
                "ft_score": 0.52,
                "improvement": 0.07,
                "improvement_pct": 15.56,
            },
            "mmlu": {
                "base_score": 0.35,
                "ft_score": 0.38,
                "improvement": 0.03,
                "improvement_pct": 8.57,
            },
        },
        statistical_analysis={
            "summary": {
                "mean_improvement_pct": 12.07,
                "median_improvement_pct": 12.07,
                "std_improvement_pct": 3.49,
            }
        },
    )

    # Create report generator
    config = ReportConfig(
        title="Model Fine-Tuning Evaluation Report",
        include_sections=list(ReportSection),
        include_visualizations=True,
    )

    generator = ReportGenerator(config)

    # Create content
    content = ReportContent(
        comparison_result=comparison, metadata={"project": "LLM Lab", "version": "1.0.0"}
    )

    # Generate reports in different formats
    print("Generating reports...")

    # Markdown report
    md_report = generator.generate_report(content, ReportFormat.MARKDOWN)
    generator.save_report(md_report, "evaluation_report.md")
    print("Markdown report saved")

    # JSON report
    json_report = generator.generate_report(content, ReportFormat.JSON)
    generator.save_report(json_report, "evaluation_report.json")
    print("JSON report saved")

    print("\nReport generation complete!")
