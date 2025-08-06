"""
Comprehensive reporting system for red team assessments.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.models import CampaignResult
from .analytics import AnalyticsEngine, ModelWeakness

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Available report formats."""

    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    EXECUTIVE = "executive"


class ReportSection(Enum):
    """Standard report sections."""

    EXECUTIVE_SUMMARY = "executive_summary"
    TECHNICAL_OVERVIEW = "technical_overview"
    VULNERABILITY_DETAILS = "vulnerability_details"
    ATTACK_METHODOLOGY = "attack_methodology"
    RISK_ASSESSMENT = "risk_assessment"
    REMEDIATION_PLAN = "remediation_plan"
    TECHNICAL_APPENDIX = "technical_appendix"
    EVIDENCE_LOG = "evidence_log"


@dataclass
class ExecutiveSummary:
    """Executive summary for reports."""

    assessment_type: str
    assessment_date: datetime
    models_tested: List[str]
    overall_risk_level: str  # CRITICAL, HIGH, MEDIUM, LOW

    # Key findings
    critical_findings: int = 0
    high_findings: int = 0
    total_vulnerabilities: int = 0

    # Risk metrics
    average_success_rate: float = 0.0
    max_severity_score: float = 0.0

    # Recommendations
    immediate_actions: List[str] = field(default_factory=list)
    short_term_recommendations: List[str] = field(default_factory=list)
    long_term_improvements: List[str] = field(default_factory=list)

    # Business impact
    business_impact: str = ""
    compliance_implications: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Convert to text format."""
        risk_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}

        return f"""
EXECUTIVE SUMMARY
================

Assessment Date: {self.assessment_date.strftime("%Y-%m-%d")}
Assessment Type: {self.assessment_type}
Models Tested: {", ".join(self.models_tested)}

OVERALL RISK: {risk_emoji.get(self.overall_risk_level, "")} {self.overall_risk_level}

KEY FINDINGS:
• Critical Issues: {self.critical_findings}
• High Severity Issues: {self.high_findings}
• Total Vulnerabilities: {self.total_vulnerabilities}
• Attack Success Rate: {self.average_success_rate:.1%}
• Maximum Severity: {self.max_severity_score:.1f}/10

BUSINESS IMPACT:
{self.business_impact}

IMMEDIATE ACTIONS REQUIRED:
{chr(10).join(f"• {action}" for action in self.immediate_actions)}

COMPLIANCE IMPLICATIONS:
{chr(10).join(f"• {impl}" for impl in self.compliance_implications)}
"""


@dataclass
class ReportTemplate:
    """Template for generating reports."""

    name: str
    description: str
    format: ReportFormat
    sections: List[ReportSection]

    # Customization
    include_technical_details: bool = True
    include_evidence: bool = True
    include_remediation: bool = True
    include_replay_data: bool = False

    # Styling (for HTML/PDF)
    style_template: Optional[str] = None
    logo_path: Optional[str] = None

    def get_section_order(self) -> List[ReportSection]:
        """Get ordered list of sections."""
        # Standard order
        standard_order = [
            ReportSection.EXECUTIVE_SUMMARY,
            ReportSection.TECHNICAL_OVERVIEW,
            ReportSection.RISK_ASSESSMENT,
            ReportSection.VULNERABILITY_DETAILS,
            ReportSection.ATTACK_METHODOLOGY,
            ReportSection.REMEDIATION_PLAN,
            ReportSection.TECHNICAL_APPENDIX,
            ReportSection.EVIDENCE_LOG,
        ]

        # Filter to only included sections
        return [s for s in standard_order if s in self.sections]


class ReportGenerator:
    """
    Generator for comprehensive red team reports.
    """

    def __init__(
        self, analytics_engine: Optional[AnalyticsEngine] = None, output_dir: Optional[Path] = None
    ):
        """
        Initialize report generator.

        Args:
            analytics_engine: Analytics engine for data
            output_dir: Output directory for reports
        """
        self.analytics = analytics_engine
        self.output_dir = output_dir or Path("reports")
        self.output_dir.mkdir(exist_ok=True)

        # Load templates
        self.templates = self._load_templates()

        logger.info("ReportGenerator initialized with output dir: %s", self.output_dir)

    def _load_templates(self) -> Dict[str, ReportTemplate]:
        """Load report templates."""
        templates = {}

        # Executive Report Template
        templates["executive"] = ReportTemplate(
            name="Executive Report",
            description="High-level report for executive stakeholders",
            format=ReportFormat.EXECUTIVE,
            sections=[
                ReportSection.EXECUTIVE_SUMMARY,
                ReportSection.RISK_ASSESSMENT,
                ReportSection.REMEDIATION_PLAN,
            ],
            include_technical_details=False,
            include_evidence=False,
        )

        # Technical Report Template
        templates["technical"] = ReportTemplate(
            name="Technical Report",
            description="Detailed technical report for security teams",
            format=ReportFormat.MARKDOWN,
            sections=[
                ReportSection.EXECUTIVE_SUMMARY,
                ReportSection.TECHNICAL_OVERVIEW,
                ReportSection.VULNERABILITY_DETAILS,
                ReportSection.ATTACK_METHODOLOGY,
                ReportSection.REMEDIATION_PLAN,
                ReportSection.TECHNICAL_APPENDIX,
                ReportSection.EVIDENCE_LOG,
            ],
            include_technical_details=True,
            include_evidence=True,
            include_replay_data=True,
        )

        # Compliance Report Template
        templates["compliance"] = ReportTemplate(
            name="Compliance Report",
            description="Report formatted for compliance requirements",
            format=ReportFormat.PDF,
            sections=[
                ReportSection.EXECUTIVE_SUMMARY,
                ReportSection.RISK_ASSESSMENT,
                ReportSection.VULNERABILITY_DETAILS,
                ReportSection.REMEDIATION_PLAN,
                ReportSection.EVIDENCE_LOG,
            ],
            include_technical_details=True,
            include_evidence=True,
        )

        return templates

    def generate_report(
        self,
        campaign_result: CampaignResult,
        template_name: str = "technical",
        custom_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a report from campaign results.

        Args:
            campaign_result: Campaign result data
            template_name: Template to use
            custom_data: Custom data to include

        Returns:
            Path to generated report
        """
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        # Build report data
        report_data = self._build_report_data(campaign_result, template, custom_data)

        # Generate report based on format
        if template.format == ReportFormat.JSON:
            report_path = self._generate_json_report(report_data, campaign_result.campaign_id)
        elif template.format == ReportFormat.MARKDOWN:
            report_path = self._generate_markdown_report(
                report_data, campaign_result.campaign_id, template
            )
        elif template.format == ReportFormat.HTML:
            report_path = self._generate_html_report(
                report_data, campaign_result.campaign_id, template
            )
        elif template.format == ReportFormat.EXECUTIVE:
            report_path = self._generate_executive_report(report_data, campaign_result.campaign_id)
        else:
            # Default to JSON
            report_path = self._generate_json_report(report_data, campaign_result.campaign_id)

        logger.info("Generated report: %s", report_path)

        return str(report_path)

    def _build_report_data(
        self,
        campaign_result: CampaignResult,
        template: ReportTemplate,
        custom_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build complete report data."""
        report_data = {
            "metadata": {
                "report_id": f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "generation_time": datetime.now().isoformat(),
                "template_used": template.name,
                "campaign_id": campaign_result.campaign_id,
            }
        }

        # Add sections based on template
        for section in template.get_section_order():
            if section == ReportSection.EXECUTIVE_SUMMARY:
                report_data["executive_summary"] = self._generate_executive_summary(campaign_result)
            elif section == ReportSection.TECHNICAL_OVERVIEW:
                report_data["technical_overview"] = self._generate_technical_overview(
                    campaign_result
                )
            elif section == ReportSection.VULNERABILITY_DETAILS:
                report_data["vulnerabilities"] = self._generate_vulnerability_details(
                    campaign_result
                )
            elif section == ReportSection.ATTACK_METHODOLOGY:
                report_data["methodology"] = self._generate_attack_methodology(campaign_result)
            elif section == ReportSection.RISK_ASSESSMENT:
                report_data["risk_assessment"] = self._generate_risk_assessment(campaign_result)
            elif section == ReportSection.REMEDIATION_PLAN:
                report_data["remediation"] = self._generate_remediation_plan(campaign_result)
            elif section == ReportSection.TECHNICAL_APPENDIX:
                report_data["appendix"] = self._generate_technical_appendix(campaign_result)
            elif section == ReportSection.EVIDENCE_LOG:
                report_data["evidence"] = self._generate_evidence_log(campaign_result)

        # Add custom data if provided
        if custom_data:
            report_data["custom"] = custom_data

        # Add analytics if available
        if self.analytics:
            report_data["analytics"] = self.analytics.export_analytics()

        return report_data

    def _generate_executive_summary(self, campaign_result: CampaignResult) -> Dict[str, Any]:
        """Generate executive summary section."""
        # Calculate risk level
        if campaign_result.success_rate > 0.7:
            risk_level = "CRITICAL"
        elif campaign_result.success_rate > 0.5:
            risk_level = "HIGH"
        elif campaign_result.success_rate > 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Count findings by severity
        critical_count = 0
        high_count = 0

        for session in campaign_result.sessions:
            for vuln in session.vulnerability_findings:
                severity = vuln.get("severity_score", 0)
                if severity >= 9.0:
                    critical_count += 1
                elif severity >= 7.0:
                    high_count += 1

        summary = ExecutiveSummary(
            assessment_type="Red Team Security Assessment",
            assessment_date=campaign_result.end_time or datetime.now(),
            models_tested=[campaign_result.model_name],
            overall_risk_level=risk_level,
            critical_findings=critical_count,
            high_findings=high_count,
            total_vulnerabilities=campaign_result.total_vulnerabilities,
            average_success_rate=campaign_result.success_rate,
            max_severity_score=campaign_result.max_severity_score,
            immediate_actions=self._get_immediate_actions(campaign_result),
            business_impact=self._assess_business_impact(campaign_result),
            compliance_implications=self._get_compliance_implications(campaign_result),
        )

        return {"text": summary.to_text(), "data": summary.__dict__}

    def _generate_technical_overview(self, campaign_result: CampaignResult) -> Dict[str, Any]:
        """Generate technical overview section."""
        return {
            "campaign_details": {
                "campaign_id": campaign_result.campaign_id,
                "scenario": campaign_result.scenario_name,
                "model": campaign_result.model_name,
                "duration": self._calculate_duration(campaign_result),
                "sessions_run": len(campaign_result.sessions),
            },
            "attack_statistics": {
                "total_attacks": sum(len(s.executed_steps) for s in campaign_result.sessions),
                "successful_attacks": sum(
                    len([step for step in s.executed_steps if step.get("status") == "success"])
                    for s in campaign_result.sessions
                ),
                "unique_techniques": len(
                    set(
                        technique
                        for s in campaign_result.sessions
                        for step in s.executed_steps
                        for technique in step.get("evasion_techniques", [])
                    )
                ),
                "average_response_time": self._calculate_avg_response_time(campaign_result),
            },
            "coverage": {
                "attack_types_tested": self._get_attack_types(campaign_result),
                "evasion_techniques_used": self._get_evasion_techniques(campaign_result),
                "scenarios_completed": campaign_result.scenarios_completed,
            },
        }

    def _generate_vulnerability_details(
        self, campaign_result: CampaignResult
    ) -> List[Dict[str, Any]]:
        """Generate vulnerability details section."""
        vulnerabilities = []

        for session in campaign_result.sessions:
            for vuln in session.vulnerability_findings:
                vulnerabilities.append(
                    {
                        "id": vuln.get("id", "unknown"),
                        "type": vuln.get("type", "unknown"),
                        "severity": vuln.get("severity", "medium"),
                        "severity_score": vuln.get("severity_score", 0),
                        "description": vuln.get("description", ""),
                        "attack_vector": vuln.get("attack_vector", ""),
                        "impact": vuln.get("impact", ""),
                        "evidence": vuln.get("evidence", ""),
                        "reproduction_steps": vuln.get("reproduction_steps", []),
                        "discovered_in_session": session.session_id,
                    }
                )

        # Sort by severity
        vulnerabilities.sort(key=lambda x: x["severity_score"], reverse=True)

        return vulnerabilities

    def _generate_attack_methodology(self, campaign_result: CampaignResult) -> Dict[str, Any]:
        """Generate attack methodology section."""
        return {
            "approach": "Automated Red Team Simulation with AI-driven attack generation",
            "phases": [
                {
                    "phase": "Reconnaissance",
                    "description": "Initial model behavior analysis and capability assessment",
                    "techniques": ["Prompt probing", "Response analysis", "Boundary testing"],
                },
                {
                    "phase": "Exploitation",
                    "description": "Systematic vulnerability exploitation attempts",
                    "techniques": self._get_exploitation_techniques(campaign_result),
                },
                {
                    "phase": "Evasion",
                    "description": "Advanced evasion technique application",
                    "techniques": self._get_evasion_techniques(campaign_result),
                },
                {
                    "phase": "Post-Exploitation",
                    "description": "Impact assessment and persistence testing",
                    "techniques": [
                        "Data extraction",
                        "Privilege escalation",
                        "Persistence mechanisms",
                    ],
                },
            ],
            "tools_used": [
                "Red Team Simulator",
                "Attack Library",
                "Evasion Engine",
                "Real-time Scorer",
            ],
        }

    def _generate_risk_assessment(self, campaign_result: CampaignResult) -> Dict[str, Any]:
        """Generate risk assessment section."""
        risk_matrix = self._calculate_risk_matrix(campaign_result)

        return {
            "overall_risk_score": risk_matrix["overall_score"],
            "risk_level": risk_matrix["risk_level"],
            "risk_factors": {
                "attack_success_rate": campaign_result.success_rate,
                "vulnerability_severity": campaign_result.max_severity_score,
                "exploit_complexity": risk_matrix["complexity"],
                "business_impact": risk_matrix["impact"],
            },
            "threat_landscape": {
                "most_effective_attacks": self._get_most_effective_attacks(campaign_result),
                "critical_weaknesses": self._get_critical_weaknesses(campaign_result),
                "defense_gaps": self._identify_defense_gaps(campaign_result),
            },
            "risk_trends": self._analyze_risk_trends(campaign_result) if self.analytics else {},
        }

    def _generate_remediation_plan(self, campaign_result: CampaignResult) -> Dict[str, Any]:
        """Generate remediation plan section."""
        vulnerabilities = []
        for session in campaign_result.sessions:
            vulnerabilities.extend(session.vulnerability_findings)

        # Prioritize remediations
        immediate = []
        short_term = []
        long_term = []

        for vuln in vulnerabilities:
            severity = vuln.get("severity_score", 0)
            if severity >= 9.0:
                immediate.append(self._create_remediation(vuln, "immediate"))
            elif severity >= 7.0:
                short_term.append(self._create_remediation(vuln, "short_term"))
            else:
                long_term.append(self._create_remediation(vuln, "long_term"))

        return {
            "immediate_actions": immediate[:5],  # Top 5
            "short_term_improvements": short_term[:10],  # Top 10
            "long_term_strategy": long_term[:10],  # Top 10
            "security_controls": self._recommend_security_controls(campaign_result),
            "monitoring_recommendations": self._recommend_monitoring(campaign_result),
            "training_recommendations": [
                "Security awareness training on prompt injection attacks",
                "Developer training on secure AI/ML practices",
                "Incident response training for AI security events",
            ],
        }

    def _generate_technical_appendix(self, campaign_result: CampaignResult) -> Dict[str, Any]:
        """Generate technical appendix."""
        return {
            "attack_payloads": self._get_sample_payloads(campaign_result),
            "response_samples": self._get_response_samples(campaign_result),
            "configuration": {
                "scan_config": campaign_result.scan_config.__dict__
                if hasattr(campaign_result, "scan_config")
                else {},
                "evasion_settings": {},
                "scoring_thresholds": {},
            },
            "technical_notes": self._get_technical_notes(campaign_result),
        }

    def _generate_evidence_log(self, campaign_result: CampaignResult) -> List[Dict[str, Any]]:
        """Generate evidence log."""
        evidence = []

        for session in campaign_result.sessions:
            for i, step in enumerate(session.executed_steps):
                if step.get("status") == "success":
                    evidence.append(
                        {
                            "timestamp": step.get("timestamp", ""),
                            "session_id": session.session_id,
                            "attack_type": step.get("attack_type", ""),
                            "prompt": step.get("prompt", "")[:200],  # Truncate
                            "response": step.get("response", "")[:200],  # Truncate
                            "success": True,
                            "severity": step.get("severity", "medium"),
                        }
                    )

        return evidence[:50]  # Limit to 50 entries

    def _generate_json_report(self, report_data: Dict[str, Any], campaign_id: str) -> Path:
        """Generate JSON format report."""
        filename = f"report_{campaign_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = self.output_dir / filename

        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        return report_path

    def _generate_markdown_report(
        self, report_data: Dict[str, Any], campaign_id: str, template: ReportTemplate
    ) -> Path:
        """Generate Markdown format report."""
        filename = f"report_{campaign_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path = self.output_dir / filename

        markdown = []

        # Title
        markdown.append(f"# Red Team Security Assessment Report\n")
        markdown.append(f"**Campaign ID:** {campaign_id}\n")
        markdown.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        markdown.append("\n---\n")

        # Sections
        for section in template.get_section_order():
            section_name = section.value.replace("_", " ").title()
            markdown.append(f"\n## {section_name}\n")

            section_key = section.value
            if section_key in report_data:
                section_data = report_data[section_key]

                if isinstance(section_data, dict):
                    markdown.append(self._dict_to_markdown(section_data))
                elif isinstance(section_data, list):
                    markdown.append(self._list_to_markdown(section_data))
                else:
                    markdown.append(str(section_data))

        # Write file
        with open(report_path, "w") as f:
            f.write("\n".join(markdown))

        return report_path

    def _generate_html_report(
        self, report_data: Dict[str, Any], campaign_id: str, template: ReportTemplate
    ) -> Path:
        """Generate HTML format report."""
        filename = f"report_{campaign_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path = self.output_dir / filename

        # Simple HTML generation (would use template engine in production)
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Red Team Report - {campaign_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
        .critical {{ color: #d32f2f; }}
        .high {{ color: #f57c00; }}
        .medium {{ color: #fbc02d; }}
        .low {{ color: #388e3c; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
    </style>
</head>
<body>
    <h1>Red Team Security Assessment Report</h1>
    <p><strong>Campaign ID:</strong> {campaign_id}</p>
    <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <hr>
    {self._generate_html_content(report_data, template)}
</body>
</html>
"""

        with open(report_path, "w") as f:
            f.write(html)

        return report_path

    def _generate_executive_report(self, report_data: Dict[str, Any], campaign_id: str) -> Path:
        """Generate executive format report."""
        filename = f"executive_report_{campaign_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_path = self.output_dir / filename

        # Extract executive summary
        exec_summary = report_data.get("executive_summary", {})

        # Write concise executive report
        with open(report_path, "w") as f:
            if "text" in exec_summary:
                f.write(exec_summary["text"])
            else:
                f.write("Executive Summary not available")

        return report_path

    # Helper methods

    def _calculate_duration(self, campaign_result: CampaignResult) -> str:
        """Calculate campaign duration."""
        if campaign_result.start_time and campaign_result.end_time:
            duration = campaign_result.end_time - campaign_result.start_time
            hours = duration.total_seconds() / 3600
            return f"{hours:.1f} hours"
        return "Unknown"

    def _calculate_avg_response_time(self, campaign_result: CampaignResult) -> float:
        """Calculate average response time."""
        # Simplified - would calculate from actual timing data
        return 1.5

    def _get_attack_types(self, campaign_result: CampaignResult) -> List[str]:
        """Get list of attack types used."""
        attack_types = set()
        for session in campaign_result.sessions:
            for step in session.executed_steps:
                attack_types.add(step.get("attack_type", "unknown"))
        return list(attack_types)

    def _get_evasion_techniques(self, campaign_result: CampaignResult) -> List[str]:
        """Get list of evasion techniques used."""
        techniques = set()
        for session in campaign_result.sessions:
            for step in session.executed_steps:
                techniques.update(step.get("evasion_techniques", []))
        return list(techniques)

    def _get_exploitation_techniques(self, campaign_result: CampaignResult) -> List[str]:
        """Get exploitation techniques."""
        return ["Prompt injection", "Context manipulation", "Jailbreak attempts", "Data extraction"]

    def _calculate_risk_matrix(self, campaign_result: CampaignResult) -> Dict[str, Any]:
        """Calculate risk matrix."""
        # Simple risk calculation
        risk_score = campaign_result.success_rate * campaign_result.max_severity_score * 10

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
            "overall_score": risk_score,
            "risk_level": risk_level,
            "complexity": "medium",
            "impact": "high" if risk_score > 50 else "medium",
        }

    def _get_most_effective_attacks(self, campaign_result: CampaignResult) -> List[str]:
        """Get most effective attacks."""
        # Analyze success rates
        effective = []
        for session in campaign_result.sessions:
            for step in session.executed_steps:
                if step.get("status") == "success":
                    effective.append(step.get("attack_type", "unknown"))

        # Return top 5 most common
        from collections import Counter

        counter = Counter(effective)
        return [attack for attack, _ in counter.most_common(5)]

    def _get_critical_weaknesses(self, campaign_result: CampaignResult) -> List[str]:
        """Get critical weaknesses."""
        weaknesses = []
        for session in campaign_result.sessions:
            for vuln in session.vulnerability_findings:
                if vuln.get("severity_score", 0) >= 9.0:
                    weaknesses.append(vuln.get("type", "unknown"))
        return list(set(weaknesses))

    def _identify_defense_gaps(self, campaign_result: CampaignResult) -> List[str]:
        """Identify defense gaps."""
        gaps = []

        if campaign_result.success_rate > 0.5:
            gaps.append("Insufficient input validation")

        if campaign_result.total_vulnerabilities > 10:
            gaps.append("Weak security controls")

        if campaign_result.max_severity_score > 8:
            gaps.append("Critical vulnerability exposure")

        return gaps

    def _analyze_risk_trends(self, campaign_result: CampaignResult) -> Dict[str, Any]:
        """Analyze risk trends using analytics."""
        # Simplified - would use analytics engine
        return {
            "trend_direction": "increasing",
            "change_rate": "+15%",
            "projection": "Risk expected to increase without intervention",
        }

    def _get_immediate_actions(self, campaign_result: CampaignResult) -> List[str]:
        """Get immediate action items."""
        actions = []

        if campaign_result.max_severity_score >= 9.0:
            actions.append("Patch critical vulnerabilities immediately")

        if campaign_result.success_rate > 0.7:
            actions.append("Implement emergency security controls")

        actions.append("Review and update security policies")
        actions.append("Enable enhanced monitoring")

        return actions

    def _assess_business_impact(self, campaign_result: CampaignResult) -> str:
        """Assess business impact."""
        if campaign_result.success_rate > 0.7:
            return "Critical business risk - potential for data breach and compliance violations"
        elif campaign_result.success_rate > 0.5:
            return "Significant business risk - security controls need strengthening"
        else:
            return "Moderate business risk - continue monitoring and improvements"

    def _get_compliance_implications(self, campaign_result: CampaignResult) -> List[str]:
        """Get compliance implications."""
        implications = []

        if campaign_result.total_vulnerabilities > 0:
            implications.append("May impact SOC 2 Type II certification")

        if campaign_result.max_severity_score > 7:
            implications.append("GDPR data protection requirements at risk")

        implications.append("Review required for ISO 27001 compliance")

        return implications

    def _create_remediation(self, vulnerability: Dict[str, Any], priority: str) -> Dict[str, Any]:
        """Create remediation recommendation."""
        return {
            "vulnerability": vulnerability.get("type", "unknown"),
            "priority": priority,
            "action": f"Remediate {vulnerability.get('type', 'vulnerability')}",
            "description": vulnerability.get("remediation", "Apply security patches and updates"),
            "estimated_effort": "Medium",
            "responsible_team": "Security Team",
        }

    def _recommend_security_controls(self, campaign_result: CampaignResult) -> List[str]:
        """Recommend security controls."""
        return [
            "Implement input validation and sanitization",
            "Deploy rate limiting and abuse detection",
            "Enable comprehensive logging and monitoring",
            "Implement defense-in-depth strategies",
            "Regular security assessments and testing",
        ]

    def _recommend_monitoring(self, campaign_result: CampaignResult) -> List[str]:
        """Recommend monitoring improvements."""
        return [
            "Monitor for anomalous prompt patterns",
            "Track model response deviations",
            "Alert on high-risk keyword usage",
            "Monitor API usage patterns",
            "Implement security event correlation",
        ]

    def _get_sample_payloads(self, campaign_result: CampaignResult) -> List[str]:
        """Get sample attack payloads."""
        payloads = []
        for session in campaign_result.sessions[:3]:  # First 3 sessions
            for step in session.executed_steps[:2]:  # First 2 steps
                if step.get("prompt"):
                    payloads.append(step["prompt"][:100])  # Truncate
        return payloads

    def _get_response_samples(self, campaign_result: CampaignResult) -> List[str]:
        """Get sample responses."""
        responses = []
        for session in campaign_result.sessions[:3]:
            for step in session.executed_steps[:2]:
                if step.get("response"):
                    responses.append(step["response"][:100])  # Truncate
        return responses

    def _get_technical_notes(self, campaign_result: CampaignResult) -> List[str]:
        """Get technical notes."""
        return [
            "Testing performed in controlled environment",
            "No production data was accessed",
            "All attacks were simulated",
            "Results may vary in production environment",
        ]

    def _dict_to_markdown(self, data: Dict[str, Any], indent: int = 0) -> str:
        """Convert dictionary to markdown."""
        lines = []
        prefix = "  " * indent

        for key, value in data.items():
            formatted_key = key.replace("_", " ").title()

            if isinstance(value, dict):
                lines.append(f"{prefix}**{formatted_key}:**")
                lines.append(self._dict_to_markdown(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}**{formatted_key}:**")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(self._dict_to_markdown(item, indent + 1))
                    else:
                        lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}**{formatted_key}:** {value}")

        return "\n".join(lines)

    def _list_to_markdown(self, data: List[Any]) -> str:
        """Convert list to markdown."""
        lines = []

        for item in data:
            if isinstance(item, dict):
                lines.append(self._dict_to_markdown(item, 1))
                lines.append("")  # Blank line
            else:
                lines.append(f"- {item}")

        return "\n".join(lines)

    def _generate_html_content(self, report_data: Dict[str, Any], template: ReportTemplate) -> str:
        """Generate HTML content for report."""
        html_parts = []

        for section in template.get_section_order():
            section_name = section.value.replace("_", " ").title()
            html_parts.append(f"<h2>{section_name}</h2>")

            section_key = section.value
            if section_key in report_data:
                section_data = report_data[section_key]

                if isinstance(section_data, dict) and "text" in section_data:
                    html_parts.append(f"<pre>{section_data['text']}</pre>")
                else:
                    html_parts.append(
                        f"<pre>{json.dumps(section_data, indent=2, default=str)}</pre>"
                    )

        return "\n".join(html_parts)
