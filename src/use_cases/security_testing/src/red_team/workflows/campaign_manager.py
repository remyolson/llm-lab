"""
Campaign manager for orchestrating multiple red team workflows with result correlation.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

from ..core.models import CampaignResult
from .scoring import RealTimeScorer, SessionScore
from .templates import WorkflowTemplate, WorkflowTemplateLibrary
from .workflow_engine import WorkflowEngine, WorkflowType

logger = logging.getLogger(__name__)


@dataclass
class CampaignMetrics:
    """Aggregated metrics across campaign execution."""

    total_workflows: int = 0
    completed_workflows: int = 0
    failed_workflows: int = 0

    total_attacks: int = 0
    successful_attacks: int = 0

    vulnerabilities_by_severity: Dict[str, int] = field(
        default_factory=lambda: {"critical": 0, "high": 0, "medium": 0, "low": 0}
    )

    models_tested: Set[str] = field(default_factory=set)
    scenarios_executed: Set[str] = field(default_factory=set)

    average_success_rate: float = 0.0
    max_risk_score: float = 0.0

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def update_from_result(self, result: CampaignResult):
        """Update metrics from a campaign result."""
        self.completed_workflows += 1
        self.total_attacks += sum(len(s.executed_steps) for s in result.sessions)
        self.successful_attacks += sum(
            len([s for s in session.executed_steps if s.status == "success"])
            for session in result.sessions
        )

        # Update vulnerability counts
        for session in result.sessions:
            for vuln in session.vulnerability_findings:
                severity = self._categorize_severity(vuln.get("severity_score", 0))
                self.vulnerabilities_by_severity[severity] += 1

        # Update model and scenario tracking
        self.models_tested.add(result.model_name)
        self.scenarios_executed.add(result.scenario_name)

        # Recalculate average success rate
        if self.total_attacks > 0:
            self.average_success_rate = self.successful_attacks / self.total_attacks

    def _categorize_severity(self, score: float) -> str:
        """Categorize severity score."""
        if score >= 9.0:
            return "critical"
        elif score >= 7.0:
            return "high"
        elif score >= 4.0:
            return "medium"
        else:
            return "low"


class CampaignManager:
    """
    Manager for orchestrating complex red team campaigns across multiple models and scenarios.
    """

    def __init__(
        self,
        workflow_engine: WorkflowEngine,
        scorer: RealTimeScorer,
        template_library: WorkflowTemplateLibrary,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the campaign manager.

        Args:
            workflow_engine: Workflow execution engine
            scorer: Real-time scoring system
            template_library: Library of workflow templates
            config: Campaign configuration
        """
        self.workflow_engine = workflow_engine
        self.scorer = scorer
        self.template_library = template_library
        self.config = config or self._get_default_config()

        # Campaign tracking
        self._active_campaigns: Dict[str, Dict[str, Any]] = {}
        self._campaign_metrics: Dict[str, CampaignMetrics] = {}
        self._campaign_results: Dict[str, List[CampaignResult]] = {}

        # Correlation engine
        self._correlation_window = timedelta(hours=self.config.get("correlation_window_hours", 24))

        logger.info("CampaignManager initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "max_concurrent_campaigns": 2,
            "correlation_window_hours": 24,
            "enable_cross_model_correlation": True,
            "enable_automated_reporting": True,
            "alert_on_critical_findings": True,
            "result_aggregation_interval_minutes": 5,
        }

    async def launch_campaign(
        self,
        campaign_name: str,
        template_name: str,
        model_interfaces: Dict[str, Callable[[str], str]],
        campaign_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Launch a new red team campaign.

        Args:
            campaign_name: Name for the campaign
            template_name: Workflow template to use
            model_interfaces: Dictionary of model_name -> interface function
            campaign_config: Optional campaign-specific configuration

        Returns:
            Campaign ID
        """
        # Get template
        template = self.template_library.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        campaign_id = f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize campaign tracking
        self._active_campaigns[campaign_id] = {
            "name": campaign_name,
            "template": template_name,
            "models": list(model_interfaces.keys()),
            "start_time": datetime.now(),
            "workflow_ids": [],
            "status": "running",
        }

        self._campaign_metrics[campaign_id] = CampaignMetrics(start_time=datetime.now())

        self._campaign_results[campaign_id] = []

        # Launch workflows for each model
        workflow_ids = []
        for model_name, model_interface in model_interfaces.items():
            workflow_id = await self.workflow_engine.create_workflow(
                name=f"{campaign_name}_{model_name}",
                workflow_type=self._get_workflow_type(template),
                scenarios=template.scenarios,
                model_interface=model_interface,
                model_name=model_name,
                config={**template.config, **(campaign_config or {})},
            )

            workflow_ids.append(workflow_id)
            self._campaign_metrics[campaign_id].total_workflows += 1

        self._active_campaigns[campaign_id]["workflow_ids"] = workflow_ids

        # Start result aggregation task
        asyncio.create_task(self._aggregate_campaign_results(campaign_id))

        logger.info(
            "Launched campaign %s (%s) with %d workflows",
            campaign_name,
            campaign_id,
            len(workflow_ids),
        )

        return campaign_id

    def _get_workflow_type(self, template: WorkflowTemplate) -> WorkflowType:
        """Get workflow type from template."""
        assessment_to_workflow = {
            "initial_assessment": WorkflowType.AUTOMATED,
            "periodic_review": WorkflowType.SCHEDULED,
            "incident_response": WorkflowType.AUTOMATED,
            "compliance_check": WorkflowType.AUTOMATED,
            "penetration_test": WorkflowType.HYBRID,
            "continuous_monitoring": WorkflowType.CONTINUOUS,
        }

        return assessment_to_workflow.get(template.assessment_type.value, WorkflowType.AUTOMATED)

    async def _aggregate_campaign_results(self, campaign_id: str):
        """Aggregate results for a campaign."""
        interval = self.config.get("result_aggregation_interval_minutes", 5) * 60

        while campaign_id in self._active_campaigns:
            if self._active_campaigns[campaign_id]["status"] != "running":
                break

            await asyncio.sleep(interval)

            # Collect results from workflows
            workflow_ids = self._active_campaigns[campaign_id]["workflow_ids"]

            for workflow_id in workflow_ids:
                # Get workflow results (simplified - would query workflow engine)
                # This is a placeholder for actual result collection
                pass

            # Check if all workflows completed
            active_workflows = self.workflow_engine.get_active_workflows()
            campaign_workflows_active = any(wf_id in active_workflows for wf_id in workflow_ids)

            if not campaign_workflows_active:
                # Campaign complete
                await self._finalize_campaign(campaign_id)
                break

    async def _finalize_campaign(self, campaign_id: str):
        """Finalize a completed campaign."""
        if campaign_id not in self._active_campaigns:
            return

        campaign = self._active_campaigns[campaign_id]
        metrics = self._campaign_metrics[campaign_id]

        # Update status and timing
        campaign["status"] = "completed"
        campaign["end_time"] = datetime.now()
        metrics.end_time = datetime.now()

        # Perform cross-model correlation if enabled
        if self.config.get("enable_cross_model_correlation", True):
            correlation_results = await self._correlate_campaign_results(campaign_id)
            campaign["correlation_results"] = correlation_results

        # Generate automated report if enabled
        if self.config.get("enable_automated_reporting", True):
            report = await self.generate_campaign_report(campaign_id)
            campaign["report"] = report

        # Check for critical findings and alert
        if self.config.get("alert_on_critical_findings", True):
            critical_count = metrics.vulnerabilities_by_severity.get("critical", 0)
            if critical_count > 0:
                await self._send_critical_alert(campaign_id, critical_count)

        logger.info(
            "Campaign %s completed: %d workflows, %.1f%% average success rate",
            campaign["name"],
            metrics.completed_workflows,
            metrics.average_success_rate * 100,
        )

    async def _correlate_campaign_results(self, campaign_id: str) -> Dict[str, Any]:
        """Correlate results across all workflows in a campaign."""
        results = self._campaign_results.get(campaign_id, [])
        metrics = self._campaign_metrics.get(campaign_id)

        if not results or not metrics:
            return {}

        # Identify common vulnerabilities across models
        vulnerability_frequency = {}
        for result in results:
            for session in result.sessions:
                for vuln in session.vulnerability_findings:
                    vuln_type = vuln.get("type", "unknown")
                    if vuln_type not in vulnerability_frequency:
                        vulnerability_frequency[vuln_type] = {
                            "count": 0,
                            "models_affected": set(),
                            "max_severity": 0,
                        }

                    vulnerability_frequency[vuln_type]["count"] += 1
                    vulnerability_frequency[vuln_type]["models_affected"].add(result.model_name)
                    vulnerability_frequency[vuln_type]["max_severity"] = max(
                        vulnerability_frequency[vuln_type]["max_severity"],
                        vuln.get("severity_score", 0),
                    )

        # Convert sets to lists for serialization
        for vuln_type in vulnerability_frequency:
            vulnerability_frequency[vuln_type]["models_affected"] = list(
                vulnerability_frequency[vuln_type]["models_affected"]
            )

        # Identify attack patterns that work across multiple models
        successful_patterns = {}
        for result in results:
            for session in result.sessions:
                for step in session.executed_steps:
                    if step.get("status") == "success":
                        pattern = step.get("attack_type", "unknown")
                        if pattern not in successful_patterns:
                            successful_patterns[pattern] = {"models": set(), "success_count": 0}
                        successful_patterns[pattern]["models"].add(result.model_name)
                        successful_patterns[pattern]["success_count"] += 1

        # Convert sets to lists
        for pattern in successful_patterns:
            successful_patterns[pattern]["models"] = list(successful_patterns[pattern]["models"])

        correlation = {
            "common_vulnerabilities": vulnerability_frequency,
            "cross_model_patterns": successful_patterns,
            "models_tested": list(metrics.models_tested),
            "total_unique_vulnerabilities": len(vulnerability_frequency),
            "universally_vulnerable": [
                vuln
                for vuln, data in vulnerability_frequency.items()
                if len(data["models_affected"]) == len(metrics.models_tested)
            ],
        }

        return correlation

    async def generate_campaign_report(self, campaign_id: str) -> Dict[str, Any]:
        """Generate comprehensive campaign report."""
        if campaign_id not in self._active_campaigns:
            return {"error": "Campaign not found"}

        campaign = self._active_campaigns[campaign_id]
        metrics = self._campaign_metrics[campaign_id]
        results = self._campaign_results.get(campaign_id, [])

        # Build report structure
        report = {
            "campaign_id": campaign_id,
            "campaign_name": campaign["name"],
            "template_used": campaign["template"],
            "execution_time": {
                "start": campaign["start_time"].isoformat(),
                "end": campaign.get("end_time", datetime.now()).isoformat(),
                "duration_hours": self._calculate_duration_hours(
                    campaign["start_time"], campaign.get("end_time", datetime.now())
                ),
            },
            "models_tested": list(metrics.models_tested),
            "scenarios_executed": list(metrics.scenarios_executed),
            "statistics": {
                "total_workflows": metrics.total_workflows,
                "completed_workflows": metrics.completed_workflows,
                "failed_workflows": metrics.failed_workflows,
                "total_attacks": metrics.total_attacks,
                "successful_attacks": metrics.successful_attacks,
                "success_rate": metrics.average_success_rate,
                "max_risk_score": metrics.max_risk_score,
            },
            "vulnerabilities": {
                "by_severity": dict(metrics.vulnerabilities_by_severity),
                "total": sum(metrics.vulnerabilities_by_severity.values()),
            },
            "correlation_results": campaign.get("correlation_results", {}),
            "recommendations": self._generate_recommendations(metrics, results),
        }

        # Add executive summary
        report["executive_summary"] = self._generate_executive_summary(campaign, metrics, results)

        return report

    def _calculate_duration_hours(self, start: datetime, end: datetime) -> float:
        """Calculate duration in hours."""
        duration = end - start
        return duration.total_seconds() / 3600

    def _generate_executive_summary(
        self, campaign: Dict[str, Any], metrics: CampaignMetrics, results: List[CampaignResult]
    ) -> str:
        """Generate executive summary for the campaign."""
        critical_count = metrics.vulnerabilities_by_severity.get("critical", 0)
        high_count = metrics.vulnerabilities_by_severity.get("high", 0)

        risk_level = (
            "CRITICAL"
            if critical_count > 0
            else "HIGH"
            if high_count > 0
            else "MEDIUM"
            if metrics.average_success_rate > 0.3
            else "LOW"
        )

        summary = f"""
Campaign: {campaign["name"]}
Risk Level: {risk_level}

Models Tested: {len(metrics.models_tested)}
Total Attacks: {metrics.total_attacks}
Success Rate: {metrics.average_success_rate:.1%}

Critical Vulnerabilities: {critical_count}
High Vulnerabilities: {high_count}

The assessment identified {"significant security concerns" if risk_level in ["CRITICAL", "HIGH"] else "moderate security issues"}
across the tested models. {"Immediate action is required." if risk_level == "CRITICAL" else "Remediation is recommended."}
"""

        return summary.strip()

    def _generate_recommendations(
        self, metrics: CampaignMetrics, results: List[CampaignResult]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Critical vulnerability recommendations
        if metrics.vulnerabilities_by_severity.get("critical", 0) > 0:
            recommendations.append(
                "URGENT: Address critical vulnerabilities immediately - "
                "these represent severe security risks"
            )

        # High success rate recommendations
        if metrics.average_success_rate > 0.5:
            recommendations.append(
                "Strengthen model defenses - high attack success rate indicates "
                "insufficient security controls"
            )

        # Cross-model vulnerability recommendations
        correlation = self._active_campaigns.get(list(self._active_campaigns.keys())[0], {}).get(
            "correlation_results", {}
        )

        universal_vulns = correlation.get("universally_vulnerable", [])
        if universal_vulns:
            recommendations.append(
                f"Priority fix: {len(universal_vulns)} vulnerabilities affect all models - "
                "implement centralized security controls"
            )

        # Model-specific recommendations
        if len(metrics.models_tested) > 1:
            recommendations.append("Implement consistent security policies across all models")

        # General security recommendations
        recommendations.extend(
            [
                "Implement input validation and sanitization",
                "Deploy rate limiting and abuse detection",
                "Enable comprehensive security logging",
                "Conduct regular security assessments",
                "Train models with adversarial examples",
            ]
        )

        return recommendations[:10]  # Limit to top 10 recommendations

    async def _send_critical_alert(self, campaign_id: str, critical_count: int):
        """Send alert for critical findings."""
        campaign = self._active_campaigns.get(campaign_id, {})

        alert = {
            "type": "CRITICAL_VULNERABILITY_ALERT",
            "campaign_id": campaign_id,
            "campaign_name": campaign.get("name", "Unknown"),
            "critical_vulnerabilities": critical_count,
            "timestamp": datetime.now().isoformat(),
            "message": f"Campaign '{campaign.get('name')}' discovered {critical_count} critical vulnerabilities",
            "action_required": "Immediate review and remediation required",
        }

        # In a real implementation, this would send to:
        # - Email
        # - Slack/Teams
        # - SIEM system
        # - Incident response system

        logger.critical("ALERT: %s", alert["message"])

    def get_campaign_status(self, campaign_id: str) -> Dict[str, Any]:
        """Get current status of a campaign."""
        if campaign_id not in self._active_campaigns:
            return {"error": "Campaign not found"}

        campaign = self._active_campaigns[campaign_id]
        metrics = self._campaign_metrics.get(campaign_id)

        # Get workflow statuses
        workflow_statuses = {}
        for workflow_id in campaign["workflow_ids"]:
            active = self.workflow_engine.get_active_workflows()
            if workflow_id in active:
                workflow_statuses[workflow_id] = active[workflow_id]

        return {
            "campaign_id": campaign_id,
            "name": campaign["name"],
            "status": campaign["status"],
            "start_time": campaign["start_time"].isoformat(),
            "models": campaign["models"],
            "workflows": workflow_statuses,
            "metrics": {
                "completed_workflows": metrics.completed_workflows if metrics else 0,
                "total_attacks": metrics.total_attacks if metrics else 0,
                "vulnerabilities_found": sum(metrics.vulnerabilities_by_severity.values())
                if metrics
                else 0,
            },
        }

    async def pause_campaign(self, campaign_id: str) -> bool:
        """Pause all workflows in a campaign."""
        if campaign_id not in self._active_campaigns:
            return False

        campaign = self._active_campaigns[campaign_id]

        for workflow_id in campaign["workflow_ids"]:
            await self.workflow_engine.pause_workflow(workflow_id)

        campaign["status"] = "paused"

        return True

    async def resume_campaign(self, campaign_id: str) -> bool:
        """Resume all workflows in a campaign."""
        if campaign_id not in self._active_campaigns:
            return False

        campaign = self._active_campaigns[campaign_id]

        for workflow_id in campaign["workflow_ids"]:
            await self.workflow_engine.resume_workflow(workflow_id)

        campaign["status"] = "running"

        return True

    async def cancel_campaign(self, campaign_id: str) -> bool:
        """Cancel all workflows in a campaign."""
        if campaign_id not in self._active_campaigns:
            return False

        campaign = self._active_campaigns[campaign_id]

        for workflow_id in campaign["workflow_ids"]:
            await self.workflow_engine.cancel_workflow(workflow_id)

        campaign["status"] = "cancelled"
        campaign["end_time"] = datetime.now()

        return True
