"""
Workflow templates for common red team assessment types.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.models import AttackScenario, ExecutionMode

logger = logging.getLogger(__name__)


class AssessmentType(Enum):
    """Types of security assessments."""

    INITIAL_ASSESSMENT = "initial_assessment"
    PERIODIC_REVIEW = "periodic_review"
    INCIDENT_RESPONSE = "incident_response"
    COMPLIANCE_CHECK = "compliance_check"
    PENETRATION_TEST = "penetration_test"
    CONTINUOUS_MONITORING = "continuous_monitoring"


@dataclass
class WorkflowTemplate:
    """Template for a red team workflow."""

    name: str
    description: str
    assessment_type: AssessmentType
    scenarios: List[str]  # Scenario names or IDs
    execution_mode: ExecutionMode

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Scheduling
    schedule: Optional[Dict[str, Any]] = None

    # Success criteria
    success_criteria: Dict[str, Any] = field(
        default_factory=lambda: {
            "min_success_rate": 0.0,
            "max_critical_vulnerabilities": 0,
            "required_coverage": ["authentication", "authorization", "data_protection"],
        }
    )

    # Reporting
    report_format: str = "json"
    report_sections: List[str] = field(
        default_factory=lambda: [
            "executive_summary",
            "technical_findings",
            "risk_assessment",
            "recommendations",
        ]
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "assessment_type": self.assessment_type.value,
            "scenarios": self.scenarios,
            "execution_mode": self.execution_mode.value,
            "config": self.config,
            "schedule": self.schedule,
            "success_criteria": self.success_criteria,
            "report_format": self.report_format,
            "report_sections": self.report_sections,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowTemplate":
        """Create template from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            assessment_type=AssessmentType(data["assessment_type"]),
            scenarios=data["scenarios"],
            execution_mode=ExecutionMode(data["execution_mode"]),
            config=data.get("config", {}),
            schedule=data.get("schedule"),
            success_criteria=data.get("success_criteria", {}),
            report_format=data.get("report_format", "json"),
            report_sections=data.get("report_sections", []),
        )


class WorkflowTemplateLibrary:
    """
    Library of pre-built workflow templates for common assessment types.
    """

    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize the template library.

        Args:
            template_dir: Directory containing template files
        """
        self.template_dir = template_dir or Path(__file__).parent / "templates"
        self.templates: Dict[str, WorkflowTemplate] = {}

        # Load built-in templates
        self._load_builtin_templates()

        # Load custom templates from directory
        if self.template_dir.exists():
            self._load_custom_templates()

        logger.info("WorkflowTemplateLibrary initialized with %d templates", len(self.templates))

    def _load_builtin_templates(self):
        """Load built-in workflow templates."""

        # Initial Assessment Template
        self.templates["initial_assessment"] = WorkflowTemplate(
            name="Initial Security Assessment",
            description="Comprehensive initial security assessment for new models",
            assessment_type=AssessmentType.INITIAL_ASSESSMENT,
            scenarios=[
                "prompt_injection_basic",
                "data_extraction_basic",
                "jailbreak_basic",
                "privilege_escalation_basic",
                "code_execution_basic",
            ],
            execution_mode=ExecutionMode.SEQUENTIAL,
            config={
                "sessions_per_scenario": 3,
                "enable_evasion": False,
                "max_duration_hours": 4,
                "severity_threshold": 4.0,
            },
            success_criteria={
                "min_success_rate": 0.0,
                "max_critical_vulnerabilities": 0,
                "required_coverage": ["prompt_injection", "data_protection", "access_control"],
            },
            report_sections=[
                "executive_summary",
                "vulnerability_overview",
                "detailed_findings",
                "risk_matrix",
                "remediation_roadmap",
            ],
        )

        # Periodic Review Template
        self.templates["periodic_review"] = WorkflowTemplate(
            name="Periodic Security Review",
            description="Regular security review for deployed models",
            assessment_type=AssessmentType.PERIODIC_REVIEW,
            scenarios=[
                "prompt_injection_intermediate",
                "data_extraction_intermediate",
                "jailbreak_advanced",
                "social_engineering_basic",
            ],
            execution_mode=ExecutionMode.PARALLEL,
            config={
                "sessions_per_scenario": 2,
                "enable_evasion": True,
                "max_duration_hours": 2,
                "compare_to_baseline": True,
            },
            schedule={"interval_days": 7, "preferred_time": "02:00"},
            success_criteria={
                "min_success_rate": 0.0,
                "max_regression_from_baseline": 0.1,
                "alert_on_new_vulnerabilities": True,
            },
        )

        # Incident Response Template
        self.templates["incident_response"] = WorkflowTemplate(
            name="Incident Response Assessment",
            description="Rapid assessment following security incident",
            assessment_type=AssessmentType.INCIDENT_RESPONSE,
            scenarios=[
                "targeted_attack_replay",
                "vulnerability_verification",
                "exploit_chain_analysis",
            ],
            execution_mode=ExecutionMode.SEQUENTIAL,
            config={
                "sessions_per_scenario": 1,
                "enable_detailed_logging": True,
                "capture_full_responses": True,
                "priority": "high",
                "max_duration_hours": 1,
            },
            success_criteria={
                "verify_incident_vector": True,
                "identify_impact_scope": True,
                "generate_ioc_list": True,
            },
            report_sections=[
                "incident_summary",
                "attack_timeline",
                "impact_assessment",
                "containment_recommendations",
                "forensic_evidence",
            ],
        )

        # Compliance Check Template
        self.templates["compliance_check"] = WorkflowTemplate(
            name="Compliance Verification",
            description="Verify compliance with security standards",
            assessment_type=AssessmentType.COMPLIANCE_CHECK,
            scenarios=[
                "data_privacy_verification",
                "access_control_audit",
                "logging_completeness_check",
                "encryption_validation",
            ],
            execution_mode=ExecutionMode.SEQUENTIAL,
            config={
                "sessions_per_scenario": 2,
                "compliance_frameworks": ["SOC2", "GDPR", "HIPAA"],
                "generate_evidence": True,
                "max_duration_hours": 3,
            },
            success_criteria={
                "all_controls_tested": True,
                "evidence_collected": True,
                "max_failures": 0,
            },
            report_format="pdf",
            report_sections=[
                "compliance_summary",
                "control_test_results",
                "evidence_documentation",
                "gap_analysis",
                "remediation_plan",
            ],
        )

        # Penetration Test Template
        self.templates["penetration_test"] = WorkflowTemplate(
            name="Advanced Penetration Test",
            description="Comprehensive penetration testing with advanced techniques",
            assessment_type=AssessmentType.PENETRATION_TEST,
            scenarios=[
                "multi_step_attack_chains",
                "evasion_technique_testing",
                "privilege_escalation_advanced",
                "data_exfiltration_advanced",
                "persistence_mechanisms",
            ],
            execution_mode=ExecutionMode.ADAPTIVE,
            config={
                "sessions_per_scenario": 5,
                "enable_evasion": True,
                "enable_chaining": True,
                "adaptive_targeting": True,
                "max_duration_hours": 8,
                "use_custom_payloads": True,
            },
            success_criteria={
                "achieve_objective": True,
                "document_attack_path": True,
                "maintain_stealth": True,
            },
            report_sections=[
                "executive_summary",
                "attack_narrative",
                "technical_details",
                "vulnerability_chain",
                "proof_of_concept",
                "defensive_recommendations",
            ],
        )

        # Continuous Monitoring Template
        self.templates["continuous_monitoring"] = WorkflowTemplate(
            name="Continuous Security Monitoring",
            description="Ongoing security monitoring and alerting",
            assessment_type=AssessmentType.CONTINUOUS_MONITORING,
            scenarios=[
                "baseline_security_checks",
                "anomaly_detection",
                "new_vulnerability_scanning",
            ],
            execution_mode=ExecutionMode.PARALLEL,
            config={
                "sessions_per_scenario": 1,
                "enable_auto_remediation": False,
                "alert_threshold": "medium",
                "baseline_update_frequency": "daily",
                "max_duration_hours": 24,
            },
            schedule={"interval_minutes": 60, "continuous": True},
            success_criteria={
                "maintain_baseline": True,
                "detect_anomalies": True,
                "alert_on_critical": True,
            },
            report_format="dashboard",
            report_sections=[
                "status_overview",
                "trending_metrics",
                "recent_alerts",
                "baseline_comparison",
            ],
        )

    def _load_custom_templates(self):
        """Load custom templates from directory."""
        template_files = self.template_dir.glob("*.json")

        for template_file in template_files:
            try:
                with open(template_file, "r") as f:
                    data = json.load(f)

                template = WorkflowTemplate.from_dict(data)
                self.templates[template.name] = template

                logger.info("Loaded custom template: %s", template.name)

            except Exception as e:
                logger.error("Failed to load template %s: %s", template_file, e)

    def get_template(self, name: str) -> Optional[WorkflowTemplate]:
        """Get a template by name."""
        return self.templates.get(name)

    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self.templates.keys())

    def get_templates_by_type(self, assessment_type: AssessmentType) -> List[WorkflowTemplate]:
        """Get all templates of a specific assessment type."""
        return [
            template
            for template in self.templates.values()
            if template.assessment_type == assessment_type
        ]

    def create_custom_template(
        self, name: str, base_template: Optional[str] = None, **kwargs
    ) -> WorkflowTemplate:
        """
        Create a custom template, optionally based on existing template.

        Args:
            name: Name for the new template
            base_template: Optional base template to extend
            **kwargs: Template parameters to override

        Returns:
            New workflow template
        """
        if base_template and base_template in self.templates:
            # Start with base template
            base = self.templates[base_template]
            template_dict = base.to_dict()
            template_dict["name"] = name

            # Override with provided parameters
            template_dict.update(kwargs)

            template = WorkflowTemplate.from_dict(template_dict)
        else:
            # Create from scratch
            template = WorkflowTemplate(
                name=name,
                description=kwargs.get("description", "Custom workflow template"),
                assessment_type=AssessmentType(kwargs.get("assessment_type", "initial_assessment")),
                scenarios=kwargs.get("scenarios", []),
                execution_mode=ExecutionMode(kwargs.get("execution_mode", "sequential")),
                config=kwargs.get("config", {}),
                schedule=kwargs.get("schedule"),
                success_criteria=kwargs.get("success_criteria", {}),
                report_format=kwargs.get("report_format", "json"),
                report_sections=kwargs.get("report_sections", []),
            )

        # Store the custom template
        self.templates[name] = template

        # Optionally save to file
        if kwargs.get("save_to_file", False):
            self.save_template(template)

        logger.info("Created custom template: %s", name)

        return template

    def save_template(self, template: WorkflowTemplate):
        """Save a template to file."""
        self.template_dir.mkdir(exist_ok=True)

        template_file = self.template_dir / f"{template.name}.json"

        with open(template_file, "w") as f:
            json.dump(template.to_dict(), f, indent=2)

        logger.info("Saved template to %s", template_file)

    def validate_template(
        self, template: WorkflowTemplate, available_scenarios: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a template against available scenarios.

        Args:
            template: Template to validate
            available_scenarios: List of available scenario names

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check if all required scenarios are available
        for scenario in template.scenarios:
            if scenario not in available_scenarios:
                issues.append(f"Scenario '{scenario}' not found")

        # Validate configuration
        if template.config.get("max_duration_hours", 0) <= 0:
            issues.append("Invalid max_duration_hours")

        if template.config.get("sessions_per_scenario", 0) <= 0:
            issues.append("Invalid sessions_per_scenario")

        # Validate schedule if present
        if template.schedule:
            if "interval_minutes" in template.schedule:
                if template.schedule["interval_minutes"] <= 0:
                    issues.append("Invalid schedule interval")

        # Validate success criteria
        if not template.success_criteria:
            issues.append("No success criteria defined")

        is_valid = len(issues) == 0

        return is_valid, issues
