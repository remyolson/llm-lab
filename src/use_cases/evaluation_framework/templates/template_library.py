"""
Evaluation Template Library

This module provides pre-configured evaluation templates for common scenarios,
including accuracy-focused, latency-optimized, cost-optimized, domain-specific,
and production readiness templates.

Example:
    library = TemplateLibrary()

    # Get accuracy-focused template
    template = library.get_template("accuracy_focused")

    # Run evaluation with template
    runner = AutoBenchmarkRunner(template.config)
    results = runner.evaluate_fine_tuning(
        base_model="model1",
        fine_tuned_model="model2"
    )
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..analysis.cost_benefit import CostAnalysisConfig

# Import evaluation components
from ..benchmark_runner import BenchmarkConfig
from ..integrations.pipeline_hooks import HookConfig, HookType, NotificationType
from ..reporting.report_generator import ReportConfig, ReportFormat, ReportSection

logger = logging.getLogger(__name__)


class TemplateType(Enum):
    """Types of evaluation templates."""

    ACCURACY_FOCUSED = "accuracy_focused"
    LATENCY_OPTIMIZED = "latency_optimized"
    COST_OPTIMIZED = "cost_optimized"
    PRODUCTION_READINESS = "production_readiness"
    QUICK_VALIDATION = "quick_validation"
    COMPREHENSIVE = "comprehensive"
    DOMAIN_SPECIFIC = "domain_specific"
    A_B_TESTING = "a_b_testing"
    REGRESSION_TESTING = "regression_testing"
    CUSTOM = "custom"


@dataclass
class EvaluationTemplate:
    """Evaluation template configuration."""

    name: str
    type: TemplateType
    description: str
    benchmark_config: BenchmarkConfig
    report_config: Optional[ReportConfig] = None
    hook_config: Optional[HookConfig] = None
    cost_config: Optional[CostAnalysisConfig] = None
    custom_metrics: List[str] = field(default_factory=list)
    recommended_for: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "benchmark_config": self.benchmark_config.__dict__,
            "report_config": self.report_config.__dict__ if self.report_config else None,
            "hook_config": self.hook_config.__dict__ if self.hook_config else None,
            "cost_config": self.cost_config.__dict__ if self.cost_config else None,
            "custom_metrics": self.custom_metrics,
            "recommended_for": self.recommended_for,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    def save(self, filepath: str):
        """Save template to file.

        Args:
            filepath: Path to save template
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, filepath: str) -> "EvaluationTemplate":
        """Load template from file.

        Args:
            filepath: Path to template file

        Returns:
            Loaded template
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        # Reconstruct configs
        benchmark_config = BenchmarkConfig(**data["benchmark_config"])

        report_config = None
        if data.get("report_config"):
            report_config = ReportConfig(**data["report_config"])

        hook_config = None
        if data.get("hook_config"):
            hook_config = HookConfig(**data["hook_config"])

        cost_config = None
        if data.get("cost_config"):
            cost_config = CostAnalysisConfig(**data["cost_config"])

        return cls(
            name=data["name"],
            type=TemplateType(data["type"]),
            description=data["description"],
            benchmark_config=benchmark_config,
            report_config=report_config,
            hook_config=hook_config,
            cost_config=cost_config,
            custom_metrics=data.get("custom_metrics", []),
            recommended_for=data.get("recommended_for", []),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


class TemplateLibrary:
    """Library of evaluation templates."""

    def __init__(self, custom_templates_dir: Optional[str] = None):
        """Initialize template library.

        Args:
            custom_templates_dir: Directory for custom templates
        """
        self.templates: Dict[str, EvaluationTemplate] = {}
        self.custom_templates_dir = Path(custom_templates_dir) if custom_templates_dir else None

        # Initialize default templates
        self._init_default_templates()

        # Load custom templates
        if self.custom_templates_dir and self.custom_templates_dir.exists():
            self._load_custom_templates()

    def _init_default_templates(self):
        """Initialize default evaluation templates."""

        # Accuracy-focused template
        self.templates["accuracy_focused"] = EvaluationTemplate(
            name="Accuracy Focused",
            type=TemplateType.ACCURACY_FOCUSED,
            description="Comprehensive accuracy evaluation with multiple benchmarks",
            benchmark_config=BenchmarkConfig(
                benchmarks=["hellaswag", "mmlu", "arc_easy", "arc_challenge", "winogrande"],
                batch_size=8,
                max_samples=None,  # Full evaluation
                cache_results=True,
                parallel_benchmarks=True,
                max_parallel_jobs=3,
                timeout_minutes=120,
            ),
            report_config=ReportConfig(
                title="Accuracy-Focused Evaluation Report",
                include_sections=[
                    ReportSection.EXECUTIVE_SUMMARY,
                    ReportSection.PERFORMANCE_METRICS,
                    ReportSection.DETAILED_BENCHMARKS,
                    ReportSection.VISUALIZATIONS,
                    ReportSection.RECOMMENDATIONS,
                ],
                format=ReportFormat.PDF,
                include_visualizations=True,
            ),
            hook_config=HookConfig(
                auto_evaluate=True,
                auto_generate_report=True,
                notification_on_completion=True,
                notification_on_regression=True,
            ),
            recommended_for=["NLU tasks", "QA systems", "General purpose models"],
            tags=["accuracy", "comprehensive", "benchmarks"],
        )

        # Latency-optimized template
        self.templates["latency_optimized"] = EvaluationTemplate(
            name="Latency Optimized",
            type=TemplateType.LATENCY_OPTIMIZED,
            description="Fast evaluation focusing on inference speed and efficiency",
            benchmark_config=BenchmarkConfig(
                benchmarks=["hellaswag", "mmlu"],  # Fewer benchmarks
                batch_size=16,  # Larger batch for throughput
                max_samples=1000,  # Limited samples for speed
                cache_results=True,
                parallel_benchmarks=False,  # Sequential for accurate timing
                timeout_minutes=30,
            ),
            report_config=ReportConfig(
                title="Latency-Optimized Evaluation Report",
                include_sections=[
                    ReportSection.EXECUTIVE_SUMMARY,
                    ReportSection.PERFORMANCE_METRICS,
                    ReportSection.COST_ANALYSIS,
                ],
                format=ReportFormat.HTML,
                include_visualizations=True,
            ),
            cost_config=CostAnalysisConfig(
                daily_requests=100000,  # High volume
                avg_input_tokens=200,
                avg_output_tokens=100,
            ),
            recommended_for=["Real-time applications", "High-traffic APIs", "Edge deployment"],
            tags=["latency", "speed", "efficiency"],
        )

        # Cost-optimized template
        self.templates["cost_optimized"] = EvaluationTemplate(
            name="Cost Optimized",
            type=TemplateType.COST_OPTIMIZED,
            description="Minimal evaluation focused on cost efficiency",
            benchmark_config=BenchmarkConfig(
                benchmarks=["hellaswag"],  # Single benchmark
                batch_size=32,  # Maximum batch size
                max_samples=500,  # Very limited samples
                cache_results=True,
                parallel_benchmarks=False,
                timeout_minutes=15,
                device="cpu",  # Use CPU to save GPU costs
            ),
            report_config=ReportConfig(
                title="Cost-Optimized Evaluation Report",
                include_sections=[
                    ReportSection.EXECUTIVE_SUMMARY,
                    ReportSection.COST_ANALYSIS,
                    ReportSection.RECOMMENDATIONS,
                ],
                format=ReportFormat.MARKDOWN,  # Simple format
                include_visualizations=False,  # Skip visualizations
            ),
            cost_config=CostAnalysisConfig(
                gpu_cost_per_hour=0,  # CPU only
                cpu_cost_per_hour=0.05,
                daily_requests=1000,
                avg_input_tokens=100,
                avg_output_tokens=50,
            ),
            recommended_for=["Budget-conscious projects", "Initial validation", "Small models"],
            tags=["cost", "budget", "minimal"],
        )

        # Production readiness template
        self.templates["production_readiness"] = EvaluationTemplate(
            name="Production Readiness",
            type=TemplateType.PRODUCTION_READINESS,
            description="Comprehensive evaluation for production deployment",
            benchmark_config=BenchmarkConfig(
                benchmarks=["hellaswag", "mmlu", "arc_easy", "truthfulqa"],
                custom_evaluations=["toxicity", "bias", "robustness"],
                batch_size=8,
                max_samples=None,
                cache_results=True,
                parallel_benchmarks=True,
                monitoring_enabled=True,
                monitoring_platforms=["tensorboard", "wandb"],
                save_intermediate=True,
            ),
            report_config=ReportConfig(
                title="Production Readiness Evaluation Report",
                include_sections=list(ReportSection),  # All sections
                format=ReportFormat.PDF,
                include_visualizations=True,
                include_raw_data=True,
                confidentiality_level="Confidential",
            ),
            hook_config=HookConfig(
                auto_evaluate=True,
                auto_generate_report=True,
                auto_deploy=False,  # Manual deployment decision
                deployment_threshold=10.0,
                notification_on_completion=True,
                notification_on_error=True,
                notification_on_regression=True,
                notification_channels=[NotificationType.EMAIL, NotificationType.SLACK],
                model_registry_enabled=True,
            ),
            cost_config=CostAnalysisConfig(
                daily_requests=50000, avg_input_tokens=500, avg_output_tokens=200
            ),
            recommended_for=[
                "Production deployment",
                "Enterprise applications",
                "Critical systems",
            ],
            tags=["production", "comprehensive", "safety", "monitoring"],
        )

        # Quick validation template
        self.templates["quick_validation"] = EvaluationTemplate(
            name="Quick Validation",
            type=TemplateType.QUICK_VALIDATION,
            description="Rapid validation for development iterations",
            benchmark_config=BenchmarkConfig(
                benchmarks=["hellaswag"],
                batch_size=16,
                max_samples=100,  # Very limited
                cache_results=False,  # Skip caching for development
                parallel_benchmarks=False,
                timeout_minutes=5,
            ),
            report_config=ReportConfig(
                title="Quick Validation Report",
                include_sections=[
                    ReportSection.EXECUTIVE_SUMMARY,
                    ReportSection.PERFORMANCE_METRICS,
                ],
                format=ReportFormat.MARKDOWN,
                include_visualizations=False,
            ),
            recommended_for=["Development", "Debugging", "Quick checks"],
            tags=["quick", "development", "validation"],
        )

        # Comprehensive template
        self.templates["comprehensive"] = EvaluationTemplate(
            name="Comprehensive Evaluation",
            type=TemplateType.COMPREHENSIVE,
            description="Full evaluation suite with all available benchmarks",
            benchmark_config=BenchmarkConfig(
                benchmarks=[
                    "hellaswag",
                    "mmlu",
                    "arc_easy",
                    "arc_challenge",
                    "winogrande",
                    "truthfulqa",
                    "gsm8k",
                    "humaneval",
                ],
                custom_evaluations=["domain_specific", "safety", "bias"],
                batch_size=8,
                max_samples=None,
                cache_results=True,
                parallel_benchmarks=True,
                max_parallel_jobs=4,
                timeout_minutes=240,
                save_intermediate=True,
            ),
            report_config=ReportConfig(
                title="Comprehensive Evaluation Report",
                include_sections=list(ReportSection),
                format=ReportFormat.PDF,
                include_visualizations=True,
                include_raw_data=True,
                executive_summary_length=1000,
            ),
            hook_config=HookConfig(
                auto_evaluate=True,
                auto_generate_report=True,
                notification_on_completion=True,
                notification_on_error=True,
                notification_on_regression=True,
                model_registry_enabled=True,
            ),
            cost_config=CostAnalysisConfig(
                daily_requests=10000, avg_input_tokens=500, avg_output_tokens=200
            ),
            recommended_for=["Research", "Model comparison", "Publication"],
            tags=["comprehensive", "research", "full"],
        )

        # Domain-specific template (Medical)
        self.templates["medical_domain"] = EvaluationTemplate(
            name="Medical Domain Evaluation",
            type=TemplateType.DOMAIN_SPECIFIC,
            description="Evaluation for medical and healthcare applications",
            benchmark_config=BenchmarkConfig(
                benchmarks=["medqa", "pubmedqa", "medmcqa"],
                custom_evaluations=["medical_safety", "terminology_accuracy"],
                batch_size=4,  # Smaller batch for longer contexts
                max_samples=None,
                cache_results=True,
                parallel_benchmarks=False,  # Sequential for safety
                timeout_minutes=180,
            ),
            report_config=ReportConfig(
                title="Medical Domain Evaluation Report",
                include_sections=list(ReportSection),
                format=ReportFormat.PDF,
                include_visualizations=True,
                confidentiality_level="Highly Confidential",
            ),
            hook_config=HookConfig(
                auto_evaluate=True,
                auto_generate_report=True,
                auto_deploy=False,  # Never auto-deploy medical models
                notification_on_completion=True,
                notification_on_error=True,
                notification_on_regression=True,
            ),
            recommended_for=["Healthcare applications", "Medical AI", "Clinical decision support"],
            tags=["medical", "healthcare", "safety-critical"],
            metadata={
                "compliance": ["HIPAA", "FDA"],
                "required_accuracy": 0.95,
                "safety_threshold": 0.99,
            },
        )

        # Domain-specific template (Legal)
        self.templates["legal_domain"] = EvaluationTemplate(
            name="Legal Domain Evaluation",
            type=TemplateType.DOMAIN_SPECIFIC,
            description="Evaluation for legal and compliance applications",
            benchmark_config=BenchmarkConfig(
                benchmarks=["legalqa", "contract_understanding"],
                custom_evaluations=["legal_accuracy", "citation_verification"],
                batch_size=4,
                max_samples=None,
                cache_results=True,
                parallel_benchmarks=False,
                timeout_minutes=120,
            ),
            report_config=ReportConfig(
                title="Legal Domain Evaluation Report",
                include_sections=list(ReportSection),
                format=ReportFormat.PDF,
                include_visualizations=True,
                confidentiality_level="Confidential",
            ),
            recommended_for=["Legal research", "Contract analysis", "Compliance"],
            tags=["legal", "compliance", "high-accuracy"],
            metadata={"required_accuracy": 0.98, "citation_accuracy": 0.99},
        )

        # Domain-specific template (Code)
        self.templates["code_generation"] = EvaluationTemplate(
            name="Code Generation Evaluation",
            type=TemplateType.DOMAIN_SPECIFIC,
            description="Evaluation for code generation and programming tasks",
            benchmark_config=BenchmarkConfig(
                benchmarks=["humaneval", "mbpp", "apps"],
                custom_evaluations=["syntax_check", "test_pass_rate"],
                batch_size=1,  # One at a time for code execution
                max_samples=None,
                cache_results=True,
                parallel_benchmarks=False,  # Sequential for safety
                timeout_minutes=180,
            ),
            report_config=ReportConfig(
                title="Code Generation Evaluation Report",
                include_sections=[
                    ReportSection.EXECUTIVE_SUMMARY,
                    ReportSection.PERFORMANCE_METRICS,
                    ReportSection.DETAILED_BENCHMARKS,
                    ReportSection.VISUALIZATIONS,
                ],
                format=ReportFormat.HTML,
                include_visualizations=True,
            ),
            recommended_for=["Code assistants", "IDE plugins", "Programming education"],
            tags=["code", "programming", "development"],
            metadata={
                "languages": ["python", "javascript", "java"],
                "test_frameworks": ["pytest", "jest", "junit"],
            },
        )

        # A/B Testing template
        self.templates["ab_testing"] = EvaluationTemplate(
            name="A/B Testing",
            type=TemplateType.A_B_TESTING,
            description="Template for A/B testing in production",
            benchmark_config=BenchmarkConfig(
                benchmarks=["custom_production_metrics"],
                batch_size=16,
                max_samples=10000,  # Statistical significance
                cache_results=True,
                parallel_benchmarks=True,
                monitoring_enabled=True,
                monitoring_platforms=["prometheus", "grafana"],
            ),
            report_config=ReportConfig(
                title="A/B Test Results",
                include_sections=[
                    ReportSection.EXECUTIVE_SUMMARY,
                    ReportSection.PERFORMANCE_METRICS,
                    ReportSection.COST_ANALYSIS,
                    ReportSection.RECOMMENDATIONS,
                ],
                format=ReportFormat.HTML,
                include_visualizations=True,
            ),
            hook_config=HookConfig(
                auto_evaluate=True, auto_generate_report=True, notification_on_completion=True
            ),
            recommended_for=["Production testing", "Feature rollout", "Performance validation"],
            tags=["ab-testing", "production", "statistics"],
            metadata={
                "confidence_level": 0.95,
                "minimum_sample_size": 1000,
                "test_duration_days": 7,
            },
        )

        # Regression testing template
        self.templates["regression_testing"] = EvaluationTemplate(
            name="Regression Testing",
            type=TemplateType.REGRESSION_TESTING,
            description="Template for detecting performance regressions",
            benchmark_config=BenchmarkConfig(
                benchmarks=["hellaswag", "mmlu", "custom_regression_suite"],
                batch_size=8,
                max_samples=1000,
                cache_results=False,  # Always fresh results
                parallel_benchmarks=True,
                save_intermediate=True,
            ),
            report_config=ReportConfig(
                title="Regression Test Report",
                include_sections=[
                    ReportSection.EXECUTIVE_SUMMARY,
                    ReportSection.PERFORMANCE_METRICS,
                    ReportSection.DETAILED_BENCHMARKS,
                    ReportSection.RECOMMENDATIONS,
                ],
                format=ReportFormat.HTML,
                include_visualizations=True,
            ),
            hook_config=HookConfig(
                auto_evaluate=True,
                auto_generate_report=True,
                notification_on_regression=True,
                notification_channels=[NotificationType.EMAIL, NotificationType.SLACK],
            ),
            recommended_for=["CI/CD pipelines", "Model updates", "Version control"],
            tags=["regression", "testing", "ci-cd"],
            metadata={
                "regression_threshold": 5.0,  # 5% regression triggers alert
                "comparison_baseline": "previous_version",
            },
        )

    def _load_custom_templates(self):
        """Load custom templates from directory."""
        for template_file in self.custom_templates_dir.glob("*.json"):
            try:
                template = EvaluationTemplate.load(str(template_file))
                self.templates[template.name.lower().replace(" ", "_")] = template
                logger.info(f"Loaded custom template: {template.name}")
            except Exception as e:
                logger.error(f"Failed to load template {template_file}: {e}")

    def get_template(self, name: str) -> Optional[EvaluationTemplate]:
        """Get template by name.

        Args:
            name: Template name

        Returns:
            Template or None if not found
        """
        return self.templates.get(name)

    def list_templates(self) -> List[str]:
        """List available template names.

        Returns:
            List of template names
        """
        return list(self.templates.keys())

    def get_templates_by_type(self, template_type: TemplateType) -> List[EvaluationTemplate]:
        """Get templates by type.

        Args:
            template_type: Type of template

        Returns:
            List of matching templates
        """
        return [template for template in self.templates.values() if template.type == template_type]

    def get_templates_by_tag(self, tag: str) -> List[EvaluationTemplate]:
        """Get templates by tag.

        Args:
            tag: Tag to search for

        Returns:
            List of matching templates
        """
        return [template for template in self.templates.values() if tag in template.tags]

    def get_recommended_templates(self, use_case: str) -> List[EvaluationTemplate]:
        """Get recommended templates for a use case.

        Args:
            use_case: Use case description

        Returns:
            List of recommended templates
        """
        recommendations = []
        use_case_lower = use_case.lower()

        for template in self.templates.values():
            # Check if use case matches recommendations
            for recommended in template.recommended_for:
                if recommended.lower() in use_case_lower or use_case_lower in recommended.lower():
                    recommendations.append(template)
                    break

            # Check tags
            for tag in template.tags:
                if tag in use_case_lower:
                    if template not in recommendations:
                        recommendations.append(template)
                    break

        return recommendations

    def create_custom_template(
        self, name: str, description: str, base_template: Optional[str] = None, **kwargs
    ) -> EvaluationTemplate:
        """Create a custom template.

        Args:
            name: Template name
            description: Template description
            base_template: Optional base template to extend
            **kwargs: Additional configuration

        Returns:
            Created template
        """
        # Start with base template if provided
        if base_template and base_template in self.templates:
            base = self.templates[base_template]
            benchmark_config = base.benchmark_config
            report_config = base.report_config
            hook_config = base.hook_config
            cost_config = base.cost_config
        else:
            # Create default configs
            benchmark_config = BenchmarkConfig()
            report_config = ReportConfig()
            hook_config = HookConfig()
            cost_config = CostAnalysisConfig()

        # Override with provided kwargs
        if "benchmarks" in kwargs:
            benchmark_config.benchmarks = kwargs["benchmarks"]
        if "batch_size" in kwargs:
            benchmark_config.batch_size = kwargs["batch_size"]
        if "max_samples" in kwargs:
            benchmark_config.max_samples = kwargs["max_samples"]

        # Create template
        template = EvaluationTemplate(
            name=name,
            type=TemplateType.CUSTOM,
            description=description,
            benchmark_config=benchmark_config,
            report_config=report_config,
            hook_config=hook_config,
            cost_config=cost_config,
            custom_metrics=kwargs.get("custom_metrics", []),
            recommended_for=kwargs.get("recommended_for", []),
            tags=kwargs.get("tags", ["custom"]),
            metadata=kwargs.get("metadata", {}),
        )

        # Add to library
        template_key = name.lower().replace(" ", "_")
        self.templates[template_key] = template

        # Save if custom directory exists
        if self.custom_templates_dir:
            self.custom_templates_dir.mkdir(parents=True, exist_ok=True)
            template.save(str(self.custom_templates_dir / f"{template_key}.json"))

        return template

    def export_template(self, name: str, filepath: str):
        """Export a template to file.

        Args:
            name: Template name
            filepath: Export path
        """
        template = self.get_template(name)
        if template:
            template.save(filepath)
            logger.info(f"Exported template {name} to {filepath}")
        else:
            raise ValueError(f"Template {name} not found")

    def import_template(self, filepath: str) -> EvaluationTemplate:
        """Import a template from file.

        Args:
            filepath: Import path

        Returns:
            Imported template
        """
        template = EvaluationTemplate.load(filepath)
        template_key = template.name.lower().replace(" ", "_")
        self.templates[template_key] = template
        logger.info(f"Imported template: {template.name}")
        return template


# Convenience functions
_library = None


def get_library() -> TemplateLibrary:
    """Get the global template library instance.

    Returns:
        Template library
    """
    global _library
    if _library is None:
        _library = TemplateLibrary()
    return _library


def get_template(name: str) -> Optional[EvaluationTemplate]:
    """Get a template by name.

    Args:
        name: Template name

    Returns:
        Template or None
    """
    return get_library().get_template(name)


def list_templates() -> List[str]:
    """List available template names.

    Returns:
        List of template names
    """
    return get_library().list_templates()


# Example usage
if __name__ == "__main__":
    # Create library
    library = TemplateLibrary()

    # List available templates
    print("Available templates:")
    for name in library.list_templates():
        template = library.get_template(name)
        print(f"  - {name}: {template.description}")

    # Get accuracy-focused template
    accuracy_template = library.get_template("accuracy_focused")
    print(f"\nAccuracy template benchmarks: {accuracy_template.benchmark_config.benchmarks}")

    # Get templates by type
    domain_templates = library.get_templates_by_type(TemplateType.DOMAIN_SPECIFIC)
    print(f"\nDomain-specific templates: {[t.name for t in domain_templates]}")

    # Get recommended templates
    recommendations = library.get_recommended_templates("medical AI application")
    print(f"\nRecommended for medical AI: {[t.name for t in recommendations]}")

    # Create custom template
    custom = library.create_custom_template(
        name="My Custom Template",
        description="Custom evaluation for my specific use case",
        base_template="quick_validation",
        benchmarks=["hellaswag", "custom_benchmark"],
        batch_size=16,
        tags=["custom", "project-x"],
    )
    print(f"\nCreated custom template: {custom.name}")

    # Export template
    library.export_template("accuracy_focused", "accuracy_template.json")
    print("Exported accuracy_focused template")
