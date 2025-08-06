"""Generate compliance documentation for regulatory frameworks."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import ComplianceReport, ModelCard

logger = logging.getLogger(__name__)


class ComplianceGenerator:
    """Generate compliance documentation."""

    def __init__(self):
        """Initialize the compliance generator."""
        self.frameworks = {
            "eu_ai_act": self._check_eu_ai_act,
            "iso_26000": self._check_iso_26000,
            "model_cards": self._check_model_cards_standard,
        }

    def generate_compliance_report(self, model_card: ModelCard, framework: str) -> ComplianceReport:
        """
        Generate compliance report for a framework.

        Args:
            model_card: The model card to assess
            framework: Regulatory framework name

        Returns:
            ComplianceReport object
        """
        if framework not in self.frameworks:
            raise ValueError(f"Unsupported framework: {framework}")

        checker = self.frameworks[framework]
        requirements_met, requirements_missing, score = checker(model_card)

        report = ComplianceReport(
            framework=framework,
            version="1.0",
            compliant=len(requirements_missing) == 0,
            score=score,
            requirements_met=requirements_met,
            requirements_missing=requirements_missing,
            recommendations=self._generate_recommendations(requirements_missing),
        )

        return report

    def _check_eu_ai_act(self, model_card: ModelCard) -> tuple:
        """Check EU AI Act compliance."""
        requirements_met = []
        requirements_missing = []

        # Check required elements
        if model_card.metadata.description:
            requirements_met.append("Model description provided")
        else:
            requirements_missing.append("Model description required")

        if model_card.ethical_considerations and model_card.ethical_considerations.intended_use:
            requirements_met.append("Intended use documented")
        else:
            requirements_missing.append("Intended use must be documented")

        if model_card.ethical_considerations and model_card.ethical_considerations.limitations:
            requirements_met.append("Limitations documented")
        else:
            requirements_missing.append("Model limitations must be documented")

        if model_card.performance:
            requirements_met.append("Performance metrics provided")
        else:
            requirements_missing.append("Performance metrics required")

        # Calculate score
        total = len(requirements_met) + len(requirements_missing)
        score = len(requirements_met) / total if total > 0 else 0

        return requirements_met, requirements_missing, score

    def _check_iso_26000(self, model_card: ModelCard) -> tuple:
        """Check ISO 26000 compliance."""
        requirements_met = []
        requirements_missing = []

        # Social responsibility checks
        if model_card.ethical_considerations:
            if model_card.ethical_considerations.potential_biases:
                requirements_met.append("Bias assessment documented")
            else:
                requirements_missing.append("Bias assessment required")

            if model_card.ethical_considerations.privacy_considerations:
                requirements_met.append("Privacy considerations documented")
            else:
                requirements_missing.append("Privacy considerations required")
        else:
            requirements_missing.append("Ethical considerations required")

        # Calculate score
        total = len(requirements_met) + len(requirements_missing)
        score = len(requirements_met) / total if total > 0 else 0

        return requirements_met, requirements_missing, score

    def _check_model_cards_standard(self, model_card: ModelCard) -> tuple:
        """Check Model Cards standard compliance."""
        requirements_met = []
        requirements_missing = []

        # Required sections
        if model_card.metadata:
            requirements_met.append("Model details provided")
        else:
            requirements_missing.append("Model details required")

        if model_card.training_config:
            requirements_met.append("Training data documented")
        else:
            requirements_missing.append("Training data documentation required")

        if model_card.performance:
            requirements_met.append("Evaluation data provided")
        else:
            requirements_missing.append("Evaluation data required")

        if model_card.ethical_considerations:
            requirements_met.append("Ethical considerations documented")
        else:
            requirements_missing.append("Ethical considerations required")

        # Calculate score
        total = len(requirements_met) + len(requirements_missing)
        score = len(requirements_met) / total if total > 0 else 0

        return requirements_met, requirements_missing, score

    def _generate_recommendations(self, requirements_missing: List[str]) -> List[str]:
        """Generate recommendations based on missing requirements."""
        recommendations = []

        for req in requirements_missing:
            if "description" in req.lower():
                recommendations.append(
                    "Add a detailed model description explaining its purpose and functionality"
                )
            elif "intended use" in req.lower():
                recommendations.append("Document the intended use cases and users for this model")
            elif "limitations" in req.lower():
                recommendations.append("Document known limitations and out-of-scope use cases")
            elif "bias" in req.lower():
                recommendations.append(
                    "Conduct and document bias assessment across different demographics"
                )
            elif "privacy" in req.lower():
                recommendations.append("Document privacy implications and data protection measures")
            elif "performance" in req.lower():
                recommendations.append(
                    "Provide comprehensive performance metrics and evaluation results"
                )

        return recommendations
