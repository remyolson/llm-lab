"""A/B testing framework for alignment strategies."""

import hashlib
import logging
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VariantType(Enum):
    """Types of variants in experiments."""

    CONTROL = "control"
    TREATMENT = "treatment"


class MetricType(Enum):
    """Types of metrics to track."""

    BINARY = "binary"  # Success/failure
    CONTINUOUS = "continuous"  # Numeric value
    CATEGORICAL = "categorical"  # Category selection
    DURATION = "duration"  # Time-based


class ExperimentStatus(Enum):
    """Status of an experiment."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class Variant:
    """A variant in an A/B test."""

    variant_id: str
    name: str
    description: str
    variant_type: VariantType

    # Configuration for this variant
    config: Dict[str, Any] = field(default_factory=dict)

    # Allocation percentage (0-100)
    allocation_percentage: float = 50.0

    # Metrics collected
    exposures: int = 0
    conversions: int = 0
    metrics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    def record_exposure(self) -> None:
        """Record an exposure to this variant."""
        self.exposures += 1

    def record_conversion(self, metric_name: str, value: float = 1.0) -> None:
        """Record a conversion/metric for this variant."""
        self.conversions += 1
        self.metrics[metric_name].append(value)

    def get_conversion_rate(self) -> float:
        """Get conversion rate for this variant."""
        if self.exposures == 0:
            return 0.0
        return self.conversions / self.exposures

    def get_metric_stats(self, metric_name: str) -> Dict[str | float]:
        """Get statistics for a specific metric."""
        values = self.metrics.get(metric_name, [])

        if not values:
            return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        return {
            "count": len(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": min(values),
            "max": max(values),
        }


@dataclass
class Experiment:
    """An A/B testing experiment."""

    experiment_id: str
    name: str
    description: str

    # Experiment configuration
    variants: List[Variant] = field(default_factory=list)
    primary_metric: str = "conversion"
    secondary_metrics: List[str] = field(default_factory=list)

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # Status
    status: ExperimentStatus = ExperimentStatus.DRAFT

    # Sample size and duration
    min_sample_size: int = 1000
    max_duration_days: int = 30

    # Statistical configuration
    confidence_level: float = 0.95
    minimum_detectable_effect: float = 0.05

    # Results
    winner: Optional[str] = None

    def add_variant(self, variant: Variant) -> None:
        """Add a variant to the experiment."""
        self.variants.append(variant)

    def get_variant(self, variant_id: str) -> Optional[Variant]:
        """Get a variant by ID."""
        for variant in self.variants:
            if variant.variant_id == variant_id:
                return variant
        return None

    def start(self) -> None:
        """Start the experiment."""
        if self.status == ExperimentStatus.DRAFT:
            self.status = ExperimentStatus.RUNNING
            self.started_at = datetime.now()
            logger.info(f"Started experiment {self.experiment_id}")

    def pause(self) -> None:
        """Pause the experiment."""
        if self.status == ExperimentStatus.RUNNING:
            self.status = ExperimentStatus.PAUSED
            logger.info(f"Paused experiment {self.experiment_id}")

    def resume(self) -> None:
        """Resume the experiment."""
        if self.status == ExperimentStatus.PAUSED:
            self.status = ExperimentStatus.RUNNING
            logger.info(f"Resumed experiment {self.experiment_id}")

    def complete(self) -> None:
        """Complete the experiment."""
        if self.status in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            self.status = ExperimentStatus.COMPLETED
            self.ended_at = datetime.now()
            logger.info(f"Completed experiment {self.experiment_id}")

    def is_active(self) -> bool:
        """Check if experiment is active."""
        return self.status == ExperimentStatus.RUNNING

    def should_complete(self) -> bool:
        """Check if experiment should be completed."""
        if not self.is_active():
            return False

        # Check sample size
        total_exposures = sum(v.exposures for v in self.variants)
        if total_exposures >= self.min_sample_size:
            return True

        # Check duration
        if self.started_at:
            duration = datetime.now() - self.started_at
            if duration.days >= self.max_duration_days:
                return True

        return False


class AllocationStrategy(ABC):
    """Abstract base class for variant allocation strategies."""

    @abstractmethod
    def allocate(self, user_id: str, experiment: Experiment) -> Variant:
        """Allocate a user to a variant."""
        pass


class RandomAllocation(AllocationStrategy):
    """Random allocation based on percentages."""

    def allocate(self, user_id: str, experiment: Experiment) -> Variant:
        """Allocate user randomly based on variant percentages."""
        # Create cumulative distribution
        cumulative = []
        total = 0.0

        for variant in experiment.variants:
            total += variant.allocation_percentage
            cumulative.append((total, variant))

        # Generate random number
        rand = random.random() * total

        # Find variant
        for threshold, variant in cumulative:
            if rand <= threshold:
                return variant

        # Fallback to last variant
        return experiment.variants[-1]


class DeterministicAllocation(AllocationStrategy):
    """Deterministic allocation based on user ID hash."""

    def allocate(self, user_id: str, experiment: Experiment) -> Variant:
        """Allocate user deterministically based on ID hash."""
        # Hash user ID
        hash_value = int(
            hashlib.md5(f"{experiment.experiment_id}:{user_id}".encode()).hexdigest(), 16
        )

        # Map to percentage
        percentage = (hash_value % 10000) / 100.0

        # Find variant
        cumulative = 0.0
        for variant in experiment.variants:
            cumulative += variant.allocation_percentage
            if percentage <= cumulative:
                return variant

        # Fallback to last variant
        return experiment.variants[-1]


class ABTestingFramework:
    """Main A/B testing framework for alignment experiments."""

    def __init__(self, allocation_strategy: Optional[AllocationStrategy] = None):
        """
        Initialize A/B testing framework.

        Args:
            allocation_strategy: Strategy for allocating users to variants
        """
        self.allocation_strategy = allocation_strategy or DeterministicAllocation()
        self.experiments: Dict[str, Experiment] = {}
        self.user_assignments: Dict[str, Dict[str, str]] = defaultdict(dict)

    def create_experiment(self, experiment: Experiment) -> str:
        """Create a new experiment."""
        self.experiments[experiment.experiment_id] = experiment
        logger.info(f"Created experiment {experiment.experiment_id}: {experiment.name}")
        return experiment.experiment_id

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        return self.experiments.get(experiment_id)

    def get_active_experiments(self) -> List[Experiment]:
        """Get all active experiments."""
        return [exp for exp in self.experiments.values() if exp.is_active()]

    def allocate_user(self, user_id: str, experiment_id: str) -> Optional[Variant]:
        """Allocate a user to a variant in an experiment."""
        experiment = self.get_experiment(experiment_id)
        if not experiment or not experiment.is_active():
            return None

        # Check if user already allocated
        if experiment_id in self.user_assignments[user_id]:
            variant_id = self.user_assignments[user_id][experiment_id]
            return experiment.get_variant(variant_id)

        # Allocate to variant
        variant = self.allocation_strategy.allocate(user_id, experiment)

        # Record assignment
        self.user_assignments[user_id][experiment_id] = variant.variant_id
        variant.record_exposure()

        logger.debug(
            f"Allocated user {user_id} to variant {variant.variant_id} "
            f"in experiment {experiment_id}"
        )

        return variant

    def record_conversion(
        self, user_id: str, experiment_id: str, metric_name: str, value: float = 1.0
    ) -> bool:
        """Record a conversion/metric for a user."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False

        # Get user's variant
        if experiment_id not in self.user_assignments[user_id]:
            return False

        variant_id = self.user_assignments[user_id][experiment_id]
        variant = experiment.get_variant(variant_id)

        if variant:
            variant.record_conversion(metric_name, value)
            return True

        return False

    def get_experiment_results(self, experiment_id: str) -> Dict[str | Any]:
        """Get results for an experiment."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return {}

        results = {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "status": experiment.status.value,
            "duration_days": 0,
            "total_exposures": 0,
            "variants": [],
        }

        # Calculate duration
        if experiment.started_at:
            end_time = experiment.ended_at or datetime.now()
            results["duration_days"] = (end_time - experiment.started_at).days

        # Collect variant results
        for variant in experiment.variants:
            variant_results = {
                "variant_id": variant.variant_id,
                "name": variant.name,
                "type": variant.variant_type.value,
                "exposures": variant.exposures,
                "conversions": variant.conversions,
                "conversion_rate": variant.get_conversion_rate(),
                "metrics": {},
            }

            # Add metric statistics
            for metric_name in [experiment.primary_metric] + experiment.secondary_metrics:
                variant_results["metrics"][metric_name] = variant.get_metric_stats(metric_name)

            results["variants"].append(variant_results)
            results["total_exposures"] += variant.exposures

        # Perform statistical analysis
        if len(experiment.variants) == 2:
            results["statistical_significance"] = self._calculate_significance(experiment)

        return results

    def _calculate_significance(self, experiment: Experiment) -> Dict[str | Any]:
        """Calculate statistical significance for two-variant experiment."""
        if len(experiment.variants) != 2:
            return {}

        control = next(
            (v for v in experiment.variants if v.variant_type == VariantType.CONTROL), None
        )
        treatment = next(
            (v for v in experiment.variants if v.variant_type == VariantType.TREATMENT), None
        )

        if not control or not treatment:
            return {}

        # Simple z-test for conversion rates
        p1 = control.get_conversion_rate()
        p2 = treatment.get_conversion_rate()
        n1 = control.exposures
        n2 = treatment.exposures

        if n1 == 0 or n2 == 0:
            return {"significant": False, "p_value": 1.0}

        # Pooled proportion
        p_pool = (control.conversions + treatment.conversions) / (n1 + n2)

        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

        if se == 0:
            return {"significant": False, "p_value": 1.0}

        # Z-score
        z = (p2 - p1) / se

        # P-value (two-tailed)
        from scipy import stats

        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return {
            "significant": p_value < (1 - experiment.confidence_level),
            "p_value": p_value,
            "z_score": z,
            "lift": (p2 - p1) / p1 if p1 > 0 else 0,
        }

    def check_and_complete_experiments(self) -> List[str]:
        """Check all experiments and complete those that should end."""
        completed = []

        for experiment in self.get_active_experiments():
            if experiment.should_complete():
                experiment.complete()

                # Determine winner
                results = self.get_experiment_results(experiment.experiment_id)
                if "statistical_significance" in results:
                    sig = results["statistical_significance"]
                    if sig.get("significant") and sig.get("lift", 0) > 0:
                        # Treatment won
                        treatment = next(
                            (
                                v
                                for v in experiment.variants
                                if v.variant_type == VariantType.TREATMENT
                            ),
                            None,
                        )
                        if treatment:
                            experiment.winner = treatment.variant_id

                completed.append(experiment.experiment_id)
                logger.info(f"Completed experiment {experiment.experiment_id}")

        return completed


# Example experiment configurations
def create_intervention_experiment() -> Experiment:
    """Create an experiment for testing intervention strategies."""
    exp = Experiment(
        experiment_id="intervention_test_001",
        name="Prompt Intervention Effectiveness",
        description="Test if prompt interventions improve safety without harming quality",
    )

    # Control: No intervention
    control = Variant(
        variant_id="control",
        name="No Intervention",
        description="Standard processing without interventions",
        variant_type=VariantType.CONTROL,
        config={"interventions_enabled": False},
    )

    # Treatment: With interventions
    treatment = Variant(
        variant_id="treatment",
        name="With Interventions",
        description="Apply runtime interventions",
        variant_type=VariantType.TREATMENT,
        config={
            "interventions_enabled": True,
            "intervention_types": ["toxicity_filter", "bias_removal"],
        },
    )

    exp.add_variant(control)
    exp.add_variant(treatment)

    return exp


def create_safety_threshold_experiment() -> Experiment:
    """Create an experiment for testing safety thresholds."""
    exp = Experiment(
        experiment_id="safety_threshold_001",
        name="Optimal Safety Threshold",
        description="Find the optimal safety threshold balancing safety and usability",
    )

    # Different threshold variants
    thresholds = [0.7, 0.8, 0.9]

    for i, threshold in enumerate(thresholds):
        variant = Variant(
            variant_id=f"threshold_{threshold}",
            name=f"Threshold {threshold}",
            description=f"Safety threshold set to {threshold}",
            variant_type=VariantType.CONTROL if i == 0 else VariantType.TREATMENT,
            config={"safety_threshold": threshold},
            allocation_percentage=100.0 / len(thresholds),
        )
        exp.add_variant(variant)

    return exp
