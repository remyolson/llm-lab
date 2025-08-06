"""
A/B Testing Framework

This module provides a comprehensive A/B testing framework for production models,
including traffic splitting, statistical significance testing, confidence intervals,
power analysis, and early stopping capabilities.

Example:
    # Create A/B test
    test = ABTestRunner(
        name="Model v2 vs v1",
        control_model="model_v1",
        treatment_model="model_v2"
    )

    # Run production test
    results = test.run_production_test(
        duration_hours=24,
        traffic_split=50
    )

    # Get statistical analysis
    analysis = test.analyze_results()
"""

import asyncio
import hashlib
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from ..analysis.cost_benefit import CostBenefitAnalyzer

# Import evaluation components
from ..benchmark_runner import AutoBenchmarkRunner, ModelVersion
from ..reporting.report_generator import ReportContent, ReportFormat, ReportGenerator

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """A/B test status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED_EARLY = "stopped_early"
    FAILED = "failed"


class StatisticalTest(Enum):
    """Statistical test types."""

    T_TEST = "t_test"
    CHI_SQUARED = "chi_squared"
    MANN_WHITNEY = "mann_whitney"
    PROPORTIONS_Z_TEST = "proportions_z_test"


@dataclass
class TrafficSplit:
    """Traffic split configuration."""

    control_percentage: float
    treatment_percentage: float
    strategy: str = "random"  # random, hash, sequential
    hash_seed: Optional[str] = None

    def __post_init__(self):
        """Validate traffic split."""
        total = self.control_percentage + self.treatment_percentage
        if abs(total - 100.0) > 0.01:
            raise ValueError(f"Traffic split must sum to 100%, got {total}%")


@dataclass
class ABTestConfig:
    """A/B test configuration."""

    name: str
    control_model: str
    treatment_model: str
    traffic_split: TrafficSplit = field(default_factory=lambda: TrafficSplit(50, 50))

    # Test parameters
    minimum_sample_size: int = 1000
    maximum_duration_hours: float = 168  # 1 week
    confidence_level: float = 0.95
    power: float = 0.8
    minimum_detectable_effect: float = 0.01

    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_threshold: float = 0.999
    check_interval_minutes: int = 60

    # Metrics to track
    primary_metric: str = "latency_p50"
    secondary_metrics: List[str] = field(default_factory=lambda: ["accuracy", "cost_per_request"])
    custom_metrics: List[str] = field(default_factory=list)

    # Guardrails
    guardrail_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Example: {"error_rate": {"max": 0.01}, "latency_p99": {"max": 1000}}

    # Segmentation
    segments: List[str] = field(default_factory=list)
    # Example: ["user_type", "region", "device_type"]


@dataclass
class SampleData:
    """Sample data for a single observation."""

    timestamp: datetime
    request_id: str
    model_variant: str  # "control" or "treatment"
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    segment: Optional[str] = None


@dataclass
class ABTestResults:
    """A/B test results."""

    test_config: ABTestConfig
    status: TestStatus
    start_time: datetime
    end_time: Optional[datetime]

    # Sample counts
    control_samples: int
    treatment_samples: int

    # Primary metric results
    primary_metric_control: Dict[str, float]  # mean, std, etc.
    primary_metric_treatment: Dict[str, float]
    primary_metric_lift: float
    primary_metric_p_value: float
    primary_metric_confidence_interval: Tuple[float, float]

    # Secondary metrics
    secondary_metrics_results: Dict[str, Dict[str, Any]]

    # Statistical power
    observed_power: float

    # Segmentation results
    segment_results: Dict[str, Dict[str, Any]]

    # Guardrail violations
    guardrail_violations: List[Dict[str, Any]]

    # Decision
    winner: Optional[str] = None
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_config": {
                "name": self.test_config.name,
                "control_model": self.test_config.control_model,
                "treatment_model": self.test_config.treatment_model,
                "traffic_split": {
                    "control": self.test_config.traffic_split.control_percentage,
                    "treatment": self.test_config.traffic_split.treatment_percentage,
                },
            },
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "samples": {"control": self.control_samples, "treatment": self.treatment_samples},
            "primary_metric": {
                "control": self.primary_metric_control,
                "treatment": self.primary_metric_treatment,
                "lift": self.primary_metric_lift,
                "p_value": self.primary_metric_p_value,
                "confidence_interval": self.primary_metric_confidence_interval,
            },
            "secondary_metrics": self.secondary_metrics_results,
            "power": self.observed_power,
            "segments": self.segment_results,
            "guardrail_violations": self.guardrail_violations,
            "winner": self.winner,
            "recommendation": self.recommendation,
        }


class ABTestRunner:
    """A/B test runner for production models."""

    def __init__(
        self,
        config: ABTestConfig,
        data_dir: str = "./ab_test_data",
        benchmark_runner: Optional[AutoBenchmarkRunner] = None,
    ):
        """Initialize A/B test runner.

        Args:
            config: A/B test configuration
            data_dir: Directory for storing test data
            benchmark_runner: Optional benchmark runner for offline evaluation
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.benchmark_runner = benchmark_runner

        # Data storage
        self.samples: List[SampleData] = []
        self.control_metrics: defaultdict = defaultdict(list)
        self.treatment_metrics: defaultdict = defaultdict(list)

        # Test state
        self.status = TestStatus.PENDING
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Production routing
        self.routing_cache: Dict[str, str] = {}

    def calculate_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.8,
    ) -> int:
        """Calculate required sample size for test.

        Args:
            baseline_rate: Baseline conversion/success rate
            minimum_detectable_effect: Minimum effect to detect
            alpha: Significance level
            power: Statistical power

        Returns:
            Required sample size per variant
        """
        # Using formula for two-proportion z-test
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)

        # Pooled proportion
        p_pooled = (p1 + p2) / 2

        # Z-scores
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        # Sample size calculation
        numerator = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
        denominator = (p2 - p1) ** 2

        n = numerator / denominator

        return int(np.ceil(n))

    def route_request(self, request_id: str) -> str:
        """Route a request to control or treatment.

        Args:
            request_id: Unique request identifier

        Returns:
            "control" or "treatment"
        """
        # Check cache
        if request_id in self.routing_cache:
            return self.routing_cache[request_id]

        strategy = self.config.traffic_split.strategy

        if strategy == "random":
            # Random assignment
            rand = random.random() * 100
            variant = (
                "control" if rand < self.config.traffic_split.control_percentage else "treatment"
            )

        elif strategy == "hash":
            # Hash-based assignment (deterministic)
            seed = self.config.traffic_split.hash_seed or "default_seed"
            hash_input = f"{request_id}_{seed}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            percentage = hash_value % 100
            variant = (
                "control"
                if percentage < self.config.traffic_split.control_percentage
                else "treatment"
            )

        elif strategy == "sequential":
            # Sequential assignment
            total_samples = len(self.control_metrics[self.config.primary_metric]) + len(
                self.treatment_metrics[self.config.primary_metric]
            )

            control_ratio = self.config.traffic_split.control_percentage / 100
            expected_control = int(total_samples * control_ratio)
            actual_control = len(self.control_metrics[self.config.primary_metric])

            variant = "control" if actual_control < expected_control else "treatment"

        else:
            raise ValueError(f"Unknown routing strategy: {strategy}")

        # Cache the decision
        self.routing_cache[request_id] = variant

        return variant

    def record_sample(self, sample: SampleData):
        """Record a sample observation.

        Args:
            sample: Sample data
        """
        self.samples.append(sample)

        # Store metrics by variant
        metrics_dict = (
            self.control_metrics if sample.model_variant == "control" else self.treatment_metrics
        )

        for metric_name, value in sample.metrics.items():
            metrics_dict[metric_name].append(value)

        # Store by segment if applicable
        if sample.segment:
            segment_key = f"{metric_name}_{sample.segment}"
            metrics_dict[segment_key].append(value)

    def check_guardrails(self) -> List[Dict[str, Any]]:
        """Check for guardrail violations.

        Returns:
            List of violations
        """
        violations = []

        for metric_name, constraints in self.config.guardrail_metrics.items():
            # Check control
            if metric_name in self.control_metrics:
                control_values = self.control_metrics[metric_name]
                if control_values:
                    control_mean = np.mean(control_values)

                    if "max" in constraints and control_mean > constraints["max"]:
                        violations.append(
                            {
                                "variant": "control",
                                "metric": metric_name,
                                "value": control_mean,
                                "constraint": f"max={constraints['max']}",
                                "severity": "high",
                            }
                        )

                    if "min" in constraints and control_mean < constraints["min"]:
                        violations.append(
                            {
                                "variant": "control",
                                "metric": metric_name,
                                "value": control_mean,
                                "constraint": f"min={constraints['min']}",
                                "severity": "high",
                            }
                        )

            # Check treatment
            if metric_name in self.treatment_metrics:
                treatment_values = self.treatment_metrics[metric_name]
                if treatment_values:
                    treatment_mean = np.mean(treatment_values)

                    if "max" in constraints and treatment_mean > constraints["max"]:
                        violations.append(
                            {
                                "variant": "treatment",
                                "metric": metric_name,
                                "value": treatment_mean,
                                "constraint": f"max={constraints['max']}",
                                "severity": "high",
                            }
                        )

                    if "min" in constraints and treatment_mean < constraints["min"]:
                        violations.append(
                            {
                                "variant": "treatment",
                                "metric": metric_name,
                                "value": treatment_mean,
                                "constraint": f"min={constraints['min']}",
                                "severity": "high",
                            }
                        )

        return violations

    def perform_statistical_test(
        self,
        control_data: List[float],
        treatment_data: List[float],
        test_type: StatisticalTest = StatisticalTest.T_TEST,
    ) -> Tuple[float | float | Tuple[float | float]]:
        """Perform statistical test.

        Args:
            control_data: Control group data
            treatment_data: Treatment group data
            test_type: Type of statistical test

        Returns:
            Tuple of (test_statistic, p_value, confidence_interval)
        """
        if test_type == StatisticalTest.T_TEST:
            # Two-sample t-test
            statistic, p_value = stats.ttest_ind(control_data, treatment_data)

            # Calculate confidence interval for difference
            control_mean = np.mean(control_data)
            treatment_mean = np.mean(treatment_data)
            diff = treatment_mean - control_mean

            # Standard error
            control_std = np.std(control_data, ddof=1)
            treatment_std = np.std(treatment_data, ddof=1)
            n_control = len(control_data)
            n_treatment = len(treatment_data)

            se = np.sqrt(control_std**2 / n_control + treatment_std**2 / n_treatment)

            # Confidence interval
            alpha = 1 - self.config.confidence_level
            z = stats.norm.ppf(1 - alpha / 2)
            ci_lower = diff - z * se
            ci_upper = diff + z * se

            return statistic, p_value, (ci_lower, ci_upper)

        elif test_type == StatisticalTest.MANN_WHITNEY:
            # Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(
                control_data, treatment_data, alternative="two-sided"
            )

            # Bootstrap confidence interval
            n_bootstrap = 1000
            diffs = []

            for _ in range(n_bootstrap):
                control_sample = np.random.choice(
                    control_data, size=len(control_data), replace=True
                )
                treatment_sample = np.random.choice(
                    treatment_data, size=len(treatment_data), replace=True
                )
                diffs.append(np.median(treatment_sample) - np.median(control_sample))

            ci_lower = np.percentile(diffs, (1 - self.config.confidence_level) / 2 * 100)
            ci_upper = np.percentile(diffs, (1 + self.config.confidence_level) / 2 * 100)

            return statistic, p_value, (ci_lower, ci_upper)

        elif test_type == StatisticalTest.PROPORTIONS_Z_TEST:
            # Z-test for proportions
            # Assuming data contains 0s and 1s
            p1 = np.mean(control_data)
            p2 = np.mean(treatment_data)
            n1 = len(control_data)
            n2 = len(treatment_data)

            # Pooled proportion
            p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)

            # Standard error
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))

            # Z-statistic
            z_stat = (p2 - p1) / se if se > 0 else 0

            # P-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

            # Confidence interval
            diff = p2 - p1
            se_unpooled = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
            z_crit = stats.norm.ppf(1 - (1 - self.config.confidence_level) / 2)
            ci_lower = diff - z_crit * se_unpooled
            ci_upper = diff + z_crit * se_unpooled

            return z_stat, p_value, (ci_lower, ci_upper)

        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def check_early_stopping(self) -> bool:
        """Check if test should be stopped early.

        Returns:
            True if test should be stopped
        """
        if not self.config.enable_early_stopping:
            return False

        # Need minimum samples
        if len(self.control_metrics[self.config.primary_metric]) < self.config.minimum_sample_size:
            return False

        if (
            len(self.treatment_metrics[self.config.primary_metric])
            < self.config.minimum_sample_size
        ):
            return False

        # Perform statistical test
        control_data = self.control_metrics[self.config.primary_metric]
        treatment_data = self.treatment_metrics[self.config.primary_metric]

        _, p_value, _ = self.perform_statistical_test(control_data, treatment_data)

        # Check for overwhelming evidence
        if p_value < (1 - self.config.early_stopping_threshold):
            logger.info(f"Early stopping triggered: p_value={p_value:.4f}")
            return True

        return False

    def analyze_results(self) -> ABTestResults:
        """Analyze test results.

        Returns:
            ABTestResults object
        """
        # Get primary metric data
        control_data = self.control_metrics[self.config.primary_metric]
        treatment_data = self.treatment_metrics[self.config.primary_metric]

        # Calculate statistics for primary metric
        control_stats = {
            "mean": np.mean(control_data) if control_data else 0,
            "std": np.std(control_data) if control_data else 0,
            "median": np.median(control_data) if control_data else 0,
            "count": len(control_data),
        }

        treatment_stats = {
            "mean": np.mean(treatment_data) if treatment_data else 0,
            "std": np.std(treatment_data) if treatment_data else 0,
            "median": np.median(treatment_data) if treatment_data else 0,
            "count": len(treatment_data),
        }

        # Perform statistical test
        if control_data and treatment_data:
            _, p_value, confidence_interval = self.perform_statistical_test(
                control_data, treatment_data
            )

            # Calculate lift
            lift = (
                ((treatment_stats["mean"] - control_stats["mean"]) / control_stats["mean"] * 100)
                if control_stats["mean"] != 0
                else 0
            )

            # Calculate observed power
            effect_size = abs(treatment_stats["mean"] - control_stats["mean"]) / np.sqrt(
                (control_stats["std"] ** 2 + treatment_stats["std"] ** 2) / 2
            )

            observed_power = self._calculate_power(
                effect_size,
                len(control_data),
                len(treatment_data),
                alpha=1 - self.config.confidence_level,
            )
        else:
            p_value = 1.0
            confidence_interval = (0, 0)
            lift = 0
            observed_power = 0

        # Analyze secondary metrics
        secondary_results = {}
        for metric in self.config.secondary_metrics:
            if metric in self.control_metrics and metric in self.treatment_metrics:
                control = self.control_metrics[metric]
                treatment = self.treatment_metrics[metric]

                if control and treatment:
                    _, p_val, ci = self.perform_statistical_test(control, treatment)

                    secondary_results[metric] = {
                        "control_mean": np.mean(control),
                        "treatment_mean": np.mean(treatment),
                        "lift": ((np.mean(treatment) - np.mean(control)) / np.mean(control) * 100)
                        if np.mean(control) != 0
                        else 0,
                        "p_value": p_val,
                        "confidence_interval": ci,
                    }

        # Analyze segments
        segment_results = {}
        for segment in self.config.segments:
            segment_key = f"{self.config.primary_metric}_{segment}"

            if segment_key in self.control_metrics and segment_key in self.treatment_metrics:
                control = self.control_metrics[segment_key]
                treatment = self.treatment_metrics[segment_key]

                if control and treatment:
                    _, p_val, ci = self.perform_statistical_test(control, treatment)

                    segment_results[segment] = {
                        "control_mean": np.mean(control),
                        "treatment_mean": np.mean(treatment),
                        "lift": ((np.mean(treatment) - np.mean(control)) / np.mean(control) * 100)
                        if np.mean(control) != 0
                        else 0,
                        "p_value": p_val,
                        "confidence_interval": ci,
                    }

        # Check guardrails
        guardrail_violations = self.check_guardrails()

        # Determine winner
        winner = None
        recommendation = None

        if p_value < (1 - self.config.confidence_level):
            if lift > 0:
                winner = "treatment"
                recommendation = f"Deploy {self.config.treatment_model} (lift: {lift:.2f}%)"
            else:
                winner = "control"
                recommendation = f"Keep {self.config.control_model} (treatment showed {abs(lift):.2f}% degradation)"
        else:
            recommendation = (
                "No significant difference detected. Continue testing or keep current model."
            )

        # Handle guardrail violations
        if guardrail_violations:
            high_severity = [v for v in guardrail_violations if v["severity"] == "high"]
            if high_severity:
                winner = "control"
                recommendation = f"Keep {self.config.control_model} due to guardrail violations"

        return ABTestResults(
            test_config=self.config,
            status=self.status,
            start_time=self.start_time,
            end_time=self.end_time,
            control_samples=len(control_data),
            treatment_samples=len(treatment_data),
            primary_metric_control=control_stats,
            primary_metric_treatment=treatment_stats,
            primary_metric_lift=lift,
            primary_metric_p_value=p_value,
            primary_metric_confidence_interval=confidence_interval,
            secondary_metrics_results=secondary_results,
            observed_power=observed_power,
            segment_results=segment_results,
            guardrail_violations=guardrail_violations,
            winner=winner,
            recommendation=recommendation,
        )

    def _calculate_power(
        self, effect_size: float, n_control: int, n_treatment: int, alpha: float = 0.05
    ) -> float:
        """Calculate statistical power.

        Args:
            effect_size: Cohen's d effect size
            n_control: Control sample size
            n_treatment: Treatment sample size
            alpha: Significance level

        Returns:
            Statistical power
        """
        # Effective sample size
        n_eff = (n_control * n_treatment) / (n_control + n_treatment)

        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n_eff)

        # Critical value
        z_crit = stats.norm.ppf(1 - alpha / 2)

        # Power calculation
        power = 1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)

        return power

    async def run_production_test(
        self, duration_hours: Optional[float] = None, simulate: bool = False
    ) -> ABTestResults:
        """Run A/B test in production.

        Args:
            duration_hours: Test duration in hours
            simulate: Whether to simulate production traffic

        Returns:
            Test results
        """
        duration_hours = duration_hours or self.config.maximum_duration_hours

        self.status = TestStatus.RUNNING
        self.start_time = datetime.now()

        logger.info(f"Starting A/B test: {self.config.name}")
        logger.info(f"Duration: {duration_hours} hours")
        logger.info(
            f"Traffic split: Control={self.config.traffic_split.control_percentage}%, "
            f"Treatment={self.config.traffic_split.treatment_percentage}%"
        )

        try:
            if simulate:
                # Simulate production traffic
                await self._simulate_production_traffic(duration_hours)
            else:
                # Real production test
                await self._run_real_production_test(duration_hours)

            self.status = TestStatus.COMPLETED

        except Exception as e:
            logger.error(f"A/B test failed: {e}")
            self.status = TestStatus.FAILED
            raise

        finally:
            self.end_time = datetime.now()

            # Save results
            results = self.analyze_results()
            self._save_results(results)

        return results

    async def _simulate_production_traffic(self, duration_hours: float):
        """Simulate production traffic for testing.

        Args:
            duration_hours: Duration in hours
        """
        requests_per_minute = 100
        end_time = datetime.now() + timedelta(hours=duration_hours)

        # Simulated baseline metrics
        baseline_latency = 100  # ms
        baseline_accuracy = 0.85
        baseline_cost = 0.001  # per request

        # Simulated treatment effects
        treatment_latency_effect = -10  # 10ms improvement
        treatment_accuracy_effect = 0.02  # 2% improvement
        treatment_cost_effect = 0.0002  # 20% cost increase

        request_count = 0
        check_counter = 0

        while datetime.now() < end_time:
            # Generate batch of requests
            for _ in range(requests_per_minute):
                request_id = f"req_{request_count}"
                variant = self.route_request(request_id)

                # Generate metrics based on variant
                if variant == "control":
                    latency = np.random.normal(baseline_latency, 10)
                    accuracy = np.random.binomial(1, baseline_accuracy)
                    cost = np.random.normal(baseline_cost, 0.0001)
                else:
                    latency = np.random.normal(baseline_latency + treatment_latency_effect, 10)
                    accuracy = np.random.binomial(1, baseline_accuracy + treatment_accuracy_effect)
                    cost = np.random.normal(baseline_cost + treatment_cost_effect, 0.0001)

                # Record sample
                sample = SampleData(
                    timestamp=datetime.now(),
                    request_id=request_id,
                    model_variant=variant,
                    metrics={
                        "latency_p50": latency,
                        "accuracy": accuracy,
                        "cost_per_request": cost,
                    },
                    metadata={"simulated": True},
                )

                self.record_sample(sample)
                request_count += 1

            # Check for early stopping
            check_counter += 1
            if check_counter >= self.config.check_interval_minutes:
                check_counter = 0

                if self.check_early_stopping():
                    logger.info("Early stopping condition met")
                    self.status = TestStatus.STOPPED_EARLY
                    break

                # Check guardrails
                violations = self.check_guardrails()
                if violations:
                    high_severity = [v for v in violations if v["severity"] == "high"]
                    if high_severity:
                        logger.warning(f"High severity guardrail violations: {high_severity}")
                        self.status = TestStatus.STOPPED_EARLY
                        break

            # Sleep for a minute
            await asyncio.sleep(60)

    async def _run_real_production_test(self, duration_hours: float):
        """Run real production A/B test.

        Args:
            duration_hours: Duration in hours
        """
        # This would integrate with actual production systems
        # For now, this is a placeholder
        raise NotImplementedError(
            "Real production testing requires integration with production systems"
        )

    def _save_results(self, results: ABTestResults):
        """Save test results to disk.

        Args:
            results: Test results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.data_dir / f"ab_test_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(results.to_dict(), f, indent=2)

        logger.info(f"Saved A/B test results to {filename}")

    def generate_report(self, results: ABTestResults) -> str:
        """Generate A/B test report.

        Args:
            results: Test results

        Returns:
            Report content
        """
        # Create report content
        content = ReportContent(
            title=f"A/B Test Report: {self.config.name}",
            model_name=self.config.treatment_model,
            timestamp=datetime.now(),
            metadata={
                "test_type": "A/B Test",
                "control_model": self.config.control_model,
                "treatment_model": self.config.treatment_model,
                "duration_hours": (results.end_time - results.start_time).total_seconds() / 3600
                if results.end_time
                else 0,
            },
        )

        # Add test results
        content.benchmark_results = {
            "primary_metric": {
                "name": self.config.primary_metric,
                "control": results.primary_metric_control,
                "treatment": results.primary_metric_treatment,
                "lift": results.primary_metric_lift,
                "p_value": results.primary_metric_p_value,
                "confidence_interval": results.primary_metric_confidence_interval,
            },
            "secondary_metrics": results.secondary_metrics_results,
            "segments": results.segment_results,
        }

        # Add statistical analysis
        content.statistical_analysis = {
            "observed_power": results.observed_power,
            "confidence_level": self.config.confidence_level,
            "minimum_detectable_effect": self.config.minimum_detectable_effect,
            "sample_sizes": {
                "control": results.control_samples,
                "treatment": results.treatment_samples,
            },
        }

        # Add recommendations
        content.recommendations = [
            results.recommendation,
            f"Winner: {results.winner}" if results.winner else "No clear winner",
        ]

        # Add guardrail violations
        if results.guardrail_violations:
            content.recommendations.append(
                f"⚠️ Guardrail violations detected: {len(results.guardrail_violations)}"
            )

        # Generate report
        report_generator = ReportGenerator()
        report = report_generator.generate_report(content, format=ReportFormat.MARKDOWN)

        return report


class MultiVariateTestRunner:
    """Multi-variate test runner for testing multiple variants."""

    def __init__(
        self,
        name: str,
        control_model: str,
        treatment_models: List[str],
        data_dir: str = "./multivariate_test_data",
    ):
        """Initialize multi-variate test runner.

        Args:
            name: Test name
            control_model: Control model path
            treatment_models: List of treatment model paths
            data_dir: Data directory
        """
        self.name = name
        self.control_model = control_model
        self.treatment_models = treatment_models
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Create pairwise A/B tests
        self.ab_tests = []
        for treatment in treatment_models:
            config = ABTestConfig(
                name=f"{name}_{Path(treatment).stem}",
                control_model=control_model,
                treatment_model=treatment,
            )
            self.ab_tests.append(ABTestRunner(config, data_dir=str(self.data_dir)))

    async def run_tests(
        self, duration_hours: float = 24, simulate: bool = True
    ) -> Dict[str, ABTestResults]:
        """Run all tests.

        Args:
            duration_hours: Test duration
            simulate: Whether to simulate

        Returns:
            Dictionary of test results
        """
        results = {}

        # Run tests in parallel
        tasks = []
        for test in self.ab_tests:
            task = test.run_production_test(duration_hours, simulate)
            tasks.append(task)

        test_results = await asyncio.gather(*tasks)

        for test, result in zip(self.ab_tests, test_results):
            results[test.config.name] = result

        return results

    def select_winner(self, results: Dict[str, ABTestResults]) -> str:
        """Select overall winner from multiple tests.

        Args:
            results: Test results

        Returns:
            Winner model path
        """
        # Score each model
        scores = {self.control_model: 0}
        for treatment in self.treatment_models:
            scores[treatment] = 0

        # Award points based on wins
        for test_name, result in results.items():
            if result.winner == "treatment":
                # Find which treatment won
                for test in self.ab_tests:
                    if test.config.name == test_name:
                        scores[test.config.treatment_model] += 1
                        break
            elif result.winner == "control":
                scores[self.control_model] += 1

        # Return model with highest score
        winner = max(scores, key=scores.get)
        logger.info(f"Overall winner: {winner} with score {scores[winner]}")

        return winner


# Example usage
if __name__ == "__main__":
    # Create A/B test configuration
    config = ABTestConfig(
        name="GPT-2 Fine-tuned vs Base",
        control_model="gpt2",
        treatment_model="./fine_tuned/gpt2-custom",
        traffic_split=TrafficSplit(50, 50, strategy="hash"),
        minimum_sample_size=1000,
        confidence_level=0.95,
        primary_metric="latency_p50",
        secondary_metrics=["accuracy", "cost_per_request"],
        guardrail_metrics={"error_rate": {"max": 0.01}, "latency_p99": {"max": 1000}},
    )

    # Create test runner
    runner = ABTestRunner(config)

    # Calculate required sample size
    sample_size = runner.calculate_sample_size(baseline_rate=0.85, minimum_detectable_effect=0.02)
    print(f"Required sample size: {sample_size} per variant")

    # Run simulated test
    async def run_test():
        results = await runner.run_production_test(
            duration_hours=0.1,  # Short test for demo
            simulate=True,
        )

        print("\nA/B Test Results:")
        print(f"Status: {results.status.value}")
        print(f"Samples: Control={results.control_samples}, Treatment={results.treatment_samples}")
        print(f"Primary metric lift: {results.primary_metric_lift:.2f}%")
        print(f"P-value: {results.primary_metric_p_value:.4f}")
        print(f"Confidence interval: {results.primary_metric_confidence_interval}")
        print(f"Winner: {results.winner}")
        print(f"Recommendation: {results.recommendation}")

        # Generate report
        report = runner.generate_report(results)
        print("\nGenerated report preview:")
        print(report[:500])

    # Run the test
    asyncio.run(run_test())
