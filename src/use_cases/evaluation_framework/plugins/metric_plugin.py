"""
Custom Metric Plugin System

This module provides a flexible plugin system for defining custom evaluation metrics,
including metric registration, validation, aggregation strategies, and plugin discovery.

Example:
    # Define custom metric
    @register_metric("code_quality")
    class CodeQualityMetric(BaseMetric):
        def evaluate(self, model_output, ground_truth):
            # Custom evaluation logic
            return score

    # Use in evaluation
    runner = AutoBenchmarkRunner()
    runner.register_custom_metric("code_quality")
    results = runner.evaluate_model(model_path)
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import importlib
import inspect
import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    final,
    overload,
)

import numpy as np
from scipy import stats
from typing_extensions import override

from ....types.generics import GenericMetricCalculator
from ....types.protocols import DataType, MetricType, NumericType

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Aggregation strategies for metrics."""

    MEAN = "mean"
    MEDIAN = "median"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    HARMONIC_MEAN = "harmonic_mean"
    GEOMETRIC_MEAN = "geometric_mean"
    WEIGHTED_MEAN = "weighted_mean"
    PERCENTILE = "percentile"
    CUSTOM = "custom"


class MetricType(Enum):
    """Types of metrics."""

    ACCURACY = "accuracy"
    LATENCY = "latency"
    COST = "cost"
    QUALITY = "quality"
    SAFETY = "safety"
    FAIRNESS = "fairness"
    ROBUSTNESS = "robustness"
    CUSTOM = "custom"


@dataclass
class MetricConfig:
    """Configuration for a custom metric."""

    name: str
    type: MetricType
    description: str
    aggregation_strategy: AggregationStrategy = AggregationStrategy.MEAN
    higher_is_better: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    required_inputs: List[str] = field(default_factory=list)
    optional_inputs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Aggregation parameters
    weights: Optional[List[float]] = None
    percentile: Optional[float] = None
    custom_aggregator: Optional[Callable] = None


@dataclass
class MetricResult:
    """Result from metric evaluation."""

    metric_name: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    sample_size: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @final
    def to_dict(self) -> Dict[str | Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "confidence_interval": self.confidence_interval,
            "sample_size": self.sample_size,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseMetric(Generic[DataType, NumericType], ABC):
    """Base class for custom metrics."""

    def __init__(self, config: MetricConfig):
        """Initialize metric.

        Args:
            config: Metric configuration
        """
        self.config = config
        self.results_history = []

    @abstractmethod
    def evaluate(
        self, model_output: DataType, ground_truth: Optional[DataType] = None, **kwargs
    ) -> NumericType | MetricResult:
        """Evaluate the metric.

        Args:
            model_output: Model output to evaluate
            ground_truth: Optional ground truth
            **kwargs: Additional inputs

        Returns:
            Metric value or MetricResult
        """
        pass

    @final
    def validate_inputs(
        self, model_output: Any, ground_truth: Optional[Any] = None, **kwargs
    ) -> bool:
        """Validate inputs for the metric.

        Args:
            model_output: Model output
            ground_truth: Ground truth
            **kwargs: Additional inputs

        Returns:
            True if inputs are valid
        """
        # Check required inputs
        for required in self.config.required_inputs:
            if required not in kwargs:
                logger.error(f"Missing required input: {required}")
                return False

        return True

    @final
    def aggregate(self, values: List[float]) -> float:
        """Aggregate multiple metric values.

        Args:
            values: List of values to aggregate

        Returns:
            Aggregated value
        """
        if not values:
            return 0.0

        strategy = self.config.aggregation_strategy

        if strategy == AggregationStrategy.MEAN:
            return np.mean(values)
        elif strategy == AggregationStrategy.MEDIAN:
            return np.median(values)
        elif strategy == AggregationStrategy.SUM:
            return np.sum(values)
        elif strategy == AggregationStrategy.MIN:
            return np.min(values)
        elif strategy == AggregationStrategy.MAX:
            return np.max(values)
        elif strategy == AggregationStrategy.HARMONIC_MEAN:
            return stats.hmean(values)
        elif strategy == AggregationStrategy.GEOMETRIC_MEAN:
            return stats.gmean(values)
        elif strategy == AggregationStrategy.WEIGHTED_MEAN:
            if self.config.weights is None:
                return np.mean(values)
            return np.average(values[: len(self.config.weights)], weights=self.config.weights)
        elif strategy == AggregationStrategy.PERCENTILE:
            percentile = self.config.percentile or 50
            return np.percentile(values, percentile)
        elif strategy == AggregationStrategy.CUSTOM:
            if self.config.custom_aggregator:
                return self.config.custom_aggregator(values)
            return np.mean(values)
        else:
            return np.mean(values)

    @final
    def normalize(self, value: float) -> float:
        """Normalize metric value.

        Args:
            value: Raw value

        Returns:
            Normalized value
        """
        if self.config.min_value is not None and self.config.max_value is not None:
            # Normalize to [0, 1]
            normalized = (value - self.config.min_value) / (
                self.config.max_value - self.config.min_value
            )
            return np.clip(normalized, 0, 1)

        return value

    @final
    def compute_confidence_interval(
        self, values: List[float], confidence_level: float = 0.95
    ) -> Tuple[float | float]:
        """Compute confidence interval for metric values.

        Args:
            values: List of values
            confidence_level: Confidence level

        Returns:
            Confidence interval (lower, upper)
        """
        if len(values) < 2:
            return (values[0], values[0]) if values else (0, 0)

        mean = np.mean(values)
        std = np.std(values, ddof=1)
        n = len(values)

        # Standard error
        se = std / np.sqrt(n)

        # Critical value
        alpha = 1 - confidence_level
        z = stats.norm.ppf(1 - alpha / 2)

        # Confidence interval
        ci_lower = mean - z * se
        ci_upper = mean + z * se

        return (ci_lower, ci_upper)


# Built-in metric implementations


class AccuracyMetric(BaseMetric):
    """Accuracy metric for classification tasks."""

    def evaluate(self, model_output: List[Any], ground_truth: List[Any], **kwargs) -> MetricResult:
        """Evaluate accuracy.

        Args:
            model_output: Model predictions
            ground_truth: True labels

        Returns:
            MetricResult
        """
        if not self.validate_inputs(model_output, ground_truth, **kwargs):
            raise ValueError("Invalid inputs")

        if len(model_output) != len(ground_truth):
            raise ValueError("Output and ground truth must have same length")

        correct = sum(1 for pred, true in zip(model_output, ground_truth) if pred == true)
        accuracy = correct / len(model_output) if model_output else 0

        # Compute confidence interval
        # Using binomial confidence interval
        n = len(model_output)
        if n > 0:
            se = np.sqrt(accuracy * (1 - accuracy) / n)
            z = 1.96  # 95% confidence
            ci_lower = max(0, accuracy - z * se)
            ci_upper = min(1, accuracy + z * se)
            ci = (ci_lower, ci_upper)
        else:
            ci = (0, 0)

        return MetricResult(
            metric_name=self.config.name,
            value=accuracy,
            confidence_interval=ci,
            sample_size=n,
            metadata={"correct": correct, "total": n},
        )


class LatencyMetric(BaseMetric):
    """Latency metric for performance evaluation."""

    def evaluate(
        self, model_output: Any, ground_truth: Optional[Any] = None, **kwargs
    ) -> MetricResult:
        """Evaluate latency.

        Args:
            model_output: Not used
            **kwargs: Must contain 'latencies' list

        Returns:
            MetricResult
        """
        latencies = kwargs.get("latencies", [])

        if not latencies:
            raise ValueError("No latency data provided")

        # Calculate percentiles
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        # Primary metric is p50 by default
        value = p50

        # Confidence interval
        ci = self.compute_confidence_interval(latencies)

        return MetricResult(
            metric_name=self.config.name,
            value=value,
            confidence_interval=ci,
            sample_size=len(latencies),
            metadata={
                "p50": p50,
                "p95": p95,
                "p99": p99,
                "mean": np.mean(latencies),
                "std": np.std(latencies),
            },
        )


class F1ScoreMetric(BaseMetric):
    """F1 score metric for classification."""

    def evaluate(self, model_output: List[Any], ground_truth: List[Any], **kwargs) -> MetricResult:
        """Evaluate F1 score.

        Args:
            model_output: Predictions
            ground_truth: True labels

        Returns:
            MetricResult
        """
        # Calculate precision and recall
        true_positives = sum(
            1 for pred, true in zip(model_output, ground_truth) if pred == 1 and true == 1
        )
        false_positives = sum(
            1 for pred, true in zip(model_output, ground_truth) if pred == 1 and true == 0
        )
        false_negatives = sum(
            1 for pred, true in zip(model_output, ground_truth) if pred == 0 and true == 1
        )

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return MetricResult(
            metric_name=self.config.name,
            value=f1,
            sample_size=len(model_output),
            metadata={
                "precision": precision,
                "recall": recall,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            },
        )


class BLEUScoreMetric(BaseMetric):
    """BLEU score metric for text generation."""

    def evaluate(self, model_output: List[str], ground_truth: List[str], **kwargs) -> MetricResult:
        """Evaluate BLEU score.

        Args:
            model_output: Generated texts
            ground_truth: Reference texts

        Returns:
            MetricResult
        """
        # Simplified BLEU calculation
        # In practice, would use nltk.translate.bleu_score

        scores = []
        for generated, reference in zip(model_output, ground_truth):
            # Tokenize
            gen_tokens = generated.lower().split()
            ref_tokens = reference.lower().split()

            # Calculate n-gram overlap (simplified)
            if not gen_tokens or not ref_tokens:
                scores.append(0)
                continue

            # Unigram precision
            matches = sum(1 for token in gen_tokens if token in ref_tokens)
            precision = matches / len(gen_tokens) if gen_tokens else 0

            # Length penalty
            brevity_penalty = min(1, len(gen_tokens) / len(ref_tokens)) if ref_tokens else 0

            bleu = precision * brevity_penalty
            scores.append(bleu)

        avg_bleu = np.mean(scores) if scores else 0

        return MetricResult(
            metric_name=self.config.name,
            value=avg_bleu,
            confidence_interval=self.compute_confidence_interval(scores),
            sample_size=len(model_output),
            metadata={
                "scores": scores,
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0,
            },
        )


class CustomMetricRegistry:
    """Registry for custom metrics."""

    def __init__(self):
        """Initialize registry."""
        self.metrics: Dict[str, type] = {}
        self.instances: Dict[str, BaseMetric] = {}

        # Register built-in metrics
        self._register_builtin_metrics()

    def _register_builtin_metrics(self):
        """Register built-in metrics."""
        # Accuracy
        self.register_metric_class(
            "accuracy",
            AccuracyMetric,
            MetricConfig(
                name="accuracy",
                type=MetricType.ACCURACY,
                description="Classification accuracy",
                higher_is_better=True,
                min_value=0,
                max_value=1,
            ),
        )

        # Latency
        self.register_metric_class(
            "latency",
            LatencyMetric,
            MetricConfig(
                name="latency",
                type=MetricType.LATENCY,
                description="Response latency",
                higher_is_better=False,
                aggregation_strategy=AggregationStrategy.PERCENTILE,
                percentile=50,
            ),
        )

        # F1 Score
        self.register_metric_class(
            "f1_score",
            F1ScoreMetric,
            MetricConfig(
                name="f1_score",
                type=MetricType.ACCURACY,
                description="F1 score for binary classification",
                higher_is_better=True,
                min_value=0,
                max_value=1,
            ),
        )

        # BLEU Score
        self.register_metric_class(
            "bleu_score",
            BLEUScoreMetric,
            MetricConfig(
                name="bleu_score",
                type=MetricType.QUALITY,
                description="BLEU score for text generation",
                higher_is_better=True,
                min_value=0,
                max_value=1,
            ),
        )

    @overload
    def register_metric_class(self, name: str, metric_class: type) -> None: ...

    @overload
    def register_metric_class(
        self, name: str, metric_class: type, config: MetricConfig
    ) -> None: ...

    def register_metric_class(
        self, name: str, metric_class: type, config: Optional[MetricConfig] = None
    ) -> None:
        """Register a metric class.

        Args:
            name: Metric name
            metric_class: Metric class (must inherit from BaseMetric)
            config: Optional configuration
        """
        if not issubclass(metric_class, BaseMetric):
            raise ValueError(f"{metric_class} must inherit from BaseMetric")

        self.metrics[name] = metric_class

        # Create instance if config provided
        if config:
            self.instances[name] = metric_class(config)

        logger.info(f"Registered metric: {name}")

    def register_metric_function(self, name: str, eval_function: Callable, config: MetricConfig):
        """Register a metric function.

        Args:
            name: Metric name
            eval_function: Evaluation function
            config: Metric configuration
        """

        # Create a dynamic class
        class FunctionMetric(BaseMetric):
            def evaluate(self, model_output, ground_truth=None, **kwargs):
                result = eval_function(model_output, ground_truth, **kwargs)

                if isinstance(result, MetricResult):
                    return result
                else:
                    return MetricResult(metric_name=self.config.name, value=float(result))

        # Register the class
        self.register_metric_class(name, FunctionMetric, config)

    def get_metric(self, name: str) -> Optional[BaseMetric]:
        """Get metric instance.

        Args:
            name: Metric name

        Returns:
            Metric instance or None
        """
        return self.instances.get(name)

    def create_metric(
        self, name: str, config: Optional[MetricConfig] = None
    ) -> Optional[BaseMetric]:
        """Create metric instance.

        Args:
            name: Metric name
            config: Optional configuration

        Returns:
            Metric instance or None
        """
        metric_class = self.metrics.get(name)

        if not metric_class:
            logger.error(f"Metric {name} not found")
            return None

        if not config:
            config = MetricConfig(
                name=name, type=MetricType.CUSTOM, description=f"Custom metric: {name}"
            )

        instance = metric_class(config)
        self.instances[name] = instance

        return instance

    def list_metrics(self) -> List[str]:
        """List available metrics.

        Returns:
            List of metric names
        """
        return list(self.metrics.keys())

    @overload
    def evaluate_metric(
        self, name: str, model_output: Any, ground_truth: Optional[Any] = None, **kwargs
    ) -> Optional[MetricResult]: ...

    @overload
    def evaluate_metric(
        self,
        name: str,
        model_output: Any,
        return_raw: Literal[True],
        ground_truth: Optional[Any] = None,
        **kwargs,
    ) -> Optional[float]: ...

    def evaluate_metric(
        self,
        name: str,
        model_output: Any,
        ground_truth: Optional[Any] = None,
        return_raw: bool = False,
        **kwargs,
    ) -> Optional[MetricResult] | Optional[float]:
        """Evaluate a metric.

        Args:
            name: Metric name
            model_output: Model output
            ground_truth: Ground truth
            **kwargs: Additional arguments

        Returns:
            MetricResult or None
        """
        metric = self.get_metric(name)

        if not metric:
            metric = self.create_metric(name)

        if not metric:
            logger.error(f"Failed to create metric: {name}")
            return None

        try:
            result = metric.evaluate(model_output, ground_truth, **kwargs)

            # Handle raw return request
            if return_raw:
                if isinstance(result, MetricResult):
                    return result.value
                return float(result)

            # Return MetricResult
            if not isinstance(result, MetricResult):
                result = MetricResult(metric_name=name, value=float(result))

            return result

        except Exception as e:
            logger.error(f"Metric evaluation failed for {name}: {e}")
            return None


class MetricPluginLoader:
    """Loader for metric plugins from external files."""

    def __init__(self, plugin_dir: str = "./metric_plugins"):
        """Initialize plugin loader.

        Args:
            plugin_dir: Directory containing plugin files
        """
        self.plugin_dir = Path(plugin_dir)
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        self.registry = CustomMetricRegistry()

    def load_plugin(self, plugin_path: str):
        """Load a plugin from file.

        Args:
            plugin_path: Path to plugin file
        """
        plugin_path = Path(plugin_path)

        if not plugin_path.exists():
            raise FileNotFoundError(f"Plugin not found: {plugin_path}")

        # Load as Python module
        if plugin_path.suffix == ".py":
            self._load_python_plugin(plugin_path)
        # Load as JSON config
        elif plugin_path.suffix == ".json":
            self._load_json_plugin(plugin_path)
        else:
            raise ValueError(f"Unsupported plugin format: {plugin_path.suffix}")

    def _load_python_plugin(self, plugin_path: Path):
        """Load Python plugin.

        Args:
            plugin_path: Path to Python file
        """
        # Import module dynamically
        spec = importlib.util.spec_from_file_location(plugin_path.stem, plugin_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find metric classes
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, BaseMetric) and obj != BaseMetric:
                # Get config if available
                config = getattr(obj, "CONFIG", None) or MetricConfig(
                    name=name.lower(),
                    type=MetricType.CUSTOM,
                    description=obj.__doc__ or f"Custom metric: {name}",
                )

                self.registry.register_metric_class(name.lower(), obj, config)
                logger.info(f"Loaded plugin metric: {name}")

    def _load_json_plugin(self, plugin_path: Path):
        """Load JSON plugin configuration.

        Args:
            plugin_path: Path to JSON file
        """
        with open(plugin_path) as f:
            plugin_data = json.load(f)

        # Create metric from JSON config
        name = plugin_data.get("name")
        config = MetricConfig(**plugin_data.get("config", {}))

        # Check if it's a composite metric
        if "components" in plugin_data:
            self._create_composite_metric(name, config, plugin_data["components"])
        else:
            logger.warning(f"JSON plugin {name} has no implementation")

    def _create_composite_metric(
        self, name: str, config: MetricConfig, components: List[Dict[str, Any]]
    ):
        """Create composite metric from components.

        Args:
            name: Metric name
            config: Metric configuration
            components: List of component metrics
        """

        class CompositeMetric(BaseMetric):
            def __init__(self, config, components, registry):
                super().__init__(config)
                self.components = components
                self.registry = registry

            def evaluate(self, model_output, ground_truth=None, **kwargs):
                component_results = []

                for comp in self.components:
                    comp_name = comp["name"]
                    weight = comp.get("weight", 1.0)

                    result = self.registry.evaluate_metric(
                        comp_name, model_output, ground_truth, **kwargs
                    )

                    if result:
                        component_results.append((result.value, weight))

                if not component_results:
                    return MetricResult(metric_name=self.config.name, value=0)

                # Weighted average
                values, weights = zip(*component_results)
                weighted_avg = np.average(values, weights=weights)

                return MetricResult(
                    metric_name=self.config.name,
                    value=weighted_avg,
                    metadata={"components": component_results},
                )

        # Create instance
        metric = CompositeMetric(config, components, self.registry)
        self.registry.instances[name] = metric
        logger.info(f"Created composite metric: {name}")

    def discover_plugins(self):
        """Discover and load all plugins in plugin directory."""
        # Python plugins
        for plugin_file in self.plugin_dir.glob("*.py"):
            if plugin_file.name != "__init__.py":
                try:
                    self.load_plugin(str(plugin_file))
                except Exception as e:
                    logger.error(f"Failed to load plugin {plugin_file}: {e}")

        # JSON plugins
        for plugin_file in self.plugin_dir.glob("*.json"):
            try:
                self.load_plugin(str(plugin_file))
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_file}: {e}")


# Decorator for easy metric registration
_global_registry = CustomMetricRegistry()


def register_metric(name: str, config: Optional[MetricConfig] = None):
    """Decorator to register a metric class.

    Args:
        name: Metric name
        config: Optional configuration
    """

    def decorator(cls):
        _global_registry.register_metric_class(name, cls, config)
        return cls

    return decorator


def get_metric(name: str) -> Optional[BaseMetric]:
    """Get metric from global registry.

    Args:
        name: Metric name

    Returns:
        Metric instance or None
    """
    return _global_registry.get_metric(name)


# Example custom metric using decorator
@register_metric(
    "semantic_similarity",
    MetricConfig(
        name="semantic_similarity",
        type=MetricType.QUALITY,
        description="Semantic similarity between texts",
        higher_is_better=True,
        min_value=0,
        max_value=1,
    ),
)
class SemanticSimilarityMetric(BaseMetric):
    """Semantic similarity metric using embeddings."""

    def evaluate(self, model_output: List[str], ground_truth: List[str], **kwargs) -> MetricResult:
        """Evaluate semantic similarity.

        Args:
            model_output: Generated texts
            ground_truth: Reference texts

        Returns:
            MetricResult
        """
        # Simplified similarity calculation
        # In practice, would use sentence embeddings

        similarities = []
        for generated, reference in zip(model_output, ground_truth):
            # Simple word overlap as proxy for semantic similarity
            gen_words = set(generated.lower().split())
            ref_words = set(reference.lower().split())

            if not gen_words or not ref_words:
                similarities.append(0)
                continue

            intersection = gen_words & ref_words
            union = gen_words | ref_words

            jaccard = len(intersection) / len(union) if union else 0
            similarities.append(jaccard)

        avg_similarity = np.mean(similarities) if similarities else 0

        return MetricResult(
            metric_name=self.config.name,
            value=avg_similarity,
            confidence_interval=self.compute_confidence_interval(similarities),
            sample_size=len(model_output),
            metadata={
                "min": min(similarities) if similarities else 0,
                "max": max(similarities) if similarities else 0,
            },
        )


# Example usage
if __name__ == "__main__":
    # Create registry
    registry = CustomMetricRegistry()

    # List available metrics
    print("Available metrics:")
    for metric_name in registry.list_metrics():
        print(f"  - {metric_name}")

    # Evaluate accuracy metric
    predictions = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    ground_truth = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1]

    result = registry.evaluate_metric("accuracy", predictions, ground_truth)
    print(f"\nAccuracy: {result.value:.3f}")
    print(f"Confidence interval: {result.confidence_interval}")

    # Evaluate latency metric
    latencies = np.random.exponential(100, 1000).tolist()  # Simulated latencies

    result = registry.evaluate_metric("latency", None, latencies=latencies)
    print(f"\nLatency P50: {result.value:.2f}ms")
    print(f"P95: {result.metadata['p95']:.2f}ms")
    print(f"P99: {result.metadata['p99']:.2f}ms")

    # Register custom metric function
    def custom_metric(model_output, ground_truth, **kwargs):
        # Custom logic
        return len(model_output) / 100

    registry.register_metric_function(
        "custom_length_metric",
        custom_metric,
        MetricConfig(
            name="custom_length_metric",
            type=MetricType.CUSTOM,
            description="Custom metric based on output length",
        ),
    )

    # Use custom metric
    result = registry.evaluate_metric("custom_length_metric", ["a"] * 50, None)
    print(f"\nCustom metric: {result.value:.3f}")

    # Test semantic similarity
    generated = ["The cat sat on the mat", "It is raining today"]
    reference = ["A cat was sitting on a mat", "Today it's rainy"]

    result = registry.evaluate_metric("semantic_similarity", generated, reference)
    print(f"\nSemantic similarity: {result.value:.3f}")

    # Plugin loader example
    loader = MetricPluginLoader()
    print(f"\nPlugin directory: {loader.plugin_dir}")

    # Discover plugins (would load from plugin_dir)
    loader.discover_plugins()
    print("Plugins loaded successfully")
