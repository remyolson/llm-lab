"""
Custom Metric Plugin System

This module provides a flexible plugin system for custom evaluation metrics.
"""

from .metric_plugin import (
    AggregationStrategy,
    BaseMetric,
    CustomMetricRegistry,
    MetricConfig,
    MetricPluginLoader,
    MetricResult,
    MetricType,
    get_metric,
    register_metric,
)

__all__ = [
    "AggregationStrategy",
    "BaseMetric",
    "CustomMetricRegistry",
    "MetricConfig",
    "MetricPluginLoader",
    "MetricResult",
    "MetricType",
    "get_metric",
    "register_metric",
]
