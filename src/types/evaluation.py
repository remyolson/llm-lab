"""
Evaluation TypedDict definitions for LLM Lab

This module provides TypedDict classes for evaluation results, benchmark data,
and metric structures used throughout the evaluation framework.
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

from typing import Dict, List

from typing_extensions import NotRequired, TypedDict

from .core import ProviderInfo


class EvaluationResult(TypedDict):
    """Type-safe evaluation result structure.

    Standard structure for evaluation results from any evaluation method.
    """

    score: float
    method: str
    timestamp: str
    confidence: NotRequired[float]
    details: NotRequired[Dict[str, str | float | int | bool]]
    metadata: NotRequired[Dict[str, str | float | int | bool]]
    execution_time: NotRequired[float]
    error_message: NotRequired[str]


class MethodResult(TypedDict):
    """Type-safe individual method result structure.

    Results from a specific evaluation method within a larger evaluation.
    """

    method_name: str
    score: float
    individual_scores: NotRequired[List[float]]
    metadata: NotRequired[Dict[str, str | float | int | bool]]
    weight: NotRequired[float]
    normalized_score: NotRequired[float]
    error_count: NotRequired[int]


class CombinedResult(TypedDict):
    """Type-safe combined evaluation result structure.

    Results from combining multiple evaluation methods.
    """

    overall_score: float
    method_results: List[MethodResult]
    combination_method: str
    weights: NotRequired[Dict[str, float]]
    confidence_interval: NotRequired[List[float]]  # [lower, upper]
    sample_size: NotRequired[int]


class BenchmarkResult(TypedDict):
    """Type-safe benchmark result structure.

    Results from running standardized benchmarks.
    """

    benchmark_name: str
    task: str
    score: float
    metrics: Dict[str, float]
    model_info: ProviderInfo
    timestamp: str
    dataset_version: NotRequired[str]
    sample_count: NotRequired[int]
    execution_time: NotRequired[float]
    configuration: NotRequired[Dict[str, str | int | float | bool]]


class MetricResult(TypedDict):
    """Type-safe metric calculation result structure.

    Individual metric calculation results with metadata.
    """

    metric_name: str
    value: float
    unit: NotRequired[str]
    description: NotRequired[str]
    higher_is_better: NotRequired[bool]
    confidence: NotRequired[float]
    sample_size: NotRequired[int]
    calculation_method: NotRequired[str]


class FuzzyMatchResult(TypedDict):
    """Type-safe fuzzy matching result structure.

    Results from fuzzy string matching evaluations.
    """

    similarity_score: float
    match_type: str  # "exact", "partial", "fuzzy", "semantic"
    matched_text: NotRequired[str]
    confidence: NotRequired[float]
    algorithm: NotRequired[str]
    threshold_used: NotRequired[float]


class BenchmarkConfig(TypedDict):
    """Type-safe benchmark configuration structure.

    Configuration settings for running benchmarks.
    """

    name: str
    tasks: List[str]
    batch_size: int
    max_samples: NotRequired[int]
    device: NotRequired[str]
    use_cache: NotRequired[bool]
    seed: NotRequired[int]
    timeout: NotRequired[int]
    retry_failed: NotRequired[bool]


class EvaluationSummary(TypedDict):
    """Type-safe evaluation summary structure.

    High-level summary of evaluation runs and results.
    """

    total_evaluations: int
    successful_evaluations: int
    failed_evaluations: int
    average_score: float
    score_distribution: Dict[str, int]  # Score ranges to counts
    evaluation_methods: List[str]
    start_time: str
    end_time: str
    total_duration: float


class PluginResult(TypedDict):
    """Type-safe evaluation plugin result structure.

    Results from evaluation framework plugins.
    """

    plugin_name: str
    plugin_version: str
    result: EvaluationResult
    plugin_metadata: NotRequired[Dict[str, str | float | int | bool]]
    execution_context: NotRequired[Dict[str, str | float | int | bool]]
    dependencies: NotRequired[List[str]]


class ComparisonResult(TypedDict):
    """Type-safe model comparison result structure.

    Results from comparing multiple models on the same tasks.
    """

    models: List[str]
    task: str
    scores: Dict[str, float]  # model_name -> score
    winner: str
    confidence: NotRequired[float]
    statistical_significance: NotRequired[float]
    effect_size: NotRequired[float]
    metadata: NotRequired[Dict[str, str | float | int | bool]]
