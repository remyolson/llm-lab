"""
Comprehensive Before/After Evaluation Framework

This module provides automated benchmarking and evaluation capabilities for
comparing model performance before and after fine-tuning, including cost/benefit
analysis, side-by-side comparisons, and comprehensive reporting.
"""

from .benchmark_runner import (
    AutoBenchmarkRunner,
    BenchmarkConfig,
    BenchmarkResult,
    EvaluationState,
    ModelVersion,
)

__all__ = [
    "AutoBenchmarkRunner",
    "BenchmarkConfig",
    "BenchmarkResult",
    "EvaluationState",
    "ModelVersion",
]
