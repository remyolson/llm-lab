"""
Benchmark runners module for LLM Lab

This module provides tools for running benchmarks across multiple models
with support for both sequential and parallel execution.
"""

from .multi_runner import (
    BenchmarkResult,
    ExecutionMode,
    ModelBenchmarkResult,
    MultiModelBenchmarkRunner,
)

__all__ = ["BenchmarkResult", "ExecutionMode", "ModelBenchmarkResult", "MultiModelBenchmarkRunner"]
