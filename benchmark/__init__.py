"""
Benchmark module for LLM Lab

This module provides tools for running benchmarks across multiple models
with support for both sequential and parallel execution.
"""

from .multi_runner import MultiModelBenchmarkRunner, BenchmarkResult, ModelBenchmarkResult, ExecutionMode

__all__ = ['MultiModelBenchmarkRunner', 'BenchmarkResult', 'ModelBenchmarkResult', 'ExecutionMode']