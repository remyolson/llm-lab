"""
Benchmarks module for LLM Lab

This module provides benchmark datasets, runners, and evaluation tools
for testing LLM performance across various tasks.
"""

# Import key components for easier access
from .runners import BenchmarkResult, ExecutionMode, ModelBenchmarkResult, MultiModelBenchmarkRunner

__all__ = ["BenchmarkResult", "ExecutionMode", "ModelBenchmarkResult", "MultiModelBenchmarkRunner"]
