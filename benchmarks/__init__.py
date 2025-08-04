"""
Benchmarks module for LLM Lab

This module provides benchmark datasets, runners, and evaluation tools
for testing LLM performance across various tasks.
"""

# Import key components for easier access
from .runners import MultiModelBenchmarkRunner, BenchmarkResult, ModelBenchmarkResult, ExecutionMode
from .datasets import validate_dataset

__all__ = [
    'MultiModelBenchmarkRunner', 
    'BenchmarkResult', 
    'ModelBenchmarkResult', 
    'ExecutionMode',
    'validate_dataset'
]