"""
Benchmark Creation Tool - Use Case 13

This module provides tools for creating custom benchmarks and evaluation datasets
for Large Language Models, with support for various task types and evaluation metrics.

Key Components:
- Benchmark dataset generation and validation
- Template-based test case creation
- Multiple evaluation metric support
- Storage backends for benchmark persistence
- Integration with existing evaluation frameworks

Usage:
    from src.use_cases.benchmark_creation import BenchmarkBuilder

    builder = BenchmarkBuilder()
    benchmark = await builder.create_benchmark(config)
"""

__version__ = "1.0.0"
__author__ = "LLM Lab Team"

# Re-export main components for easy access
try:
    from .src.benchmark_builder.generators.factory import GeneratorFactory
    from .src.benchmark_builder.generators.text_generator import TextGenerator
    from .src.benchmark_builder.validators.base import BaseValidator

    __all__ = ["TextGenerator", "GeneratorFactory", "BaseValidator"]
except ImportError:
    # Handle graceful import failures for development
    __all__ = []
