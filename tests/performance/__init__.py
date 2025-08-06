"""
Performance benchmarking suite for LLM providers

This module provides comprehensive performance testing and benchmarking tools
for measuring and comparing LLM provider performance across various metrics.

Key Features:
- Response time measurement and analysis
- Throughput testing (requests per second)
- Token efficiency analysis
- Memory usage monitoring
- Concurrent request performance
- Comparative benchmarking across providers
- Statistical analysis and reporting

Usage:
    from tests.performance import PerformanceBenchmark, BenchmarkSuite

    # Create benchmark suite
    suite = BenchmarkSuite()

    # Add providers
    suite.add_provider(OpenAIProvider("gpt-3.5-turbo"))
    suite.add_provider(AnthropicProvider("claude-3-haiku-20240307"))

    # Run benchmarks
    results = suite.run_all_benchmarks()

    # Generate report
    suite.generate_report(results)
"""

from .benchmark_config import BenchmarkConfig
from .benchmark_reporter import BenchmarkReporter
from .benchmark_suite import BenchmarkMetrics, BenchmarkResult, BenchmarkSuite
from .performance_analyzer import PerformanceAnalyzer
from .performance_tests import (
    ConcurrencyTest,
    MemoryUsageTest,
    ResponseTimeTest,
    StressTest,
    ThroughputTest,
    TokenEfficiencyTest,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkMetrics",
    "BenchmarkReporter",
    "BenchmarkResult",
    "BenchmarkSuite",
    "ConcurrencyTest",
    "MemoryUsageTest",
    "PerformanceAnalyzer",
    "ResponseTimeTest",
    "StressTest",
    "ThroughputTest",
    "TokenEfficiencyTest",
]
