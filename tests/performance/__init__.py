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

from .benchmark_suite import BenchmarkSuite, BenchmarkResult, BenchmarkMetrics
from .performance_tests import (
    ResponseTimeTest,
    ThroughputTest,
    TokenEfficiencyTest,
    ConcurrencyTest,
    MemoryUsageTest,
    StressTest
)
from .benchmark_config import BenchmarkConfig
from .performance_analyzer import PerformanceAnalyzer
from .benchmark_reporter import BenchmarkReporter

__all__ = [
    'BenchmarkSuite',
    'BenchmarkResult', 
    'BenchmarkMetrics',
    'ResponseTimeTest',
    'ThroughputTest',
    'TokenEfficiencyTest',
    'ConcurrencyTest',
    'MemoryUsageTest',
    'StressTest',
    'BenchmarkConfig',
    'PerformanceAnalyzer',
    'BenchmarkReporter'
]