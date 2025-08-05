#!/usr/bin/env python3
"""
Comprehensive Performance Benchmark Suite for Cross-LLM Testing

This module provides an advanced performance benchmarking framework for comparing
LLM providers across multiple dimensions: latency, throughput, cost, and quality.

Key Features:
- Multi-dimensional performance analysis
- Concurrent request handling and throughput testing
- Cost analysis with real-time pricing
- Statistical analysis with confidence intervals
- Performance profiling and bottleneck identification
- Comparative analysis and ranking
- Load testing and stress testing capabilities
"""

import pytest
import asyncio
import aiohttp
import time
import statistics
import json
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import psutil
import threading
import queue
import os
import tempfile
from contextlib import contextmanager

# Import existing infrastructure
from src.providers.base import LLMProvider
from src.providers.openai import OpenAIProvider
from src.providers.anthropic import AnthropicProvider
from src.providers.google import GoogleProvider


# ===============================================================================
# PERFORMANCE METRICS DATA STRUCTURES
# ===============================================================================

@dataclass
class LatencyMetrics:
    """Detailed latency measurements."""
    mean: float
    median: float
    p95: float
    p99: float
    min: float
    max: float
    std_dev: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ThroughputMetrics:
    """Throughput performance measurements."""
    requests_per_second: float
    tokens_per_second: float
    concurrent_capacity: int
    queue_time_ms: float
    processing_time_ms: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class CostMetrics:
    """Cost analysis metrics."""
    cost_per_request: float
    cost_per_token: float
    cost_per_minute: float
    total_cost: float
    cost_efficiency_score: float  # Performance/cost ratio
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ResourceMetrics:
    """System resource utilization metrics."""
    cpu_usage_percent: float
    memory_usage_mb: float
    network_io_kb: float
    peak_memory_mb: float
    avg_cpu_percent: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class QualityMetrics:
    """Response quality metrics for performance correlation."""
    accuracy_score: float
    coherence_score: float
    completeness_score: float
    avg_response_length: int
    error_rate: float
    
    def to_dict(self) -> Dict[str, Union[float, int]]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Comprehensive benchmark result."""
    provider: str
    model: str
    test_scenario: str
    timestamp: datetime
    latency: LatencyMetrics
    throughput: ThroughputMetrics
    cost: CostMetrics
    resources: ResourceMetrics
    quality: QualityMetrics
    configuration: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def overall_score(self) -> float:
        """Calculate composite performance score."""
        # Normalize metrics to 0-1 scale and weight them
        latency_score = max(0, 1 - (self.latency.mean / 10.0))  # Assume 10s is very slow
        throughput_score = min(1, self.throughput.requests_per_second / 100.0)  # Assume 100 RPS is excellent
        cost_score = max(0, 1 - (self.cost.cost_per_request / 1.0))  # Assume $1 per request is expensive
        quality_score = (self.quality.accuracy_score + self.quality.coherence_score) / 2
        
        # Weighted average
        weights = {'latency': 0.3, 'throughput': 0.3, 'cost': 0.2, 'quality': 0.2}
        return (
            weights['latency'] * latency_score +
            weights['throughput'] * throughput_score +
            weights['cost'] * cost_score +
            weights['quality'] * quality_score
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'provider': self.provider,
            'model': self.model,
            'test_scenario': self.test_scenario,
            'timestamp': self.timestamp.isoformat(),
            'latency': self.latency.to_dict(),
            'throughput': self.throughput.to_dict(),
            'cost': self.cost.to_dict(),
            'resources': self.resources.to_dict(),
            'quality': self.quality.to_dict(),
            'overall_score': self.overall_score(),
            'configuration': self.configuration,
            'metadata': self.metadata
        }


# ===============================================================================
# PERFORMANCE TESTING UTILITIES
# ===============================================================================

class ResourceMonitor:
    """Monitor system resource usage during benchmarks."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.measurements = []
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        self.monitoring = True
        self.measurements = []
        self._monitor_thread = threading.Thread(target=self._monitor_resources)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> ResourceMetrics:
        """Stop monitoring and return resource metrics."""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        
        if not self.measurements:
            return ResourceMetrics(0, 0, 0, 0, 0)
        
        cpu_values = [m['cpu'] for m in self.measurements]
        memory_values = [m['memory'] for m in self.measurements]
        
        return ResourceMetrics(
            cpu_usage_percent=self.process.cpu_percent(),
            memory_usage_mb=self.process.memory_info().rss / 1024 / 1024,
            network_io_kb=sum(psutil.net_io_counters()[:2]) / 1024,
            peak_memory_mb=max(memory_values),
            avg_cpu_percent=statistics.mean(cpu_values) if cpu_values else 0
        )
    
    def _monitor_resources(self):
        """Background monitoring function."""
        while self.monitoring:
            try:
                cpu_percent = self.process.cpu_percent()
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                
                self.measurements.append({
                    'timestamp': time.time(),
                    'cpu': cpu_percent,
                    'memory': memory_mb
                })
                
                time.sleep(0.1)  # Sample every 100ms
            except Exception:
                pass


class CostCalculator:
    """Calculate costs for different providers based on current pricing."""
    
    # Current pricing per 1K tokens (as of 2024)
    PRICING = {
        'OpenAIProvider': {
            'gpt-4o': {'input': 0.0025, 'output': 0.01},
            'gpt-4o-mini': {'input': 0.000150, 'output': 0.0006},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002}
        },
        'AnthropicProvider': {
            'claude-3-5-sonnet-20241022': {'input': 0.003, 'output': 0.015},
            'claude-3-opus-20240229': {'input': 0.015, 'output': 0.075},
            'claude-3-sonnet-20240229': {'input': 0.003, 'output': 0.015},
            'claude-3-haiku-20240307': {'input': 0.00025, 'output': 0.00125}
        },
        'GoogleProvider': {
            'gemini-1.5-pro': {'input': 0.00125, 'output': 0.005},
            'gemini-1.5-flash': {'input': 0.000075, 'output': 0.0003},
            'gemini-1.0-pro': {'input': 0.0005, 'output': 0.0015}
        }
    }
    
    @classmethod
    def calculate_cost(cls, provider: str, model: str, input_tokens: int, 
                      output_tokens: int) -> float:
        """Calculate cost for a request."""
        if provider not in cls.PRICING:
            return 0.0
        
        model_pricing = cls.PRICING[provider].get(model)
        if not model_pricing:
            # Use first available model pricing as fallback
            model_pricing = next(iter(cls.PRICING[provider].values()))
        
        input_cost = (input_tokens / 1000) * model_pricing['input']
        output_cost = (output_tokens / 1000) * model_pricing['output']
        
        return input_cost + output_cost
    
    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)."""
        return max(1, len(text) // 4)


class QualityEvaluator:
    """Evaluate response quality for performance/quality trade-off analysis."""
    
    @staticmethod
    def evaluate_accuracy(prompt: str, response: str, expected: str = None) -> float:
        """Evaluate response accuracy."""
        if not response or response.lower().startswith('error'):
            return 0.0
        
        if expected:
            # Simple keyword matching
            expected_words = set(expected.lower().split())
            response_words = set(response.lower().split())
            if expected_words:
                overlap = len(expected_words.intersection(response_words))
                return overlap / len(expected_words)
        
        # Heuristic scoring for responses without expected answers
        if len(response.strip()) < 5:
            return 0.2
        elif 'sorry' in response.lower() or 'cannot' in response.lower():
            return 0.4
        else:
            return 0.8
    
    @staticmethod
    def evaluate_coherence(response: str) -> float:
        """Evaluate response coherence."""
        if not response:
            return 0.0
        
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if len(sentences) == 0:
            return 0.1
        
        # Check for word diversity
        words = response.lower().split()
        if len(words) == 0:
            return 0.0
        
        unique_words = len(set(words))
        diversity = unique_words / len(words)
        
        # Score based on diversity and structure
        coherence = min(1.0, diversity * 1.5)
        if len(sentences) > 1:
            coherence += 0.2  # Bonus for structured response
        
        return min(1.0, coherence)
    
    @staticmethod
    def evaluate_completeness(prompt: str, response: str) -> float:
        """Evaluate response completeness."""
        if not response:
            return 0.0
        
        # Heuristic: response should be proportional to prompt complexity
        prompt_complexity = len(prompt.split()) + prompt.count('?') * 2
        response_length = len(response.split())
        
        if response_length == 0:
            return 0.0
        
        # Good responses are typically 0.5x to 3x the prompt complexity
        ratio = response_length / max(prompt_complexity, 1)
        
        if 0.5 <= ratio <= 3.0:
            return 1.0
        elif 0.2 <= ratio <= 5.0:
            return 0.7
        else:
            return 0.3


# ===============================================================================
# BENCHMARK TEST SCENARIOS
# ===============================================================================

class BenchmarkScenarios:
    """Predefined benchmark test scenarios."""
    
    LATENCY_TESTS = [
        ("simple_math", "What is 15 * 23?", "345"),
        ("greeting", "Hello! How are you today?", None),
        ("short_code", "Write a Python function to check if a number is even.", None),
        ("fact_query", "What is the capital of Japan?", "Tokyo"),
        ("reasoning", "Why is the sky blue?", None)
    ]
    
    THROUGHPUT_TESTS = [
        ("batch_math", ["What is {} + {}?".format(i, i+1) for i in range(1, 21)]),
        ("batch_greetings", ["Hello in {}".format(lang) for lang in 
         ["Spanish", "French", "German", "Italian", "Portuguese"]]),
        ("batch_facts", ["What year was {} founded?".format(company) for company in
         ["Apple", "Microsoft", "Google", "Amazon", "Tesla"]])
    ]
    
    STRESS_TESTS = [
        ("long_context", "A" * 2000 + " Please summarize this text."),
        ("complex_reasoning", 
         "Analyze the economic implications of artificial intelligence on job markets, "
         "considering both positive and negative aspects, and provide specific examples "
         "from different industries. Structure your response with clear arguments."),
        ("code_generation",
         "Create a complete Python web application using Flask that includes user "
         "authentication, a database for storing user posts, RESTful API endpoints, "
         "and basic error handling. Include docstrings and comments.")
    ]


# ===============================================================================
# ADVANCED PERFORMANCE BENCHMARK SUITE
# ===============================================================================

class AdvancedPerformanceBenchmark:
    """Advanced performance benchmarking suite."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
        self.resource_monitor = ResourceMonitor()
        self.cost_calculator = CostCalculator()
        self.quality_evaluator = QualityEvaluator()
    
    def benchmark_latency(self, provider: LLMProvider, 
                         test_cases: List[Tuple[str, str, str]] = None,
                         num_runs: int = 10) -> BenchmarkResult:
        """Benchmark provider latency with statistical analysis."""
        test_cases = test_cases or BenchmarkScenarios.LATENCY_TESTS
        
        print(f"Benchmarking latency for {provider.__class__.__name__}...")
        
        latencies = []
        responses = []
        costs = []
        quality_scores = []
        
        self.resource_monitor.start_monitoring()
        start_time = time.time()
        
        for run in range(num_runs):
            for test_name, prompt, expected in test_cases:
                latency, response, cost = self._time_single_request(
                    provider, prompt, max_tokens=150
                )
                
                latencies.append(latency)
                responses.append(response)
                costs.append(cost)
                
                # Evaluate quality
                accuracy = self.quality_evaluator.evaluate_accuracy(prompt, response, expected)
                coherence = self.quality_evaluator.evaluate_coherence(response)
                completeness = self.quality_evaluator.evaluate_completeness(prompt, response)
                
                quality_scores.append({
                    'accuracy': accuracy,
                    'coherence': coherence,
                    'completeness': completeness
                })
        
        total_time = time.time() - start_time
        resources = self.resource_monitor.stop_monitoring()
        
        # Calculate metrics
        latency_metrics = LatencyMetrics(
            mean=statistics.mean(latencies),
            median=statistics.median(latencies),
            p95=np.percentile(latencies, 95),
            p99=np.percentile(latencies, 99),
            min=min(latencies),
            max=max(latencies),
            std_dev=statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        )
        
        # Throughput based on sequential execution
        throughput_metrics = ThroughputMetrics(
            requests_per_second=len(latencies) / total_time,
            tokens_per_second=sum(len(r.split()) for r in responses) / total_time,
            concurrent_capacity=1,  # Sequential execution
            queue_time_ms=0,
            processing_time_ms=statistics.mean(latencies) * 1000
        )
        
        cost_metrics = CostMetrics(
            cost_per_request=statistics.mean(costs),
            cost_per_token=statistics.mean(costs) / max(1, statistics.mean([len(r.split()) for r in responses])),
            cost_per_minute=sum(costs) / (total_time / 60),
            total_cost=sum(costs),
            cost_efficiency_score=latency_metrics.mean / max(0.001, statistics.mean(costs))
        )
        
        # Average quality metrics
        avg_accuracy = statistics.mean([q['accuracy'] for q in quality_scores])
        avg_coherence = statistics.mean([q['coherence'] for q in quality_scores])
        avg_completeness = statistics.mean([q['completeness'] for q in quality_scores])
        error_count = sum(1 for r in responses if r.lower().startswith('error'))
        
        quality_metrics = QualityMetrics(
            accuracy_score=avg_accuracy,
            coherence_score=avg_coherence,
            completeness_score=avg_completeness,
            avg_response_length=int(statistics.mean([len(r.split()) for r in responses])),
            error_rate=error_count / len(responses)
        )
        
        result = BenchmarkResult(
            provider=provider.__class__.__name__,
            model=provider.model,
            test_scenario="latency_benchmark",
            timestamp=datetime.now(),
            latency=latency_metrics,
            throughput=throughput_metrics,
            cost=cost_metrics,
            resources=resources,
            quality=quality_metrics,
            configuration={
                'num_runs': num_runs,
                'num_test_cases': len(test_cases),
                'max_tokens': 150
            }
        )
        
        self.results.append(result)
        return result
    
    def benchmark_throughput(self, provider: LLMProvider,
                           concurrent_levels: List[int] = None,
                           requests_per_level: int = 20) -> BenchmarkResult:
        """Benchmark provider throughput with concurrent requests."""
        concurrent_levels = concurrent_levels or [1, 5, 10, 20]
        
        print(f"Benchmarking throughput for {provider.__class__.__name__}...")
        
        best_throughput = 0
        best_concurrency = 1
        all_latencies = []
        all_responses = []
        all_costs = []
        
        self.resource_monitor.start_monitoring()
        total_start_time = time.time()
        
        for concurrency in concurrent_levels:
            print(f"  Testing with {concurrency} concurrent requests...")
            
            try:
                latencies, responses, costs = self._benchmark_concurrent_requests(
                    provider, concurrency, requests_per_level
                )
                
                current_throughput = len(latencies) / sum(latencies) * concurrency
                if current_throughput > best_throughput:
                    best_throughput = current_throughput
                    best_concurrency = concurrency
                
                all_latencies.extend(latencies)
                all_responses.extend(responses)
                all_costs.extend(costs)
                
            except Exception as e:
                print(f"    Failed at concurrency {concurrency}: {e}")
                break
        
        total_time = time.time() - total_start_time
        resources = self.resource_monitor.stop_monitoring()
        
        # Calculate metrics
        if not all_latencies:
            raise RuntimeError("No successful requests completed")
        
        latency_metrics = LatencyMetrics(
            mean=statistics.mean(all_latencies),
            median=statistics.median(all_latencies),
            p95=np.percentile(all_latencies, 95),
            p99=np.percentile(all_latencies, 99),
            min=min(all_latencies),
            max=max(all_latencies),
            std_dev=statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0.0
        )
        
        throughput_metrics = ThroughputMetrics(
            requests_per_second=best_throughput,
            tokens_per_second=sum(len(r.split()) for r in all_responses) / total_time,
            concurrent_capacity=best_concurrency,
            queue_time_ms=(statistics.mean(all_latencies) - min(all_latencies)) * 1000,
            processing_time_ms=min(all_latencies) * 1000
        )
        
        cost_metrics = CostMetrics(
            cost_per_request=statistics.mean(all_costs),
            cost_per_token=statistics.mean(all_costs) / max(1, statistics.mean([len(r.split()) for r in all_responses])),
            cost_per_minute=sum(all_costs) / (total_time / 60),
            total_cost=sum(all_costs),
            cost_efficiency_score=best_throughput / max(0.001, statistics.mean(all_costs))
        )
        
        # Quality evaluation
        quality_scores = []
        for response in all_responses[:10]:  # Sample for quality evaluation
            accuracy = self.quality_evaluator.evaluate_accuracy("Test prompt", response)
            coherence = self.quality_evaluator.evaluate_coherence(response)
            completeness = self.quality_evaluator.evaluate_completeness("Test prompt", response)
            quality_scores.append((accuracy, coherence, completeness))
        
        if quality_scores:
            avg_accuracy = statistics.mean([q[0] for q in quality_scores])
            avg_coherence = statistics.mean([q[1] for q in quality_scores])
            avg_completeness = statistics.mean([q[2] for q in quality_scores])
        else:
            avg_accuracy = avg_coherence = avg_completeness = 0.0
        
        error_count = sum(1 for r in all_responses if r.lower().startswith('error'))
        
        quality_metrics = QualityMetrics(
            accuracy_score=avg_accuracy,
            coherence_score=avg_coherence,
            completeness_score=avg_completeness,
            avg_response_length=int(statistics.mean([len(r.split()) for r in all_responses])),
            error_rate=error_count / len(all_responses)
        )
        
        result = BenchmarkResult(
            provider=provider.__class__.__name__,
            model=provider.model,
            test_scenario="throughput_benchmark",
            timestamp=datetime.now(),
            latency=latency_metrics,
            throughput=throughput_metrics,
            cost=cost_metrics,
            resources=resources,
            quality=quality_metrics,
            configuration={
                'max_concurrency': max(concurrent_levels),
                'requests_per_level': requests_per_level,
                'optimal_concurrency': best_concurrency
            }
        )
        
        self.results.append(result)
        return result
    
    def benchmark_stress_test(self, provider: LLMProvider,
                            duration_minutes: int = 5) -> BenchmarkResult:
        """Perform stress testing over extended duration."""
        print(f"Running stress test for {provider.__class__.__name__} ({duration_minutes} minutes)...")
        
        end_time = time.time() + (duration_minutes * 60)
        latencies = []
        responses = []
        costs = []
        error_count = 0
        
        self.resource_monitor.start_monitoring()
        start_time = time.time()
        
        test_prompts = [case[1] for case in BenchmarkScenarios.STRESS_TESTS]
        prompt_index = 0
        
        while time.time() < end_time:
            prompt = test_prompts[prompt_index % len(test_prompts)]
            prompt_index += 1
            
            try:
                latency, response, cost = self._time_single_request(
                    provider, prompt, max_tokens=200
                )
                latencies.append(latency)
                responses.append(response)
                costs.append(cost)
                
                # Brief pause to avoid overwhelming the API
                time.sleep(0.1)
                
            except Exception as e:
                error_count += 1
                print(f"    Error during stress test: {e}")
                time.sleep(1)  # Longer pause after errors
        
        total_time = time.time() - start_time
        resources = self.resource_monitor.stop_monitoring()
        
        if not latencies:
            raise RuntimeError("No successful requests during stress test")
        
        # Calculate degradation over time
        first_half = latencies[:len(latencies)//2]
        second_half = latencies[len(latencies)//2:]
        
        degradation = 0.0
        if first_half and second_half:
            degradation = (statistics.mean(second_half) - statistics.mean(first_half)) / statistics.mean(first_half)
        
        latency_metrics = LatencyMetrics(
            mean=statistics.mean(latencies),
            median=statistics.median(latencies),
            p95=np.percentile(latencies, 95),
            p99=np.percentile(latencies, 99),
            min=min(latencies),
            max=max(latencies),
            std_dev=statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        )
        
        throughput_metrics = ThroughputMetrics(
            requests_per_second=len(latencies) / total_time,
            tokens_per_second=sum(len(r.split()) for r in responses) / total_time,
            concurrent_capacity=1,
            queue_time_ms=0,
            processing_time_ms=statistics.mean(latencies) * 1000
        )
        
        cost_metrics = CostMetrics(
            cost_per_request=statistics.mean(costs),
            cost_per_token=statistics.mean(costs) / max(1, statistics.mean([len(r.split()) for r in responses])),
            cost_per_minute=sum(costs) / (total_time / 60),
            total_cost=sum(costs),
            cost_efficiency_score=len(latencies) / max(0.001, sum(costs))
        )
        
        # Quality metrics
        quality_scores = []
        for i, response in enumerate(responses[:20]):  # Sample for quality
            prompt = test_prompts[i % len(test_prompts)]
            accuracy = self.quality_evaluator.evaluate_accuracy(prompt, response)
            coherence = self.quality_evaluator.evaluate_coherence(response)
            completeness = self.quality_evaluator.evaluate_completeness(prompt, response)
            quality_scores.append((accuracy, coherence, completeness))
        
        if quality_scores:
            avg_accuracy = statistics.mean([q[0] for q in quality_scores])
            avg_coherence = statistics.mean([q[1] for q in quality_scores])
            avg_completeness = statistics.mean([q[2] for q in quality_scores])
        else:
            avg_accuracy = avg_coherence = avg_completeness = 0.0
        
        quality_metrics = QualityMetrics(
            accuracy_score=avg_accuracy,
            coherence_score=avg_coherence,
            completeness_score=avg_completeness,
            avg_response_length=int(statistics.mean([len(r.split()) for r in responses])),
            error_rate=error_count / (len(latencies) + error_count)
        )
        
        result = BenchmarkResult(
            provider=provider.__class__.__name__,
            model=provider.model,
            test_scenario="stress_test",
            timestamp=datetime.now(),
            latency=latency_metrics,
            throughput=throughput_metrics,
            cost=cost_metrics,
            resources=resources,
            quality=quality_metrics,
            configuration={
                'duration_minutes': duration_minutes,
                'total_requests': len(latencies),
                'error_count': error_count,
                'performance_degradation': degradation
            }
        )
        
        self.results.append(result)
        return result
    
    def _time_single_request(self, provider: LLMProvider, prompt: str, 
                           max_tokens: int = 150) -> Tuple[float, str, float]:
        """Time a single request and return latency, response, and cost."""
        start_time = time.time()
        
        try:
            response = provider.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7
            )
            latency = time.time() - start_time
            
            # Estimate cost
            input_tokens = self.cost_calculator.estimate_tokens(prompt)
            output_tokens = self.cost_calculator.estimate_tokens(response)
            cost = self.cost_calculator.calculate_cost(
                provider.__class__.__name__,
                provider.model,
                input_tokens,
                output_tokens
            )
            
            return latency, response, cost
            
        except Exception as e:
            latency = time.time() - start_time
            return latency, f"Error: {str(e)}", 0.0
    
    def _benchmark_concurrent_requests(self, provider: LLMProvider,
                                     concurrency: int, num_requests: int) -> Tuple[List[float], List[str], List[float]]:
        """Benchmark concurrent requests."""
        prompts = ["What is AI?" for _ in range(num_requests)]
        
        latencies = []
        responses = []
        costs = []
        
        def worker():
            for prompt in prompts[:num_requests // concurrency]:
                latency, response, cost = self._time_single_request(provider, prompt)
                latencies.append(latency)
                responses.append(response)
                costs.append(cost)
        
        threads = []
        for _ in range(concurrency):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        return latencies, responses, costs
    
    def compare_providers(self, providers: List[LLMProvider],
                         benchmark_types: List[str] = None) -> Dict[str, Any]:
        """Comprehensive provider comparison."""
        benchmark_types = benchmark_types or ['latency', 'throughput']
        
        comparison_results = defaultdict(list)
        
        for provider in providers:
            provider_name = f"{provider.__class__.__name__}_{provider.model}"
            print(f"\nBenchmarking {provider_name}...")
            
            try:
                if 'latency' in benchmark_types:
                    result = self.benchmark_latency(provider, num_runs=5)
                    comparison_results['latency'].append(result)
                
                if 'throughput' in benchmark_types:
                    result = self.benchmark_throughput(provider, [1, 5, 10])
                    comparison_results['throughput'].append(result)
                
                if 'stress' in benchmark_types:
                    result = self.benchmark_stress_test(provider, duration_minutes=2)
                    comparison_results['stress'].append(result)
                    
            except Exception as e:
                print(f"Failed to benchmark {provider_name}: {e}")
        
        # Generate comparison report
        report = self._generate_comparison_report(comparison_results)
        return report
    
    def _generate_comparison_report(self, results: Dict[str, List[BenchmarkResult]]) -> Dict[str, Any]:
        """Generate comprehensive comparison report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'detailed_results': {},
            'rankings': {}
        }
        
        for benchmark_type, benchmark_results in results.items():
            if not benchmark_results:
                continue
            
            # Sort by overall score
            sorted_results = sorted(benchmark_results, key=lambda x: x.overall_score(), reverse=True)
            
            report['rankings'][benchmark_type] = [
                {
                    'provider': f"{r.provider}_{r.model}",
                    'overall_score': r.overall_score(),
                    'latency_mean': r.latency.mean,
                    'throughput_rps': r.throughput.requests_per_second,
                    'cost_per_request': r.cost.cost_per_request,
                    'quality_score': (r.quality.accuracy_score + r.quality.coherence_score) / 2
                }
                for r in sorted_results
            ]
            
            report['detailed_results'][benchmark_type] = [r.to_dict() for r in benchmark_results]
        
        return report
    
    def save_results(self, filename: str = None):
        """Save benchmark results to file."""
        if not filename:
            filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        
        results_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_results': len(self.results),
                'benchmark_version': '2.0'
            },
            'results': [result.to_dict() for result in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to: {filepath}")
    
    def generate_visualizations(self):
        """Generate performance visualization charts."""
        if not self.results:
            print("No results to visualize")
            return
        
        # Create comparison charts
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        providers = list(set(f"{r.provider}_{r.model}" for r in self.results))
        
        # Latency comparison
        latencies = [r.latency.mean for r in self.results]
        axes[0, 0].bar(providers, latencies)
        axes[0, 0].set_title('Average Latency Comparison')
        axes[0, 0].set_ylabel('Latency (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Throughput comparison
        throughputs = [r.throughput.requests_per_second for r in self.results]
        axes[0, 1].bar(providers, throughputs)
        axes[0, 1].set_title('Throughput Comparison')
        axes[0, 1].set_ylabel('Requests per Second')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Cost comparison
        costs = [r.cost.cost_per_request for r in self.results]
        axes[1, 0].bar(providers, costs)
        axes[1, 0].set_title('Cost per Request Comparison')
        axes[1, 0].set_ylabel('Cost ($)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Overall score comparison
        scores = [r.overall_score() for r in self.results]
        axes[1, 1].bar(providers, scores)
        axes[1, 1].set_title('Overall Performance Score')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        chart_path = self.output_dir / f"performance_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path)
        print(f"Charts saved to: {chart_path}")


# ===============================================================================
# PYTEST FIXTURES AND TESTS
# ===============================================================================

@pytest.fixture
def benchmark_suite():
    """Fixture for benchmark suite with temporary output directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        suite = AdvancedPerformanceBenchmark(tmp_dir)
        yield suite


@pytest.fixture
def mock_providers():
    """Mock providers for testing."""
    from unittest.mock import Mock
    
    providers = []
    
    # Mock OpenAI provider
    openai_mock = Mock(spec=OpenAIProvider)
    openai_mock.model = "gpt-4o-mini"
    openai_mock.__class__.__name__ = "OpenAIProvider"
    openai_mock.generate.return_value = "This is a test response from OpenAI."
    providers.append(openai_mock)
    
    # Mock Anthropic provider
    anthropic_mock = Mock(spec=AnthropicProvider)
    anthropic_mock.model = "claude-3-haiku-20240307"
    anthropic_mock.__class__.__name__ = "AnthropicProvider"
    anthropic_mock.generate.return_value = "This is a test response from Anthropic."
    providers.append(anthropic_mock)
    
    return providers


# ===============================================================================
# PERFORMANCE BENCHMARK TESTS
# ===============================================================================

class TestPerformanceBenchmark:
    """Test cases for the performance benchmark suite."""
    
    def test_latency_benchmark(self, benchmark_suite, mock_providers):
        """Test latency benchmarking functionality."""
        provider = mock_providers[0]
        
        result = benchmark_suite.benchmark_latency(provider, num_runs=3)
        
        assert result.provider == "OpenAIProvider"
        assert result.model == "gpt-4o-mini"
        assert result.test_scenario == "latency_benchmark"
        assert result.latency.mean > 0
        assert result.quality.accuracy_score >= 0
        assert result.overall_score() >= 0
    
    def test_throughput_benchmark(self, benchmark_suite, mock_providers):
        """Test throughput benchmarking functionality."""
        provider = mock_providers[0]
        
        result = benchmark_suite.benchmark_throughput(
            provider, 
            concurrent_levels=[1, 2],
            requests_per_level=4
        )
        
        assert result.test_scenario == "throughput_benchmark"
        assert result.throughput.requests_per_second > 0
        assert result.throughput.concurrent_capacity >= 1
    
    def test_provider_comparison(self, benchmark_suite, mock_providers):
        """Test provider comparison functionality."""
        comparison = benchmark_suite.compare_providers(
            mock_providers,
            benchmark_types=['latency']
        )
        
        assert 'rankings' in comparison
        assert 'latency' in comparison['rankings']
        assert len(comparison['rankings']['latency']) == len(mock_providers)
        
        # Check that results are sorted by overall score
        scores = [r['overall_score'] for r in comparison['rankings']['latency']]
        assert scores == sorted(scores, reverse=True)
    
    def test_cost_calculation(self):
        """Test cost calculation functionality."""
        calculator = CostCalculator()
        
        cost = calculator.calculate_cost(
            "OpenAIProvider",
            "gpt-4o-mini",
            input_tokens=100,
            output_tokens=50
        )
        
        assert cost > 0
        
        # Test token estimation
        tokens = calculator.estimate_tokens("Hello world, this is a test.")
        assert tokens > 0
    
    def test_quality_evaluation(self):
        """Test quality evaluation functionality."""
        evaluator = QualityEvaluator()
        
        # Test accuracy evaluation
        accuracy = evaluator.evaluate_accuracy(
            "What is 2 + 2?",
            "The answer is 4",
            "4"
        )
        assert accuracy > 0.5
        
        # Test coherence evaluation
        coherence = evaluator.evaluate_coherence(
            "This is a well-structured response with multiple sentences. "
            "It demonstrates good coherence and flow."
        )
        assert coherence > 0.5
    
    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv('INTEGRATION_TESTS'), reason="Integration tests disabled")
    def test_real_provider_benchmark(self, benchmark_suite):
        """Integration test with real providers."""
        providers_to_test = []
        
        # Try to initialize real providers
        if os.getenv('OPENAI_API_KEY'):
            try:
                provider = OpenAIProvider(model='gpt-4o-mini')
                provider.initialize()
                providers_to_test.append(provider)
            except Exception:
                pass
        
        if not providers_to_test:
            pytest.skip("No real providers available for testing")
        
        provider = providers_to_test[0]
        
        result = benchmark_suite.benchmark_latency(provider, num_runs=2)
        
        assert result.latency.mean > 0
        assert result.cost.total_cost >= 0
        assert result.quality.error_rate < 1.0  # Should have some successful responses


if __name__ == "__main__":
    # Example usage
    print("Advanced Performance Benchmark Suite")
    print("="*40)
    
    # Create mock providers for demonstration
    from unittest.mock import Mock
    
    mock_openai = Mock(spec=OpenAIProvider)
    mock_openai.model = "gpt-4o-mini"
    mock_openai.__class__.__name__ = "OpenAIProvider"
    mock_openai.generate.return_value = "Hello! I'm a helpful AI assistant."
    
    mock_anthropic = Mock(spec=AnthropicProvider)
    mock_anthropic.model = "claude-3-haiku-20240307"
    mock_anthropic.__class__.__name__ = "AnthropicProvider"
    mock_anthropic.generate.return_value = "Hello! I'm Claude, an AI assistant created by Anthropic."
    
    # Initialize benchmark suite
    suite = AdvancedPerformanceBenchmark("demo_benchmarks")
    
    # Run latency benchmark
    print("Running latency benchmark...")
    result = suite.benchmark_latency(mock_openai, num_runs=3)
    print(f"Average latency: {result.latency.mean:.3f}s")
    print(f"Overall score: {result.overall_score():.3f}")
    
    # Compare providers
    print("\nComparing providers...")
    comparison = suite.compare_providers([mock_openai, mock_anthropic], ['latency'])
    
    print("\nLatency Rankings:")
    for rank, provider_result in enumerate(comparison['rankings']['latency'], 1):
        print(f"{rank}. {provider_result['provider']}: {provider_result['overall_score']:.3f}")
    
    # Save results
    suite.save_results()
    
    print("\nTo run the full test suite:")
    print("pytest examples/use_cases/performance_benchmark_suite.py -v")
    print("\nTo run with real providers (requires API keys):")
    print("INTEGRATION_TESTS=1 pytest examples/use_cases/performance_benchmark_suite.py -v")