"""
Individual performance test implementations

This module contains specific performance test classes for different
types of benchmarking scenarios.
"""

import time
import statistics
import threading
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging

from llm_providers.base import LLMProvider
from llm_providers.exceptions import ProviderError, RateLimitError
from .benchmark_config import BenchmarkConfig, rate_performance
from .benchmark_suite import BenchmarkMetrics

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result from a single performance test."""
    test_name: str
    provider: str
    model: str
    success: bool
    duration: float
    metrics: Dict[str, Any]
    error: Optional[str] = None


class BasePerformanceTest(ABC):
    """Base class for performance tests."""
    
    def __init__(self, name: str):
        self.name = name
        self.results: List[TestResult] = []
    
    @abstractmethod
    def run_test(self, provider: LLMProvider, **kwargs) -> TestResult:
        """Run the performance test."""
        pass
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of test results."""
        if not self.results:
            return {"test_name": self.name, "results": []}
        
        successful = [r for r in self.results if r.success]
        
        return {
            "test_name": self.name,
            "total_runs": len(self.results),
            "successful_runs": len(successful),
            "success_rate": len(successful) / len(self.results) * 100,
            "avg_duration": statistics.mean(r.duration for r in successful) if successful else 0,
            "results": [
                {
                    "provider": r.provider,
                    "model": r.model,
                    "success": r.success,
                    "duration": r.duration,
                    "metrics": r.metrics,
                    "error": r.error
                }
                for r in self.results
            ]
        }


class ResponseTimeTest(BasePerformanceTest):
    """Test response time performance across different prompt types."""
    
    def __init__(self):
        super().__init__("Response Time Test")
    
    def run_test(self, provider: LLMProvider, 
                 prompt_category: str = "medium",
                 num_requests: int = 10,
                 **kwargs) -> TestResult:
        """
        Run response time test.
        
        Args:
            provider: LLM provider to test
            prompt_category: Category of prompts to use
            num_requests: Number of requests to make
        """
        logger.info(f"Running response time test for {provider.__class__.__name__}")
        
        prompts = BenchmarkConfig.get_prompts(prompt_category, num_requests)
        token_limits = BenchmarkConfig.get_token_limits(prompt_category)
        
        response_times = []
        errors = []
        
        start_time = time.time()
        
        for i, prompt in enumerate(prompts):
            try:
                request_start = time.time()
                response = provider.generate(
                    prompt, 
                    max_tokens=token_limits['max'],
                    **kwargs
                )
                request_end = time.time()
                
                response_time = request_end - request_start
                response_times.append(response_time)
                
                # Rate limiting
                if i < len(prompts) - 1:  # Don't delay after last request
                    delay = BenchmarkConfig.get_rate_limit_delay(
                        provider.__class__.__name__.lower().replace('provider', '')
                    )
                    time.sleep(delay)
                
            except Exception as e:
                errors.append(str(e))
                logger.warning(f"Request failed: {e}")
        
        total_duration = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            "total_requests": len(prompts),
            "successful_requests": len(response_times),
            "failed_requests": len(errors),
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "median_response_time": statistics.median(response_times) if response_times else 0,
            "response_time_std": statistics.stdev(response_times) if len(response_times) > 1 else 0,
            "errors": errors,
            "prompt_category": prompt_category,
            "performance_rating": rate_performance(
                statistics.mean(response_times) if response_times else float('inf'),
                "response_time",
                higher_is_better=False
            )
        }
        
        result = TestResult(
            test_name=self.name,
            provider=provider.__class__.__name__,
            model=provider.model_name,
            success=len(response_times) > 0,
            duration=total_duration,
            metrics=metrics,
            error=f"{len(errors)} requests failed" if errors else None
        )
        
        self.results.append(result)
        return result


class ThroughputTest(BasePerformanceTest):
    """Test throughput (requests per second) performance."""
    
    def __init__(self):
        super().__init__("Throughput Test")
    
    def run_test(self, provider: LLMProvider,
                 num_requests: int = 20,
                 max_workers: int = 3,
                 **kwargs) -> TestResult:
        """
        Run throughput test with concurrent requests.
        
        Args:
            provider: LLM provider to test
            num_requests: Total number of requests
            max_workers: Maximum concurrent workers
        """
        logger.info(f"Running throughput test for {provider.__class__.__name__}")
        
        # Use simple prompts for throughput testing
        base_prompt = "Respond briefly: What is artificial intelligence?"
        
        successful_requests = 0
        failed_requests = 0
        response_times = []
        errors = []
        
        start_time = time.time()
        
        def make_request(request_id: int):
            try:
                request_start = time.time()
                response = provider.generate(
                    f"{base_prompt} (Request {request_id})",
                    max_tokens=50,
                    **kwargs
                )
                request_end = time.time()
                
                return {
                    "success": True,
                    "response_time": request_end - request_start,
                    "request_id": request_id
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "request_id": request_id
                }
        
        # Execute concurrent requests
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(make_request, i)
                for i in range(num_requests)
            ]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result["success"]:
                        successful_requests += 1
                        response_times.append(result["response_time"])
                    else:
                        failed_requests += 1
                        errors.append(result["error"])
                except Exception as e:
                    failed_requests += 1
                    errors.append(str(e))
        
        total_duration = time.time() - start_time
        
        # Calculate throughput metrics
        throughput = successful_requests / total_duration if total_duration > 0 else 0
        
        metrics = {
            "total_requests": num_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "total_duration": total_duration,
            "throughput_rps": throughput,
            "throughput_rpm": throughput * 60,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "concurrent_workers": max_workers,
            "errors": errors[:10],  # Limit error list size
            "performance_rating": rate_performance(
                throughput * 60,  # Convert to RPM for rating
                "throughput",
                higher_is_better=True
            )
        }
        
        result = TestResult(
            test_name=self.name,
            provider=provider.__class__.__name__,
            model=provider.model_name,
            success=successful_requests > 0,
            duration=total_duration,
            metrics=metrics,
            error=f"{failed_requests} requests failed" if failed_requests > 0 else None
        )
        
        self.results.append(result)
        return result


class TokenEfficiencyTest(BasePerformanceTest):
    """Test token processing efficiency (tokens per second)."""
    
    def __init__(self):
        super().__init__("Token Efficiency Test")
    
    def run_test(self, provider: LLMProvider,
                 num_requests: int = 5,
                 **kwargs) -> TestResult:
        """
        Run token efficiency test.
        
        Args:
            provider: LLM provider to test
            num_requests: Number of requests to make
        """
        logger.info(f"Running token efficiency test for {provider.__class__.__name__}")
        
        # Use prompts that generate substantial responses
        prompts = [
            "Write a comprehensive explanation of machine learning algorithms and their applications in modern technology.",
            "Describe the complete software development lifecycle from planning to deployment and maintenance.",
            "Explain the principles of renewable energy systems and their impact on environmental sustainability.",
            "Analyze the evolution of artificial intelligence and its potential future developments.",
            "Discuss the fundamentals of cloud computing and its advantages for modern businesses."
        ]
        
        token_rates = []
        response_times = []
        total_tokens_processed = 0
        errors = []
        
        start_time = time.time()
        
        for i, prompt in enumerate(prompts[:num_requests]):
            try:
                request_start = time.time()
                response = provider.generate(
                    prompt,
                    max_tokens=300,
                    temperature=0.7,
                    **kwargs
                )
                request_end = time.time()
                
                response_time = request_end - request_start
                response_times.append(response_time)
                
                # Estimate tokens (rough approximation)
                prompt_tokens = len(prompt.split())
                response_tokens = len(response.split()) if response else 0
                total_tokens = prompt_tokens + response_tokens
                total_tokens_processed += total_tokens
                
                # Calculate tokens per second for this request
                if response_time > 0:
                    tokens_per_second = total_tokens / response_time
                    token_rates.append(tokens_per_second)
                
                # Rate limiting for token-heavy requests
                if i < num_requests - 1:
                    time.sleep(2.0)  # Longer delay for comprehensive requests
                
            except Exception as e:
                errors.append(str(e))
                logger.warning(f"Token efficiency request failed: {e}")
        
        total_duration = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            "total_requests": num_requests,
            "successful_requests": len(token_rates),
            "failed_requests": len(errors),
            "total_duration": total_duration,
            "total_tokens_processed": total_tokens_processed,
            "avg_tokens_per_second": statistics.mean(token_rates) if token_rates else 0,
            "max_tokens_per_second": max(token_rates) if token_rates else 0,
            "min_tokens_per_second": min(token_rates) if token_rates else 0,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "errors": errors,
            "performance_rating": rate_performance(
                statistics.mean(token_rates) if token_rates else 0,
                "token_efficiency",
                higher_is_better=True
            )
        }
        
        result = TestResult(
            test_name=self.name,
            provider=provider.__class__.__name__,
            model=provider.model_name,
            success=len(token_rates) > 0,
            duration=total_duration,
            metrics=metrics,
            error=f"{len(errors)} requests failed" if errors else None
        )
        
        self.results.append(result)
        return result


class ConcurrencyTest(BasePerformanceTest):
    """Test concurrent request handling performance."""
    
    def __init__(self):
        super().__init__("Concurrency Test")
    
    def run_test(self, provider: LLMProvider,
                 num_requests: int = 15,
                 max_workers: int = 5,
                 **kwargs) -> TestResult:
        """
        Run concurrency test with varying levels of concurrent requests.
        
        Args:
            provider: LLM provider to test
            num_requests: Total number of requests
            max_workers: Maximum concurrent workers
        """
        logger.info(f"Running concurrency test for {provider.__class__.__name__}")
        
        prompt = "Explain the concept of parallel processing in computing."
        
        # Test different concurrency levels
        concurrency_levels = [1, 2, max_workers]
        concurrency_results = {}
        
        start_time = time.time()
        
        for workers in concurrency_levels:
            logger.info(f"Testing with {workers} concurrent workers")
            
            successful = 0
            failed = 0
            response_times = []
            errors = []
            
            level_start = time.time()
            
            def make_concurrent_request(request_id: int):
                try:
                    req_start = time.time()
                    response = provider.generate(
                        f"{prompt} (Worker test {request_id})",
                        max_tokens=100,
                        **kwargs
                    )
                    req_end = time.time()
                    
                    return {
                        "success": True,
                        "response_time": req_end - req_start,
                        "request_id": request_id
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "request_id": request_id
                    }
            
            # Execute requests with current concurrency level
            requests_per_level = num_requests // len(concurrency_levels)
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(make_concurrent_request, i)
                    for i in range(requests_per_level)
                ]
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result["success"]:
                            successful += 1
                            response_times.append(result["response_time"])
                        else:
                            failed += 1
                            errors.append(result["error"])
                    except Exception as e:
                        failed += 1
                        errors.append(str(e))
            
            level_duration = time.time() - level_start
            
            concurrency_results[workers] = {
                "workers": workers,
                "successful_requests": successful,
                "failed_requests": failed,
                "avg_response_time": statistics.mean(response_times) if response_times else 0,
                "total_duration": level_duration,
                "throughput": successful / level_duration if level_duration > 0 else 0,
                "errors": errors[:5]  # Limit error list
            }
            
            # Brief pause between concurrency levels
            time.sleep(1.0)
        
        total_duration = time.time() - start_time
        
        # Analyze concurrency scaling
        best_throughput = max(
            (result["throughput"] for result in concurrency_results.values()),
            default=0
        )
        
        scaling_efficiency = {}
        baseline_throughput = concurrency_results.get(1, {}).get("throughput", 0)
        
        for workers, result in concurrency_results.items():
            if baseline_throughput > 0:
                expected_throughput = baseline_throughput * workers
                actual_throughput = result["throughput"]
                efficiency = (actual_throughput / expected_throughput) * 100 if expected_throughput > 0 else 0
                scaling_efficiency[workers] = efficiency
        
        metrics = {
            "concurrency_levels_tested": list(concurrency_levels),
            "concurrency_results": concurrency_results,
            "best_throughput": best_throughput,
            "scaling_efficiency": scaling_efficiency,
            "total_duration": total_duration,
            "optimal_workers": max(
                concurrency_results.items(),
                key=lambda x: x[1]["throughput"],
                default=(1, {})
            )[0],
            "performance_rating": rate_performance(
                best_throughput * 60,  # Convert to RPM
                "throughput",
                higher_is_better=True
            )
        }
        
        # Overall success based on best performing concurrency level
        best_result = max(concurrency_results.values(), key=lambda x: x["successful_requests"], default={})
        
        result = TestResult(
            test_name=self.name,
            provider=provider.__class__.__name__,
            model=provider.model_name,
            success=best_result.get("successful_requests", 0) > 0,
            duration=total_duration,
            metrics=metrics,
            error=None
        )
        
        self.results.append(result)
        return result


class MemoryUsageTest(BasePerformanceTest):
    """Test memory usage during requests."""
    
    def __init__(self):
        super().__init__("Memory Usage Test")
    
    def run_test(self, provider: LLMProvider,
                 num_requests: int = 10,
                 **kwargs) -> TestResult:
        """
        Run memory usage test.
        
        Args:
            provider: LLM provider to test
            num_requests: Number of requests to make
        """
        logger.info(f"Running memory usage test for {provider.__class__.__name__}")
        
        import psutil
        import gc
        
        # Force garbage collection before starting
        gc.collect()
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_measurements = []
        response_times = []
        successful_requests = 0
        errors = []
        
        prompt = "Generate a detailed analysis of data structures and algorithms used in computer science."
        
        start_time = time.time()
        
        for i in range(num_requests):
            try:
                # Measure memory before request
                pre_request_memory = process.memory_info().rss / 1024 / 1024
                
                request_start = time.time()
                response = provider.generate(
                    f"{prompt} (Request {i+1})",
                    max_tokens=200,
                    **kwargs
                )
                request_end = time.time()
                
                # Measure memory after request
                post_request_memory = process.memory_info().rss / 1024 / 1024
                
                response_times.append(request_end - request_start)
                memory_measurements.append({
                    "request": i + 1,
                    "pre_request_mb": pre_request_memory,
                    "post_request_mb": post_request_memory,
                    "memory_delta_mb": post_request_memory - pre_request_memory
                })
                
                successful_requests += 1
                
                # Brief pause to allow memory measurement
                time.sleep(0.5)
                
            except Exception as e:
                errors.append(str(e))
                logger.warning(f"Memory test request failed: {e}")
        
        # Force garbage collection and final measurement
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024
        
        total_duration = time.time() - start_time
        
        # Calculate memory metrics
        peak_memory = max(
            (m["post_request_mb"] for m in memory_measurements),
            default=initial_memory
        )
        
        avg_memory_per_request = statistics.mean(
            (m["memory_delta_mb"] for m in memory_measurements)
        ) if memory_measurements else 0
        
        memory_growth = final_memory - initial_memory
        
        metrics = {
            "total_requests": num_requests,
            "successful_requests": successful_requests,
            "failed_requests": len(errors),
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "peak_memory_mb": peak_memory,
            "memory_growth_mb": memory_growth,
            "avg_memory_per_request_mb": avg_memory_per_request,
            "memory_measurements": memory_measurements,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "total_duration": total_duration,
            "errors": errors[:5],
            "memory_efficiency_rating": "good" if memory_growth < 50 else "acceptable" if memory_growth < 100 else "poor"
        }
        
        result = TestResult(
            test_name=self.name,
            provider=provider.__class__.__name__,
            model=provider.model_name,
            success=successful_requests > 0,
            duration=total_duration,
            metrics=metrics,
            error=f"{len(errors)} requests failed" if errors else None
        )
        
        self.results.append(result)
        return result


class StressTest(BasePerformanceTest):
    """High-load stress testing."""
    
    def __init__(self):
        super().__init__("Stress Test")
    
    def run_test(self, provider: LLMProvider,
                 num_requests: int = 50,
                 max_workers: int = 8,
                 duration_limit: int = 300,  # 5 minutes
                 **kwargs) -> TestResult:
        """
        Run stress test with high concurrent load.
        
        Args:
            provider: LLM provider to test
            num_requests: Total number of requests
            max_workers: Maximum concurrent workers
            duration_limit: Maximum test duration in seconds
        """
        if not BenchmarkConfig.should_run_stress_tests():
            logger.info("Stress tests disabled, skipping...")
            return TestResult(
                test_name=self.name,
                provider=provider.__class__.__name__,
                model=provider.model_name,
                success=False,
                duration=0,
                metrics={"skipped": True},
                error="Stress tests disabled"
            )
        
        logger.info(f"Running stress test for {provider.__class__.__name__}")
        
        prompt = "Provide a quick summary of machine learning concepts."
        
        successful_requests = 0
        failed_requests = 0
        response_times = []
        errors = []
        rate_limit_errors = 0
        timeout_errors = 0
        
        start_time = time.time()
        
        def stress_request(request_id: int):
            try:
                req_start = time.time()
                response = provider.generate(
                    f"{prompt} #{request_id}",
                    max_tokens=100,
                    **kwargs
                )
                req_end = time.time()
                
                return {
                    "success": True,
                    "response_time": req_end - req_start,
                    "request_id": request_id
                }
            except RateLimitError as e:
                return {
                    "success": False,
                    "error": "rate_limit",
                    "error_detail": str(e),
                    "request_id": request_id
                }
            except Exception as e:
                error_type = "timeout" if "timeout" in str(e).lower() else "other"
                return {
                    "success": False,
                    "error": error_type,
                    "error_detail": str(e),
                    "request_id": request_id
                }
        
        # Execute high-load concurrent requests
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(stress_request, i)
                for i in range(num_requests)
            ]
            
            for future in as_completed(futures, timeout=duration_limit):
                try:
                    result = future.result()
                    if result["success"]:
                        successful_requests += 1
                        response_times.append(result["response_time"])
                    else:
                        failed_requests += 1
                        error_type = result.get("error", "unknown")
                        
                        if error_type == "rate_limit":
                            rate_limit_errors += 1
                        elif error_type == "timeout":
                            timeout_errors += 1
                        
                        errors.append(result.get("error_detail", "Unknown error"))
                        
                except Exception as e:
                    failed_requests += 1
                    errors.append(str(e))
        
        total_duration = time.time() - start_time
        
        # Calculate stress test metrics
        throughput = successful_requests / total_duration if total_duration > 0 else 0
        
        metrics = {
            "total_requests": num_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "rate_limit_errors": rate_limit_errors,
            "timeout_errors": timeout_errors,
            "other_errors": failed_requests - rate_limit_errors - timeout_errors,
            "success_rate": (successful_requests / num_requests) * 100,
            "total_duration": total_duration,
            "throughput_rps": throughput,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "response_time_std": statistics.stdev(response_times) if len(response_times) > 1 else 0,
            "max_workers": max_workers,
            "errors_sample": errors[:10],  # Limit error list size
            "stress_tolerance_rating": (
                "excellent" if successful_requests / num_requests > 0.9 else
                "good" if successful_requests / num_requests > 0.7 else
                "acceptable" if successful_requests / num_requests > 0.5 else
                "poor"
            )
        }
        
        result = TestResult(
            test_name=self.name,
            provider=provider.__class__.__name__,
            model=provider.model_name,
            success=successful_requests > 0,
            duration=total_duration,
            metrics=metrics,
            error=f"High failure rate: {failed_requests}/{num_requests}" if failed_requests > num_requests * 0.5 else None
        )
        
        self.results.append(result)
        return result