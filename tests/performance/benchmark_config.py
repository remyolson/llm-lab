"""
Configuration for performance benchmarks

This module defines configuration settings for performance benchmarking,
including test parameters, thresholds, and execution settings.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class BenchmarkMode(Enum):
    """Benchmark execution modes."""

    QUICK = "quick"  # Fast tests, minimal requests
    STANDARD = "standard"  # Standard benchmark suite
    COMPREHENSIVE = "comprehensive"  # Full benchmark suite
    STRESS = "stress"  # High-load stress testing


@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarks."""

    # Test execution settings
    EXECUTION_SETTINGS = {
        "quick": {
            "num_requests": 5,
            "concurrent_workers": 2,
            "warmup_requests": 1,
            "timeout": 15.0,
            "rate_limit_delay": 0.5,
        },
        "standard": {
            "num_requests": 20,
            "concurrent_workers": 3,
            "warmup_requests": 3,
            "timeout": 30.0,
            "rate_limit_delay": 1.0,
        },
        "comprehensive": {
            "num_requests": 50,
            "concurrent_workers": 5,
            "warmup_requests": 5,
            "timeout": 60.0,
            "rate_limit_delay": 1.2,
        },
        "stress": {
            "num_requests": 100,
            "concurrent_workers": 10,
            "warmup_requests": 10,
            "timeout": 120.0,
            "rate_limit_delay": 0.8,
        },
    }

    # Performance thresholds (in seconds)
    PERFORMANCE_THRESHOLDS = {
        "response_time": {"excellent": 1.0, "good": 3.0, "acceptable": 8.0, "poor": 15.0},
        "throughput": {  # requests per minute
            "excellent": 60,
            "good": 30,
            "acceptable": 15,
            "poor": 5,
        },
        "token_efficiency": {  # tokens per second
            "excellent": 100,
            "good": 50,
            "acceptable": 20,
            "poor": 10,
        },
    }

    # Test prompts for different categories
    BENCHMARK_PROMPTS = {
        "short": ["Hello", "What is 2+2?", "Name a color", "Say yes or no", "Count to 3"],
        "medium": [
            "Explain what machine learning is in one paragraph.",
            "Write a short poem about nature.",
            "List 5 benefits of renewable energy.",
            "Describe how to make a sandwich.",
            "What are the main differences between Python and JavaScript?",
        ],
        "long": [
            "Write a detailed explanation of quantum computing, including its principles, applications, and potential impact on various industries.",
            "Create a comprehensive guide for someone learning to program, covering the basics, best practices, and career advice.",
            "Analyze the causes and effects of climate change, discussing both scientific evidence and potential solutions.",
            "Explain the history of artificial intelligence, from its origins to modern developments and future possibilities.",
            "Describe the process of developing a mobile application from concept to deployment, including technical and business considerations.",
        ],
        "creative": [
            "Write a creative story about a robot who discovers emotions.",
            "Create a dialogue between two characters from different time periods.",
            "Invent a new sport and explain its rules.",
            "Write song lyrics about the beauty of mathematics.",
            "Describe a day in the life of a cloud.",
        ],
        "technical": [
            "Explain the difference between SQL and NoSQL databases.",
            "Write Python code to implement a binary search algorithm.",
            "Describe how HTTP/HTTPS protocols work.",
            "Explain the concept of microservices architecture.",
            "What is the difference between machine learning and deep learning?",
        ],
    }

    # Token limits for different test categories
    TOKEN_LIMITS = {
        "short": {"min": 10, "max": 50},
        "medium": {"min": 50, "max": 200},
        "long": {"min": 200, "max": 500},
        "creative": {"min": 100, "max": 300},
        "technical": {"min": 100, "max": 400},
    }

    # Provider-specific configurations
    PROVIDER_CONFIGS = {
        "openai": {
            "rate_limit": 3500,  # requests per minute
            "recommended_delay": 1.0,
            "max_concurrent": 5,
            "test_models": ["gpt-3.5-turbo", "gpt-4o-mini"],
        },
        "anthropic": {
            "rate_limit": 1000,
            "recommended_delay": 1.5,
            "max_concurrent": 3,
            "test_models": ["claude-3-haiku-20240307", "claude-3-sonnet-20240229"],
        },
        "google": {
            "rate_limit": 1500,
            "recommended_delay": 1.2,
            "max_concurrent": 4,
            "test_models": ["gemini-1.5-flash", "gemini-1.5-pro"],
        },
    }

    # Memory usage monitoring settings
    MEMORY_SETTINGS = {
        "monitor_interval": 0.1,  # seconds
        "memory_threshold_mb": 100,  # MB increase threshold
        "gc_between_tests": True,
        "track_peak_usage": True,
    }

    # Statistical analysis settings
    STATISTICS_SETTINGS = {
        "confidence_level": 0.95,
        "outlier_threshold": 2.0,  # standard deviations
        "min_samples": 10,
        "percentiles": [50, 75, 90, 95, 99],
    }

    # Output and reporting settings
    OUTPUT_SETTINGS = {
        "save_raw_data": True,
        "generate_charts": True,
        "chart_formats": ["png", "svg"],
        "export_formats": ["json", "csv", "html"],
        "detailed_logs": True,
    }

    @classmethod
    def get_mode_config(cls, mode: BenchmarkMode) -> Dict[str, Any]:
        """Get configuration for a specific benchmark mode."""
        return cls.EXECUTION_SETTINGS.get(mode.value, cls.EXECUTION_SETTINGS["standard"])

    @classmethod
    def get_benchmark_mode(cls) -> BenchmarkMode:
        """Get benchmark mode from environment variable."""
        mode_str = os.getenv("BENCHMARK_MODE", "standard").lower()
        try:
            return BenchmarkMode(mode_str)
        except ValueError:
            return BenchmarkMode.STANDARD

    @classmethod
    def get_provider_config(cls, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        return cls.PROVIDER_CONFIGS.get(provider.lower(), {})

    @classmethod
    def get_prompts(cls, category: str, count: Optional[int] = None) -> List[str]:
        """Get prompts for a specific category."""
        prompts = cls.BENCHMARK_PROMPTS.get(category, cls.BENCHMARK_PROMPTS["medium"])
        if count:
            return prompts[:count]
        return prompts

    @classmethod
    def get_token_limits(cls, category: str) -> Dict[str, int]:
        """Get token limits for a category."""
        return cls.TOKEN_LIMITS.get(category, cls.TOKEN_LIMITS["medium"])

    @classmethod
    def get_performance_threshold(cls, metric: str, level: str) -> float:
        """Get performance threshold for a metric and level."""
        thresholds = cls.PERFORMANCE_THRESHOLDS.get(metric, {})
        return thresholds.get(level, 0.0)

    @classmethod
    def should_run_expensive_benchmarks(cls) -> bool:
        """Check if expensive benchmarks should be run."""
        return os.getenv("RUN_EXPENSIVE_BENCHMARKS", "").lower() == "true"

    @classmethod
    def should_run_stress_tests(cls) -> bool:
        """Check if stress tests should be run."""
        return os.getenv("RUN_STRESS_TESTS", "").lower() == "true"

    @classmethod
    def get_output_directory(cls) -> str:
        """Get output directory for benchmark results."""
        return os.getenv("BENCHMARK_OUTPUT_DIR", "benchmark_results")

    @classmethod
    def is_provider_benchmark_enabled(cls, provider: str) -> bool:
        """Check if benchmarks are enabled for a provider."""
        env_var = f"BENCHMARK_{provider.upper()}_ENABLED"
        return os.getenv(env_var, "").lower() == "true"

    @classmethod
    def get_custom_model(cls, provider: str) -> Optional[str]:
        """Get custom model override for a provider."""
        env_var = f"BENCHMARK_MODEL_{provider.upper()}"
        return os.getenv(env_var)

    @classmethod
    def get_max_concurrent_requests(cls, provider: str) -> int:
        """Get maximum concurrent requests for a provider."""
        config = cls.get_provider_config(provider)
        override = os.getenv(f"BENCHMARK_MAX_CONCURRENT_{provider.upper()}")

        if override:
            try:
                return int(override)
            except ValueError:
                pass

        return config.get("max_concurrent", 3)

    @classmethod
    def get_rate_limit_delay(cls, provider: str) -> float:
        """Get rate limit delay for a provider."""
        config = cls.get_provider_config(provider)
        override = os.getenv(f"BENCHMARK_RATE_DELAY_{provider.upper()}")

        if override:
            try:
                return float(override)
            except ValueError:
                pass

        return config.get("recommended_delay", 1.0)


# Environment setup for benchmarks
def setup_benchmark_environment():
    """Setup environment for performance benchmarks."""
    import logging

    # Configure logging for benchmarks
    log_level = logging.INFO
    if os.getenv("BENCHMARK_VERBOSE", "").lower() == "true":
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create output directory
    output_dir = BenchmarkConfig.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)

    # Check benchmark mode
    mode = BenchmarkConfig.get_benchmark_mode()
    config = BenchmarkConfig.get_mode_config(mode)

    print(f"Benchmark Mode: {mode.value}")
    print(f"Configuration: {config}")

    # Check enabled providers
    providers = ["openai", "anthropic", "google"]
    enabled_providers = [p for p in providers if BenchmarkConfig.is_provider_benchmark_enabled(p)]

    if enabled_providers:
        print(f"Enabled providers: {', '.join(enabled_providers)}")
    else:
        print("No providers explicitly enabled - will attempt to benchmark available providers")

    return mode, enabled_providers


# Performance rating helper
def rate_performance(value: float, metric: str, higher_is_better: bool = True) -> str:
    """Rate performance based on thresholds."""
    thresholds = BenchmarkConfig.PERFORMANCE_THRESHOLDS.get(metric, {})

    if not thresholds:
        return "unknown"

    if higher_is_better:
        # For metrics like throughput where higher is better
        if value >= thresholds.get("excellent", float("inf")):
            return "excellent"
        elif value >= thresholds.get("good", float("inf")):
            return "good"
        elif value >= thresholds.get("acceptable", float("inf")):
            return "acceptable"
        else:
            return "poor"
    else:
        # For metrics like response time where lower is better
        if value <= thresholds.get("excellent", 0):
            return "excellent"
        elif value <= thresholds.get("good", 0):
            return "good"
        elif value <= thresholds.get("acceptable", 0):
            return "acceptable"
        else:
            return "poor"
