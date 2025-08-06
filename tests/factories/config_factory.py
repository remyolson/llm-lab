"""
Configuration Factory

Factory for creating test configuration objects.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ConfigFactory:
    """Factory for creating configuration objects."""

    # Default configuration templates
    TEMPLATES = {
        "minimal": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
        },
        "standard": {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9,
            "timeout": 30,
            "max_retries": 3,
        },
        "conservative": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.2,
            "max_tokens": 500,
            "top_p": 0.5,
            "timeout": 15,
            "max_retries": 2,
        },
        "aggressive": {
            "model": "gpt-4",
            "temperature": 1.5,
            "max_tokens": 2000,
            "top_p": 0.95,
            "timeout": 60,
            "max_retries": 5,
        },
        "experimental": {
            "model": "gpt-4-turbo",
            "temperature": 0.8,
            "max_tokens": 4000,
            "top_p": 0.9,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
            "timeout": 120,
            "max_retries": 3,
        },
    }

    @classmethod
    def create(cls, template: Optional[str] = None, **overrides) -> Dict[str, Any]:
        """
        Create a configuration dictionary.

        Args:
            template: Name of template to use
            **overrides: Values to override in template

        Returns:
            Configuration dictionary
        """
        if template is None:
            template = random.choice(list(cls.TEMPLATES.keys()))

        config = cls.TEMPLATES.get(template, cls.TEMPLATES["standard"]).copy()
        config.update(overrides)

        return config

    @classmethod
    def create_batch(cls, count: int = 3, template: Optional[str] = None) -> List[Dict[str, Any]]:
        """Create multiple configuration objects."""
        return [cls.create(template) for _ in range(count)]

    @classmethod
    def create_invalid(cls, error_type: str = "temperature") -> Dict[str, Any]:
        """
        Create an invalid configuration for error testing.

        Args:
            error_type: Type of validation error to create

        Returns:
            Invalid configuration dictionary
        """
        invalid_configs = {
            "temperature": {"temperature": 3.0},  # Out of range
            "max_tokens": {"max_tokens": -100},  # Negative value
            "top_p": {"top_p": 1.5},  # Out of range
            "timeout": {"timeout": 0},  # Zero timeout
            "model": {"model": None},  # Missing model
            "type_error": {"temperature": "not_a_number"},  # Wrong type
        }

        base_config = cls.create("minimal")
        base_config.update(invalid_configs.get(error_type, invalid_configs["temperature"]))

        return base_config


class ProviderConfigFactory:
    """Factory for provider-specific configurations."""

    @staticmethod
    def create_openai_config(**overrides) -> Dict[str, Any]:
        """Create OpenAI provider configuration."""
        config = {
            "api_key": "sk-test-key",
            "organization": "org-test",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "timeout": 30,
        }
        config.update(overrides)
        return config

    @staticmethod
    def create_anthropic_config(**overrides) -> Dict[str, Any]:
        """Create Anthropic provider configuration."""
        config = {
            "api_key": "claude-test-key",
            "model": "claude-3-opus",
            "temperature": 0.7,
            "max_tokens": 100000,
            "top_p": 0.9,
            "top_k": 40,
            "timeout": 60,
        }
        config.update(overrides)
        return config

    @staticmethod
    def create_google_config(**overrides) -> Dict[str, Any]:
        """Create Google provider configuration."""
        config = {
            "api_key": "gemini-test-key",
            "model": "gemini-pro",
            "temperature": 0.7,
            "max_tokens": 32768,
            "top_p": 0.9,
            "top_k": 40,
            "timeout": 45,
        }
        config.update(overrides)
        return config

    @staticmethod
    def create_local_model_config(**overrides) -> Dict[str, Any]:
        """Create local model configuration."""
        config = {
            "model_path": "/models/llama-2-7b",
            "model": "llama-2-7b",
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.9,
            "top_k": 40,
            "device": "cpu",
            "threads": 4,
            "batch_size": 8,
        }
        config.update(overrides)
        return config


class ExperimentConfigFactory:
    """Factory for experiment and benchmark configurations."""

    @staticmethod
    def create_benchmark_config(benchmark_type: str = "standard", **overrides) -> Dict[str, Any]:
        """
        Create benchmark configuration.

        Args:
            benchmark_type: Type of benchmark
            **overrides: Configuration overrides

        Returns:
            Benchmark configuration dictionary
        """
        configs = {
            "standard": {
                "name": "Standard Benchmark",
                "iterations": 100,
                "warmup_iterations": 10,
                "models": ["gpt-3.5-turbo", "gpt-4"],
                "datasets": ["truthfulness", "reasoning"],
                "metrics": ["accuracy", "latency", "cost"],
                "timeout": 300,
                "parallel": True,
            },
            "quick": {
                "name": "Quick Test",
                "iterations": 10,
                "warmup_iterations": 2,
                "models": ["gpt-3.5-turbo"],
                "datasets": ["truthfulness"],
                "metrics": ["accuracy"],
                "timeout": 60,
                "parallel": False,
            },
            "comprehensive": {
                "name": "Comprehensive Benchmark",
                "iterations": 500,
                "warmup_iterations": 50,
                "models": ["gpt-3.5-turbo", "gpt-4", "claude-3", "gemini-pro"],
                "datasets": ["truthfulness", "reasoning", "coding", "translation"],
                "metrics": ["accuracy", "latency", "cost", "consistency", "toxicity"],
                "timeout": 3600,
                "parallel": True,
            },
        }

        config = configs.get(benchmark_type, configs["standard"]).copy()
        config.update(overrides)

        return config

    @staticmethod
    def create_evaluation_config(eval_type: str = "accuracy", **overrides) -> Dict[str, Any]:
        """
        Create evaluation configuration.

        Args:
            eval_type: Type of evaluation
            **overrides: Configuration overrides

        Returns:
            Evaluation configuration dictionary
        """
        configs = {
            "accuracy": {
                "type": "accuracy",
                "threshold": 0.8,
                "metrics": ["exact_match", "fuzzy_match", "semantic_similarity"],
                "confidence_level": 0.95,
            },
            "performance": {
                "type": "performance",
                "max_latency_ms": 1000,
                "min_throughput": 10,
                "percentiles": [50, 90, 95, 99],
            },
            "cost": {
                "type": "cost",
                "max_cost_per_1k": 0.02,
                "currency": "USD",
                "include_prompt_tokens": True,
            },
            "safety": {
                "type": "safety",
                "toxicity_threshold": 0.1,
                "bias_threshold": 0.2,
                "check_pii": True,
                "check_harmful_content": True,
            },
        }

        config = configs.get(eval_type, configs["accuracy"]).copy()
        config.update(overrides)

        return config

    @staticmethod
    def create_monitoring_config(**overrides) -> Dict[str, Any]:
        """Create monitoring configuration."""
        config = {
            "enabled": True,
            "metrics_interval": 60,  # seconds
            "log_level": "INFO",
            "export_format": "json",
            "alerts": {
                "error_rate_threshold": 0.05,
                "latency_p99_threshold": 5000,  # ms
                "cost_threshold": 100.0,  # USD
            },
            "storage": {
                "type": "sqlite",
                "path": ":memory:",
                "retention_days": 30,
            },
            "notifications": {
                "enabled": False,
                "channels": ["email", "slack"],
            },
        }
        config.update(overrides)
        return config
