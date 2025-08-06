"""
Test Data Factories with Faker Integration

Comprehensive data factories for generating realistic test data for all domain models,
including providers, configurations, responses, and evaluation results.
"""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from faker import Faker
from faker.providers import BaseProvider

from src.types.core import (
    APIResponse,
    ConfigDict,
    ErrorResponse,
    ModelParameters,
    ProviderInfo,
)
from src.types.custom import (
    MaxTokens,
    ModelName,
    ProviderName,
    Temperature,
    TopP,
)
from src.types.evaluation import (
    BenchmarkResult,
    EvaluationResult,
    MethodResult,
    MetricResult,
)

from .base import TestFactory

# Initialize Faker with seed for reproducibility in tests
fake = Faker()
Faker.seed(42)


class LLMProvider(BaseProvider):
    """Custom Faker provider for LLM-specific data."""

    def provider_name(self) -> str:
        """Generate a provider name."""
        return random.choice(["openai", "anthropic", "google", "local", "custom"])

    def model_name(self) -> str:
        """Generate a model name."""
        models = {
            "openai": ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo-preview"],
            "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-2.1"],
            "google": ["gemini-pro", "gemini-ultra", "palm-2"],
            "local": ["llama-2-7b", "mistral-7b", "phi-2"],
        }
        provider = self.provider_name()
        return random.choice(models.get(provider, ["custom-model"]))

    def api_key(self) -> str:
        """Generate a fake API key."""
        prefix = random.choice(["sk-", "key-", "api-", "token-"])
        return prefix + fake.sha256()[:48]

    def prompt(self) -> str:
        """Generate a test prompt."""
        templates = [
            "Explain {topic} in simple terms.",
            "What are the benefits of {topic}?",
            "How does {topic} work?",
            "Compare {topic1} and {topic2}.",
            "Write a short story about {topic}.",
            "Summarize the following text: {text}",
        ]
        template = random.choice(templates)
        return template.format(
            topic=fake.word(),
            topic1=fake.word(),
            topic2=fake.word(),
            text=fake.text(max_nb_chars=200),
        )

    def llm_response(self) -> str:
        """Generate a fake LLM response."""
        responses = [
            fake.text(max_nb_chars=500),
            fake.paragraph(nb_sentences=5),
            f"Based on my analysis, {fake.sentence()}",
            f"The answer is {fake.word()}. {fake.paragraph()}",
            json.dumps({"result": fake.word(), "confidence": random.random()}),
        ]
        return random.choice(responses)

    def temperature(self) -> float:
        """Generate a temperature value."""
        return round(random.uniform(0.0, 2.0), 2)

    def max_tokens(self) -> int:
        """Generate max tokens value."""
        return random.choice([100, 250, 500, 1000, 2000, 4000])

    def top_p(self) -> float:
        """Generate top_p value."""
        return round(random.uniform(0.1, 1.0), 2)


# Add custom provider to Faker
fake.add_provider(LLMProvider)


class ProviderFactory(TestFactory[ProviderInfo]):
    """Factory for creating ProviderInfo test data."""

    def create(self, **kwargs) -> ProviderInfo:
        """Create a ProviderInfo instance with optional overrides."""
        data = {
            "name": fake.provider_name(),
            "models": [fake.model_name() for _ in range(random.randint(1, 5))],
            "api_key": fake.api_key(),
            "base_url": fake.url(),
            "max_retries": random.randint(1, 5),
            "timeout": random.randint(10, 120),
            "organization": fake.company(),
            "headers": {
                "User-Agent": fake.user_agent(),
                "X-Request-ID": str(uuid.uuid4()),
            },
            "supports_streaming": random.choice([True, False]),
            "supports_functions": random.choice([True, False]),
            "rate_limit": random.randint(10, 1000),
        }
        data.update(kwargs)
        return ProviderInfo(data)

    def create_batch(self, count: int, **kwargs) -> List[ProviderInfo]:
        """Create multiple ProviderInfo instances."""
        return [self.create(**kwargs) for _ in range(count)]

    def create_valid(self) -> ProviderInfo:
        """Create a valid ProviderInfo instance."""
        return self.create(
            name="openai",
            models=["gpt-4", "gpt-3.5-turbo"],
            api_key=fake.api_key(),
            supports_streaming=True,
            supports_functions=True,
        )

    def create_invalid(self) -> ProviderInfo:
        """Create an invalid ProviderInfo instance for error testing."""
        return self.create(
            name="",  # Invalid empty name
            models=[],  # No models
            api_key="invalid-key",  # Invalid API key format
            timeout=-1,  # Invalid negative timeout
            rate_limit=0,  # Invalid zero rate limit
        )

    def create_edge_case(self) -> ProviderInfo:
        """Create an edge case ProviderInfo instance."""
        return self.create(
            name="custom-provider-with-very-long-name-" + "x" * 100,
            models=[f"model-{i}" for i in range(100)],  # Many models
            timeout=1,  # Very short timeout
            rate_limit=1,  # Minimal rate limit
            headers={f"Header-{i}": fake.uuid4() for i in range(50)},  # Many headers
        )


class ConfigFactory(TestFactory[ConfigDict]):
    """Factory for creating configuration test data."""

    def create(self, **kwargs) -> ConfigDict:
        """Create a ConfigDict instance with optional overrides."""
        data = {
            "providers": {
                fake.provider_name(): {
                    "api_key": fake.api_key(),
                    "model": fake.model_name(),
                    "temperature": fake.temperature(),
                    "max_tokens": fake.max_tokens(),
                }
                for _ in range(random.randint(1, 3))
            },
            "evaluation": {
                "methods": ["semantic_similarity", "exact_match", "fuzzy_match"],
                "threshold": round(random.uniform(0.5, 0.95), 2),
                "sample_size": random.randint(10, 1000),
            },
            "monitoring": {
                "enabled": random.choice([True, False]),
                "log_level": random.choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
                "metrics_interval": random.randint(10, 300),
            },
            "cache": {
                "enabled": random.choice([True, False]),
                "ttl": random.randint(60, 3600),
                "max_size": random.randint(100, 10000),
            },
            "retry": {
                "max_attempts": random.randint(1, 5),
                "backoff_factor": round(random.uniform(1.0, 3.0), 1),
                "max_delay": random.randint(10, 60),
            },
        }
        data.update(kwargs)
        return ConfigDict(data)

    def create_batch(self, count: int, **kwargs) -> List[ConfigDict]:
        """Create multiple ConfigDict instances."""
        return [self.create(**kwargs) for _ in range(count)]

    def create_valid(self) -> ConfigDict:
        """Create a valid ConfigDict instance."""
        return self.create(
            providers={
                "openai": {
                    "api_key": fake.api_key(),
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                }
            },
            monitoring={"enabled": True, "log_level": "INFO"},
            cache={"enabled": True, "ttl": 3600},
        )

    def create_invalid(self) -> ConfigDict:
        """Create an invalid ConfigDict instance."""
        return self.create(
            providers={},  # No providers configured
            evaluation={"threshold": 2.0},  # Invalid threshold > 1
            monitoring={"log_level": "INVALID"},  # Invalid log level
            retry={"max_attempts": -1},  # Negative attempts
        )

    def create_edge_case(self) -> ConfigDict:
        """Create an edge case ConfigDict instance."""
        return self.create(
            providers={f"provider_{i}": {} for i in range(50)},  # Many providers
            evaluation={"sample_size": 1000000},  # Very large sample
            cache={"max_size": 1},  # Minimal cache
            retry={"max_attempts": 100, "max_delay": 3600},  # Extreme retry
        )


class ResponseFactory(TestFactory[APIResponse]):
    """Factory for creating API response test data."""

    def create(self, **kwargs) -> APIResponse:
        """Create an APIResponse instance with optional overrides."""
        data = {
            "content": fake.llm_response(),
            "model": fake.model_name(),
            "provider": fake.provider_name(),
            "usage": {
                "prompt_tokens": random.randint(10, 500),
                "completion_tokens": random.randint(10, 1000),
                "total_tokens": random.randint(20, 1500),
            },
            "metadata": {
                "request_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "latency": round(random.uniform(0.1, 5.0), 3),
                "cached": random.choice([True, False]),
            },
            "finish_reason": random.choice(["stop", "length", "content_filter"]),
        }
        data.update(kwargs)
        return APIResponse(data)

    def create_batch(self, count: int, **kwargs) -> List[APIResponse]:
        """Create multiple APIResponse instances."""
        return [self.create(**kwargs) for _ in range(count)]

    def create_valid(self) -> APIResponse:
        """Create a valid APIResponse instance."""
        return self.create(
            content="This is a valid response from the model.",
            finish_reason="stop",
            usage={"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
        )

    def create_invalid(self) -> APIResponse:
        """Create an invalid APIResponse instance."""
        return self.create(
            content="",  # Empty content
            usage={},  # Missing usage data
            finish_reason="unknown",  # Invalid finish reason
        )

    def create_edge_case(self) -> APIResponse:
        """Create an edge case APIResponse instance."""
        return self.create(
            content="x" * 10000,  # Very long response
            usage={
                "prompt_tokens": 8000,
                "completion_tokens": 8000,
                "total_tokens": 16000,  # Near token limit
            },
            metadata={"latency": 30.0},  # High latency
        )

    def create_error_response(self, **kwargs) -> ErrorResponse:
        """Create an ErrorResponse instance."""
        data = {
            "error": {
                "message": fake.sentence(),
                "type": random.choice(
                    ["invalid_request", "authentication", "rate_limit", "server_error"]
                ),
                "code": random.choice(["invalid_api_key", "model_not_found", "quota_exceeded"]),
            },
            "provider": fake.provider_name(),
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4()),
        }
        data.update(kwargs)
        return ErrorResponse(data)


class ModelParametersFactory(TestFactory[ModelParameters]):
    """Factory for creating ModelParameters test data."""

    def create(self, **kwargs) -> ModelParameters:
        """Create a ModelParameters instance with optional overrides."""
        data = {
            "temperature": fake.temperature(),
            "max_tokens": fake.max_tokens(),
            "top_p": fake.top_p(),
            "top_k": random.randint(1, 100),
            "frequency_penalty": round(random.uniform(-2.0, 2.0), 2),
            "presence_penalty": round(random.uniform(-2.0, 2.0), 2),
            "stop": [fake.word() for _ in range(random.randint(0, 4))],
            "seed": random.randint(0, 1000000),
            "response_format": random.choice([None, {"type": "json_object"}]),
            "tools": None
            if random.random() > 0.5
            else [
                {
                    "type": "function",
                    "function": {
                        "name": fake.word(),
                        "description": fake.sentence(),
                        "parameters": {},
                    },
                }
            ],
        }
        data.update(kwargs)
        return ModelParameters(data)

    def create_batch(self, count: int, **kwargs) -> List[ModelParameters]:
        """Create multiple ModelParameters instances."""
        return [self.create(**kwargs) for _ in range(count)]

    def create_valid(self) -> ModelParameters:
        """Create valid ModelParameters."""
        return self.create(
            temperature=0.7,
            max_tokens=1000,
            top_p=0.95,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

    def create_invalid(self) -> ModelParameters:
        """Create invalid ModelParameters."""
        return self.create(
            temperature=3.0,  # Too high
            max_tokens=-100,  # Negative
            top_p=1.5,  # > 1.0
            frequency_penalty=5.0,  # Out of range
        )

    def create_edge_case(self) -> ModelParameters:
        """Create edge case ModelParameters."""
        return self.create(
            temperature=0.0,  # Deterministic
            max_tokens=1,  # Minimal tokens
            top_p=0.01,  # Very selective
            stop=[""] * 20,  # Many stop sequences
        )


class EvaluationResultFactory(TestFactory[EvaluationResult]):
    """Factory for creating EvaluationResult test data."""

    def create(self, **kwargs) -> EvaluationResult:
        """Create an EvaluationResult instance with optional overrides."""
        methods = ["semantic_similarity", "exact_match", "fuzzy_match", "bleu", "rouge"]
        data = {
            "model": fake.model_name(),
            "provider": fake.provider_name(),
            "benchmark": random.choice(["truthfulness", "reasoning", "creativity", "safety"]),
            "results": {
                method: {
                    "score": round(random.uniform(0.0, 1.0), 3),
                    "confidence": round(random.uniform(0.8, 1.0), 3),
                    "samples": random.randint(10, 1000),
                }
                for method in random.sample(methods, k=random.randint(1, len(methods)))
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "duration": round(random.uniform(1.0, 300.0), 2),
                "version": f"v{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
                "config": {"temperature": fake.temperature(), "max_tokens": fake.max_tokens()},
            },
            "summary": {
                "average_score": round(random.uniform(0.0, 1.0), 3),
                "best_method": random.choice(methods),
                "worst_method": random.choice(methods),
            },
        }
        data.update(kwargs)
        return EvaluationResult(data)

    def create_batch(self, count: int, **kwargs) -> List[EvaluationResult]:
        """Create multiple EvaluationResult instances."""
        return [self.create(**kwargs) for _ in range(count)]

    def create_valid(self) -> EvaluationResult:
        """Create a valid EvaluationResult."""
        return self.create(
            results={
                "semantic_similarity": {"score": 0.85, "confidence": 0.95, "samples": 100},
                "exact_match": {"score": 0.72, "confidence": 0.98, "samples": 100},
            },
            summary={
                "average_score": 0.785,
                "best_method": "semantic_similarity",
                "worst_method": "exact_match",
            },
        )

    def create_invalid(self) -> EvaluationResult:
        """Create an invalid EvaluationResult."""
        return self.create(
            results={},  # No results
            summary={"average_score": 1.5},  # Invalid score > 1
        )

    def create_edge_case(self) -> EvaluationResult:
        """Create an edge case EvaluationResult."""
        return self.create(
            results={f"method_{i}": {"score": i / 100} for i in range(100)},  # Many methods
            metadata={"duration": 86400.0},  # 24 hours duration
        )


class MetricFactory(TestFactory[MetricResult]):
    """Factory for creating MetricResult test data."""

    def create(self, **kwargs) -> MetricResult:
        """Create a MetricResult instance with optional overrides."""
        data = {
            "name": random.choice(["accuracy", "latency", "cost", "quality", "safety"]),
            "value": round(random.uniform(0.0, 100.0), 2),
            "unit": random.choice(["percentage", "milliseconds", "dollars", "score", None]),
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "model": fake.model_name(),
                "provider": fake.provider_name(),
                "sample_size": random.randint(10, 1000),
            },
            "confidence_interval": (
                round(random.uniform(0.0, 50.0), 2),
                round(random.uniform(50.0, 100.0), 2),
            )
            if random.random() > 0.5
            else None,
        }
        data.update(kwargs)
        return MetricResult(data)

    def create_batch(self, count: int, **kwargs) -> List[MetricResult]:
        """Create multiple MetricResult instances."""
        return [self.create(**kwargs) for _ in range(count)]

    def create_valid(self) -> MetricResult:
        """Create a valid MetricResult."""
        return self.create(
            name="accuracy",
            value=85.5,
            unit="percentage",
            confidence_interval=(83.2, 87.8),
        )

    def create_invalid(self) -> MetricResult:
        """Create an invalid MetricResult."""
        return self.create(
            name="",  # Empty name
            value=-10.0,  # Negative value for percentage
            unit="invalid_unit",
        )

    def create_edge_case(self) -> MetricResult:
        """Create an edge case MetricResult."""
        return self.create(
            value=0.0000001,  # Very small value
            confidence_interval=(0.0, 0.0000002),  # Tiny interval
            metadata={f"key_{i}": fake.word() for i in range(100)},  # Large metadata
        )


class BenchmarkDataFactory(TestFactory[Dict[str, Any]]):
    """Factory for creating benchmark test data."""

    def create(self, **kwargs) -> Dict[str, Any]:
        """Create benchmark data with optional overrides."""
        question_types = ["multiple_choice", "true_false", "open_ended", "completion"]
        data = {
            "id": str(uuid.uuid4()),
            "question": fake.sentence() + "?",
            "type": random.choice(question_types),
            "answer": fake.sentence()
            if random.random() > 0.5
            else random.choice(["A", "B", "C", "D"]),
            "options": [fake.sentence() for _ in range(4)] if random.random() > 0.5 else None,
            "context": fake.text(max_nb_chars=500) if random.random() > 0.5 else None,
            "category": random.choice(["reasoning", "knowledge", "creativity", "ethics"]),
            "difficulty": random.choice(["easy", "medium", "hard"]),
            "metadata": {
                "source": fake.url(),
                "created_at": fake.date_time().isoformat(),
                "tags": [fake.word() for _ in range(random.randint(1, 5))],
            },
        }
        data.update(kwargs)
        return data

    def create_batch(self, count: int, **kwargs) -> List[Dict[str, Any]]:
        """Create multiple benchmark data instances."""
        return [self.create(**kwargs) for _ in range(count)]

    def create_valid(self) -> Dict[str, Any]:
        """Create valid benchmark data."""
        return self.create(
            question="What is the capital of France?",
            type="multiple_choice",
            answer="Paris",
            options=["London", "Paris", "Berlin", "Madrid"],
            category="knowledge",
            difficulty="easy",
        )

    def create_invalid(self) -> Dict[str, Any]:
        """Create invalid benchmark data."""
        return self.create(
            question="",  # Empty question
            type="invalid_type",
            answer=None,  # No answer
            difficulty="invalid",
        )

    def create_edge_case(self) -> Dict[str, Any]:
        """Create edge case benchmark data."""
        return self.create(
            question="?" * 1000,  # Very long question
            options=[""] * 100,  # Many empty options
            context="x" * 10000,  # Very long context
            metadata={"nested": {"deep": {"structure": {"value": "test"}}}},
        )
