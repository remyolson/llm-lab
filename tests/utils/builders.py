"""
Builder Patterns for Complex Test Objects

Builder classes for constructing complex test objects with fluent interfaces,
supporting incremental construction and validation.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar

from src.types.core import APIResponse, ConfigDict, ModelParameters, ProviderInfo
from src.types.evaluation import EvaluationResult, MetricResult

from .factories import fake

T = TypeVar("T")


class BuilderBase(Generic[T]):
    """Base class for all builders."""

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._validators: List[callable] = []

    def validate(self) -> bool:
        """Validate the current builder state."""
        for validator in self._validators:
            if not validator(self._data):
                return False
        return True

    def reset(self) -> BuilderBase[T]:
        """Reset the builder to initial state."""
        self._data = {}
        return self

    def build(self) -> T:
        """Build the final object."""
        raise NotImplementedError


class ProviderBuilder(BuilderBase[ProviderInfo]):
    """Builder for creating ProviderInfo objects."""

    def with_name(self, name: str) -> ProviderBuilder:
        """Set provider name."""
        self._data["name"] = name
        return self

    def with_model(self, model: str) -> ProviderBuilder:
        """Add a single model."""
        if "models" not in self._data:
            self._data["models"] = []
        self._data["models"].append(model)
        return self

    def with_models(self, models: List[str]) -> ProviderBuilder:
        """Set multiple models."""
        self._data["models"] = models
        return self

    def with_api_key(self, api_key: str) -> ProviderBuilder:
        """Set API key."""
        self._data["api_key"] = api_key
        return self

    def with_base_url(self, base_url: str) -> ProviderBuilder:
        """Set base URL."""
        self._data["base_url"] = base_url
        return self

    def with_timeout(self, timeout: int) -> ProviderBuilder:
        """Set timeout."""
        self._data["timeout"] = timeout
        return self

    def with_retry_config(self, max_retries: int, backoff: float = 2.0) -> ProviderBuilder:
        """Set retry configuration."""
        self._data["max_retries"] = max_retries
        self._data["retry_backoff"] = backoff
        return self

    def with_rate_limit(self, limit: int, window: int = 60) -> ProviderBuilder:
        """Set rate limiting."""
        self._data["rate_limit"] = limit
        self._data["rate_limit_window"] = window
        return self

    def with_features(
        self, streaming: bool = False, functions: bool = False, vision: bool = False
    ) -> ProviderBuilder:
        """Set supported features."""
        self._data["supports_streaming"] = streaming
        self._data["supports_functions"] = functions
        self._data["supports_vision"] = vision
        return self

    def with_headers(self, headers: Dict[str, str]) -> ProviderBuilder:
        """Set custom headers."""
        self._data["headers"] = headers
        return self

    def build(self) -> ProviderInfo:
        """Build the ProviderInfo object."""
        # Set defaults
        defaults = {
            "name": "test-provider",
            "models": ["test-model"],
            "api_key": "test-key",
            "max_retries": 3,
            "timeout": 30,
        }

        for key, value in defaults.items():
            if key not in self._data:
                self._data[key] = value

        return ProviderInfo(self._data)


class ConfigBuilder(BuilderBase[ConfigDict]):
    """Builder for creating ConfigDict objects."""

    def with_provider(self, name: str, api_key: str, model: str, **kwargs) -> ConfigBuilder:
        """Add a provider configuration."""
        if "providers" not in self._data:
            self._data["providers"] = {}

        self._data["providers"][name] = {"api_key": api_key, "model": model, **kwargs}
        return self

    def with_evaluation(
        self, methods: Optional[List[str]] = None, threshold: float = 0.8, sample_size: int = 100
    ) -> ConfigBuilder:
        """Set evaluation configuration."""
        self._data["evaluation"] = {
            "methods": methods or ["semantic_similarity"],
            "threshold": threshold,
            "sample_size": sample_size,
        }
        return self

    def with_monitoring(
        self, enabled: bool = True, log_level: str = "INFO", metrics_interval: int = 60
    ) -> ConfigBuilder:
        """Set monitoring configuration."""
        self._data["monitoring"] = {
            "enabled": enabled,
            "log_level": log_level,
            "metrics_interval": metrics_interval,
        }
        return self

    def with_cache(
        self, enabled: bool = True, ttl: int = 3600, max_size: int = 1000
    ) -> ConfigBuilder:
        """Set cache configuration."""
        self._data["cache"] = {"enabled": enabled, "ttl": ttl, "max_size": max_size}
        return self

    def with_retry(
        self, max_attempts: int = 3, backoff_factor: float = 2.0, max_delay: int = 60
    ) -> ConfigBuilder:
        """Set retry configuration."""
        self._data["retry"] = {
            "max_attempts": max_attempts,
            "backoff_factor": backoff_factor,
            "max_delay": max_delay,
        }
        return self

    def build(self) -> ConfigDict:
        """Build the ConfigDict object."""
        # Ensure at least one provider
        if "providers" not in self._data:
            self._data["providers"] = {"default": {"api_key": "test-key", "model": "test-model"}}

        return ConfigDict(self._data)


class ResponseBuilder(BuilderBase[APIResponse]):
    """Builder for creating APIResponse objects."""

    def with_content(self, content: str) -> ResponseBuilder:
        """Set response content."""
        self._data["content"] = content
        return self

    def with_model(self, model: str) -> ResponseBuilder:
        """Set model name."""
        self._data["model"] = model
        return self

    def with_provider(self, provider: str) -> ResponseBuilder:
        """Set provider name."""
        self._data["provider"] = provider
        return self

    def with_usage(
        self, prompt_tokens: int, completion_tokens: int, total_tokens: Optional[int] = None
    ) -> ResponseBuilder:
        """Set token usage."""
        self._data["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens or (prompt_tokens + completion_tokens),
        }
        return self

    def with_metadata(self, **kwargs) -> ResponseBuilder:
        """Set metadata."""
        if "metadata" not in self._data:
            self._data["metadata"] = {}
        self._data["metadata"].update(kwargs)
        return self

    def with_latency(self, latency: float) -> ResponseBuilder:
        """Set response latency."""
        if "metadata" not in self._data:
            self._data["metadata"] = {}
        self._data["metadata"]["latency"] = latency
        return self

    def with_finish_reason(self, reason: str) -> ResponseBuilder:
        """Set finish reason."""
        self._data["finish_reason"] = reason
        return self

    def as_streaming(self) -> ResponseBuilder:
        """Mark as streaming response."""
        if "metadata" not in self._data:
            self._data["metadata"] = {}
        self._data["metadata"]["streaming"] = True
        return self

    def as_cached(self) -> ResponseBuilder:
        """Mark as cached response."""
        if "metadata" not in self._data:
            self._data["metadata"] = {}
        self._data["metadata"]["cached"] = True
        return self

    def build(self) -> APIResponse:
        """Build the APIResponse object."""
        # Set defaults
        if "content" not in self._data:
            self._data["content"] = "Test response"
        if "model" not in self._data:
            self._data["model"] = "test-model"
        if "provider" not in self._data:
            self._data["provider"] = "test-provider"
        if "usage" not in self._data:
            self._data["usage"] = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        if "finish_reason" not in self._data:
            self._data["finish_reason"] = "stop"
        if "metadata" not in self._data:
            self._data["metadata"] = {}

        # Add timestamp if not present
        if "timestamp" not in self._data["metadata"]:
            self._data["metadata"]["timestamp"] = datetime.now().isoformat()

        return APIResponse(self._data)


class EvaluationBuilder(BuilderBase[EvaluationResult]):
    """Builder for creating EvaluationResult objects."""

    def with_model(self, model: str) -> EvaluationBuilder:
        """Set model name."""
        self._data["model"] = model
        return self

    def with_provider(self, provider: str) -> EvaluationBuilder:
        """Set provider name."""
        self._data["provider"] = provider
        return self

    def with_benchmark(self, benchmark: str) -> EvaluationBuilder:
        """Set benchmark name."""
        self._data["benchmark"] = benchmark
        return self

    def with_result(
        self,
        method: str,
        score: float,
        confidence: Optional[float] = None,
        samples: Optional[int] = None,
    ) -> EvaluationBuilder:
        """Add an evaluation result."""
        if "results" not in self._data:
            self._data["results"] = {}

        self._data["results"][method] = {
            "score": score,
            "confidence": confidence or 0.95,
            "samples": samples or 100,
        }
        return self

    def with_multiple_results(self, results: Dict[str, Dict[str, Any]]) -> EvaluationBuilder:
        """Set multiple results at once."""
        self._data["results"] = results
        return self

    def with_summary(
        self, average_score: float, best_method: str, worst_method: str
    ) -> EvaluationBuilder:
        """Set summary statistics."""
        self._data["summary"] = {
            "average_score": average_score,
            "best_method": best_method,
            "worst_method": worst_method,
        }
        return self

    def with_metadata(self, **kwargs) -> EvaluationBuilder:
        """Set metadata."""
        if "metadata" not in self._data:
            self._data["metadata"] = {}
        self._data["metadata"].update(kwargs)
        return self

    def with_duration(self, duration: float) -> EvaluationBuilder:
        """Set evaluation duration."""
        if "metadata" not in self._data:
            self._data["metadata"] = {}
        self._data["metadata"]["duration"] = duration
        return self

    def build(self) -> EvaluationResult:
        """Build the EvaluationResult object."""
        # Set defaults
        if "model" not in self._data:
            self._data["model"] = "test-model"
        if "provider" not in self._data:
            self._data["provider"] = "test-provider"
        if "benchmark" not in self._data:
            self._data["benchmark"] = "test-benchmark"
        if "results" not in self._data:
            self._data["results"] = {"default": {"score": 0.85, "confidence": 0.95, "samples": 100}}

        # Auto-generate summary if not provided
        if "summary" not in self._data and "results" in self._data:
            scores = [r["score"] for r in self._data["results"].values()]
            self._data["summary"] = {
                "average_score": sum(scores) / len(scores) if scores else 0,
                "best_method": max(
                    self._data["results"], key=lambda k: self._data["results"][k]["score"]
                ),
                "worst_method": min(
                    self._data["results"], key=lambda k: self._data["results"][k]["score"]
                ),
            }

        if "metadata" not in self._data:
            self._data["metadata"] = {}

        # Add timestamp if not present
        if "timestamp" not in self._data["metadata"]:
            self._data["metadata"]["timestamp"] = datetime.now().isoformat()

        return EvaluationResult(self._data)


class TestScenarioBuilder:
    """Builder for creating complex test scenarios."""

    def __init__(self):
        self.providers: List[ProviderInfo] = []
        self.config: Optional[ConfigDict] = None
        self.prompts: List[str] = []
        self.expected_responses: Dict[str, str] = {}
        self.evaluation_criteria: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def with_provider(self, provider: ProviderInfo) -> TestScenarioBuilder:
        """Add a provider to the scenario."""
        self.providers.append(provider)
        return self

    def with_multiple_providers(self, *providers: ProviderInfo) -> TestScenarioBuilder:
        """Add multiple providers."""
        self.providers.extend(providers)
        return self

    def with_config(self, config: ConfigDict) -> TestScenarioBuilder:
        """Set configuration."""
        self.config = config
        return self

    def with_prompt(self, prompt: str) -> TestScenarioBuilder:
        """Add a test prompt."""
        self.prompts.append(prompt)
        return self

    def with_prompts(self, prompts: List[str]) -> TestScenarioBuilder:
        """Add multiple prompts."""
        self.prompts.extend(prompts)
        return self

    def expect_response(self, provider: str, response: str) -> TestScenarioBuilder:
        """Set expected response for a provider."""
        self.expected_responses[provider] = response
        return self

    def with_evaluation_criteria(
        self,
        min_score: float = 0.7,
        max_latency: float = 5000,
        required_methods: Optional[List[str]] = None,
    ) -> TestScenarioBuilder:
        """Set evaluation criteria."""
        self.evaluation_criteria = {
            "min_score": min_score,
            "max_latency": max_latency,
            "required_methods": required_methods or ["semantic_similarity"],
        }
        return self

    def with_metadata(self, **kwargs) -> TestScenarioBuilder:
        """Add metadata."""
        self.metadata.update(kwargs)
        return self

    def build(self) -> Dict[str, Any]:
        """Build the test scenario."""
        # Ensure we have at least one provider
        if not self.providers:
            self.providers = [ProviderBuilder().build()]

        # Ensure we have configuration
        if not self.config:
            config_builder = ConfigBuilder()
            for provider in self.providers:
                config_builder.with_provider(
                    provider["name"],
                    provider.get("api_key", "test-key"),
                    provider.get("models", ["test-model"])[0],
                )
            self.config = config_builder.build()

        # Ensure we have prompts
        if not self.prompts:
            self.prompts = ["Test prompt"]

        return {
            "id": str(uuid.uuid4()),
            "providers": self.providers,
            "config": self.config,
            "prompts": self.prompts,
            "expected_responses": self.expected_responses,
            "evaluation_criteria": self.evaluation_criteria,
            "metadata": self.metadata,
            "created_at": datetime.now().isoformat(),
        }
