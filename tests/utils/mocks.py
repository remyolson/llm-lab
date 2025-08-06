"""
Mock Provider Classes with Configurable Responses

Mock implementations of all provider interfaces with realistic behavior simulation,
supporting success, failure, timeout, and rate limiting scenarios.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional
from unittest.mock import AsyncMock, MagicMock

from src.providers.base import BaseProvider
from src.types.core import APIResponse, ErrorResponse, ModelParameters, ProviderInfo
from src.types.custom import ProviderName

from .base import MockProvider
from .factories import ResponseFactory, fake


@dataclass
class MockResponse:
    """Configurable mock response."""

    content: str
    delay: float = 0.0
    should_fail: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_api_response(self, provider: str, model: str) -> APIResponse:
        """Convert to APIResponse."""
        return APIResponse(
            {
                "content": self.content,
                "model": model,
                "provider": provider,
                "usage": {
                    "prompt_tokens": len(self.content.split()) * 2,
                    "completion_tokens": len(self.content.split()),
                    "total_tokens": len(self.content.split()) * 3,
                },
                "metadata": self.metadata,
                "finish_reason": "stop",
            }
        )

    def to_error_response(self, provider: str) -> ErrorResponse:
        """Convert to ErrorResponse."""
        return ErrorResponse(
            {
                "error": {
                    "message": self.error_message or "Mock error",
                    "type": self.error_type or "mock_error",
                    "code": "mock_error_code",
                },
                "provider": provider,
                "timestamp": datetime.now().isoformat(),
                "request_id": fake.uuid4(),
            }
        )


class MockOpenAIProvider(MockProvider):
    """Mock implementation of OpenAI provider."""

    def __init__(self, model: str = "gpt-4", **kwargs):
        super().__init__()
        self.model = model
        self.provider_name = "openai"
        self.config = kwargs
        self.response_queue = []
        self.response_patterns = {}
        self.rate_limit_remaining = 100
        self.rate_limit_reset = time.time() + 60

        # Response simulation settings
        self.simulate_rate_limit = False
        self.simulate_timeout = False
        self.timeout_after = 30.0
        self.simulate_streaming = True

    def generate(self, prompt: str, **kwargs) -> str | APIResponse:
        """Generate a mock response."""
        self.call_count += 1
        self.last_request = {"prompt": prompt, "kwargs": kwargs}

        # Simulate delay
        if self.delay > 0:
            time.sleep(self.delay)

        # Check for rate limiting
        if self.simulate_rate_limit:
            if self.rate_limit_remaining <= 0:
                if time.time() < self.rate_limit_reset:
                    raise Exception("Rate limit exceeded")
                else:
                    self.rate_limit_remaining = 100
                    self.rate_limit_reset = time.time() + 60
            self.rate_limit_remaining -= 1

        # Check for timeout
        if self.simulate_timeout:
            time.sleep(self.timeout_after)
            raise TimeoutError("Request timed out")

        # Check for configured failure
        if self.should_fail:
            raise Exception(self.failure_message)

        # Check for pattern-based responses
        for pattern, response in self.response_patterns.items():
            if pattern in prompt:
                return response

        # Return queued response if available
        if self.response_queue:
            response = self.response_queue.pop(0)
            if isinstance(response, MockResponse):
                if response.delay > 0:
                    time.sleep(response.delay)
                if response.should_fail:
                    raise Exception(response.error_message or "Mock failure")
                return response.to_api_response(self.provider_name, self.model)
            return response

        # Return from configured responses
        if self.responses:
            response = self.responses[self.call_count % len(self.responses) - 1]
            return response

        # Generate default response
        response_factory = ResponseFactory()
        return response_factory.create(
            content=f"Mock response from {self.model} for: {prompt[:50]}...",
            model=self.model,
            provider=self.provider_name,
        )

    def generate_streaming(self, prompt: str, **kwargs) -> Iterator[str]:
        """Generate a streaming mock response."""
        if not self.simulate_streaming:
            yield self.generate(prompt, **kwargs)
            return

        response = self.generate(prompt, **kwargs)
        if isinstance(response, str):
            words = response.split()
            for word in words:
                yield word + " "
                time.sleep(0.01)  # Simulate streaming delay
        else:
            yield json.dumps(response)

    async def generate_async(self, prompt: str, **kwargs) -> str | APIResponse:
        """Generate an async mock response."""
        await asyncio.sleep(self.delay)
        return self.generate(prompt, **kwargs)

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str | APIResponse]:
        """Generate batch mock responses."""
        return [self.generate(prompt, **kwargs) for prompt in prompts]

    def add_response_pattern(self, pattern: str, response: str | MockResponse):
        """Add a pattern-based response."""
        self.response_patterns[pattern] = response

    def queue_response(self, response: str | MockResponse):
        """Queue a response to be returned."""
        self.response_queue.append(response)

    def set_rate_limit(self, remaining: int, reset_in_seconds: int = 60):
        """Configure rate limiting."""
        self.simulate_rate_limit = True
        self.rate_limit_remaining = remaining
        self.rate_limit_reset = time.time() + reset_in_seconds


class MockAnthropicProvider(MockProvider):
    """Mock implementation of Anthropic provider."""

    def __init__(self, model: str = "claude-3-opus", **kwargs):
        super().__init__()
        self.model = model
        self.provider_name = "anthropic"
        self.config = kwargs
        self.conversation_history = []
        self.system_prompt = None

    def generate(self, prompt: str, **kwargs) -> str | APIResponse:
        """Generate a mock response."""
        self.call_count += 1
        self.last_request = {"prompt": prompt, "kwargs": kwargs}

        # Store conversation history
        self.conversation_history.append({"role": "user", "content": prompt})

        if self.delay > 0:
            time.sleep(self.delay)

        if self.should_fail:
            raise Exception(self.failure_message)

        # Check for system prompt
        if "system" in kwargs:
            self.system_prompt = kwargs["system"]

        # Generate response based on conversation context
        if self.responses:
            response = self.responses[self.call_count % len(self.responses) - 1]
        else:
            context_info = f" (with {len(self.conversation_history)} messages in history)"
            response = f"Claude response{context_info}: {fake.text(max_nb_chars=200)}"

        self.conversation_history.append({"role": "assistant", "content": response})

        if kwargs.get("return_full", False):
            response_factory = ResponseFactory()
            return response_factory.create(
                content=response,
                model=self.model,
                provider=self.provider_name,
                metadata={"conversation_id": fake.uuid4()},
            )

        return response

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.system_prompt = None


class MockGoogleProvider(MockProvider):
    """Mock implementation of Google provider."""

    def __init__(self, model: str = "gemini-pro", **kwargs):
        super().__init__()
        self.model = model
        self.provider_name = "google"
        self.config = kwargs
        self.safety_settings = kwargs.get("safety_settings", {})

    def generate(self, prompt: str, **kwargs) -> str | APIResponse:
        """Generate a mock response."""
        self.call_count += 1
        self.last_request = {"prompt": prompt, "kwargs": kwargs}

        if self.delay > 0:
            time.sleep(self.delay)

        # Simulate safety filtering
        if self.safety_settings.get("block_harmful", False):
            harmful_keywords = ["dangerous", "harmful", "illegal"]
            if any(keyword in prompt.lower() for keyword in harmful_keywords):
                raise Exception("Content blocked by safety filter")

        if self.should_fail:
            raise Exception(self.failure_message)

        if self.responses:
            response = self.responses[self.call_count % len(self.responses) - 1]
        else:
            response = f"Gemini response: {fake.paragraph()}"

        if kwargs.get("return_full", False):
            response_factory = ResponseFactory()
            return response_factory.create(
                content=response,
                model=self.model,
                provider=self.provider_name,
                metadata={"safety_ratings": self._generate_safety_ratings()},
            )

        return response

    def _generate_safety_ratings(self) -> Dict[str, str]:
        """Generate mock safety ratings."""
        categories = ["harassment", "hate_speech", "dangerous_content", "sexually_explicit"]
        ratings = ["NEGLIGIBLE", "LOW", "MEDIUM", "HIGH"]
        return {category: random.choice(ratings) for category in categories}


class MockLocalProvider(MockProvider):
    """Mock implementation of local/custom provider."""

    def __init__(self, model: str = "llama-2-7b", **kwargs):
        super().__init__()
        self.model = model
        self.provider_name = "local"
        self.config = kwargs
        self.model_loaded = False
        self.memory_usage = 0.0

    def load_model(self):
        """Simulate model loading."""
        if not self.model_loaded:
            time.sleep(0.5)  # Simulate loading time
            self.model_loaded = True
            self.memory_usage = random.uniform(2.0, 8.0)  # GB

    def generate(self, prompt: str, **kwargs) -> str | APIResponse:
        """Generate a mock response."""
        if not self.model_loaded:
            self.load_model()

        self.call_count += 1
        self.last_request = {"prompt": prompt, "kwargs": kwargs}

        if self.delay > 0:
            time.sleep(self.delay)

        if self.should_fail:
            raise Exception(self.failure_message)

        # Simulate memory constraints
        if self.memory_usage > 7.0:
            if random.random() > 0.9:
                raise MemoryError("Out of memory")

        if self.responses:
            response = self.responses[self.call_count % len(self.responses) - 1]
        else:
            response = f"Local model response: {fake.sentence()}"

        if kwargs.get("return_full", False):
            response_factory = ResponseFactory()
            return response_factory.create(
                content=response,
                model=self.model,
                provider=self.provider_name,
                metadata={
                    "memory_usage_gb": self.memory_usage,
                    "inference_time": random.uniform(0.1, 2.0),
                },
            )

        return response

    def unload_model(self):
        """Simulate model unloading."""
        self.model_loaded = False
        self.memory_usage = 0.0


class MockEvaluator:
    """Mock evaluator for testing evaluation pipelines."""

    def __init__(self):
        self.call_count = 0
        self.last_evaluation = None
        self.default_score = 0.85

    def evaluate(self, prediction: str, target: str, method: str = "semantic_similarity") -> float:
        """Mock evaluation."""
        self.call_count += 1
        self.last_evaluation = {
            "prediction": prediction,
            "target": target,
            "method": method,
        }

        # Simple mock scoring logic
        if prediction == target:
            return 1.0
        elif method == "fuzzy_match":
            # Simulate fuzzy matching
            common_chars = set(prediction) & set(target)
            return len(common_chars) / max(len(prediction), len(target))
        else:
            return self.default_score

    def batch_evaluate(
        self, predictions: List[str], targets: List[str], method: str = "semantic_similarity"
    ) -> List[float]:
        """Mock batch evaluation."""
        return [self.evaluate(pred, tgt, method) for pred, tgt in zip(predictions, targets)]


class MockLogger:
    """Mock logger for testing logging functionality."""

    def __init__(self):
        self.messages = []
        self.call_count = 0

    def log(self, level: str, message: str, **kwargs):
        """Mock logging."""
        self.call_count += 1
        self.messages.append(
            {
                "level": level,
                "message": message,
                "timestamp": datetime.now(),
                "extra": kwargs,
            }
        )

    def debug(self, message: str, **kwargs):
        """Mock debug logging."""
        self.log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs):
        """Mock info logging."""
        self.log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Mock warning logging."""
        self.log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs):
        """Mock error logging."""
        self.log("ERROR", message, **kwargs)

    def get_messages(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get logged messages, optionally filtered by level."""
        if level:
            return [msg for msg in self.messages if msg["level"] == level]
        return self.messages

    def clear(self):
        """Clear logged messages."""
        self.messages = []
        self.call_count = 0


class MockCache:
    """Mock cache for testing caching functionality."""

    def __init__(self):
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0
        self.set_count = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]["value"]
        else:
            self.miss_count += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        self.set_count += 1
        self.cache[key] = {
            "value": value,
            "ttl": ttl,
            "timestamp": time.time(),
        }

    def delete(self, key: str):
        """Delete value from cache."""
        if key in self.cache:
            del self.cache[key]

    def clear(self):
        """Clear all cache entries."""
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0
        self.set_count = 0

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        return {
            "hits": self.hit_count,
            "misses": self.miss_count,
            "sets": self.set_count,
            "size": len(self.cache),
            "hit_rate": hit_rate,
        }


def create_mock_provider(provider_name: str = "openai", **kwargs) -> MockProvider:
    """Factory function to create mock providers."""
    providers = {
        "openai": MockOpenAIProvider,
        "anthropic": MockAnthropicProvider,
        "google": MockGoogleProvider,
        "local": MockLocalProvider,
    }

    provider_class = providers.get(provider_name, MockOpenAIProvider)
    return provider_class(**kwargs)


class MockDependencyInjector:
    """Mock dependency injector for testing DI system."""

    def __init__(self):
        self.services = {}
        self.instances = {}

    def register(self, name: str, service: Any, singleton: bool = False):
        """Register a service."""
        self.services[name] = {
            "service": service,
            "singleton": singleton,
        }

    def get(self, name: str) -> Any:
        """Get a service instance."""
        if name not in self.services:
            raise KeyError(f"Service {name} not registered")

        service_config = self.services[name]

        if service_config["singleton"]:
            if name not in self.instances:
                self.instances[name] = service_config["service"]()
            return self.instances[name]
        else:
            return service_config["service"]()

    def clear(self):
        """Clear all services and instances."""
        self.services = {}
        self.instances = {}
