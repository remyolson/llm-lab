"""
Provider Factory

Factory for creating test provider instances.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock


@dataclass
class ProviderFactory:
    """Factory for creating provider instances with test data."""

    # Available provider types
    PROVIDER_TYPES = ["openai", "anthropic", "google", "cohere", "huggingface"]

    # Model configurations
    MODEL_CONFIGS = {
        "openai": {
            "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            "max_tokens": 4096,
            "supports_streaming": True,
            "supports_functions": True,
        },
        "anthropic": {
            "models": ["claude-2", "claude-3-opus", "claude-3-sonnet"],
            "max_tokens": 100000,
            "supports_streaming": True,
            "supports_functions": False,
        },
        "google": {
            "models": ["gemini-pro", "gemini-pro-vision", "palm-2"],
            "max_tokens": 32768,
            "supports_streaming": True,
            "supports_functions": True,
        },
        "cohere": {
            "models": ["command", "command-light", "command-nightly"],
            "max_tokens": 4096,
            "supports_streaming": False,
            "supports_functions": False,
        },
        "huggingface": {
            "models": ["meta-llama/Llama-2-7b", "mistralai/Mistral-7B"],
            "max_tokens": 2048,
            "supports_streaming": False,
            "supports_functions": False,
        },
    }

    @classmethod
    def create(
        cls, provider_type: Optional[str] = None, model_name: Optional[str] = None, **kwargs
    ) -> Mock:
        """
        Create a mock provider instance.

        Args:
            provider_type: Type of provider (openai, anthropic, etc.)
            model_name: Specific model name
            **kwargs: Additional provider configuration

        Returns:
            Mock provider instance
        """
        # Select random provider type if not specified
        if provider_type is None:
            provider_type = random.choice(cls.PROVIDER_TYPES)

        # Get provider configuration
        config = cls.MODEL_CONFIGS.get(provider_type, cls.MODEL_CONFIGS["openai"])

        # Select model if not specified
        if model_name is None:
            model_name = random.choice(config["models"])

        # Create mock provider
        provider = Mock()
        provider.provider_name = provider_type
        provider.model_name = model_name
        provider.max_tokens = config["max_tokens"]
        provider.supports_streaming = config["supports_streaming"]
        provider.supports_functions = config["supports_functions"]

        # Configure methods
        provider.generate.return_value = cls._generate_response()
        provider.batch_generate.return_value = [cls._generate_response() for _ in range(3)]
        provider.validate_credentials.return_value = True
        provider.get_model_info.return_value = {
            "provider": provider_type,
            "model": model_name,
            "max_tokens": config["max_tokens"],
            "capabilities": cls._get_capabilities(config),
        }

        # Apply additional configuration
        for key, value in kwargs.items():
            setattr(provider, key, value)

        return provider

    @classmethod
    def create_batch(cls, count: int = 3, **kwargs) -> List[Mock]:
        """Create multiple provider instances."""
        return [cls.create(**kwargs) for _ in range(count)]

    @staticmethod
    def _generate_response() -> str:
        """Generate a realistic test response."""
        responses = [
            "This is a test response from the language model.",
            "The answer to your question involves multiple factors that should be considered.",
            "Based on the analysis, here are the key findings:\n1. First point\n2. Second point\n3. Third point",
            "I understand your request. Let me provide a comprehensive response.",
            "The implementation would look something like this:\n```python\ndef example():\n    return 'test'\n```",
        ]
        return random.choice(responses)

    @staticmethod
    def _get_capabilities(config: Dict) -> List[str]:
        """Get provider capabilities."""
        capabilities = ["text_generation"]

        if config.get("supports_streaming"):
            capabilities.append("streaming")

        if config.get("supports_functions"):
            capabilities.append("function_calling")

        return capabilities


class MockProviderFactory:
    """Factory for creating mock providers with specific behaviors."""

    @staticmethod
    def create_failing_provider(exception: Optional[Exception] = None, fail_after: int = 0) -> Mock:
        """
        Create a provider that fails with an exception.

        Args:
            exception: Exception to raise (default: generic Exception)
            fail_after: Number of successful calls before failing

        Returns:
            Mock provider that fails
        """
        provider = Mock()
        provider.provider_name = "failing_provider"
        provider.model_name = "fail-model"

        if exception is None:
            exception = Exception("Provider error")

        # Create a call counter
        call_count = {"count": 0}

        def generate_with_failure(*args, **kwargs):
            call_count["count"] += 1
            if call_count["count"] > fail_after:
                raise exception
            return "Success before failure"

        provider.generate.side_effect = generate_with_failure
        provider.validate_credentials.return_value = False

        return provider

    @staticmethod
    def create_slow_provider(delay_seconds: float = 2.0) -> Mock:
        """
        Create a provider with simulated latency.

        Args:
            delay_seconds: Response delay in seconds

        Returns:
            Mock provider with delay
        """
        import time

        provider = Mock()
        provider.provider_name = "slow_provider"
        provider.model_name = "slow-model"

        def slow_generate(*args, **kwargs):
            time.sleep(delay_seconds)
            return "Slow response"

        provider.generate.side_effect = slow_generate
        provider.validate_credentials.return_value = True

        return provider

    @staticmethod
    def create_rate_limited_provider(limit: int = 5, window_seconds: int = 60) -> Mock:
        """
        Create a provider with rate limiting.

        Args:
            limit: Number of allowed requests
            window_seconds: Time window for rate limit

        Returns:
            Mock provider with rate limiting
        """
        import time

        provider = Mock()
        provider.provider_name = "rate_limited_provider"
        provider.model_name = "limited-model"

        # Track requests
        requests = {"count": 0, "window_start": time.time()}

        def rate_limited_generate(*args, **kwargs):
            current_time = time.time()

            # Reset window if expired
            if current_time - requests["window_start"] > window_seconds:
                requests["count"] = 0
                requests["window_start"] = current_time

            # Check rate limit
            if requests["count"] >= limit:
                raise Exception(
                    f"Rate limit exceeded: {limit} requests per {window_seconds} seconds"
                )

            requests["count"] += 1
            return f"Response {requests['count']}/{limit}"

        provider.generate.side_effect = rate_limited_generate
        provider.validate_credentials.return_value = True

        return provider

    @staticmethod
    def create_streaming_provider() -> Mock:
        """
        Create a provider that supports streaming responses.

        Returns:
            Mock provider with streaming support
        """
        provider = Mock()
        provider.provider_name = "streaming_provider"
        provider.model_name = "stream-model"
        provider.supports_streaming = True

        def stream_generate(*args, stream=False, **kwargs):
            if stream:
                # Return generator for streaming
                def token_generator():
                    tokens = ["This ", "is ", "a ", "streaming ", "response."]
                    for token in tokens:
                        yield token

                return token_generator()
            else:
                return "This is a streaming response."

        provider.generate.side_effect = stream_generate
        provider.validate_credentials.return_value = True

        return provider
