"""
OpenAI Provider Integration Tests

These tests make real API calls to OpenAI to verify the provider implementation
works correctly with the actual OpenAI API.

To run these tests:
    TEST_OPENAI_INTEGRATION=true pytest tests/integration/test_openai_integration.py

Required environment variables:
    - OPENAI_API_KEY: Valid OpenAI API key
    - TEST_OPENAI_INTEGRATION: Set to 'true' to enable tests
"""

import os
import time
from typing import List

import pytest

from llm_providers import OpenAIProvider
from src.providers.exceptions import (
    InvalidCredentialsError,
    ModelNotSupportedError,
    ProviderError,
    RateLimitError,
)

from .test_config import IntegrationTestConfig

# Skip all tests if not enabled
pytestmark = pytest.mark.skipif(
    not IntegrationTestConfig.is_provider_enabled("openai"),
    reason="OpenAI integration tests not enabled. Set TEST_OPENAI_INTEGRATION=true to enable.",
)


class TestOpenAIIntegration:
    """Integration tests for OpenAI provider."""

    @pytest.fixture(scope="class")
    def provider(self):
        """Create OpenAI provider instance."""
        if not IntegrationTestConfig.has_valid_api_key("openai"):
            pytest.skip("Valid OpenAI API key not found")

        model = IntegrationTestConfig.get_test_model("openai")
        provider = OpenAIProvider(model_name=model)
        provider.initialize()
        return provider

    @pytest.fixture(scope="class")
    def rate_limiter(self):
        """Rate limiter to avoid hitting API limits."""
        last_call = [0]
        config = IntegrationTestConfig.get_provider_config("openai")
        delay = config.get("rate_limit_delay", 1.0)

        def rate_limit():
            current_time = time.time()
            time_since_last = current_time - last_call[0]
            if time_since_last < delay:
                time.sleep(delay - time_since_last)
            last_call[0] = time.time()

        return rate_limit

    def test_basic_generation(self, provider, rate_limiter):
        """Test basic text generation."""
        rate_limiter()

        response = provider.generate("What is 2 + 2?", max_tokens=10)

        assert isinstance(response, str)
        assert len(response) > 0
        assert "4" in response

    def test_generation_with_parameters(self, provider, rate_limiter):
        """Test generation with various parameters."""
        rate_limiter()

        response = provider.generate(
            "Write a very short story about a cat.", temperature=0.7, max_tokens=50, top_p=0.9
        )

        assert isinstance(response, str)
        assert len(response) > 0
        assert len(response.split()) <= 60  # Rough token limit check

    def test_system_prompt_support(self, provider, rate_limiter):
        """Test system prompt functionality."""
        rate_limiter()

        response = provider.generate(
            "What's your favorite color?",
            system_prompt="You are a helpful assistant who always says your favorite color is blue.",
            max_tokens=20,
        )

        assert isinstance(response, str)
        assert len(response) > 0
        # Check if system prompt influenced the response
        assert "blue" in response.lower() or "Blue" in response

    def test_longer_generation(self, provider, rate_limiter):
        """Test longer text generation."""
        if not IntegrationTestConfig.should_run_expensive_tests():
            pytest.skip("Expensive tests not enabled")

        rate_limiter()

        response = provider.generate(
            "Explain the concept of machine learning in simple terms.", max_tokens=200
        )

        assert isinstance(response, str)
        assert len(response) > 100  # Should be a substantial response
        assert "machine learning" in response.lower()

    def test_multiple_requests(self, provider, rate_limiter):
        """Test multiple sequential requests."""
        prompts = ["What is 1 + 1?", "What is the capital of France?", "Name a color."]

        responses = []
        for prompt in prompts:
            rate_limiter()
            response = provider.generate(prompt, max_tokens=20)
            responses.append(response)

        assert len(responses) == 3
        for response in responses:
            assert isinstance(response, str)
            assert len(response) > 0

        # Check that responses are different (not cached)
        assert responses[0] != responses[1] or responses[1] != responses[2]

    def test_temperature_effects(self, provider, rate_limiter):
        """Test that temperature affects response variability."""
        if not IntegrationTestConfig.should_run_expensive_tests():
            pytest.skip("Expensive tests not enabled")

        prompt = "Complete this sentence: The weather today is"

        # Low temperature (more deterministic)
        rate_limiter()
        response_low = provider.generate(prompt, temperature=0.1, max_tokens=10)

        # High temperature (more random)
        rate_limiter()
        response_high = provider.generate(prompt, temperature=0.9, max_tokens=10)

        assert isinstance(response_low, str)
        assert isinstance(response_high, str)
        assert len(response_low) > 0
        assert len(response_high) > 0

    def test_max_tokens_limit(self, provider, rate_limiter):
        """Test that max_tokens parameter is respected."""
        rate_limiter()

        # Request a very short response
        response = provider.generate(
            "Write a long essay about artificial intelligence.", max_tokens=5
        )

        assert isinstance(response, str)
        assert len(response) > 0
        # Response should be short due to token limit
        assert len(response.split()) <= 10  # Rough check

    def test_empty_prompt_rejection(self, provider):
        """Test that empty prompts are rejected."""
        with pytest.raises(ProviderError) as exc_info:
            provider.generate("")

        assert "empty" in str(exc_info.value).lower() or "prompt" in str(exc_info.value).lower()

    def test_invalid_parameters(self, provider):
        """Test validation of invalid parameters."""
        # Invalid temperature
        with pytest.raises(ProviderError) as exc_info:
            provider.generate("test", temperature=3.0)

        assert "temperature" in str(exc_info.value).lower()

        # Invalid max_tokens
        with pytest.raises(ProviderError) as exc_info:
            provider.generate("test", max_tokens=-1)

        assert "max_tokens" in str(exc_info.value).lower()

    def test_model_info(self, provider):
        """Test getting model information."""
        model_info = provider.get_model_info()

        assert isinstance(model_info, dict)
        assert "model_name" in model_info
        assert "provider" in model_info
        assert model_info["provider"].lower() == "openai"

    @pytest.mark.slow_integration
    def test_concurrent_requests(self, provider):
        """Test concurrent request handling."""
        if not IntegrationTestConfig.should_run_slow_tests():
            pytest.skip("Slow tests not enabled")

        import concurrent.futures

        def make_request(prompt_id):
            # Each request gets a small delay to avoid rate limits
            time.sleep(prompt_id * 0.5)
            return provider.generate(f"What is {prompt_id} + 1?", max_tokens=10)

        # Make 3 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request, i) for i in range(3)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]

        assert len(responses) == 3
        for response in responses:
            assert isinstance(response, str)
            assert len(response) > 0


class TestOpenAIModels:
    """Test different OpenAI models."""

    def test_gpt_35_turbo(self):
        """Test GPT-3.5-turbo specifically."""
        if not IntegrationTestConfig.is_provider_enabled("openai"):
            pytest.skip("OpenAI integration tests not enabled")

        if not IntegrationTestConfig.has_valid_api_key("openai"):
            pytest.skip("Valid OpenAI API key not found")

        provider = OpenAIProvider(model_name="gpt-3.5-turbo")
        provider.initialize()

        response = provider.generate("Say hello in exactly 3 words.", max_tokens=10)

        assert isinstance(response, str)
        assert len(response) > 0
        # Should be approximately 3 words
        word_count = len(response.split())
        assert 2 <= word_count <= 5  # Allow some flexibility

    def test_gpt_4o_mini(self):
        """Test GPT-4o-mini if available."""
        if not IntegrationTestConfig.is_provider_enabled("openai"):
            pytest.skip("OpenAI integration tests not enabled")

        if not IntegrationTestConfig.has_valid_api_key("openai"):
            pytest.skip("Valid OpenAI API key not found")

        try:
            provider = OpenAIProvider(model_name="gpt-4o-mini")
            provider.initialize()

            response = provider.generate("What is the square root of 16?", max_tokens=10)

            assert isinstance(response, str)
            assert len(response) > 0
            assert "4" in response

        except ModelNotSupportedError:
            pytest.skip("GPT-4o-mini model not available")

    def test_invalid_model(self):
        """Test that invalid models are rejected."""
        with pytest.raises(ModelNotSupportedError):
            OpenAIProvider(model_name="gpt-invalid-model")


class TestOpenAIErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_api_key(self):
        """Test handling of invalid API key."""
        # Temporarily override API key
        original_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-invalid-key-12345"

        try:
            provider = OpenAIProvider(model_name="gpt-3.5-turbo")

            with pytest.raises(InvalidCredentialsError):
                provider.initialize()

        finally:
            # Restore original key
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)

    def test_missing_api_key(self):
        """Test handling of missing API key."""
        original_key = os.environ.get("OPENAI_API_KEY")

        # Remove API key
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        try:
            provider = OpenAIProvider(model_name="gpt-3.5-turbo")

            with pytest.raises(InvalidCredentialsError):
                provider.validate_credentials()

        finally:
            # Restore original key
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key


if __name__ == "__main__":
    # Run with: python -m pytest tests/integration/test_openai_integration.py -v
    pytest.main([__file__, "-v"])
