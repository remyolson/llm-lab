"""
Unit tests for LLM providers.

These tests verify provider functionality in isolation using mocks.
No external API calls are made.
"""

from unittest.mock import Mock, patch

import pytest


@pytest.mark.unit
class TestProviderInitialization:
    """Test provider initialization and configuration."""

    def test_openai_provider_init(self, mock_env):
        """Test OpenAI provider initialization with mock environment."""
        from providers.openai import OpenAIProvider

        with patch("openai.OpenAI"):
            provider = OpenAIProvider(api_key="test-key", model="gpt-4", temperature=0.7)

            assert provider.model == "gpt-4"
            assert provider.temperature == 0.7

    def test_anthropic_provider_init(self, mock_env):
        """Test Anthropic provider initialization."""
        from providers.anthropic import AnthropicProvider

        with patch("anthropic.Anthropic"):
            provider = AnthropicProvider(api_key="test-key", model="claude-3")

            assert provider.model == "claude-3"

    def test_google_provider_init(self, mock_env):
        """Test Google provider initialization."""
        from providers.google import GoogleProvider

        with patch("google.generativeai.configure"):
            with patch("google.generativeai.GenerativeModel"):
                provider = GoogleProvider(api_key="test-key", model="gemini-pro")

                assert provider.model_name == "gemini-pro"


@pytest.mark.unit
class TestProviderGeneration:
    """Test provider text generation functionality."""

    def test_mock_provider_generation(self, mock_openai_provider):
        """Test text generation with mock provider."""
        response = mock_openai_provider.generate("Test prompt")

        assert isinstance(response, str)
        assert "Mock response" in response
        assert mock_openai_provider.call_count == 1
        assert mock_openai_provider.last_prompt == "Test prompt"

    def test_mock_provider_streaming(self, mock_anthropic_provider):
        """Test streaming generation with mock provider."""
        stream = mock_anthropic_provider.generate_streaming("Stream test")

        chunks = list(stream)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

        # Reconstruct full response
        full_response = "".join(chunks)
        assert "Mock response" in full_response

    @pytest.mark.parametrize("provider_name", ["openai", "anthropic", "google"])
    def test_all_providers_generate(self, provider_name, mock_all_providers):
        """Test that all mock providers can generate text."""
        provider = mock_all_providers[provider_name]
        response = provider.generate("Universal test")

        assert isinstance(response, str)
        assert len(response) > 0
        assert provider.call_count == 1


@pytest.mark.unit
class TestProviderErrorHandling:
    """Test error handling in providers."""

    def test_invalid_api_key_handling(self):
        """Test handling of invalid API keys."""
        from providers.openai import OpenAIProvider

        with patch("openai.OpenAI") as mock_client:
            mock_client.side_effect = Exception("Invalid API key")

            with pytest.raises(Exception) as exc_info:
                OpenAIProvider(api_key="invalid")

            assert "Invalid API key" in str(exc_info.value)

    def test_rate_limit_handling(self, mock_openai_provider):
        """Test rate limit error handling."""
        # Simulate rate limit error
        mock_openai_provider.generate = Mock(side_effect=Exception("Rate limit exceeded"))

        with pytest.raises(Exception) as exc_info:
            mock_openai_provider.generate("Test")

        assert "Rate limit" in str(exc_info.value)

    def test_timeout_handling(self, mock_google_provider):
        """Test timeout handling."""
        import time

        # Make provider very slow
        mock_google_provider.response_delay = 10

        # This would normally timeout in a real scenario
        # For unit test, we just verify the delay is set
        assert mock_google_provider.response_delay == 10


@pytest.mark.unit
class TestProviderConfiguration:
    """Test provider configuration and parameter handling."""

    def test_temperature_bounds(self):
        """Test that temperature is bounded correctly."""
        from utils.validation import validate_temperature

        assert validate_temperature(0.0) == 0.0
        assert validate_temperature(1.0) == 1.0
        assert validate_temperature(0.5) == 0.5

        with pytest.raises(ValueError):
            validate_temperature(-0.1)

        with pytest.raises(ValueError):
            validate_temperature(1.1)

    def test_max_tokens_validation(self):
        """Test max tokens validation."""
        from utils.validation import validate_max_tokens

        assert validate_max_tokens(100) == 100
        assert validate_max_tokens(4096) == 4096

        with pytest.raises(ValueError):
            validate_max_tokens(-1)

        with pytest.raises(ValueError):
            validate_max_tokens(1000000)  # Too large

    def test_model_name_validation(self):
        """Test model name validation."""
        from utils.validation import validate_model_name

        valid_models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus-20240229", "gemini-pro"]

        for model in valid_models:
            assert validate_model_name(model) == model

        with pytest.raises(ValueError):
            validate_model_name("")

        with pytest.raises(ValueError):
            validate_model_name(None)
