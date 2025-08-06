"""
Basic mocked tests for all providers

This module tests basic functionality with mocked API calls.
"""

import json
import os
import time
from unittest.mock import Mock, patch

import pytest

# Import providers directly to avoid base_test issues
try:
    from llm_providers import AnthropicProvider, GoogleProvider, OpenAIProvider
    from llm_providers.exceptions import (
        InvalidCredentialsError,
        ModelNotSupportedError,
        ProviderError,
        ProviderTimeoutError,
        RateLimitError,
    )

    PROVIDERS_AVAILABLE = True
except ImportError:
    PROVIDERS_AVAILABLE = False


@pytest.mark.skipif(not PROVIDERS_AVAILABLE, reason="Provider modules not available")
class TestBasicProviderMocks:
    """Basic mocked tests for all providers."""

    @pytest.fixture
    def providers(self):
        """Get instances of all providers."""
        return [
            OpenAIProvider(model_name="gpt-3.5-turbo"),
            AnthropicProvider(model_name="claude-3-haiku-20240307"),
            GoogleProvider(model_name="gemini-1.5-flash"),
        ]

    @patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "sk-test_key",
            "ANTHROPIC_API_KEY": "test_key",
            "GOOGLE_API_KEY": "test_key",
        },
    )
    def test_empty_prompt_handling(self, providers):
        """Test handling of empty prompts."""
        for provider in providers:
            # Mock the validate_credentials method to skip actual API calls
            with patch.object(provider, "validate_credentials", return_value=True):
                with pytest.raises(ProviderError) as exc_info:
                    provider.generate("")

                error_msg = str(exc_info.value).lower()
                assert "empty" in error_msg or "prompt" in error_msg or "required" in error_msg

    @patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test_key",
            "ANTHROPIC_API_KEY": "test_key",
            "GOOGLE_API_KEY": "test_key",
        },
    )
    def test_none_prompt_handling(self, providers):
        """Test handling of None prompt."""
        for provider in providers:
            with pytest.raises(ProviderError) as exc_info:
                provider.generate(None)

            error_msg = str(exc_info.value).lower()
            assert "none" in error_msg or "prompt" in error_msg or "required" in error_msg

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test_key"})
    @patch("openai.OpenAI")
    def test_openai_mock_generation(self, mock_openai_class):
        """Test OpenAI provider with mocked API."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content="Test response from OpenAI"))]
        mock_completion.usage = Mock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        mock_client.chat.completions.create.return_value = mock_completion

        # Test generation
        provider = OpenAIProvider(model_name="gpt-3.5-turbo")

        # Mock credential validation to avoid API calls
        with patch.object(provider, "validate_credentials", return_value=True):
            provider.initialize()
            response = provider.generate("Test prompt")

        assert response == "Test response from OpenAI"
        mock_client.chat.completions.create.assert_called_once()

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"})
    @patch("anthropic.Anthropic")
    def test_anthropic_mock_generation(self, mock_anthropic_class):
        """Test Anthropic provider with mocked API."""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_message = Mock()
        mock_message.content = [Mock(text="Test response from Anthropic")]
        mock_message.usage = Mock(input_tokens=10, output_tokens=20)
        mock_client.messages.create.return_value = mock_message

        # Test generation
        provider = AnthropicProvider(model_name="claude-3-haiku-20240307")

        # Mock credential validation to avoid API calls
        with patch.object(provider, "validate_credentials", return_value=True):
            provider.initialize()
            response = provider.generate("Test prompt")

        assert response == "Test response from Anthropic"
        mock_client.messages.create.assert_called_once()

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test_key"})
    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    def test_google_mock_generation(self, mock_model_class, mock_configure):
        """Test Google provider with mocked API."""
        # Setup mock
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        mock_response = Mock()
        mock_response.text = "Test response from Google"
        mock_model.generate_content.return_value = mock_response

        # Test generation
        provider = GoogleProvider(model_name="gemini-1.5-flash")

        # Mock credential validation to avoid API calls
        with patch.object(provider, "validate_credentials", return_value=True):
            provider.initialize()
            response = provider.generate("Test prompt")

        assert response == "Test response from Google"
        mock_model.generate_content.assert_called_once()

    def test_model_validation(self):
        """Test that invalid models are rejected."""
        # Test invalid models
        with pytest.raises(ModelNotSupportedError):
            OpenAIProvider(model_name="invalid-gpt-model")

        with pytest.raises(ModelNotSupportedError):
            AnthropicProvider(model_name="invalid-claude-model")

        with pytest.raises(ModelNotSupportedError):
            GoogleProvider(model_name="invalid-gemini-model")

    @patch("openai.OpenAI")
    def test_parameter_passing(self, mock_openai_class):
        """Test that parameters are passed correctly to API."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content="Parameterized response"))]
        mock_client.chat.completions.create.return_value = mock_completion

        # Test with parameters
        provider = OpenAIProvider(model_name="gpt-3.5-turbo")
        provider.initialize()
        response = provider.generate("Test prompt", temperature=0.7, max_tokens=100, top_p=0.9)

        # Verify parameters were passed
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["temperature"] == 0.7
        assert call_args["max_tokens"] == 100
        assert call_args["top_p"] == 0.9


class TestProviderErrorHandling:
    """Test error handling across providers."""

    @patch("openai.OpenAI")
    def test_openai_error_handling(self, mock_openai_class):
        """Test OpenAI error handling."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Simulate API error
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        provider = OpenAIProvider(model_name="gpt-3.5-turbo")
        provider.initialize()

        with pytest.raises(ProviderError):
            provider.generate("Test prompt")

    @patch("anthropic.Anthropic")
    def test_anthropic_error_handling(self, mock_anthropic_class):
        """Test Anthropic error handling."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Simulate API error
        mock_client.messages.create.side_effect = Exception("API Error")

        provider = AnthropicProvider(model_name="claude-3-haiku-20240307")
        provider.initialize()

        with pytest.raises(ProviderError):
            provider.generate("Test prompt")

    @patch("google.generativeai.GenerativeModel")
    def test_google_error_handling(self, mock_model_class):
        """Test Google error handling."""
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        # Simulate API error
        mock_model.generate_content.side_effect = Exception("API Error")

        provider = GoogleProvider(model_name="gemini-1.5-flash")
        provider.initialize()

        with pytest.raises(ProviderError):
            provider.generate("Test prompt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
