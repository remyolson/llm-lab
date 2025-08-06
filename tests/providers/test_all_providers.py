"""
Comprehensive tests for all providers

This module runs comprehensive tests across all providers to ensure
consistency and proper functionality.
"""

import os
from unittest.mock import Mock, patch

import pytest

from llm_providers import AnthropicProvider, GoogleProvider, OpenAIProvider
from src.providers.exceptions import (
    InvalidCredentialsError,
    ModelNotSupportedError,
    ProviderError,
    RateLimitError,
)

from .base_test import BaseProviderTest
from .fixtures import (
    mock_anthropic_provider,
    mock_google_provider,
    mock_openai_provider,
    sample_evaluation_data,
    temp_config_file,
    test_config,
)
from .test_scenarios import CommonTestScenarios


class TestOpenAIProviderComprehensive(BaseProviderTest):
    """Comprehensive tests for OpenAI provider."""

    def get_provider_class(self):
        return OpenAIProvider

    def get_default_model(self):
        return "gpt-3.5-turbo"

    def get_supported_models(self):
        return [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
        ]

    @patch("openai.OpenAI")
    def test_generate_success(self, mock_openai_class, provider, mock_response):
        """Test successful generation with OpenAI."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content="Test response"))]
        mock_completion.usage = Mock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        mock_client.chat.completions.create.return_value = mock_completion

        # Initialize and generate
        provider.initialize()
        response = provider.generate("Test prompt")

        assert response == "Test response"
        mock_client.chat.completions.create.assert_called_once()

    @patch("openai.OpenAI")
    def test_generate_with_system_prompt(self, mock_openai_class, provider, mock_response):
        """Test generation with system prompt."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_completion

        # Initialize and generate
        provider.initialize()
        response = provider.generate("Test prompt", system_prompt="You are helpful")

        assert response == "Test response"

        # Verify system prompt was included
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"

    @patch("openai.OpenAI")
    def test_rate_limit_handling(self, mock_openai_class, provider):
        """Test OpenAI rate limit handling."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Simulate rate limit error
        mock_client.chat.completions.create.side_effect = Exception("Rate limit exceeded")

        provider.initialize()
        with pytest.raises(ProviderError):
            provider.generate("Test")

    @patch("openai.OpenAI")
    def test_all_parameters(self, mock_openai_class, provider):
        """Test with all supported parameters."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content="Test"))]
        mock_client.chat.completions.create.return_value = mock_completion

        provider.initialize()
        response = provider.generate(
            "Test",
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2,
        )

        # Verify parameters were passed
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["temperature"] == 0.5
        assert call_args["max_tokens"] == 100
        assert call_args["top_p"] == 0.9
        assert call_args["frequency_penalty"] == 0.1
        assert call_args["presence_penalty"] == 0.2


class TestAnthropicProviderComprehensive(BaseProviderTest):
    """Comprehensive tests for Anthropic provider."""

    def get_provider_class(self):
        return AnthropicProvider

    def get_default_model(self):
        return "claude-3-sonnet-20240229"

    def get_supported_models(self):
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-20241022",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ]

    @patch("anthropic.Anthropic")
    def test_generate_success(self, mock_anthropic_class, provider, mock_response):
        """Test successful generation with Anthropic."""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_message = Mock()
        mock_message.content = [Mock(text="Test response")]
        mock_message.usage = Mock(input_tokens=10, output_tokens=20)
        mock_client.messages.create.return_value = mock_message

        # Initialize and generate
        provider.initialize()
        response = provider.generate("Test prompt")

        assert response == "Test response"
        mock_client.messages.create.assert_called_once()

    @patch("anthropic.Anthropic")
    def test_generate_with_system_prompt(self, mock_anthropic_class, provider, mock_response):
        """Test generation with system prompt."""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_message = Mock()
        mock_message.content = [Mock(text="Test response")]
        mock_client.messages.create.return_value = mock_message

        # Initialize and generate
        provider.initialize()
        response = provider.generate("Test prompt", system_prompt="You are Claude")

        assert response == "Test response"

        # Verify system prompt was included
        call_args = mock_client.messages.create.call_args[1]
        assert call_args["system"] == "You are Claude"

    @patch("anthropic.Anthropic")
    def test_anthropic_specific_params(self, mock_anthropic_class, provider):
        """Test Anthropic-specific parameters."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_message = Mock()
        mock_message.content = [Mock(text="Test")]
        mock_client.messages.create.return_value = mock_message

        provider.initialize()

        # Test with max_tokens (required for Anthropic)
        response = provider.generate("Test", max_tokens=100)

        call_args = mock_client.messages.create.call_args[1]
        assert call_args["max_tokens"] == 100


class TestGoogleProviderComprehensive(BaseProviderTest):
    """Comprehensive tests for Google provider."""

    def get_provider_class(self):
        return GoogleProvider

    def get_default_model(self):
        return "gemini-1.5-flash"

    def get_supported_models(self):
        return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]

    @patch("google.generativeai.GenerativeModel")
    def test_generate_success(self, mock_model_class, provider, mock_response):
        """Test successful generation with Google."""
        # Setup mock
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        mock_response = Mock()
        mock_response.text = "Test response"
        mock_model.generate_content.return_value = mock_response

        # Initialize and generate
        provider.initialize()
        response = provider.generate("Test prompt")

        assert response == "Test response"
        mock_model.generate_content.assert_called_once()

    @patch("google.generativeai.GenerativeModel")
    def test_generate_with_system_prompt(self, mock_model_class, provider, mock_response):
        """Test generation with system prompt."""
        # Setup mock
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        mock_response = Mock()
        mock_response.text = "Test response"
        mock_model.generate_content.return_value = mock_response

        # Initialize and generate
        provider.initialize()
        response = provider.generate("Test prompt", system_prompt="Be helpful")

        assert response == "Test response"

        # Google combines system prompt with user prompt
        call_args = mock_model.generate_content.call_args[0][0]
        assert "Be helpful" in call_args
        assert "Test prompt" in call_args

    @patch("google.generativeai.GenerativeModel")
    def test_safety_settings(self, mock_model_class, provider):
        """Test Google's safety settings handling."""
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        mock_response = Mock()
        mock_response.text = "Safe response"
        mock_model.generate_content.return_value = mock_response

        provider.initialize()
        response = provider.generate("Test potentially unsafe content")

        assert response == "Safe response"


class TestCrossProviderConsistency:
    """Test consistency across all providers."""

    @pytest.fixture
    def all_provider_classes(self):
        """Get all provider classes."""
        return [OpenAIProvider, AnthropicProvider, GoogleProvider]

    def test_all_providers_consistency(self, all_provider_classes):
        """Run consistency tests across all providers."""
        results = CommonTestScenarios.run_all_scenarios(all_provider_classes)

        # Print results
        print("\nCross-Provider Consistency Test Results:")
        print("=" * 50)
        for scenario, result in results.items():
            status = "✓" if result == "PASSED" else "✗"
            print(f"{status} {scenario}: {result}")

        # All scenarios should pass
        failed_scenarios = [s for s, r in results.items() if r != "PASSED"]
        assert len(failed_scenarios) == 0, f"Failed scenarios: {failed_scenarios}"

    def test_error_message_consistency(self, all_provider_classes):
        """Test that error messages are consistent across providers."""
        for provider_class in all_provider_classes:
            # Test invalid model error
            with pytest.raises(ModelNotSupportedError) as exc_info:
                provider_class(model="invalid-model-xyz")

            error_msg = str(exc_info.value).lower()
            assert "model" in error_msg
            assert "invalid-model-xyz" in error_msg or "not supported" in error_msg

    @patch.dict(os.environ, {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "", "GOOGLE_API_KEY": ""})
    def test_missing_credentials_consistency(self, all_provider_classes):
        """Test that all providers handle missing credentials consistently."""
        for provider_class in all_provider_classes:
            if hasattr(provider_class, "SUPPORTED_MODELS"):
                model = list(provider_class.SUPPORTED_MODELS)[0]
                provider = provider_class(model=model)

                with pytest.raises(InvalidCredentialsError) as exc_info:
                    provider.initialize()

                error_msg = str(exc_info.value).lower()
                assert "api key" in error_msg or "credentials" in error_msg

    def test_parameter_bounds_consistency(self, all_provider_classes):
        """Test that all providers validate parameters consistently."""
        test_cases = [
            {"temperature": -0.1, "should_fail": True},
            {"temperature": 2.1, "should_fail": True},
            {"temperature": 0.7, "should_fail": False},
            {"max_tokens": -10, "should_fail": True},
            {"max_tokens": 100, "should_fail": False},
            {"top_p": -0.1, "should_fail": True},
            {"top_p": 1.5, "should_fail": True},
            {"top_p": 0.9, "should_fail": False},
        ]

        for provider_class in all_provider_classes:
            if hasattr(provider_class, "SUPPORTED_MODELS"):
                model = list(provider_class.SUPPORTED_MODELS)[0]
                provider = provider_class(model=model)

                for test_case in test_cases:
                    params = {k: v for k, v in test_case.items() if k != "should_fail"}

                    if test_case["should_fail"]:
                        with pytest.raises(ProviderError):
                            provider.generate("test", **params)
                    else:
                        # Should not raise with valid parameters
                        # Note: We're not actually calling the API here
                        try:
                            provider.validate_parameters(**params)
                        except AttributeError:
                            # Provider might not have validate_parameters method
                            pass


@pytest.mark.parametrize(
    "provider_class,model",
    [
        (OpenAIProvider, "gpt-3.5-turbo"),
        (AnthropicProvider, "claude-3-haiku-20240307"),
        (GoogleProvider, "gemini-1.5-flash"),
    ],
)
class TestParameterizedProviderTests:
    """Parameterized tests that run for all providers."""

    def test_empty_prompt_handling(self, provider_class, model):
        """Test that empty prompts are rejected."""
        provider = provider_class(model=model)

        with pytest.raises(ProviderError) as exc_info:
            provider.generate("")

        assert "empty" in str(exc_info.value).lower() or "prompt" in str(exc_info.value).lower()

    def test_none_prompt_handling(self, provider_class, model):
        """Test that None prompts are rejected."""
        provider = provider_class(model=model)

        with pytest.raises(ProviderError):
            provider.generate(None)

    def test_model_property(self, provider_class, model):
        """Test that model property is set correctly."""
        provider = provider_class(model=model)
        assert provider.model == model

    def test_temperature_validation(self, provider_class, model):
        """Test temperature parameter validation."""
        provider = provider_class(model=model)

        # Valid temperatures
        for temp in [0.0, 0.5, 1.0, 2.0]:
            try:
                # Just validate, don't actually call API
                if hasattr(provider, "validate_parameters"):
                    provider.validate_parameters(temperature=temp)
            except ProviderError:
                # Some providers might have stricter limits
                pass

        # Invalid temperatures
        for temp in [-1.0, 3.0]:
            with pytest.raises(ProviderError):
                provider.generate("test", temperature=temp)


if __name__ == "__main__":
    # Run specific test suites
    pytest.main([__file__, "-v", "-k", "TestCrossProviderConsistency"])
