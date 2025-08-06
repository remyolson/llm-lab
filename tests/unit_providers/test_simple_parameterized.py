"""
Simple parameterized cross-provider tests

A simplified version of parameterized tests that focuses on the core functionality
while ensuring proper mocking to avoid real API calls.
"""

import os
from unittest.mock import Mock, patch

import pytest

# Import providers
try:
    from llm_providers import AnthropicProvider, GoogleProvider, OpenAIProvider
    from src.providers.exceptions import ModelNotSupportedError, ProviderError

    PROVIDERS_AVAILABLE = True
except ImportError:
    PROVIDERS_AVAILABLE = False


# Simple provider test configurations
PROVIDERS = [
    ("OpenAI", OpenAIProvider, "gpt-3.5-turbo"),
    ("Anthropic", AnthropicProvider, "claude-3-haiku-20240307"),
    ("Google", GoogleProvider, "gemini-1.5-flash"),
]

SIMPLE_PROMPTS = ["What is 2 + 2?", "Say hello.", "Name a color."]

INVALID_PROMPTS = ["", "   ", None]


@pytest.mark.skipif(not PROVIDERS_AVAILABLE, reason="Provider modules not available")
class TestSimpleParameterized:
    """Simple parameterized tests for basic provider functionality."""

    @pytest.mark.parametrize("provider_name,provider_class,model", PROVIDERS)
    @pytest.mark.parametrize("prompt", SIMPLE_PROMPTS)
    def test_basic_generation_all_providers(self, provider_name, provider_class, model, prompt):
        """Test basic generation works across all providers."""
        # Set up environment
        env_vars = {
            "OPENAI_API_KEY": "sk-test_key",
            "ANTHROPIC_API_KEY": "test_key",
            "GOOGLE_API_KEY": "test_key",
        }

        with patch.dict("os.environ", env_vars):
            provider = provider_class(model_name=model)

            # Mock the validate_credentials method
            with patch.object(provider, "validate_credentials", return_value=True):
                # Mock the actual API call method
                expected_response = f"{provider_name} response to: {prompt}"
                with patch.object(
                    provider, "_call_api", return_value=expected_response
                ) as mock_api:
                    provider.initialize()
                    response = provider.generate(prompt)

                    # Verify the response
                    assert isinstance(response, str)
                    assert len(response) > 0
                    assert response == expected_response

                    # Verify the API was called
                    mock_api.assert_called_once()

    @pytest.mark.parametrize("provider_name,provider_class,model", PROVIDERS)
    @pytest.mark.parametrize("invalid_prompt", INVALID_PROMPTS)
    def test_invalid_prompt_rejection_all_providers(
        self, provider_name, provider_class, model, invalid_prompt
    ):
        """Test that all providers reject invalid prompts."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test_key",
            "ANTHROPIC_API_KEY": "test_key",
            "GOOGLE_API_KEY": "test_key",
        }

        with patch.dict("os.environ", env_vars):
            provider = provider_class(model_name=model)

            # Mock credential validation
            with patch.object(provider, "validate_credentials", return_value=True):
                with pytest.raises(ProviderError) as exc_info:
                    provider.generate(invalid_prompt)

                error_msg = str(exc_info.value).lower()
                # Should contain meaningful error message
                assert any(
                    keyword in error_msg
                    for keyword in ["empty", "prompt", "required", "none", "invalid", "missing"]
                )

    @pytest.mark.parametrize("provider_name,provider_class,model", PROVIDERS)
    def test_model_property_all_providers(self, provider_name, provider_class, model):
        """Test that model property is set correctly for all providers."""
        provider = provider_class(model_name=model)
        assert provider.model == model
        assert hasattr(provider, "model_name")
        assert provider.model_name == model

    @pytest.mark.parametrize("provider_name,provider_class,model", PROVIDERS)
    def test_invalid_model_rejection_all_providers(self, provider_name, provider_class, model):
        """Test that all providers reject invalid models."""
        invalid_model = f"invalid-{provider_name.lower()}-model-xyz"

        with pytest.raises(ModelNotSupportedError) as exc_info:
            provider_class(model_name=invalid_model)

        error_msg = str(exc_info.value).lower()
        assert "model" in error_msg
        assert invalid_model.lower() in error_msg or "not supported" in error_msg

    def test_all_providers_exist(self):
        """Test that all expected providers are available."""
        expected_providers = ["OpenAIProvider", "AnthropicProvider", "GoogleProvider"]

        for provider_name in expected_providers:
            assert provider_name in globals() or hasattr(__import__("llm_providers"), provider_name)

    @pytest.mark.parametrize("provider_name,provider_class,model", PROVIDERS)
    def test_provider_initialization_all_providers(self, provider_name, provider_class, model):
        """Test that all providers can be initialized."""
        provider = provider_class(model_name=model)

        # Basic properties should be set
        assert provider.model == model
        assert provider.model_name == model
        assert hasattr(provider, "initialize")
        assert hasattr(provider, "generate")
        assert hasattr(provider, "validate_credentials")

    @pytest.mark.parametrize("provider_name,provider_class,model", PROVIDERS)
    def test_parameter_validation_all_providers(self, provider_name, provider_class, model):
        """Test parameter validation across all providers."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test_key",
            "ANTHROPIC_API_KEY": "test_key",
            "GOOGLE_API_KEY": "test_key",
        }

        with patch.dict("os.environ", env_vars):
            provider = provider_class(model_name=model)

            # Mock credential validation
            with patch.object(provider, "validate_credentials", return_value=True):
                # Test invalid temperature
                with pytest.raises(ProviderError) as exc_info:
                    provider.generate("test", temperature=2.5)

                error_msg = str(exc_info.value).lower()
                assert (
                    "temperature" in error_msg or "parameter" in error_msg or "range" in error_msg
                )

                # Test invalid max_tokens
                with pytest.raises(ProviderError) as exc_info:
                    provider.generate("test", max_tokens=-10)

                error_msg = str(exc_info.value).lower()
                assert (
                    "max_tokens" in error_msg or "parameter" in error_msg or "negative" in error_msg
                )


@pytest.mark.skipif(not PROVIDERS_AVAILABLE, reason="Provider modules not available")
class TestProviderComparison:
    """Tests that compare behavior across providers."""

    def test_response_consistency_comparison(self):
        """Compare response consistency across providers."""
        test_prompt = "What is 2 + 2?"
        env_vars = {
            "OPENAI_API_KEY": "sk-test_key",
            "ANTHROPIC_API_KEY": "test_key",
            "GOOGLE_API_KEY": "test_key",
        }

        responses = {}

        for provider_name, provider_class, model in PROVIDERS:
            with patch.dict("os.environ", env_vars):
                provider = provider_class(model_name=model)

                with patch.object(provider, "validate_credentials", return_value=True):
                    with patch.object(
                        provider, "_call_api", return_value=f"{provider_name}: 4"
                    ) as mock_api:
                        provider.initialize()
                        response = provider.generate(test_prompt)
                        responses[provider_name] = response

                        # Verify API was called
                        mock_api.assert_called_once()

        # All providers should have responded
        assert len(responses) == len(PROVIDERS)

        # All responses should be strings
        for provider_name, response in responses.items():
            assert isinstance(response, str), f"{provider_name} didn't return a string"
            assert len(response) > 0, f"{provider_name} returned empty response"
            assert provider_name in response, (
                f"{provider_name} response doesn't contain provider name"
            )

    def test_error_handling_comparison(self):
        """Compare error handling across providers."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test_key",
            "ANTHROPIC_API_KEY": "test_key",
            "GOOGLE_API_KEY": "test_key",
        }

        error_types = {}

        for provider_name, provider_class, model in PROVIDERS:
            with patch.dict("os.environ", env_vars):
                provider = provider_class(model_name=model)

                with patch.object(provider, "validate_credentials", return_value=True):
                    with patch.object(provider, "_call_api", side_effect=Exception("API Error")):
                        provider.initialize()

                        with pytest.raises(ProviderError) as exc_info:
                            provider.generate("test")

                        error_types[provider_name] = type(exc_info.value).__name__

        # All providers should raise ProviderError for API failures
        for provider_name, error_type in error_types.items():
            assert error_type == "ProviderError", (
                f"{provider_name} raised {error_type} instead of ProviderError"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
