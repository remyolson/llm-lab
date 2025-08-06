"""
Parameterized cross-provider tests

This module runs the same tests across all providers using pytest parametrization
to ensure consistent behavior and identify provider-specific differences.
"""

import concurrent.futures
import time
from typing import Any, Dict, List, Type
from unittest.mock import Mock, patch

import pytest

# Import providers
try:
    from llm_providers import AnthropicProvider, GoogleProvider, OpenAIProvider
    from src.providers.exceptions import (
        InvalidCredentialsError,
        ModelNotSupportedError,
        ProviderError,
        ProviderTimeoutError,
        RateLimitError,
    )

    PROVIDERS_AVAILABLE = True
except ImportError:
    PROVIDERS_AVAILABLE = False


# Provider configurations for testing
PROVIDER_CONFIGS = [
    (OpenAIProvider, "gpt-3.5-turbo", "OPENAI_API_KEY", "sk-test_key", "openai.OpenAI"),
    (
        AnthropicProvider,
        "claude-3-haiku-20240307",
        "ANTHROPIC_API_KEY",
        "test_key",
        "anthropic.Anthropic",
    ),
    (
        GoogleProvider,
        "gemini-1.5-flash",
        "GOOGLE_API_KEY",
        "test_key",
        "google.generativeai.GenerativeModel",
    ),
]

# Test prompts for different categories
TEST_PROMPTS = {
    "simple": [
        "What is 2 + 2?",
        "What color is the sky?",
        "Say hello.",
        "Count to 3.",
        "Name a fruit.",
    ],
    "complex": [
        "Explain quantum physics in simple terms.",
        "Write a haiku about technology.",
        "Compare Python and JavaScript.",
        "Describe machine learning.",
        "What are the benefits of renewable energy?",
    ],
    "code": [
        "Write a Python function to reverse a string.",
        "Create a JavaScript function to find the maximum in an array.",
        "Show a SQL query to count records.",
        "Write HTML for a simple form.",
        "Create a CSS class for centering text.",
    ],
    "edge_cases": [
        "a",  # Very short
        "Test" * 100,  # Repeated content
        "What is 🌟 + 🚀?",  # Unicode/emoji
        "Explain.\n\nWith multiple.\n\nLine breaks.",  # Multi-line
        "Test with special chars: @#$%^&*()",  # Special characters
    ],
}

# Expected behavior patterns
EXPECTED_BEHAVIORS = {
    "response_type": str,
    "min_response_length": 1,
    "max_response_length": 10000,
    "response_not_empty": True,
}


@pytest.mark.skipif(not PROVIDERS_AVAILABLE, reason="Provider modules not available")
class TestParameterizedProviders:
    """Parameterized tests that run across all providers."""

    @pytest.mark.parametrize("provider_class,model,env_var,api_key,mock_path", PROVIDER_CONFIGS)
    @pytest.mark.parametrize("prompt", TEST_PROMPTS["simple"])
    def test_simple_generation_consistency(
        self, provider_class, model, env_var, api_key, mock_path, prompt
    ):
        """Test that all providers handle simple prompts consistently."""
        with patch.dict("os.environ", {env_var: api_key}):
            provider = provider_class(model_name=model)

            # Mock the API calls and credentials
            with patch(mock_path) as mock_api:
                if provider_class == OpenAIProvider:
                    mock_client = Mock()
                    mock_api.return_value = mock_client
                    mock_completion = Mock()
                    mock_completion.choices = [Mock(message=Mock(content=f"Response to: {prompt}"))]
                    mock_client.chat.completions.create.return_value = mock_completion
                elif provider_class == AnthropicProvider:
                    mock_client = Mock()
                    mock_api.return_value = mock_client
                    mock_message = Mock()
                    mock_message.content = [Mock(text=f"Response to: {prompt}")]
                    mock_client.messages.create.return_value = mock_message
                else:  # GoogleProvider
                    mock_model = Mock()
                    mock_api.return_value = mock_model
                    mock_response = Mock()
                    mock_response.text = f"Response to: {prompt}"
                    mock_model.generate_content.return_value = mock_response

                # Mock credential validation
                with patch.object(provider, "validate_credentials", return_value=True):
                    provider.initialize()
                    response = provider.generate(prompt)

                    # Check consistent behavior
                    assert isinstance(response, EXPECTED_BEHAVIORS["response_type"])
                    assert len(response) >= EXPECTED_BEHAVIORS["min_response_length"]
                    assert len(response) <= EXPECTED_BEHAVIORS["max_response_length"]
                    assert bool(response.strip()) == EXPECTED_BEHAVIORS["response_not_empty"]
                    assert prompt.lower() in response.lower() or "response" in response.lower()

    @pytest.mark.parametrize("provider_class,model,env_var,api_key,mock_path", PROVIDER_CONFIGS)
    @pytest.mark.parametrize("invalid_prompt", ["", "   ", None])
    def test_invalid_prompt_handling_consistency(
        self, provider_class, model, env_var, api_key, mock_path, invalid_prompt
    ):
        """Test that all providers handle invalid prompts consistently."""
        with patch.dict("os.environ", {env_var: api_key}):
            provider = provider_class(model_name=model)

            # Mock credential validation
            with patch.object(provider, "validate_credentials", return_value=True):
                with pytest.raises(ProviderError) as exc_info:
                    provider.generate(invalid_prompt)

                error_msg = str(exc_info.value).lower()
                # All providers should reject invalid prompts with meaningful errors
                assert any(
                    keyword in error_msg
                    for keyword in ["empty", "prompt", "required", "none", "invalid"]
                )

    @pytest.mark.parametrize("provider_class,model,env_var,api_key,mock_path", PROVIDER_CONFIGS)
    @pytest.mark.parametrize("temp", [0.0, 0.5, 1.0])
    def test_temperature_parameter_consistency(
        self, provider_class, model, env_var, api_key, mock_path, temp
    ):
        """Test that all providers handle temperature parameter consistently."""
        with patch.dict("os.environ", {env_var: api_key}):
            provider = provider_class(model_name=model)

            with patch(mock_path) as mock_api:
                # Setup appropriate mock response
                if provider_class == OpenAIProvider:
                    mock_client = Mock()
                    mock_api.return_value = mock_client
                    mock_completion = Mock()
                    mock_completion.choices = [
                        Mock(message=Mock(content=f"Response with temp {temp}"))
                    ]
                    mock_client.chat.completions.create.return_value = mock_completion
                elif provider_class == AnthropicProvider:
                    mock_client = Mock()
                    mock_api.return_value = mock_client
                    mock_message = Mock()
                    mock_message.content = [Mock(text=f"Response with temp {temp}")]
                    mock_client.messages.create.return_value = mock_message
                else:  # GoogleProvider
                    mock_model = Mock()
                    mock_api.return_value = mock_model
                    mock_response = Mock()
                    mock_response.text = f"Response with temp {temp}"
                    mock_model.generate_content.return_value = mock_response

                with patch.object(provider, "validate_credentials", return_value=True):
                    provider.initialize()
                    response = provider.generate("Test prompt", temperature=temp)

                    # All providers should accept valid temperature values
                    assert isinstance(response, str)
                    assert len(response) > 0

    @pytest.mark.parametrize("provider_class,model,env_var,api_key,mock_path", PROVIDER_CONFIGS)
    @pytest.mark.parametrize("invalid_temp", [-0.1, 2.1, -1.0, 3.0])
    def test_invalid_temperature_handling_consistency(
        self, provider_class, model, env_var, api_key, mock_path, invalid_temp
    ):
        """Test that all providers reject invalid temperature values consistently."""
        with patch.dict("os.environ", {env_var: api_key}):
            provider = provider_class(model_name=model)

            with patch.object(provider, "validate_credentials", return_value=True):
                with pytest.raises(ProviderError) as exc_info:
                    provider.generate("Test prompt", temperature=invalid_temp)

                error_msg = str(exc_info.value).lower()
                # All providers should reject invalid temperature with meaningful error
                assert (
                    "temperature" in error_msg or "parameter" in error_msg or "range" in error_msg
                )

    @pytest.mark.parametrize("provider_class,model,env_var,api_key,mock_path", PROVIDER_CONFIGS)
    @pytest.mark.parametrize("max_tokens", [1, 10, 100, 1000])
    def test_max_tokens_parameter_consistency(
        self, provider_class, model, env_var, api_key, mock_path, max_tokens
    ):
        """Test that all providers handle max_tokens parameter consistently."""
        with patch.dict("os.environ", {env_var: api_key}):
            provider = provider_class(model_name=model)

            with patch(mock_path) as mock_api:
                # Generate response that respects token limit (roughly)
                expected_response = (
                    "Short"
                    if max_tokens < 10
                    else "This is a longer response"
                    if max_tokens < 50
                    else "This is a much longer response with more content to test token limits properly."
                )

                if provider_class == OpenAIProvider:
                    mock_client = Mock()
                    mock_api.return_value = mock_client
                    mock_completion = Mock()
                    mock_completion.choices = [Mock(message=Mock(content=expected_response))]
                    mock_client.chat.completions.create.return_value = mock_completion
                elif provider_class == AnthropicProvider:
                    mock_client = Mock()
                    mock_api.return_value = mock_client
                    mock_message = Mock()
                    mock_message.content = [Mock(text=expected_response)]
                    mock_client.messages.create.return_value = mock_message
                else:  # GoogleProvider
                    mock_model = Mock()
                    mock_api.return_value = mock_model
                    mock_response = Mock()
                    mock_response.text = expected_response
                    mock_model.generate_content.return_value = mock_response

                with patch.object(provider, "validate_credentials", return_value=True):
                    provider.initialize()
                    response = provider.generate("Generate some text", max_tokens=max_tokens)

                    assert isinstance(response, str)
                    assert len(response) > 0
                    # Response should be roughly appropriate for the token limit
                    if max_tokens <= 10:
                        assert len(response) <= 50  # Very short response

    @pytest.mark.parametrize("provider_class,model,env_var,api_key,mock_path", PROVIDER_CONFIGS)
    def test_system_prompt_support_consistency(
        self, provider_class, model, env_var, api_key, mock_path
    ):
        """Test that all providers handle system prompts consistently."""
        system_prompt = "You are a helpful assistant that speaks like a pirate."
        user_prompt = "What is the weather like?"

        with patch.dict("os.environ", {env_var: api_key}):
            provider = provider_class(model_name=model)

            with patch(mock_path) as mock_api:
                if provider_class == OpenAIProvider:
                    mock_client = Mock()
                    mock_api.return_value = mock_client
                    mock_completion = Mock()
                    mock_completion.choices = [
                        Mock(message=Mock(content="Arrr, the weather be fine!"))
                    ]
                    mock_client.chat.completions.create.return_value = mock_completion
                elif provider_class == AnthropicProvider:
                    mock_client = Mock()
                    mock_api.return_value = mock_client
                    mock_message = Mock()
                    mock_message.content = [Mock(text="Arrr, the weather be fine!")]
                    mock_client.messages.create.return_value = mock_message
                else:  # GoogleProvider
                    mock_model = Mock()
                    mock_api.return_value = mock_model
                    mock_response = Mock()
                    mock_response.text = "Arrr, the weather be fine!"
                    mock_model.generate_content.return_value = mock_response

                with patch.object(provider, "validate_credentials", return_value=True):
                    provider.initialize()
                    response = provider.generate(user_prompt, system_prompt=system_prompt)

                    # All providers should support system prompts
                    assert isinstance(response, str)
                    assert len(response) > 0
                    # The response should reflect the system prompt influence (pirate theme)
                    assert (
                        "arr" in response.lower()
                        or "pirate" in response.lower()
                        or "fine" in response.lower()
                    )

    @pytest.mark.parametrize("provider_class,model,env_var,api_key,mock_path", PROVIDER_CONFIGS)
    def test_error_recovery_consistency(self, provider_class, model, env_var, api_key, mock_path):
        """Test that all providers handle API errors consistently."""
        with patch.dict("os.environ", {env_var: api_key}):
            provider = provider_class(model_name=model)

            with patch(mock_path) as mock_api:
                # Simulate API error
                if provider_class == OpenAIProvider:
                    mock_client = Mock()
                    mock_api.return_value = mock_client
                    mock_client.chat.completions.create.side_effect = Exception("API Error")
                elif provider_class == AnthropicProvider:
                    mock_client = Mock()
                    mock_api.return_value = mock_client
                    mock_client.messages.create.side_effect = Exception("API Error")
                else:  # GoogleProvider
                    mock_model = Mock()
                    mock_api.return_value = mock_model
                    mock_model.generate_content.side_effect = Exception("API Error")

                with patch.object(provider, "validate_credentials", return_value=True):
                    provider.initialize()

                    # All providers should convert API errors to ProviderError
                    with pytest.raises(ProviderError) as exc_info:
                        provider.generate("Test prompt")

                    error_msg = str(exc_info.value).lower()
                    assert "error" in error_msg or "failed" in error_msg


@pytest.mark.skipif(not PROVIDERS_AVAILABLE, reason="Provider modules not available")
class TestCrossProviderComparison:
    """Tests that compare behavior across providers."""

    def test_response_time_comparison(self):
        """Compare response times across providers (mocked)."""
        results = {}

        for provider_class, model, env_var, api_key, mock_path in PROVIDER_CONFIGS:
            with patch.dict("os.environ", {env_var: api_key}):
                provider = provider_class(model_name=model)

                with patch(mock_path) as mock_api:
                    # Setup mock with simulated delay
                    def delayed_response(*args, **kwargs):
                        # Simulate different response times for different providers
                        delays = {
                            "OpenAIProvider": 0.5,
                            "AnthropicProvider": 0.7,
                            "GoogleProvider": 0.3,
                        }
                        time.sleep(delays.get(provider_class.__name__, 0.5))

                        if provider_class == OpenAIProvider:
                            mock_completion = Mock()
                            mock_completion.choices = [Mock(message=Mock(content="Fast response"))]
                            return mock_completion
                        elif provider_class == AnthropicProvider:
                            mock_message = Mock()
                            mock_message.content = [Mock(text="Fast response")]
                            return mock_message
                        else:  # GoogleProvider
                            mock_response = Mock()
                            mock_response.text = "Fast response"
                            return mock_response

                    if provider_class == OpenAIProvider:
                        mock_client = Mock()
                        mock_api.return_value = mock_client
                        mock_client.chat.completions.create.side_effect = delayed_response
                    elif provider_class == AnthropicProvider:
                        mock_client = Mock()
                        mock_api.return_value = mock_client
                        mock_client.messages.create.side_effect = delayed_response
                    else:  # GoogleProvider
                        mock_model = Mock()
                        mock_api.return_value = mock_model
                        mock_model.generate_content.side_effect = delayed_response

                    with patch.object(provider, "validate_credentials", return_value=True):
                        provider.initialize()

                        start_time = time.time()
                        response = provider.generate("Quick test")
                        end_time = time.time()

                        results[provider_class.__name__] = {
                            "response_time": end_time - start_time,
                            "response_length": len(response),
                        }

        # Verify all providers responded
        assert len(results) == len(PROVIDER_CONFIGS)

        # All response times should be reasonable (under 2 seconds for mocked tests)
        for provider_name, metrics in results.items():
            assert metrics["response_time"] < 2.0
            assert metrics["response_length"] > 0

    def test_concurrent_request_handling(self):
        """Test that all providers can handle concurrent requests."""

        def make_concurrent_request(provider_config, request_id):
            provider_class, model, env_var, api_key, mock_path = provider_config

            with patch.dict("os.environ", {env_var: api_key}):
                provider = provider_class(model_name=model)

                with patch(mock_path) as mock_api:
                    if provider_class == OpenAIProvider:
                        mock_client = Mock()
                        mock_api.return_value = mock_client
                        mock_completion = Mock()
                        mock_completion.choices = [
                            Mock(message=Mock(content=f"Concurrent response {request_id}"))
                        ]
                        mock_client.chat.completions.create.return_value = mock_completion
                    elif provider_class == AnthropicProvider:
                        mock_client = Mock()
                        mock_api.return_value = mock_client
                        mock_message = Mock()
                        mock_message.content = [Mock(text=f"Concurrent response {request_id}")]
                        mock_client.messages.create.return_value = mock_message
                    else:  # GoogleProvider
                        mock_model = Mock()
                        mock_api.return_value = mock_model
                        mock_response = Mock()
                        mock_response.text = f"Concurrent response {request_id}"
                        mock_model.generate_content.return_value = mock_response

                    with patch.object(provider, "validate_credentials", return_value=True):
                        provider.initialize()
                        return provider.generate(f"Concurrent test {request_id}")

        # Test concurrent requests for each provider
        for provider_config in PROVIDER_CONFIGS:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for i in range(5):
                    future = executor.submit(make_concurrent_request, provider_config, i)
                    futures.append(future)

                # Collect all results
                results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        pytest.fail(f"Concurrent request failed: {e}")

                # All requests should complete successfully
                assert len(results) == 5
                for result in results:
                    assert isinstance(result, str)
                    assert "concurrent response" in result.lower()

    def test_parameter_validation_comparison(self):
        """Compare parameter validation across providers."""
        invalid_params = [
            {"temperature": -1.0},
            {"temperature": 3.0},
            {"max_tokens": -10},
            {"top_p": -0.5},
            {"top_p": 1.5},
        ]

        validation_results = {}

        for provider_class, model, env_var, api_key, mock_path in PROVIDER_CONFIGS:
            provider_name = provider_class.__name__
            validation_results[provider_name] = {}

            with patch.dict("os.environ", {env_var: api_key}):
                provider = provider_class(model_name=model)

                with patch.object(provider, "validate_credentials", return_value=True):
                    for param_dict in invalid_params:
                        param_name = list(param_dict.keys())[0]
                        param_value = param_dict[param_name]

                        try:
                            provider.generate("Test", **param_dict)
                            validation_results[provider_name][f"{param_name}_{param_value}"] = (
                                "ACCEPTED"
                            )
                        except ProviderError:
                            validation_results[provider_name][f"{param_name}_{param_value}"] = (
                                "REJECTED"
                            )
                        except Exception as e:
                            validation_results[provider_name][f"{param_name}_{param_value}"] = (
                                f"ERROR: {type(e).__name__}"
                            )

        # All providers should reject clearly invalid parameters
        for provider_name, results in validation_results.items():
            for param_test, result in results.items():
                # Invalid parameters should be rejected, not accepted
                assert result != "ACCEPTED", (
                    f"{provider_name} incorrectly accepted invalid parameter: {param_test}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
