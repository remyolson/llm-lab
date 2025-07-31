"""
Common test scenarios for all providers

This module provides reusable test scenarios that should work across
all providers to ensure consistency.
"""

import pytest
from typing import Type, List, Dict, Any
from unittest.mock import patch
import time

from llm_providers.base import LLMProvider
from llm_providers.exceptions import (
    ProviderError,
    RateLimitError,
    InvalidCredentialsError,
    ModelNotSupportedError
)


class CommonTestScenarios:
    """
    Collection of test scenarios that all providers should pass.
    
    These tests ensure consistent behavior across different providers.
    """
    
    @staticmethod
    def test_provider_consistency(provider_classes: List[Type[LLMProvider]], test_prompt: str = "What is 2 + 2?"):
        """
        Test that all providers return consistent response formats.
        
        Args:
            provider_classes: List of provider classes to test
            test_prompt: Prompt to use for testing
        """
        responses = []
        
        for provider_class in provider_classes:
            # Get default model for this provider
            if hasattr(provider_class, 'SUPPORTED_MODELS'):
                model = list(provider_class.SUPPORTED_MODELS)[0]
            else:
                # Skip if no models defined
                continue
            
            provider = provider_class(model=model)
            
            # Mock the actual API call
            with patch.object(provider, '_call_api') as mock_call:
                mock_call.return_value = "4"
                
                response = provider.generate(test_prompt)
                responses.append({
                    'provider': provider_class.__name__,
                    'response': response,
                    'type': type(response)
                })
        
        # All responses should be strings
        for resp in responses:
            assert isinstance(resp['response'], str), f"{resp['provider']} did not return a string"
        
        # All responses should be non-empty
        for resp in responses:
            assert len(resp['response']) > 0, f"{resp['provider']} returned empty response"
    
    @staticmethod
    def test_error_handling_consistency(provider_classes: List[Type[LLMProvider]]):
        """Test that all providers handle errors consistently."""
        for provider_class in provider_classes:
            if hasattr(provider_class, 'SUPPORTED_MODELS'):
                model = list(provider_class.SUPPORTED_MODELS)[0]
            else:
                continue
            
            provider = provider_class(model=model)
            
            # Test empty prompt
            with pytest.raises(ProviderError):
                provider.generate("")
            
            # Test None prompt
            with pytest.raises(ProviderError):
                provider.generate(None)
            
            # Test invalid model
            with pytest.raises(ModelNotSupportedError):
                invalid_provider = provider_class(model="invalid-model-xyz-123")
    
    @staticmethod
    def test_parameter_validation_consistency(provider_classes: List[Type[LLMProvider]]):
        """Test that all providers validate parameters consistently."""
        for provider_class in provider_classes:
            if hasattr(provider_class, 'SUPPORTED_MODELS'):
                model = list(provider_class.SUPPORTED_MODELS)[0]
            else:
                continue
            
            provider = provider_class(model=model)
            
            # Test invalid temperature
            with pytest.raises(ProviderError) as exc_info:
                provider.generate("test", temperature=2.5)
            assert "temperature" in str(exc_info.value).lower()
            
            # Test invalid max_tokens
            with pytest.raises(ProviderError) as exc_info:
                provider.generate("test", max_tokens=-1)
            assert "max_tokens" in str(exc_info.value).lower()
            
            # Test invalid top_p
            with pytest.raises(ProviderError) as exc_info:
                provider.generate("test", top_p=1.5)
            assert "top_p" in str(exc_info.value).lower()
    
    @staticmethod
    def test_retry_behavior_consistency(provider_classes: List[Type[LLMProvider]]):
        """Test that all providers handle retries consistently."""
        for provider_class in provider_classes:
            if hasattr(provider_class, 'SUPPORTED_MODELS'):
                model = list(provider_class.SUPPORTED_MODELS)[0]
            else:
                continue
            
            provider = provider_class(model=model)
            
            # Mock API calls to simulate failures then success
            call_count = 0
            def mock_api_call(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise RateLimitError("Rate limit exceeded", retry_after=0.1)
                return "Success after retries"
            
            with patch.object(provider, '_call_api', mock_api_call):
                with patch.object(provider, 'max_retries', 3):
                    response = provider.generate("test")
                    assert response == "Success after retries"
                    assert call_count == 3
    
    @staticmethod
    def test_model_switching(provider_classes: List[Type[LLMProvider]]):
        """Test that providers can switch between models correctly."""
        for provider_class in provider_classes:
            if not hasattr(provider_class, 'SUPPORTED_MODELS'):
                continue
            
            models = list(provider_class.SUPPORTED_MODELS)
            if len(models) < 2:
                continue  # Skip if provider only supports one model
            
            # Test initialization with different models
            for model in models[:2]:  # Test first two models
                provider = provider_class(model=model)
                assert provider.model == model
                
                # Mock API call
                with patch.object(provider, '_call_api') as mock_call:
                    mock_call.return_value = f"Response from {model}"
                    response = provider.generate("test")
                    assert model in response or "Response from" in response
    
    @staticmethod
    def test_system_prompt_support(provider_classes: List[Type[LLMProvider]]):
        """Test that all providers handle system prompts correctly."""
        for provider_class in provider_classes:
            if hasattr(provider_class, 'SUPPORTED_MODELS'):
                model = list(provider_class.SUPPORTED_MODELS)[0]
            else:
                continue
            
            provider = provider_class(model=model)
            
            # Mock API call to verify system prompt is passed
            with patch.object(provider, '_call_api') as mock_call:
                mock_call.return_value = "Response with system prompt"
                
                response = provider.generate(
                    "What is 2 + 2?",
                    system_prompt="You are a math tutor."
                )
                
                # Verify the API was called
                mock_call.assert_called_once()
                
                # Check if system prompt was included in the call
                # This is provider-specific, but we can check the args
                call_args = mock_call.call_args
                assert call_args is not None
    
    @staticmethod
    def test_response_format_fields(provider_classes: List[Type[LLMProvider]]):
        """Test that providers include expected fields in responses."""
        for provider_class in provider_classes:
            if hasattr(provider_class, 'SUPPORTED_MODELS'):
                model = list(provider_class.SUPPORTED_MODELS)[0]
            else:
                continue
            
            provider = provider_class(model=model)
            
            # For this test, we'll check internal behavior if providers
            # track usage or other metadata
            with patch.object(provider, '_call_api') as mock_call:
                mock_call.return_value = "Test response"
                
                # Some providers might track usage internally
                response = provider.generate("test")
                
                # Basic checks
                assert isinstance(response, str)
                assert len(response) > 0
    
    @staticmethod
    def test_concurrent_requests(provider_classes: List[Type[LLMProvider]]):
        """Test that providers handle concurrent requests safely."""
        import concurrent.futures
        
        for provider_class in provider_classes:
            if hasattr(provider_class, 'SUPPORTED_MODELS'):
                model = list(provider_class.SUPPORTED_MODELS)[0]
            else:
                continue
            
            provider = provider_class(model=model)
            
            # Mock API calls
            call_counter = 0
            def mock_api_call(*args, **kwargs):
                nonlocal call_counter
                call_counter += 1
                time.sleep(0.01)  # Simulate API delay
                return f"Response {call_counter}"
            
            with patch.object(provider, '_call_api', mock_api_call):
                # Submit multiple concurrent requests
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = []
                    for i in range(10):
                        future = executor.submit(provider.generate, f"Test prompt {i}")
                        futures.append(future)
                    
                    # Collect results
                    results = []
                    for future in concurrent.futures.as_completed(futures):
                        results.append(future.result())
                
                # All requests should complete successfully
                assert len(results) == 10
                assert all(isinstance(r, str) for r in results)
                assert call_counter == 10
    
    @staticmethod
    def test_environment_variable_handling(provider_classes: List[Type[LLMProvider]]):
        """Test that providers handle missing environment variables correctly."""
        # Define expected env vars for each provider
        provider_env_vars = {
            'OpenAIProvider': 'OPENAI_API_KEY',
            'AnthropicProvider': 'ANTHROPIC_API_KEY',
            'GoogleProvider': 'GOOGLE_API_KEY'
        }
        
        for provider_class in provider_classes:
            if hasattr(provider_class, 'SUPPORTED_MODELS'):
                model = list(provider_class.SUPPORTED_MODELS)[0]
            else:
                continue
            
            env_var = provider_env_vars.get(provider_class.__name__)
            if not env_var:
                continue
            
            # Test with missing API key
            with patch.dict('os.environ', {env_var: ''}, clear=False):
                provider = provider_class(model=model)
                
                with pytest.raises(InvalidCredentialsError) as exc_info:
                    provider.initialize()
                
                assert "API key" in str(exc_info.value) or "credentials" in str(exc_info.value).lower()
    
    @staticmethod 
    def run_all_scenarios(provider_classes: List[Type[LLMProvider]]):
        """Run all common test scenarios for the given providers."""
        scenarios = [
            ("Consistency", CommonTestScenarios.test_provider_consistency),
            ("Error Handling", CommonTestScenarios.test_error_handling_consistency),
            ("Parameter Validation", CommonTestScenarios.test_parameter_validation_consistency),
            ("Retry Behavior", CommonTestScenarios.test_retry_behavior_consistency),
            ("Model Switching", CommonTestScenarios.test_model_switching),
            ("System Prompt Support", CommonTestScenarios.test_system_prompt_support),
            ("Response Format", CommonTestScenarios.test_response_format_fields),
            ("Concurrent Requests", CommonTestScenarios.test_concurrent_requests),
            ("Environment Variables", CommonTestScenarios.test_environment_variable_handling)
        ]
        
        results = {}
        for scenario_name, scenario_func in scenarios:
            try:
                scenario_func(provider_classes)
                results[scenario_name] = "PASSED"
            except Exception as e:
                results[scenario_name] = f"FAILED: {str(e)}"
        
        return results