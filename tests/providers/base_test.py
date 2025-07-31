"""
Base test classes for provider testing

This module provides abstract base classes that all provider tests should inherit from.
It ensures consistent testing patterns and provides common functionality.
"""

import os
import pytest
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type
from unittest.mock import Mock, patch
import json
import time

from llm_providers.base import LLMProvider
from llm_providers.exceptions import (
    ProviderError, 
    InvalidCredentialsError,
    RateLimitError,
    ModelNotSupportedError,
    ProviderTimeoutError
)


class BaseProviderTest(ABC):
    """
    Abstract base class for all provider unit tests.
    
    This class provides common test infrastructure and ensures all providers
    are tested consistently with mocked API calls.
    """
    
    @abstractmethod
    def get_provider_class(self) -> Type[LLMProvider]:
        """Return the provider class being tested."""
        pass
    
    @abstractmethod
    def get_default_model(self) -> str:
        """Return the default model name for this provider."""
        pass
    
    @abstractmethod
    def get_supported_models(self) -> list[str]:
        """Return list of supported model names."""
        pass
    
    @pytest.fixture
    def provider(self):
        """Create a provider instance with mocked credentials."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'ANTHROPIC_API_KEY': 'test-key',
            'GOOGLE_API_KEY': 'test-key',
        }, clear=False):
            provider_class = self.get_provider_class()
            return provider_class(model=self.get_default_model())
    
    @pytest.fixture
    def mock_response(self):
        """Standard mock response for successful API calls."""
        return {
            'content': 'This is a test response',
            'model': self.get_default_model(),
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 20,
                'total_tokens': 30
            }
        }
    
    def test_initialization(self):
        """Test provider can be initialized with valid model."""
        provider_class = self.get_provider_class()
        provider = provider_class(model=self.get_default_model())
        assert provider.model == self.get_default_model()
    
    def test_initialization_with_invalid_model(self):
        """Test provider raises error with invalid model."""
        provider_class = self.get_provider_class()
        with pytest.raises(ModelNotSupportedError) as exc_info:
            provider_class(model="invalid-model-xyz")
        assert "invalid-model-xyz" in str(exc_info.value)
    
    def test_all_supported_models(self):
        """Test that all supported models can be initialized."""
        provider_class = self.get_provider_class()
        for model in self.get_supported_models():
            provider = provider_class(model=model)
            assert provider.model == model
    
    @abstractmethod
    def test_generate_success(self, provider, mock_response):
        """Test successful generation - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def test_generate_with_system_prompt(self, provider, mock_response):
        """Test generation with system prompt - must be implemented by subclasses."""
        pass
    
    def test_missing_api_key(self):
        """Test that missing API key raises appropriate error."""
        # Clear relevant environment variables
        env_vars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']
        with patch.dict(os.environ, {key: '' for key in env_vars}, clear=False):
            provider_class = self.get_provider_class()
            provider = provider_class(model=self.get_default_model())
            with pytest.raises(InvalidCredentialsError):
                provider.initialize()
    
    def test_rate_limit_handling(self, provider):
        """Test that rate limit errors are handled properly."""
        # This should be overridden by provider-specific implementations
        pass
    
    def test_timeout_handling(self, provider):
        """Test that timeout errors are handled properly."""
        # This should be overridden by provider-specific implementations
        pass
    
    def test_retry_logic(self, provider):
        """Test that retry logic works correctly."""
        # This should be overridden by provider-specific implementations
        pass
    
    def test_generate_with_all_parameters(self, provider):
        """Test generation with all supported parameters."""
        # This should be overridden by provider-specific implementations
        pass


class BaseProviderIntegrationTest(ABC):
    """
    Abstract base class for provider integration tests.
    
    These tests actually call the provider APIs and should only run
    when explicitly enabled via environment variables.
    """
    
    @abstractmethod
    def get_provider_class(self) -> Type[LLMProvider]:
        """Return the provider class being tested."""
        pass
    
    @abstractmethod
    def get_test_model(self) -> str:
        """Return model name to use for integration tests."""
        pass
    
    @abstractmethod
    def get_env_var_name(self) -> str:
        """Return environment variable name that enables integration tests."""
        pass
    
    @pytest.fixture
    def skip_integration(self):
        """Skip test if integration testing is not enabled."""
        if not os.getenv(self.get_env_var_name(), '').lower() == 'true':
            pytest.skip(f"Integration tests not enabled. Set {self.get_env_var_name()}=true to run.")
    
    @pytest.fixture
    def provider(self, skip_integration):
        """Create a real provider instance for integration testing."""
        provider_class = self.get_provider_class()
        provider = provider_class(model=self.get_test_model())
        provider.initialize()
        return provider
    
    def test_simple_generation(self, provider):
        """Test a simple generation request."""
        prompt = "What is 2 + 2? Answer with just the number."
        response = provider.generate(prompt)
        
        assert response is not None
        assert len(response) > 0
        assert "4" in response
    
    def test_generation_with_system_prompt(self, provider):
        """Test generation with system prompt."""
        prompt = "What is the capital of France?"
        system_prompt = "You are a geography expert. Answer concisely."
        
        response = provider.generate(prompt, system_prompt=system_prompt)
        
        assert response is not None
        assert len(response) > 0
        assert "Paris" in response
    
    def test_generation_parameters(self, provider):
        """Test generation with various parameters."""
        prompt = "Generate a random number between 1 and 10."
        
        # Test with low temperature (more deterministic)
        response1 = provider.generate(
            prompt,
            temperature=0.1,
            max_tokens=10
        )
        
        # Test with high temperature (more random)
        response2 = provider.generate(
            prompt,
            temperature=0.9,
            max_tokens=10
        )
        
        assert response1 is not None
        assert response2 is not None
        assert len(response1) <= 50  # Rough token limit check
        assert len(response2) <= 50
    
    @pytest.mark.slow
    def test_performance_benchmark(self, provider):
        """Benchmark provider response times."""
        prompts = [
            "What is 2 + 2?",
            "Name a color.",
            "Is water wet?",
            "Count to 5.",
            "Say hello."
        ]
        
        response_times = []
        
        for prompt in prompts:
            start_time = time.time()
            response = provider.generate(prompt, max_tokens=20)
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            assert response is not None
            assert len(response) > 0
        
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        
        # Log performance metrics
        print(f"\nPerformance Metrics for {provider.__class__.__name__}:")
        print(f"  Average response time: {avg_response_time:.2f}s")
        print(f"  Min response time: {min_response_time:.2f}s")
        print(f"  Max response time: {max_response_time:.2f}s")
        
        # Assert reasonable performance (adjust as needed)
        assert avg_response_time < 5.0  # Average should be under 5 seconds
        assert max_response_time < 10.0  # Max should be under 10 seconds
    
    def test_error_handling(self, provider):
        """Test error handling with invalid inputs."""
        # Test with empty prompt
        with pytest.raises(ProviderError):
            provider.generate("")
        
        # Test with None prompt
        with pytest.raises(ProviderError):
            provider.generate(None)
        
        # Test with extremely long prompt (if provider has limits)
        # This is provider-specific and may need adjustment
        very_long_prompt = "a" * 100000
        try:
            response = provider.generate(very_long_prompt, max_tokens=1)
            # Some providers might handle this gracefully
            assert response is not None
        except ProviderError:
            # Others might raise an error, which is also acceptable
            pass