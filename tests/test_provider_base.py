"""Tests for the base LLM provider infrastructure."""

import pytest
from typing import Dict, Any
from unittest.mock import MagicMock, patch

from llm_providers.base import LLMProvider, ProviderConfig
from llm_providers.exceptions import (
    ModelNotSupportedError,
    ProviderConfigurationError,
    InvalidCredentialsError
)


class MockProvider(LLMProvider):
    """Mock provider for testing abstract base class."""
    
    SUPPORTED_MODELS = ['mock-model-1', 'mock-model-2']
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Mock generate method."""
        return f"Mock response for: {prompt}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Mock get_model_info method."""
        return {
            "model_name": self.model_name,
            "provider": "mock",
            "max_tokens": 1000,
            "capabilities": ["text-generation"],
            "version": "1.0"
        }
    
    def validate_credentials(self) -> bool:
        """Mock validate_credentials method."""
        return True


class InvalidProvider(LLMProvider):
    """Invalid provider missing required methods (for testing)."""
    pass


class TestProviderConfig:
    """Test the ProviderConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ProviderConfig()
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.top_p == 1.0
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.additional_params == {}
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ProviderConfig(
            temperature=0.5,
            max_tokens=2000,
            top_p=0.9,
            timeout=60,
            max_retries=5,
            retry_delay=2.0,
            additional_params={"custom": "value"}
        )
        assert config.temperature == 0.5
        assert config.max_tokens == 2000
        assert config.top_p == 0.9
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.additional_params == {"custom": "value"}
    
    def test_invalid_temperature(self):
        """Test invalid temperature values."""
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            ProviderConfig(temperature=-0.1)
        
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            ProviderConfig(temperature=2.1)
    
    def test_invalid_max_tokens(self):
        """Test invalid max_tokens values."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            ProviderConfig(max_tokens=0)
        
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            ProviderConfig(max_tokens=-100)
    
    def test_invalid_top_p(self):
        """Test invalid top_p values."""
        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            ProviderConfig(top_p=-0.1)
        
        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            ProviderConfig(top_p=1.1)
    
    def test_invalid_timeout(self):
        """Test invalid timeout values."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            ProviderConfig(timeout=0)
        
        with pytest.raises(ValueError, match="timeout must be positive"):
            ProviderConfig(timeout=-10)
    
    def test_invalid_max_retries(self):
        """Test invalid max_retries values."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            ProviderConfig(max_retries=-1)


class TestLLMProvider:
    """Test the abstract LLMProvider base class."""
    
    def test_abstract_class_cannot_be_instantiated(self):
        """Test that abstract base class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMProvider("test-model")
    
    def test_provider_missing_methods_cannot_be_instantiated(self):
        """Test that providers missing required methods cannot be instantiated."""
        with pytest.raises(TypeError):
            InvalidProvider("test-model")
    
    def test_mock_provider_initialization(self):
        """Test successful initialization of a mock provider."""
        provider = MockProvider("mock-model-1")
        assert provider.model_name == "mock-model-1"
        assert provider.provider_name == "mock"
        assert provider.supported_models == ['mock-model-1', 'mock-model-2']
        assert isinstance(provider.config, ProviderConfig)
    
    def test_unsupported_model_error(self):
        """Test error when initializing with unsupported model."""
        with pytest.raises(ModelNotSupportedError) as exc_info:
            MockProvider("unsupported-model")
        
        error = exc_info.value
        assert "unsupported-model" in str(error)
        assert "mock" in str(error)
        assert "mock-model-1" in str(error)
    
    def test_provider_with_custom_config(self):
        """Test provider initialization with custom configuration."""
        provider = MockProvider(
            "mock-model-1",
            temperature=0.5,
            max_tokens=2000,
            custom_param="custom_value"
        )
        assert provider.config.temperature == 0.5
        assert provider.config.max_tokens == 2000
        assert provider.config.additional_params["custom_param"] == "custom_value"
    
    def test_provider_with_invalid_config(self):
        """Test provider initialization with invalid configuration."""
        with pytest.raises(ProviderConfigurationError):
            MockProvider("mock-model-1", temperature=3.0)
    
    def test_generate_method(self):
        """Test the generate method."""
        provider = MockProvider("mock-model-1")
        response = provider.generate("Test prompt")
        assert response == "Mock response for: Test prompt"
    
    def test_get_model_info_method(self):
        """Test the get_model_info method."""
        provider = MockProvider("mock-model-1")
        info = provider.get_model_info()
        assert info["model_name"] == "mock-model-1"
        assert info["provider"] == "mock"
        assert info["max_tokens"] == 1000
        assert "text-generation" in info["capabilities"]
    
    def test_validate_credentials_method(self):
        """Test the validate_credentials method."""
        provider = MockProvider("mock-model-1")
        assert provider.validate_credentials() is True
    
    def test_batch_generate_method(self):
        """Test the batch_generate method."""
        provider = MockProvider("mock-model-1")
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = provider.batch_generate(prompts)
        
        assert len(responses) == 3
        assert responses[0] == "Mock response for: Prompt 1"
        assert responses[1] == "Mock response for: Prompt 2"
        assert responses[2] == "Mock response for: Prompt 3"
    
    def test_batch_generate_with_error(self):
        """Test batch_generate when one generation fails."""
        provider = MockProvider("mock-model-1")
        
        # Mock generate to fail on second prompt
        original_generate = provider.generate
        def mock_generate(prompt, **kwargs):
            if prompt == "Prompt 2":
                raise Exception("Generation failed")
            return original_generate(prompt, **kwargs)
        
        provider.generate = mock_generate
        
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = provider.batch_generate(prompts)
        
        assert len(responses) == 3
        assert responses[0] == "Mock response for: Prompt 1"
        assert responses[1] == ""  # Failed generation returns empty string
        assert responses[2] == "Mock response for: Prompt 3"
    
    def test_get_default_parameters(self):
        """Test the get_default_parameters method."""
        provider = MockProvider("mock-model-1")
        defaults = provider.get_default_parameters()
        
        assert defaults["temperature"] == 0.7
        assert defaults["max_tokens"] == 1000
        assert defaults["top_p"] == 1.0
    
    def test_str_representation(self):
        """Test string representation of provider."""
        provider = MockProvider("mock-model-1")
        assert str(provider) == "mock:mock-model-1"
    
    def test_repr_representation(self):
        """Test detailed representation of provider."""
        provider = MockProvider("mock-model-1")
        assert repr(provider) == "MockProvider(model_name='mock-model-1')"
    
    def test_initialize_method(self):
        """Test the initialize method."""
        provider = MockProvider("mock-model-1")
        assert provider._initialized is False
        
        provider.initialize()
        assert provider._initialized is True
        
        # Test that re-initialization is a no-op
        provider.initialize()
        assert provider._initialized is True
    
    def test_initialize_with_invalid_credentials(self):
        """Test initialize method with invalid credentials."""
        provider = MockProvider("mock-model-1")
        provider.validate_credentials = lambda: False
        
        with pytest.raises(InvalidCredentialsError) as exc_info:
            provider.initialize()
        
        error = exc_info.value
        assert "mock" in str(error)
        assert "Credential validation failed" in str(error)
    
    def test_validate_config_method(self):
        """Test the validate_config method."""
        provider = MockProvider("mock-model-1")
        
        # Valid config (dict)
        provider.validate_config({"key": "value"})
        
        # Invalid config (not a dict)
        with pytest.raises(ProviderConfigurationError) as exc_info:
            provider.validate_config("not a dict")
        
        error = exc_info.value
        assert "Configuration must be a dictionary" in str(error)


class TestProviderConfigParsing:
    """Test configuration parsing functionality."""
    
    def test_parse_config_with_known_params(self):
        """Test parsing configuration with known parameters."""
        provider = MockProvider("mock-model-1", temperature=0.5, max_tokens=2000)
        
        assert provider.config.temperature == 0.5
        assert provider.config.max_tokens == 2000
        assert provider.config.top_p == 1.0  # Default value
    
    def test_parse_config_with_unknown_params(self):
        """Test parsing configuration with unknown parameters."""
        provider = MockProvider(
            "mock-model-1",
            temperature=0.5,
            custom_param1="value1",
            custom_param2="value2"
        )
        
        assert provider.config.temperature == 0.5
        assert provider.config.additional_params["custom_param1"] == "value1"
        assert provider.config.additional_params["custom_param2"] == "value2"
    
    def test_parse_config_with_mixed_params(self):
        """Test parsing configuration with mix of known and unknown parameters."""
        provider = MockProvider(
            "mock-model-1",
            temperature=0.8,
            max_tokens=1500,
            top_p=0.95,
            timeout=45,
            max_retries=4,
            retry_delay=1.5,
            api_key="test-key",
            custom_setting=True
        )
        
        # Check known parameters
        assert provider.config.temperature == 0.8
        assert provider.config.max_tokens == 1500
        assert provider.config.top_p == 0.95
        assert provider.config.timeout == 45
        assert provider.config.max_retries == 4
        assert provider.config.retry_delay == 1.5
        
        # Check unknown parameters
        assert provider.config.additional_params["api_key"] == "test-key"
        assert provider.config.additional_params["custom_setting"] is True