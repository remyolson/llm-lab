"""
Base LLM Provider Interface

This module defines the abstract base class for all LLM providers in the LLM Lab framework.
It provides a consistent interface that all provider implementations must follow, ensuring
compatibility and standardization across different AI model providers.

Key features:
- Abstract base class with enforced method implementation
- Type hints for all methods and return values
- Comprehensive error handling through custom exceptions
- Configuration validation framework
- Provider metadata and model information
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging
from dataclasses import dataclass, field

from .exceptions import ProviderConfigurationError, InvalidCredentialsError

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """
    Configuration dataclass for provider settings.
    
    This provides a type-safe way to handle provider configurations
    with validation and default values.
    """
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    top_k: Optional[int] = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration values."""
        if not 0 <= self.temperature <= 2:
            raise ValueError(f"Temperature must be between 0 and 2, got {self.temperature}")
        
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        
        if not 0 <= self.top_p <= 1:
            raise ValueError(f"top_p must be between 0 and 1, got {self.top_p}")
        
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {self.max_retries}")


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    This class defines the interface that all LLM provider implementations must follow.
    It ensures consistent behavior across different providers and enables easy addition
    of new providers to the framework.
    
    Attributes:
        model_name: The name of the model being used
        provider_name: The name of the provider (e.g., "openai", "anthropic")
        supported_models: List of model names supported by this provider
        config: Configuration dictionary for the provider
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the LLM provider.
        
        Args:
            model_name: The name of the model to use
            **kwargs: Additional configuration parameters specific to the provider
        """
        self.model_name = model_name
        self.provider_name = self.__class__.__name__.replace('Provider', '').lower()
        self.supported_models: List[str] = []
        
        # Validate that the model is supported
        if hasattr(self.__class__, 'SUPPORTED_MODELS'):
            self.supported_models = self.__class__.SUPPORTED_MODELS
            if model_name not in self.supported_models:
                from .exceptions import ModelNotSupportedError
                raise ModelNotSupportedError(
                    model_name=model_name,
                    provider_name=self.provider_name,
                    supported_models=self.supported_models
                )
        
        # Load configuration from config manager
        config_from_manager = self._load_config_from_manager(model_name)
        
        # Merge configuration (kwargs override config file settings)
        merged_config = {**config_from_manager, **kwargs}
        
        # Parse and validate configuration
        try:
            self.config = self._parse_config(merged_config)
        except Exception as e:
            raise ProviderConfigurationError(
                provider_name=self.provider_name,
                config_issue=str(e)
            )
        
        # Validate credentials on initialization
        self._initialized = False
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The input prompt to send to the model
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            The generated text response from the model
            
        Raises:
            ProviderError: If generation fails
            RateLimitError: If rate limits are exceeded
            InvalidCredentialsError: If authentication fails
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary containing model metadata such as:
            - model_name: The name of the model
            - provider: The provider name
            - max_tokens: Maximum token limit
            - capabilities: List of model capabilities
            - version: Model version if available
        """
        pass
    
    @abstractmethod
    def validate_credentials(self) -> bool:
        """
        Validate that the provider credentials are properly configured.
        
        Returns:
            True if credentials are valid, False otherwise
            
        Raises:
            InvalidCredentialsError: If credentials are missing or invalid
        """
        pass
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Default implementation calls generate() for each prompt sequentially.
        Providers can override this for more efficient batch processing.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        responses = []
        for prompt in prompts:
            try:
                response = self.generate(prompt, **kwargs)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error generating response for prompt: {e}")
                responses.append("")  # Add empty response for failed generation
        return responses
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get default generation parameters for this provider.
        
        Returns:
            Dictionary of default parameters
        """
        return {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0,
        }
    
    def __str__(self) -> str:
        """String representation of the provider."""
        return f"{self.provider_name}:{self.model_name}"
    
    def __repr__(self) -> str:
        """Detailed representation of the provider."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
    
    def _load_config_from_manager(self, model_name: str) -> Dict[str, Any]:
        """
        Load configuration from the configuration manager.
        
        Args:
            model_name: The model name to load config for
            
        Returns:
            Configuration dictionary
        """
        try:
            from config.provider_config import get_config_manager
            config_manager = get_config_manager()
            
            # Get model configuration
            model_config = config_manager.get_model_config(model_name)
            
            # Extract parameters
            if 'parameters' in model_config:
                return model_config['parameters']
            
        except Exception as e:
            logger.debug(f"Failed to load config from manager: {e}")
        
        return {}
    
    def _parse_config(self, kwargs: Dict[str, Any]) -> ProviderConfig:
        """
        Parse and validate configuration parameters.
        
        Args:
            kwargs: Raw configuration parameters
            
        Returns:
            Validated ProviderConfig instance
            
        Raises:
            ValueError: If configuration values are invalid
        """
        # Extract known parameters
        config_params = {}
        additional_params = {}
        
        known_params = {
            'temperature', 'max_tokens', 'top_p', 'top_k',
            'timeout', 'max_retries', 'retry_delay'
        }
        
        for key, value in kwargs.items():
            if key in known_params:
                config_params[key] = value
            else:
                additional_params[key] = value
        
        # Create config with additional parameters
        if additional_params:
            config_params['additional_params'] = additional_params
        
        return ProviderConfig(**config_params)
    
    def initialize(self) -> None:
        """
        Initialize the provider connection.
        
        This method should be called before first use to ensure
        credentials are valid and connections are established.
        
        Raises:
            InvalidCredentialsError: If credentials are invalid
            ProviderError: If initialization fails
        """
        if self._initialized:
            return
        
        # Validate credentials
        if not self.validate_credentials():
            raise InvalidCredentialsError(
                provider_name=self.provider_name,
                details="Credential validation failed"
            )
        
        self._initialized = True
        logger.info(f"Successfully initialized {self.provider_name} provider")
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate provider-specific configuration.
        
        Subclasses can override this to add provider-specific validation.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ProviderConfigurationError: If configuration is invalid
        """
        # Base implementation just ensures config is a dict
        if not isinstance(config, dict):
            raise ProviderConfigurationError(
                provider_name=self.provider_name,
                config_issue="Configuration must be a dictionary"
            )