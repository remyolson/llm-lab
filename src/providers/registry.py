"""
Provider Registry

This module implements a registry system for dynamically loading and managing LLM providers.
It allows providers to self-register and enables easy provider selection based on model names.

The registry uses a singleton pattern to ensure a single source of truth for provider
mappings across the application.
"""

import logging
from typing import Dict, List, Type, Optional
from functools import wraps

from .base import LLMProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """
    Singleton registry for managing LLM providers.
    
    This class maintains a mapping between model names and their corresponding
    provider classes, enabling dynamic provider selection at runtime.
    """
    
    _instance = None
    _providers: Dict[str, Type[LLMProvider]] = {}
    _model_to_provider: Dict[str, str] = {}
    
    def __new__(cls):
        """Ensure only one instance of the registry exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register(self, provider_class: Type[LLMProvider], model_names: List[str]) -> None:
        """
        Register a provider class with its supported models.
        
        Args:
            provider_class: The provider class to register
            model_names: List of model names supported by this provider
            
        Raises:
            ValueError: If a model is already registered to another provider
        """
        provider_name = provider_class.__name__.replace('Provider', '').lower()
        
        # Check for conflicts
        for model_name in model_names:
            normalized_model = self._normalize_model_name(model_name)
            if normalized_model in self._model_to_provider:
                existing_provider = self._model_to_provider[normalized_model]
                if existing_provider != provider_name:
                    raise ValueError(
                        f"Model '{model_name}' is already registered to provider "
                        f"'{existing_provider}'. Cannot register to '{provider_name}'."
                    )
        
        # Register the provider
        self._providers[provider_name] = provider_class
        
        # Map models to provider
        for model_name in model_names:
            normalized_model = self._normalize_model_name(model_name)
            self._model_to_provider[normalized_model] = provider_name
            
        logger.info(
            f"Registered provider '{provider_name}' with models: {', '.join(model_names)}"
        )
    
    def get_provider(self, model_name: str) -> Optional[Type[LLMProvider]]:
        """
        Get the provider class for a given model name.
        
        Args:
            model_name: The name of the model
            
        Returns:
            The provider class if found, None otherwise
        """
        normalized_model = self._normalize_model_name(model_name)
        
        # Check direct model mapping
        if normalized_model in self._model_to_provider:
            provider_name = self._model_to_provider[normalized_model]
            return self._providers.get(provider_name)
        
        # Check if model name contains provider prefix
        for provider_name, provider_class in self._providers.items():
            if model_name.lower().startswith(provider_name):
                return provider_class
        
        return None
    
    def list_providers(self) -> Dict[str, List[str]]:
        """
        List all registered providers and their supported models.
        
        Returns:
            Dictionary mapping provider names to lists of supported models
        """
        result = {}
        for model_name, provider_name in self._model_to_provider.items():
            if provider_name not in result:
                result[provider_name] = []
            # Denormalize the model name for display
            result[provider_name].append(model_name.replace('_', '-'))
        return result
    
    def list_all_models(self) -> List[str]:
        """
        List all supported models across all providers.
        
        Returns:
            List of all supported model names
        """
        models = []
        for model_name in self._model_to_provider.keys():
            # Denormalize the model name for display
            models.append(model_name.replace('_', '-'))
        return sorted(models)
    
    def clear(self) -> None:
        """Clear all registered providers (mainly for testing)."""
        self._providers.clear()
        self._model_to_provider.clear()
    
    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        """
        Normalize model names for consistent matching.
        
        Args:
            model_name: The model name to normalize
            
        Returns:
            Normalized model name
        """
        # Convert to lowercase and replace hyphens with underscores
        return model_name.lower().replace('-', '_').strip()


# Global registry instance
registry = ProviderRegistry()


def register_provider(models: List[str]):
    """
    Decorator for automatically registering providers.
    
    Usage:
        @register_provider(models=['gpt-4', 'gpt-3.5-turbo'])
        class OpenAIProvider(LLMProvider):
            ...
    
    Args:
        models: List of model names supported by the provider
    """
    def decorator(provider_class: Type[LLMProvider]):
        # Set supported models as class attribute
        provider_class.SUPPORTED_MODELS = models
        
        # Register with the global registry
        registry.register(provider_class, models)
        
        return provider_class
    
    return decorator


def get_provider_for_model(model_name: str) -> Type[LLMProvider]:
    """
    Get the provider class for a given model name.
    
    This function first checks the configuration manager for model aliases,
    then falls back to the registry's direct model mappings.
    
    Args:
        model_name: The name of the model (can be an alias)
        
    Returns:
        The provider class
        
    Raises:
        ValueError: If no provider is found for the model
    """
    # Try to resolve alias through configuration manager
    try:
        from src.config.provider_config import get_config_manager
        config_manager = get_config_manager()
        
        # Resolve any aliases
        resolved_name = config_manager.resolve_model_alias(model_name)
        
        # First try the resolved name
        provider_class = registry.get_provider(resolved_name)
        if provider_class is not None:
            return provider_class
            
        # If not found, check if config knows about this model
        provider_name = config_manager.get_provider_for_model(resolved_name)
        if provider_name and provider_name in registry._providers:
            return registry._providers[provider_name]
    except Exception as e:
        logger.debug(f"Config manager lookup failed: {e}")
    
    # Fall back to direct registry lookup with original name
    provider_class = registry.get_provider(model_name)
    if provider_class is None:
        available_models = registry.list_all_models()
        
        # Also list available aliases from config
        try:
            from src.config.provider_config import get_config_manager
            config_manager = get_config_manager()
            all_models = config_manager.get_available_models(include_aliases=True)
            
            # Flatten the model lists
            config_models = []
            for provider_models in all_models.values():
                config_models.extend(provider_models)
            
            # Combine and deduplicate
            all_available = sorted(set(available_models + config_models))
            
            raise ValueError(
                f"No provider found for model '{model_name}'. "
                f"Available models and aliases: {', '.join(all_available)}"
            )
        except Exception:
            # If config loading fails, just use registry models
            raise ValueError(
                f"No provider found for model '{model_name}'. "
                f"Available models: {', '.join(available_models)}"
            )
    return provider_class