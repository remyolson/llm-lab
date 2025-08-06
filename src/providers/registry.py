"""
Provider Registry

This module implements a registry system for dynamically loading and managing LLM providers.
It allows providers to self-register and enables easy provider selection based on model names.

The registry uses a singleton pattern to ensure a single source of truth for provider
mappings across the application.
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import logging
from typing import Dict, Generic, List, Optional, Type, TypeVar

from ..types.generics import GenericFactory, GenericRepository
from ..types.protocols import ProviderType, ResponseType
from .base import LLMProvider

logger = logging.getLogger(__name__)


ProviderT = TypeVar("ProviderT", bound=LLMProvider)


class ProviderRegistry(GenericRepository[str, Type[LLMProvider]]):
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

    def save(self, key: str, item: Type[LLMProvider]) -> None:
        """Generic save method for repository pattern.

        Args:
            key: Provider name as key
            item: Provider class to save
        """
        if key not in self._providers:
            self._providers[key] = item

    def get_by_id(self, key: str) -> Optional[Type[LLMProvider]]:
        """Generic get method for repository pattern.

        Args:
            key: Provider name

        Returns:
            Provider class if found, None otherwise
        """
        return self._providers.get(key)

    def delete(self, key: str) -> bool:
        """Generic delete method for repository pattern.

        Args:
            key: Provider name to delete

        Returns:
            True if deleted, False if not found
        """
        if key in self._providers:
            del self._providers[key]
            # Also clean up model mappings
            models_to_remove = [
                model for model, provider in self._model_to_provider.items() if provider == key
            ]
            for model in models_to_remove:
                del self._model_to_provider[model]
            return True
        return False

    def get_all(self) -> List[Type[LLMProvider]]:
        """Generic get_all method for repository pattern.

        Returns:
            List of all registered provider classes
        """
        return list(self._providers.values())

    def register(self, provider_class: Type[LLMProvider], model_names: List[str]) -> None:
        """
        Register a provider class with its supported models.

        Args:
            provider_class: The provider class to register
            model_names: List of model names supported by this provider

        Raises:
            ValueError: If a model is already registered to another provider
        """
        provider_name = provider_class.__name__.replace("Provider", "").lower()

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

        logger.info(f"Registered provider '{provider_name}' with models: {', '.join(model_names)}")

    def get_provider(self, model_name: str) -> Type[LLMProvider | None]:
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

    def list_providers(self) -> Dict[str | List[str]]:
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
            result[provider_name].append(model_name.replace("_", "-"))
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
            models.append(model_name.replace("_", "-"))
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
        return model_name.lower().replace("-", "_").strip()


# Global registry instance
registry = ProviderRegistry()

# Global provider factory
provider_factory = GenericProviderFactory(registry)


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
        from .config.provider_config import get_config_manager

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
            from .config.provider_config import get_config_manager

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


class GenericProviderFactory(GenericFactory[str, LLMProvider]):
    """Generic factory for creating LLM provider instances.

    Implements the GenericFactory pattern with type-safe provider creation.
    """

    def __init__(self, registry: ProviderRegistry):
        self.registry = registry
        self._provider_configs: Dict[str, Dict] = {}

    def create(self, identifier: str, **kwargs) -> LLMProvider:
        """Create a provider instance from a model name.

        Args:
            identifier: Model name or provider identifier
            **kwargs: Additional configuration for the provider

        Returns:
            Configured provider instance

        Raises:
            ValueError: If provider cannot be found or created
        """
        provider_class = get_provider_for_model(identifier)

        # Extract model name from identifier
        model_name = self._extract_model_name(identifier)

        # Get stored configuration for this provider if available
        provider_name = provider_class.__name__.replace("Provider", "").lower()
        config = self._provider_configs.get(provider_name, {})

        # Merge with provided kwargs (kwargs take precedence)
        final_config = {**config, **kwargs}

        try:
            return provider_class(model_name, **final_config)
        except Exception as e:
            raise ValueError(f"Failed to create provider for {identifier}: {e}")

    def set_provider_config(self, provider_name: str, config: Dict) -> None:
        """Set default configuration for a provider.

        Args:
            provider_name: Name of the provider
            config: Configuration dictionary
        """
        self._provider_configs[provider_name] = config

    def get_available_types(self) -> List[str]:
        """Get list of available provider types.

        Returns:
            List of provider type names
        """
        return list(self.registry._providers.keys())

    def _extract_model_name(self, identifier: str) -> str:
        """Extract model name from identifier.

        Args:
            identifier: Model identifier (could be prefixed)

        Returns:
            Clean model name
        """
        # Handle provider:model format
        if ":" in identifier:
            return identifier.split(":", 1)[1]

        # Handle provider-prefixed models
        for provider_name in self.registry._providers.keys():
            if identifier.lower().startswith(provider_name + "-"):
                return identifier[len(provider_name) + 1 :]

        return identifier
