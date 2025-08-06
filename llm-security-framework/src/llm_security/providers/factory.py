"""Model adapter factory for creating and managing adapters."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from ..config import ConfigLoader, ProviderConfig
from .anthropic_adapter import AnthropicAdapter
from .async_adapter import AsyncOpenAIAdapter
from .azure_adapter import AzureOpenAIAdapter
from .base import ModelAdapter, ProviderType
from .fingerprinting import ModelFingerprint, ModelRegistry
from .local_adapter import LocalModelAdapter
from .openai_adapter import OpenAIAdapter

logger = logging.getLogger(__name__)


class ModelAdapterFactory:
    """Factory for creating and managing model adapters."""

    # Registry of adapter classes
    _adapters: Dict[str, Type[ModelAdapter]] = {
        "openai": OpenAIAdapter,
        "anthropic": AnthropicAdapter,
        "azure_openai": AzureOpenAIAdapter,
        "local": LocalModelAdapter,
    }

    # Registry of async adapter classes
    _async_adapters: Dict[str, Type] = {
        "openai": AsyncOpenAIAdapter,
        # Can add more async adapters here
    }

    def __init__(
        self,
        config_loader: Optional[ConfigLoader] = None,
        model_registry: Optional[ModelRegistry] = None,
    ):
        """
        Initialize adapter factory.

        Args:
            config_loader: Configuration loader
            model_registry: Model registry for fingerprinting
        """
        self.config_loader = config_loader or ConfigLoader()
        self.model_registry = model_registry or ModelRegistry()
        self._adapter_cache: Dict[str, ModelAdapter] = {}

    @classmethod
    def register_adapter(
        cls,
        provider_type: str,
        adapter_class: Type[ModelAdapter],
        async_class: Optional[Type] = None,
    ) -> None:
        """
        Register a new adapter type.

        Args:
            provider_type: Provider type name
            adapter_class: Adapter class
            async_class: Optional async adapter class
        """
        cls._adapters[provider_type] = adapter_class
        if async_class:
            cls._async_adapters[provider_type] = async_class

        logger.info(f"Registered adapter for provider: {provider_type}")

    def create_adapter(
        self,
        provider: str,
        model_name: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> ModelAdapter:
        """
        Create a model adapter.

        Args:
            provider: Provider name
            model_name: Optional model name override
            config_overrides: Configuration overrides
            use_cache: Whether to use cached adapter

        Returns:
            ModelAdapter instance

        Raises:
            ValueError: If provider is not supported
        """
        # Create cache key
        cache_key = f"{provider}:{model_name or 'default'}"

        # Check cache
        if use_cache and cache_key in self._adapter_cache:
            return self._adapter_cache[cache_key]

        # Get provider configuration
        config = self.config_loader.get_config(provider)
        if not config:
            raise ValueError(f"No configuration found for provider: {provider}")

        # Apply overrides
        config_dict = config.to_dict()
        if model_name:
            config_dict["model_name"] = model_name
        if config_overrides:
            config_dict.update(config_overrides)

        # Get adapter class
        provider_type = config.provider_type
        adapter_class = self._adapters.get(provider_type)

        if not adapter_class:
            raise ValueError(f"No adapter available for provider type: {provider_type}")

        # Create adapter
        adapter = adapter_class(config_dict)

        # Register in model registry
        fingerprint = self.model_registry.create_fingerprint_from_adapter(
            adapter,
            test_capabilities=False,  # Skip testing for factory creation
        )

        logger.info(f"Created adapter for {provider} (fingerprint: {fingerprint.fingerprint_hash})")

        # Cache adapter
        if use_cache:
            self._adapter_cache[cache_key] = adapter

        return adapter

    def create_async_adapter(
        self,
        provider: str,
        model_name: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        Create an async model adapter.

        Args:
            provider: Provider name
            model_name: Optional model name override
            config_overrides: Configuration overrides

        Returns:
            Async adapter instance
        """
        # Get provider configuration
        config = self.config_loader.get_config(provider)
        if not config:
            raise ValueError(f"No configuration found for provider: {provider}")

        # Apply overrides
        config_dict = config.to_dict()
        if model_name:
            config_dict["model_name"] = model_name
        if config_overrides:
            config_dict.update(config_overrides)

        # Get async adapter class
        provider_type = config.provider_type
        adapter_class = self._async_adapters.get(provider_type)

        if not adapter_class:
            raise ValueError(f"No async adapter available for provider type: {provider_type}")

        # Create adapter
        adapter = adapter_class(config_dict)

        logger.info(f"Created async adapter for {provider}")
        return adapter

    def get_available_providers(self) -> list[str]:
        """Get list of available providers."""
        return self.config_loader.list_providers()

    def get_available_models(
        self, provider: Optional[str] = None, capability_filter: Optional[Dict[str, Any]] = None
    ) -> list[ModelFingerprint]:
        """
        Get available models, optionally filtered.

        Args:
            provider: Provider to filter by
            capability_filter: Capability requirements

        Returns:
            List of model fingerprints
        """
        return self.model_registry.find_fingerprints(
            provider=provider, capability_filter=capability_filter
        )

    def create_best_adapter(
        self, requirements: Dict[str, Any], preference_order: Optional[list[str]] = None
    ) -> Optional[ModelAdapter]:
        """
        Create the best adapter for given requirements.

        Args:
            requirements: Model requirements
            preference_order: Preferred provider order

        Returns:
            Best matching adapter or None
        """
        # Find compatible models
        compatible_models = self.model_registry.get_compatible_models(requirements)

        if not compatible_models:
            logger.warning("No compatible models found for requirements")
            return None

        # Apply preference order
        if preference_order:

            def sort_key(fp: ModelFingerprint) -> int:
                try:
                    return preference_order.index(fp.provider)
                except ValueError:
                    return len(preference_order)  # Put unknown providers at end

            compatible_models.sort(key=sort_key)

        # Try to create adapter for best match
        for fingerprint in compatible_models:
            try:
                adapter = self.create_adapter(fingerprint.provider)
                logger.info(
                    f"Selected {fingerprint.provider}/{fingerprint.model_name} for requirements"
                )
                return adapter
            except Exception as e:
                logger.warning(f"Failed to create adapter for {fingerprint.provider}: {e}")
                continue

        return None

    def create_multi_provider_adapter(
        self,
        providers: list[str],
        fallback_strategy: str = "round_robin",  # round_robin, failover, load_balance
    ) -> "MultiProviderAdapter":
        """
        Create adapter that can use multiple providers.

        Args:
            providers: List of provider names
            fallback_strategy: Strategy for provider selection

        Returns:
            MultiProviderAdapter instance
        """
        adapters = []
        for provider in providers:
            try:
                adapter = self.create_adapter(provider)
                adapters.append(adapter)
            except Exception as e:
                logger.warning(f"Failed to create adapter for {provider}: {e}")

        if not adapters:
            raise ValueError("No adapters could be created")

        return MultiProviderAdapter(adapters, fallback_strategy)

    def clear_cache(self) -> None:
        """Clear adapter cache."""
        self._adapter_cache.clear()
        logger.info("Cleared adapter cache")

    def get_factory_stats(self) -> Dict[str, Any]:
        """Get factory statistics."""
        return {
            "available_providers": len(self._adapters),
            "configured_providers": len(self.config_loader.list_providers()),
            "cached_adapters": len(self._adapter_cache),
            "registered_models": len(self.model_registry._fingerprints),
            "async_providers": len(self._async_adapters),
        }


class MultiProviderAdapter(ModelAdapter):
    """Adapter that can use multiple providers with fallback strategies."""

    def __init__(self, adapters: list[ModelAdapter], strategy: str = "round_robin"):
        """
        Initialize multi-provider adapter.

        Args:
            adapters: List of adapters
            strategy: Fallback strategy
        """
        super().__init__()
        self.adapters = adapters
        self.strategy = strategy
        self.current_index = 0
        self._provider_type = ProviderType.CUSTOM
        self._model_name = "multi-provider"

    def configure(self, config_dict: Dict[str, Any]) -> None:
        """Configure all adapters."""
        for adapter in self.adapters:
            adapter.configure(config_dict)

    def send_prompt(self, prompt: Union[str, list[Dict[str, str]]], **kwargs):
        """Send prompt using strategy."""
        if self.strategy == "round_robin":
            return self._round_robin_send(prompt, **kwargs)
        elif self.strategy == "failover":
            return self._failover_send(prompt, **kwargs)
        elif self.strategy == "load_balance":
            return self._load_balance_send(prompt, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _round_robin_send(self, prompt, **kwargs):
        """Round-robin strategy."""
        adapter = self.adapters[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.adapters)
        return adapter.send_prompt(prompt, **kwargs)

    def _failover_send(self, prompt, **kwargs):
        """Failover strategy - try adapters in order until success."""
        last_error = None

        for adapter in self.adapters:
            try:
                response = adapter.send_prompt(prompt, **kwargs)
                if response.is_success:
                    return response
                last_error = response.error_message
            except Exception as e:
                last_error = str(e)
                continue

        # All adapters failed
        return ModelResponse(
            content="",
            model=self._model_name,
            provider="multi-provider",
            request_id=self.generate_request_id(),
            status=ResponseStatus.ERROR,
            error_message=f"All providers failed. Last error: {last_error}",
        )

    def _load_balance_send(self, prompt, **kwargs):
        """Load balancing based on adapter metrics."""
        # Sort by success rate and latency
        sorted_adapters = sorted(
            self.adapters,
            key=lambda a: (
                -a.get_metrics().get("success_rate", 0),
                a.get_metrics().get("average_latency_ms", float("inf")),
            ),
        )

        # Use best performing adapter
        return sorted_adapters[0].send_prompt(prompt, **kwargs)

    def get_response(self, prompt_id: str):
        """Not implemented for multi-provider."""
        return None

    def handle_errors(self, error: Exception):
        """Handle errors."""
        return ModelError(
            message=str(error), provider="multi-provider", error_type=ResponseStatus.ERROR
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get info for all adapters."""
        return {
            "provider": "multi-provider",
            "model": "multiple",
            "strategy": self.strategy,
            "adapters": [adapter.get_model_info() for adapter in self.adapters],
            "metrics": self.get_metrics(),
        }
