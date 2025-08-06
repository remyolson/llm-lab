"""
Unified Local LLM Provider

This module provides a unified interface for loading and running local models
across multiple backends (Transformers, llama.cpp, Ollama) with intelligent
backend selection and resource management.
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import logging
import time
from typing import Any, Dict, List, Optional

from ..base import LLMProvider, ProviderConfig
from ..exceptions import (
    ModelNotSupportedError,
    ProviderConfigurationError,
    ProviderError,
)
from .backends import GenerationConfig, LocalBackend, ModelInfo
from .registry import ModelRegistry

logger = logging.getLogger(__name__)


class UnifiedLocalProvider(LLMProvider):
    """
    Unified provider for local models supporting multiple backends.

    This provider automatically discovers available local models and selects
    the best backend for loading and running them. It supports intelligent
    resource management, automatic fallback, and concurrent model execution
    where possible.
    """

    # Supported model patterns (for registration)
    SUPPORTED_MODELS = [
        # Pattern-based matching for flexible model support
        "local:*",
        "transformers:*",
        "llamacpp:*",
        "ollama:*",
        # Direct model names from small-llms
        "pythia-70m",
        "pythia-160m",
        "smollm-135m",
        "smollm-360m",
        "qwen-0.5b",
        "llama3.2:1b",
        "gpt-oss:20b",
    ]

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the unified local provider.

        Args:
            model_name: Name or pattern of the model to load
            preferred_backend: Preferred backend name (optional)
            memory_threshold: Memory usage threshold (0.8 = 80%)
            auto_unload: Whether to automatically unload unused models
            **kwargs: Additional configuration parameters
        """
        # Initialize registry first
        self.registry = ModelRegistry()
        self.preferred_backend = kwargs.pop("preferred_backend", None)
        self.memory_threshold = kwargs.pop("memory_threshold", 0.8)
        self.auto_unload = kwargs.pop("auto_unload", True)

        # Current model state
        self._current_backend: LocalBackend | None = None
        self._current_model_id: str | None = None
        self._current_model_info: ModelInfo | None = None

        # Resource management
        self._loaded_models: Dict[
            str, Dict[str, Any]
        ] = {}  # model_id -> {backend, info, last_used}

        super().__init__(model_name, **kwargs)

        # Discover and validate model
        self._discover_and_validate_model()

    def _discover_and_validate_model(self) -> None:
        """Discover available models and validate the requested model."""
        # Discover models
        logger.info("Discovering local models...")
        discovered = self.registry.discover_models()

        total_models = sum(len(models) for models in discovered.values())
        logger.info(f"Found {total_models} local models across {len(discovered)} backends")

        # Find the requested model
        self._current_model_info = self._resolve_model_name(self.model_name)
        if not self._current_model_info:
            available_models = [model.name for model in self.registry.list_models()]
            raise ModelNotSupportedError(
                model_name=self.model_name,
                provider_name="unified_local",
                details=f"Available models: {', '.join(available_models[:10])}",
            )

        # Select backend
        self._select_backend()

        logger.info(f"Selected {self._current_backend.name} backend for {self.model_name}")

    def _resolve_model_name(self, model_name: str) -> ModelInfo | None:
        """Resolve model name to ModelInfo."""
        # Handle prefixed names (e.g., "ollama:llama3.2:1b")
        if ":" in model_name:
            parts = model_name.split(":", 1)
            if len(parts) == 2:
                backend_hint, actual_name = parts
                # Try to find model with this backend hint
                models = self.registry.list_models(backend_name=backend_hint)
                for model in models:
                    if model.name == actual_name or actual_name in model.name:
                        return model

        # Try exact match
        model_info = self.registry.get_model(model_name)
        if model_info:
            return model_info

        # Try partial match
        matches = self.registry.find_models_by_name(model_name)
        if matches:
            # Return the first match, or prefer smaller models for better performance
            return min(matches, key=lambda m: m.size_mb or float("inf"))

        return None

    def _select_backend(self) -> None:
        """Select the best backend for the current model."""
        if not self._current_model_info:
            raise ProviderConfigurationError(
                provider_name="unified_local",
                config_issue="No model info available for backend selection",
            )

        # Get compatible backends
        compatible_backends = self.registry.get_compatible_backends(self._current_model_info)
        if not compatible_backends:
            raise ProviderConfigurationError(
                provider_name="unified_local",
                config_issue=f"No compatible backends found for {self._current_model_info.format}",
            )

        # Use preferred backend if specified and compatible
        if self.preferred_backend and self.preferred_backend in compatible_backends:
            backend_name = self.preferred_backend
        else:
            # Use registry's recommendation
            backend_name = self.registry.get_best_backend_for_model(self._current_model_info)

        # Get backend instance
        self._current_backend = self.registry.get_backend(backend_name)
        if not self._current_backend:
            raise ProviderError(
                provider_name="unified_local",
                error_type="backend_error",
                details=f"Backend {backend_name} not available",
            )

    def initialize(self) -> None:
        """Initialize the provider and load the model."""
        if self._initialized:
            return

        # Check memory before loading
        if self.auto_unload:
            self._manage_memory()

        try:
            # Load the model
            self._current_model_id = self._current_backend.load_model(
                self._current_model_info, **self.config.additional_params
            )

            # Track loaded model
            self._loaded_models[self._current_model_id] = {
                "backend": self._current_backend,
                "info": self._current_model_info,
                "last_used": time.time(),
            }

            self._initialized = True
            logger.info(f"Successfully initialized {self.model_name}")

        except Exception as e:
            # Try fallback backends if available
            if not self._try_fallback_backends():
                raise ProviderError(
                    provider_name="unified_local",
                    error_type="initialization_error",
                    details=f"Failed to initialize model: {e}",
                )

    def _try_fallback_backends(self) -> bool:
        """Try loading the model with alternative backends."""
        if not self._current_model_info:
            return False

        compatible_backends = self.registry.get_compatible_backends(self._current_model_info)
        current_backend_name = self._current_backend.name if self._current_backend else None

        # Try other compatible backends
        for backend_name in compatible_backends:
            if backend_name == current_backend_name:
                continue  # Skip current backend

            backend = self.registry.get_backend(backend_name)
            if not backend:
                continue

            try:
                logger.info(f"Trying fallback backend: {backend_name}")
                model_id = backend.load_model(
                    self._current_model_info, **self.config.additional_params
                )

                # Success! Switch to this backend
                self._current_backend = backend
                self._current_model_id = model_id

                self._loaded_models[model_id] = {
                    "backend": backend,
                    "info": self._current_model_info,
                    "last_used": time.time(),
                }

                self._initialized = True
                logger.info(f"Successfully loaded with fallback backend: {backend_name}")
                return True

            except Exception as e:
                logger.warning(f"Fallback backend {backend_name} failed: {e}")
                continue

        return False

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the loaded model."""
        if not self._initialized or not self._current_backend or not self._current_model_id:
            raise ProviderError(
                provider_name="unified_local",
                error_type="not_initialized",
                details="Provider not initialized. Call initialize() first.",
            )

        # Update last used time
        if self._current_model_id in self._loaded_models:
            self._loaded_models[self._current_model_id]["last_used"] = time.time()

        # Prepare generation config
        generation_config = GenerationConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=kwargs.get("top_k", self.config.top_k),
            stop=kwargs.get("stop"),
            stream=kwargs.get("stream", False),
            seed=kwargs.get("seed"),
            repeat_penalty=kwargs.get("repeat_penalty", 1.1),
            extra_params=kwargs.get("extra_params", {}),
        )

        try:
            start_time = time.time()

            # Generate response
            response = self._current_backend.generate(
                self._current_model_id, prompt, generation_config
            )

            generation_time = time.time() - start_time
            logger.debug(f"Generation completed in {generation_time:.2f}s")

            # Handle streaming vs regular response
            if generation_config.stream:
                # For streaming, we need to collect the full response
                if hasattr(response, "__iter__") and not isinstance(response, str):
                    collected_response = "".join(str(chunk) for chunk in response)
                    return collected_response

            return str(response)

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise ProviderError(
                provider_name="unified_local",
                error_type="generation_error",
                details=f"Failed to generate response: {e}",
            )

    def get_model_info(self) -> Dict[str | Any]:
        """Get information about the current model."""
        base_info = {
            "model_name": self.model_name,
            "provider": "unified_local",
            "backend": self._current_backend.name if self._current_backend else None,
            "model_loaded": self._initialized,
        }

        if self._current_model_info:
            base_info.update(
                {
                    "model_path": str(self._current_model_info.path),
                    "model_format": self._current_model_info.format.value,
                    "model_size_mb": self._current_model_info.size_mb,
                    "parameters": self._current_model_info.parameters,
                    "context_length": self._current_model_info.context_length,
                    "description": self._current_model_info.description,
                }
            )

        # Add backend-specific info
        if self._current_backend and self._current_model_id:
            try:
                backend_info = self._current_backend.get_model_info(self._current_model_id)
                if backend_info:
                    base_info["backend_info"] = backend_info
            except:
                pass

        # Add resource usage
        base_info["memory_usage"] = self.get_memory_usage()

        # Add loaded models summary
        base_info["loaded_models_count"] = len(self._loaded_models)

        return base_info

    def validate_credentials(self) -> bool:
        """Validate that local models can be accessed."""
        # For local models, check that we have available backends and models
        available_backends = self.registry.get_available_backends()
        if not available_backends:
            logger.error("No local model backends available")
            return False

        # Check that we can discover at least some models
        try:
            models = self.registry.list_models()
            if not models:
                logger.warning("No local models discovered")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating local model setup: {e}")
            return False

    def get_memory_usage(self) -> Dict[str | float]:
        """Get current memory usage across all loaded models."""
        total_usage = {"ram": 0.0, "vram": 0.0}

        for model_id, model_data in self._loaded_models.items():
            try:
                backend = model_data["backend"]
                usage = backend.get_model_memory_usage(model_id)
                total_usage["ram"] += usage.get("ram", 0.0)
                total_usage["vram"] += usage.get("vram", 0.0)
            except Exception as e:
                logger.warning(f"Could not get memory usage for {model_id}: {e}")

        return total_usage

    def _manage_memory(self) -> None:
        """Manage memory by unloading least recently used models if needed."""
        if not self.auto_unload:
            return

        try:
            # Get current memory usage
            current_usage = self.get_memory_usage()
            total_memory_mb = current_usage["ram"] + current_usage["vram"]

            # Rough estimate of system memory (this could be improved)
            try:
                import psutil

                system_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
                memory_ratio = total_memory_mb / system_memory_mb
            except ImportError:
                # Fallback: assume 8GB system memory
                system_memory_mb = 8 * 1024
                memory_ratio = total_memory_mb / system_memory_mb

            # Unload models if we're above threshold
            if memory_ratio > self.memory_threshold:
                self._unload_least_recently_used()

        except Exception as e:
            logger.warning(f"Error during memory management: {e}")

    def _unload_least_recently_used(self) -> None:
        """Unload the least recently used models."""
        if len(self._loaded_models) <= 1:
            return  # Keep at least one model loaded

        # Sort by last used time
        sorted_models = sorted(self._loaded_models.items(), key=lambda x: x[1]["last_used"])

        # Unload oldest models (keep current model)
        for model_id, model_data in sorted_models[:-1]:
            if model_id != self._current_model_id:
                try:
                    backend = model_data["backend"]
                    backend.unload_model(model_id)
                    del self._loaded_models[model_id]
                    logger.info(f"Unloaded LRU model: {model_id}")
                    break  # Only unload one at a time
                except Exception as e:
                    logger.warning(f"Failed to unload model {model_id}: {e}")

    def list_available_models(self) -> List[Dict[str | Any]]:
        """List all available local models."""
        models = []
        for model_info in self.registry.list_models():
            models.append(
                {
                    "name": model_info.name,
                    "format": model_info.format.value,
                    "size_mb": model_info.size_mb,
                    "parameters": model_info.parameters,
                    "backends": self.registry.get_compatible_backends(model_info),
                    "description": model_info.description,
                }
            )
        return models

    def switch_model(self, model_name: str, **kwargs) -> None:
        """Switch to a different model."""
        # Save current state
        old_model_name = self.model_name
        old_backend = self._current_backend
        old_model_id = self._current_model_id

        try:
            # Update model name and re-initialize
            self.model_name = model_name
            self._initialized = False
            self._current_backend = None
            self._current_model_id = None
            self._current_model_info = None

            # Update config with new kwargs
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

            # Discover and initialize new model
            self._discover_and_validate_model()
            self.initialize()

            logger.info(f"Successfully switched from {old_model_name} to {model_name}")

        except Exception as e:
            # Restore previous state on failure
            self.model_name = old_model_name
            self._current_backend = old_backend
            self._current_model_id = old_model_id
            self._initialized = old_model_id is not None

            raise ProviderError(
                provider_name="unified_local",
                error_type="model_switch_error",
                details=f"Failed to switch to {model_name}: {e}",
            )

    def cleanup(self) -> None:
        """Clean up all loaded models and resources."""
        for model_id, model_data in self._loaded_models.items():
            try:
                backend = model_data["backend"]
                backend.unload_model(model_id)
            except Exception as e:
                logger.warning(f"Error unloading model {model_id}: {e}")

        self._loaded_models.clear()
        self._current_backend = None
        self._current_model_id = None
        self._initialized = False

        logger.info("Cleaned up all local models")

    def __del__(self):
        """Cleanup when provider is destroyed."""
        try:
            self.cleanup()
        except:
            pass
