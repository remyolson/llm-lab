"""
Base Backend Interface for Local Models

This module defines the abstract interface that all local model backends must implement.
It provides a consistent API for model loading, inference, and management across different
backends (Transformers, llama.cpp, Ollama).
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List

logger = logging.getLogger(__name__)


class ModelFormat(Enum):
    """Supported model formats."""

    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    GGUF = "gguf"
    GGML = "ggml"
    OLLAMA = "ollama"


@dataclass
class BackendCapabilities:
    """Capabilities supported by a backend."""

    streaming: bool = False
    embeddings: bool = False
    function_calling: bool = False
    batch_generation: bool = False
    gpu_acceleration: bool = False
    quantization: bool = False
    concurrent_models: bool = False


@dataclass
class ModelInfo:
    """Information about a discovered model."""

    name: str
    path: str | Path
    format: ModelFormat
    size_mb: float | None = None
    parameters: int | None = None
    context_length: int | None = None
    description: str | None = None
    capabilities: BackendCapabilities | None = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.capabilities is None:
            self.capabilities = BackendCapabilities()


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    top_k: int | None = None
    stop: List[str | None] = None
    stream: bool = False
    seed: int | None = None
    repeat_penalty: float = 1.1
    # Backend-specific parameters
    extra_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


class LocalBackend(ABC):
    """
    Abstract base class for local model backends.

    Each backend (Transformers, llama.cpp, Ollama) implements this interface
    to provide a consistent API for model loading and inference.
    """

    def __init__(self, name: str):
        """
        Initialize the backend.

        Args:
            name: Name of the backend (e.g., "transformers", "llamacpp", "ollama")
        """
        self.name = name
        self._loaded_models: Dict[str, Any] = {}
        self._model_metadata: Dict[str, ModelInfo] = {}

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this backend is available on the system.

        Returns:
            True if backend can be used, False otherwise
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> BackendCapabilities:
        """
        Get the capabilities supported by this backend.

        Returns:
            BackendCapabilities object describing what this backend supports
        """
        pass

    @abstractmethod
    def discover_models(self, search_paths: List[str, Path]) -> List[ModelInfo]:
        """
        Discover available models in the given paths.

        Args:
            search_paths: List of directories to search for models

        Returns:
            List of discovered models
        """
        pass

    @abstractmethod
    def can_load_model(self, model_info: ModelInfo) -> bool:
        """
        Check if this backend can load the given model.

        Args:
            model_info: Information about the model

        Returns:
            True if backend can load this model, False otherwise
        """
        pass

    @abstractmethod
    def load_model(self, model_info: ModelInfo, **kwargs) -> str:
        """
        Load a model into memory.

        Args:
            model_info: Information about the model to load
            **kwargs: Backend-specific loading parameters

        Returns:
            Model ID that can be used for generation

        Raises:
            ModelLoadError: If model loading fails
        """
        pass

    @abstractmethod
    def unload_model(self, model_id: str) -> None:
        """
        Unload a model from memory.

        Args:
            model_id: ID of the model to unload
        """
        pass

    @abstractmethod
    def generate(self, model_id: str, prompt: str, config: GenerationConfig) -> str | Iterator[str]:
        """
        Generate text using the loaded model.

        Args:
            model_id: ID of the loaded model
            prompt: Input prompt
            config: Generation configuration

        Returns:
            Generated text (string) or iterator of tokens (if streaming)

        Raises:
            GenerationError: If generation fails
        """
        pass

    @abstractmethod
    def get_model_memory_usage(self, model_id: str) -> Dict[str, float]:
        """
        Get memory usage of a loaded model.

        Args:
            model_id: ID of the loaded model

        Returns:
            Dictionary with memory usage in MB (e.g., {"ram": 1024.0, "vram": 512.0})
        """
        pass

    def list_loaded_models(self) -> List[str]:
        """
        Get list of currently loaded model IDs.

        Returns:
            List of loaded model IDs
        """
        return list(self._loaded_models.keys())

    def is_model_loaded(self, model_id: str) -> bool:
        """
        Check if a model is currently loaded.

        Args:
            model_id: Model ID to check

        Returns:
            True if model is loaded, False otherwise
        """
        return model_id in self._loaded_models

    def get_model_info(self, model_id: str) -> ModelInfo | None:
        """
        Get information about a loaded model.

        Args:
            model_id: ID of the model

        Returns:
            ModelInfo if model is loaded, None otherwise
        """
        return self._model_metadata.get(model_id)

    def get_total_memory_usage(self) -> Dict[str, float]:
        """
        Get total memory usage across all loaded models.

        Returns:
            Dictionary with total memory usage in MB
        """
        total = {"ram": 0.0, "vram": 0.0}
        for model_id in self._loaded_models:
            try:
                usage = self.get_model_memory_usage(model_id)
                total["ram"] += usage.get("ram", 0.0)
                total["vram"] += usage.get("vram", 0.0)
            except Exception as e:
                logger.warning(f"Failed to get memory usage for {model_id}: {e}")
        return total

    def validate_generation_config(self, config: GenerationConfig) -> None:
        """
        Validate generation configuration for this backend.

        Args:
            config: Generation configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        if not 0 <= config.temperature <= 2:
            raise ValueError(f"Temperature must be between 0 and 2, got {config.temperature}")

        if config.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {config.max_tokens}")

        if not 0 <= config.top_p <= 1:
            raise ValueError(f"top_p must be between 0 and 1, got {config.top_p}")

    def __repr__(self) -> str:
        """String representation of the backend."""
        return f"{self.__class__.__name__}(name='{self.name}', loaded_models={len(self._loaded_models)})"
