"""
Model Registry for Local Models

This module provides automatic model discovery and management across
different local model backends (Transformers, llama.cpp, Ollama).
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from .backends import (
    LlamaCppBackend,
    LocalBackend,
    ModelInfo,
    OllamaBackend,
    TransformersBackend,
)

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for discovering and managing local models across multiple backends.

    The registry scans standard directories and backend-specific locations
    to build a comprehensive catalog of available local models.
    """

    def __init__(self, cache_file: str | None = None):
        """
        Initialize the model registry.

        Args:
            cache_file: Optional path to model cache file
        """
        self.cache_file = cache_file or os.path.expanduser("~/.cache/llm-lab/model_registry.json")

        # Initialize backends
        self.backends: List[LocalBackend] = []
        self._init_backends()

        # Model storage
        self._models: Dict[str, ModelInfo] = {}
        self._backend_models: Dict[str, List[str]] = {}  # backend_name -> model_names

        # Load cached models
        self._load_cache()

    def _init_backends(self) -> None:
        """Initialize available backends."""
        backend_classes = [
            TransformersBackend,
            LlamaCppBackend,
            OllamaBackend,
        ]

        for backend_class in backend_classes:
            try:
                backend = backend_class()
                if backend.is_available():
                    self.backends.append(backend)
                    self._backend_models[backend.name] = []
                    logger.info(f"Initialized {backend.name} backend")
                else:
                    logger.debug(f"{backend.name} backend not available")
            except Exception as e:
                logger.warning(f"Failed to initialize {backend_class.__name__}: {e}")

    def get_available_backends(self) -> List[str]:
        """Get list of available backend names."""
        return [backend.name for backend in self.backends]

    def get_backend(self, name: str) -> LocalBackend | None:
        """Get backend by name."""
        for backend in self.backends:
            if backend.name == name:
                return backend
        return None

    def discover_models(
        self, search_paths: List[str, Path | None] = None, force_refresh: bool = False
    ) -> Dict[str | List[ModelInfo]]:
        """
        Discover models across all backends.

        Args:
            search_paths: Custom search paths (uses defaults if None)
            force_refresh: Force rediscovery even if cache exists

        Returns:
            Dictionary mapping backend names to lists of discovered models
        """
        if not force_refresh and self._models:
            logger.info("Using cached model discovery results")
            return self._group_models_by_backend()

        if search_paths is None:
            search_paths = self._get_default_search_paths()

        logger.info(
            f"Discovering models in {len(search_paths)} paths across {len(self.backends)} backends"
        )

        # Clear existing models
        self._models.clear()
        for backend_name in self._backend_models:
            self._backend_models[backend_name].clear()

        # Discover models for each backend
        discovered_count = 0
        for backend in self.backends:
            try:
                models = backend.discover_models(search_paths)
                logger.info(f"Found {len(models)} models for {backend.name} backend")

                for model in models:
                    # Create unique model ID
                    model_id = f"{backend.name}_{model.name}"

                    # Handle name conflicts
                    counter = 1
                    original_id = model_id
                    while model_id in self._models:
                        model_id = f"{original_id}_{counter}"
                        counter += 1

                    self._models[model_id] = model
                    self._backend_models[backend.name].append(model_id)
                    discovered_count += 1

            except Exception as e:
                logger.error(f"Error during model discovery for {backend.name}: {e}")

        logger.info(f"Discovered {discovered_count} total models")

        # Save to cache
        self._save_cache()

        return self._group_models_by_backend()

    def _get_default_search_paths(self) -> List[Path]:
        """Get default search paths for model discovery."""
        paths = []

        # Common local model directories
        home = Path.home()

        # Hugging Face cache
        hf_cache = home / ".cache" / "huggingface" / "hub"
        if hf_cache.exists():
            paths.append(hf_cache)

        # Project models directory
        project_models = Path("models")
        if project_models.exists():
            paths.append(project_models)

        # Small LLMs directory
        small_llms = Path("models/small-llms")
        if small_llms.exists():
            paths.append(small_llms)

        # User models directory
        user_models = home / "models"
        if user_models.exists():
            paths.append(user_models)

        # Ollama models (if available)
        ollama_dir = home / ".ollama" / "models"
        if ollama_dir.exists():
            paths.append(ollama_dir)

        logger.debug(f"Using search paths: {[str(p) for p in paths]}")
        return paths

    def _group_models_by_backend(self) -> Dict[str | List[ModelInfo]]:
        """Group models by backend."""
        result = {}
        for backend_name, model_ids in self._backend_models.items():
            result[backend_name] = [self._models[model_id] for model_id in model_ids]
        return result

    def list_models(
        self, backend_name: str | None = None, format_filter: str | None = None
    ) -> List[ModelInfo]:
        """
        List available models.

        Args:
            backend_name: Filter by backend name
            format_filter: Filter by model format

        Returns:
            List of matching models
        """
        models = []

        if backend_name:
            model_ids = self._backend_models.get(backend_name, [])
            models = [self._models[model_id] for model_id in model_ids]
        else:
            models = list(self._models.values())

        if format_filter:
            models = [m for m in models if m.format.value == format_filter]

        return models

    def get_model(self, model_name: str) -> ModelInfo | None:
        """Get model by name."""
        # Try exact match first
        if model_name in self._models:
            return self._models[model_name]

        # Try partial match
        for model_id, model_info in self._models.items():
            if model_info.name == model_name:
                return model_info

        return None

    def find_models_by_name(self, name_pattern: str) -> List[ModelInfo]:
        """Find models matching a name pattern."""
        matches = []
        pattern_lower = name_pattern.lower()

        for model_info in self._models.values():
            if pattern_lower in model_info.name.lower():
                matches.append(model_info)

        return matches

    def get_compatible_backends(self, model_info: ModelInfo) -> List[str]:
        """Get list of backends that can handle this model."""
        compatible = []

        for backend in self.backends:
            if backend.can_load_model(model_info):
                compatible.append(backend.name)

        return compatible

    def get_best_backend_for_model(self, model_info: ModelInfo) -> str | None:
        """Get the best backend for loading a specific model."""
        compatible = self.get_compatible_backends(model_info)

        if not compatible:
            return None

        # Priority order for backend selection
        priority_order = ["ollama", "llamacpp", "transformers"]

        for preferred in priority_order:
            if preferred in compatible:
                return preferred

        # Return first compatible if no preferred match
        return compatible[0]

    def get_models_summary(self) -> Dict[str | any]:
        """Get summary statistics about discovered models."""
        total_models = len(self._models)
        backend_counts = {name: len(models) for name, models in self._backend_models.items()}

        # Count by format
        format_counts = {}
        total_size_mb = 0

        for model in self._models.values():
            format_name = model.format.value
            format_counts[format_name] = format_counts.get(format_name, 0) + 1

            if model.size_mb:
                total_size_mb += model.size_mb

        return {
            "total_models": total_models,
            "backend_counts": backend_counts,
            "format_counts": format_counts,
            "total_size_mb": round(total_size_mb, 2),
            "total_size_gb": round(total_size_mb / 1024, 2),
        }

    def _load_cache(self) -> None:
        """Load models from cache file."""
        if not os.path.exists(self.cache_file):
            return

        try:
            with open(self.cache_file, "r") as f:
                cache_data = json.load(f)

            # Validate cache version/format
            if cache_data.get("version") != "1.0":
                logger.debug("Cache version mismatch, ignoring cache")
                return

            # Load models
            models_data = cache_data.get("models", {})
            for model_id, model_data in models_data.items():
                try:
                    # Reconstruct ModelInfo from cached data
                    model_info = self._model_info_from_dict(model_data)
                    self._models[model_id] = model_info

                    # Add to backend mapping
                    backend_name = model_data.get("backend")
                    if backend_name and backend_name in self._backend_models:
                        self._backend_models[backend_name].append(model_id)

                except Exception as e:
                    logger.warning(f"Failed to load cached model {model_id}: {e}")

            logger.debug(f"Loaded {len(self._models)} models from cache")

        except Exception as e:
            logger.warning(f"Failed to load model cache: {e}")

    def _save_cache(self) -> None:
        """Save models to cache file."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

            # Prepare cache data
            cache_data = {"version": "1.0", "models": {}}

            for model_id, model_info in self._models.items():
                # Find backend for this model
                backend_name = None
                for bname, model_ids in self._backend_models.items():
                    if model_id in model_ids:
                        backend_name = bname
                        break

                cache_data["models"][model_id] = self._model_info_to_dict(model_info, backend_name)

            # Write cache file
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)

            logger.debug(f"Saved {len(self._models)} models to cache")

        except Exception as e:
            logger.warning(f"Failed to save model cache: {e}")

    def _model_info_to_dict(self, model_info: ModelInfo, backend_name: str) -> Dict:
        """Convert ModelInfo to dictionary for caching."""
        return {
            "name": model_info.name,
            "path": str(model_info.path),
            "format": model_info.format.value,
            "size_mb": model_info.size_mb,
            "parameters": model_info.parameters,
            "context_length": model_info.context_length,
            "description": model_info.description,
            "metadata": model_info.metadata,
            "backend": backend_name,
        }

    def _model_info_from_dict(self, data: Dict) -> ModelInfo:
        """Reconstruct ModelInfo from dictionary."""
        from .backends.base import ModelFormat

        return ModelInfo(
            name=data["name"],
            path=data["path"],
            format=ModelFormat(data["format"]),
            size_mb=data.get("size_mb"),
            parameters=data.get("parameters"),
            context_length=data.get("context_length"),
            description=data.get("description"),
            metadata=data.get("metadata", {}),
        )

    def clear_cache(self) -> None:
        """Clear the model cache."""
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
            logger.info("Model cache cleared")

        self._models.clear()
        for backend_name in self._backend_models:
            self._backend_models[backend_name].clear()
