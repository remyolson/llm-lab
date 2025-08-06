"""
Local Model Provider System

This module provides a unified interface for running local LLM models
through multiple backends including Transformers, llama.cpp, and Ollama.
"""

# Import backends for registration
from .backends import LlamaCppBackend, OllamaBackend, TransformersBackend
from .registry import ModelRegistry
from .resource_manager import ResourceManager
from .unified_provider import UnifiedLocalProvider

__all__ = [
    "ModelRegistry",
    "UnifiedLocalProvider",
    "ResourceManager",
    "TransformersBackend",
    "LlamaCppBackend",
    "OllamaBackend",
]


# Auto-register the unified provider when this module is imported
def register_local_provider():
    """Register the unified local provider with the main provider registry."""
    try:
        # Import the main registry
        from ..registry import registry

        # Register our unified provider
        registry.register(UnifiedLocalProvider, UnifiedLocalProvider.SUPPORTED_MODELS)

        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"Registered UnifiedLocalProvider with {len(UnifiedLocalProvider.SUPPORTED_MODELS)} supported models"
        )

    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to auto-register UnifiedLocalProvider: {e}")


# Auto-register when module is imported
register_local_provider()
