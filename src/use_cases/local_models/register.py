"""
Register LocalModelProvider with the provider registry.

This module handles the registration of the local model provider
with the LLM Lab provider registry.
"""

import logging

from .provider import LocalModelProvider
from .providers.registry import registry

logger = logging.getLogger(__name__)


def register_local_provider():
    """Register the LocalModelProvider with supported models."""
    try:
        # Register with all supported models
        supported_models = LocalModelProvider.SUPPORTED_MODELS

        # Also register with generic "local" prefix for custom models
        extended_models = supported_models + [
            "local-llama-2-7b",
            "local-llama-2-13b",
            "local-mistral-7b",
            "local-phi-2",
            "local-custom",
        ]

        registry.register(LocalModelProvider, extended_models)
        logger.info("Successfully registered LocalModelProvider")

        return True

    except Exception as e:
        logger.error(f"Failed to register LocalModelProvider: {e!s}")
        return False


# Auto-register when imported
if __name__ != "__main__":
    register_local_provider()
