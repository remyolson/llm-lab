"""
Local Model Backends

This module provides backend abstractions for different local model inference engines.
"""

from .base import BackendCapabilities, GenerationConfig, LocalBackend, ModelInfo
from .llamacpp_backend import LlamaCppBackend
from .ollama_backend import OllamaBackend
from .transformers_backend import TransformersBackend

__all__ = [
    "LocalBackend",
    "ModelInfo",
    "BackendCapabilities",
    "GenerationConfig",
    "TransformersBackend",
    "LlamaCppBackend",
    "OllamaBackend",
]
