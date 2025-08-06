"""LLM Providers Package"""

# Import local providers to trigger auto-registration
from . import local
from .anthropic import AnthropicProvider
from .base import LLMProvider, ProviderConfig
from .exceptions import (
    InvalidCredentialsError,
    ModelNotSupportedError,
    ProviderConfigurationError,
    ProviderError,
    ProviderNotFoundError,
    ProviderResponseError,
    ProviderTimeoutError,
    RateLimitError,
)
from .google import GoogleProvider

# Temporarily disable OpenAI provider due to syntax issues
# from .openai import OpenAIProvider
from .registry import get_provider_for_model, register_provider, registry

__all__ = [
    "AnthropicProvider",
    "GoogleProvider",
    "InvalidCredentialsError",
    "LLMProvider",
    "ModelNotSupportedError",
    # "OpenAIProvider",  # Temporarily disabled
    "ProviderConfig",
    "ProviderConfigurationError",
    "ProviderError",
    "ProviderNotFoundError",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "RateLimitError",
    "get_provider_for_model",
    "register_provider",
    "registry",
    "local",
]
