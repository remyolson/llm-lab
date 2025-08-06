"""
OpenAI Provider Module

This module implements the OpenAI API integration for the LLM Lab framework.
It provides a wrapper around OpenAI's Python SDK, supporting GPT-4 and GPT-3.5-turbo models.

The module handles:
- API authentication using OpenAI API keys
- Model initialization and configuration
- Text generation with parameter mapping
- Exponential backoff for rate limiting
- Comprehensive error handling and mapping

Example:
    provider = OpenAIProvider("gpt-4")
    response = provider.generate("What is the capital of France?")
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import logging
import os
import time
from typing import Any, Dict, Optional

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False

from .base import LLMProvider
from .exceptions import (
    InvalidCredentialsError,
    ModelNotSupportedError,
    ProviderConfigurationError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
    RateLimitError,
    map_provider_exception,
)
from .registry import register_provider

logger = logging.getLogger(__name__)


if OPENAI_AVAILABLE:

    @register_provider(
        models=[
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-4o",
            "gpt-4o-mini",
        ]
    )
    class OpenAIProvider(LLMProvider):
        """OpenAI provider implementation for GPT models."""

        def __init__(self, model_name: str, **kwargs):
            """
            Initialize OpenAI Provider.

            Args:
                model_name: The OpenAI model to use
                **kwargs: Additional configuration parameters
            """
            super().__init__(model_name, **kwargs)
            self._client: OpenAI | None = None

        def validate_credentials(self) -> bool:
            """
            Validate OpenAI API credentials.

            Returns:
                True if credentials are valid

            Raises:
                InvalidCredentialsError: If API key is missing or invalid
            """
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise InvalidCredentialsError(
                    provider_name=self.provider_name,
                    details="OPENAI_API_KEY environment variable not set",
                )

            # Validate key format (should start with 'sk-')
            if not api_key.startswith("sk-"):
                logger.warning("OpenAI API key does not start with 'sk-', it might be invalid")

            try:
                # Initialize client to test credentials
                client = OpenAI(api_key=api_key)
                # Try to list models to validate the API key
                client.models.list()
                return True

            except Exception as e:
                error_msg = str(e).lower()
                if (
                    "invalid" in error_msg
                    or "unauthorized" in error_msg
                    or "authentication" in error_msg
                ):
                    raise InvalidCredentialsError(
                        provider_name=self.provider_name, details=f"Invalid API key: {e!s}"
                    )
                # Re-raise other exceptions
                raise

        def _initialize_client(self):
            """Initialize the OpenAI client."""
            if self._client is not None:
                return

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise InvalidCredentialsError(
                    provider_name=self.provider_name,
                    details="OPENAI_API_KEY environment variable not set",
                )

            try:
                self._client = OpenAI(
                    api_key=api_key,
                    timeout=self.config.timeout,
                    max_retries=0,  # We handle retries ourselves
                )

            except Exception as e:
                raise ProviderConfigurationError(
                    provider_name=self.provider_name,
                    config_issue=f"Failed to initialize OpenAI client: {e!s}",
                )

        def generate(self, prompt: str, **kwargs) -> str:
            """
            Generate a response from the OpenAI model.

            Args:
                prompt: The input prompt to send to the model
                **kwargs: Additional generation parameters

            Returns:
                The generated text response

            Raises:
                ProviderError: If generation fails
            """
            # Input validation
            from ..utils.validation import ValidationError, validate_prompt, validate_range

            try:
                prompt = validate_prompt(prompt, max_length=50000)
            except ValidationError as e:
                raise ProviderError(self.provider_name, f"Invalid prompt: {e}")

            # Validate generation parameters
            temperature = kwargs.get("temperature", self.config.temperature)
            max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

            try:
                if temperature is not None:
                    temperature = validate_range(temperature, 0.0, 2.0, "temperature")
                if max_tokens is not None:
                    max_tokens = validate_range(max_tokens, 1, 4096, "max_tokens")
            except ValidationError as e:
                raise ProviderError(self.provider_name, f"Invalid parameter: {e}")

            # Ensure client is initialized
            if not self._initialized:
                self.initialize()

            self._initialize_client()

            # Use validated parameters
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens
            top_p = kwargs.get("top_p", self.config.top_p)

            # Implement retry logic with exponential backoff
            max_retries = self.config.max_retries
            retry_delay = self.config.retry_delay

            for attempt in range(max_retries + 1):
                try:
                    # Make the API call
                    response = self._client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        **{
                            k: v
                            for k, v in kwargs.items()
                            if k not in ["temperature", "max_tokens", "top_p"]
                        },
                    )

                    # Extract and return the response
                    if response.choices and response.choices[0].message.content:
                        return response.choices[0].message.content.strip()
                    else:
                        raise ProviderResponseError(
                            provider_name=self.provider_name,
                            details="No text in model response",
                            response_data={
                                "choices": len(response.choices) if response.choices else 0
                            },
                        )

                except Exception as e:
                    # If it's already one of our exceptions, re-raise
                    if isinstance(
                        e,
                        (
                            InvalidCredentialsError,
                            RateLimitError,
                            ProviderTimeoutError,
                            ProviderResponseError,
                            ModelNotSupportedError,
                        ),
                    ):
                        raise

                    # Handle all other errors
                    error_type = type(e).__name__
                    error_msg = str(e)

                    # Check if it's a rate limit error
                    if (
                        "ratelimit" in error_type.lower()
                        or "rate_limit" in error_type.lower()
                        or "rate limit" in error_msg.lower()
                    ):
                        if attempt < max_retries:
                            # Calculate backoff time
                            wait_time = retry_delay * (2**attempt)
                            logger.warning(
                                f"Rate limit hit, retrying in {wait_time}s "
                                f"(attempt {attempt + 1}/{max_retries + 1})"
                            )
                            time.sleep(wait_time)
                            continue
                        else:
                            raise RateLimitError(
                                provider_name=self.provider_name,
                                retry_after=int(retry_delay * (2**max_retries)),
                                limit_type="requests",
                            )

                    # Check for timeout errors
                    elif "timeout" in error_type.lower() or "timeout" in error_msg.lower():
                        raise ProviderTimeoutError(
                            provider_name=self.provider_name,
                            timeout_seconds=self.config.timeout,
                            operation="chat completion",
                        )

                    # Check for authentication errors
                    elif (
                        "authentication" in error_type.lower()
                        or "unauthorized" in error_msg.lower()
                    ):
                        raise InvalidCredentialsError(
                            provider_name=self.provider_name, details=error_msg
                        )

                    # Check for model not found
                    elif (
                        "model_not_found" in error_type.lower()
                        or "does not exist" in error_msg.lower()
                    ):
                        raise ModelNotSupportedError(
                            model_name=self.model_name,
                            provider_name=self.provider_name,
                            supported_models=self.supported_models,
                        )

                    # For other errors, map them
                    else:
                        raise map_provider_exception(self.provider_name, e)

        # This should not be reached due to the retry logic above
        raise ProviderResponseError(
            provider_name=self.provider_name, details="Max retries exceeded"
        )

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary containing model metadata
        """
        # Model-specific information
        model_info = {
            "gpt-4": {
                "max_tokens": 8192,
                "context_window": 8192,
                "training_cutoff": "September 2021",
            },
            "gpt-4-turbo": {
                "max_tokens": 4096,
                "context_window": 128000,
                "training_cutoff": "April 2023",
            },
            "gpt-4-turbo-preview": {
                "max_tokens": 4096,
                "context_window": 128000,
                "training_cutoff": "April 2023",
            },
            "gpt-4o": {
                "max_tokens": 4096,
                "context_window": 128000,
                "training_cutoff": "October 2023",
            },
            "gpt-4o-mini": {
                "max_tokens": 16384,
                "context_window": 128000,
                "training_cutoff": "October 2023",
            },
            "gpt-3.5-turbo": {
                "max_tokens": 4096,
                "context_window": 16385,
                "training_cutoff": "September 2021",
            },
            "gpt-3.5-turbo-16k": {
                "max_tokens": 4096,
                "context_window": 16385,
                "training_cutoff": "September 2021",
            },
        }

        # Get model-specific info or use defaults
        specific_info = model_info.get(
            self.model_name,
            {"max_tokens": 4096, "context_window": 8192, "training_cutoff": "Unknown"},
        )

        return {
            "model_name": self.model_name,
            "provider": "openai",
            "max_tokens": specific_info["max_tokens"],
            "capabilities": ["text-generation", "chat"],
            "supports_streaming": True,
            "supports_functions": True,
            "context_window": specific_info["context_window"],
            "training_cutoff": specific_info["training_cutoff"],
        }

else:
    # Create a placeholder class when OpenAI dependencies are not available
    class OpenAIProvider:
        """Placeholder class when OpenAI dependencies are not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "OpenAI library is not available. Install it with: pip install openai"
            )
