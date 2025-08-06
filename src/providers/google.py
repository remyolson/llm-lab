"""
Google Gemini Provider Module

This module implements the Google Gemini API integration for the LLM Lab framework.
It provides a wrapper around Google's Generative AI Python SDK, specifically for
the Gemini 1.5 Flash model.

The module handles:
- API authentication using Google API keys
- Model initialization and configuration
- Text generation requests with error handling
- Graceful failure modes for API errors

Example:
    provider = GoogleProvider("gemini-1.5-flash")
    response = provider.generate("What is the capital of France?")
"""

import os
from typing import Any, Dict

try:
    import google.generativeai as genai

    GOOGLE_AVAILABLE = True
except ImportError:
    genai = None
    GOOGLE_AVAILABLE = False

from .base import LLMProvider
from .exceptions import (
    InvalidCredentialsError,
    ProviderConfigurationError,
    ProviderResponseError,
    ProviderTimeoutError,
    RateLimitError,
    map_provider_exception,
)
from .registry import register_provider

if GOOGLE_AVAILABLE:

    @register_provider(models=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"])
    class GoogleProvider(LLMProvider):
        """Google Gemini provider implementation."""

        def __init__(self, model_name: str = "gemini-1.5-flash", **kwargs):
            """
            Initialize GoogleProvider.

            Args:
                model_name: The Gemini model to use
                **kwargs: Additional configuration parameters
            """
            super().__init__(model_name, **kwargs)
            self._client = None
            self._model = None

        def validate_credentials(self) -> bool:
            """
            Validate Google API credentials.

            Returns:
                True if credentials are valid

            Raises:
                InvalidCredentialsError: If API key is missing or invalid
            """
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise InvalidCredentialsError(
                    provider_name=self.provider_name,
                    details="GOOGLE_API_KEY environment variable not set",
                )

            try:
                # Configure the SDK with the API key
                genai.configure(api_key=api_key)

                # Try to list models to validate the API key
                list(genai.list_models())
                return True

            except Exception as e:
                error_msg = str(e).lower()
                if "api key" in error_msg or "invalid" in error_msg or "unauthorized" in error_msg:
                    raise InvalidCredentialsError(
                        provider_name=self.provider_name, details=f"Invalid API key: {e!s}"
                    )
                # Re-raise other exceptions
                raise

        def _initialize_client(self):
            """Initialize the Gemini client and model."""
            if self._model is not None:
                return

            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise InvalidCredentialsError(
                    provider_name=self.provider_name,
                    details="GOOGLE_API_KEY environment variable not set",
                )

            try:
                # Configure the SDK
                genai.configure(api_key=api_key)

                # Create model instance
                self._model = genai.GenerativeModel(self.model_name)

            except Exception as e:
                raise ProviderConfigurationError(
                    provider_name=self.provider_name,
                    config_issue=f"Failed to initialize Gemini model: {e!s}",
                )

        def generate(self, prompt: str, **kwargs) -> str:
            """
            Generate a response from the Gemini model.

            Args:
                prompt: The input prompt to send to the model
                **kwargs: Additional generation parameters

            Returns:
                The generated text response

            Raises:
                ProviderError: If generation fails
            """
            # Ensure client is initialized
            if not self._initialized:
                self.initialize()

            self._initialize_client()

            # Validate prompt
            if not prompt or not prompt.strip():
                raise ProviderResponseError(
                    provider_name=self.provider_name, details="Empty prompt provided"
                )

            try:
                # Prepare generation config
                generation_config = {
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "max_output_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                    "top_p": kwargs.get("top_p", self.config.top_p),
                }

                # Remove None values
                generation_config = {k: v for k, v in generation_config.items() if v is not None}

                # Call the Gemini API
                response = self._model.generate_content(prompt, generation_config=generation_config)

                # Extract text from response
                if response.text:
                    return response.text.strip()
                else:
                    # Check if content was blocked
                    if hasattr(response, "prompt_feedback"):
                        raise ProviderResponseError(
                            provider_name=self.provider_name,
                            details="Content was blocked by safety filters",
                            response_data={"prompt_feedback": str(response.prompt_feedback)},
                        )
                    raise ProviderResponseError(
                        provider_name=self.provider_name, details="No text in model response"
                    )

            except Exception as e:
                # If it's already one of our exceptions, re-raise
                if isinstance(e, (ProviderResponseError, InvalidCredentialsError)):
                    raise

                # Map other exceptions
                error_msg = str(e).lower()

                if "quota" in error_msg or "rate limit" in error_msg:
                    raise RateLimitError(provider_name=self.provider_name, limit_type="API quota")
                elif "timeout" in error_msg or "deadline" in error_msg:
                    raise ProviderTimeoutError(
                        provider_name=self.provider_name, operation="Generate"
                    )
                elif "safety" in error_msg or "blocked" in error_msg:
                    raise ProviderResponseError(
                        provider_name=self.provider_name,
                        details="Content was blocked by safety filters",
                    )
                else:
                    # Use generic exception mapping
                    raise map_provider_exception(self.provider_name, e)

        def get_model_info(self) -> Dict[str | Any]:
            """
            Get information about the current model.

            Returns:
                Dictionary containing model metadata
            """
            return {
                "model_name": self.model_name,
                "provider": "google",
                "max_tokens": 8192 if "1.5" in self.model_name else 2048,
                "capabilities": ["text-generation"],
                "version": self.model_name,
                "supports_streaming": True,
                "context_window": 1048576 if "1.5" in self.model_name else 32768,
            }

else:
    # Create a placeholder class when Google dependencies are not available
    class GoogleProvider:
        """Placeholder class when Google dependencies are not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Google Generative AI library is not available. "
                "Install it with: pip install google-generativeai"
            )


# For backward compatibility
GeminiProvider = GoogleProvider
