"""
Anthropic Claude Provider Module

This module implements the Anthropic API integration for the LLM Lab framework.
It provides a wrapper around Anthropic's Python SDK, supporting Claude 3 models
including Opus, Sonnet, and Haiku variants.

The module handles:
- API authentication using Anthropic API keys
- Model initialization and configuration
- Message format conversion between standard interface and Anthropic format
- Text generation requests with error handling and retries
- Rate limiting and exponential backoff
- Graceful failure modes for API errors

Example:
    provider = AnthropicProvider("claude-3-opus-20240229")
    response = provider.generate("What is the capital of France?")
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import logging
import os
import time
from typing import Any, Dict, List, Optional

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

logger = logging.getLogger(__name__)

# Import anthropic SDK (will be installed via requirements)
try:
    import anthropic
    from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic
    from anthropic.types import Message

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None
    Anthropic = None
    HUMAN_PROMPT = "\n\nHuman: "
    AI_PROMPT = "\n\nAssistant: "


@register_provider(
    models=[
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-20241022",
        "claude-2.1",
        "claude-2.0",
        "claude-instant-1.2",
    ]
)
class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation."""

    # List of supported model names for base class
    SUPPORTED_MODELS = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-20241022",
        "claude-2.1",
        "claude-2.0",
        "claude-instant-1.2",
    ]

    # Model details with context windows and other info
    MODEL_INFO = {
        "claude-3-opus-20240229": {
            "context_window": 200000,
            "max_output": 4096,
            "description": "Most capable Claude 3 model",
        },
        "claude-3-sonnet-20240229": {
            "context_window": 200000,
            "max_output": 4096,
            "description": "Balanced performance Claude 3 model",
        },
        "claude-3-haiku-20240307": {
            "context_window": 200000,
            "max_output": 4096,
            "description": "Fastest Claude 3 model",
        },
        "claude-3-5-sonnet-20240620": {
            "context_window": 200000,
            "max_output": 4096,
            "description": "Claude 3.5 Sonnet - June 2024",
        },
        "claude-3-5-sonnet-20241022": {
            "context_window": 200000,
            "max_output": 8192,
            "description": "Claude 3.5 Sonnet - October 2024",
        },
        "claude-2.1": {
            "context_window": 200000,
            "max_output": 4096,
            "description": "Claude 2.1 with 200k context",
        },
        "claude-2.0": {"context_window": 100000, "max_output": 4096, "description": "Claude 2.0"},
        "claude-instant-1.2": {
            "context_window": 100000,
            "max_output": 4096,
            "description": "Fast, cost-effective Claude model",
        },
    }

    def __init__(self, model_name: str = "claude-3-sonnet-20240229", **kwargs):
        """
        Initialize AnthropicProvider.

        Args:
            model_name: The Claude model to use
            **kwargs: Additional configuration parameters
        """
        if not ANTHROPIC_AVAILABLE:
            raise ProviderConfigurationError(
                provider_name="anthropic",
                config_issue="anthropic package not installed. Install with: pip install anthropic",
            )

        super().__init__(model_name, **kwargs)
        self._client: Anthropic | None = None
        self._retry_count = 0

    def validate_credentials(self) -> bool:
        """
        Validate Anthropic API credentials.

        Returns:
            True if credentials are valid

        Raises:
            InvalidCredentialsError: If API key is missing or invalid
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise InvalidCredentialsError(
                provider_name=self.provider_name,
                details="ANTHROPIC_API_KEY environment variable not set",
            )

        try:
            # Create a test client
            test_client = anthropic.Anthropic(api_key=api_key)

            # Try a minimal API call to validate the key
            # Using messages API for Claude 3 models
            test_client.messages.create(
                model="claude-3-haiku-20240307",  # Use cheapest model for test
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
            )
            return True

        except Exception as e:
            error_msg = str(e).lower()
            if (
                "authentication" in error_msg
                or "api key" in error_msg
                or "unauthorized" in error_msg
            ):
                raise InvalidCredentialsError(
                    provider_name=self.provider_name, details=f"Invalid API key: {e!s}"
                )
            # For other errors during validation, we'll consider credentials valid
            # as the error might be rate limiting or other transient issues
            logger.warning(f"Credential validation warning: {e!s}")
            return True

    def _initialize_client(self):
        """Initialize the Anthropic client."""
        if self._client is not None:
            return

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise InvalidCredentialsError(
                provider_name=self.provider_name,
                details="ANTHROPIC_API_KEY environment variable not set",
            )

        try:
            self._client = anthropic.Anthropic(
                api_key=api_key,
                max_retries=0,  # We'll handle retries ourselves
            )
        except Exception as e:
            raise ProviderConfigurationError(
                provider_name=self.provider_name,
                config_issue=f"Failed to initialize Anthropic client: {e!s}",
            )

    def _convert_to_messages_format(self, prompt: str) -> List[Dict[str, str]]:
        """
        Convert a simple prompt to Anthropic's messages format.

        Args:
            prompt: The input prompt

        Returns:
            List of message dictionaries
        """
        # Check if prompt already contains Human/Assistant markers
        if HUMAN_PROMPT in prompt or AI_PROMPT in prompt:
            # Parse existing format
            messages = []
            parts = prompt.split(HUMAN_PROMPT)

            for part in parts[1:]:  # Skip empty first element
                if AI_PROMPT in part:
                    human_part, ai_part = part.split(AI_PROMPT, 1)
                    messages.append({"role": "user", "content": human_part.strip()})
                    if ai_part.strip():
                        messages.append({"role": "assistant", "content": ai_part.strip()})
                else:
                    messages.append({"role": "user", "content": part.strip()})

            return messages
        else:
            # Simple prompt - convert to single user message
            return [{"role": "user", "content": prompt}]

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the Claude model.

        Args:
            prompt: The input prompt to send to the model
            **kwargs: Additional generation parameters

        Returns:
            The generated text response

        Raises:
            ProviderError: If generation fails
            RateLimitError: If rate limits are exceeded
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

        # Convert prompt to messages format
        messages = self._convert_to_messages_format(prompt)

        # Prepare generation parameters
        model_info = self.MODEL_INFO.get(self.model_name, {})
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        # Ensure max_tokens doesn't exceed model limit
        max_output = model_info.get("max_output", 4096)
        if max_tokens > max_output:
            logger.warning(
                f"Requested max_tokens ({max_tokens}) exceeds model limit ({max_output}). Using model limit."
            )
            max_tokens = max_output

        # Build request parameters
        request_params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        # Add optional parameters if provided
        if "top_p" in kwargs or hasattr(self.config, "top_p"):
            request_params["top_p"] = kwargs.get("top_p", self.config.top_p)

        if "stop_sequences" in kwargs:
            request_params["stop_sequences"] = kwargs["stop_sequences"]

        # Implement retry logic with exponential backoff
        max_retries = self.config.max_retries
        retry_delay = self.config.retry_delay

        for attempt in range(max_retries + 1):
            try:
                # Make the API call
                response = self._client.messages.create(**request_params)

                # Extract text from response
                if hasattr(response, "content") and response.content:
                    # Claude 3 returns a list of content blocks
                    text_content = []
                    for content_block in response.content:
                        if hasattr(content_block, "text"):
                            text_content.append(content_block.text)

                    if text_content:
                        return "".join(text_content).strip()
                    else:
                        raise ProviderResponseError(
                            provider_name=self.provider_name, details="No text content in response"
                        )
                else:
                    raise ProviderResponseError(
                        provider_name=self.provider_name,
                        details="Invalid response structure from API",
                    )

            except anthropic.RateLimitError:
                if attempt < max_retries:
                    wait_time = retry_delay * (2**attempt)  # Exponential backoff
                    logger.warning(
                        f"Rate limit hit, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    raise RateLimitError(
                        provider_name=self.provider_name, limit_type="API rate limit"
                    )

            except anthropic.APITimeoutError:
                raise ProviderTimeoutError(provider_name=self.provider_name, operation="Generate")

            except anthropic.AuthenticationError as e:
                raise InvalidCredentialsError(
                    provider_name=self.provider_name, details=f"Authentication failed: {e!s}"
                )

            except Exception as e:
                # If it's already one of our exceptions, re-raise
                if isinstance(
                    e,
                    (
                        ProviderResponseError,
                        InvalidCredentialsError,
                        RateLimitError,
                        ProviderTimeoutError,
                    ),
                ):
                    raise

                # For other Anthropic-specific errors
                if hasattr(e, "__class__") and "anthropic" in str(e.__class__):
                    error_msg = str(e)
                    if "context length" in error_msg or "too long" in error_msg:
                        raise ProviderResponseError(
                            provider_name=self.provider_name,
                            details=f"Context length exceeded: {error_msg}",
                        )

                # Use generic exception mapping for unknown errors
                raise map_provider_exception(self.provider_name, e)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary containing model metadata
        """
        model_details = self.MODEL_INFO.get(self.model_name, {})

        return {
            "model_name": self.model_name,
            "provider": "anthropic",
            "max_tokens": model_details.get("max_output", 4096),
            "context_window": model_details.get("context_window", 100000),
            "capabilities": ["text-generation", "conversation"],
            "description": model_details.get("description", "Claude model"),
            "supports_streaming": True,
            "supports_system_messages": True,
            "version": self.model_name.split("-")[-1] if "-" in self.model_name else "unknown",
        }

    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get default generation parameters for Anthropic models.

        Returns:
            Dictionary of default parameters
        """
        return {"temperature": 0.7, "max_tokens": 1000, "top_p": 1.0, "stop_sequences": []}
