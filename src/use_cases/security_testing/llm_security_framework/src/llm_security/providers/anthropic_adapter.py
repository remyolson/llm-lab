"""Anthropic API adapter for LLM security testing."""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import ProviderConfig
from .base import ModelAdapter, ModelError, ModelResponse, ProviderType, ResponseStatus

logger = logging.getLogger(__name__)


class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic API (Claude models)."""

    DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
    DEFAULT_MODEL = "claude-3-opus-20240229"
    DEFAULT_API_VERSION = "2023-06-01"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Anthropic adapter.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._provider_type = ProviderType.ANTHROPIC
        self._session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry logic."""
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.get("max_retries", 3),
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy, pool_maxsize=self.config.get("max_connections", 10)
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def configure(self, config_dict: Dict[str, Any]) -> None:
        """
        Configure the Anthropic adapter.

        Args:
            config_dict: Configuration dictionary
        """
        self.config = config_dict

        # Set API key
        self.api_key = config_dict.get("api_key") or config_dict.get("anthropic_api_key")
        if not self.api_key:
            logger.warning("No Anthropic API key configured")

        # Set base URL
        self.base_url = config_dict.get("api_base_url", self.DEFAULT_BASE_URL)

        # Set model
        self._model_name = config_dict.get("model_name", self.DEFAULT_MODEL)

        # Set API version
        self._api_version = config_dict.get("api_version", self.DEFAULT_API_VERSION)

        # Set capabilities
        self._is_async_capable = True

        # Set timeout
        self.timeout = config_dict.get("timeout_seconds", 30)

    def send_prompt(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> ModelResponse:
        """
        Send a prompt to Anthropic.

        Args:
            prompt: Text prompt or conversation history
            **kwargs: Additional parameters

        Returns:
            ModelResponse object
        """
        request_id = self.generate_request_id()
        start_time = time.time()

        try:
            # Convert messages to Anthropic format
            messages = self._convert_to_anthropic_format(prompt)

            # Extract system message if present
            system_message = None
            if isinstance(prompt, list):
                system_msgs = [m for m in prompt if m.get("role") == "system"]
                if system_msgs:
                    system_message = system_msgs[0]["content"]
                    # Remove system messages from regular messages
                    messages = [m for m in messages if m.get("role") != "system"]

            # Build request payload
            payload = {
                "model": kwargs.get("model", self._model_name),
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", 4096),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p"),
                "top_k": kwargs.get("top_k"),
                "stream": kwargs.get("stream", False),
            }

            # Add system message if present
            if system_message or kwargs.get("system"):
                payload["system"] = system_message or kwargs.get("system")

            # Add stop sequences if provided
            if "stop_sequences" in kwargs:
                payload["stop_sequences"] = kwargs["stop_sequences"]

            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}

            # Prepare headers
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": self._api_version,
                "Content-Type": "application/json",
            }

            # Add custom headers
            headers.update(self.config.get("custom_headers", {}))

            # Make request
            response = self._session.post(
                f"{self.base_url}/messages", json=payload, headers=headers, timeout=self.timeout
            )

            # Handle response
            if response.status_code == 200:
                return self._handle_success_response(response.json(), request_id, start_time)
            else:
                return self._handle_error_response(response, request_id, start_time)

        except requests.exceptions.Timeout:
            error = ModelError(
                message="Request timed out",
                provider=self.provider_name,
                error_type=ResponseStatus.TIMEOUT,
            )
            return self._create_error_response(error, request_id, start_time)

        except Exception as e:
            error = self.handle_errors(e)
            return self._create_error_response(error, request_id, start_time)

    def _convert_to_anthropic_format(
        self, prompt: Union[str, List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """
        Convert messages to Anthropic format.

        Args:
            prompt: Text prompt or conversation history

        Returns:
            Messages in Anthropic format
        """
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]

        # Anthropic requires alternating user/assistant messages
        messages = []
        last_role = None

        for msg in prompt:
            role = msg.get("role")
            content = msg.get("content", "")

            # Skip system messages (handled separately)
            if role == "system":
                continue

            # Convert role names
            if role in ["human", "user"]:
                role = "user"
            elif role in ["ai", "assistant"]:
                role = "assistant"

            # Ensure alternating pattern
            if role == last_role:
                # Merge with previous message
                if messages:
                    messages[-1]["content"] += f"\n\n{content}"
            else:
                messages.append({"role": role, "content": content})
                last_role = role

        # Ensure first message is from user
        if messages and messages[0]["role"] != "user":
            messages.insert(0, {"role": "user", "content": "Continue the conversation"})

        # Ensure last message is from user
        if messages and messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": "Please respond"})

        return messages

    def _handle_success_response(
        self, data: Dict[str, Any], request_id: str, start_time: float
    ) -> ModelResponse:
        """Handle successful API response."""
        latency_ms = (time.time() - start_time) * 1000

        # Extract content
        content_blocks = data.get("content", [])
        if not content_blocks:
            content = ""
        elif isinstance(content_blocks, list):
            # Combine text from all content blocks
            content_parts = []
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    content_parts.append(block.get("text", ""))
            content = "\n".join(content_parts)
        else:
            content = str(content_blocks)

        # Extract token usage
        usage = data.get("usage", {})
        tokens_used = {
            "prompt": usage.get("input_tokens", 0),
            "completion": usage.get("output_tokens", 0),
            "total": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        }

        # Create response
        response = ModelResponse(
            content=content,
            model=data.get("model", self._model_name),
            provider=self.provider_name,
            request_id=request_id,
            status=ResponseStatus.SUCCESS,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            metadata={
                "stop_reason": data.get("stop_reason"),
                "stop_sequence": data.get("stop_sequence"),
                "id": data.get("id"),
                "type": data.get("type"),
            },
            raw_response=data,
        )

        self.track_response(response)
        return response

    def _handle_error_response(
        self, response: requests.Response, request_id: str, start_time: float
    ) -> ModelResponse:
        """Handle error API response."""
        latency_ms = (time.time() - start_time) * 1000

        # Determine error type
        if response.status_code == 401:
            status = ResponseStatus.AUTHENTICATION_ERROR
            message = "Invalid API key"
        elif response.status_code == 429:
            status = ResponseStatus.RATE_LIMITED
            message = "Rate limit exceeded"
        elif response.status_code == 400:
            status = ResponseStatus.INVALID_REQUEST
            message = "Invalid request"
        else:
            status = ResponseStatus.ERROR
            message = f"API error: {response.status_code}"

        # Try to extract error details
        try:
            error_data = response.json()
            if "error" in error_data:
                error_info = error_data["error"]
                if isinstance(error_info, dict):
                    message = error_info.get("message", message)
                else:
                    message = str(error_info)
        except:
            pass

        response_obj = ModelResponse(
            content="",
            model=self._model_name,
            provider=self.provider_name,
            request_id=request_id,
            status=status,
            latency_ms=latency_ms,
            error_message=message,
            metadata={"status_code": response.status_code, "headers": dict(response.headers)},
        )

        self.track_response(response_obj)
        return response_obj

    def _create_error_response(
        self, error: ModelError, request_id: str, start_time: float
    ) -> ModelResponse:
        """Create error response from ModelError."""
        latency_ms = (time.time() - start_time) * 1000

        response = ModelResponse(
            content="",
            model=self._model_name,
            provider=self.provider_name,
            request_id=request_id,
            status=error.error_type,
            latency_ms=latency_ms,
            error_message=error.message,
            metadata=error.details or {},
        )

        self.track_response(response)
        return response

    def get_response(self, prompt_id: str) -> Optional[ModelResponse]:
        """
        Get a response by prompt ID.

        Args:
            prompt_id: ID of the prompt

        Returns:
            ModelResponse or None
        """
        # For async operations, would check a queue or database
        # For now, return None as Anthropic is synchronous
        return None

    def handle_errors(self, error: Exception) -> ModelError:
        """
        Handle and convert Anthropic-specific errors.

        Args:
            error: Original exception

        Returns:
            ModelError with standardized information
        """
        if isinstance(error, requests.exceptions.ConnectionError):
            return ModelError(
                message="Connection error",
                provider=self.provider_name,
                error_type=ResponseStatus.ERROR,
                original_exception=error,
            )
        elif isinstance(error, requests.exceptions.Timeout):
            return ModelError(
                message="Request timeout",
                provider=self.provider_name,
                error_type=ResponseStatus.TIMEOUT,
                original_exception=error,
            )
        else:
            return ModelError(
                message=str(error),
                provider=self.provider_name,
                error_type=ResponseStatus.ERROR,
                original_exception=error,
            )

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model and provider.

        Returns:
            Dictionary with model information
        """
        return {
            "provider": self.provider_name,
            "model": self._model_name,
            "api_version": self._api_version,
            "base_url": self.base_url,
            "capabilities": {
                "streaming": True,
                "functions": False,
                "tools": False,
                "vision": "claude-3" in self._model_name,
                "max_tokens": self._get_max_tokens(),
                "async": self._is_async_capable,
                "system_messages": True,
            },
            "metrics": self.get_metrics(),
        }

    def _get_max_tokens(self) -> int:
        """Get maximum tokens for current model."""
        model_limits = {
            "claude-3-opus": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-haiku": 200000,
            "claude-2.1": 100000,
            "claude-2.0": 100000,
            "claude-instant": 100000,
        }

        for model_prefix, limit in model_limits.items():
            if self._model_name.startswith(model_prefix):
                return limit

        return 100000  # Default for Claude models

    def stream_prompt(
        self, prompt: Union[str, List[Dict[str, str]]], callback: callable, **kwargs
    ) -> None:
        """
        Stream responses from Anthropic.

        Args:
            prompt: Text prompt or conversation history
            callback: Function to call with each chunk
            **kwargs: Additional parameters
        """
        kwargs["stream"] = True
        request_id = self.generate_request_id()

        try:
            messages = self._convert_to_anthropic_format(prompt)

            payload = {
                "model": kwargs.get("model", self._model_name),
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", 4096),
                "stream": True,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["model", "messages", "max_tokens", "stream"]
                },
            }

            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": self._api_version,
                "Content-Type": "application/json",
            }

            response = self._session.post(
                f"{self.base_url}/messages",
                json=payload,
                headers=headers,
                stream=True,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            data = line[6:]
                            if data != "[DONE]":
                                chunk = json.loads(data)
                                callback(chunk)
            else:
                error = self._handle_error_response(response, request_id, time.time())
                callback({"error": error.error_message})

        except Exception as e:
            callback({"error": str(e)})
