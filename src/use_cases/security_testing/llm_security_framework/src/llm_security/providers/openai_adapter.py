"""OpenAI API adapter for LLM security testing."""

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


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI API (GPT-3.5, GPT-4, etc.)."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-3.5-turbo"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OpenAI adapter.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._provider_type = ProviderType.OPENAI
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
        Configure the OpenAI adapter.

        Args:
            config_dict: Configuration dictionary
        """
        self.config = config_dict

        # Set API key
        self.api_key = config_dict.get("api_key") or config_dict.get("openai_api_key")
        if not self.api_key:
            logger.warning("No OpenAI API key configured")

        # Set base URL
        self.base_url = config_dict.get("api_base_url", self.DEFAULT_BASE_URL)

        # Set model
        self._model_name = config_dict.get("model_name", self.DEFAULT_MODEL)

        # Set organization if provided
        self.organization = config_dict.get("organization_id")

        # Set API version
        self._api_version = config_dict.get("api_version", "v1")

        # Set capabilities
        self._is_async_capable = True

        # Set timeout
        self.timeout = config_dict.get("timeout_seconds", 30)

    def send_prompt(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> ModelResponse:
        """
        Send a prompt to OpenAI.

        Args:
            prompt: Text prompt or conversation history
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            ModelResponse object
        """
        request_id = self.generate_request_id()
        start_time = time.time()

        try:
            # Prepare messages
            messages = self.prepare_messages(prompt)

            # Build request payload
            payload = {
                "model": kwargs.get("model", self._model_name),
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens"),
                "top_p": kwargs.get("top_p", 1.0),
                "n": kwargs.get("n", 1),
                "stream": kwargs.get("stream", False),
                "presence_penalty": kwargs.get("presence_penalty", 0),
                "frequency_penalty": kwargs.get("frequency_penalty", 0),
            }

            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}

            # Add functions if provided
            if "functions" in kwargs:
                payload["functions"] = kwargs["functions"]
                if "function_call" in kwargs:
                    payload["function_call"] = kwargs["function_call"]

            # Add tools if provided (for newer models)
            if "tools" in kwargs:
                payload["tools"] = kwargs["tools"]
                if "tool_choice" in kwargs:
                    payload["tool_choice"] = kwargs["tool_choice"]

            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            if self.organization:
                headers["OpenAI-Organization"] = self.organization

            # Add custom headers
            headers.update(self.config.get("custom_headers", {}))

            # Make request
            response = self._session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
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

    def _handle_success_response(
        self, data: Dict[str, Any], request_id: str, start_time: float
    ) -> ModelResponse:
        """Handle successful API response."""
        latency_ms = (time.time() - start_time) * 1000

        # Extract content
        choices = data.get("choices", [])
        if not choices:
            content = ""
        else:
            message = choices[0].get("message", {})
            content = message.get("content", "")

            # Handle function calls
            if "function_call" in message:
                content = json.dumps(message["function_call"])

            # Handle tool calls
            if "tool_calls" in message:
                content = json.dumps(message["tool_calls"])

        # Extract token usage
        usage = data.get("usage", {})
        tokens_used = {
            "prompt": usage.get("prompt_tokens", 0),
            "completion": usage.get("completion_tokens", 0),
            "total": usage.get("total_tokens", 0),
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
                "finish_reason": choices[0].get("finish_reason") if choices else None,
                "system_fingerprint": data.get("system_fingerprint"),
                "created": data.get("created"),
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
                message = error_data["error"].get("message", message)
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
        # For now, return None as OpenAI is synchronous
        return None

    def handle_errors(self, error: Exception) -> ModelError:
        """
        Handle and convert OpenAI-specific errors.

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
                "functions": True,
                "tools": self._model_name.startswith("gpt-4"),
                "vision": "vision" in self._model_name,
                "max_tokens": self._get_max_tokens(),
                "async": self._is_async_capable,
            },
            "metrics": self.get_metrics(),
        }

    def _get_max_tokens(self) -> int:
        """Get maximum tokens for current model."""
        model_limits = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
        }

        for model_prefix, limit in model_limits.items():
            if self._model_name.startswith(model_prefix):
                return limit

        return 4096  # Default

    def stream_prompt(
        self, prompt: Union[str, List[Dict[str, str]]], callback: callable, **kwargs
    ) -> None:
        """
        Stream responses from OpenAI.

        Args:
            prompt: Text prompt or conversation history
            callback: Function to call with each chunk
            **kwargs: Additional parameters
        """
        kwargs["stream"] = True
        request_id = self.generate_request_id()

        try:
            messages = self.prepare_messages(prompt)

            payload = {
                "model": kwargs.get("model", self._model_name),
                "messages": messages,
                "stream": True,
                **{k: v for k, v in kwargs.items() if k not in ["model", "messages", "stream"]},
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            if self.organization:
                headers["OpenAI-Organization"] = self.organization

            response = self._session.post(
                f"{self.base_url}/chat/completions",
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
