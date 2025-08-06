"""Local model adapter for HTTP endpoint-based LLMs."""

import base64
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import ProviderConfig
from .base import ModelAdapter, ModelError, ModelResponse, ProviderType, ResponseStatus

logger = logging.getLogger(__name__)


class LocalModelAdapter(ModelAdapter):
    """Adapter for local models accessed via HTTP endpoints."""

    DEFAULT_BASE_URL = "http://localhost:8000"
    DEFAULT_MODEL = "local-model"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize local model adapter.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._provider_type = ProviderType.LOCAL
        self._session = self._create_session()
        self._connection_pool = {}

    def _create_session(self) -> requests.Session:
        """Create HTTP session with connection pooling."""
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.get("max_retries", 3),
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
        )

        # Use larger connection pool for local models
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_maxsize=self.config.get("max_connections", 20),
            pool_connections=self.config.get("max_connections", 20),
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def configure(self, config_dict: Dict[str, Any]) -> None:
        """
        Configure the local model adapter.

        Args:
            config_dict: Configuration dictionary
        """
        self.config = config_dict

        # Set base URL
        self.base_url = config_dict.get("api_base_url", self.DEFAULT_BASE_URL)

        # Set model name
        self._model_name = config_dict.get("model_name", self.DEFAULT_MODEL)

        # Set endpoints
        self.endpoints = {
            "generate": config_dict.get("generate_endpoint", "/generate"),
            "chat": config_dict.get("chat_endpoint", "/chat"),
            "completions": config_dict.get("completions_endpoint", "/completions"),
            "health": config_dict.get("health_endpoint", "/health"),
            "info": config_dict.get("info_endpoint", "/info"),
        }

        # Set authentication
        self.auth_type = config_dict.get("auth_type", "none")  # none, bearer, basic, custom
        self.auth_token = config_dict.get("auth_token")
        self.auth_username = config_dict.get("auth_username")
        self.auth_password = config_dict.get("auth_password")

        # Set request/response format
        self.request_format = config_dict.get("request_format", "standard")
        self.response_format = config_dict.get("response_format", "standard")

        # Custom field mappings
        self.field_mappings = config_dict.get("field_mappings", {})

        # Set capabilities
        self._is_async_capable = config_dict.get("supports_async", False)

        # Set timeout
        self.timeout = config_dict.get("timeout_seconds", 60)

        # Set API version
        self._api_version = config_dict.get("api_version", "v1")

        # Check health on configure
        if config_dict.get("check_health_on_start", True):
            self._check_health()

    def _check_health(self) -> bool:
        """
        Check if the local model endpoint is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            url = urljoin(self.base_url, self.endpoints["health"])
            headers = self._get_auth_headers()

            response = self._session.get(url, headers=headers, timeout=5)

            if response.status_code == 200:
                logger.info(f"Local model endpoint is healthy: {self.base_url}")
                return True
            else:
                logger.warning(f"Local model endpoint unhealthy: {response.status_code}")
                return False

        except Exception as e:
            logger.warning(f"Failed to check local model health: {e}")
            return False

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers based on auth type."""
        headers = {}

        if self.auth_type == "bearer" and self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        elif self.auth_type == "basic" and self.auth_username and self.auth_password:
            credentials = f"{self.auth_username}:{self.auth_password}"
            encoded = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"
        elif self.auth_type == "custom" and self.auth_token:
            headers["X-API-Key"] = self.auth_token

        return headers

    def _format_request(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Format request based on configured format.

        Args:
            messages: Message list
            **kwargs: Additional parameters

        Returns:
            Formatted request payload
        """
        if self.request_format == "openai":
            # OpenAI-compatible format
            payload = {
                "model": kwargs.get("model", self._model_name),
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1024),
                "stream": kwargs.get("stream", False),
            }
        elif self.request_format == "llama":
            # Llama.cpp server format
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            payload = {
                "prompt": prompt,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1024),
                "stop": kwargs.get("stop", []),
            }
        elif self.request_format == "custom":
            # Custom format with field mappings
            payload = {}

            # Map standard fields to custom names
            field_map = self.field_mappings
            payload[field_map.get("messages", "messages")] = messages
            payload[field_map.get("temperature", "temperature")] = kwargs.get("temperature", 0.7)
            payload[field_map.get("max_tokens", "max_tokens")] = kwargs.get("max_tokens", 1024)

            # Add any additional custom fields
            for key, value in kwargs.items():
                if key in field_map:
                    payload[field_map[key]] = value
                elif key not in ["model", "temperature", "max_tokens"]:
                    payload[key] = value
        else:
            # Standard format
            payload = {"messages": messages, **kwargs}

        return payload

    def _parse_response(self, response_data: Dict[str, Any]) -> tuple[str, Dict[str, int]]:
        """
        Parse response based on configured format.

        Args:
            response_data: Raw response data

        Returns:
            Tuple of (content, tokens_used)
        """
        content = ""
        tokens_used = {}

        if self.response_format == "openai":
            # OpenAI-compatible format
            choices = response_data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")

            usage = response_data.get("usage", {})
            tokens_used = {
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
                "total": usage.get("total_tokens", 0),
            }
        elif self.response_format == "llama":
            # Llama.cpp server format
            content = response_data.get("content", "")
            tokens_used = {
                "prompt": response_data.get("tokens_evaluated", 0),
                "completion": response_data.get("tokens_predicted", 0),
                "total": response_data.get("tokens_evaluated", 0)
                + response_data.get("tokens_predicted", 0),
            }
        elif self.response_format == "custom":
            # Custom format with field mappings
            field_map = self.field_mappings

            # Extract content
            content_field = field_map.get("content", "content")
            if content_field in response_data:
                content = response_data[content_field]
            elif "response" in response_data:
                content = response_data["response"]
            elif "text" in response_data:
                content = response_data["text"]
            else:
                content = str(response_data)

            # Extract token usage if available
            tokens_used = {
                "prompt": response_data.get(field_map.get("prompt_tokens", "prompt_tokens"), 0),
                "completion": response_data.get(
                    field_map.get("completion_tokens", "completion_tokens"), 0
                ),
                "total": response_data.get(field_map.get("total_tokens", "total_tokens"), 0),
            }
        else:
            # Try to extract content from common fields
            content = (
                response_data.get("content")
                or response_data.get("response")
                or response_data.get("text")
                or response_data.get("output")
                or str(response_data)
            )

            # Try to extract token counts
            tokens_used = {
                "prompt": response_data.get("prompt_tokens", 0),
                "completion": response_data.get("completion_tokens", 0),
                "total": response_data.get("total_tokens", 0),
            }

        return content, tokens_used

    def send_prompt(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> ModelResponse:
        """
        Send a prompt to the local model.

        Args:
            prompt: Text prompt or conversation history
            **kwargs: Additional parameters

        Returns:
            ModelResponse object
        """
        request_id = self.generate_request_id()
        start_time = time.time()

        try:
            # Prepare messages
            messages = self.prepare_messages(prompt)

            # Format request
            payload = self._format_request(messages, **kwargs)

            # Prepare headers
            headers = {"Content-Type": "application/json", **self._get_auth_headers()}

            # Add custom headers
            headers.update(self.config.get("custom_headers", {}))

            # Determine endpoint
            endpoint = self.endpoints.get(kwargs.get("endpoint_type", "chat"))
            url = urljoin(self.base_url, endpoint)

            # Make request
            response = self._session.post(url, json=payload, headers=headers, timeout=self.timeout)

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

        # Parse response
        content, tokens_used = self._parse_response(data)

        # Create response
        response = ModelResponse(
            content=content,
            model=data.get("model", self._model_name),
            provider=self.provider_name,
            request_id=request_id,
            status=ResponseStatus.SUCCESS,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            metadata={"endpoint": self.base_url, "format": self.response_format},
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
            message = "Authentication failed"
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
            message = error_data.get("error", message)
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
            metadata={"status_code": response.status_code, "endpoint": self.base_url},
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
        # Could implement polling for async endpoints
        return None

    def handle_errors(self, error: Exception) -> ModelError:
        """
        Handle and convert local model errors.

        Args:
            error: Original exception

        Returns:
            ModelError with standardized information
        """
        if isinstance(error, requests.exceptions.ConnectionError):
            return ModelError(
                message="Connection error - is the local model running?",
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
        # Try to get info from endpoint
        info = {}
        try:
            url = urljoin(self.base_url, self.endpoints["info"])
            headers = self._get_auth_headers()

            response = self._session.get(url, headers=headers, timeout=5)

            if response.status_code == 200:
                info = response.json()
        except:
            pass

        return {
            "provider": self.provider_name,
            "model": self._model_name,
            "api_version": self._api_version,
            "endpoint": self.base_url,
            "auth_type": self.auth_type,
            "request_format": self.request_format,
            "response_format": self.response_format,
            "capabilities": {
                "streaming": info.get("streaming", False),
                "functions": info.get("functions", False),
                "max_tokens": info.get("max_tokens", 2048),
                "async": self._is_async_capable,
            },
            "server_info": info,
            "metrics": self.get_metrics(),
        }
