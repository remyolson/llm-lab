"""Azure OpenAI API adapter for LLM security testing."""

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


class AzureOpenAIAdapter(ModelAdapter):
    """Adapter for Azure OpenAI Service."""

    DEFAULT_API_VERSION = "2024-02-01"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Azure OpenAI adapter.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._provider_type = ProviderType.AZURE_OPENAI
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
        Configure the Azure OpenAI adapter.

        Args:
            config_dict: Configuration dictionary
        """
        self.config = config_dict

        # Set API key
        self.api_key = config_dict.get("api_key") or config_dict.get("azure_api_key")
        if not self.api_key:
            logger.warning("No Azure OpenAI API key configured")

        # Set endpoint (required for Azure)
        self.endpoint = config_dict.get("api_base_url") or config_dict.get("azure_endpoint")
        if not self.endpoint:
            raise ValueError("Azure OpenAI endpoint is required")

        # Ensure endpoint doesn't end with slash
        self.endpoint = self.endpoint.rstrip("/")

        # Set deployment name (required for Azure)
        self.deployment_name = config_dict.get("deployment_name")
        if not self.deployment_name:
            logger.warning("No deployment name configured")

        # Set API version
        self._api_version = config_dict.get("api_version", self.DEFAULT_API_VERSION)

        # Set model name (may be same as deployment name)
        self._model_name = config_dict.get("model_name") or self.deployment_name

        # Set capabilities
        self._is_async_capable = True

        # Set timeout
        self.timeout = config_dict.get("timeout_seconds", 30)

        # Authentication type (api-key or azure-ad)
        self.auth_type = config_dict.get("auth_type", "api-key")

    def send_prompt(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> ModelResponse:
        """
        Send a prompt to Azure OpenAI.

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

            # Build request payload (same as OpenAI)
            payload = {
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

            # Add tools if provided
            if "tools" in kwargs:
                payload["tools"] = kwargs["tools"]
                if "tool_choice" in kwargs:
                    payload["tool_choice"] = kwargs["tool_choice"]

            # Prepare headers
            headers = {"Content-Type": "application/json"}

            # Add authentication
            if self.auth_type == "api-key":
                headers["api-key"] = self.api_key
            else:
                # Azure AD authentication
                headers["Authorization"] = f"Bearer {self.api_key}"

            # Add custom headers
            headers.update(self.config.get("custom_headers", {}))

            # Build URL
            deployment = kwargs.get("deployment_name", self.deployment_name)
            url = (
                f"{self.endpoint}/openai/deployments/{deployment}"
                f"/chat/completions?api-version={self._api_version}"
            )

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

        # Extract content (same format as OpenAI)
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
                "created": data.get("created"),
                "deployment": self.deployment_name,
                "api_version": self._api_version,
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
            message = "Invalid API key or unauthorized"
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
            metadata={
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "deployment": self.deployment_name,
            },
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
        # For now, return None as Azure OpenAI is synchronous
        return None

    def handle_errors(self, error: Exception) -> ModelError:
        """
        Handle and convert Azure-specific errors.

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
            "deployment": self.deployment_name,
            "api_version": self._api_version,
            "endpoint": self.endpoint,
            "auth_type": self.auth_type,
            "capabilities": {
                "streaming": True,
                "functions": True,
                "tools": True,
                "vision": "vision" in str(self._model_name).lower(),
                "max_tokens": 4096,  # Depends on deployment
                "async": self._is_async_capable,
            },
            "metrics": self.get_metrics(),
        }

    def list_deployments(self) -> List[Dict[str, Any]]:
        """
        List available deployments (if accessible).

        Returns:
            List of deployment information
        """
        try:
            headers = {"Content-Type": "application/json"}

            if self.auth_type == "api-key":
                headers["api-key"] = self.api_key
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"

            url = f"{self.endpoint}/openai/deployments?api-version={self._api_version}"

            response = self._session.get(url, headers=headers, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
            else:
                logger.warning(f"Failed to list deployments: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error listing deployments: {e}")
            return []
