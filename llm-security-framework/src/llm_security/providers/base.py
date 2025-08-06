"""Abstract base class for LLM provider adapters."""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported LLM provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    LOCAL = "local"
    GOOGLE = "google"
    COHERE = "cohere"
    CUSTOM = "custom"


class ResponseStatus(Enum):
    """Response status types."""

    SUCCESS = "success"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION_ERROR = "authentication_error"


@dataclass
class ModelResponse:
    """Standardized response from model adapters."""

    content: str
    model: str
    provider: str
    request_id: str
    status: ResponseStatus = ResponseStatus.SUCCESS
    tokens_used: Optional[Dict[str, int]] = None  # {"prompt": X, "completion": Y, "total": Z}
    latency_ms: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_response: Optional[Any] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "request_id": self.request_id,
            "status": self.status.value,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "error_message": self.error_message,
        }

    @property
    def is_success(self) -> bool:
        """Check if response was successful."""
        return self.status == ResponseStatus.SUCCESS

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        if self.tokens_used:
            return self.tokens_used.get("total", 0)
        return 0


@dataclass
class ModelError(Exception):
    """Custom exception for model adapter errors."""

    message: str
    provider: str
    error_type: ResponseStatus
    details: Optional[Dict[str, Any]] = None
    original_exception: Optional[Exception] = None

    def __str__(self):
        return f"[{self.provider}] {self.error_type.value}: {self.message}"


class ModelAdapter(ABC):
    """Abstract base class for LLM provider adapters."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model adapter.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._provider_type = ProviderType.CUSTOM
        self._model_name = "unknown"
        self._api_version = "unknown"
        self._is_async_capable = False
        self._request_history: List[ModelResponse] = []
        self._metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_latency_ms": 0,
        }

        # Configure the adapter
        self.configure(self.config)

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_type.value

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def api_version(self) -> str:
        """Get the API version."""
        return self._api_version

    @property
    def is_async_capable(self) -> bool:
        """Check if adapter supports async operations."""
        return self._is_async_capable

    @abstractmethod
    def send_prompt(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> ModelResponse:
        """
        Send a prompt to the model.

        Args:
            prompt: Text prompt or conversation history
            **kwargs: Additional provider-specific parameters

        Returns:
            ModelResponse object
        """
        pass

    @abstractmethod
    def get_response(self, prompt_id: str) -> Optional[ModelResponse]:
        """
        Get a response by prompt ID (for async operations).

        Args:
            prompt_id: ID of the prompt

        Returns:
            ModelResponse or None if not ready
        """
        pass

    @abstractmethod
    def handle_errors(self, error: Exception) -> ModelError:
        """
        Handle and convert provider-specific errors.

        Args:
            error: Original exception

        Returns:
            ModelError with standardized information
        """
        pass

    @abstractmethod
    def configure(self, config_dict: Dict[str, Any]) -> None:
        """
        Configure the adapter with provider-specific settings.

        Args:
            config_dict: Configuration dictionary
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model and provider.

        Returns:
            Dictionary with model information
        """
        pass

    def generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return f"{self.provider_name}_{uuid.uuid4().hex[:12]}_{int(time.time())}"

    def track_response(self, response: ModelResponse) -> None:
        """
        Track response for metrics and history.

        Args:
            response: ModelResponse to track
        """
        self._request_history.append(response)
        self._metrics["total_requests"] += 1

        if response.is_success:
            self._metrics["successful_requests"] += 1
        else:
            self._metrics["failed_requests"] += 1

        if response.tokens_used:
            self._metrics["total_tokens"] += response.total_tokens

        if response.latency_ms:
            self._metrics["total_latency_ms"] += response.latency_ms

    def normalize_response(
        self, raw_response: Any, request_id: str, start_time: float
    ) -> ModelResponse:
        """
        Normalize provider response to standard format.

        Args:
            raw_response: Raw response from provider
            request_id: Request ID
            start_time: Request start time

        Returns:
            Normalized ModelResponse
        """
        latency_ms = (time.time() - start_time) * 1000

        # Default implementation - should be overridden by subclasses
        response = ModelResponse(
            content="",
            model=self.model_name,
            provider=self.provider_name,
            request_id=request_id,
            latency_ms=latency_ms,
            raw_response=raw_response,
        )

        self.track_response(response)
        return response

    def validate_config(self, required_fields: List[str]) -> None:
        """
        Validate configuration has required fields.

        Args:
            required_fields: List of required field names

        Raises:
            ValueError: If required fields are missing
        """
        missing_fields = [
            field for field in required_fields if field not in self.config or not self.config[field]
        ]

        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {', '.join(missing_fields)}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get adapter performance metrics.

        Returns:
            Dictionary of metrics
        """
        metrics = self._metrics.copy()

        # Calculate averages
        if metrics["total_requests"] > 0:
            metrics["success_rate"] = metrics["successful_requests"] / metrics["total_requests"]
            metrics["average_latency_ms"] = metrics["total_latency_ms"] / metrics["total_requests"]
            metrics["average_tokens"] = metrics["total_tokens"] / metrics["total_requests"]
        else:
            metrics["success_rate"] = 0.0
            metrics["average_latency_ms"] = 0.0
            metrics["average_tokens"] = 0.0

        return metrics

    def get_request_history(
        self, limit: Optional[int] = None, only_errors: bool = False
    ) -> List[ModelResponse]:
        """
        Get request history.

        Args:
            limit: Maximum number of responses to return
            only_errors: Only return error responses

        Returns:
            List of ModelResponse objects
        """
        history = self._request_history

        if only_errors:
            history = [r for r in history if not r.is_success]

        if limit:
            history = history[-limit:]

        return history

    def clear_history(self) -> None:
        """Clear request history and reset metrics."""
        self._request_history.clear()
        self._metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_latency_ms": 0,
        }
        logger.info(f"Cleared history for {self.provider_name} adapter")

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Simple estimation - ~4 characters per token
        # Should be overridden with provider-specific tokenizers
        return len(text) // 4

    def prepare_messages(self, prompt: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """
        Prepare messages in standard format.

        Args:
            prompt: Text prompt or conversation history

        Returns:
            List of message dictionaries
        """
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            return prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

    def __repr__(self) -> str:
        """String representation of adapter."""
        return (
            f"{self.__class__.__name__}("
            f"provider={self.provider_name}, "
            f"model={self.model_name}, "
            f"version={self.api_version})"
        )
