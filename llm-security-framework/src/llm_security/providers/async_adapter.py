"""Asynchronous adapter base class and implementations."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import aiohttp

from .base import ModelAdapter, ModelError, ModelResponse, ResponseStatus

logger = logging.getLogger(__name__)


class AsyncModelAdapter(ABC):
    """Abstract base class for asynchronous model adapters."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize async adapter.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self._create_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _create_session(self) -> None:
        """Create async HTTP session."""
        # Configure connector with connection pooling
        self._connector = aiohttp.TCPConnector(
            limit=self.config.get("max_connections", 100),
            limit_per_host=self.config.get("max_connections_per_host", 20),
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
        )

        # Configure timeout
        timeout = aiohttp.ClientTimeout(
            total=self.config.get("timeout_seconds", 30),
            connect=self.config.get("connect_timeout", 10),
        )

        self._session = aiohttp.ClientSession(connector=self._connector, timeout=timeout)

    @abstractmethod
    async def send_prompt_async(
        self, prompt: Union[str, List[Dict[str, str]]], **kwargs
    ) -> ModelResponse:
        """
        Send a prompt asynchronously.

        Args:
            prompt: Text prompt or conversation history
            **kwargs: Additional parameters

        Returns:
            ModelResponse object
        """
        pass

    async def send_batch_prompts(
        self, prompts: List[Union[str, List[Dict[str, str]]]], **kwargs
    ) -> List[ModelResponse]:
        """
        Send multiple prompts concurrently.

        Args:
            prompts: List of prompts
            **kwargs: Additional parameters

        Returns:
            List of ModelResponse objects
        """
        tasks = [self.send_prompt_async(prompt, **kwargs) for prompt in prompts]

        return await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        """Close async session and connections."""
        if self._session:
            await self._session.close()
        if self._connector:
            await self._connector.close()


class AsyncOpenAIAdapter(AsyncModelAdapter):
    """Async version of OpenAI adapter."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize async OpenAI adapter."""
        super().__init__(config)
        self.base_url = config.get("api_base_url", "https://api.openai.com/v1")
        self.api_key = config.get("api_key")
        self.model_name = config.get("model_name", "gpt-3.5-turbo")

    async def send_prompt_async(
        self, prompt: Union[str, List[Dict[str, str]]], **kwargs
    ) -> ModelResponse:
        """Send prompt to OpenAI asynchronously."""
        if not self._session:
            await self._create_session()

        request_id = f"async_openai_{int(time.time())}_{id(prompt)}"
        start_time = time.time()

        try:
            # Prepare messages
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt

            # Build payload
            payload = {
                "model": kwargs.get("model", self.model_name),
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens"),
            }

            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}

            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Make async request
            async with self._session.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            ) as response:
                response_data = await response.json()
                latency_ms = (time.time() - start_time) * 1000

                if response.status == 200:
                    # Parse successful response
                    choices = response_data.get("choices", [])
                    content = choices[0]["message"]["content"] if choices else ""

                    usage = response_data.get("usage", {})
                    tokens_used = {
                        "prompt": usage.get("prompt_tokens", 0),
                        "completion": usage.get("completion_tokens", 0),
                        "total": usage.get("total_tokens", 0),
                    }

                    return ModelResponse(
                        content=content,
                        model=response_data.get("model", self.model_name),
                        provider="openai",
                        request_id=request_id,
                        status=ResponseStatus.SUCCESS,
                        tokens_used=tokens_used,
                        latency_ms=latency_ms,
                        raw_response=response_data,
                    )
                else:
                    # Handle error
                    error_msg = response_data.get("error", {}).get(
                        "message", f"HTTP {response.status}"
                    )
                    return ModelResponse(
                        content="",
                        model=self.model_name,
                        provider="openai",
                        request_id=request_id,
                        status=ResponseStatus.ERROR,
                        latency_ms=latency_ms,
                        error_message=error_msg,
                    )

        except Exception as e:
            return ModelResponse(
                content="",
                model=self.model_name,
                provider="openai",
                request_id=request_id,
                status=ResponseStatus.ERROR,
                latency_ms=(time.time() - start_time) * 1000,
                error_message=str(e),
            )
