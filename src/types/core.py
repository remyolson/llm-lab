"""
Core TypedDict definitions for LLM Lab

This module provides TypedDict classes for core data structures used throughout
the LLM Lab framework, including provider information, model parameters, and
API response structures.
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

from typing import Dict, List

from typing_extensions import NotRequired, TypedDict


class ProviderInfo(TypedDict):
    """Type-safe provider information structure.

    Used for describing LLM provider capabilities and metadata.
    """

    model_name: str
    provider: str
    max_tokens: int
    capabilities: List[str]
    version: NotRequired[str]
    provider_id: NotRequired[str]
    api_version: NotRequired[str]


class ModelParameters(TypedDict):
    """Type-safe model generation parameters.

    Defines the structure for model generation settings with optional
    advanced parameters.
    """

    temperature: float
    max_tokens: int
    top_p: float
    top_k: NotRequired[int]
    frequency_penalty: NotRequired[float]
    presence_penalty: NotRequired[float]
    stop_sequences: NotRequired[List[str]]

    # Context and sequence limits
    max_prompt_length: NotRequired[int]
    default_max_length: NotRequired[int]
    extended_max_length: NotRequired[int]

    # Evaluation-specific parameters
    eval_temperature_conservative: NotRequired[float]
    eval_temperature_standard: NotRequired[float]
    eval_max_new_tokens_short: NotRequired[int]
    eval_max_new_tokens_medium: NotRequired[int]
    eval_max_new_tokens_long: NotRequired[int]


class ConfigDict(TypedDict):
    """Type-safe configuration dictionary structure.

    Used for nested configuration objects with domain-specific settings.
    """

    network: NotRequired[Dict[str, str | int | float]]
    system: NotRequired[Dict[str, str | int | float]]
    server: NotRequired[Dict[str, str | int | float]]
    monitoring: NotRequired[Dict[str, str | int | float | bool]]
    validation: NotRequired[Dict[str, str | int | float | bool]]
    providers: NotRequired[Dict[str, Dict[str, str | int | float]]]


class APIResponse(TypedDict):
    """Type-safe API response structure.

    Standard structure for API responses throughout the system.
    """

    success: bool
    data: NotRequired[Dict | List | str | int | float | None]
    message: NotRequired[str]
    timestamp: NotRequired[str]
    request_id: NotRequired[str]
    metadata: NotRequired[Dict[str, str | int | float | bool]]


class ErrorResponse(TypedDict):
    """Type-safe error response structure.

    Standardized error response format for API and internal errors.
    """

    success: bool  # Always False for error responses
    error_code: str
    error_message: str
    details: NotRequired[Dict[str, str | int | float | bool]]
    timestamp: NotRequired[str]
    request_id: NotRequired[str]
    stack_trace: NotRequired[str]  # Only in debug mode


class RetryConfig(TypedDict):
    """Type-safe retry configuration structure.

    Configuration for retry behavior and error handling.
    """

    max_retries: int
    retry_delay: float
    exponential_backoff: bool
    backoff_factor: NotRequired[float]
    max_retry_delay: NotRequired[float]
    timeout_seconds: NotRequired[int]
    retry_jitter: NotRequired[bool]
    retry_timeout: NotRequired[int]


class NetworkConfig(TypedDict):
    """Type-safe network configuration structure.

    Network and connectivity settings for timeouts and endpoints.
    """

    default_timeout: int
    generation_timeout: int
    model_pull_timeout: int
    api_request_timeout: NotRequired[int]
    ollama_base_url: NotRequired[str]
    monitoring_poll_interval: NotRequired[float]
    buffer_poll_interval: NotRequired[float]


class SystemConfig(TypedDict):
    """Type-safe system configuration structure.

    System performance and resource management settings.
    """

    default_batch_size: int
    small_batch_size: NotRequired[int]
    large_batch_size: NotRequired[int]
    max_batch_size: NotRequired[int]
    train_batch_size: NotRequired[int]
    eval_batch_size: NotRequired[int]
    download_chunk_size: NotRequired[int]
    buffer_size: NotRequired[int]
    max_workers: NotRequired[int]
    dataloader_workers: NotRequired[int]
    memory_threshold: NotRequired[float]
    vram_threshold: NotRequired[float]


class ServerConfig(TypedDict):
    """Type-safe server configuration structure.

    Server and API configuration settings.
    """

    api_port: int
    api_host: str
    websocket_port: NotRequired[int]
    dashboard_port: NotRequired[int]
    monitoring_port: NotRequired[int]
    cors_origins: NotRequired[List[str]]
    cors_allow_credentials: NotRequired[bool]
    api_prefix: NotRequired[str]
    docs_url: NotRequired[str]
