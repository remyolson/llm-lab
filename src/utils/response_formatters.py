"""
Response Formatting Utilities

This module provides standardized response formatting patterns used throughout the LLM Lab.
It ensures consistent API response structures and evaluation result formats.
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..types import APIResponse, ErrorResponse, ResponseData


def create_success_response(
    data: ResponseData = None,
    message: str = "Operation successful",
    metadata: Dict[str, Any | None] = None,
) -> APIResponse:
    """
    Create a standardized success response.

    Args:
        data: The response data
        message: Success message
        metadata: Additional metadata

    Returns:
        Standardized success response dictionary
    """
    response = {
        "success": True,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if data is not None:
        response["data"] = data

    if metadata:
        response["metadata"] = metadata

    return response


def create_error_response(
    error: str | Exception,
    error_code: str | None = None,
    details: Dict[str, Any | None] = None,
    status_code: int = 500,
) -> Dict[str, Any]:
    """
    Create a standardized error response.

    Args:
        error: Error message or exception object
        error_code: Optional error code for categorization
        details: Additional error details
        status_code: HTTP status code

    Returns:
        Standardized error response dictionary
    """
    error_message = str(error)

    response = {
        "success": False,
        "error": error_message,
        "timestamp": datetime.utcnow().isoformat(),
        "status_code": status_code,
    }

    if error_code:
        response["error_code"] = error_code

    if details:
        response["details"] = details

    # Add exception type if error is an Exception
    if isinstance(error, Exception):
        response["error_type"] = type(error).__name__

    return response


def create_evaluation_result(
    success: bool,
    score: float,
    method: str,
    matched_keywords: List[str | None] = None,
    details: Dict[str, Any | None] = None,
    metadata: Dict[str, Any | None] = None,
) -> Dict[str, Any]:
    """
    Create a standardized evaluation result format.

    Args:
        success: Whether the evaluation was successful
        score: Evaluation score (0.0 to 1.0)
        method: Evaluation method used
        matched_keywords: List of matched keywords
        details: Additional evaluation details
        metadata: Additional metadata

    Returns:
        Standardized evaluation result dictionary
    """
    result = {
        "success": success,
        "score": score,
        "method": method,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if matched_keywords is not None:
        result["matched_keywords"] = matched_keywords

    if details:
        result["details"] = details

    if metadata:
        result["metadata"] = metadata

    return result


def create_provider_response(
    provider_name: str,
    model_name: str,
    response_text: str,
    usage_stats: Dict[str, Any | None] = None,
    generation_time: float | None = None,
    metadata: Dict[str, Any | None] = None,
) -> Dict[str, Any]:
    """
    Create a standardized provider response format.

    Args:
        provider_name: Name of the provider
        model_name: Name of the model used
        response_text: Generated response text
        usage_stats: Token usage and other statistics
        generation_time: Time taken to generate response in seconds
        metadata: Additional metadata

    Returns:
        Standardized provider response dictionary
    """
    response = {
        "provider": provider_name,
        "model": model_name,
        "response": response_text,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if usage_stats:
        response["usage"] = usage_stats

    if generation_time is not None:
        response["generation_time"] = generation_time

    if metadata:
        response["metadata"] = metadata

    return response


def create_batch_response(
    results: List[Dict[str, Any]],
    total_items: int | None = None,
    success_count: int | None = None,
    error_count: int | None = None,
    metadata: Dict[str, Any | None] = None,
) -> Dict[str, Any]:
    """
    Create a standardized batch operation response.

    Args:
        results: List of individual results
        total_items: Total number of items processed
        success_count: Number of successful operations
        error_count: Number of failed operations
        metadata: Additional metadata

    Returns:
        Standardized batch response dictionary
    """
    # Calculate counts if not provided
    if total_items is None:
        total_items = len(results)

    if success_count is None:
        success_count = sum(1 for r in results if r.get("success", False))

    if error_count is None:
        error_count = total_items - success_count

    response = {
        "success": error_count == 0,
        "total_items": total_items,
        "success_count": success_count,
        "error_count": error_count,
        "results": results,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if metadata:
        response["metadata"] = metadata

    return response


def create_comparison_result(
    models: List[str],
    metrics: Dict[str, Any],
    best_model: str | None = None,
    rankings: List[Dict[str, Any | None]] = None,
    summary: str | None = None,
    metadata: Dict[str, Any | None] = None,
) -> Dict[str, Any]:
    """
    Create a standardized model comparison result format.

    Args:
        models: List of model names compared
        metrics: Dictionary of comparison metrics
        best_model: Name of the best performing model
        rankings: List of model rankings with scores
        summary: Summary of the comparison
        metadata: Additional metadata

    Returns:
        Standardized comparison result dictionary
    """
    result = {
        "models": models,
        "metrics": metrics,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if best_model:
        result["best_model"] = best_model

    if rankings:
        result["rankings"] = rankings

    if summary:
        result["summary"] = summary

    if metadata:
        result["metadata"] = metadata

    return result


def format_model_info(
    model_name: str,
    provider: str,
    capabilities: List[str],
    max_tokens: int | None = None,
    context_window: int | None = None,
    additional_info: Dict[str, Any | None] = None,
) -> Dict[str, Any]:
    """
    Format model information in a standardized structure.

    Args:
        model_name: Name of the model
        provider: Provider name
        capabilities: List of model capabilities
        max_tokens: Maximum tokens the model can generate
        context_window: Context window size
        additional_info: Additional model information

    Returns:
        Formatted model information dictionary
    """
    info = {
        "model_name": model_name,
        "provider": provider,
        "capabilities": capabilities,
    }

    if max_tokens is not None:
        info["max_tokens"] = max_tokens

    if context_window is not None:
        info["context_window"] = context_window

    if additional_info:
        info.update(additional_info)

    return info


def format_usage_stats(
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    requests_made: int | None = None,
    cost_estimate: float | None = None,
) -> Dict[str, Any]:
    """
    Format usage statistics in a standardized structure.

    Args:
        prompt_tokens: Number of prompt tokens used
        completion_tokens: Number of completion tokens generated
        total_tokens: Total tokens used
        requests_made: Number of API requests made
        cost_estimate: Estimated cost of the operations

    Returns:
        Formatted usage statistics dictionary
    """
    stats = {}

    if prompt_tokens is not None:
        stats["prompt_tokens"] = prompt_tokens

    if completion_tokens is not None:
        stats["completion_tokens"] = completion_tokens

    if total_tokens is not None:
        stats["total_tokens"] = total_tokens
    elif prompt_tokens is not None and completion_tokens is not None:
        stats["total_tokens"] = prompt_tokens + completion_tokens

    if requests_made is not None:
        stats["requests_made"] = requests_made

    if cost_estimate is not None:
        stats["cost_estimate"] = cost_estimate

    return stats


def paginate_results(
    items: List[Any], page: int = 1, page_size: int = 20, total_count: int | None = None
) -> Dict[str, Any]:
    """
    Create a paginated response structure.

    Args:
        items: List of items to paginate
        page: Current page number (1-based)
        page_size: Number of items per page
        total_count: Total number of items available

    Returns:
        Paginated response dictionary
    """
    if total_count is None:
        total_count = len(items)

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_items = items[start_idx:end_idx]

    total_pages = (total_count + page_size - 1) // page_size
    has_next = page < total_pages
    has_prev = page > 1

    return {
        "items": page_items,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_items": total_count,
            "total_pages": total_pages,
            "has_next": has_next,
            "has_prev": has_prev,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


class ResponseTimer:
    """Context manager for timing operations and including timing in responses."""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    def add_timing_to_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Add timing information to a response dictionary."""
        response["execution_time"] = self.elapsed_time
        return response
