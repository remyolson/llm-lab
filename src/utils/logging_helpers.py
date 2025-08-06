"""
Logging Helper Utilities

This module provides standardized logging setup and utilities used throughout the LLM Lab.
It centralizes logger configuration, formatting, and common logging patterns.
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import functools
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """
    Get a standardized logger instance.

    Args:
        name: Logger name (typically __name__)
        level: Optional logging level override

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Set level if specified
    if level:
        numeric_level = getattr(logging, level.upper(), None)
        if isinstance(numeric_level, int):
            logger.setLevel(numeric_level)

    return logger


def setup_module_logger(
    name: str,
    level: str = "INFO",
    format_string: str | None = None,
    include_timestamp: bool = True,
    include_level: bool = True,
    include_name: bool = True,
) -> logging.Logger:
    """
    Set up a logger with standardized configuration for a module.

    Args:
        name: Logger name (typically __name__)
        level: Logging level
        format_string: Custom format string
        include_timestamp: Whether to include timestamp
        include_level: Whether to include log level
        include_name: Whether to include logger name

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Set level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    # Create formatter
    if format_string is None:
        format_parts = []
        if include_timestamp:
            format_parts.append("%(asctime)s")
        if include_level:
            format_parts.append("%(levelname)s")
        if include_name:
            format_parts.append("%(name)s")
        format_parts.append("%(message)s")
        format_string = " - ".join(format_parts)

    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def log_function_calls(
    logger: logging.Logger | None = None,
    level: str = "DEBUG",
    include_args: bool = False,
    include_result: bool = False,
    include_timing: bool = True,
):
    """
    Decorator to log function calls with optional arguments, results, and timing.

    Args:
        logger: Logger instance to use (creates one if None)
        level: Logging level for the messages
        include_args: Whether to log function arguments
        include_result: Whether to log function result
        include_timing: Whether to log execution time
    """

    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        log_level = getattr(logging, level.upper(), logging.DEBUG)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time() if include_timing else None

            # Log function entry
            if include_args:
                args_str = ", ".join([repr(arg) for arg in args])
                kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
                all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                logger.log(log_level, f"Calling {func.__name__}({all_args})")
            else:
                logger.log(log_level, f"Calling {func.__name__}")

            try:
                result = func(*args, **kwargs)

                # Log function exit
                if include_result:
                    logger.log(log_level, f"{func.__name__} returned: {repr(result)}")
                else:
                    logger.log(log_level, f"{func.__name__} completed successfully")

                if include_timing and start_time:
                    elapsed = time.time() - start_time
                    logger.log(log_level, f"{func.__name__} took {elapsed:.3f}s")

                return result

            except Exception as e:
                logger.error(f"{func.__name__} failed with error: {e}", exc_info=True)
                if include_timing and start_time:
                    elapsed = time.time() - start_time
                    logger.error(f"{func.__name__} failed after {elapsed:.3f}s")
                raise

        return wrapper

    return decorator


def log_performance(
    logger: logging.Logger | None = None, level: str = "INFO", threshold_seconds: float = 1.0
):
    """
    Decorator to log performance warnings for slow functions.

    Args:
        logger: Logger instance to use
        level: Logging level for performance messages
        threshold_seconds: Threshold in seconds to trigger performance warning
    """

    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        log_level = getattr(logging, level.upper(), logging.INFO)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time

                if elapsed > threshold_seconds:
                    logger.log(
                        log_level,
                        f"PERFORMANCE: {func.__name__} took {elapsed:.3f}s "
                        f"(threshold: {threshold_seconds}s)",
                    )

                return result

            except Exception:
                elapsed = time.time() - start_time
                logger.error(f"{func.__name__} failed after {elapsed:.3f}s")
                raise

        return wrapper

    return decorator


class LoggerContext:
    """Context manager for temporary logger configuration changes."""

    def __init__(
        self,
        logger: logging.Logger,
        level: str | None = None,
        handler: logging.Handler | None = None,
    ):
        self.logger = logger
        self.new_level = getattr(logging, level.upper()) if level else None
        self.new_handler = handler
        self.original_level = logger.level
        self.original_handlers = logger.handlers.copy()

    def __enter__(self):
        if self.new_level is not None:
            self.logger.setLevel(self.new_level)

        if self.new_handler is not None:
            self.logger.addHandler(self.new_handler)

        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original level
        self.logger.setLevel(self.original_level)

        # Restore original handlers
        self.logger.handlers.clear()
        self.logger.handlers.extend(self.original_handlers)


def create_file_logger(
    name: str,
    file_path: Path,
    level: str = "INFO",
    format_string: str | None = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Create a logger that writes to a file with rotation.

    Args:
        name: Logger name
        file_path: Path to log file
        level: Logging level
        format_string: Custom format string
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured file logger
    """
    from logging.handlers import RotatingFileHandler

    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create rotating file handler
    handler = RotatingFileHandler(file_path, maxBytes=max_bytes, backupCount=backup_count)
    handler.setLevel(numeric_level)

    # Create formatter
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def log_exception_details(
    logger: logging.Logger,
    exception: Exception,
    context: str | None = None,
    extra_data: dict | None = None,
):
    """
    Log detailed exception information.

    Args:
        logger: Logger instance
        exception: Exception to log
        context: Additional context information
        extra_data: Additional data to include in log
    """
    error_msg = f"Exception: {type(exception).__name__}: {str(exception)}"

    if context:
        error_msg = f"{context} - {error_msg}"

    logger.error(error_msg, exc_info=True)

    if extra_data:
        logger.error(f"Additional context: {extra_data}")


def structured_log(logger: logging.Logger, level: str, message: str, **kwargs):
    """
    Log a structured message with additional data.

    Args:
        logger: Logger instance
        level: Logging level
        message: Main log message
        **kwargs: Additional structured data
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    if kwargs:
        structured_data = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        full_message = f"{message} | {structured_data}"
    else:
        full_message = message

    logger.log(log_level, full_message)


# Common logger instances that can be imported directly
default_logger = get_logger(__name__)
performance_logger = get_logger(f"{__name__}.performance")
error_logger = get_logger(f"{__name__}.errors")
