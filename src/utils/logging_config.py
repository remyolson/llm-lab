"""
Configurable logging utilities for LLM Lab.

This module provides a centralized logging configuration system with:
- Multiple verbosity levels
- Colored output for different log levels
- Structured logging for JSON output
- Performance monitoring capabilities
- Provider-specific loggers
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class LogLevel(Enum):
    """Logging levels with descriptions."""

    QUIET = (logging.CRITICAL + 1, "Only show critical errors")
    ERROR = (logging.ERROR, "Show errors only")
    WARNING = (logging.WARNING, "Show warnings and errors")
    INFO = (logging.INFO, "Show general information (default)")
    DEBUG = (logging.DEBUG, "Show detailed debug information")
    TRACE = (logging.DEBUG - 5, "Show all possible information")

    def __init__(self, level: int, description: str):
        self.level = level
        self.description = description


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        """Format the log record with colors."""
        if not hasattr(record, "no_color") or not record.no_color:
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        return super().format(record)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        """Format the log record as JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "provider"):
            log_entry["provider"] = record.provider
        if hasattr(record, "model"):
            log_entry["model"] = record.model
        if hasattr(record, "duration"):
            log_entry["duration"] = record.duration
        if hasattr(record, "tokens"):
            log_entry["tokens"] = record.tokens
        if hasattr(record, "cost"):
            log_entry["cost"] = record.cost

        return json.dumps(log_entry)


class PerformanceLogger:
    """Logger for tracking performance metrics."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_api_call(
        self,
        provider: str,
        model: str,
        duration: float,
        tokens: int | None = None,
        cost: float | None = None,
        success: bool = True,
    ):
        """Log an API call with performance metrics."""
        extra = {
            "provider": provider,
            "model": model,
            "duration": duration,
            "tokens": tokens,
            "cost": cost,
            "success": success,
        }

        if success:
            self.logger.info(
                f"API call to {provider}/{model} completed in {duration:.2f}s", extra=extra
            )
        else:
            self.logger.error(
                f"API call to {provider}/{model} failed after {duration:.2f}s", extra=extra
            )

    def log_benchmark_progress(
        self, dataset: str, model: str, current: int, total: int, avg_time: float | None = None
    ):
        """Log benchmark progress."""
        progress = (current / total) * 100
        extra = {
            "dataset": dataset,
            "model": model,
            "progress": progress,
            "current": current,
            "total": total,
        }

        if avg_time:
            extra["avg_time"] = avg_time

        self.logger.info(f"Benchmark progress: {current}/{total} ({progress:.1f}%)", extra=extra)


class LoggingConfig:
    """Centralized logging configuration."""

    def __init__(
        self,
        level: str | LogLevel = LogLevel.INFO,
        format_style: str = "colored",  # "colored", "plain", "json"
        output_file: str | None = None,
        enable_performance_logging: bool = True,
        provider_specific_levels: Dict[str, str | None] = None,
    ):
        """
        Initialize logging configuration.

        Args:
            level: Global logging level
            format_style: Output format style
            output_file: Optional file to write logs to
            enable_performance_logging: Whether to enable performance metrics
            provider_specific_levels: Specific log levels for providers
        """
        self.level = self._parse_level(level)
        self.format_style = format_style
        self.output_file = output_file
        self.enable_performance_logging = enable_performance_logging
        self.provider_specific_levels = provider_specific_levels or {}

        # Store original handlers to allow reset
        self._original_handlers = {}

    def _parse_level(self, level: str | LogLevel) -> LogLevel:
        """Parse log level from string or enum."""
        if isinstance(level, LogLevel):
            return level

        level_str = level.upper()
        for log_level in LogLevel:
            if log_level.name == level_str:
                return log_level

        # Handle standard logging level names
        standard_levels = {
            "CRITICAL": LogLevel.ERROR,
            "FATAL": LogLevel.ERROR,
        }

        if level_str in standard_levels:
            return standard_levels[level_str]

        raise ValueError(f"Invalid log level: {level}")

    def setup(self):
        """Set up logging configuration."""
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.level.level)

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create console handler
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(self.level.level)

        # Set formatter based on style
        if self.format_style == "json":
            formatter = StructuredFormatter()
        elif self.format_style == "plain":
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        else:  # colored
            formatter = ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Add file handler if specified
        if self.output_file:
            file_handler = logging.FileHandler(self.output_file)
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file

            # Use JSON format for file output
            file_formatter = StructuredFormatter()
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

        # Configure provider-specific loggers
        for provider, level_str in self.provider_specific_levels.items():
            provider_level = self._parse_level(level_str)
            provider_logger = logging.getLogger(f"llm_lab.providers.{provider}")
            provider_logger.setLevel(provider_level.level)

        # Set up performance logging
        if self.enable_performance_logging:
            perf_logger = logging.getLogger("llm_lab.performance")
            perf_logger.setLevel(logging.INFO)

    @classmethod
    def from_env(cls) -> "LoggingConfig":
        """Create logging config from environment variables."""
        level = os.getenv("LOG_LEVEL", "INFO")
        format_style = os.getenv("LOG_FORMAT", "colored")
        output_file = os.getenv("LOG_FILE")
        enable_perf = os.getenv("ENABLE_PERFORMANCE_LOGGING", "true").lower() == "true"

        # Parse provider-specific levels from env
        provider_levels = {}
        for key, value in os.environ.items():
            if key.startswith("LOG_LEVEL_"):
                provider = key.replace("LOG_LEVEL_", "").lower()
                provider_levels[provider] = value

        return cls(
            level=level,
            format_style=format_style,
            output_file=output_file,
            enable_performance_logging=enable_perf,
            provider_specific_levels=provider_levels,
        )

    @classmethod
    def from_file(cls, config_path: str) -> "LoggingConfig":
        """Create logging config from JSON file."""
        with open(config_path, "r") as f:
            config = json.load(f)

        return cls(
            level=config.get("level", "INFO"),
            format_style=config.get("format_style", "colored"),
            output_file=config.get("output_file"),
            enable_performance_logging=config.get("enable_performance_logging", True),
            provider_specific_levels=config.get("provider_specific_levels", {}),
        )


# Global logging configuration
_global_config: LoggingConfig | None = None


def setup_logging(
    level: str | LogLevel = LogLevel.INFO,
    format_style: str = "colored",
    output_file: str | None = None,
    config_file: str | None = None,
    **kwargs,
):
    """
    Set up logging for the entire application.

    Args:
        level: Global logging level
        format_style: Output format ("colored", "plain", "json")
        output_file: Optional file to write logs to
        config_file: Optional JSON config file to load settings from
        **kwargs: Additional arguments for LoggingConfig
    """
    global _global_config

    if config_file and os.path.exists(config_file):
        _global_config = LoggingConfig.from_file(config_file)
    else:
        _global_config = LoggingConfig(
            level=level, format_style=format_style, output_file=output_file, **kwargs
        )

    _global_config.setup()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    # Set up default logging if not already configured
    if _global_config is None:
        setup_logging()

    return logging.getLogger(name)


def get_performance_logger() -> PerformanceLogger:
    """Get a performance logger instance."""
    logger = get_logger("llm_lab.performance")
    return PerformanceLogger(logger)


@contextmanager
def log_execution_time(logger: logging.Logger, operation: str, **extra_fields):
    """
    Context manager to log execution time of an operation.

    Args:
        logger: Logger to use
        operation: Description of the operation
        **extra_fields: Additional fields to include in log
    """
    start_time = time.time()
    try:
        yield
        duration = time.time() - start_time
        extra = {"duration": duration, **extra_fields}
        logger.info(f"{operation} completed in {duration:.2f}s", extra=extra)
    except Exception as e:
        duration = time.time() - start_time
        extra = {"duration": duration, "error": str(e), **extra_fields}
        logger.error(f"{operation} failed after {duration:.2f}s: {e}", extra=extra)
        raise


def log_provider_call(
    provider: str, model: str, operation: str, duration: float, success: bool = True, **extra_fields
):
    """
    Log a provider API call.

    Args:
        provider: Provider name
        model: Model name
        operation: Operation being performed
        duration: Duration in seconds
        success: Whether the operation succeeded
        **extra_fields: Additional fields to log
    """
    perf_logger = get_performance_logger()
    perf_logger.log_api_call(
        provider=provider, model=model, duration=duration, success=success, **extra_fields
    )


# Convenience functions for different log levels
def set_quiet_mode():
    """Set logging to quiet mode (errors only)."""
    setup_logging(level=LogLevel.QUIET)


def set_verbose_mode():
    """Set logging to verbose mode (debug level)."""
    setup_logging(level=LogLevel.DEBUG)


def set_trace_mode():
    """Set logging to trace mode (maximum verbosity)."""
    setup_logging(level=LogLevel.TRACE)


# Default setup using environment variables
def setup_default_logging():
    """Set up logging using environment variables and sensible defaults."""
    global _global_config

    if _global_config is None:
        _global_config = LoggingConfig.from_env()
        _global_config.setup()


# Auto-setup on import
setup_default_logging()
