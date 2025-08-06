"""
Mock Objects and Utilities

Provides mock objects for testing without external dependencies.
"""

import json
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest


@pytest.fixture
def mock_response():
    """Create a mock API response."""

    def _create_response(
        content: str = "Mock response",
        status_code: int = 200,
        headers: Dict[str, str] = None,
        elapsed_ms: float = 100.0,
    ):
        response = Mock()
        response.content = content
        response.text = content
        response.status_code = status_code
        response.headers = headers or {"content-type": "application/json"}
        response.elapsed.total_seconds.return_value = elapsed_ms / 1000
        response.json.return_value = (
            {"response": content} if status_code == 200 else {"error": content}
        )
        response.raise_for_status = (
            Mock() if status_code == 200 else Mock(side_effect=Exception(f"HTTP {status_code}"))
        )
        return response

    return _create_response


@pytest.fixture
def mock_api_client():
    """Create a mock API client."""
    client = Mock()

    # Configure default behaviors
    client.post.return_value = Mock(
        status_code=200,
        json=Mock(return_value={"result": "success"}),
    )

    client.get.return_value = Mock(
        status_code=200,
        json=Mock(return_value={"data": []}),
    )

    client.headers = {"Authorization": "Bearer mock_token"}
    client.base_url = "https://api.mock.com"

    return client


@pytest.fixture
def mock_logger():
    """Create a mock logger with tracking capabilities."""
    logger = Mock()

    # Track logged messages
    logger.debug_messages = []
    logger.info_messages = []
    logger.warning_messages = []
    logger.error_messages = []
    logger.critical_messages = []

    # Configure logging methods to track messages
    logger.debug.side_effect = lambda msg, *args, **kwargs: logger.debug_messages.append(str(msg))
    logger.info.side_effect = lambda msg, *args, **kwargs: logger.info_messages.append(str(msg))
    logger.warning.side_effect = lambda msg, *args, **kwargs: logger.warning_messages.append(
        str(msg)
    )
    logger.error.side_effect = lambda msg, *args, **kwargs: logger.error_messages.append(str(msg))
    logger.critical.side_effect = lambda msg, *args, **kwargs: logger.critical_messages.append(
        str(msg)
    )

    # Add utility methods
    logger.get_all_messages = lambda: (
        logger.debug_messages
        + logger.info_messages
        + logger.warning_messages
        + logger.error_messages
        + logger.critical_messages
    )

    logger.clear = lambda: (
        logger.debug_messages.clear(),
        logger.info_messages.clear(),
        logger.warning_messages.clear(),
        logger.error_messages.clear(),
        logger.critical_messages.clear(),
    )

    return logger


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = Mock()

    # Default configuration values
    config_data = {
        "api_key": "mock_api_key",
        "model": "mock-model",
        "temperature": 0.7,
        "max_tokens": 1000,
        "timeout": 30,
        "retry_count": 3,
        "log_level": "INFO",
        "cache_enabled": True,
        "database_url": "sqlite:///:memory:",
    }

    # Configure get method
    config.get.side_effect = lambda key, default=None: config_data.get(key, default)

    # Configure attribute access
    for key, value in config_data.items():
        setattr(config, key, value)

    # Add update method
    config.update = lambda updates: config_data.update(updates)

    return config


@pytest.fixture
def mock_metrics():
    """Create a mock metrics collector."""
    metrics = Mock()

    # Storage for metrics
    metrics._counters = {}
    metrics._gauges = {}
    metrics._histograms = {}
    metrics._timers = {}

    # Counter methods
    def increment_counter(name: str, value: int = 1, tags: Dict = None):
        if name not in metrics._counters:
            metrics._counters[name] = 0
        metrics._counters[name] += value

    # Gauge methods
    def set_gauge(name: str, value: float, tags: Dict = None):
        metrics._gauges[name] = value

    # Histogram methods
    def record_histogram(name: str, value: float, tags: Dict = None):
        if name not in metrics._histograms:
            metrics._histograms[name] = []
        metrics._histograms[name].append(value)

    # Timer context manager
    class Timer:
        def __init__(self, name: str):
            self.name = name
            self.start_time = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, *args):
            elapsed = time.time() - self.start_time
            if self.name not in metrics._timers:
                metrics._timers[self.name] = []
            metrics._timers[self.name].append(elapsed)

    metrics.increment = increment_counter
    metrics.gauge = set_gauge
    metrics.histogram = record_histogram
    metrics.timer = lambda name: Timer(name)

    # Utility methods
    metrics.get_counter = lambda name: metrics._counters.get(name, 0)
    metrics.get_gauge = lambda name: metrics._gauges.get(name)
    metrics.get_histogram = lambda name: metrics._histograms.get(name, [])
    metrics.get_timer_stats = lambda name: {
        "count": len(metrics._timers.get(name, [])),
        "mean": sum(metrics._timers.get(name, [])) / len(metrics._timers.get(name, []))
        if metrics._timers.get(name)
        else 0,
        "values": metrics._timers.get(name, []),
    }

    metrics.reset = lambda: (
        metrics._counters.clear(),
        metrics._gauges.clear(),
        metrics._histograms.clear(),
        metrics._timers.clear(),
    )

    return metrics


@pytest.fixture
def mock_database():
    """Create a mock database connection."""
    db = Mock()

    # In-memory storage
    db._data = {}
    db._transactions = []

    # Query execution
    def execute(query: str, params: Dict = None):
        result = Mock()
        result.fetchall.return_value = []
        result.fetchone.return_value = None
        result.rowcount = 0
        db._transactions.append({"query": query, "params": params})
        return result

    # CRUD operations
    def insert(table: str, data: Dict):
        if table not in db._data:
            db._data[table] = []
        db._data[table].append(data)
        return len(db._data[table]) - 1

    def select(table: str, conditions: Dict = None):
        if table not in db._data:
            return []
        if not conditions:
            return db._data[table]
        # Simple filtering
        results = []
        for row in db._data[table]:
            if all(row.get(k) == v for k, v in conditions.items()):
                results.append(row)
        return results

    def update(table: str, data: Dict, conditions: Dict):
        if table not in db._data:
            return 0
        updated = 0
        for row in db._data[table]:
            if all(row.get(k) == v for k, v in conditions.items()):
                row.update(data)
                updated += 1
        return updated

    def delete(table: str, conditions: Dict):
        if table not in db._data:
            return 0
        original_len = len(db._data[table])
        db._data[table] = [
            row
            for row in db._data[table]
            if not all(row.get(k) == v for k, v in conditions.items())
        ]
        return original_len - len(db._data[table])

    db.execute = execute
    db.insert = insert
    db.select = select
    db.update = update
    db.delete = delete

    # Transaction methods
    db.begin = Mock()
    db.commit = Mock()
    db.rollback = Mock()

    # Connection methods
    db.connect = Mock(return_value=True)
    db.disconnect = Mock()
    db.is_connected = Mock(return_value=True)

    return db


@pytest.fixture
def mock_async_client():
    """Create a mock async API client."""
    client = AsyncMock()

    # Configure default async behaviors
    async def mock_post(*args, **kwargs):
        return Mock(
            status_code=200,
            json=Mock(return_value={"result": "async success"}),
        )

    async def mock_get(*args, **kwargs):
        return Mock(
            status_code=200,
            json=Mock(return_value={"data": []}),
        )

    client.post = mock_post
    client.get = mock_get
    client.close = AsyncMock()

    return client


@pytest.fixture
def mock_file_system():
    """Create a mock file system for testing file operations."""
    fs = Mock()

    # Virtual file storage
    fs._files = {}
    fs._directories = {"/": True, "/tmp": True}

    def read_file(path: str) -> str:
        if path not in fs._files:
            raise FileNotFoundError(f"File not found: {path}")
        return fs._files[path]

    def write_file(path: str, content: str):
        # Ensure parent directory exists
        parent = "/".join(path.split("/")[:-1]) or "/"
        if parent not in fs._directories:
            raise FileNotFoundError(f"Directory not found: {parent}")
        fs._files[path] = content

    def delete_file(path: str):
        if path in fs._files:
            del fs._files[path]
        else:
            raise FileNotFoundError(f"File not found: {path}")

    def exists(path: str) -> bool:
        return path in fs._files or path in fs._directories

    def mkdir(path: str):
        fs._directories[path] = True

    def list_dir(path: str) -> List[str]:
        if path not in fs._directories:
            raise FileNotFoundError(f"Directory not found: {path}")

        results = []
        for file_path in fs._files:
            if file_path.startswith(path):
                relative = file_path[len(path) :].lstrip("/")
                if "/" not in relative:
                    results.append(relative)
        return results

    fs.read = read_file
    fs.write = write_file
    fs.delete = delete_file
    fs.exists = exists
    fs.mkdir = mkdir
    fs.list_dir = list_dir

    return fs
