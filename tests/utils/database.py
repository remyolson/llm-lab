"""
Test Database Utilities

Database test utilities for SQLite-based monitoring and results storage with
automatic cleanup, transaction rollback, and pre-populated database states.
"""

from __future__ import annotations

import json
import random
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from .base import DatabaseFixture
from .factories import fake


@dataclass
class DatabaseState:
    """Represents a database state for testing."""

    name: str
    description: str
    tables: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def apply(self, connection: sqlite3.Connection):
        """Apply this state to a database connection."""
        cursor = connection.cursor()

        for table_name, records in self.tables.items():
            if records:
                # Get column names from first record
                columns = list(records[0].keys())
                placeholders = ", ".join(["?" for _ in columns])
                column_names = ", ".join(columns)

                for record in records:
                    values = [record.get(col) for col in columns]
                    cursor.execute(
                        f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})", values
                    )

        connection.commit()


class TestDatabase(DatabaseFixture):
    """Enhanced test database with pre-defined schemas and data states."""

    def __init__(self, db_path: Optional[Path] = None):
        super().__init__(db_path)
        self.transaction_savepoint = None

    def create_schema(self):
        """Create comprehensive test database schema."""
        cursor = self.connection.cursor()

        # Results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                provider TEXT NOT NULL,
                benchmark TEXT,
                score REAL,
                latency REAL,
                tokens_used INTEGER,
                cost REAL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)

        # Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT,
                model TEXT,
                provider TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)

        # Evaluations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id TEXT UNIQUE,
                model TEXT NOT NULL,
                provider TEXT NOT NULL,
                method TEXT,
                score REAL,
                confidence REAL,
                samples INTEGER,
                duration REAL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                config TEXT,
                results TEXT
            )
        """)

        # Prompts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_text TEXT NOT NULL,
                response_text TEXT,
                model TEXT,
                provider TEXT,
                temperature REAL,
                max_tokens INTEGER,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)

        # Cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                ttl INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                accessed_at TEXT,
                hit_count INTEGER DEFAULT 0
            )
        """)

        # Monitoring events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS monitoring_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                level TEXT,
                message TEXT,
                source TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_model ON results(model)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_timestamp ON results(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluations_model ON evaluations(model)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_key ON cache(key)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_monitoring_timestamp ON monitoring_events(timestamp)"
        )

        self.connection.commit()

    def populate_data(self):
        """Populate with default test data."""
        self.populate_results()
        self.populate_metrics()
        self.populate_evaluations()

    def populate_results(self, count: int = 10):
        """Populate results table with test data."""
        cursor = self.connection.cursor()

        models = ["gpt-4", "claude-3", "gemini-pro", "llama-2"]
        providers = ["openai", "anthropic", "google", "local"]
        benchmarks = ["truthfulness", "reasoning", "creativity", "safety"]

        for i in range(count):
            cursor.execute(
                """
                INSERT INTO results (model, provider, benchmark, score, latency, tokens_used, cost, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    random.choice(models),
                    random.choice(providers),
                    random.choice(benchmarks),
                    random.uniform(0.5, 1.0),
                    random.uniform(100, 5000),
                    random.randint(100, 10000),
                    random.uniform(0.01, 1.0),
                    json.dumps({"iteration": i, "test": True}),
                ),
            )

        self.connection.commit()

    def populate_metrics(self, count: int = 20):
        """Populate metrics table with test data."""
        cursor = self.connection.cursor()

        metric_names = ["accuracy", "latency", "cost", "throughput", "error_rate"]
        units = ["percentage", "milliseconds", "dollars", "requests/second", "percentage"]
        models = ["gpt-4", "claude-3", "gemini-pro"]
        providers = ["openai", "anthropic", "google"]

        for i in range(count):
            metric_idx = random.randint(0, len(metric_names) - 1)
            cursor.execute(
                """
                INSERT INTO metrics (name, value, unit, model, provider, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    metric_names[metric_idx],
                    random.uniform(0, 100),
                    units[metric_idx],
                    random.choice(models),
                    random.choice(providers),
                    json.dumps({"test_run": i}),
                ),
            )

        self.connection.commit()

    def populate_evaluations(self, count: int = 5):
        """Populate evaluations table with test data."""
        cursor = self.connection.cursor()

        methods = ["semantic_similarity", "exact_match", "fuzzy_match", "bleu"]
        models = ["gpt-4", "claude-3"]
        providers = ["openai", "anthropic"]

        for i in range(count):
            eval_id = fake.uuid4()
            results = {
                "scores": [random.uniform(0.6, 1.0) for _ in range(10)],
                "average": random.uniform(0.7, 0.95),
            }
            config = {"temperature": 0.7, "max_tokens": 1000, "top_p": 0.95}

            cursor.execute(
                """
                INSERT INTO evaluations
                (evaluation_id, model, provider, method, score, confidence, samples, duration, config, results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    eval_id,
                    random.choice(models),
                    random.choice(providers),
                    random.choice(methods),
                    random.uniform(0.6, 1.0),
                    random.uniform(0.8, 1.0),
                    random.randint(10, 1000),
                    random.uniform(1, 300),
                    json.dumps(config),
                    json.dumps(results),
                ),
            )

        self.connection.commit()

    @contextmanager
    def transaction(self):
        """Context manager for database transactions with automatic rollback."""
        cursor = self.connection.cursor()
        cursor.execute("BEGIN TRANSACTION")

        try:
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    @contextmanager
    def savepoint(self, name: str = "test_savepoint"):
        """Context manager for savepoints within transactions."""
        cursor = self.connection.cursor()
        cursor.execute(f"SAVEPOINT {name}")

        try:
            yield self.connection
            cursor.execute(f"RELEASE SAVEPOINT {name}")
        except Exception:
            cursor.execute(f"ROLLBACK TO SAVEPOINT {name}")
            raise

    def clear_table(self, table_name: str):
        """Clear all data from a specific table."""
        cursor = self.connection.cursor()
        cursor.execute(f"DELETE FROM {table_name}")
        self.connection.commit()

    def clear_all_tables(self):
        """Clear all data from all tables."""
        tables = ["results", "metrics", "evaluations", "prompts", "cache", "monitoring_events"]
        for table in tables:
            self.clear_table(table)

    def get_table_count(self, table_name: str) -> int:
        """Get the number of records in a table."""
        cursor = self.connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cursor.fetchone()[0]

    def apply_state(self, state: DatabaseState):
        """Apply a pre-defined database state."""
        self.clear_all_tables()
        state.apply(self.connection)


def create_test_db(schema_only: bool = False) -> TestDatabase:
    """Create a test database with optional data population."""
    db = TestDatabase()
    db.setup()

    if not schema_only:
        db.populate_data()

    return db


def cleanup_test_db(db: TestDatabase):
    """Clean up a test database."""
    db.teardown()


def populate_test_data(connection: sqlite3.Connection, table: str, records: List[Dict[str, Any]]):
    """Populate a table with test data."""
    if not records:
        return

    cursor = connection.cursor()
    columns = list(records[0].keys())
    placeholders = ", ".join(["?" for _ in columns])
    column_names = ", ".join(columns)

    for record in records:
        values = [record.get(col) for col in columns]
        cursor.execute(f"INSERT INTO {table} ({column_names}) VALUES ({placeholders})", values)

    connection.commit()


def reset_database(connection: sqlite3.Connection):
    """Reset database to clean state."""
    cursor = connection.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    # Clear all tables
    for table in tables:
        cursor.execute(f"DELETE FROM {table[0]}")

    connection.commit()


# Pre-defined database states for testing
class DatabaseStates:
    """Collection of pre-defined database states."""

    @staticmethod
    def empty() -> DatabaseState:
        """Empty database state."""
        return DatabaseState(name="empty", description="Empty database with schema only", tables={})

    @staticmethod
    def with_results() -> DatabaseState:
        """Database with results data."""
        return DatabaseState(
            name="with_results",
            description="Database with sample results",
            tables={
                "results": [
                    {
                        "model": "gpt-4",
                        "provider": "openai",
                        "benchmark": "truthfulness",
                        "score": 0.85,
                        "latency": 1250.5,
                        "tokens_used": 1500,
                        "cost": 0.05,
                    },
                    {
                        "model": "claude-3",
                        "provider": "anthropic",
                        "benchmark": "reasoning",
                        "score": 0.92,
                        "latency": 980.3,
                        "tokens_used": 1200,
                        "cost": 0.04,
                    },
                ]
            },
        )

    @staticmethod
    def with_cache() -> DatabaseState:
        """Database with cache data."""
        return DatabaseState(
            name="with_cache",
            description="Database with cached responses",
            tables={
                "cache": [
                    {
                        "key": "prompt_hash_1",
                        "value": json.dumps({"response": "Cached response 1"}),
                        "ttl": 3600,
                        "hit_count": 5,
                    },
                    {
                        "key": "prompt_hash_2",
                        "value": json.dumps({"response": "Cached response 2"}),
                        "ttl": 7200,
                        "hit_count": 10,
                    },
                ]
            },
        )

    @staticmethod
    def with_monitoring() -> DatabaseState:
        """Database with monitoring events."""
        events = []
        levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        event_types = ["api_call", "cache_hit", "error", "performance"]

        for i in range(20):
            events.append(
                {
                    "event_type": random.choice(event_types),
                    "level": random.choice(levels),
                    "message": f"Test event {i}",
                    "source": random.choice(["provider", "cache", "evaluator"]),
                    "metadata": json.dumps({"index": i}),
                }
            )

        return DatabaseState(
            name="with_monitoring",
            description="Database with monitoring events",
            tables={"monitoring_events": events},
        )

    @staticmethod
    def corrupted() -> DatabaseState:
        """Database with corrupted/invalid data for error testing."""
        return DatabaseState(
            name="corrupted",
            description="Database with invalid data",
            tables={
                "results": [
                    {
                        "model": None,  # Invalid NULL
                        "provider": "invalid_provider",
                        "score": 1.5,  # Invalid score > 1
                        "latency": -100,  # Invalid negative latency
                        "tokens_used": None,
                        "cost": "not_a_number",  # Invalid type
                    }
                ],
                "metrics": [
                    {
                        "name": "",  # Empty name
                        "value": float("inf"),  # Infinity
                        "unit": None,
                        "model": "x" * 1000,  # Very long string
                        "provider": None,
                    }
                ],
            },
        )
