"""Tests for results_logger module."""

import csv
import json
import os
import tempfile
from datetime import datetime
from logging.results_logger import (
    CSVResultLogger,
    ResultRecord,
    clean_multiline_text,
    format_timestamp,
    truncate_text,
)
from pathlib import Path

import pytest


class TestHelperFunctions:
    """Test helper functions."""

    def test_truncate_text_short(self):
        """Test truncation with short text."""
        text = "Short text"
        assert truncate_text(text, 20) == text

    def test_truncate_text_long(self):
        """Test truncation with long text."""
        text = "A" * 100
        result = truncate_text(text, 10)
        assert len(result) == 10
        assert result.endswith("...")
        assert result == "AAAAAAA..."

    def test_clean_multiline_text(self):
        """Test multiline text cleaning."""
        text = "Line 1\nLine 2\r\nLine 3\r\nWith  multiple   spaces"
        result = clean_multiline_text(text)
        assert "\n" not in result
        assert "\r" not in result
        assert "  " not in result
        assert result == "Line 1 Line 2 Line 3 With multiple spaces"

    def test_clean_multiline_text_empty(self):
        """Test cleaning empty text."""
        assert clean_multiline_text("") == ""
        assert clean_multiline_text(None) == ""

    def test_format_timestamp_datetime(self):
        """Test formatting datetime object."""
        dt = datetime(2024, 1, 15, 10, 30, 45)
        result = format_timestamp(dt)
        assert result == "2024-01-15T10:30:45"

    def test_format_timestamp_string(self):
        """Test formatting string timestamp."""
        timestamp = "2024-01-15T10:30:45"
        result = format_timestamp(timestamp)
        assert result == timestamp

    def test_format_timestamp_none(self):
        """Test formatting with None (current time)."""
        result = format_timestamp(None)
        # Should be a valid ISO format timestamp
        datetime.fromisoformat(result)  # Will raise if invalid


class TestResultRecord:
    """Test ResultRecord class."""

    def test_result_record_full_data(self):
        """Test creating ResultRecord with full data."""
        eval_data = {
            "timestamp": "2024-01-15T10:30:45",
            "model_name": "test-model",
            "benchmark_name": "test-benchmark",
            "prompt_id": "test-001",
            "prompt": "Test prompt",
            "response": "Test response",
            "expected_keywords": ["keyword1", "keyword2"],
            "matched_keywords": ["keyword1"],
            "score": 0.5,
            "success": True,
            "evaluation_method": "keyword_match",
            "response_time_seconds": 1.234,
        }

        record = ResultRecord(eval_data)
        csv_dict = record.to_csv_dict()

        assert csv_dict["timestamp"] == eval_data["timestamp"]
        assert csv_dict["model_name"] == "test-model"
        assert csv_dict["score"] == "0.5000"
        assert csv_dict["success"] == "pass"
        assert csv_dict["response_time_seconds"] == "1.234"
        assert json.loads(csv_dict["expected_keywords"]) == ["keyword1", "keyword2"]
        assert json.loads(csv_dict["matched_keywords"]) == ["keyword1"]

    def test_result_record_minimal_data(self):
        """Test creating ResultRecord with minimal data."""
        eval_data = {"prompt": "Test prompt", "response": "Test response"}

        record = ResultRecord(eval_data)
        csv_dict = record.to_csv_dict()

        assert csv_dict["score"] == "0.0000"
        assert csv_dict["success"] == "fail"
        assert csv_dict["expected_keywords"] == ""
        assert csv_dict["matched_keywords"] == ""
        assert csv_dict["error"] == ""


class TestCSVResultLogger:
    """Test CSVResultLogger class."""

    def test_init_creates_directory(self):
        """Test that initialization creates output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_results"
            assert not output_dir.exists()

            logger = CSVResultLogger(str(output_dir))
            assert output_dir.exists()

    def test_generate_filename(self):
        """Test filename generation."""
        logger = CSVResultLogger()
        filename = logger.generate_filename("google", "truthfulness")

        assert filename.startswith("benchmark_google_truthfulness_")
        assert filename.endswith(".csv")
        # Check timestamp format (YYYYMMDD_HHMMSS)
        parts = filename.split("_")
        assert len(parts) == 5  # benchmark, google, truthfulness, YYYYMMDD, HHMMSS.csv
        assert parts[-1].endswith(".csv")

    def test_write_results_success(self):
        """Test successful results writing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVResultLogger(tmpdir)

            results = {
                "provider": "test-provider",
                "dataset": "test-dataset",
                "evaluations": [
                    {
                        "timestamp": "2024-01-15T10:30:45",
                        "model_name": "test-model",
                        "prompt": "Test prompt 1",
                        "response": "Test response 1",
                        "score": 1.0,
                        "success": True,
                    },
                    {
                        "timestamp": "2024-01-15T10:30:46",
                        "model_name": "test-model",
                        "prompt": "Test prompt 2",
                        "response": "Test response 2",
                        "score": 0.0,
                        "success": False,
                    },
                ],
            }

            csv_path = logger.write_results(results)
            assert os.path.exists(csv_path)

            # Verify content
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert rows[0]["prompt_text"] == "Test prompt 1"
            assert rows[0]["success"] == "pass"
            assert rows[1]["prompt_text"] == "Test prompt 2"
            assert rows[1]["success"] == "fail"

    def test_write_results_no_evaluations(self):
        """Test writing with no evaluations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVResultLogger(tmpdir)

            results = {"provider": "test", "dataset": "test", "evaluations": []}

            with pytest.raises(ValueError) as exc_info:
                logger.write_results(results)
            assert "No evaluation results" in str(exc_info.value)

    def test_append_result(self):
        """Test appending single result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVResultLogger(tmpdir)
            csv_path = Path(tmpdir) / "test_append.csv"

            # First append (should create file with header)
            eval_data1 = {"prompt": "Test 1", "response": "Response 1", "score": 1.0}
            logger.append_result(csv_path, eval_data1)

            # Second append (should not add header)
            eval_data2 = {"prompt": "Test 2", "response": "Response 2", "score": 0.5}
            logger.append_result(csv_path, eval_data2)

            # Verify content
            with open(csv_path, newline="") as f:
                lines = f.readlines()

            # Should have header + 2 data rows
            assert len(lines) == 3
            assert lines[0].startswith("timestamp,")

    def test_context_manager(self):
        """Test context manager protocol."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with CSVResultLogger(tmpdir) as logger:
                assert isinstance(logger, CSVResultLogger)
                # Should be able to use logger here
                filename = logger.generate_filename("test", "test")
                assert filename
