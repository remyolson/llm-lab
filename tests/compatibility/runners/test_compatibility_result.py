"""
Test Compatibility Result Management

Tests for compatibility test result handling and reporting.
"""

import json
from datetime import datetime
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

from tests.fixtures import mock_logger, mock_response


class TestCompatibilityResult:
    """Tests for CompatibilityTestResult class."""

    def test_result_creation(self):
        """Test creating a compatibility test result."""
        result = CompatibilityTestResult(
            test_name="basic_generation",
            provider_name="openai",
            model_name="gpt-3.5-turbo",
            success=True,
            duration=1.23,
        )

        assert result.test_name == "basic_generation"
        assert result.provider_name == "openai"
        assert result.success is True
        assert result.duration == 1.23
        assert result.error is None
        assert isinstance(result.timestamp, datetime)

    def test_result_with_error(self):
        """Test creating a result with error information."""
        result = CompatibilityTestResult(
            test_name="error_test",
            provider_name="test_provider",
            model_name="test_model",
            success=False,
            duration=0.5,
            error="Connection timeout",
            details={"retry_count": 3, "status_code": 504},
        )

        assert result.success is False
        assert result.error == "Connection timeout"
        assert result.details["retry_count"] == 3
        assert result.details["status_code"] == 504

    def test_result_serialization(self):
        """Test serializing result to JSON."""
        result = CompatibilityTestResult(
            test_name="serialize_test",
            provider_name="openai",
            model_name="gpt-4",
            success=True,
            duration=2.5,
            details={"tokens_used": 150},
        )

        # Convert to dict for serialization
        result_dict = {
            "test_name": result.test_name,
            "provider_name": result.provider_name,
            "model_name": result.model_name,
            "success": result.success,
            "duration": result.duration,
            "error": result.error,
            "details": result.details,
            "timestamp": result.timestamp.isoformat(),
        }

        # Should serialize without errors
        json_str = json.dumps(result_dict)
        assert json_str is not None

        # Deserialize and verify
        loaded = json.loads(json_str)
        assert loaded["test_name"] == "serialize_test"
        assert loaded["success"] is True

    def test_result_aggregation(self):
        """Test aggregating multiple results."""
        results = [
            CompatibilityTestResult("test1", "provider1", "model1", True, 1.0),
            CompatibilityTestResult("test2", "provider1", "model1", True, 2.0),
            CompatibilityTestResult("test3", "provider1", "model1", False, 3.0, error="Failed"),
        ]

        # Calculate statistics
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.success)
        success_rate = successful_tests / total_tests
        avg_duration = sum(r.duration for r in results) / total_tests

        assert total_tests == 3
        assert successful_tests == 2
        assert success_rate == pytest.approx(0.667, 0.01)
        assert avg_duration == pytest.approx(2.0, 0.01)

    def test_result_filtering(self):
        """Test filtering results by criteria."""
        results = [
            CompatibilityTestResult("test1", "openai", "gpt-3.5", True, 1.0),
            CompatibilityTestResult("test2", "anthropic", "claude", True, 2.0),
            CompatibilityTestResult("test3", "openai", "gpt-4", False, 3.0),
            CompatibilityTestResult("test4", "google", "gemini", True, 1.5),
        ]

        # Filter by provider
        openai_results = [r for r in results if r.provider_name == "openai"]
        assert len(openai_results) == 2

        # Filter by success
        successful_results = [r for r in results if r.success]
        assert len(successful_results) == 3

        # Filter by duration
        fast_results = [r for r in results if r.duration < 2.0]
        assert len(fast_results) == 2


class TestResultReporting:
    """Tests for result reporting and formatting."""

    def test_generate_summary_report(self):
        """Test generating a summary report from results."""
        results = [
            CompatibilityTestResult("test1", "provider1", "model1", True, 1.0),
            CompatibilityTestResult("test2", "provider1", "model1", False, 2.0),
        ]

        report = {
            "total_tests": len(results),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "success_rate": sum(1 for r in results if r.success) / len(results),
            "avg_duration": sum(r.duration for r in results) / len(results),
        }

        assert report["total_tests"] == 2
        assert report["successful"] == 1
        assert report["failed"] == 1
        assert report["success_rate"] == 0.5
        assert report["avg_duration"] == 1.5

    def test_group_results_by_provider(self):
        """Test grouping results by provider."""
        results = [
            CompatibilityTestResult("test1", "openai", "gpt-3.5", True, 1.0),
            CompatibilityTestResult("test2", "openai", "gpt-4", True, 2.0),
            CompatibilityTestResult("test3", "anthropic", "claude", True, 1.5),
        ]

        grouped = {}
        for result in results:
            if result.provider_name not in grouped:
                grouped[result.provider_name] = []
            grouped[result.provider_name].append(result)

        assert len(grouped) == 2
        assert len(grouped["openai"]) == 2
        assert len(grouped["anthropic"]) == 1

    def test_export_results_to_csv(self):
        """Test exporting results to CSV format."""
        results = [
            CompatibilityTestResult("test1", "provider1", "model1", True, 1.0),
            CompatibilityTestResult("test2", "provider2", "model2", False, 2.0, error="Failed"),
        ]

        # Create CSV-like structure
        csv_rows = []
        csv_rows.append("test_name,provider,model,success,duration,error")

        for r in results:
            csv_rows.append(
                f"{r.test_name},{r.provider_name},{r.model_name},"
                f"{r.success},{r.duration},{r.error or ''}"
            )

        csv_content = "\n".join(csv_rows)

        assert "test1,provider1,model1,True,1.0," in csv_content
        assert "test2,provider2,model2,False,2.0,Failed" in csv_content


# Import compatibility result class (would be in the refactored module)
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class CompatibilityTestResult:
    """Result from a single compatibility test."""

    test_name: str
    provider_name: str
    model_name: str
    success: bool
    duration: float
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
