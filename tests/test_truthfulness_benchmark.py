"""Tests for truthfulness benchmark module."""

import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from benchmarks.truthfulness import validate_dataset


class TestValidateDataset:
    """Test validate_dataset function."""

    @patch("os.path.exists")
    def test_validate_dataset_success(self, mock_exists):
        """Test successful dataset validation."""
        mock_exists.return_value = True
        test_data = (
            json.dumps(
                {
                    "id": "truth_001",
                    "prompt": "Test prompt",
                    "evaluation_method": "keyword_match",
                    "expected_keywords": ["test"],
                    "category": "science",
                }
            )
            + "\n"
        )

        with patch("builtins.open", mock_open(read_data=test_data)):
            result = validate_dataset()
            assert result is True

    @patch("os.path.exists")
    def test_validate_dataset_file_not_found(self, mock_exists):
        """Test validation with missing dataset file."""
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError) as exc_info:
            validate_dataset()
        assert "Dataset file not found" in str(exc_info.value)

    @patch("os.path.exists")
    def test_validate_dataset_empty_file(self, mock_exists):
        """Test validation with empty dataset file."""
        mock_exists.return_value = True

        with patch("builtins.open", mock_open(read_data="")):
            # Empty file is actually valid (returns True)
            result = validate_dataset()
            assert result is True

    @patch("os.path.exists")
    def test_validate_dataset_invalid_json(self, mock_exists):
        """Test validation with invalid JSON."""
        mock_exists.return_value = True

        mock_file = MagicMock()
        mock_file.__enter__ = lambda self: self
        mock_file.__exit__ = lambda self, *args: None
        mock_file.__iter__ = lambda self: iter(["invalid json\n"])

        with patch("builtins.open", return_value=mock_file):
            with pytest.raises(json.JSONDecodeError) as exc_info:
                validate_dataset()
            assert "Invalid JSON on line 1" in str(exc_info.value)

    @patch("os.path.exists")
    def test_validate_dataset_missing_required_fields(self, mock_exists):
        """Test validation with missing required fields."""
        mock_exists.return_value = True
        test_data = json.dumps({"prompt": "Test"}) + "\n"  # Missing id, evaluation_method

        with patch("builtins.open", mock_open(read_data=test_data)):
            with pytest.raises(ValueError) as exc_info:
                validate_dataset()
            assert "Missing required fields" in str(exc_info.value)

    @patch("os.path.exists")
    def test_validate_dataset_invalid_expected_keywords_type(self, mock_exists):
        """Test validation with invalid expected_keywords type."""
        mock_exists.return_value = True
        test_data = (
            json.dumps(
                {
                    "id": "123",
                    "prompt": "Test",
                    "evaluation_method": "keyword_match",
                    "expected_keywords": "not a list",  # Should be list
                }
            )
            + "\n"
        )

        with patch("builtins.open", mock_open(read_data=test_data)):
            with pytest.raises(ValueError) as exc_info:
                validate_dataset()
            assert "'expected_keywords' must be a list" in str(exc_info.value)

    @patch("os.path.exists")
    def test_validate_dataset_empty_expected_keywords(self, mock_exists):
        """Test validation with empty expected_keywords."""
        mock_exists.return_value = True
        test_data = (
            json.dumps(
                {
                    "id": "test",
                    "prompt": "Test",
                    "evaluation_method": "keyword_match",
                    "expected_keywords": [],  # Empty list
                }
            )
            + "\n"
        )

        with patch("builtins.open", mock_open(read_data=test_data)):
            with pytest.raises(ValueError) as exc_info:
                validate_dataset()
            assert "'expected_keywords' cannot be empty" in str(exc_info.value)

    @patch("os.path.exists")
    def test_validate_dataset_keyword_match_missing_keywords(self, mock_exists):
        """Test validation of keyword_match without expected_keywords."""
        mock_exists.return_value = True
        test_data = (
            json.dumps(
                {
                    "id": "test",
                    "prompt": "Test",
                    "evaluation_method": "keyword_match",
                    # Missing expected_keywords
                }
            )
            + "\n"
        )

        with patch("builtins.open", mock_open(read_data=test_data)):
            with pytest.raises(ValueError) as exc_info:
                validate_dataset()
            assert "Missing required fields" in str(exc_info.value)
            assert "expected_keywords" in str(exc_info.value)

    @patch("os.path.exists")
    def test_validate_dataset_skips_empty_lines(self, mock_exists):
        """Test that validation skips empty lines."""
        mock_exists.return_value = True
        lines = [
            "",  # Empty line
            json.dumps(
                {
                    "id": "test1",
                    "prompt": "Test",
                    "evaluation_method": "keyword_match",
                    "expected_keywords": ["test"],
                }
            ),
            "   ",  # Whitespace only
            json.dumps(
                {
                    "id": "test2",
                    "prompt": "Test 2",
                    "evaluation_method": "keyword_match",
                    "expected_keywords": ["test2"],
                }
            ),
        ]

        mock_file = MagicMock()
        mock_file.__enter__ = lambda self: self
        mock_file.__exit__ = lambda self, *args: None
        mock_file.__iter__ = lambda self: iter(lines)

        with patch("builtins.open", return_value=mock_file):
            result = validate_dataset()
            assert result is True

    @patch("os.path.exists")
    def test_validate_dataset_multiple_errors(self, mock_exists):
        """Test validation with multiple entries, some invalid."""
        mock_exists.return_value = True
        lines = [
            json.dumps(
                {
                    "id": "test1",
                    "prompt": "Valid entry",
                    "evaluation_method": "keyword_match",
                    "expected_keywords": ["test"],
                }
            ),
            json.dumps(
                {
                    "id": "test2",
                    "prompt": "Invalid - missing keywords",
                    # Missing evaluation_method and expected_keywords
                }
            ),
        ]

        mock_file = MagicMock()
        mock_file.__enter__ = lambda self: self
        mock_file.__exit__ = lambda self, *args: None
        mock_file.__iter__ = lambda self: iter(lines)

        with patch("builtins.open", return_value=mock_file):
            # Should fail on the second entry
            with pytest.raises(ValueError) as exc_info:
                validate_dataset()
            assert "Missing required fields on line 2" in str(exc_info.value)
