"""
Simple test to verify basic testing infrastructure works.
"""

import pytest


def test_basic_function():
    """Test that basic testing works."""
    assert 2 + 2 == 4


def test_string_operations():
    """Test string operations."""
    test_string = "hello world"
    assert test_string.upper() == "HELLO WORLD"
    assert "world" in test_string


class TestBasicClass:
    """Basic test class."""

    def test_list_operations(self):
        """Test list operations."""
        test_list = [1, 2, 3, 4, 5]
        assert len(test_list) == 5
        assert max(test_list) == 5
        assert min(test_list) == 1

    def test_dict_operations(self):
        """Test dictionary operations."""
        test_dict = {"a": 1, "b": 2, "c": 3}
        assert test_dict["a"] == 1
        assert "b" in test_dict
        assert len(test_dict) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
