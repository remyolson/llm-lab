"""
Unit tests for utility functions and helpers.
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest


@pytest.mark.unit
class TestDataValidation:
    """Test data validation utilities."""

    def test_prompt_validation(self):
        """Test prompt text validation."""

        def validate_prompt(prompt):
            if not isinstance(prompt, str):
                raise ValueError("Prompt must be a string")

            if not prompt.strip():
                raise ValueError("Prompt cannot be empty")

            if len(prompt) > 10000:
                raise ValueError("Prompt too long (max 10000 characters)")

            # Check for potentially harmful content
            harmful_patterns = ["<script>", "javascript:", "eval("]
            for pattern in harmful_patterns:
                if pattern.lower() in prompt.lower():
                    raise ValueError(f"Potentially harmful content detected: {pattern}")

            return prompt.strip()

        # Valid prompts
        assert validate_prompt("Hello world") == "Hello world"
        assert validate_prompt("  What is AI?  ") == "What is AI?"

        # Invalid prompts
        with pytest.raises(ValueError, match="must be a string"):
            validate_prompt(123)

        with pytest.raises(ValueError, match="cannot be empty"):
            validate_prompt("")

        with pytest.raises(ValueError, match="cannot be empty"):
            validate_prompt("   ")

        with pytest.raises(ValueError, match="too long"):
            validate_prompt("x" * 10001)

        with pytest.raises(ValueError, match="harmful content"):
            validate_prompt("Click here: <script>alert('xss')</script>")

    def test_response_validation(self):
        """Test response validation and sanitization."""

        def validate_response(response):
            if not isinstance(response, str):
                raise ValueError("Response must be a string")

            # Basic sanitization
            response = response.strip()

            # Check for minimum content
            if len(response) < 1:
                raise ValueError("Response too short")

            # Check for reasonable length
            if len(response) > 50000:
                raise ValueError("Response too long")

            return response

        # Valid responses
        assert validate_response("This is a good response.") == "This is a good response."
        assert validate_response("  Response with whitespace  ") == "Response with whitespace"

        # Invalid responses
        with pytest.raises(ValueError, match="must be a string"):
            validate_response(None)

        with pytest.raises(ValueError, match="too short"):
            validate_response("")

        with pytest.raises(ValueError, match="too long"):
            validate_response("x" * 50001)

    def test_json_validation(self):
        """Test JSON data validation."""

        def validate_json_structure(data, required_fields=None, optional_fields=None):
            required_fields = required_fields or []
            optional_fields = optional_fields or []

            if not isinstance(data, dict):
                raise ValueError("Data must be a dictionary")

            # Check required fields
            missing_fields = []
            for field in required_fields:
                if field not in data:
                    missing_fields.append(field)

            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")

            # Check for unknown fields
            allowed_fields = set(required_fields + optional_fields)
            unknown_fields = set(data.keys()) - allowed_fields

            if unknown_fields:
                raise ValueError(f"Unknown fields: {list(unknown_fields)}")

            return data

        # Valid data
        data = {"id": "123", "prompt": "test", "response": "result"}
        validated = validate_json_structure(
            data, required_fields=["id", "prompt"], optional_fields=["response", "metadata"]
        )
        assert validated == data

        # Missing required field
        with pytest.raises(ValueError, match="Missing required fields"):
            validate_json_structure({"prompt": "test"}, required_fields=["id", "prompt"])

        # Unknown field
        with pytest.raises(ValueError, match="Unknown fields"):
            validate_json_structure({"id": "123", "unknown": "value"}, required_fields=["id"])


@pytest.mark.unit
class TestStringUtilities:
    """Test string manipulation utilities."""

    def test_text_truncation(self):
        """Test text truncation with ellipsis."""

        def truncate_text(text, max_length=100, suffix="..."):
            if len(text) <= max_length:
                return text

            return text[: max_length - len(suffix)] + suffix

        # No truncation needed
        short_text = "This is short"
        assert truncate_text(short_text, 100) == short_text

        # Truncation needed
        long_text = "This is a very long text that needs to be truncated"
        truncated = truncate_text(long_text, 20)
        assert len(truncated) == 20
        assert truncated.endswith("...")
        assert truncated == "This is a very lo..."

    def test_text_sanitization(self):
        """Test text sanitization for logs and display."""

        def sanitize_text(text, max_length=1000):
            # Remove control characters
            import re

            text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)

            # Normalize whitespace
            text = re.sub(r"\s+", " ", text)

            # Truncate if too long
            if len(text) > max_length:
                text = text[: max_length - 3] + "..."

            return text.strip()

        # Test control character removal
        dirty_text = "Hello\x00\x1f\x7fWorld"
        clean_text = sanitize_text(dirty_text)
        assert clean_text == "HelloWorld"

        # Test whitespace normalization
        messy_text = "Too   much\n\n\twhitespace"
        normalized = sanitize_text(messy_text)
        assert normalized == "Too muchwhitespace"

        # Test truncation
        long_text = "x" * 1500
        truncated = sanitize_text(long_text, max_length=1000)
        assert len(truncated) == 1000
        assert truncated.endswith("...")

    def test_template_substitution(self):
        """Test simple template variable substitution."""

        def substitute_variables(template, variables):
            import re

            def replace_var(match):
                var_name = match.group(1)
                return str(variables.get(var_name, match.group(0)))

            return re.sub(r"\${(\w+)}", replace_var, template)

        template = "Hello ${name}, you have ${count} messages"
        variables = {"name": "Alice", "count": 5}

        result = substitute_variables(template, variables)
        assert result == "Hello Alice, you have 5 messages"

        # Missing variable
        incomplete_vars = {"name": "Bob"}
        result = substitute_variables(template, incomplete_vars)
        assert result == "Hello Bob, you have ${count} messages"

    def test_slug_generation(self):
        """Test URL slug generation from text."""

        def generate_slug(text, max_length=50):
            import re

            # Convert to lowercase
            slug = text.lower()

            # Replace spaces and special chars with hyphens
            slug = re.sub(r"[^\w\s-]", "", slug)
            slug = re.sub(r"[\s_-]+", "-", slug)

            # Remove leading/trailing hyphens
            slug = slug.strip("-")

            # Truncate if too long
            if len(slug) > max_length:
                slug = slug[:max_length].rstrip("-")

            return slug

        assert generate_slug("Hello World!") == "hello-world"
        assert generate_slug("Test_with-various chars123") == "test-with-various-chars123"
        assert generate_slug("Multiple   spaces") == "multiple-spaces"
        assert generate_slug("x" * 100, max_length=20) == "x" * 20


@pytest.mark.unit
class TestFileUtilities:
    """Test file system utilities."""

    def test_safe_filename_generation(self):
        """Test generation of safe filenames."""

        def make_safe_filename(filename, max_length=255):
            import re

            # Remove/replace unsafe characters
            safe_chars = r"[^\w\s.-]"
            filename = re.sub(safe_chars, "_", filename)

            # Replace spaces with underscores
            filename = re.sub(r"\s+", "_", filename)

            # Truncate if too long
            if len(filename) > max_length:
                name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
                max_name_length = max_length - len(ext) - 1 if ext else max_length
                filename = name[:max_name_length] + ("." + ext if ext else "")

            return filename

        assert make_safe_filename("normal_file.txt") == "normal_file.txt"
        assert make_safe_filename("file with spaces.txt") == "file_with_spaces.txt"
        assert make_safe_filename("file/with\\bad:chars.txt") == "file_with_bad_chars.txt"

        # Test truncation
        long_name = "x" * 300 + ".txt"
        safe_name = make_safe_filename(long_name, max_length=255)
        assert len(safe_name) <= 255
        assert safe_name.endswith(".txt")

    def test_file_size_formatting(self):
        """Test human-readable file size formatting."""

        def format_file_size(size_bytes):
            if size_bytes == 0:
                return "0 B"

            size_names = ["B", "KB", "MB", "GB", "TB"]
            import math

            i = int(math.floor(math.log(size_bytes, 1024)))
            p = math.pow(1024, i)
            s = round(size_bytes / p, 2)

            return f"{s} {size_names[i]}"

        assert format_file_size(0) == "0 B"
        assert format_file_size(512) == "512.0 B"
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1536) == "1.5 KB"
        assert format_file_size(1048576) == "1.0 MB"
        assert format_file_size(1073741824) == "1.0 GB"

    def test_directory_creation(self, temp_dir):
        """Test safe directory creation."""

        def ensure_directory(path):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            return path

        # Create nested directory
        test_dir = temp_dir / "level1" / "level2" / "level3"
        created_dir = ensure_directory(test_dir)

        assert created_dir.exists()
        assert created_dir.is_dir()

        # Creating again should not raise error
        ensure_directory(test_dir)
        assert created_dir.exists()


@pytest.mark.unit
class TestDataStructureUtilities:
    """Test data structure manipulation utilities."""

    def test_deep_merge(self):
        """Test deep merging of dictionaries."""

        def deep_merge(dict1, dict2):
            result = dict1.copy()

            for key, value in dict2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value

            return result

        dict1 = {"a": 1, "b": {"x": 10, "y": 20}, "c": [1, 2, 3]}

        dict2 = {
            "a": 2,  # Override
            "b": {"y": 30, "z": 40},  # Merge nested
            "d": 4,  # New key
        }

        merged = deep_merge(dict1, dict2)

        assert merged["a"] == 2  # Overridden
        assert merged["b"] == {"x": 10, "y": 30, "z": 40}  # Merged
        assert merged["c"] == [1, 2, 3]  # Unchanged
        assert merged["d"] == 4  # Added

    def test_flatten_dict(self):
        """Test flattening nested dictionaries."""

        def flatten_dict(d, parent_key="", sep="."):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k

                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))

            return dict(items)

        nested = {"level1": {"level2": {"value": 42}, "other": "test"}, "simple": "value"}

        flattened = flatten_dict(nested)

        assert flattened == {"level1.level2.value": 42, "level1.other": "test", "simple": "value"}

    def test_list_chunking(self):
        """Test splitting lists into chunks."""

        def chunk_list(lst, chunk_size):
            for i in range(0, len(lst), chunk_size):
                yield lst[i : i + chunk_size]

        data = list(range(10))  # [0, 1, 2, ..., 9]
        chunks = list(chunk_list(data, 3))

        assert chunks == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

        # Test with exact division
        chunks = list(chunk_list(data, 5))
        assert chunks == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

        # Test with chunk size larger than list
        chunks = list(chunk_list([1, 2], 5))
        assert chunks == [[1, 2]]

    def test_list_deduplication(self):
        """Test removing duplicates while preserving order."""

        def deduplicate_list(lst):
            seen = set()
            result = []

            for item in lst:
                if item not in seen:
                    seen.add(item)
                    result.append(item)

            return result

        data = [1, 2, 3, 2, 4, 1, 5]
        deduped = deduplicate_list(data)

        assert deduped == [1, 2, 3, 4, 5]

        # Test with strings
        strings = ["a", "b", "a", "c", "b"]
        deduped_strings = deduplicate_list(strings)
        assert deduped_strings == ["a", "b", "c"]


@pytest.mark.unit
class TestTimeUtilities:
    """Test time and date utilities."""

    def test_duration_formatting(self):
        """Test formatting durations in human-readable format."""

        def format_duration(seconds):
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                minutes = seconds // 60
                secs = seconds % 60
                return f"{int(minutes)}m {secs:.1f}s"
            else:
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                secs = seconds % 60
                return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"

        assert format_duration(5.5) == "5.5s"
        assert format_duration(65) == "1m 5.0s"
        assert format_duration(3665) == "1h 1m 5.0s"
        assert format_duration(7200) == "2h 0m 0.0s"

    def test_timestamp_generation(self):
        """Test timestamp generation and formatting."""

        def generate_timestamp(format_type="iso"):
            import datetime

            now = datetime.datetime.now(datetime.timezone.utc)

            if format_type == "iso":
                return now.isoformat()
            elif format_type == "filename":
                return now.strftime("%Y%m%d_%H%M%S")
            elif format_type == "human":
                return now.strftime("%Y-%m-%d %H:%M:%S UTC")
            else:
                return str(int(now.timestamp()))

        iso_timestamp = generate_timestamp("iso")
        assert "T" in iso_timestamp
        assert iso_timestamp.endswith("+00:00") or iso_timestamp.endswith("Z")

        filename_timestamp = generate_timestamp("filename")
        assert len(filename_timestamp) == 15  # YYYYMMDD_HHMMSS
        assert "_" in filename_timestamp

        human_timestamp = generate_timestamp("human")
        assert "UTC" in human_timestamp

    def test_timeout_checker(self):
        """Test timeout checking utility."""

        class TimeoutChecker:
            def __init__(self, timeout_seconds):
                self.timeout = timeout_seconds
                self.start_time = time.time()

            def is_timeout(self):
                return time.time() - self.start_time > self.timeout

            def remaining_time(self):
                elapsed = time.time() - self.start_time
                return max(0, self.timeout - elapsed)

        # Test immediate check
        checker = TimeoutChecker(1.0)
        assert not checker.is_timeout()
        assert checker.remaining_time() > 0.9

        # Test after delay
        time.sleep(0.1)
        assert not checker.is_timeout()
        assert checker.remaining_time() < 1.0


@pytest.mark.unit
class TestCryptoUtilities:
    """Test cryptographic and hashing utilities."""

    def test_hash_generation(self):
        """Test generating hashes for data integrity."""
        import hashlib

        def generate_hash(data, algorithm="sha256"):
            if isinstance(data, str):
                data = data.encode("utf-8")

            hash_obj = hashlib.new(algorithm)
            hash_obj.update(data)
            return hash_obj.hexdigest()

        # Test with string
        text = "Hello, World!"
        hash1 = generate_hash(text)
        hash2 = generate_hash(text)

        assert hash1 == hash2  # Same input, same hash
        assert len(hash1) == 64  # SHA256 produces 64 hex chars

        # Test with different input
        hash3 = generate_hash("Different text")
        assert hash1 != hash3

        # Test with bytes
        hash4 = generate_hash(b"Hello, World!")
        assert hash1 == hash4  # Same result for string vs bytes

    def test_uuid_generation(self):
        """Test UUID generation for unique identifiers."""
        import uuid

        def generate_id(id_type="uuid4"):
            if id_type == "uuid4":
                return str(uuid.uuid4())
            elif id_type == "short":
                return str(uuid.uuid4())[:8]
            elif id_type == "timestamp":
                return f"{int(time.time() * 1000)}"
            else:
                return str(uuid.uuid4())

        # Test UUID4
        id1 = generate_id("uuid4")
        id2 = generate_id("uuid4")

        assert id1 != id2  # Should be unique
        assert len(id1) == 36  # Standard UUID format
        assert id1.count("-") == 4  # Standard UUID has 4 hyphens

        # Test short ID
        short_id = generate_id("short")
        assert len(short_id) == 8

        # Test timestamp ID
        timestamp_id = generate_id("timestamp")
        assert timestamp_id.isdigit()
        assert len(timestamp_id) >= 13  # Millisecond timestamp


@pytest.mark.integration
class TestUtilityIntegration:
    """Test utility functions working together."""

    def test_data_processing_pipeline(self, temp_dir):
        """Test a complete data processing pipeline using utilities."""
        # Create test data
        test_data = [
            {"id": "1", "text": "  Hello World!  ", "score": 0.95},
            {"id": "2", "text": "Another text sample", "score": 0.87},
            {"id": "1", "text": "Duplicate ID", "score": 0.92},  # Duplicate
        ]

        # Processing pipeline
        def process_data(data):
            # 1. Deduplicate by ID (keep first occurrence)
            seen_ids = set()
            deduped = []
            for item in data:
                if item["id"] not in seen_ids:
                    seen_ids.add(item["id"])
                    deduped.append(item)

            # 2. Clean text
            for item in deduped:
                item["text"] = item["text"].strip()
                item["text_hash"] = hashlib.sha256(item["text"].encode()).hexdigest()[:8]

            # 3. Add metadata
            for item in deduped:
                item["processed_at"] = time.time()
                item["text_length"] = len(item["text"])

            return deduped

        import hashlib

        processed = process_data(test_data)

        # Verify processing
        assert len(processed) == 2  # Deduplication worked
        assert processed[0]["text"] == "Hello World!"  # Text cleaned
        assert "text_hash" in processed[0]  # Hash added
        assert "processed_at" in processed[0]  # Timestamp added
        assert processed[0]["text_length"] == 12  # Length calculated

    def test_file_processing_workflow(self, temp_dir):
        """Test file processing workflow with utilities."""
        # Create test files
        test_files = []
        for i in range(3):
            file_path = temp_dir / f"test_file_{i}.txt"
            content = f"This is test file {i}\nWith some content"

            with open(file_path, "w") as f:
                f.write(content)

            test_files.append(file_path)

        # Process files
        import hashlib

        results = []
        for file_path in test_files:
            with open(file_path, "r") as f:
                content = f.read()

            result = {
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "lines": len(content.splitlines()),
                "checksum": hashlib.sha256(content.encode()).hexdigest()[:16],
            }
            results.append(result)

        # Verify results
        assert len(results) == 3
        assert all("filename" in r for r in results)
        assert all("checksum" in r for r in results)
        assert all(r["lines"] == 2 for r in results)  # Each file has 2 lines
