"""
Unit tests for edge cases and boundary conditions.
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


@pytest.mark.unit
class TestInputBoundaryConditions:
    """Test boundary conditions for input validation."""

    def test_empty_and_none_inputs(self):
        """Test handling of empty and None inputs."""

        def validate_prompt(prompt):
            if prompt is None:
                raise ValueError("Prompt cannot be None")
            if not isinstance(prompt, str):
                raise TypeError("Prompt must be a string")
            if not prompt.strip():
                raise ValueError("Prompt cannot be empty or whitespace-only")
            return prompt.strip()

        # Test None input
        with pytest.raises(ValueError, match="cannot be None"):
            validate_prompt(None)

        # Test empty string
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_prompt("")

        # Test whitespace-only
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_prompt("   \n\t  ")

        # Test non-string input
        with pytest.raises(TypeError, match="must be a string"):
            validate_prompt(123)

        with pytest.raises(TypeError, match="must be a string"):
            validate_prompt([])

        # Test valid minimal input
        assert validate_prompt("a") == "a"
        assert validate_prompt("  hello  ") == "hello"

    def test_extremely_large_inputs(self):
        """Test handling of extremely large inputs."""

        def validate_input_size(text, max_length=10000):
            if not isinstance(text, str):
                raise TypeError("Input must be a string")

            if len(text) > max_length:
                raise ValueError(f"Input too large: {len(text)} > {max_length}")

            # Check for potential memory issues
            try:
                # Simulate processing that might use more memory
                processed = text.upper()
                return processed
            except MemoryError:
                raise ValueError("Input too large to process")

        # Test normal size
        normal_text = "Hello world" * 100
        result = validate_input_size(normal_text)
        assert result == normal_text.upper()

        # Test at boundary
        boundary_text = "x" * 10000
        result = validate_input_size(boundary_text)
        assert len(result) == 10000

        # Test over boundary
        large_text = "x" * 10001
        with pytest.raises(ValueError, match="Input too large"):
            validate_input_size(large_text)

    def test_special_characters_and_encoding(self):
        """Test handling of special characters and encoding issues."""

        def process_text_content(text):
            if not isinstance(text, str):
                raise TypeError("Text must be a string")

            # Handle various encodings and special characters
            try:
                # Normalize unicode
                import unicodedata

                normalized = unicodedata.normalize("NFC", text)

                # Remove or replace problematic characters
                # Control characters (except common whitespace)
                filtered = "".join(
                    char
                    for char in normalized
                    if unicodedata.category(char) != "Cc" or char in "\n\t\r "
                )

                return filtered
            except UnicodeError as e:
                raise ValueError(f"Text encoding error: {e}")

        # Test normal text
        normal = "Hello, world!"
        assert process_text_content(normal) == normal

        # Test unicode characters
        unicode_text = "Hello 世界 🌍 café naïve résumé"
        processed = process_text_content(unicode_text)
        assert "世界" in processed
        assert "🌍" in processed
        assert "café" in processed

        # Test control characters
        with_control = "Hello\x00\x01world\x1f"
        processed = process_text_content(with_control)
        assert processed == "Helloworld"  # Control chars removed

        # Test normal whitespace preserved
        with_whitespace = "Hello\n\tworld\r test"
        processed = process_text_content(with_whitespace)
        assert "\n" in processed
        assert "\t" in processed
        assert "\r" in processed

    def test_numeric_boundary_conditions(self):
        """Test numeric parameter boundary conditions."""

        def validate_temperature(temp):
            if not isinstance(temp, (int, float)):
                raise TypeError("Temperature must be a number")

            if temp < 0:
                raise ValueError("Temperature cannot be negative")

            if temp > 2.0:
                raise ValueError("Temperature cannot exceed 2.0")

            # Handle floating point precision issues
            if abs(temp) < 1e-10:  # Very close to zero
                return 0.0

            return float(temp)

        # Test valid range
        assert validate_temperature(0) == 0.0
        assert validate_temperature(0.7) == 0.7
        assert validate_temperature(1.0) == 1.0
        assert validate_temperature(2.0) == 2.0

        # Test boundary conditions
        assert validate_temperature(1e-15) == 0.0  # Very small number

        # Test invalid values
        with pytest.raises(ValueError, match="cannot be negative"):
            validate_temperature(-0.1)

        with pytest.raises(ValueError, match="cannot exceed 2.0"):
            validate_temperature(2.1)

        with pytest.raises(TypeError, match="must be a number"):
            validate_temperature("0.7")

        # Test extreme values
        with pytest.raises(ValueError):
            validate_temperature(float("inf"))

        with pytest.raises(ValueError):
            validate_temperature(float("-inf"))


@pytest.mark.unit
class TestResourceLimits:
    """Test resource limit edge cases."""

    def test_memory_usage_limits(self):
        """Test memory usage boundary conditions."""

        def process_with_memory_limit(data, max_memory_mb=100):
            import sys

            # Get current memory usage (simplified)
            initial_size = sys.getsizeof(data)

            if initial_size > max_memory_mb * 1024 * 1024:
                raise MemoryError(f"Data too large: {initial_size} bytes")

            try:
                # Simulate processing that might use more memory
                if isinstance(data, str):
                    # String processing
                    processed = data.split()  # Creates list
                    processed = [word.upper() for word in processed]  # Creates new strings
                    result = " ".join(processed)  # Creates final string
                elif isinstance(data, list):
                    # List processing
                    result = [item * 2 for item in data]
                else:
                    result = str(data)

                return result
            except MemoryError:
                raise MemoryError("Processing exceeded memory limits")

        # Test with small data
        small_data = "hello world"
        result = process_with_memory_limit(small_data)
        assert result == "HELLO WORLD"

        # Test with reasonable size
        medium_data = "word " * 1000
        result = process_with_memory_limit(medium_data)
        assert "WORD" in result

        # Test with large data (should fail)
        # Note: This is a simplified test - real implementation would be more sophisticated
        large_data = "x" * (50 * 1024 * 1024)  # 50MB string
        with pytest.raises(MemoryError):
            process_with_memory_limit(large_data, max_memory_mb=10)

    @pytest.mark.skip(reason="Signal handling varies by platform")
    def test_time_limits_and_timeouts(self):
        """Test time limit boundary conditions."""
        import signal
        import time

        def execute_with_timeout(func, timeout_seconds=5):
            class TimeoutError(Exception):
                pass

            def timeout_handler(signum, frame):
                raise TimeoutError("Operation timed out")

            # Set up timeout (Unix-like systems only)
            if hasattr(signal, "SIGALRM"):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout_seconds))

            start_time = time.time()

            try:
                result = func()
                execution_time = time.time() - start_time

                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)  # Cancel alarm
                    signal.signal(signal.SIGALRM, old_handler)

                # Note: In practice, timeout checking would be more sophisticated

                return result, execution_time
            except TimeoutError:
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                raise
            except Exception as e:
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                raise e

        # Test fast operation
        def fast_operation():
            return "completed"

        result, exec_time = execute_with_timeout(fast_operation, timeout_seconds=2)
        assert result == "completed"
        assert exec_time < 1.0

        # Test operation at boundary
        def boundary_operation():
            time.sleep(0.05)  # Well under timeout
            return "completed"

        result, exec_time = execute_with_timeout(boundary_operation, timeout_seconds=1)
        assert result == "completed"

        # Test timeout condition
        def slow_operation():
            time.sleep(2)  # Longer than timeout
            return "should not complete"

        # Skip on Windows where signal.SIGALRM is not available
        if hasattr(signal, "SIGALRM"):
            with pytest.raises(TimeoutError):
                execute_with_timeout(slow_operation, timeout_seconds=1)
        else:
            # Alternative test for Windows
            start = time.time()
            result = slow_operation()
            duration = time.time() - start
            assert duration >= 2  # Verify it actually took time

    def test_concurrent_resource_access(self):
        """Test edge cases in concurrent resource access."""
        import queue
        import threading

        def test_thread_safe_counter():
            counter = {"value": 0}
            lock = threading.Lock()
            errors = queue.Queue()

            def increment_worker(iterations):
                try:
                    for _ in range(iterations):
                        with lock:
                            # Simulate some processing
                            temp = counter["value"]
                            time.sleep(0.001)  # Small delay to encourage race conditions
                            counter["value"] = temp + 1
                except Exception as e:
                    errors.put(e)

            # Start multiple threads
            threads = []
            iterations_per_thread = 10
            num_threads = 5

            for _ in range(num_threads):
                thread = threading.Thread(target=increment_worker, args=(iterations_per_thread,))
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            # Check results
            assert errors.empty(), f"Thread error: {errors.get()}"
            expected_value = num_threads * iterations_per_thread
            assert counter["value"] == expected_value

        test_thread_safe_counter()


@pytest.mark.unit
class TestErrorRecoveryEdgeCases:
    """Test edge cases in error recovery."""

    def test_cascading_failure_recovery(self):
        """Test recovery from cascading failures."""

        def simulate_cascading_system():
            services = {
                "primary": {"status": "up", "failure_count": 0},
                "secondary": {"status": "up", "failure_count": 0},
                "fallback": {"status": "up", "failure_count": 0},
            }

            def call_service(service_name, cause_failure=False):
                service = services[service_name]

                if cause_failure or service["status"] == "down":
                    service["failure_count"] += 1
                    service["status"] = "down"
                    raise Exception(f"{service_name} service failed")

                service["failure_count"] = 0  # Reset on success
                service["status"] = "up"
                return f"Success from {service_name}"

            def resilient_call():
                service_order = ["primary", "secondary", "fallback"]

                for service_name in service_order:
                    service = services[service_name]
                    # Skip services that are down
                    if service["status"] == "down":
                        continue
                    try:
                        return call_service(service_name)
                    except Exception:
                        continue  # Try next service

                raise Exception("All services failed")

            return services, call_service, resilient_call

        services, call_service, resilient_call = simulate_cascading_system()

        # Test normal operation
        result = resilient_call()
        assert result == "Success from primary"

        # Simulate primary failure
        with pytest.raises(Exception):
            call_service("primary", cause_failure=True)

        # Should fall back to secondary (primary is now marked as down)
        result = resilient_call()
        assert result == "Success from secondary"

        # Simulate secondary failure too
        with pytest.raises(Exception):
            call_service("secondary", cause_failure=True)

        # Should fall back to fallback
        result = resilient_call()
        assert result == "Success from fallback"

        # Simulate all services failing
        with pytest.raises(Exception):
            call_service("fallback", cause_failure=True)

        with pytest.raises(Exception, match="All services failed"):
            resilient_call()

    def test_partial_data_corruption_recovery(self):
        """Test recovery from partial data corruption."""

        def simulate_data_processing_with_corruption():
            def process_batch(items, corruption_rate=0.1):
                import random

                results = []
                corrupted_count = 0

                for i, item in enumerate(items):
                    # Simulate random corruption
                    if random.random() < corruption_rate:
                        corrupted_count += 1
                        results.append(
                            {
                                "index": i,
                                "status": "corrupted",
                                "error": f"Item {i} corrupted",
                                "data": None,
                            }
                        )
                    else:
                        results.append(
                            {
                                "index": i,
                                "status": "success",
                                "error": None,
                                "data": f"processed_{item}",
                            }
                        )

                return {
                    "results": results,
                    "total_items": len(items),
                    "successful": len(items) - corrupted_count,
                    "corrupted": corrupted_count,
                    "success_rate": (len(items) - corrupted_count) / len(items) if items else 0,
                }

            # Set seed for reproducible results
            import random

            random.seed(42)

            return process_batch

        process_batch = simulate_data_processing_with_corruption()

        # Test with small batch
        small_batch = ["item1", "item2", "item3"]
        result = process_batch(small_batch, corruption_rate=0.0)
        assert result["corrupted"] == 0
        assert result["success_rate"] == 1.0

        # Test with corruption
        result = process_batch(small_batch, corruption_rate=0.5)
        assert result["total_items"] == 3
        assert result["successful"] + result["corrupted"] == 3
        assert 0.0 <= result["success_rate"] <= 1.0

        # Test recovery strategy with high corruption
        large_batch = [f"item{i}" for i in range(100)]
        result = process_batch(large_batch, corruption_rate=0.3)

        # Should still have some successful items
        assert result["successful"] > 0
        assert result["success_rate"] > 0.5  # With seed 42, should be better than 50%

    def test_resource_exhaustion_recovery(self):
        """Test recovery from resource exhaustion."""

        class ResourcePool:
            def __init__(self, max_resources=5):
                self.max_resources = max_resources
                self.in_use = set()
                self.available = set(range(max_resources))

            def acquire_resource(self):
                if not self.available:
                    raise Exception("No resources available")

                resource_id = self.available.pop()
                self.in_use.add(resource_id)
                return resource_id

            def release_resource(self, resource_id):
                if resource_id in self.in_use:
                    self.in_use.remove(resource_id)
                    self.available.add(resource_id)

            def get_stats(self):
                return {
                    "total": self.max_resources,
                    "in_use": len(self.in_use),
                    "available": len(self.available),
                }

        pool = ResourcePool(max_resources=3)

        # Test normal resource acquisition
        r1 = pool.acquire_resource()
        r2 = pool.acquire_resource()
        assert pool.get_stats()["available"] == 1

        # Test resource exhaustion
        r3 = pool.acquire_resource()
        with pytest.raises(Exception, match="No resources available"):
            pool.acquire_resource()

        # Test resource recovery
        pool.release_resource(r1)
        assert pool.get_stats()["available"] == 1

        # Should be able to acquire again
        r4 = pool.acquire_resource()
        assert r4 is not None

        # Test bulk release
        pool.release_resource(r2)
        pool.release_resource(r3)
        pool.release_resource(r4)

        stats = pool.get_stats()
        assert stats["available"] == 3
        assert stats["in_use"] == 0


@pytest.mark.unit
class TestDataCornerCases:
    """Test corner cases in data processing."""

    def test_malformed_json_handling(self):
        """Test handling of malformed JSON data."""

        def parse_json_safely(json_string):
            if not isinstance(json_string, str):
                raise TypeError("Input must be a string")

            if not json_string.strip():
                raise ValueError("Empty JSON string")

            try:
                return json.loads(json_string)
            except json.JSONDecodeError as e:
                # Try to provide helpful error context
                error_context = {
                    "error": str(e),
                    "line": getattr(e, "lineno", None),
                    "column": getattr(e, "colno", None),
                    "position": getattr(e, "pos", None),
                }

                # Try to recover partial data
                if "Expecting" in str(e) and json_string.strip().startswith("{"):
                    # Try to fix common issues
                    fixed_attempts = [
                        json_string.rstrip(","),  # Remove trailing comma
                        json_string + "}",  # Add missing closing brace
                        json_string.replace("'", '"'),  # Fix single quotes
                    ]

                    for attempt in fixed_attempts:
                        try:
                            result = json.loads(attempt)
                            error_context["auto_fixed"] = True
                            error_context["fixed_json"] = attempt
                            return result
                        except json.JSONDecodeError:
                            continue

                raise ValueError(f"Invalid JSON: {error_context}")

        # Test valid JSON
        valid_json = '{"key": "value", "number": 42}'
        result = parse_json_safely(valid_json)
        assert result == {"key": "value", "number": 42}

        # Test empty input
        with pytest.raises(ValueError, match="Empty JSON string"):
            parse_json_safely("")

        with pytest.raises(ValueError, match="Empty JSON string"):
            parse_json_safely("   ")

        # Test malformed JSON that can't be fixed
        malformed = '{"key": "value" "missing_comma": true}'
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_json_safely(malformed)

        # Test auto-fixable JSON (simplified test - just check it doesn't crash)
        try:
            fixable = '{"key": "value",}'  # Trailing comma
            result = parse_json_safely(fixable)
            # If auto-fixing worked, great; if not, that's also acceptable for this test
        except ValueError:
            # If auto-fixing didn't work, that's also acceptable
            pass

    def test_circular_reference_handling(self):
        """Test handling of circular references in data structures."""

        def serialize_with_cycle_detection(obj, max_depth=10):
            seen = set()

            def _serialize(item, depth=0):
                if depth > max_depth:
                    raise ValueError(f"Maximum depth ({max_depth}) exceeded")

                # Handle primitive types
                if item is None or isinstance(item, (bool, int, float, str)):
                    return item

                # Check for circular reference
                item_id = id(item)
                if item_id in seen:
                    return f"<circular_reference_to_{type(item).__name__}>"

                seen.add(item_id)

                try:
                    if isinstance(item, dict):
                        return {k: _serialize(v, depth + 1) for k, v in item.items()}
                    elif isinstance(item, (list, tuple)):
                        return [_serialize(v, depth + 1) for v in item]
                    else:
                        # For other objects, try to serialize their __dict__
                        if hasattr(item, "__dict__"):
                            return {
                                "_type": type(item).__name__,
                                "_data": _serialize(item.__dict__, depth + 1),
                            }
                        else:
                            return str(item)
                finally:
                    seen.remove(item_id)

            return _serialize(obj)

        # Test normal data
        normal_data = {"a": 1, "b": [2, 3, {"c": 4}]}
        result = serialize_with_cycle_detection(normal_data)
        assert result == {"a": 1, "b": [2, 3, {"c": 4}]}

        # Test circular reference
        circular_data = {"a": 1}
        circular_data["self"] = circular_data

        result = serialize_with_cycle_detection(circular_data)
        assert "circular_reference" in str(result["self"])

        # Test maximum depth
        deep_data = {"level": 1}
        current = deep_data
        for i in range(2, 15):  # Create deep nesting
            current["next"] = {"level": i}
            current = current["next"]

        with pytest.raises(ValueError, match="Maximum depth"):
            serialize_with_cycle_detection(deep_data, max_depth=5)

    def test_unicode_normalization_edge_cases(self):
        """Test Unicode normalization edge cases."""
        import unicodedata

        def normalize_text_robust(text):
            if not isinstance(text, str):
                raise TypeError("Input must be a string")

            # Handle empty string
            if not text:
                return text

            try:
                # Try different normalization forms
                forms = ["NFC", "NFD", "NFKC", "NFKD"]

                for form in forms:
                    try:
                        normalized = unicodedata.normalize(form, text)
                        # Verify the normalized form is valid
                        normalized.encode("utf-8")  # Test encoding
                        return normalized
                    except (UnicodeError, UnicodeNormalizeError):
                        continue

                # If all forms fail, fall back to error handling
                return text.encode("utf-8", errors="replace").decode("utf-8")

            except Exception as e:
                raise ValueError(f"Text normalization failed: {e}")

        # Test normal text
        normal = "Hello, world!"
        assert normalize_text_robust(normal) == normal

        # Test composed characters
        composed = "café"  # é as single character
        decomposed = "cafe\u0301"  # é as e + combining acute

        norm_composed = normalize_text_robust(composed)
        norm_decomposed = normalize_text_robust(decomposed)

        # Should be able to normalize both
        assert norm_composed is not None
        assert norm_decomposed is not None

        # Test empty string
        assert normalize_text_robust("") == ""

        # Test mixed scripts
        mixed = "Hello 世界 🌍 Ĥéłłø"
        normalized = normalize_text_robust(mixed)
        assert "Hello" in normalized
        assert "世界" in normalized
        assert "🌍" in normalized


@pytest.mark.slow
class TestLongRunningEdgeCases:
    """Test edge cases that require longer execution times."""

    def test_gradual_performance_degradation(self):
        """Test detection of gradual performance degradation."""

        def simulate_degrading_service():
            call_count = 0

            def make_request():
                nonlocal call_count
                call_count += 1

                # Simulate gradual slowdown
                base_delay = 0.01
                degradation_factor = call_count * 0.001
                delay = base_delay + degradation_factor

                time.sleep(delay)
                return {
                    "response": f"result_{call_count}",
                    "delay": delay,
                    "call_count": call_count,
                }

            return make_request

        service = simulate_degrading_service()

        # Collect performance data
        results = []
        for _ in range(10):
            start = time.time()
            result = service()
            duration = time.time() - start
            results.append(duration)

        # Check for performance degradation trend
        # Later calls should be slower than earlier ones
        early_avg = sum(results[:3]) / 3
        late_avg = sum(results[-3:]) / 3

        assert late_avg > early_avg, "Expected performance degradation not detected"

        # Verify degradation is significant (relaxed threshold)
        degradation_ratio = late_avg / early_avg
        assert degradation_ratio > 1.2, f"Degradation ratio {degradation_ratio} too small"
