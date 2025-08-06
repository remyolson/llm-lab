"""
Automated testing suite for deployed models
Includes latency benchmarks, quality regression tests, and API compatibility checks
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import asyncio
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestType(Enum):
    LATENCY = "latency"
    QUALITY = "quality"
    REGRESSION = "regression"
    API_COMPATIBILITY = "api_compatibility"
    LOAD = "load"
    FUNCTIONALITY = "functionality"


class TestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TestCase:
    """Represents a single test case"""

    id: str
    name: str
    test_type: TestType
    input_data: Any
    expected_output: Optional[Any]
    actual_output: Optional[Any]
    status: TestStatus
    execution_time: Optional[float]
    error_message: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class TestResult:
    """Results from a test execution"""

    test_id: str
    test_name: str
    status: TestStatus
    passed: int
    failed: int
    skipped: int
    total_time: float
    test_cases: List[TestCase]
    metrics: Dict[str, Any]
    timestamp: datetime


class ModelTestSuite:
    """Comprehensive testing suite for deployed models"""

    def __init__(self, model_endpoint: str, model_name: str):
        self.model_endpoint = model_endpoint
        self.model_name = model_name
        self.tokenizer = None
        self.test_results: List[TestResult] = []

    async def run_full_suite(self) -> TestResult:
        """Run complete test suite"""
        logger.info(f"Starting full test suite for {self.model_name}")

        start_time = time.time()
        test_cases = []

        # Run different test types
        latency_tests = await self.test_latency()
        test_cases.extend(latency_tests)

        quality_tests = await self.test_quality()
        test_cases.extend(quality_tests)

        regression_tests = await self.test_regression()
        test_cases.extend(regression_tests)

        api_tests = await self.test_api_compatibility()
        test_cases.extend(api_tests)

        load_tests = await self.test_load()
        test_cases.extend(load_tests)

        # Calculate summary
        passed = sum(1 for tc in test_cases if tc.status == TestStatus.PASSED)
        failed = sum(1 for tc in test_cases if tc.status == TestStatus.FAILED)
        skipped = sum(1 for tc in test_cases if tc.status == TestStatus.SKIPPED)

        result = TestResult(
            test_id=f"suite_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            test_name=f"Full Test Suite - {self.model_name}",
            status=TestStatus.PASSED if failed == 0 else TestStatus.FAILED,
            passed=passed,
            failed=failed,
            skipped=skipped,
            total_time=time.time() - start_time,
            test_cases=test_cases,
            metrics=self._calculate_metrics(test_cases),
            timestamp=datetime.utcnow(),
        )

        self.test_results.append(result)
        return result

    async def test_latency(self) -> List[TestCase]:
        """Test model response latency"""
        test_cases = []

        # Test prompts of different lengths
        test_prompts = [
            ("short", "Hello"),
            ("medium", "Write a short story about a robot learning to paint."),
            (
                "long",
                "Explain quantum computing in detail, including its principles, current applications, potential future uses, and the main challenges facing the field. "
                * 3,
            ),
        ]

        for prompt_type, prompt in test_prompts:
            test_case = TestCase(
                id=f"latency_{prompt_type}",
                name=f"Latency Test - {prompt_type} prompt",
                test_type=TestType.LATENCY,
                input_data=prompt,
                expected_output=None,
                actual_output=None,
                status=TestStatus.PENDING,
                execution_time=None,
                error_message=None,
                metadata={},
            )

            try:
                # Measure latency
                latencies = []
                for _ in range(5):  # Run multiple times for average
                    start = time.time()
                    response = await self._call_model(prompt)
                    latency = (time.time() - start) * 1000  # Convert to ms
                    latencies.append(latency)

                avg_latency = statistics.mean(latencies)
                p95_latency = np.percentile(latencies, 95)

                test_case.actual_output = {
                    "avg_latency_ms": avg_latency,
                    "p95_latency_ms": p95_latency,
                    "all_latencies": latencies,
                }
                test_case.execution_time = avg_latency

                # Check against thresholds
                if avg_latency < 1000:  # Under 1 second
                    test_case.status = TestStatus.PASSED
                else:
                    test_case.status = TestStatus.FAILED
                    test_case.error_message = f"Latency {avg_latency}ms exceeds threshold of 1000ms"

            except Exception as e:
                test_case.status = TestStatus.FAILED
                test_case.error_message = str(e)

            test_cases.append(test_case)

        return test_cases

    async def test_quality(self) -> List[TestCase]:
        """Test model output quality"""
        test_cases = []

        # Quality test scenarios
        quality_tests = [
            {
                "id": "coherence",
                "prompt": "Write a coherent paragraph about climate change.",
                "check": self._check_coherence,
            },
            {
                "id": "factuality",
                "prompt": "What is the capital of France?",
                "expected": "Paris",
                "check": self._check_factuality,
            },
            {
                "id": "instruction_following",
                "prompt": "List exactly 3 benefits of exercise, numbered 1-3.",
                "check": self._check_instruction_following,
            },
        ]

        for test in quality_tests:
            test_case = TestCase(
                id=f"quality_{test['id']}",
                name=f"Quality Test - {test['id']}",
                test_type=TestType.QUALITY,
                input_data=test["prompt"],
                expected_output=test.get("expected"),
                actual_output=None,
                status=TestStatus.PENDING,
                execution_time=None,
                error_message=None,
                metadata={},
            )

            try:
                response = await self._call_model(test["prompt"])
                test_case.actual_output = response

                # Run quality check
                passed, score, message = test["check"](response, test.get("expected"))

                test_case.metadata["quality_score"] = score
                test_case.status = TestStatus.PASSED if passed else TestStatus.FAILED
                if not passed:
                    test_case.error_message = message

            except Exception as e:
                test_case.status = TestStatus.FAILED
                test_case.error_message = str(e)

            test_cases.append(test_case)

        return test_cases

    async def test_regression(self) -> List[TestCase]:
        """Test for regression against baseline"""
        test_cases = []

        # Regression test cases
        regression_tests = [
            {
                "prompt": "Translate 'Hello world' to Spanish",
                "baseline_response": "Hola mundo",
                "similarity_threshold": 0.8,
            },
            {"prompt": "What is 2 + 2?", "baseline_response": "4", "similarity_threshold": 1.0},
        ]

        for i, test in enumerate(regression_tests):
            test_case = TestCase(
                id=f"regression_{i}",
                name=f"Regression Test #{i + 1}",
                test_type=TestType.REGRESSION,
                input_data=test["prompt"],
                expected_output=test["baseline_response"],
                actual_output=None,
                status=TestStatus.PENDING,
                execution_time=None,
                error_message=None,
                metadata={"threshold": test["similarity_threshold"]},
            )

            try:
                response = await self._call_model(test["prompt"])
                test_case.actual_output = response

                # Calculate similarity
                similarity = self._calculate_similarity(response, test["baseline_response"])

                test_case.metadata["similarity"] = similarity

                if similarity >= test["similarity_threshold"]:
                    test_case.status = TestStatus.PASSED
                else:
                    test_case.status = TestStatus.FAILED
                    test_case.error_message = (
                        f"Similarity {similarity} below threshold {test['similarity_threshold']}"
                    )

            except Exception as e:
                test_case.status = TestStatus.FAILED
                test_case.error_message = str(e)

            test_cases.append(test_case)

        return test_cases

    async def test_api_compatibility(self) -> List[TestCase]:
        """Test API compatibility"""
        test_cases = []

        # API compatibility tests
        api_tests = [
            {
                "id": "openai_format",
                "payload": {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "temperature": 0.7,
                },
            },
            {
                "id": "streaming",
                "payload": {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": "Count to 5"}],
                    "stream": True,
                },
            },
        ]

        for test in api_tests:
            test_case = TestCase(
                id=f"api_{test['id']}",
                name=f"API Compatibility - {test['id']}",
                test_type=TestType.API_COMPATIBILITY,
                input_data=test["payload"],
                expected_output=None,
                actual_output=None,
                status=TestStatus.PENDING,
                execution_time=None,
                error_message=None,
                metadata={},
            )

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.model_endpoint}/v1/chat/completions", json=test["payload"]
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            test_case.actual_output = data
                            test_case.status = TestStatus.PASSED
                        else:
                            test_case.status = TestStatus.FAILED
                            test_case.error_message = f"API returned status {response.status}"

            except Exception as e:
                test_case.status = TestStatus.FAILED
                test_case.error_message = str(e)

            test_cases.append(test_case)

        return test_cases

    async def test_load(self) -> List[TestCase]:
        """Test model under load"""
        test_cases = []

        # Load test configuration
        concurrent_requests = 10
        total_requests = 50

        test_case = TestCase(
            id="load_test",
            name=f"Load Test - {concurrent_requests} concurrent requests",
            test_type=TestType.LOAD,
            input_data={"concurrent": concurrent_requests, "total": total_requests},
            expected_output=None,
            actual_output=None,
            status=TestStatus.PENDING,
            execution_time=None,
            error_message=None,
            metadata={},
        )

        try:
            start_time = time.time()
            latencies = []
            errors = 0

            # Create tasks for concurrent requests
            async def make_request():
                try:
                    req_start = time.time()
                    await self._call_model("Test prompt for load testing")
                    latency = time.time() - req_start
                    return latency, None
                except Exception as e:
                    return None, str(e)

            # Run requests in batches
            for batch_start in range(0, total_requests, concurrent_requests):
                batch_size = min(concurrent_requests, total_requests - batch_start)
                tasks = [make_request() for _ in range(batch_size)]
                results = await asyncio.gather(*tasks)

                for latency, error in results:
                    if error:
                        errors += 1
                    elif latency:
                        latencies.append(latency * 1000)  # Convert to ms

            total_time = time.time() - start_time

            test_case.actual_output = {
                "total_requests": total_requests,
                "successful_requests": total_requests - errors,
                "failed_requests": errors,
                "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
                "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
                "requests_per_second": total_requests / total_time,
            }

            test_case.execution_time = total_time * 1000

            # Pass if error rate is below 5%
            error_rate = errors / total_requests
            if error_rate < 0.05:
                test_case.status = TestStatus.PASSED
            else:
                test_case.status = TestStatus.FAILED
                test_case.error_message = f"Error rate {error_rate * 100:.1f}% exceeds 5% threshold"

        except Exception as e:
            test_case.status = TestStatus.FAILED
            test_case.error_message = str(e)

        test_cases.append(test_case)
        return [test_case]

    async def _call_model(self, prompt: str) -> str:
        """Call the model endpoint"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.7,
            }

            async with session.post(
                f"{self.model_endpoint}/v1/chat/completions", json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"Model API error: {response.status}")

    def _check_coherence(self, response: str, expected: Optional[str]) -> Tuple[bool | float | str]:
        """Check response coherence"""
        # Simple coherence check based on length and structure
        sentences = response.split(".")
        word_count = len(response.split())

        if len(sentences) >= 2 and word_count >= 20:
            return True, 0.8, "Response is coherent"
        else:
            return False, 0.3, "Response lacks coherence"

    def _check_factuality(
        self, response: str, expected: Optional[str]
    ) -> Tuple[bool | float | str]:
        """Check factual accuracy"""
        if expected and expected.lower() in response.lower():
            return True, 1.0, "Factually correct"
        else:
            return False, 0.0, f"Expected '{expected}' not found in response"

    def _check_instruction_following(
        self, response: str, expected: Optional[str]
    ) -> Tuple[bool | float | str]:
        """Check if model follows instructions"""
        # Check for numbered list
        has_numbers = all(f"{i}." in response or f"{i})" in response for i in [1, 2, 3])
        lines = response.strip().split("\n")

        if has_numbers and len(lines) >= 3:
            return True, 0.9, "Instructions followed correctly"
        else:
            return False, 0.4, "Did not follow instructions properly"

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        # Simple character-based similarity
        text1_lower = text1.lower().strip()
        text2_lower = text2.lower().strip()

        if text1_lower == text2_lower:
            return 1.0

        # Calculate Jaccard similarity of words
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _calculate_metrics(self, test_cases: List[TestCase]) -> Dict[str | Any]:
        """Calculate overall metrics from test cases"""
        latency_tests = [tc for tc in test_cases if tc.test_type == TestType.LATENCY]
        quality_tests = [tc for tc in test_cases if tc.test_type == TestType.QUALITY]

        metrics = {"avg_latency_ms": 0, "quality_score": 0, "pass_rate": 0}

        if latency_tests:
            latencies = []
            for tc in latency_tests:
                if tc.actual_output and isinstance(tc.actual_output, dict):
                    latencies.append(tc.actual_output.get("avg_latency_ms", 0))
            if latencies:
                metrics["avg_latency_ms"] = statistics.mean(latencies)

        if quality_tests:
            scores = []
            for tc in quality_tests:
                if tc.metadata.get("quality_score"):
                    scores.append(tc.metadata["quality_score"])
            if scores:
                metrics["quality_score"] = statistics.mean(scores)

        total_tests = len(test_cases)
        passed_tests = sum(1 for tc in test_cases if tc.status == TestStatus.PASSED)

        if total_tests > 0:
            metrics["pass_rate"] = passed_tests / total_tests

        return metrics

    def generate_report(self, result: TestResult) -> str:
        """Generate a test report"""
        report = f"""
# Test Report - {result.test_name}

## Summary
- **Test ID**: {result.test_id}
- **Timestamp**: {result.timestamp}
- **Total Tests**: {len(result.test_cases)}
- **Passed**: {result.passed}
- **Failed**: {result.failed}
- **Skipped**: {result.skipped}
- **Pass Rate**: {(result.passed / len(result.test_cases) * 100):.1f}%
- **Total Time**: {result.total_time:.2f}s

## Metrics
- **Average Latency**: {result.metrics.get("avg_latency_ms", 0):.2f}ms
- **Quality Score**: {result.metrics.get("quality_score", 0):.2f}
- **Overall Pass Rate**: {result.metrics.get("pass_rate", 0) * 100:.1f}%

## Test Results

"""

        # Group by test type
        for test_type in TestType:
            type_tests = [tc for tc in result.test_cases if tc.test_type == test_type]
            if type_tests:
                report += f"### {test_type.value.replace('_', ' ').title()} Tests\n\n"

                for tc in type_tests:
                    status_icon = "✅" if tc.status == TestStatus.PASSED else "❌"
                    report += f"- {status_icon} **{tc.name}**\n"

                    if tc.execution_time:
                        report += f"  - Execution Time: {tc.execution_time:.2f}ms\n"

                    if tc.status == TestStatus.FAILED and tc.error_message:
                        report += f"  - Error: {tc.error_message}\n"

                    report += "\n"

        return report


# Example usage
if __name__ == "__main__":

    async def main():
        test_suite = ModelTestSuite(
            model_endpoint="http://localhost:8000", model_name="fine-tuned-model"
        )

        result = await test_suite.run_full_suite()
        report = test_suite.generate_report(result)
        print(report)

    asyncio.run(main())
