"""
Custom Pytest Plugins

Domain-specific pytest plugins for LLM testing, including response comparison,
cost tracking, performance benchmarking, and test report generation.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
from _pytest.config import Config
from _pytest.nodes import Item
from _pytest.reports import TestReport
from _pytest.terminal import TerminalReporter

from .base import TestMetrics, TestPlugin


@dataclass
class LLMTestMetrics:
    """Extended metrics for LLM-specific testing."""

    test_name: str
    provider: Optional[str] = None
    model: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: Optional[float] = None

    # LLM-specific metrics
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0
    api_errors: int = 0

    # Cost tracking
    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    total_cost: float = 0.0

    # Performance metrics
    latency_p50: Optional[float] = None
    latency_p95: Optional[float] = None
    latency_p99: Optional[float] = None
    throughput: Optional[float] = None

    # Quality metrics
    accuracy: Optional[float] = None
    similarity_score: Optional[float] = None

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_cost(self, prompt_price: float = 0.00003, completion_price: float = 0.00006):
        """Calculate cost based on token usage."""
        self.prompt_cost = (self.prompt_tokens / 1000) * prompt_price
        self.completion_cost = (self.completion_tokens / 1000) * completion_price
        self.total_cost = self.prompt_cost + self.completion_cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class LLMTestPlugin(TestPlugin):
    """Main pytest plugin for LLM testing enhancements."""

    def __init__(self):
        self.metrics: Dict[str, LLMTestMetrics] = {}
        self.config: Optional[Config] = None
        self.report_path: Optional[Path] = None
        self.collect_metrics = True
        self.track_costs = True
        self.compare_responses = False

    def pytest_configure(self, config: Config):
        """Configure the plugin."""
        self.config = config

        # Add custom markers
        config.addinivalue_line("markers", "llm: mark test as LLM-specific test")
        config.addinivalue_line("markers", "provider(name): mark test for specific provider")
        config.addinivalue_line("markers", "model(name): mark test for specific model")
        config.addinivalue_line("markers", "benchmark: mark test as benchmark test")
        config.addinivalue_line("markers", "slow: mark test as slow running")

        # Get configuration options
        self.collect_metrics = config.getoption("--llm-metrics", default=True)
        self.track_costs = config.getoption("--track-costs", default=True)
        self.compare_responses = config.getoption("--compare-responses", default=False)
        self.report_path = Path(config.getoption("--llm-report", default="llm_test_report.json"))

    def pytest_collection_modifyitems(self, items: List[Item]):
        """Modify collected test items."""
        for item in items:
            # Auto-mark LLM tests
            if "provider" in item.fixturenames or "mock_provider" in item.fixturenames:
                item.add_marker(pytest.mark.llm)

            # Add provider/model info from fixtures
            if hasattr(item, "callspec"):
                params = item.callspec.params
                if "provider_name" in params:
                    item.add_marker(pytest.mark.provider(params["provider_name"]))
                if "model_name" in params:
                    item.add_marker(pytest.mark.model(params["model_name"]))

    def pytest_runtest_setup(self, item: Item):
        """Called before running a test."""
        if self.collect_metrics:
            # Initialize metrics for this test
            test_id = item.nodeid
            self.metrics[test_id] = LLMTestMetrics(test_name=test_id)

            # Extract provider/model from markers
            for marker in item.iter_markers(name="provider"):
                self.metrics[test_id].provider = marker.args[0]
            for marker in item.iter_markers(name="model"):
                self.metrics[test_id].model = marker.args[0]

    def pytest_runtest_teardown(self, item: Item):
        """Called after running a test."""
        if self.collect_metrics:
            test_id = item.nodeid
            if test_id in self.metrics:
                metrics = self.metrics[test_id]
                metrics.end_time = datetime.now()
                metrics.duration = (metrics.end_time - metrics.start_time).total_seconds()

                # Calculate costs if tracking
                if self.track_costs:
                    metrics.calculate_cost()

    def pytest_runtest_makereport(self, item: Item, call):
        """Create test report."""
        if call.when == "call" and self.collect_metrics:
            test_id = item.nodeid
            if test_id in self.metrics:
                # Extract metrics from test function if available
                if hasattr(item, "llm_metrics"):
                    test_metrics = item.llm_metrics
                    self.metrics[test_id].prompt_tokens = test_metrics.get("prompt_tokens", 0)
                    self.metrics[test_id].completion_tokens = test_metrics.get(
                        "completion_tokens", 0
                    )
                    self.metrics[test_id].api_calls = test_metrics.get("api_calls", 0)

    def pytest_terminal_summary(self, terminalreporter: TerminalReporter):
        """Add custom summary to terminal output."""
        if self.metrics:
            terminalreporter.section("LLM Test Metrics Summary")

            # Calculate totals
            total_tokens = sum(m.total_tokens for m in self.metrics.values())
            total_cost = sum(m.total_cost for m in self.metrics.values())
            total_api_calls = sum(m.api_calls for m in self.metrics.values())

            terminalreporter.write_line(f"Total API Calls: {total_api_calls}")
            terminalreporter.write_line(f"Total Tokens Used: {total_tokens:,}")
            terminalreporter.write_line(f"Total Cost: ${total_cost:.4f}")

            # Write detailed report
            self._write_report()

    def _write_report(self):
        """Write detailed metrics report to file."""
        if self.report_path:
            report_data = {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_tests": len(self.metrics),
                    "total_tokens": sum(m.total_tokens for m in self.metrics.values()),
                    "total_cost": sum(m.total_cost for m in self.metrics.values()),
                    "total_api_calls": sum(m.api_calls for m in self.metrics.values()),
                },
                "tests": {test_id: metrics.to_dict() for test_id, metrics in self.metrics.items()},
            }

            self.report_path.write_text(json.dumps(report_data, indent=2, default=str))


class CostTrackingPlugin(TestPlugin):
    """Plugin for tracking API costs during testing."""

    # Pricing per 1K tokens (example rates)
    PRICING = {
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        "gpt-3.5-turbo": {"prompt": 0.001, "completion": 0.002},
        "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
        "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
        "gemini-pro": {"prompt": 0.00025, "completion": 0.0005},
    }

    def __init__(self):
        self.costs: Dict[str, float] = defaultdict(float)
        self.token_usage: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"prompt": 0, "completion": 0}
        )
        self.cost_limit: Optional[float] = None

    def pytest_configure(self, config: Config):
        """Configure the plugin."""
        self.cost_limit = config.getoption("--cost-limit", default=None)
        if self.cost_limit:
            self.cost_limit = float(self.cost_limit)

    def track_usage(self, model: str, prompt_tokens: int, completion_tokens: int):
        """Track token usage and calculate cost."""
        self.token_usage[model]["prompt"] += prompt_tokens
        self.token_usage[model]["completion"] += completion_tokens

        if model in self.PRICING:
            pricing = self.PRICING[model]
            cost = (prompt_tokens / 1000) * pricing["prompt"] + (
                completion_tokens / 1000
            ) * pricing["completion"]
            self.costs[model] += cost

            # Check cost limit
            if self.cost_limit and sum(self.costs.values()) > self.cost_limit:
                pytest.fail(
                    f"Cost limit exceeded: ${sum(self.costs.values()):.4f} > ${self.cost_limit}"
                )

    def pytest_terminal_summary(self, terminalreporter: TerminalReporter):
        """Add cost summary to terminal output."""
        if self.costs:
            terminalreporter.section("API Cost Summary")

            for model, cost in self.costs.items():
                tokens = self.token_usage[model]
                terminalreporter.write_line(
                    f"{model}: ${cost:.4f} "
                    f"(Prompt: {tokens['prompt']:,}, Completion: {tokens['completion']:,})"
                )

            total_cost = sum(self.costs.values())
            terminalreporter.write_line(f"\nTotal Cost: ${total_cost:.4f}")

            if self.cost_limit:
                remaining = self.cost_limit - total_cost
                terminalreporter.write_line(f"Budget Remaining: ${remaining:.4f}")


class PerformancePlugin(TestPlugin):
    """Plugin for performance benchmarking and monitoring."""

    def __init__(self):
        self.performance_data: Dict[str, List[float]] = defaultdict(list)
        self.slow_threshold: float = 5.0  # seconds
        self.performance_regression_threshold: float = 0.2  # 20% slower

    def pytest_configure(self, config: Config):
        """Configure the plugin."""
        self.slow_threshold = float(config.getoption("--slow-threshold", default=5.0))

    def pytest_runtest_makereport(self, item: Item, call):
        """Track test performance."""
        if call.when == "call":
            duration = call.duration
            test_name = item.nodeid

            self.performance_data[test_name].append(duration)

            # Check for slow tests
            if duration > self.slow_threshold:
                item.add_marker(pytest.mark.slow)
                item.user_properties.append(("slow_test", duration))

    def pytest_terminal_summary(self, terminalreporter: TerminalReporter):
        """Add performance summary to terminal output."""
        if self.performance_data:
            terminalreporter.section("Performance Summary")

            # Find slowest tests
            slowest = sorted(
                [(name, max(durations)) for name, durations in self.performance_data.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:5]

            terminalreporter.write_line("Slowest Tests:")
            for name, duration in slowest:
                short_name = name.split("::")[-1] if "::" in name else name
                terminalreporter.write_line(f"  {short_name}: {duration:.3f}s")

            # Calculate statistics
            all_durations = [d for durations in self.performance_data.values() for d in durations]
            if all_durations:
                import statistics

                terminalreporter.write_line(
                    f"\nAverage Duration: {statistics.mean(all_durations):.3f}s"
                )
                terminalreporter.write_line(
                    f"Median Duration: {statistics.median(all_durations):.3f}s"
                )

                if len(all_durations) > 1:
                    terminalreporter.write_line(f"Std Dev: {statistics.stdev(all_durations):.3f}s")


class ResponseComparisonPlugin(TestPlugin):
    """Plugin for comparing LLM responses across providers."""

    def __init__(self):
        self.responses: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.comparison_results: List[Dict[str, Any]] = []

    def pytest_configure(self, config: Config):
        """Configure the plugin."""
        config.addinivalue_line("markers", "compare_responses: mark test for response comparison")

    def compare_responses(
        self, prompt: str, responses: Dict[str, str], method: str = "similarity"
    ) -> Dict[str, Any]:
        """Compare responses from different providers."""
        from difflib import SequenceMatcher

        result = {
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "responses": responses,
            "similarities": {},
        }

        providers = list(responses.keys())
        for i in range(len(providers)):
            for j in range(i + 1, len(providers)):
                p1, p2 = providers[i], providers[j]
                similarity = SequenceMatcher(None, responses[p1], responses[p2]).ratio()
                result["similarities"][f"{p1}_vs_{p2}"] = similarity

        self.comparison_results.append(result)
        return result

    def pytest_terminal_summary(self, terminalreporter: TerminalReporter):
        """Add comparison summary to terminal output."""
        if self.comparison_results:
            terminalreporter.section("Response Comparison Summary")

            for result in self.comparison_results[:5]:  # Show first 5
                terminalreporter.write_line(f"\nPrompt: {result['prompt']}")
                for comparison, similarity in result["similarities"].items():
                    terminalreporter.write_line(f"  {comparison}: {similarity:.2%} similar")


def pytest_configure(config):
    """Register plugins with pytest."""
    # Register LLM test plugin
    llm_plugin = LLMTestPlugin()
    config.pluginmanager.register(llm_plugin, "llm_test_plugin")

    # Register cost tracking plugin
    cost_plugin = CostTrackingPlugin()
    config.pluginmanager.register(cost_plugin, "cost_tracking_plugin")

    # Register performance plugin
    perf_plugin = PerformancePlugin()
    config.pluginmanager.register(perf_plugin, "performance_plugin")

    # Register response comparison plugin
    comparison_plugin = ResponseComparisonPlugin()
    config.pluginmanager.register(comparison_plugin, "response_comparison_plugin")


def pytest_addoption(parser):
    """Add custom command-line options."""
    group = parser.getgroup("llm", "LLM testing options")

    group.addoption(
        "--llm-metrics",
        action="store_true",
        default=False,
        help="Collect LLM-specific metrics during testing",
    )

    group.addoption(
        "--track-costs", action="store_true", default=False, help="Track API costs during testing"
    )

    group.addoption(
        "--cost-limit", type=float, default=None, help="Maximum allowed cost for test run"
    )

    group.addoption(
        "--compare-responses",
        action="store_true",
        default=False,
        help="Compare responses across providers",
    )

    group.addoption(
        "--llm-report", default="llm_test_report.json", help="Path to save LLM test report"
    )

    group.addoption(
        "--slow-threshold",
        type=float,
        default=5.0,
        help="Threshold in seconds for marking tests as slow",
    )
