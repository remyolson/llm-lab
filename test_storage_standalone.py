#!/usr/bin/env python3
"""Standalone test of result storage (no external dependencies)."""

import sys
import os
import json
import tempfile
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

print("Testing Result Storage System (Standalone)")
print("=" * 60)

# Define minimal data structures for testing
@dataclass
class MockModelResponse:
    model: str
    provider: str
    prompt: str
    rendered_prompt: str
    response: Optional[str]
    success: bool
    error: Optional[str]
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    retry_count: int
    metadata: Dict[str, Any]
    
    def to_dict(self):
        return {
            "model": self.model,
            "provider": self.provider,
            "prompt": self.prompt,
            "rendered_prompt": self.rendered_prompt,
            "response": self.response,
            "success": self.success,
            "error": self.error,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "retry_count": self.retry_count,
            "metadata": self.metadata
        }

@dataclass
class MockExecutionResult:
    prompt_template: str
    template_variables: Dict[str, Any]
    models_requested: List[str]
    models_succeeded: List[str]
    models_failed: List[str]
    total_duration_seconds: float
    responses: List[MockModelResponse]
    execution_mode: str

# Import result_storage module directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src/use_cases/custom_prompts'))

# Mock the imports in result_storage
class MockModule:
    ExecutionResult = MockExecutionResult
    ModelResponse = MockModelResponse
    MetricResult = None

sys.modules['src.use_cases.custom_prompts.prompt_runner'] = MockModule()
sys.modules['src.use_cases.custom_prompts.evaluation_metrics'] = MockModule()

# Now import result_storage
import result_storage

# Create test data
print("\n1. Creating Mock Data")
print("-" * 40)

responses = [
    MockModelResponse(
        model="gpt-4",
        provider="openai",
        prompt="What is {topic}?",
        rendered_prompt="What is the meaning of life?",
        response="The meaning of life is 42, according to Douglas Adams.",
        success=True,
        error=None,
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(seconds=1.5),
        duration_seconds=1.5,
        retry_count=0,
        metadata={"temperature": 0.7}
    ),
    MockModelResponse(
        model="claude-3",
        provider="anthropic",
        prompt="What is {topic}?",
        rendered_prompt="What is the meaning of life?",
        response="Life's meaning is subjective and personal to each individual.",
        success=True,
        error=None,
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(seconds=2.0),
        duration_seconds=2.0,
        retry_count=0,
        metadata={"temperature": 0.7}
    )
]

exec_result = MockExecutionResult(
    prompt_template="What is {topic}?",
    template_variables={"topic": "the meaning of life"},
    models_requested=["gpt-4", "claude-3"],
    models_succeeded=["gpt-4", "claude-3"],
    models_failed=[],
    total_duration_seconds=3.5,
    responses=responses,
    execution_mode="parallel"
)

print(f"Created execution result with {len(responses)} responses")

# Test formatters
print("\n2. Testing Formatters")
print("-" * 40)

# JSON Formatter
json_formatter = result_storage.JSONFormatter(indent=2)
custom_result = result_storage.CustomPromptResult.from_execution_result(exec_result)
json_output = json_formatter.format(custom_result)
print("JSON output preview:")
print(json_output[:200] + "...")

# CSV Formatter
print("\n3. CSV Formatting")
print("-" * 40)
csv_formatter = result_storage.CSVFormatter()
csv_output = csv_formatter.format(custom_result)
print("CSV output:")
print(csv_output)

# Markdown Formatter
print("\n4. Markdown Formatting")
print("-" * 40)
md_formatter = result_storage.MarkdownFormatter()
md_output = md_formatter.format(custom_result)
print("Markdown preview:")
print(md_output[:400] + "...")

# Test storage
print("\n5. Testing Storage System")
print("-" * 40)
with tempfile.TemporaryDirectory() as temp_dir:
    storage = result_storage.ResultStorage(temp_dir)
    
    # Save results
    json_path = storage.save(custom_result, format="json")
    print(f"Saved to: {json_path}")
    
    # Verify file exists
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            saved_data = json.load(f)
        print(f"Verified save - execution_id: {saved_data['execution_id']}")
    
    # Test cache
    print("\n6. Testing Cache")
    print("-" * 40)
    cached = storage.check_cache(
        custom_result.prompt_template,
        custom_result.template_variables
    )
    print(f"Found in cache: {cached is not None}")

# Test comparison
print("\n7. Testing Result Comparison")
print("-" * 40)
comparison = result_storage.ResultComparator.compare_responses(custom_result)
print(f"Models compared: {comparison['models']}")
print(f"Response lengths: {comparison['response_lengths']}")
print(f"Agreement score: {comparison.get('agreement_score', 0):.3f}")

print("\n✅ All storage tests completed successfully!")