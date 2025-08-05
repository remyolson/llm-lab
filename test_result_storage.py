#!/usr/bin/env python3
"""Test the result storage functionality."""

import sys
import os
import json
import tempfile
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules directly to avoid provider dependencies
import importlib.util

# Load modules
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load required modules
prompt_runner = load_module("prompt_runner", "src/use_cases/custom_prompts/prompt_runner.py")
result_storage = load_module("result_storage", "src/use_cases/custom_prompts/result_storage.py")

# Import classes
ExecutionResult = prompt_runner.ExecutionResult
ModelResponse = prompt_runner.ModelResponse
CustomPromptResult = result_storage.CustomPromptResult
JSONFormatter = result_storage.JSONFormatter
CSVFormatter = result_storage.CSVFormatter
MarkdownFormatter = result_storage.MarkdownFormatter
ResultStorage = result_storage.ResultStorage
ResultComparator = result_storage.ResultComparator

print("Testing Result Storage System")
print("=" * 60)

# Create mock data
def create_mock_response(model, success=True, response_text=None):
    """Create a mock ModelResponse."""
    if response_text is None:
        response_text = f"This is a response from {model}. The answer is 42."
    
    return ModelResponse(
        model=model,
        provider=model.split('-')[0],
        prompt="What is the meaning of life?",
        rendered_prompt="What is the meaning of life?",
        response=response_text if success else None,
        success=success,
        error=None if success else "API Error",
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(seconds=2),
        duration_seconds=2.0,
        retry_count=0,
        metadata={"temperature": 0.7}
    )

# Create mock execution result
responses = [
    create_mock_response("gpt-4", True, "The meaning of life is a profound philosophical question that has been pondered throughout history."),
    create_mock_response("claude-3", True, "Life's meaning is subjective and varies from person to person. Many find meaning through relationships, achievements, and personal growth."),
    create_mock_response("gemini-pro", False)
]

exec_result = ExecutionResult(
    prompt_template="What is {topic}?",
    template_variables={"topic": "the meaning of life"},
    models_requested=["gpt-4", "claude-3", "gemini-pro"],
    models_succeeded=["gpt-4", "claude-3"],
    models_failed=["gemini-pro"],
    total_duration_seconds=6.5,
    responses=responses,
    execution_mode="parallel"
)

# Test 1: Create CustomPromptResult
print("\n1. Creating CustomPromptResult from ExecutionResult")
print("-" * 40)
custom_result = CustomPromptResult.from_execution_result(exec_result)
print(f"Execution ID: {custom_result.execution_id}")
print(f"Prompt hash: {custom_result.prompt_hash}")
print(f"Models succeeded: {custom_result.models_succeeded}")
print(f"Models failed: {custom_result.models_failed}")

# Test 2: JSON Formatting
print("\n2. JSON Formatting")
print("-" * 40)
json_formatter = JSONFormatter(indent=2, include_responses=False)
json_output = json_formatter.format(custom_result)
print("JSON preview (without full responses):")
print(json_output[:300] + "...")

# Test 3: CSV Formatting
print("\n3. CSV Formatting")
print("-" * 40)
csv_formatter = CSVFormatter()
csv_output = csv_formatter.format(custom_result)
print("CSV output:")
print(csv_output)

# Test 4: Markdown Formatting
print("\n4. Markdown Formatting")
print("-" * 40)
md_formatter = MarkdownFormatter()
md_output = md_formatter.format(custom_result)
print("Markdown preview:")
print(md_output[:500] + "...")

# Test 5: Storage System
print("\n5. Storage System")
print("-" * 40)
with tempfile.TemporaryDirectory() as temp_dir:
    storage = ResultStorage(temp_dir)
    
    # Save in different formats
    json_path = storage.save(custom_result, format="json")
    csv_path = storage.save(custom_result, format="csv")
    md_path = storage.save(custom_result, format="markdown")
    
    print(f"Saved JSON to: {json_path}")
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved Markdown to: {md_path}")
    
    # Test cache
    print("\n6. Cache Testing")
    print("-" * 40)
    cached = storage.check_cache(
        exec_result.prompt_template,
        exec_result.template_variables
    )
    print(f"Found in cache: {cached is not None}")
    if cached:
        print(f"Cached execution ID: {cached.execution_id}")
    
    # List results
    print("\n7. Listing Results")
    print("-" * 40)
    results_list = storage.list_results()
    print(f"Found {len(results_list)} results")
    for result in results_list:
        print(f"  - {result['execution_id']}: {result['prompt_preview']}")

# Test 8: Result Comparison
print("\n8. Result Comparison")
print("-" * 40)
comparison = ResultComparator.compare_responses(custom_result)
print(f"Comparison analysis:")
print(f"  Models: {comparison['models']}")
print(f"  Response lengths: {comparison['response_lengths']}")
print(f"  Common phrases: {comparison['common_phrases'][:5]}...")
print(f"  Agreement score: {comparison['agreement_score']:.3f}")

# Test 9: Metrics Integration (mock)
print("\n9. Metrics Integration")
print("-" * 40)
# Add mock metrics
custom_result.metrics = {
    "gpt-4": {
        "response_length": {"words": 50},
        "sentiment": {"score": 0.8, "label": "positive"},
        "coherence": {"score": 0.92}
    },
    "claude-3": {
        "response_length": {"words": 45},
        "sentiment": {"score": 0.85, "label": "positive"},
        "coherence": {"score": 0.95}
    }
}

custom_result.aggregated_metrics = {
    "response_length": {"mean": 47.5, "std": 2.5},
    "sentiment": {"mean": 0.825},
    "coherence": {"mean": 0.935},
    "diversity": {"score": 0.72}
}

# Format with metrics
json_with_metrics = JSONFormatter().format(custom_result)
print("JSON with metrics preview:")
data = json.loads(json_with_metrics)
print(f"  Metrics: {list(data.get('metrics', {}).keys())}")
print(f"  Aggregated metrics: {list(data.get('aggregated_metrics', {}).keys())}")

print("\n✅ All result storage tests completed!")