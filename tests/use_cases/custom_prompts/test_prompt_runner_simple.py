#!/usr/bin/env python3
"""Simple test of prompt runner core functionality."""

# Import paths fixed - sys.path manipulation removed
import json
import os
import sys
from datetime import datetime

from src.use_cases.custom_prompts.prompt_runner import ExecutionResult, ModelResponse

# Test just the data structures
from src.use_cases.custom_prompts.template_engine import PromptTemplate

print("Testing PromptRunner data structures...")
print("-" * 50)

# Test ModelResponse
print("\nTest 1: ModelResponse creation and serialization")
response = ModelResponse(
    model="gpt-4",
    provider="openai",
    prompt="What is {question}?",
    rendered_prompt="What is the meaning of life?",
    response="The answer is 42",
    success=True,
    error=None,
    start_time=datetime.now(),
    end_time=datetime.now(),
    duration_seconds=1.23,
    retry_count=0,
    metadata={"temperature": 0.7},
)

print(f"Created ModelResponse for {response.model}")
print(f"Response preview: {response.response[:30]}...")
response_dict = response.to_dict()
print(f"Serialized to dict with keys: {list(response_dict.keys())}")

# Test ExecutionResult
print("\n\nTest 2: ExecutionResult creation")
exec_result = ExecutionResult(
    prompt_template="Calculate {expression}",
    template_variables={"expression": "2+2"},
    models_requested=["gpt-4", "claude-3"],
    models_succeeded=["gpt-4"],
    models_failed=["claude-3"],
    total_duration_seconds=5.67,
    responses=[response],
    execution_mode="parallel",
)

print(f"Created ExecutionResult")
print(f"Models requested: {exec_result.models_requested}")
print(f"Models succeeded: {exec_result.models_succeeded}")
print(f"Models failed: {exec_result.models_failed}")
print(f"Execution mode: {exec_result.execution_mode}")

# Test serialization
print("\n\nTest 3: JSON serialization")
result_dict = exec_result.to_dict()
json_str = json.dumps(result_dict, indent=2)
print(f"Successfully serialized to JSON ({len(json_str)} chars)")
print("JSON preview:")
print(json_str[:300] + "...")

print("\n✅ All data structure tests passed!")
