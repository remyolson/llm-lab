#!/usr/bin/env python3
"""Test the prompt runner functionality."""

# Import paths fixed - sys.path manipulation removed
import os
import sys

from src.use_cases.custom_prompts import PromptRunner, PromptTemplate


# Mock provider for testing without API calls
class MockProvider:
    def __init__(self, model_name):
        self.model_name = model_name

    def initialize(self):
        pass

    def generate(self, prompt, **kwargs):
        return f"Mock response from {self.model_name}: The answer is 42."


# Monkey patch for testing
original_get_provider = None


def mock_get_provider_for_model(model_name):
    return MockProvider


# Test the prompt runner
def test_prompt_runner():
    print("Testing PromptRunner...")
    print("-" * 50)

    # Create runner
    runner = PromptRunner(max_retries=1, retry_delay=0.1)

    # Test 1: Single model execution
    print("\nTest 1: Single model execution")
    template = PromptTemplate("Hello {model_name}, what is {question}?")

    # Patch the provider getter
    import src.providers

    original_get_provider = src.providers.get_provider_for_model
    src.providers.get_provider_for_model = mock_get_provider_for_model

    try:
        response = runner.run_single(template, "test-model", {"question": "the meaning of life"})

        print(f"Model: {response.model}")
        print(f"Success: {response.success}")
        print(f"Response: {response.response}")
        print(f"Duration: {response.duration_seconds:.2f}s")
        print(f"Rendered prompt: {response.rendered_prompt}")

        # Test 2: Multiple models
        print("\n\nTest 2: Multiple models (sequential)")
        result = runner.run_multiple(
            "Calculate {expression} for me",
            ["model-1", "model-2", "model-3"],
            {"expression": "2 + 2"},
            parallel=False,
        )

        print(f"Models requested: {result.models_requested}")
        print(f"Models succeeded: {result.models_succeeded}")
        print(f"Models failed: {result.models_failed}")
        print(f"Total duration: {result.total_duration_seconds:.2f}s")
        print(f"Execution mode: {result.execution_mode}")

        for resp in result.responses:
            print(f"  - {resp.model}: {resp.response[:50]}...")

        # Test 3: Parallel execution
        print("\n\nTest 3: Multiple models (parallel)")
        result = runner.run_multiple(
            "What is {topic}?",
            ["gpt-4", "claude-3", "gemini-pro"],
            {"topic": "quantum computing"},
            parallel=True,
        )

        print(f"Execution mode: {result.execution_mode}")
        print(f"Total duration: {result.total_duration_seconds:.2f}s")
        print(f"Responses collected: {len(result.responses)}")

        # Test 4: Template from file
        print("\n\nTest 4: Template from file")
        if os.path.exists("examples/prompts/reasoning_template.txt"):
            result = runner.run_from_file(
                "examples/prompts/reasoning_template.txt",
                ["test-model"],
                {"question": "Why is the sky blue?", "context": "Physics explanation"},
            )
            print(f"Template loaded and executed successfully")
            print(f"Response preview: {result.responses[0].response[:100]}...")
        else:
            print("Skipping file test - template file not found")

    finally:
        # Restore original
        src.providers.get_provider_for_model = original_get_provider

    print("\n✅ All prompt runner tests completed!")


if __name__ == "__main__":
    test_prompt_runner()
