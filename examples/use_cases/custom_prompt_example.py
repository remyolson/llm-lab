#!/usr/bin/env python3
"""
Example: Using the Custom Prompt CLI and Template Engine

This example demonstrates how to use custom prompts with the LLM Lab
benchmarking system. It shows both CLI usage and programmatic usage
of the template engine and prompt runner.

Note: This example shows the intended usage. Actual execution requires
installing dependencies with: pip install -r requirements.txt
"""

import json

# Example 1: CLI Usage Examples
print("=" * 60)
print("CUSTOM PROMPT CLI EXAMPLES")
print("=" * 60)

print("\n1. Simple custom prompt:")
print("   python scripts/run_benchmarks.py \\")
print('   --custom-prompt "What is 2+2?" \\')
print("   --models gpt-4o-mini")

print("\n2. Custom prompt with template variables:")
print("   python scripts/run_benchmarks.py \\")
print('   --custom-prompt "Hello {model_name}, solve {equation}" \\')
print('   --prompt-variables \'{"equation": "x^2 + 5x + 6 = 0"}\' \\')
print("   --models gpt-4o-mini,claude-3-haiku")

print("\n3. Load prompt from file:")
print("   python scripts/run_benchmarks.py \\")
print("   --prompt-file examples/prompts/reasoning_template.txt \\")
print('   --prompt-variables \'{"question": "Why do stars twinkle?", "context": "astronomy"}\' \\')
print("   --models gpt-4,claude-3-sonnet,gemini-pro")

print("\n4. Parallel execution across multiple models:")
print("   python scripts/run_benchmarks.py \\")
print("   --prompt-file examples/prompts/code_review_template.txt \\")
print('   --prompt-variables \'{"language": "python", "code": "def add(a,b): return a+b"}\' \\')
print("   --models gpt-4o-mini,claude-3-haiku,gemini-flash \\")
print("   --parallel")

# Example 2: Template Engine Usage
print("\n\n" + "=" * 60)
print("TEMPLATE ENGINE USAGE EXAMPLES")
print("=" * 60)

# Show template content
template_content = """You are {model_name}, a helpful AI assistant. Today's date is {date}.

{?context}Given the following context:
{context}

{/context}Please answer the following question step by step:

Question: {question}

Let's think about this carefully and provide a detailed answer."""

print("\n1. Template with conditional sections:")
print(template_content)

print("\n2. Rendering with context:")
example_context = {
    "model_name": "gpt-4",
    "question": "What causes rainbows?",
    "context": "Consider the physics of light refraction",
}
print(f"\nContext: {json.dumps(example_context, indent=2)}")

print("\n3. Template variables detected:")
print("   - Required: model_name, question")
print("   - Optional: context (conditional)")
print("   - Built-in: date, timestamp, time")

# Example 3: Prompt Runner Usage (Programmatic)
print("\n\n" + "=" * 60)
print("PROMPT RUNNER PROGRAMMATIC USAGE")
print("=" * 60)

print("""
from src.use_cases.custom_prompts import PromptRunner, PromptTemplate

# Initialize runner
runner = PromptRunner(
    max_retries=3,
    retry_delay=1.0,
    progress_callback=lambda msg, pct: print(f"[{pct:3.0f}%] {msg}")
)

# Run on single model
response = runner.run_single(
    "Explain {concept} in simple terms",
    "gpt-4",
    {"concept": "quantum entanglement"}
)

# Run on multiple models in parallel
result = runner.run_multiple(
    prompt="Compare {item1} and {item2}",
    models=["gpt-4", "claude-3-opus", "gemini-pro"],
    template_variables={"item1": "Python", "item2": "JavaScript"},
    parallel=True
)

# Save results
runner.save_results(result, "results/custom_prompt_comparison.json")
""")

# Example 4: Expected Output Format
print("\n\n" + "=" * 60)
print("EXPECTED OUTPUT FORMAT")
print("=" * 60)

example_output = {
    "prompt_template": "Compare {item1} and {item2}",
    "template_variables": {"item1": "Python", "item2": "JavaScript"},
    "models_requested": ["gpt-4", "claude-3-opus", "gemini-pro"],
    "models_succeeded": ["gpt-4", "gemini-pro"],
    "models_failed": ["claude-3-opus"],
    "total_duration_seconds": 12.45,
    "execution_mode": "parallel",
    "responses": [
        {
            "model": "gpt-4",
            "provider": "openai",
            "prompt": "Compare {item1} and {item2}",
            "rendered_prompt": "Compare Python and JavaScript",
            "response": "Python and JavaScript are both popular programming languages...",
            "success": True,
            "error": None,
            "start_time": "2025-08-04T12:00:00",
            "end_time": "2025-08-04T12:00:04",
            "duration_seconds": 4.23,
            "retry_count": 0,
            "metadata": {
                "generation_params": {"temperature": 0.7, "max_tokens": 1000},
                "template_variables": {"item1": "Python", "item2": "JavaScript"},
            },
        }
    ],
}

print("\nExample execution result:")
print(json.dumps(example_output, indent=2)[:500] + "...")

# Example 5: Advanced Features
print("\n\n" + "=" * 60)
print("ADVANCED FEATURES")
print("=" * 60)

print("\n1. Progress tracking:")
print("   - Real-time progress updates during parallel execution")
print("   - Shows completion status for each model")
print("   - Indicates success (✓) or failure (✗) for each model")

print("\n2. Error handling:")
print("   - Automatic retry with exponential backoff")
print("   - Graceful handling of API errors")
print("   - Timeout protection for long-running requests")

print("\n3. Template features:")
print("   - Variable interpolation: {variable_name}")
print("   - Conditional sections: {?condition}...{/condition}")
print("   - Built-in variables: {timestamp}, {date}, {time}")
print("   - Safe JSON serialization for complex values")

print("\n4. Output formats:")
print("   - JSON: Complete results in single file")
print("   - JSONL: One response per line for streaming")
print("   - CSV: Via --csv flag in CLI (legacy format)")

print("\n\n" + "=" * 60)
print("For more examples, see:")
print("- examples/prompts/reasoning_template.txt")
print("- examples/prompts/code_review_template.txt")
print("- docs/guides/USE_CASE_3_HOW_TO.md")
print("=" * 60)
