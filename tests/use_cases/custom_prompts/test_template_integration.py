#!/usr/bin/env python3
"""Test the template engine integration."""

# Import paths fixed - sys.path manipulation removed
import os
import sys

from src.use_cases.custom_prompts import PromptTemplate

# Test 1: Simple template with model_name and date
print("Test 1: Simple template")
print("-" * 50)
template1 = PromptTemplate("Hello {model_name}, what is 2+2? Today is {date}.")
result1 = template1.render({"model_name": "gpt-4"})
print(f"Template: {template1.template_str}")
print(f"Required vars: {template1.get_required_variables()}")
print(f"Result: {result1}")
print()

# Test 2: Template with conditional sections
print("Test 2: Conditional template")
print("-" * 50)
template2 = PromptTemplate("""Model: {model_name}
{?context}Context: {context}
{/context}Question: {question}""")
result2 = template2.render(
    {
        "model_name": "claude-3",
        "question": "What is the capital of France?",
        "context": "We are discussing European geography.",
    }
)
print(f"Template: {template2.template_str}")
print(f"Result: {result2}")
print()

# Test 3: Load from file
print("Test 3: Load template from file")
print("-" * 50)
template3 = PromptTemplate.from_file("examples/prompts/reasoning_template.txt")
result3 = template3.render(
    {
        "model_name": "gpt-4",
        "question": "Explain quantum computing",
        "context": "Focus on practical applications",
    }
)
print(f"Template name: {template3.name}")
print(f"Required vars: {template3.get_required_variables()}")
print(f"Result preview: {result3[:200]}...")
print()

# Test 4: Missing variables (non-strict mode)
print("Test 4: Missing variables (non-strict)")
print("-" * 50)
template4 = PromptTemplate("Analyze {topic} using {method} approach")
missing = template4.validate_context({"topic": "climate change"})
print(f"Missing variables: {missing}")
result4 = template4.render({"topic": "climate change"}, strict=False)
print(f"Result: {result4}")
print()

print("✅ All template tests completed!")
