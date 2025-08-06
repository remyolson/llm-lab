#!/usr/bin/env python3
"""
Example: Using Custom Prompts with Evaluation Metrics

This example demonstrates how to combine the prompt runner with evaluation metrics
to get comprehensive analysis of model responses.
"""

import json

# Example of how to use the custom prompts module with evaluation metrics
print("=" * 60)
print("CUSTOM PROMPTS WITH EVALUATION METRICS")
print("=" * 60)

# Example 1: Basic Integration Pattern
print("\n1. Basic Integration Pattern:")
print("-" * 40)
print("""
from src.use_cases.custom_prompts import (
    PromptRunner,
    PromptTemplate,
    MetricSuite,
    evaluate_response
)

# Initialize components
runner = PromptRunner()
metrics = MetricSuite()

# Run prompt on model
response = runner.run_single(
    "Explain {concept} in simple terms",
    "gpt-4",
    {"concept": "machine learning"}
)

# Evaluate the response if successful
if response.success:
    evaluation = metrics.evaluate(response.response)
    print(f"Response metrics: {evaluation}")
""")

# Example 2: Batch Evaluation with Multiple Models
print("\n2. Batch Evaluation with Multiple Models:")
print("-" * 40)
print("""
# Run on multiple models
result = runner.run_multiple(
    prompt="Write a haiku about {topic}",
    models=["gpt-4", "claude-3", "gemini-pro"],
    template_variables={"topic": "programming"},
    parallel=True
)

# Extract successful responses
responses = [r.response for r in result.responses if r.success]

# Evaluate all responses including diversity
batch_evaluation = metrics.evaluate_batch(responses)

# Display aggregated metrics
print(f"Average sentiment: {batch_evaluation['aggregated']['sentiment']['score']['mean']}")
print(f"Response diversity: {batch_evaluation['aggregated']['diversity']['value']['diversity_score']}")
""")

# Example 3: Custom Metrics for Specific Use Cases
print("\n3. Custom Metrics for Code Review:")
print("-" * 40)
print("""
from src.use_cases.custom_prompts import CustomMetric

# Define custom metrics for code review responses
def code_snippet_count(response: str, **kwargs) -> int:
    \"\"\"Count code snippets in the response.\"\"\"
    return response.count('```')

def suggestion_count(response: str, **kwargs) -> int:
    \"\"\"Count improvement suggestions.\"\"\"
    keywords = ['should', 'could', 'recommend', 'suggest', 'improve']
    return sum(1 for word in keywords if word in response.lower())

# Create custom metric suite
code_review_metrics = MetricSuite([
    ResponseLengthMetric(),
    CustomMetric("code_snippets", code_snippet_count),
    CustomMetric("suggestions", suggestion_count)
])

# Evaluate code review response
code_review_prompt = PromptTemplate('''
Review this {language} code:
```{language}
{code}
```
Provide specific suggestions for improvement.
''')

response = runner.run_single(
    code_review_prompt,
    "gpt-4",
    {
        "language": "python",
        "code": "def add(a,b):\\n    return a+b"
    }
)

if response.success:
    review_metrics = code_review_metrics.evaluate(response.response)
    print(f"Code snippets in review: {review_metrics['code_snippets']['value']}")
    print(f"Suggestions made: {review_metrics['suggestions']['value']}")
""")

# Example 4: Comparing Models with Metrics
print("\n4. Model Comparison with Metrics:")
print("-" * 40)

# Sample comparison data structure
model_comparison = {
    "prompt": "Explain quantum computing to a 10-year-old",
    "models": ["gpt-4", "claude-3-opus", "gemini-pro"],
    "results": {
        "gpt-4": {
            "response_length": {"words": 150, "sentences": 8},
            "sentiment": {"score": 0.85, "label": "positive"},
            "coherence": {"score": 0.92},
            "readability": "Grade 5",
        },
        "claude-3-opus": {
            "response_length": {"words": 180, "sentences": 10},
            "sentiment": {"score": 0.90, "label": "positive"},
            "coherence": {"score": 0.95},
            "readability": "Grade 4",
        },
        "gemini-pro": {
            "response_length": {"words": 120, "sentences": 7},
            "sentiment": {"score": 0.80, "label": "positive"},
            "coherence": {"score": 0.88},
            "readability": "Grade 6",
        },
    },
    "diversity_score": 0.72,
}

print("Model Comparison Results:")
print(json.dumps(model_comparison, indent=2))

# Example 5: Workflow for A/B Testing Prompts
print("\n\n5. A/B Testing Different Prompts:")
print("-" * 40)
print("""
# Define two prompt variations
prompt_a = "Explain {topic} clearly and concisely"
prompt_b = "You are an expert teacher. Help a student understand {topic}"

# Test both prompts on same model
responses_a = []
responses_b = []

for i in range(5):  # Multiple runs for statistical significance
    result_a = runner.run_single(prompt_a, "gpt-4", {"topic": "recursion"})
    result_b = runner.run_single(prompt_b, "gpt-4", {"topic": "recursion"})

    if result_a.success:
        responses_a.append(result_a.response)
    if result_b.success:
        responses_b.append(result_b.response)

# Evaluate both sets
eval_a = metrics.evaluate_batch(responses_a)
eval_b = metrics.evaluate_batch(responses_b)

# Compare key metrics
print(f"Prompt A - Avg coherence: {eval_a['aggregated']['coherence']['score']['mean']}")
print(f"Prompt B - Avg coherence: {eval_b['aggregated']['coherence']['score']['mean']}")
print(f"Prompt A - Response diversity: {eval_a['aggregated']['diversity']['value']['diversity_score']}")
print(f"Prompt B - Response diversity: {eval_b['aggregated']['diversity']['value']['diversity_score']}")
""")

# Example 6: Integrating Metrics into CLI
print("\n6. CLI Integration Example:")
print("-" * 40)
print("""
# Extended CLI usage with metrics
python scripts/run_benchmarks.py \\
  --custom-prompt "Analyze the {algorithm} algorithm" \\
  --prompt-variables '{"algorithm": "quicksort"}' \\
  --models gpt-4,claude-3 \\
  --metrics all \\
  --output-format json

# The output would include both responses and metrics:
{
  "execution_result": {
    "prompt_template": "Analyze the {algorithm} algorithm",
    "responses": [...],
    "metrics": {
      "individual": [
        {
          "model": "gpt-4",
          "response_length": {"words": 250, "sentences": 12},
          "sentiment": {"score": 0.2, "label": "neutral"},
          "coherence": {"score": 0.88}
        },
        ...
      ],
      "aggregated": {
        "response_length": {"words": {"mean": 235, "std": 21}},
        "diversity": {"score": 0.65}
      }
    }
  }
}
""")

# Example 7: Real-time Metric Tracking
print("\n7. Real-time Metric Tracking:")
print("-" * 40)
print("""
# Custom progress callback with metrics
def progress_with_metrics(message: str, progress: float, response: str = None):
    print(f"[{progress:3.0f}%] {message}")
    if response:
        # Quick metrics on partial response
        word_count = len(response.split())
        print(f"      Current length: {word_count} words")

runner_with_tracking = PromptRunner(
    progress_callback=progress_with_metrics
)

# Use with streaming responses (if supported by provider)
response = runner_with_tracking.run_single(
    "Write a detailed guide about {topic}",
    "gpt-4",
    {"topic": "API design best practices"}
)
""")

print("\n" + "=" * 60)
print("BENEFITS OF METRICS INTEGRATION")
print("=" * 60)
print("""
1. **Objective Comparison**: Compare models based on measurable criteria
2. **Quality Assurance**: Ensure responses meet minimum quality standards
3. **A/B Testing**: Scientifically test different prompt strategies
4. **Cost Optimization**: Balance response quality with API costs
5. **User Preference Learning**: Track which response characteristics users prefer
6. **Automated Filtering**: Reject responses that don't meet metric thresholds
7. **Performance Tracking**: Monitor model performance over time
""")

print("\nFor implementation details, see:")
print("- src/use_cases/custom_prompts/evaluation_metrics.py")
print("- examples/use_cases/custom_prompt_example.py")
print("- docs/guides/USE_CASE_3_HOW_TO.md")
