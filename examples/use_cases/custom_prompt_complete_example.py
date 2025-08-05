#!/usr/bin/env python3
"""
Complete Example: Custom Prompts with Template Engine, Metrics, and Storage

This example demonstrates the complete workflow of using custom prompts
with all the implemented components working together.
"""

import json
from datetime import datetime
from pathlib import Path

print("=" * 70)
print("COMPLETE CUSTOM PROMPT SYSTEM EXAMPLE")
print("=" * 70)

# Example 1: Complete Workflow
print("\n1. Complete Workflow Example")
print("-" * 40)
print("""
from src.use_cases.custom_prompts import (
    PromptTemplate,
    PromptRunner,
    MetricSuite,
    ResultStorage,
    save_execution_result
)

# Step 1: Create a prompt template
template = PromptTemplate('''
You are {model_name}, an AI assistant specializing in {domain}.

{?context}Context: {context}
{/context}

Please provide a {style} explanation of: {topic}

Requirements:
- Use simple language
- Provide concrete examples
- Keep it under {max_words} words
''')

# Step 2: Initialize components
runner = PromptRunner(
    max_retries=3,
    retry_delay=1.0,
    progress_callback=lambda msg, pct: print(f"[{pct:3.0f}%] {msg}")
)

storage = ResultStorage("./results/custom_prompts")
metrics = MetricSuite()

# Step 3: Define template variables
variables = {
    "domain": "computer science",
    "style": "beginner-friendly",
    "topic": "recursion",
    "max_words": 200,
    "context": "Focus on practical programming examples"
}

# Step 4: Run on multiple models
result = runner.run_multiple(
    template,
    models=["gpt-4", "claude-3-sonnet", "gemini-pro"],
    template_variables=variables,
    parallel=True
)

# Step 5: Save with metrics
saved_path = save_execution_result(
    result,
    storage_dir="./results/custom_prompts",
    format="json",
    include_metrics=True
)

print(f"Results saved to: {saved_path}")
""")

# Example 2: Working with Cached Results
print("\n2. Caching and Performance")
print("-" * 40)
print("""
# Check if prompt was already run
cached_result = storage.check_cache(template.template_str, variables)

if cached_result:
    print(f"Using cached result from {cached_result.execution_timestamp}")
    # Use cached result instead of re-running
else:
    # Run the prompt
    result = runner.run_multiple(template, models, variables)
    # Save to cache
    storage.save(CustomPromptResult.from_execution_result(result))
""")

# Example 3: Advanced Metrics Analysis
print("\n3. Advanced Metrics Analysis")
print("-" * 40)
print("""
# Create custom metrics for technical content
def technical_depth_metric(response: str, **kwargs) -> dict:
    # Count technical terms
    tech_terms = ['algorithm', 'function', 'variable', 'loop', 'condition', 
                  'stack', 'base case', 'recursive call']
    
    term_count = sum(1 for term in tech_terms if term.lower() in response.lower())
    total_words = len(response.split())
    
    return {
        "technical_terms": term_count,
        "technical_density": term_count / total_words if total_words > 0 else 0
    }

# Add custom metric
custom_metrics = MetricSuite()
custom_metrics.add_metric(CustomMetric("technical_depth", technical_depth_metric))

# Evaluate responses
for response in result.responses:
    if response.success:
        evaluation = custom_metrics.evaluate(response.response)
        print(f"{response.model}: {evaluation}")
""")

# Example 4: Comparison and Reporting
print("\n4. Model Comparison and Reporting")
print("-" * 40)
print("""
# Compare responses
comparison = ResultComparator.compare_responses(custom_result)

# Generate comparison report
report = f'''
# Model Comparison Report

## Prompt
{custom_result.prompt_template}

## Results Summary
- Models tested: {', '.join(comparison['models'])}
- Agreement score: {comparison['agreement_score']:.2%}
- Common themes: {', '.join(comparison['common_phrases'][:5])}

## Response Characteristics
'''

for model, length in comparison['response_lengths'].items():
    report += f"- {model}: {length} words\\n"

# Save report
report_path = Path("./results/reports") / f"comparison_{datetime.now():%Y%m%d_%H%M%S}.md"
report_path.parent.mkdir(exist_ok=True)
report_path.write_text(report)
""")

# Example 5: Integration with CLI
print("\n5. CLI Integration")
print("-" * 40)
print("""
# Run with full metrics and storage
python scripts/run_benchmarks.py \\
  --prompt-file templates/code_explanation.txt \\
  --prompt-variables '{"topic": "binary search", "level": "beginner"}' \\
  --models gpt-4,claude-3,gemini-pro \\
  --parallel \\
  --metrics all \\
  --output-dir ./results/explanations \\
  --output-format json,csv,markdown

# View saved results
python -m src.use_cases.custom_prompts.view_result \\
  ./results/explanations/2024-01-15/20240115_140230_gpt-4.json
""")

# Example 6: Batch Processing
print("\n6. Batch Processing Multiple Prompts")
print("-" * 40)
print("""
# Process multiple topics
topics = ["recursion", "dynamic programming", "graph algorithms", "sorting"]
all_results = []

for topic in topics:
    # Check cache first
    cached = storage.check_cache(template.template_str, {"topic": topic, **base_vars})
    
    if cached:
        all_results.append(cached)
    else:
        # Run and save
        result = runner.run_multiple(
            template,
            models=["gpt-4", "claude-3"],
            template_variables={"topic": topic, **base_vars}
        )
        
        custom_result = CustomPromptResult.from_execution_result(result)
        storage.save(custom_result)
        all_results.append(custom_result)

# Analyze trends across topics
topic_metrics = {}
for i, result in enumerate(all_results):
    if result.aggregated_metrics:
        topic_metrics[topics[i]] = result.aggregated_metrics

# Generate summary report
summary = {
    "topics_analyzed": len(topics),
    "total_responses": sum(len(r.responses) for r in all_results),
    "average_coherence": statistics.mean([
        m.get('coherence', {}).get('mean', 0) 
        for m in topic_metrics.values()
    ])
}
""")

# Example 7: A/B Testing Prompts
print("\n7. A/B Testing Different Prompt Strategies")
print("-" * 40)
print("""
# Define two prompt variations
prompt_a = PromptTemplate("Explain {topic} in simple terms.")
prompt_b = PromptTemplate('''
You are an expert teacher. A student asks: "What is {topic}?"
Provide a clear, engaging explanation with examples.
''')

# Run both prompts
results_a = []
results_b = []

for i in range(5):  # Multiple runs for statistical significance
    result_a = runner.run_single(prompt_a, "gpt-4", {"topic": "recursion"})
    result_b = runner.run_single(prompt_b, "gpt-4", {"topic": "recursion"})
    
    if result_a.success:
        metrics_a = metrics.evaluate(result_a.response)
        results_a.append(metrics_a)
    
    if result_b.success:
        metrics_b = metrics.evaluate(result_b.response)
        results_b.append(metrics_b)

# Compare results
print("Prompt A - Average coherence:", 
      statistics.mean([r['coherence']['value']['score'] for r in results_a]))
print("Prompt B - Average coherence:", 
      statistics.mean([r['coherence']['value']['score'] for r in results_b]))
""")

# Example output structure
print("\n8. Example Output Structure")
print("-" * 40)

example_output = {
    "execution_id": "20240804_143022_gpt-4",
    "prompt_template": "Explain {topic} in {style} terms",
    "template_variables": {
        "topic": "recursion",
        "style": "simple"
    },
    "responses": [
        {
            "model": "gpt-4",
            "success": True,
            "response": "Recursion is when a function calls itself...",
            "duration_seconds": 1.23,
            "metrics": {
                "response_length": {"words": 150, "sentences": 8},
                "sentiment": {"score": 0.2, "label": "neutral"},
                "coherence": {"score": 0.89}
            }
        }
    ],
    "aggregated_metrics": {
        "response_length": {"mean": 145, "std": 12},
        "coherence": {"mean": 0.87, "std": 0.05},
        "diversity": {"score": 0.72}
    }
}

print("Example output structure:")
print(json.dumps(example_output, indent=2))

print("\n" + "=" * 70)
print("KEY FEATURES DEMONSTRATED")
print("=" * 70)
print("""
1. **Template Engine**: Variable substitution, conditionals, built-in variables
2. **Prompt Runner**: Parallel execution, retry logic, progress tracking
3. **Evaluation Metrics**: Length, sentiment, coherence, diversity, custom metrics
4. **Result Storage**: JSON/CSV/Markdown formats, caching, comparison
5. **CLI Integration**: Command-line interface with all features
6. **Batch Processing**: Process multiple prompts efficiently with caching
7. **A/B Testing**: Compare prompt effectiveness scientifically
""")

print("\nFor implementation details, see:")
print("- src/use_cases/custom_prompts/")
print("- examples/prompts/")
print("- docs/guides/USE_CASE_3_HOW_TO.md")