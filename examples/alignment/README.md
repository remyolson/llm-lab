# Alignment Examples

This directory contains examples demonstrating LLM Lab's alignment and safety features.

## Overview

These examples show how to implement various alignment techniques to ensure LLMs behave safely and according to specified principles.

## Examples

### 1. Constitutional AI Demo (`constitutional_ai_demo.py`)

Demonstrates how to implement Constitutional AI principles to guide model behavior.

**What it does:**
- Loads constitutional rules from YAML files
- Applies rules to model outputs
- Provides feedback and corrections based on principles

**Prerequisites:**
- API key for at least one provider
- Constitutional rules defined in YAML

**How to run:**
```bash
python constitutional_ai_demo.py --provider openai --model gpt-4
```

**Expected output:**
- Original model response
- Constitutional evaluation results
- Corrected response (if needed)

### 2. Human-in-the-Loop Demo (`human_loop_demo.py`)

Shows how to integrate human feedback into the alignment process.

**What it does:**
- Generates responses from multiple models
- Presents options to human reviewer
- Learns from human preferences
- Updates alignment based on feedback

**Prerequisites:**
- Multiple provider API keys for comparison
- Terminal access for interactive feedback

**How to run:**
```bash
python human_loop_demo.py --providers openai,anthropic --interactive
```

**Expected output:**
- Side-by-side model comparisons
- Human preference collection interface
- Updated alignment metrics

### 3. Runtime Intervention Demo (`runtime_intervention_demo.py`)

Demonstrates real-time safety interventions during model generation.

**What it does:**
- Monitors model output in real-time
- Detects potential safety violations
- Intervenes to correct or stop generation
- Logs intervention events

**Prerequisites:**
- Streaming-capable provider (OpenAI or Anthropic)
- Safety rules configuration

**How to run:**
```bash
python runtime_intervention_demo.py --provider anthropic --stream --safety-level high
```

**Expected output:**
- Real-time generation with interventions
- Safety violation logs
- Final sanitized output

### 4. Safety Demo (`safety_demo.py`)

Comprehensive safety filtering and evaluation example.

**What it does:**
- Tests multiple safety filters
- Evaluates responses for various safety criteria
- Generates safety reports
- Compares safety across providers

**Prerequisites:**
- API keys for providers to test
- Safety filter configurations

**How to run:**
```bash
python safety_demo.py --test-suite comprehensive --output-report safety_report.html
```

**Expected output:**
- Safety evaluation scores
- Detailed violation reports
- HTML report with visualizations

## Constitutional Rules

The `constitutional_rules/` directory contains example rule sets:

### Educational Rules (`educational_rules.yaml`)
```yaml
principles:
  - id: factual_accuracy
    description: "Provide factually accurate information"
    weight: 0.9
  
  - id: age_appropriate
    description: "Ensure content is age-appropriate"
    weight: 0.8
```

### Safety Rules (`safety_rules.yaml`)
```yaml
filters:
  - type: toxicity
    threshold: 0.1
    action: block
  
  - type: personal_info
    patterns: ["ssn", "credit card"]
    action: redact
```

## Common Patterns

### Basic Alignment Check
```python
from src.use_cases.alignment import SafetyEvaluator

evaluator = SafetyEvaluator()
result = evaluator.evaluate(response_text)
if result.is_safe:
    print("Response passed safety checks")
else:
    print(f"Safety violations: {result.violations}")
```

### Constitutional AI Application
```python
from src.use_cases.alignment.constitutional import ConstitutionalEngine

engine = ConstitutionalEngine.from_yaml("constitutional_rules/educational_rules.yaml")
corrected = engine.apply(original_response)
```

### Human Preference Learning
```python
from src.use_cases.alignment.human_loop import PreferenceLearner

learner = PreferenceLearner()
learner.add_preference(response_a, response_b, preferred="a")
learner.update_model()
```

## Configuration Files

- `alignment_rules.yaml` - General alignment configuration
- `safety_filters.yaml` - Safety filter definitions
- `monitoring_config.yaml` - Monitoring settings for alignment

## Tips

1. **Start Simple**: Begin with `safety_demo.py` to understand basic safety filtering
2. **Custom Rules**: Modify YAML files to create domain-specific rules
3. **Combine Approaches**: Use multiple alignment techniques together
4. **Monitor Performance**: Track alignment metrics over time

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No safety violations detected" | Lower threshold values in safety rules |
| "Too many false positives" | Adjust filter sensitivity or add exceptions |
| "Slow performance" | Disable real-time monitoring for batch processing |

## Next Steps

- Explore [Use Case 7: Alignment Research](../../docs/guides/USE_CASE_7_HOW_TO.md)
- Read about [Constitutional AI](https://www.anthropic.com/constitutional.pdf)
- Implement custom safety filters
- Contribute your own alignment examples

## Related Examples

- [Custom Prompts](../custom_prompts/) - Prompt-based alignment
- [Monitoring](../use_cases/monitoring_demo.py) - Track alignment metrics
- [Fine-tuning](../use_cases/fine_tuning_demo.py) - Align through fine-tuning