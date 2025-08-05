# Use Case Examples

This directory contains complete examples for each of LLM Lab's main use cases.

## Overview

These examples demonstrate end-to-end workflows for common LLM evaluation and optimization tasks. Each example is self-contained and production-ready.

## Quick Start Examples

### 🚀 Benchmarking Workflow (`benchmarking_workflow.py`)

The most common use case - comparing multiple models on standard benchmarks.

```bash
# Quick start - benchmark popular models
python benchmarking_workflow.py --quick-start

# Custom benchmark
python benchmarking_workflow.py \
    --providers openai,anthropic,google \
    --models gpt-4,claude-3-5-sonnet-20241022,gemini-1.5-pro \
    --dataset truthfulqa \
    --output results/benchmark_results.html
```

## Complete Examples by Use Case

### 📊 1. Performance Benchmarking

#### `performance_benchmark_suite.py`
Complete performance testing across providers.

**Features:**
- Latency measurement
- Throughput testing  
- Token rate analysis
- Cost per token calculation
- Statistical significance testing

**Run:**
```bash
python performance_benchmark_suite.py --comprehensive
```

### 💰 2. Cost Analysis

#### `cost_analysis.py`
Analyze and optimize costs across providers.

**Features:**
- Cost breakdowns by model
- Budget forecasting
- Cost vs quality trade-offs
- Optimization recommendations

**Run:**
```bash
python cost_analysis.py --budget 1000 --quality-threshold 0.9
```

#### `cost_quality_metrics.py`
Detailed cost-quality analysis with visualizations.

**Run:**
```bash
python cost_quality_metrics.py --generate-report
```

#### `cost_scenarios.py`
Test different usage scenarios and their costs.

**Run:**
```bash
python cost_scenarios.py --scenario high-volume
```

### 🔍 3. Cross-LLM Testing

#### `cross_llm_testing_examples.py`
Test prompt consistency across different LLMs.

**Features:**
- Response consistency metrics
- Output format validation
- Error rate comparison
- Fallback strategies

**Run:**
```bash
python cross_llm_testing_examples.py --test-suite consistency
```

#### `regression_testing_suite.py`
Ensure model updates don't break existing functionality.

**Run:**
```bash
python regression_testing_suite.py --baseline results/baseline.json
```

### 📝 4. Custom Prompts

#### `custom_prompt_complete_example.py`
Full workflow for testing custom prompts.

**Features:**
- Prompt template management
- Variable substitution
- Response validation
- Metric collection

**Run:**
```bash
python custom_prompt_complete_example.py --template medical_diagnosis
```

#### `custom_prompt_with_metrics.py`
Custom prompts with detailed metrics.

**Run:**
```bash
python custom_prompt_with_metrics.py --metrics accuracy,relevance,completeness
```

### 🖥️ 5. Local Models

#### `local_model_demo.py`
Work with locally hosted models.

**Features:**
- Local model loading
- Memory optimization
- Batch processing
- GPU utilization

**Run:**
```bash
python local_model_demo.py --model llama-2-7b --device cuda
```

### 🎯 6. Fine-tuning

#### `fine_tuning_complete_demo.py`
Complete fine-tuning workflow with evaluation.

**Features:**
- Dataset preparation
- Training configuration
- Progress monitoring
- Before/after comparison

**Run:**
```bash
python fine_tuning_complete_demo.py --base-model gpt-3.5-turbo --dataset custom.jsonl
```

### 🛡️ 7. Alignment

#### `alignment_demo.py`
Implement safety and alignment measures.

**Features:**
- Safety filtering
- Constitutional AI
- Response validation
- Alignment metrics

**Run:**
```bash
python alignment_demo.py --safety-level high --constitutional-rules ethics.yaml
```

### 📈 8. Monitoring

#### `monitoring_complete_demo.py`
Set up comprehensive monitoring.

**Features:**
- Real-time dashboards
- Alert configuration
- Performance tracking
- Anomaly detection

**Run:**
```bash
python monitoring_complete_demo.py --dashboard-port 8080 --enable-alerts
```

### 🔄 Integrated Workflows

#### `integrated_workflow_demo.py`
Combines multiple use cases into a complete pipeline.

**Features:**
- Multi-stage pipeline
- Conditional workflows
- Result aggregation
- Report generation

**Run:**
```bash
python integrated_workflow_demo.py --pipeline production
```

## Output Examples

The `results/` directory contains example outputs:

- `cost_quality_analysis.json` - Detailed cost/quality metrics
- `cost_quality_report.md` - Human-readable analysis report
- `cost_scenarios_analysis.json` - Scenario comparison data
- `cost_scenarios_report.md` - Scenario recommendations

## Common Patterns

### Loading Multiple Providers
```python
from src.providers import ProviderRegistry

registry = ProviderRegistry()
providers = {
    name: registry.create_provider(name, model=model)
    for name, model in [
        ("openai", "gpt-4"),
        ("anthropic", "claude-3-5-sonnet-20241022"),
        ("google", "gemini-1.5-pro")
    ]
}
```

### Running Parallel Evaluations
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {
        provider: executor.submit(evaluate_provider, provider, dataset)
        for provider in providers
    }
    results = {
        provider: future.result()
        for provider, future in futures.items()
    }
```

### Generating Reports
```python
from src.analysis import ReportGenerator

generator = ReportGenerator()
report = generator.create_report(
    results,
    format="html",
    include_charts=True,
    save_path="results/report.html"
)
```

## Best Practices

1. **Start Small**: Test with small datasets first
2. **Use Caching**: Enable caching to avoid redundant API calls
3. **Monitor Costs**: Always track API usage and costs
4. **Version Results**: Save results with timestamps
5. **Document Changes**: Keep notes on configuration changes

## Environment Setup

Create a `.env` file with your API keys:
```bash
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
GOOGLE_API_KEY=your-key-here
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Rate limit errors | Reduce parallel workers or add delays |
| Out of memory | Use smaller batch sizes or models |
| Inconsistent results | Set temperature=0 for deterministic output |
| High costs | Use smaller models or sample datasets |

## Contributing

We welcome new examples! Please:
1. Follow the existing naming pattern
2. Include comprehensive docstrings
3. Add error handling
4. Update this README

## Next Steps

- Read the [Use Cases Overview](../../docs/guides/USE_CASES_OVERVIEW.md)
- Try the [Quick Start Tutorial](../../docs/guides/PREREQUISITES.md)
- Explore [API Documentation](../../docs/api/README.md)
- Join [Discussions](https://github.com/remyolson/llm-lab/discussions)