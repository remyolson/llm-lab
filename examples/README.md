# LLM Lab Examples

Welcome to the LLM Lab examples directory! This collection demonstrates practical implementations of all major features.

## 🚀 Quick Start

New to LLM Lab? Start here:

```bash
# Run the quick start example - compares models on a benchmark
python examples/quick_start.py

# With specific providers
python examples/quick_start.py --providers openai,anthropic

# Generate an HTML report
python examples/quick_start.py --report html --samples 20
```

## 📁 Directory Structure

```
examples/
├── quick_start.py           # 🚀 Start here! Quick benchmarking demo
├── alignment/               # 🛡️ Safety and alignment examples
│   ├── README.md           # Detailed alignment guide
│   ├── constitutional_ai_demo.py
│   ├── human_loop_demo.py
│   ├── runtime_intervention_demo.py
│   └── safety_demo.py
├── custom_prompts/          # 📝 Custom prompt testing
│   ├── README.md           # Prompt testing guide
│   ├── code_generation_examples.sh
│   ├── creative_writing_examples.sh
│   └── customer_service_examples.sh
├── notebooks/               # 📓 Interactive tutorials
│   ├── 01_basic_multi_model_comparison.py
│   ├── 02_performance_analysis.py
│   └── integrated_workflows.ipynb
├── use_cases/              # 💼 Production-ready workflows
│   ├── README.md          # Comprehensive use case guide
│   ├── benchmarking_workflow.py
│   ├── cost_analysis.py
│   ├── fine_tuning_demo.py
│   ├── monitoring_demo.py
│   └── [more examples...]
└── results/                # 📊 Example outputs
    └── [generated results]
```

## 📚 Examples by Category

### 🚀 Getting Started

| Example | Description | Time | Difficulty |
|---------|-------------|------|------------|
| [`quick_start.py`](quick_start.py) | Compare multiple LLMs on benchmarks | 2-5 min | Beginner |
| [`notebooks/01_basic_multi_model_comparison.py`](notebooks/01_basic_multi_model_comparison.py) | Interactive model comparison | 5 min | Beginner |
| [`use_cases/benchmarking_workflow.py`](use_cases/benchmarking_workflow.py) | Complete benchmarking pipeline | 10 min | Intermediate |

### 🛡️ Alignment & Safety

| Example | Description | Features |
|---------|-------------|----------|
| [`alignment/safety_demo.py`](alignment/safety_demo.py) | Basic safety filtering | Toxicity detection, PII filtering |
| [`alignment/constitutional_ai_demo.py`](alignment/constitutional_ai_demo.py) | Constitutional AI implementation | Rule-based alignment |
| [`alignment/human_loop_demo.py`](alignment/human_loop_demo.py) | Human feedback integration | Interactive preference learning |
| [`alignment/runtime_intervention_demo.py`](alignment/runtime_intervention_demo.py) | Real-time safety monitoring | Stream intervention |

### 💰 Cost Analysis

| Example | Description | Business Value |
|---------|-------------|----------------|
| [`use_cases/cost_analysis.py`](use_cases/cost_analysis.py) | Comprehensive cost tracking | Budget management |
| [`use_cases/cost_quality_metrics.py`](use_cases/cost_quality_metrics.py) | Cost vs quality analysis | ROI optimization |
| [`use_cases/cost_scenarios.py`](use_cases/cost_scenarios.py) | Usage scenario modeling | Capacity planning |

### 📊 Performance & Benchmarking

| Example | Description | Metrics |
|---------|-------------|---------|
| [`notebooks/02_performance_analysis.py`](notebooks/02_performance_analysis.py) | Performance deep dive | Latency, throughput |
| [`use_cases/performance_benchmark_suite.py`](use_cases/performance_benchmark_suite.py) | Comprehensive benchmarks | All metrics |
| [`use_cases/regression_testing_suite.py`](use_cases/regression_testing_suite.py) | Regression detection | Quality assurance |

### 📝 Custom Prompts

| Example | Description | Use Case |
|---------|-------------|----------|
| [`custom_prompts/code_generation_examples.sh`](custom_prompts/code_generation_examples.sh) | Code generation prompts | Development |
| [`custom_prompts/creative_writing_examples.sh`](custom_prompts/creative_writing_examples.sh) | Creative content | Content creation |
| [`custom_prompts/customer_service_examples.sh`](custom_prompts/customer_service_examples.sh) | Support responses | Customer service |

### 🔧 Advanced Features

| Example | Description | Advanced Topics |
|---------|-------------|-----------------|
| [`use_cases/fine_tuning_demo.py`](use_cases/fine_tuning_demo.py) | Model fine-tuning | Custom training |
| [`use_cases/monitoring_demo.py`](use_cases/monitoring_demo.py) | Production monitoring | Dashboards, alerts |
| [`use_cases/local_model_demo.py`](use_cases/local_model_demo.py) | Local model usage | Self-hosted models |

## 🎯 Learning Paths

### Path 1: Basic User (30 minutes)
1. Start with [`quick_start.py`](quick_start.py)
2. Try [`notebooks/01_basic_multi_model_comparison.py`](notebooks/01_basic_multi_model_comparison.py)
3. Explore [`use_cases/cost_analysis.py`](use_cases/cost_analysis.py)

### Path 2: Developer (2 hours)
1. Complete Basic User path
2. Study [`alignment/README.md`](alignment/README.md) and run safety demos
3. Implement custom prompts with [`custom_prompts/`](custom_prompts/) examples
4. Build a benchmarking pipeline with [`use_cases/benchmarking_workflow.py`](use_cases/benchmarking_workflow.py)

### Path 3: Production Engineer (4 hours)
1. Complete Developer path
2. Set up monitoring with [`use_cases/monitoring_demo.py`](use_cases/monitoring_demo.py)
3. Implement fine-tuning with [`use_cases/fine_tuning_demo.py`](use_cases/fine_tuning_demo.py)
4. Create integrated workflow from [`notebooks/integrated_workflows.ipynb`](notebooks/integrated_workflows.ipynb)

## 💡 Best Practices

### Environment Setup
```bash
# 1. Create a .env file
cp .env.example .env

# 2. Add your API keys
# Edit .env and add:
# OPENAI_API_KEY=your-key
# ANTHROPIC_API_KEY=your-key
# GOOGLE_API_KEY=your-key

# 3. Install dependencies
pip install -r requirements.txt
```

### Running Examples
```bash
# Always run from project root
cd /path/to/llm-lab

# Run with Python
python examples/quick_start.py

# Or make executable
chmod +x examples/quick_start.py
./examples/quick_start.py
```

### Customizing Examples

1. **Copy an example as a template**:
   ```bash
   cp examples/quick_start.py my_custom_benchmark.py
   ```

2. **Modify for your needs**:
   - Change datasets
   - Add custom metrics
   - Adjust parameters
   - Add visualizations

3. **Follow the pattern**:
   - Clear documentation
   - Progress indicators
   - Error handling
   - Results export

## 📊 Example Outputs

All examples save outputs to `examples/results/` with timestamps:

```
results/
├── quick_start_truthfulqa_20250105_143022.json
├── benchmark_results_20250105_143500.csv
├── cost_analysis_report_20250105_144000.html
└── safety_evaluation_20250105_144500.pdf
```

### Sample Output: Quick Start
```json
{
  "metadata": {
    "dataset": "truthfulqa",
    "sample_size": 10,
    "providers": ["openai", "anthropic"],
    "timestamp": "20250105_143022"
  },
  "results": {
    "openai": {
      "average_score": 0.85,
      "average_latency": 0.89,
      "success_rate": 1.0
    },
    "anthropic": {
      "average_score": 0.88,
      "average_latency": 1.23,
      "success_rate": 1.0
    }
  }
}
```

## 🔧 Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| "No API key found" | Check your .env file and environment variables |
| "Rate limit exceeded" | Add delays or reduce parallel requests |
| "Module not found" | Run from project root, check Python path |
| "Out of memory" | Reduce batch size or use smaller models |

### Debug Mode

Run examples with debug output:
```bash
# Set log level
export LOG_LEVEL=DEBUG
python examples/quick_start.py

# Or use Python's verbose mode
python -v examples/quick_start.py
```

## 🤝 Contributing Examples

We welcome new examples! See our [contribution guidelines](../CONTRIBUTING.md).

### Example Template
```python
#!/usr/bin/env python3
"""
Example Title - Brief Description

Detailed description of what this example demonstrates.

Prerequisites:
    - Required API keys
    - Required packages
    - Required data

Usage:
    python examples/category/your_example.py [options]
"""

# Your implementation here
```

### Checklist for New Examples
- [ ] Clear documentation and purpose
- [ ] Error handling and validation
- [ ] Progress indicators for long operations
- [ ] Results saved to `examples/results/`
- [ ] Added to this README
- [ ] Tested with multiple providers

## 📚 Additional Resources

- [Full Documentation](../docs/README.md)
- [API Reference](../docs/api/README.md)
- [Provider Guides](../docs/providers/)
- [Troubleshooting Guide](../docs/guides/TROUBLESHOOTING.md)

## 🎓 Learning More

1. **Read the guides**: Start with use case guides in [`docs/guides/`](../docs/guides/)
2. **Explore the API**: Check [`docs/api/`](../docs/api/) for detailed API docs
3. **Join discussions**: Participate in [GitHub Discussions](https://github.com/remyolson/llm-lab/discussions)
4. **Contribute**: Share your examples and improvements!

---

Happy experimenting with LLM Lab! 🚀