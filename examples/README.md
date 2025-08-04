# LLM Lab Examples and Use Cases

This directory contains practical examples and real-world use cases demonstrating the capabilities of the LLM Lab multi-model framework.

## 📚 Examples Overview

### 📓 Notebooks (`notebooks/`)

Interactive examples that can be run as Python scripts or converted to Jupyter notebooks:

#### 1. Basic Multi-Model Comparison (`01_basic_multi_model_comparison.py`)
**Purpose**: Learn the basics of comparing responses across multiple LLM providers.

**Features**:
- Provider initialization and configuration
- Interactive prompt selection
- Side-by-side response comparison
- Performance metrics (response time, success rate)
- Results export in JSON format

**Usage**:
```bash
python examples/notebooks/01_basic_multi_model_comparison.py
```

**What you'll learn**:
- How to set up multiple providers
- Basic error handling and retry logic
- Response quality comparison techniques
- Performance measurement basics

#### 2. Performance Analysis (`02_performance_analysis.py`)
**Purpose**: Deep dive into performance analysis and optimization strategies.

**Features**:
- Response time benchmarking across providers
- Throughput measurement (requests per minute)
- Cost efficiency analysis
- Statistical performance reporting
- Optimization recommendations

**Usage**:
```bash
python examples/notebooks/02_performance_analysis.py
```

**What you'll learn**:
- Performance benchmarking methodologies
- Statistical analysis of LLM performance
- Cost-performance trade-offs
- Rate limiting and reliability strategies

### 🎯 Use Cases (`use_cases/`)

Production-ready examples for specific business scenarios:

#### 1. Cost Analysis (`cost_analysis.py`)
**Purpose**: Comprehensive cost tracking and optimization for multi-provider LLM usage.

**Business Value**:
- Budget management and cost control
- Provider cost comparison
- ROI optimization
- Cost-per-use case analysis

**Features**:
- Real-time cost estimation using current pricing
- Daily/monthly budget tracking with alerts
- Cost-per-token analysis across providers
- Automated cost optimization recommendations
- Cost efficiency scoring

**Usage**:
```bash
python examples/use_cases/cost_analysis.py
```

**Use Cases**:
- Enterprise LLM budget management
- Cost-conscious model selection
- Multi-provider cost optimization
- Billing and chargeback systems

#### 2. Benchmarking Workflow (`benchmarking_workflow.py`)
**Purpose**: Complete end-to-end benchmarking pipeline for production environments.

**Business Value**:
- Quality assurance for LLM deployments
- Model performance monitoring
- Vendor evaluation and selection
- Continuous improvement tracking

**Features**:
- Custom dataset creation and validation
- Multi-provider benchmark execution
- Statistical analysis and reporting
- Results export in multiple formats (CSV, JSON)
- Automated performance comparison

**Usage**:
```bash
python examples/use_cases/benchmarking_workflow.py
```

**Use Cases**:
- Model evaluation for specific domains
- Vendor selection processes
- Quality assurance pipelines
- Performance regression detection

## 🚀 Getting Started

### Prerequisites

1. **API Keys**: Configure at least one provider in your `.env` file:
   ```bash
   GOOGLE_API_KEY=your-google-key-here
   OPENAI_API_KEY=your-openai-key-here
   ANTHROPIC_API_KEY=your-anthropic-key-here
   ```

2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements-dev.txt
   # or
   make install-dev
   ```

3. **Environment Setup**: Load environment variables:
   ```bash
   # Copy example environment file
   cp .env.example .env
   # Edit .env with your API keys
   ```

### Quick Start

1. **Basic Comparison** (5 minutes):
   ```bash
   python examples/notebooks/01_basic_multi_model_comparison.py
   ```

2. **Performance Analysis** (10-15 minutes):
   ```bash
   python examples/notebooks/02_performance_analysis.py
   ```

3. **Cost Analysis** (5-10 minutes):
   ```bash
   python examples/use_cases/cost_analysis.py
   ```

4. **Full Benchmarking** (15-30 minutes):
   ```bash
   python examples/use_cases/benchmarking_workflow.py
   ```

## 📊 Example Outputs

### Results Directory Structure

Examples save their outputs to `examples/results/` with organized structure:

```
examples/results/
├── comparison_20241231_143022.json        # Multi-model comparison
├── performance_analysis_20241231_143500.json  # Performance data
├── performance_report_20241231_143500.txt     # Human-readable report
├── cost_analysis_20241231_144000.json         # Cost tracking data
├── cost_report_20241231_144000.txt             # Cost analysis report
├── benchmark_results_20241231_144500.csv       # Benchmark results
├── benchmark_results_20241231_144500.json      # Detailed benchmark data
└── benchmark_report_20241231_144500.txt        # Comprehensive report
```

### Sample Outputs

#### Multi-Model Comparison Results
```json
{
  "prompt": "Explain machine learning in simple terms.",
  "timestamp": "2024-12-31T14:30:22",
  "results": {
    "google": {
      "response": "Machine learning is a type of artificial intelligence...",
      "response_time": 1.23,
      "success": true
    },
    "openai": {
      "response": "Machine learning is a method of data analysis...",
      "response_time": 0.89,
      "success": true
    }
  },
  "analysis": {
    "fastest_provider": "openai",
    "average_response_time": 1.06
  }
}
```

#### Performance Analysis Summary
```
📊 PERFORMANCE ANALYSIS REPORT
================================
Provider      Success   Avg Score  Avg Time
google        100.0%    0.85       1.23s
openai        100.0%    0.82       0.89s
anthropic     100.0%    0.88       1.45s

🏆 BEST PERFORMERS
Highest accuracy: anthropic
Fastest response: openai
Most reliable: google
```

#### Cost Analysis Report
```
💰 COST ANALYSIS REPORT
=======================
Daily budget: $5.00
Total spent today: $0.0127
Remaining budget: $4.9873
Budget utilization: 0.3%

💡 RECOMMENDATIONS
Most cost-effective provider: openai
Estimated cost per 1000 requests: $3.45
```

## 🛠️ Customization Guide

### Creating Custom Examples

1. **Copy a Template**:
   ```bash
   cp examples/notebooks/01_basic_multi_model_comparison.py my_custom_example.py
   ```

2. **Modify for Your Use Case**:
   - Update prompts and test data
   - Add custom evaluation metrics
   - Customize output formats
   - Add domain-specific analysis

3. **Test Your Example**:
   ```bash
   python my_custom_example.py
   ```

### Adding New Providers

To test additional providers in examples:

1. **Install Provider SDK**:
   ```bash
   pip install your-provider-sdk
   ```

2. **Create Provider Wrapper**:
   ```python
   from llm_providers.base import LLMProvider
   
   class YourProvider(LLMProvider):
       def __init__(self, model_name: str, **kwargs):
           # Initialize your provider
           pass
       
       def generate(self, prompt: str) -> str:
           # Implement generation logic
           pass
   ```

3. **Add to Examples**:
   ```python
   providers['your_provider'] = YourProvider(model_name="your-model")
   ```

### Custom Evaluation Metrics

Add domain-specific evaluation beyond keyword matching:

```python
def custom_evaluate(prompt: str, response: str, expected: dict) -> dict:
    """Custom evaluation logic for your domain."""
    # Implement your evaluation criteria
    return {
        'score': 0.85,
        'details': {'custom_metric': 'value'},
        'success': True
    }
```

## 🔧 Advanced Usage

### Batch Processing

Process multiple prompts efficiently:

```python
import asyncio
from examples.notebooks.01_basic_multi_model_comparison import setup_providers

async def batch_process(prompts):
    providers = setup_providers()
    
    async def process_prompt(prompt):
        results = {}
        for name, provider in providers.items():
            try:
                response = await provider.generate_async(prompt)
                results[name] = response
            except Exception as e:
                results[name] = f"Error: {e}"
        return results
    
    tasks = [process_prompt(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)
```

### Integration with CI/CD

Run examples as part of continuous integration:

```yaml
# .github/workflows/examples.yml
name: Examples Test
on: [push, pull_request]

jobs:
  test-examples:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      - name: Run examples
        env:
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          python examples/notebooks/01_basic_multi_model_comparison.py
          python examples/use_cases/cost_analysis.py
```

## 📈 Performance Tips

### Rate Limiting

Examples include built-in rate limiting, but you can customize:

```python
import time

class RateLimitedProvider:
    def __init__(self, provider, requests_per_minute=10):
        self.provider = provider
        self.min_interval = 60 / requests_per_minute
        self.last_request = 0
    
    def generate(self, prompt):
        # Enforce rate limiting
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        
        self.last_request = time.time()
        return self.provider.generate(prompt)
```

### Memory Optimization

For large-scale processing:

```python
import gc
from typing import Iterator

def process_in_batches(data: list, batch_size: int = 10) -> Iterator:
    """Process data in batches to manage memory usage."""
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        yield batch
        gc.collect()  # Force garbage collection between batches
```

### Error Recovery

Robust error handling for production use:

```python
import logging
from typing import Optional

def robust_generate(provider, prompt: str, max_retries: int = 3) -> Optional[str]:
    """Generate with comprehensive error handling."""
    for attempt in range(max_retries):
        try:
            return provider.generate(prompt)
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logging.error(f"All attempts failed for prompt: {prompt[:50]}...")
                raise
```

## 🤝 Contributing Examples

We welcome contributions of new examples and use cases!

### Guidelines

1. **Follow the Template Structure**:
   - Clear docstring with purpose and usage
   - Environment setup and validation
   - Progress tracking and user feedback
   - Error handling and recovery
   - Results export and reporting

2. **Include Documentation**:
   - Purpose and business value
   - Prerequisites and setup
   - Expected outputs
   - Customization options

3. **Test Thoroughly**:
   - Test with different provider combinations
   - Verify error handling scenarios
   - Validate output formats
   - Check performance with various inputs

4. **Submit a Pull Request**:
   - Add your example to the appropriate directory
   - Update this README with a description
   - Include sample outputs
   - Test with the CI pipeline

### Example Submission Template

```python
#!/usr/bin/env python3
"""
Your Example Title

Brief description of what this example demonstrates and its business value.

Usage:
    python examples/category/your_example.py

Requirements:
    - List specific requirements
    - API keys needed
    - Special dependencies

Features:
    - Feature 1
    - Feature 2
    - Feature 3
"""

# Implementation here...

if __name__ == "__main__":
    exit(main())
```

## 📞 Support

If you have questions about the examples or need help customizing them:

1. **Check the Documentation**: Review the provider-specific docs in `docs/providers/`
2. **Review Test Cases**: Look at the test suite for additional usage patterns
3. **Check Issues**: Search existing GitHub issues for similar questions
4. **Create an Issue**: Submit a detailed question with your use case

## 🗺️ Roadmap

Planned additions to the examples collection:

- **Domain-Specific Examples**:
  - Content generation workflows
  - Code analysis and review
  - Data extraction and summarization
  - Multi-language support

- **Advanced Use Cases**:
  - A/B testing frameworks
  - Model ensemble techniques
  - Streaming response handling
  - Fine-tuning integration

- **Integration Examples**:
  - Database integration
  - Web API development
  - Microservices architecture
  - Event-driven processing

- **Production Patterns**:
  - Monitoring and alerting
  - Caching strategies
  - Load balancing
  - Deployment automation

---

These examples provide a solid foundation for building production-ready LLM applications with the LLM Lab framework. Use them as starting points for your own implementations and contribute back improvements for the community!