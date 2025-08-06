# Evaluation API Reference

## Overview

The evaluation module provides comprehensive tools for evaluating LLM responses across various metrics and benchmarks.

## Base Evaluator

### `BaseEvaluator`

Abstract base class for all evaluators.

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseEvaluator(ABC):
    """Base class for all evaluators."""

    @abstractmethod
    def evaluate(
        self,
        response: str,
        expected: Any,
        **kwargs
    ) -> Dict[str, float]:
        """Evaluate a response against expected output.

        Args:
            response: Model response
            expected: Expected output or ground truth
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary of metric names to scores
        """
        pass
```

## Built-in Evaluators

### `TruthfulnessEvaluator`

Evaluates factual accuracy and truthfulness.

```python
from src.evaluation import TruthfulnessEvaluator

evaluator = TruthfulnessEvaluator()

# Simple evaluation
result = evaluator.evaluate(
    response="Paris is the capital of France",
    expected=["Paris", "capital", "France"]
)
# result: {"accuracy": 1.0, "keyword_match": 1.0}

# With confidence scores
result = evaluator.evaluate_with_confidence(
    response="The Earth orbits the Sun",
    expected="heliocentric model",
    confidence_threshold=0.8
)
# result: {"accuracy": 0.95, "confidence": 0.92}
```

### `SafetyEvaluator`

Evaluates response safety and appropriateness.

```python
from src.evaluation import SafetyEvaluator

evaluator = SafetyEvaluator(
    filters=["toxicity", "bias", "personal_info"]
)

result = evaluator.evaluate(
    response="I can help you with that math problem",
    context="homework help"
)
# result: {"safety_score": 1.0, "flags": []}
```

### `PerformanceEvaluator`

Evaluates response performance metrics.

```python
from src.evaluation import PerformanceEvaluator

evaluator = PerformanceEvaluator()

result = evaluator.evaluate(
    response=response_object,
    metrics=["latency", "tokens_per_second", "time_to_first_token"]
)
# result: {
#     "latency": 1.23,
#     "tokens_per_second": 45.6,
#     "time_to_first_token": 0.15
# }
```

## Metric Implementations

### `ExactMatchMetric`

```python
from src.evaluation.metrics import ExactMatchMetric

metric = ExactMatchMetric(case_sensitive=False)
score = metric.compute("Hello World", "hello world")
# score: 1.0
```

### `BLEUMetric`

```python
from src.evaluation.metrics import BLEUMetric

metric = BLEUMetric(n_gram=4)
score = metric.compute(
    hypothesis="The cat sat on the mat",
    reference="The cat is sitting on the mat"
)
# score: 0.65
```

### `RougeMetric`

```python
from src.evaluation.metrics import RougeMetric

metric = RougeMetric(rouge_types=["rouge1", "rouge2", "rougeL"])
scores = metric.compute(
    summary="AI is transforming technology",
    reference="Artificial intelligence is revolutionizing tech"
)
# scores: {"rouge1": 0.5, "rouge2": 0.0, "rougeL": 0.33}
```

## Benchmark Datasets

### `DatasetLoader`

```python
from src.evaluation.datasets import DatasetLoader

# Load built-in dataset
loader = DatasetLoader()
dataset = loader.load("truthfulqa", split="test", sample_size=100)

# Load custom dataset
custom_dataset = loader.load_custom(
    path="./my_dataset.jsonl",
    format="jsonl",
    fields={"prompt": "question", "expected": "answer"}
)
```

### Supported Datasets

- **TruthfulQA**: Factual accuracy evaluation
- **GSM8K**: Mathematical reasoning
- **HumanEval**: Code generation
- **MMLU**: Multi-domain knowledge
- **ARC**: Science reasoning
- **HellaSwag**: Common sense reasoning

## Evaluation Pipelines

### `EvaluationPipeline`

Run complete evaluation workflows.

```python
from src.evaluation import EvaluationPipeline

# Create pipeline
pipeline = EvaluationPipeline(
    evaluators=[
        TruthfulnessEvaluator(),
        SafetyEvaluator(),
        PerformanceEvaluator()
    ]
)

# Run evaluation
results = pipeline.run(
    provider=openai_provider,
    dataset="truthfulqa",
    sample_size=100,
    output_dir="./results"
)

# Get summary
summary = results.summary()
print(f"Overall score: {summary['overall_score']:.2f}")
```

### Multi-Provider Evaluation

```python
# Evaluate multiple providers
providers = [openai_provider, anthropic_provider, google_provider]

results = pipeline.run_multi_provider(
    providers=providers,
    dataset="truthfulqa",
    parallel=True
)

# Compare results
comparison = results.compare_providers()
print(comparison.to_dataframe())
```

## Custom Evaluators

### Creating Custom Evaluators

```python
from src.evaluation import BaseEvaluator

class DomainSpecificEvaluator(BaseEvaluator):
    """Custom evaluator for domain-specific content."""

    def __init__(self, domain_keywords: List[str]):
        self.domain_keywords = domain_keywords

    def evaluate(
        self,
        response: str,
        expected: Any,
        **kwargs
    ) -> Dict[str, float]:
        # Count domain keyword matches
        matches = sum(
            1 for keyword in self.domain_keywords
            if keyword.lower() in response.lower()
        )

        relevance = matches / len(self.domain_keywords)

        return {
            "domain_relevance": relevance,
            "keyword_coverage": matches
        }

# Use custom evaluator
evaluator = DomainSpecificEvaluator(
    domain_keywords=["medical", "diagnosis", "treatment", "patient"]
)
```

### Composite Evaluators

```python
class CompositeEvaluator(BaseEvaluator):
    """Combine multiple evaluators."""

    def __init__(self, evaluators: List[BaseEvaluator], weights: Dict[str, float]):
        self.evaluators = evaluators
        self.weights = weights

    def evaluate(self, response: str, expected: Any, **kwargs) -> Dict[str, float]:
        all_scores = {}

        for evaluator in self.evaluators:
            scores = evaluator.evaluate(response, expected, **kwargs)
            all_scores.update(scores)

        # Calculate weighted average
        weighted_score = sum(
            score * self.weights.get(metric, 1.0)
            for metric, score in all_scores.items()
        ) / len(all_scores)

        all_scores["weighted_average"] = weighted_score
        return all_scores
```

## Statistical Analysis

### `StatisticalAnalyzer`

```python
from src.evaluation.analysis import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

# Analyze results
stats = analyzer.analyze(evaluation_results)
print(f"Mean score: {stats['mean']:.3f}")
print(f"Std deviation: {stats['std']:.3f}")
print(f"95% CI: [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]")

# Compare two models
comparison = analyzer.compare_models(
    results_a=model_a_results,
    results_b=model_b_results,
    test="wilcoxon"  # or "ttest"
)
print(f"P-value: {comparison['p_value']:.4f}")
print(f"Significant: {comparison['is_significant']}")
```

## Evaluation Configuration

### YAML Configuration

```yaml
evaluation:
  pipeline:
    - evaluator: truthfulness
      config:
        threshold: 0.8
        strict_mode: true

    - evaluator: safety
      config:
        filters:
          - toxicity
          - bias
          - pii

    - evaluator: performance
      config:
        metrics:
          - latency
          - throughput

  dataset:
    name: truthfulqa
    split: test
    sample_size: 500

  output:
    format: json
    include_raw_responses: true
    save_plots: true
```

## Best Practices

1. **Use Appropriate Metrics**
   ```python
   # Choose metrics based on task
   if task_type == "generation":
       metrics = ["bleu", "rouge", "perplexity"]
   elif task_type == "classification":
       metrics = ["accuracy", "f1", "precision", "recall"]
   ```

2. **Handle Edge Cases**
   ```python
   def safe_evaluate(response, expected):
       if not response or not expected:
           return {"score": 0.0, "error": "Empty input"}

       try:
           return evaluator.evaluate(response, expected)
       except Exception as e:
           return {"score": 0.0, "error": str(e)}
   ```

3. **Statistical Significance**
   ```python
   # Always check statistical significance
   if results.sample_size < 30:
       logger.warning("Small sample size, results may not be significant")

   # Use bootstrap for confidence intervals
   ci = analyzer.bootstrap_ci(results, n_iterations=1000)
   ```

4. **Cache Evaluations**
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=1000)
   def cached_evaluate(response: str, expected: str) -> float:
       return evaluator.evaluate(response, expected)["score"]
   ```

## See Also

- [Benchmarking Guide](../guides/USE_CASE_1_HOW_TO.md) - Running benchmarks
- [Custom Metrics Guide](../guides/CUSTOM_EVALUATION_METRICS.md) - Creating metrics
- [Statistical Analysis](analysis.md) - Advanced analysis tools
