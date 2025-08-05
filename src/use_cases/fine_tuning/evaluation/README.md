# Fine-Tuning Evaluation Suite

## Overview

The Fine-Tuning Evaluation Suite provides comprehensive assessment tools for evaluating model performance before and after fine-tuning. It includes support for standard benchmarks, custom metrics, and recipe-based evaluation functions.

## Features

### Core Capabilities

- **Standard Benchmarks**: Support for GLUE, SuperGLUE, MMLU, HellaSwag, and more
- **Custom Metrics**: BLEU, ROUGE, BERTScore, perplexity, and task-specific metrics
- **Before/After Comparison**: Compare base and fine-tuned model performance
- **Regression Detection**: Identify performance degradation between checkpoints
- **Recipe Integration**: Custom evaluation functions defined in training recipes
- **Report Generation**: Detailed evaluation reports in JSON and Markdown formats

### Supported Benchmarks

#### Language Understanding
- **GLUE Tasks**: CoLA, SST-2, MRPC, QQP, MNLI, QNLI, RTE, WNLI
- **SuperGLUE Tasks**: BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC
- **Other**: MMLU, HellaSwag, WinoGrande, ARC, LAMBADA, PIQA, SIQA, OpenBookQA

#### Generation Tasks
- **Summarization**: XSum, CNN/DailyMail
- **Question Answering**: SQuAD, SQuAD v2
- **Code Generation**: HumanEval, MBPP

## Usage

### Basic Evaluation

```python
from src.use_cases.fine_tuning.evaluation import EvaluationSuite, EvaluationConfig

# Configure evaluation
config = EvaluationConfig(
    benchmarks=["hellaswag", "mmlu", "winogrande"],
    custom_metrics=["perplexity"],
    batch_size=8,
    save_results=True,
    output_dir="evaluation_results"
)

# Initialize suite
suite = EvaluationSuite(config)

# Run evaluation
results = suite.evaluate(model, tokenizer)

# Print results
print(f"Perplexity: {results.perplexity:.2f}")
for benchmark in results.benchmarks:
    print(f"{benchmark.name}: {benchmark.overall_score:.4f}")
```

### Model Comparison

```python
# Compare base and fine-tuned models
comparison = suite.compare_models(
    base_model=base_model,
    fine_tuned_model=ft_model,
    tokenizer=tokenizer,
    benchmarks=["hellaswag", "mmlu"]
)

# Detect regressions
regressions = suite.detect_regressions(
    current_results=ft_results,
    baseline_results=base_results,
    threshold=0.05  # 5% degradation threshold
)
```

### Recipe-Based Evaluation

```python
# Define recipe with custom evaluation
recipe = {
    "name": "instruction_tuning",
    "evaluation": {
        "benchmarks": ["hellaswag"],
        "custom_eval_function": "instruction_following",
        "eval_function_config": {
            "test_instructions": [
                "Explain machine learning",
                "Write a Python function"
            ]
        }
    }
}

# Run evaluation with recipe
results = suite.evaluate_with_recipe(
    model=model,
    tokenizer=tokenizer,
    recipe=recipe
)
```

## Custom Evaluation Functions

### Available Functions

1. **instruction_following**: Evaluates instruction-following capabilities
2. **code_generation**: Tests code generation and syntax validity
3. **domain_specific_medical**: Medical domain knowledge assessment
4. **chat_coherence**: Multi-turn conversation coherence
5. **summarization_quality**: Text summarization evaluation

### Creating Custom Evaluations

```python
from src.use_cases.fine_tuning.evaluation import CustomEvaluationRegistry

@CustomEvaluationRegistry.register("my_custom_eval")
def evaluate_custom(model, tokenizer, config=None):
    """Custom evaluation function."""
    results = []
    
    # Your evaluation logic here
    test_prompts = config.get("prompts", ["Test prompt"])
    
    for prompt in test_prompts:
        # Generate and evaluate
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs)
        response = tokenizer.decode(outputs[0])
        
        # Calculate metrics
        score = calculate_custom_metric(response)
        
        results.append({
            "name": f"custom_{prompt[:20]}",
            "value": score,
            "metadata": {"prompt": prompt}
        })
    
    return results
```

### Using in Recipes

```yaml
# recipe.yaml
name: custom_training
evaluation:
  benchmarks:
    - hellaswag
    - mmlu
  custom_eval_function: my_custom_eval
  eval_function_config:
    prompts:
      - "Test prompt 1"
      - "Test prompt 2"
    min_score: 0.8
```

## Integration with Training Pipeline

```python
from src.use_cases.fine_tuning.recipes import RecipeManager
from src.use_cases.fine_tuning.evaluation import EvaluationSuite

# Load recipe
manager = RecipeManager()
recipe = manager.load_recipe("instruction_tuning")

# Train model
model = train_with_recipe(recipe)

# Evaluate with recipe settings
suite = EvaluationSuite()
results = suite.evaluate_with_recipe(
    model=model,
    tokenizer=tokenizer,
    recipe=recipe.to_dict()
)

# Generate report
report = suite.generate_report(results, output_format="markdown")
print(report)
```

## Configuration Options

### EvaluationConfig

```python
EvaluationConfig(
    benchmarks=["hellaswag", "mmlu"],  # Benchmarks to run
    custom_metrics=["perplexity"],      # Additional metrics
    batch_size=8,                        # Evaluation batch size
    max_samples=1000,                    # Max samples per benchmark
    device="cuda",                       # Device for evaluation
    seed=42,                             # Random seed
    use_cache=True,                      # Cache evaluation results
    save_results=True,                   # Save results to disk
    output_dir="results",                # Output directory
    regression_threshold=0.05            # Regression detection threshold
)
```

## Performance Considerations

### Memory Optimization
- Use smaller batch sizes for large models
- Enable gradient checkpointing for memory-constrained environments
- Use FP16/BF16 precision when appropriate

### Speed Optimization
- Enable result caching to avoid re-evaluation
- Use parallel evaluation for multiple benchmarks
- Leverage hardware acceleration (GPU/TPU)

### Best Practices
1. Always evaluate base model first for comparison
2. Use consistent seeds for reproducible results
3. Monitor memory usage during evaluation
4. Save intermediate results for long evaluations
5. Use appropriate benchmarks for your use case

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch_size or max_samples
2. **Slow Evaluation**: Enable caching, reduce benchmark count
3. **Import Errors**: Install required dependencies (evaluate, lm-eval)
4. **Metric Failures**: Check tokenizer compatibility

### Dependencies

Required packages:
- transformers >= 4.30.0
- torch >= 2.0.0
- evaluate >= 0.4.0
- datasets >= 2.10.0
- numpy
- pandas
- tqdm

Optional packages:
- lm-eval (for additional benchmarks)
- bert-score (for BERTScore metric)
- rouge-score (for ROUGE metrics)

## Examples

See `examples/evaluation/` for complete examples:
- `basic_evaluation.py`: Simple model evaluation
- `recipe_evaluation.py`: Recipe-based evaluation
- `model_comparison.py`: Compare multiple models
- `custom_metrics.py`: Adding custom metrics