# Custom Evaluation Metrics Guide

*Comprehensive guide to creating, implementing, and using custom evaluation metrics for domain-specific LLM evaluation.*

## 🎯 Overview

Custom evaluation metrics allow you to assess LLM outputs based on your specific requirements beyond standard metrics like length and sentiment. This guide covers:

- **Metric Design Principles**: How to design effective evaluation criteria
- **Implementation Patterns**: Code examples for common metric types
- **Integration**: Connecting metrics with the benchmark framework
- **Visualization**: Creating charts and reports for metric analysis
- **Domain-Specific Examples**: Customer service, code quality, creativity metrics
- **Advanced Techniques**: Weighted scoring, threshold configuration, comparative analysis

## 📋 Metric Architecture

### Core Components

The custom metrics system consists of several key components:

```python
# Core metric interface
class CustomMetric:
    def __init__(self, name: str, evaluator_function: Callable)
    def evaluate(self, response: str, **kwargs) -> Dict[str, Any]
    def aggregate(self, results: List[Dict]) -> Dict[str, Any]

# Metric suite for managing multiple metrics
class MetricSuite:
    def add_metric(self, metric: CustomMetric)
    def evaluate(self, response: str, **kwargs) -> Dict[str, Any]
    def compare_results(self, results: List[Dict]) -> Dict[str, Any]

# Integration with benchmark framework
def save_execution_result(result, metrics: MetricSuite = None)
```

### Metric Types

| Type | Purpose | Example Use Cases | Complexity |
|------|---------|-------------------|------------|
| **Binary** | Pass/fail evaluation | Syntax validity, policy compliance | Low |
| **Scalar** | Numeric scoring (0-1 or 0-100) | Quality scores, similarity measures | Medium |
| **Categorical** | Classification into groups | Tone classification, content type | Medium |
| **Composite** | Multiple sub-metrics combined | Overall quality, multi-factor assessment | High |
| **Comparative** | Ranking between responses | Best response selection, A/B testing | High |

## 🏗️ Creating Custom Metrics

### 1. Basic Metric Implementation

Here's the structure for implementing a custom metric:

```python
# src/use_cases/custom_prompts/custom_metrics.py

from typing import Dict, Any, List, Callable
import re
import statistics
from dataclasses import dataclass

@dataclass
class CustomMetric:
    """Base class for custom evaluation metrics."""
    
    name: str
    evaluator_function: Callable
    description: str = ""
    metric_type: str = "scalar"  # binary, scalar, categorical, composite
    
    def evaluate(self, response: str, **kwargs) -> Dict[str, Any]:
        """Evaluate a single response."""
        try:
            result = self.evaluator_function(response, **kwargs)
            return {
                "metric_name": self.name,
                "metric_type": self.metric_type,
                "value": result,
                "success": True,
                "error": None
            }
        except Exception as e:
            return {
                "metric_name": self.name,
                "metric_type": self.metric_type,
                "value": None,
                "success": False,
                "error": str(e)
            }
    
    def aggregate(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate multiple evaluation results."""
        successful_results = [r for r in results if r.get("success", False)]
        
        if not successful_results:
            return {"mean": None, "std": None, "count": 0}
        
        values = [r["value"] for r in successful_results]
        
        if self.metric_type == "scalar":
            return {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
        elif self.metric_type == "binary":
            true_count = sum(1 for v in values if v)
            return {
                "pass_rate": true_count / len(values),
                "pass_count": true_count,
                "total_count": len(values)
            }
        elif self.metric_type == "categorical":
            from collections import Counter
            counts = Counter(values)
            return {
                "distribution": dict(counts),
                "most_common": counts.most_common(1)[0] if counts else None,
                "unique_count": len(counts),
                "total_count": len(values)
            }
        else:
            return {"values": values, "count": len(values)}
```

### 2. Domain-Specific Metric Examples

#### Customer Service Quality Metric

```python
def customer_service_quality_evaluator(response: str, **kwargs) -> Dict[str, float]:
    """Evaluate customer service response quality."""
    
    # Extract parameters
    customer_issue = kwargs.get("customer_issue", "")
    company_policies = kwargs.get("company_policies", [])
    
    scores = {}
    
    # 1. Empathy Score (0-1)
    empathy_keywords = [
        "understand", "sorry", "apologize", "frustrating", "concerned",
        "appreciate", "thank you", "help", "assist", "support"
    ]
    empathy_count = sum(1 for word in empathy_keywords if word.lower() in response.lower())
    scores["empathy"] = min(empathy_count / 3.0, 1.0)  # Normalize to 0-1
    
    # 2. Professionalism Score (0-1)
    # Check for professional language, proper grammar
    professional_indicators = [
        len(re.findall(r'[.!?]', response)) > 0,  # Proper punctuation
        not bool(re.search(r'[A-Z]{3,}', response)),  # No excessive caps
        len(response.split()) >= 20,  # Adequate length
        not any(word in response.lower() for word in ["dunno", "gonna", "wanna"])  # Formal language
    ]
    scores["professionalism"] = sum(professional_indicators) / len(professional_indicators)
    
    # 3. Solution Orientation (0-1)
    solution_keywords = [
        "will", "can", "next steps", "resolution", "solve", "fix",
        "contact", "follow up", "schedule", "escalate", "resolve"
    ]
    solution_count = sum(1 for word in solution_keywords if word.lower() in response.lower())
    scores["solution_orientation"] = min(solution_count / 2.0, 1.0)
    
    # 4. Policy Compliance (0-1)
    if company_policies:
        policy_mentions = sum(1 for policy in company_policies 
                            if policy.lower() in response.lower())
        scores["policy_compliance"] = min(policy_mentions / len(company_policies), 1.0)
    else:
        scores["policy_compliance"] = 1.0  # Default if no policies specified
    
    # 5. Overall Quality (weighted average)
    weights = {"empathy": 0.25, "professionalism": 0.30, "solution_orientation": 0.35, "policy_compliance": 0.10}
    overall_score = sum(scores[key] * weights[key] for key in weights)
    scores["overall"] = overall_score
    
    return scores

# Create the metric
customer_service_metric = CustomMetric(
    name="customer_service_quality",
    evaluator_function=customer_service_quality_evaluator,
    description="Evaluates customer service responses on empathy, professionalism, solution orientation, and policy compliance",
    metric_type="composite"
)
```

#### Code Quality Metric

```python
import ast
import subprocess
import tempfile
import os

def code_quality_evaluator(response: str, **kwargs) -> Dict[str, float]:
    """Evaluate generated code quality."""
    
    language = kwargs.get("language", "python")
    
    if language.lower() != "python":
        return {"error": "Only Python code evaluation supported"}
    
    scores = {}
    
    # Extract code blocks from response
    code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
    if not code_blocks:
        # Try to extract code without markdown
        code_blocks = [response]
    
    code = "\n".join(code_blocks)
    
    # 1. Syntax Validity (0-1)
    try:
        ast.parse(code)
        scores["syntax_valid"] = 1.0
    except SyntaxError:
        scores["syntax_valid"] = 0.0
        return scores  # Return early if syntax invalid
    
    # 2. Code Structure Score (0-1)
    tree = ast.parse(code)
    
    # Check for functions/classes
    has_functions = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
    has_docstrings = any(isinstance(node, ast.Str) for node in ast.walk(tree))
    has_type_hints = any(hasattr(node, 'annotation') and node.annotation 
                        for node in ast.walk(tree) if hasattr(node, 'annotation'))
    
    structure_indicators = [
        has_functions,
        has_docstrings,
        has_type_hints,
        len(code.split('\n')) > 5  # Reasonable length
    ]
    scores["code_structure"] = sum(structure_indicators) / len(structure_indicators)
    
    # 3. Best Practices Score (0-1)
    best_practices = [
        not re.search(r'print\(', code),  # No debug prints
        'import' in code or 'from' in code,  # Proper imports
        not re.search(r'[a-z]{20,}', code),  # No excessively long variable names
        re.search(r'def \w+\(', code),  # Has function definitions
    ]
    scores["best_practices"] = sum(best_practices) / len(best_practices)
    
    # 4. Complexity Analysis (0-1, lower complexity is better)
    try:
        # Count control flow statements
        control_flow_count = len([node for node in ast.walk(tree) 
                                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try))])
        # Normalize complexity (assumes reasonable complexity < 10)
        scores["complexity"] = max(0, 1 - (control_flow_count / 10))
    except:
        scores["complexity"] = 0.5  # Default if analysis fails
    
    # 5. Overall Code Quality
    weights = {"syntax_valid": 0.4, "code_structure": 0.25, "best_practices": 0.25, "complexity": 0.1}
    overall_score = sum(scores[key] * weights[key] for key in weights if key in scores)
    scores["overall"] = overall_score
    
    return scores

# Create the metric
code_quality_metric = CustomMetric(
    name="code_quality",
    evaluator_function=code_quality_evaluator,
    description="Evaluates generated code on syntax validity, structure, best practices, and complexity",
    metric_type="composite"
)
```

#### Creative Writing Metric

```python
from textstat import flesch_reading_ease, flesch_kincaid_grade
import nltk
from collections import Counter

def creativity_evaluator(response: str, **kwargs) -> Dict[str, float]:
    """Evaluate creativity in written content."""
    
    genre = kwargs.get("genre", "general")
    target_audience = kwargs.get("target_audience", "adult")
    
    scores = {}
    
    # 1. Vocabulary Diversity (0-1)
    words = response.lower().split()
    unique_words = set(words)
    if len(words) > 0:
        scores["vocabulary_diversity"] = len(unique_words) / len(words)
    else:
        scores["vocabulary_diversity"] = 0
    
    # 2. Sentence Structure Variety (0-1)
    sentences = re.split(r'[.!?]+', response)
    sentence_lengths = [len(sent.split()) for sent in sentences if sent.strip()]
    
    if sentence_lengths:
        length_variety = statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0
        # Normalize variety score (assumes good variety has stdev around 5-10)
        scores["sentence_variety"] = min(length_variety / 10, 1.0)
    else:
        scores["sentence_variety"] = 0
    
    # 3. Imagery and Descriptive Language (0-1)
    descriptive_patterns = [
        r'\b\w+ly\b',  # Adverbs
        r'\b(bright|dark|soft|rough|smooth|loud|quiet|warm|cold|sweet|bitter)\b',  # Sensory adjectives
        r'\b(like|as)\s+\w+',  # Similes
        r'\b(sparkled|glowed|whispered|thundered|danced|crept)\b'  # Vivid verbs
    ]
    
    descriptive_count = sum(len(re.findall(pattern, response, re.IGNORECASE)) 
                          for pattern in descriptive_patterns)
    word_count = len(words)
    scores["imagery"] = min(descriptive_count / (word_count * 0.05), 1.0) if word_count > 0 else 0
    
    # 4. Originality (0-1) - Based on uncommon word usage
    # This is a simplified approach - in practice, you might use more sophisticated NLP
    common_words = set([
        "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "a", "an", "is", "was", "are", "were", "have", "has", "had", "will", "would",
        "very", "really", "quite", "just", "only", "also", "much", "many", "some"
    ])
    
    uncommon_words = [word for word in unique_words if word not in common_words and len(word) > 3]
    scores["originality"] = min(len(uncommon_words) / (len(unique_words) * 0.3), 1.0) if unique_words else 0
    
    # 5. Readability Appropriateness (0-1)
    try:
        reading_ease = flesch_reading_ease(response)
        # Score based on target audience
        if target_audience == "child":
            target_ease = 90  # Very easy
        elif target_audience == "young_adult":
            target_ease = 70  # Fairly easy
        else:
            target_ease = 60  # Standard
        
        # Score based on how close to target readability
        ease_diff = abs(reading_ease - target_ease)
        scores["readability_fit"] = max(0, 1 - (ease_diff / 50))
    except:
        scores["readability_fit"] = 0.5  # Default if calculation fails
    
    # 6. Overall Creativity Score
    weights = {
        "vocabulary_diversity": 0.25,
        "sentence_variety": 0.20,
        "imagery": 0.25,
        "originality": 0.20,
        "readability_fit": 0.10
    }
    overall_score = sum(scores[key] * weights[key] for key in weights if key in scores)
    scores["overall"] = overall_score
    
    return scores

# Create the metric
creativity_metric = CustomMetric(
    name="creativity",
    evaluator_function=creativity_evaluator,
    description="Evaluates creative writing on vocabulary diversity, sentence variety, imagery, originality, and readability",
    metric_type="composite"
)
```

## 🔧 Integration with Benchmark Framework

### 1. Adding Metrics to Evaluation Suite

```python
# src/use_cases/custom_prompts/evaluation_metrics.py

class MetricSuite:
    """Suite for managing multiple custom metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.default_metrics_enabled = True
    
    def add_metric(self, metric: CustomMetric):
        """Add a custom metric to the suite."""
        self.metrics[metric.name] = metric
    
    def remove_metric(self, metric_name: str):
        """Remove a metric from the suite."""
        if metric_name in self.metrics:
            del self.metrics[metric_name]
    
    def list_metrics(self) -> List[str]:
        """List all available metrics."""
        base_metrics = ["response_length", "sentiment", "coherence"] if self.default_metrics_enabled else []
        return base_metrics + list(self.metrics.keys())
    
    def evaluate(self, response: str, **kwargs) -> Dict[str, Any]:
        """Evaluate response with all metrics."""
        results = {}
        
        # Run default metrics if enabled
        if self.default_metrics_enabled:
            # Call existing metric functions
            results.update(self._evaluate_default_metrics(response))
        
        # Run custom metrics
        for metric_name, metric in self.metrics.items():
            try:
                metric_result = metric.evaluate(response, **kwargs)
                results[metric_name] = metric_result
            except Exception as e:
                results[metric_name] = {
                    "success": False,
                    "error": f"Metric evaluation failed: {str(e)}"
                }
        
        return results
    
    def aggregate_results(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across multiple evaluations."""
        aggregated = {}
        
        # Group results by metric
        for metric_name in self.list_metrics():
            metric_results = []
            for result in all_results:
                if metric_name in result:
                    metric_results.append(result[metric_name])
            
            if metric_results and metric_name in self.metrics:
                # Use custom metric aggregation
                aggregated[metric_name] = self.metrics[metric_name].aggregate(metric_results)
            elif metric_results:
                # Use default aggregation
                aggregated[metric_name] = self._default_aggregate(metric_results)
        
        return aggregated
    
    def _evaluate_default_metrics(self, response: str) -> Dict[str, Any]:
        """Evaluate with default metrics."""
        # Implementation of existing metrics
        return {
            "response_length": {"words": len(response.split()), "sentences": len(re.findall(r'[.!?]', response))},
            "sentiment": self._analyze_sentiment(response),
            "coherence": self._analyze_coherence(response)
        }
    
    def _analyze_sentiment(self, response: str) -> Dict[str, Any]:
        """Basic sentiment analysis."""
        # Simplified implementation
        positive_words = ["good", "great", "excellent", "happy", "positive", "wonderful"]
        negative_words = ["bad", "terrible", "awful", "sad", "negative", "horrible"]
        
        pos_count = sum(1 for word in positive_words if word in response.lower())
        neg_count = sum(1 for word in negative_words if word in response.lower())
        
        if pos_count + neg_count == 0:
            return {"score": 0.0, "label": "neutral"}
        
        sentiment_score = (pos_count - neg_count) / (pos_count + neg_count)
        
        if sentiment_score > 0.1:
            label = "positive"
        elif sentiment_score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        return {"score": sentiment_score, "label": label}
    
    def _analyze_coherence(self, response: str) -> Dict[str, Any]:
        """Basic coherence analysis."""
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return {"score": 1.0 if len(sentences) == 1 else 0.0}
        
        # Simple coherence based on sentence length consistency and transition words
        transition_words = ["however", "therefore", "moreover", "furthermore", "consequently", "meanwhile", "additionally"]
        transition_count = sum(1 for word in transition_words if word in response.lower())
        
        # Length consistency
        lengths = [len(s.split()) for s in sentences]
        avg_length = statistics.mean(lengths)
        length_consistency = 1.0 - (statistics.stdev(lengths) / avg_length if avg_length > 0 else 1.0)
        
        # Transition usage
        transition_score = min(transition_count / len(sentences), 1.0)
        
        # Combined coherence score
        coherence_score = (length_consistency * 0.7) + (transition_score * 0.3)
        
        return {"score": min(max(coherence_score, 0.0), 1.0)}
    
    def _default_aggregate(self, results: List[Dict]) -> Dict[str, Any]:
        """Default aggregation for built-in metrics."""
        # Simple implementation for default metrics
        if not results:
            return {}
        
        # For metrics with numeric values
        if all(isinstance(r.get("score"), (int, float)) for r in results):
            scores = [r["score"] for r in results if "score" in r]
            return {
                "mean": statistics.mean(scores),
                "std": statistics.stdev(scores) if len(scores) > 1 else 0,
                "count": len(scores)
            }
        
        return {"count": len(results)}
```

### 2. CLI Integration

```python
# Integration with run_benchmarks.py

def add_custom_metrics_support():
    """Add custom metrics support to CLI."""
    
    # Add argument parser options
    parser.add_argument(
        '--custom-metrics',
        type=str,
        nargs='+',
        help='Enable custom metrics: customer_service, code_quality, creativity, or custom module path'
    )
    
    parser.add_argument(
        '--metric-config',
        type=str,
        help='Path to metric configuration JSON file'
    )
    
    # Load and configure metrics
    def setup_metrics(args):
        suite = MetricSuite()
        
        if hasattr(args, 'custom_metrics') and args.custom_metrics:
            for metric_name in args.custom_metrics:
                if metric_name == 'customer_service':
                    suite.add_metric(customer_service_metric)
                elif metric_name == 'code_quality':
                    suite.add_metric(code_quality_metric)
                elif metric_name == 'creativity':
                    suite.add_metric(creativity_metric)
                else:
                    # Try to load as custom module
                    try:
                        metric = load_custom_metric(metric_name)
                        suite.add_metric(metric)
                    except Exception as e:
                        print(f"Failed to load custom metric {metric_name}: {e}")
        
        return suite

def load_custom_metric(module_path: str) -> CustomMetric:
    """Load custom metric from external module."""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("custom_metric", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Expect module to have 'get_metric()' function
    return module.get_metric()
```

## 📊 Metric Visualization and Reporting

### 1. Visualization Tools

```python
# src/use_cases/custom_prompts/metric_visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any
import numpy as np

class MetricVisualizer:
    """Create visualizations for custom metrics."""
    
    def __init__(self, style: str = "seaborn"):
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_metric_comparison(self, results: Dict[str, Dict], metric_name: str, save_path: str = None):
        """Plot metric comparison across models."""
        
        models = list(results.keys())
        values = []
        
        for model in models:
            if metric_name in results[model]:
                metric_result = results[model][metric_name]
                if "overall" in metric_result.get("value", {}):
                    values.append(metric_result["value"]["overall"])
                elif isinstance(metric_result.get("value"), (int, float)):
                    values.append(metric_result["value"])
                else:
                    values.append(0)
            else:
                values.append(0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(models, values, color=sns.color_palette("husl", len(models)))
        
        ax.set_title(f'{metric_name.replace("_", " ").title()} Comparison', fontsize=16, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add grid for better readability
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_composite_metric_breakdown(self, results: Dict[str, Dict], metric_name: str, save_path: str = None):
        """Plot breakdown of composite metric components."""
        
        models = list(results.keys())
        
        # Extract sub-metric data
        sub_metrics = set()
        data = {}
        
        for model in models:
            if metric_name in results[model]:
                metric_value = results[model][metric_name].get("value", {})
                if isinstance(metric_value, dict):
                    sub_metrics.update(metric_value.keys())
                    data[model] = metric_value
        
        sub_metrics = sorted([m for m in sub_metrics if m != "overall"])
        
        if not sub_metrics:
            print(f"No sub-metrics found for {metric_name}")
            return None
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(models))
        width = 0.15
        
        for i, sub_metric in enumerate(sub_metrics):
            values = [data.get(model, {}).get(sub_metric, 0) for model in models]
            ax.bar(x + i * width, values, width, label=sub_metric.replace('_', ' ').title())
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{metric_name.replace("_", " ").title()} - Component Breakdown', fontsize=16, fontweight='bold')
        ax.set_xticks(x + width * (len(sub_metrics) - 1) / 2)
        ax.set_xticklabels(models)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_matrix(self, results: Dict[str, Dict], save_path: str = None):
        """Plot correlation matrix between different metrics."""
        
        # Prepare data for correlation analysis
        metrics_data = {}
        
        for model, model_results in results.items():
            for metric_name, metric_result in model_results.items():
                if metric_result.get("success", True):
                    value = metric_result.get("value")
                    
                    if isinstance(value, dict) and "overall" in value:
                        metrics_data.setdefault(metric_name, []).append(value["overall"])
                    elif isinstance(value, (int, float)):
                        metrics_data.setdefault(metric_name, []).append(value)
        
        # Create DataFrame
        df = pd.DataFrame(metrics_data)
        
        if df.empty:
            print("No numeric data available for correlation analysis")
            return None
        
        # Calculate correlation matrix
        correlation_matrix = df.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        
        ax.set_title('Metric Correlation Matrix', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_metric_report(self, results: Dict[str, Dict], output_dir: str):
        """Create comprehensive metric visualization report."""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all unique metrics
        all_metrics = set()
        for model_results in results.values():
            all_metrics.update(model_results.keys())
        
        plots_created = []
        
        # Create individual metric comparisons
        for metric in all_metrics:
            try:
                fig = self.plot_metric_comparison(results, metric, 
                                                f"{output_dir}/{metric}_comparison.png")
                if fig:
                    plt.close(fig)
                    plots_created.append(f"{metric}_comparison.png")
            except Exception as e:
                print(f"Failed to create comparison plot for {metric}: {e}")
        
        # Create composite metric breakdowns
        composite_metrics = ["customer_service_quality", "code_quality", "creativity"]
        for metric in composite_metrics:
            if metric in all_metrics:
                try:
                    fig = self.plot_composite_metric_breakdown(results, metric,
                                                             f"{output_dir}/{metric}_breakdown.png")
                    if fig:
                        plt.close(fig)
                        plots_created.append(f"{metric}_breakdown.png")
                except Exception as e:
                    print(f"Failed to create breakdown plot for {metric}: {e}")
        
        # Create correlation matrix
        try:
            fig = self.plot_correlation_matrix(results, f"{output_dir}/correlation_matrix.png")
            if fig:
                plt.close(fig)
                plots_created.append("correlation_matrix.png")
        except Exception as e:
            print(f"Failed to create correlation matrix: {e}")
        
        return plots_created
```

### 2. Reporting Tools

```python
# src/use_cases/custom_prompts/metric_reports.py

from jinja2 import Template
import json
from datetime import datetime
from pathlib import Path

class MetricReporter:
    """Generate comprehensive metric reports."""
    
    def __init__(self):
        self.report_template = """
# Custom Metrics Evaluation Report

**Generated:** {{ timestamp }}
**Models Evaluated:** {{ models|join(', ') }}
**Total Metrics:** {{ total_metrics }}

## Executive Summary

{% for metric_name, summary in metric_summaries.items() %}
### {{ metric_name.replace('_', ' ').title() }}

{% if summary.type == 'composite' %}
**Overall Performance:**
{% for model, score in summary.model_scores.items() %}
- **{{ model }}**: {{ "%.3f"|format(score) }}
{% endfor %}

**Component Analysis:**
{% for component, scores in summary.components.items() %}
- **{{ component.replace('_', ' ').title() }}**: Avg {{ "%.3f"|format(scores.mean) }} (σ={{ "%.3f"|format(scores.std) }})
{% endfor %}

{% else %}
**Model Performance:**
{% for model, score in summary.model_scores.items() %}
- **{{ model }}**: {{ score }}
{% endfor %}
{% endif %}

---
{% endfor %}

## Detailed Analysis

{% for model in models %}
### {{ model }} Performance

| Metric | Score | Interpretation |
|--------|-------|----------------|
{% for metric_name, result in detailed_results[model].items() %}
{% if result.success %}
| {{ metric_name.replace('_', ' ').title() }} | {{ format_score(result.value) }} | {{ interpret_score(metric_name, result.value) }} |
{% endif %}
{% endfor %}

{% endfor %}

## Recommendations

Based on the evaluation results:

{% for recommendation in recommendations %}
- {{ recommendation }}
{% endfor %}

## Methodology

This report was generated using custom evaluation metrics designed for domain-specific assessment. Each metric uses weighted scoring algorithms to provide comprehensive evaluation of model outputs.

**Metric Descriptions:**
{% for metric_name, description in metric_descriptions.items() %}
- **{{ metric_name.replace('_', ' ').title() }}**: {{ description }}
{% endfor %}

---

*Report generated by LLM Lab Custom Metrics System*
"""
    
    def generate_report(self, results: Dict[str, Dict], output_path: str = None) -> str:
        """Generate comprehensive metric report."""
        
        # Process results for template
        models = list(results.keys())
        total_metrics = len(set().union(*(r.keys() for r in results.values())))
        
        # Create metric summaries
        metric_summaries = self._create_metric_summaries(results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        
        # Metric descriptions
        metric_descriptions = {
            "customer_service_quality": "Evaluates empathy, professionalism, solution orientation, and policy compliance",
            "code_quality": "Assesses syntax validity, code structure, best practices, and complexity",
            "creativity": "Measures vocabulary diversity, sentence variety, imagery, and originality",
            "response_length": "Basic response length analysis",
            "sentiment": "Sentiment polarity analysis",
            "coherence": "Logical flow and consistency assessment"
        }
        
        # Template context
        context = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models": models,
            "total_metrics": total_metrics,
            "metric_summaries": metric_summaries,
            "detailed_results": results,
            "recommendations": recommendations,
            "metric_descriptions": metric_descriptions,
            "format_score": self._format_score,
            "interpret_score": self._interpret_score
        }
        
        # Render report
        template = Template(self.report_template)
        report = template.render(**context)
        
        # Save if path provided
        if output_path:
            Path(output_path).write_text(report)
        
        return report
    
    def _create_metric_summaries(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Create summary statistics for each metric."""
        
        summaries = {}
        
        # Get all unique metrics
        all_metrics = set()
        for model_results in results.values():
            all_metrics.update(model_results.keys())
        
        for metric_name in all_metrics:
            summary = {"model_scores": {}}
            
            # Collect scores for this metric across models
            for model, model_results in results.items():
                if metric_name in model_results and model_results[metric_name].get("success", True):
                    value = model_results[metric_name].get("value")
                    
                    if isinstance(value, dict):
                        if "overall" in value:
                            summary["model_scores"][model] = value["overall"]
                            summary["type"] = "composite"
                            
                            # Track components
                            if "components" not in summary:
                                summary["components"] = {}
                            
                            for comp_name, comp_value in value.items():
                                if comp_name != "overall" and isinstance(comp_value, (int, float)):
                                    if comp_name not in summary["components"]:
                                        summary["components"][comp_name] = []
                                    summary["components"][comp_name].append(comp_value)
                        else:
                            # Handle other dict formats
                            summary["model_scores"][model] = str(value)
                            summary["type"] = "categorical"
                    elif isinstance(value, (int, float)):
                        summary["model_scores"][model] = value
                        summary["type"] = "scalar"
                    else:
                        summary["model_scores"][model] = str(value)
                        summary["type"] = "other"
            
            # Calculate component statistics
            if "components" in summary:
                for comp_name, values in summary["components"].items():
                    summary["components"][comp_name] = {
                        "mean": statistics.mean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0
                    }
            
            summaries[metric_name] = summary
        
        return summaries
    
    def _generate_recommendations(self, results: Dict[str, Dict]) -> List[str]:
        """Generate recommendations based on results."""
        
        recommendations = []
        
        # Analyze customer service results
        if any("customer_service_quality" in r for r in results.values()):
            cs_scores = {}
            for model, model_results in results.items():
                if "customer_service_quality" in model_results:
                    value = model_results["customer_service_quality"].get("value", {})
                    if "overall" in value:
                        cs_scores[model] = value["overall"]
            
            if cs_scores:
                best_model = max(cs_scores, key=cs_scores.get)
                recommendations.append(f"For customer service applications, {best_model} shows the best overall performance (score: {cs_scores[best_model]:.3f})")
        
        # Analyze code quality results
        if any("code_quality" in r for r in results.values()):
            code_scores = {}
            for model, model_results in results.items():
                if "code_quality" in model_results:
                    value = model_results["code_quality"].get("value", {})
                    if "overall" in value:
                        code_scores[model] = value["overall"]
            
            if code_scores:
                best_model = max(code_scores, key=code_scores.get)
                recommendations.append(f"For code generation tasks, {best_model} demonstrates superior code quality (score: {code_scores[best_model]:.3f})")
        
        # Analyze creativity results
        if any("creativity" in r for r in results.values()):
            creativity_scores = {}
            for model, model_results in results.items():
                if "creativity" in model_results:
                    value = model_results["creativity"].get("value", {})
                    if "overall" in value:
                        creativity_scores[model] = value["overall"]
            
            if creativity_scores:
                best_model = max(creativity_scores, key=creativity_scores.get)
                recommendations.append(f"For creative writing tasks, {best_model} exhibits the highest creativity scores (score: {creativity_scores[best_model]:.3f})")
        
        # Add general recommendations
        if not recommendations:
            recommendations.append("Consider running additional evaluations with more diverse prompts to identify optimal models for specific use cases")
        
        recommendations.append("Regular evaluation with custom metrics helps identify the most suitable models for specific business requirements")
        
        return recommendations
    
    def _format_score(self, value) -> str:
        """Format score for display."""
        if isinstance(value, dict) and "overall" in value:
            return f"{value['overall']:.3f}"
        elif isinstance(value, (int, float)):
            return f"{value:.3f}"
        else:
            return str(value)
    
    def _interpret_score(self, metric_name: str, value) -> str:
        """Provide interpretation of score."""
        
        if isinstance(value, dict) and "overall" in value:
            score = value["overall"]
        elif isinstance(value, (int, float)):
            score = value
        else:
            return "N/A"
        
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        elif score >= 0.2:
            return "Poor"
        else:
            return "Very Poor"
```

## 🚀 Usage Examples

### 1. CLI Usage with Custom Metrics

```bash
# Customer service evaluation
python scripts/run_benchmarks.py \
  --prompt-file templates/customer_service/customer_service_response.txt \
  --prompt-variables '{"company_name": "TechCorp", "customer_type": "Premium", "severity": "High", "customer_message": "My system is down", "tone": "professional", "max_words": "150"}' \
  --models gpt-4,claude-3-sonnet \
  --custom-metrics customer_service \
  --output-format json,markdown \
  --output-dir ./results/custom-metrics/customer-service

# Code quality evaluation
python scripts/run_benchmarks.py \
  --prompt-file templates/code_generation/code_generation.txt \
  --prompt-variables '{"function_name": "fibonacci", "function_purpose": "Calculate fibonacci sequence", "complexity_level": "intermediate"}' \
  --models gpt-4,claude-3-sonnet \
  --custom-metrics code_quality \
  --parallel \
  --output-dir ./results/custom-metrics/code-quality

# Creative writing evaluation
python scripts/run_benchmarks.py \
  --prompt-file templates/creative_writing/story_generation.txt \
  --prompt-variables '{"genre": "science fiction", "word_count": "500", "protagonist": "A space engineer", "theme": "isolation"}' \
  --models gpt-4,claude-3-opus \
  --custom-metrics creativity \
  --output-dir ./results/custom-metrics/creative-writing

# Multiple custom metrics
python scripts/run_benchmarks.py \
  --custom-prompt "Write a professional email responding to a customer complaint about a software bug" \
  --models gpt-4,claude-3-sonnet,gemini-pro \
  --custom-metrics customer_service code_quality \
  --parallel \
  --output-dir ./results/custom-metrics/multi-metric
```

### 2. Programmatic Usage

```python
# Example script: examples/custom_metrics/metric_evaluation_example.py

from src.use_cases.custom_prompts.evaluation_metrics import MetricSuite
from src.use_cases.custom_prompts.custom_metrics import customer_service_metric, code_quality_metric, creativity_metric
from src.use_cases.custom_prompts.metric_visualization import MetricVisualizer
from src.use_cases.custom_prompts.metric_reports import MetricReporter

# Setup metric suite
suite = MetricSuite()
suite.add_metric(customer_service_metric)
suite.add_metric(code_quality_metric) 
suite.add_metric(creativity_metric)

# Example responses from different models
responses = {
    "gpt-4": "Thank you for reaching out about this issue. I understand how frustrating software bugs can be, especially when they impact your daily workflow. Let me help you resolve this promptly...",
    "claude-3-sonnet": "I apologize for the inconvenience you've experienced with the software bug. Your feedback is valuable, and I want to ensure we address this concern thoroughly...",
    "gemini-pro": "I'm sorry to hear about the software issue you're encountering. Let me work with you to find a solution and prevent this from happening again..."
}

# Evaluate all responses
results = {}
for model, response in responses.items():
    results[model] = suite.evaluate(
        response,
        customer_issue="software bug",
        company_policies=["respond within 24 hours", "escalate critical issues"],
        language="python"
    )

# Generate visualizations
visualizer = MetricVisualizer()
plots = visualizer.create_metric_report(results, "./results/metric_visualizations")

# Generate report
reporter = MetricReporter()
report = reporter.generate_report(results, "./results/metric_report.md")

print("Metric evaluation completed!")
print(f"Visualizations saved: {len(plots)} plots")
print(f"Report saved to: ./results/metric_report.md")
```

## 📚 Advanced Techniques

### 1. Weighted Metric Scoring

```python
def create_weighted_metric(base_metrics: List[CustomMetric], weights: Dict[str, float]) -> CustomMetric:
    """Create a weighted combination of multiple metrics."""
    
    def weighted_evaluator(response: str, **kwargs) -> float:
        total_score = 0
        total_weight = 0
        
        for metric in base_metrics:
            if metric.name in weights:
                result = metric.evaluate(response, **kwargs)
                if result["success"]:
                    value = result["value"]
                    if isinstance(value, dict) and "overall" in value:
                        score = value["overall"]
                    elif isinstance(value, (int, float)):
                        score = value
                    else:
                        continue
                    
                    weight = weights[metric.name]
                    total_score += score * weight
                    total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    
    return CustomMetric(
        name="weighted_combined",
        evaluator_function=weighted_evaluator,
        description=f"Weighted combination of: {', '.join(m.name for m in base_metrics)}",
        metric_type="scalar"
    )

# Usage example
combined_metric = create_weighted_metric(
    [customer_service_metric, creativity_metric],
    {"customer_service_quality": 0.7, "creativity": 0.3}
)
```

### 2. Threshold-Based Metrics

```python
def create_threshold_metric(base_metric: CustomMetric, thresholds: Dict[str, float]) -> CustomMetric:
    """Create a threshold-based version of a metric."""
    
    def threshold_evaluator(response: str, **kwargs) -> Dict[str, bool]:
        base_result = base_metric.evaluate(response, **kwargs)
        
        if not base_result["success"]:
            return base_result
        
        value = base_result["value"]
        if isinstance(value, dict) and "overall" in value:
            score = value["overall"]
        elif isinstance(value, (int, float)):
            score = value
        else:
            return {"error": "Cannot apply thresholds to non-numeric metric"}
        
        threshold_results = {}
        for threshold_name, threshold_value in thresholds.items():
            threshold_results[threshold_name] = score >= threshold_value
        
        return threshold_results
    
    return CustomMetric(
        name=f"{base_metric.name}_thresholds",
        evaluator_function=threshold_evaluator,
        description=f"Threshold-based evaluation of {base_metric.name}",
        metric_type="binary"
    )

# Usage example
quality_thresholds = create_threshold_metric(
    customer_service_metric,
    {"acceptable": 0.6, "good": 0.75, "excellent": 0.9}
)
```

### 3. Comparative Metrics

```python
def create_comparative_metric(base_metric: CustomMetric) -> CustomMetric:
    """Create a metric that compares multiple responses."""
    
    def comparative_evaluator(responses: List[str], **kwargs) -> Dict[str, Any]:
        """Compare multiple responses and rank them."""
        
        scores = []
        for i, response in enumerate(responses):
            result = base_metric.evaluate(response, **kwargs)
            if result["success"]:
                value = result["value"]
                if isinstance(value, dict) and "overall" in value:
                    score = value["overall"]
                elif isinstance(value, (int, float)):
                    score = value
                else:
                    score = 0
            else:
                score = 0
            scores.append((i, score))
        
        # Sort by score (descending)
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        
        return {
            "rankings": [{"response_index": idx, "Score": score} for idx, score in ranked],
            "best_response_index": ranked[0][0] if ranked else None,
            "score_spread": max(s[1] for s in scores) - min(s[1] for s in scores) if scores else 0
        }
    
    return CustomMetric(
        name=f"{base_metric.name}_comparative",
        evaluator_function=comparative_evaluator,
        description=f"Comparative ranking using {base_metric.name}",
        metric_type="comparative"
    )
```

## 🎯 Best Practices

### 1. Metric Design Guidelines

- **Define Clear Objectives**: What specific quality are you measuring?
- **Use Appropriate Scales**: 0-1 for probabilities, 0-100 for percentages
- **Handle Edge Cases**: Empty responses, errors, unexpected formats
- **Provide Interpretable Results**: Include explanations with scores
- **Validate Against Human Judgment**: Test metrics against expert evaluations

### 2. Performance Considerations

- **Optimize for Speed**: Use efficient algorithms for large-scale evaluation
- **Cache Expensive Operations**: Store intermediate results when possible
- **Batch Processing**: Evaluate multiple responses together when applicable
- **Error Handling**: Gracefully handle failures without stopping evaluation

### 3. Integration Tips

- **Modular Design**: Keep metrics independent and reusable
- **Configuration Support**: Allow parameter tuning via config files
- **Documentation**: Provide clear usage examples and parameter descriptions
- **Testing**: Write unit tests for metric functions
- **Versioning**: Track metric versions for reproducibility

---

*This comprehensive guide provides everything needed to create, implement, and use custom evaluation metrics for domain-specific LLM assessment. The examples and tools provided enable sophisticated evaluation workflows tailored to specific business requirements.*