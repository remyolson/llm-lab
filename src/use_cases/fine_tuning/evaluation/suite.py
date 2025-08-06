"""
Comprehensive Evaluation Suite for Fine-Tuned Models

This module provides a comprehensive evaluation framework for assessing model
performance across various benchmarks and metrics, with support for before/after
comparisons and regression detection.

Example:
    suite = EvaluationSuite()

    # Evaluate model
    results = suite.evaluate(
        model=model,
        tokenizer=tokenizer,
        benchmarks=["hellaswag", "mmlu", "winogrande"]
    )

    # Compare with baseline
    comparison = suite.compare_models(
        base_model=base_model,
        fine_tuned_model=ft_model,
        tokenizer=tokenizer
    )
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import evaluation libraries
try:
    from evaluate import load as load_metric

    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False

try:
    from lm_eval import evaluator, tasks

    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False

# Import transformers components
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# Import datasets
from datasets import Dataset, load_dataset

# Import custom evaluation support
from .custom_evaluations import create_recipe_evaluation_function

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result from a single metric evaluation."""

    name: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result from a benchmark evaluation."""

    name: str
    overall_score: float
    task_scores: Dict[str, float] = field(default_factory=dict)
    metrics: List[MetricResult] = field(default_factory=list)
    runtime_seconds: float = 0.0
    samples_evaluated: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation results for a model."""

    model_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    benchmarks: List[BenchmarkResult] = field(default_factory=list)
    custom_metrics: List[MetricResult] = field(default_factory=list)
    perplexity: Optional[float] = None
    total_runtime_seconds: float = 0.0
    hardware_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "benchmarks": [
                {
                    "name": b.name,
                    "overall_score": b.overall_score,
                    "task_scores": b.task_scores,
                    "metrics": [{"name": m.name, "value": m.value} for m in b.metrics],
                    "runtime_seconds": b.runtime_seconds,
                    "samples_evaluated": b.samples_evaluated,
                }
                for b in self.benchmarks
            ],
            "custom_metrics": [{"name": m.name, "value": m.value} for m in self.custom_metrics],
            "perplexity": self.perplexity,
            "total_runtime_seconds": self.total_runtime_seconds,
            "hardware_info": self.hardware_info,
        }


@dataclass
class EvaluationConfig:
    """Configuration for evaluation suite."""

    benchmarks: List[str] = field(default_factory=lambda: ["hellaswag", "mmlu", "winogrande"])
    custom_metrics: List[str] = field(default_factory=lambda: ["bleu", "rouge", "bertscore"])
    batch_size: int = 8
    max_samples: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    seed: int = 42
    calculate_perplexity: bool = True
    save_results: bool = True
    results_dir: str = "./evaluation_results"
    use_cache: bool = True
    regression_threshold: float = 0.05  # 5% performance drop threshold


class EvaluationSuite:
    """Comprehensive evaluation suite for fine-tuned models."""

    # Available benchmarks
    SUPPORTED_BENCHMARKS = {
        "hellaswag": {"dataset": "hellaswag", "metric": "accuracy", "num_choices": 4},
        "mmlu": {
            "dataset": "cais/mmlu",
            "metric": "accuracy",
            "num_choices": 4,
            "subjects": ["all"],  # Can specify individual subjects
        },
        "winogrande": {
            "dataset": "winogrande",
            "config": "winogrande_xl",
            "metric": "accuracy",
            "num_choices": 2,
        },
        "arc": {
            "dataset": "ai2_arc",
            "config": "ARC-Challenge",
            "metric": "accuracy",
            "num_choices": 4,
        },
        "lambada": {"dataset": "lambada", "metric": "accuracy", "generation": True},
        "piqa": {"dataset": "piqa", "metric": "accuracy", "num_choices": 2},
        "siqa": {"dataset": "social_i_qa", "metric": "accuracy", "num_choices": 3},
        "openbookqa": {"dataset": "openbookqa", "metric": "accuracy", "num_choices": 4},
    }

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize evaluation suite.

        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()

        # Create results directory
        self.results_dir = Path(self.config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metric cache
        self._metric_cache = {}

        # Set random seed
        self._set_seed(self.config.seed)

        # Check available libraries
        if not EVALUATE_AVAILABLE:
            logger.warning("'evaluate' library not available. Some metrics will be disabled.")
        if not LM_EVAL_AVAILABLE:
            logger.warning("'lm-eval' library not available. Some benchmarks will be disabled.")

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        benchmarks: Optional[List[str]] = None,
        custom_eval_fn: Optional[Callable] = None,
    ) -> EvaluationResult:
        """Run comprehensive evaluation on a model.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            benchmarks: List of benchmarks to run
            custom_eval_fn: Optional custom evaluation function

        Returns:
            Complete evaluation results
        """
        benchmarks = benchmarks or self.config.benchmarks

        # Start evaluation
        start_time = time.time()
        logger.info(f"Starting evaluation for model: {model.config.name_or_path}")

        # Initialize results
        results = EvaluationResult(
            model_name=model.config.name_or_path, hardware_info=self._get_hardware_info()
        )

        # Calculate perplexity if requested
        if self.config.calculate_perplexity:
            logger.info("Calculating perplexity...")
            results.perplexity = self._calculate_perplexity(model, tokenizer)

        # Run benchmarks
        for benchmark_name in benchmarks:
            if benchmark_name in self.SUPPORTED_BENCHMARKS:
                logger.info(f"Running benchmark: {benchmark_name}")
                benchmark_result = self._run_benchmark(model, tokenizer, benchmark_name)
                results.benchmarks.append(benchmark_result)
            else:
                logger.warning(f"Unsupported benchmark: {benchmark_name}")

        # Run custom metrics
        if self.config.custom_metrics:
            logger.info("Running custom metrics...")
            custom_results = self._run_custom_metrics(model, tokenizer, self.config.custom_metrics)
            results.custom_metrics.extend(custom_results)

        # Run custom evaluation function if provided
        if custom_eval_fn:
            logger.info("Running custom evaluation function...")
            custom_result = custom_eval_fn(model, tokenizer)
            if isinstance(custom_result, list):
                results.custom_metrics.extend(custom_result)
            else:
                results.custom_metrics.append(custom_result)

        # Calculate total runtime
        results.total_runtime_seconds = time.time() - start_time

        # Save results if requested
        if self.config.save_results:
            self._save_results(results)

        logger.info(f"Evaluation complete in {results.total_runtime_seconds:.2f} seconds")
        return results

    def evaluate_with_recipe(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        recipe: Dict[str, Any],
        benchmarks: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Run evaluation based on recipe configuration.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            recipe: Recipe configuration dictionary
            benchmarks: Optional override for benchmarks

        Returns:
            Complete evaluation results
        """
        # Extract evaluation config from recipe
        eval_config = recipe.get("evaluation", {})

        # Use recipe benchmarks if not overridden
        if benchmarks is None:
            benchmarks = eval_config.get("benchmarks", self.config.benchmarks)

        # Get custom evaluation function from recipe
        custom_eval_fn = create_recipe_evaluation_function(recipe)

        # Run standard evaluation with recipe customizations
        results = self.evaluate(
            model=model, tokenizer=tokenizer, benchmarks=benchmarks, custom_eval_fn=custom_eval_fn
        )

        # Add recipe metadata to results
        results.metadata["recipe_name"] = recipe.get("name", "unknown")
        results.metadata["recipe_type"] = recipe.get("recipe_type", "unknown")

        return results

    def _calculate_perplexity(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
    ) -> float:
        """Calculate model perplexity on a dataset.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            dataset_name: Dataset to use
            dataset_config: Dataset configuration

        Returns:
            Perplexity value
        """
        # Load dataset
        dataset = load_dataset(dataset_name, dataset_config, split="test")

        # Prepare data
        encodings = tokenizer(
            dataset["text"][:100],  # Use subset for efficiency
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )

        # Calculate perplexity
        model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for i in range(0, len(encodings["input_ids"]), self.config.batch_size):
                batch_ids = encodings["input_ids"][i : i + self.config.batch_size].to(model.device)
                batch_mask = encodings["attention_mask"][i : i + self.config.batch_size].to(
                    model.device
                )

                outputs = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_ids)

                total_loss += outputs.loss.item() * batch_ids.size(0)
                total_tokens += batch_mask.sum().item()

        avg_loss = total_loss / len(encodings["input_ids"])
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        logger.info(f"Perplexity: {perplexity:.2f}")
        return perplexity

    def _run_benchmark(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, benchmark_name: str
    ) -> BenchmarkResult:
        """Run a specific benchmark.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            benchmark_name: Name of benchmark

        Returns:
            Benchmark results
        """
        benchmark_info = self.SUPPORTED_BENCHMARKS[benchmark_name]
        start_time = time.time()

        # Use lm-eval if available for supported benchmarks
        if LM_EVAL_AVAILABLE and benchmark_name in ["hellaswag", "mmlu", "arc", "lambada"]:
            return self._run_lm_eval_benchmark(model, tokenizer, benchmark_name)

        # Otherwise use custom implementation
        dataset_name = benchmark_info["dataset"]
        dataset_config = benchmark_info.get("config", None)

        # Load dataset
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split="validation")
        else:
            dataset = load_dataset(dataset_name, split="validation")

        # Limit samples if specified
        if self.config.max_samples:
            dataset = dataset.select(range(min(len(dataset), self.config.max_samples)))

        # Run evaluation based on task type
        if benchmark_info.get("generation", False):
            scores = self._evaluate_generation_task(model, tokenizer, dataset, benchmark_info)
        else:
            scores = self._evaluate_multiple_choice_task(model, tokenizer, dataset, benchmark_info)

        # Calculate overall score
        overall_score = np.mean(list(scores.values()))

        runtime = time.time() - start_time

        return BenchmarkResult(
            name=benchmark_name,
            overall_score=overall_score,
            task_scores=scores,
            runtime_seconds=runtime,
            samples_evaluated=len(dataset),
            metadata={"dataset": dataset_name, "config": dataset_config},
        )

    def _evaluate_multiple_choice_task(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset,
        benchmark_info: Dict[str, Any],
    ) -> Dict[str, float]:
        """Evaluate multiple choice task.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            dataset: Dataset
            benchmark_info: Benchmark configuration

        Returns:
            Task scores
        """
        model.eval()
        correct = 0
        total = 0

        # Create dataloader
        dataloader = DataLoader(
            dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers
        )

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {benchmark_info['dataset']}"):
                # Prepare inputs based on dataset format
                if "question" in batch:
                    questions = batch["question"]
                    choices = [batch[f"choice{i}"] for i in range(benchmark_info["num_choices"])]
                    answers = batch["answer"]
                else:
                    # Handle different dataset formats
                    continue

                # Score each choice
                batch_correct = 0
                for q_idx in range(len(questions)):
                    question = questions[q_idx]
                    choice_scores = []

                    for choice_idx in range(benchmark_info["num_choices"]):
                        choice = choices[choice_idx][q_idx]

                        # Format input
                        input_text = f"{question} {choice}"
                        inputs = tokenizer(
                            input_text,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=512,
                        ).to(model.device)

                        # Get model score
                        outputs = model(**inputs)
                        score = outputs.logits[0, -1, :].mean().item()
                        choice_scores.append(score)

                    # Check if prediction is correct
                    predicted = np.argmax(choice_scores)
                    if predicted == answers[q_idx]:
                        batch_correct += 1

                correct += batch_correct
                total += len(questions)

        accuracy = correct / total if total > 0 else 0
        return {"accuracy": accuracy}

    def _evaluate_generation_task(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset,
        benchmark_info: Dict[str, Any],
    ) -> Dict[str, float]:
        """Evaluate generation task.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            dataset: Dataset
            benchmark_info: Benchmark configuration

        Returns:
            Task scores
        """
        model.eval()
        scores = []

        # Configure generation
        gen_config = GenerationConfig(
            max_new_tokens=50, temperature=0.7, top_p=0.95, do_sample=True
        )

        with torch.no_grad():
            for example in tqdm(dataset, desc="Evaluating generation"):
                # Prepare input
                input_text = example.get("text", example.get("prompt", ""))
                inputs = tokenizer(
                    input_text, return_tensors="pt", truncation=True, max_length=512
                ).to(model.device)

                # Generate
                outputs = model.generate(**inputs, generation_config=gen_config)

                # Decode and evaluate
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Simple accuracy check (can be improved based on task)
                target = example.get("target", example.get("completion", ""))
                score = 1.0 if target.lower() in generated.lower() else 0.0
                scores.append(score)

        return {"accuracy": np.mean(scores)}

    def _run_lm_eval_benchmark(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, benchmark_name: str
    ) -> BenchmarkResult:
        """Run benchmark using lm-eval harness.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            benchmark_name: Benchmark name

        Returns:
            Benchmark results
        """
        # Map benchmark names to lm-eval task names
        task_map = {
            "hellaswag": "hellaswag",
            "mmlu": "hendrycksTest-*",
            "arc": "arc_challenge",
            "lambada": "lambada_openai",
        }

        task_name = task_map.get(benchmark_name, benchmark_name)

        # Run evaluation
        results = evaluator.simple_evaluate(
            model=model,
            tasks=[task_name],
            batch_size=self.config.batch_size,
            device=self.config.device,
            no_cache=not self.config.use_cache,
            limit=self.config.max_samples,
        )

        # Extract scores
        task_results = results["results"][task_name]
        overall_score = task_results.get("acc", task_results.get("acc_norm", 0))

        return BenchmarkResult(
            name=benchmark_name,
            overall_score=overall_score,
            task_scores={"accuracy": overall_score},
            runtime_seconds=results.get("total_time", 0),
            samples_evaluated=results.get("total_instances", 0),
            metadata=task_results,
        )

    def _run_custom_metrics(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, metric_names: List[str]
    ) -> List[MetricResult]:
        """Run custom metrics.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            metric_names: List of metric names

        Returns:
            List of metric results
        """
        results = []

        for metric_name in metric_names:
            if EVALUATE_AVAILABLE:
                try:
                    # Load metric
                    if metric_name not in self._metric_cache:
                        self._metric_cache[metric_name] = load_metric(metric_name)

                    metric = self._metric_cache[metric_name]

                    # Run metric evaluation (simplified example)
                    # In practice, this would use appropriate test data
                    score = self._evaluate_with_metric(model, tokenizer, metric, metric_name)

                    results.append(MetricResult(name=metric_name, value=score))
                except Exception as e:
                    logger.warning(f"Failed to evaluate metric {metric_name}: {e}")
            else:
                logger.warning(f"Metric {metric_name} requires 'evaluate' library")

        return results

    def _evaluate_with_metric(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, metric: Any, metric_name: str
    ) -> float:
        """Evaluate model with a specific metric.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            metric: Loaded metric
            metric_name: Name of metric

        Returns:
            Metric score
        """
        # This is a simplified example - in practice would use appropriate test data
        # and metric-specific evaluation logic

        if metric_name == "perplexity":
            return self._calculate_perplexity(model, tokenizer)

        # For other metrics, would need appropriate test data and evaluation logic
        # Returning dummy value for now
        return 0.0

    def compare_models(
        self,
        base_model: PreTrainedModel | str,
        fine_tuned_model: PreTrainedModel | str,
        tokenizer: PreTrainedTokenizer,
        benchmarks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare base and fine-tuned models.

        Args:
            base_model: Base model or path
            fine_tuned_model: Fine-tuned model or path
            tokenizer: Tokenizer
            benchmarks: Benchmarks to run

        Returns:
            Comparison results
        """
        # Load models if paths provided
        if isinstance(base_model, str):
            base_model = AutoModelForCausalLM.from_pretrained(base_model)
        if isinstance(fine_tuned_model, str):
            fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model)

        # Evaluate both models
        logger.info("Evaluating base model...")
        base_results = self.evaluate(base_model, tokenizer, benchmarks)

        logger.info("Evaluating fine-tuned model...")
        ft_results = self.evaluate(fine_tuned_model, tokenizer, benchmarks)

        # Compare results
        comparison = {
            "base_model": base_results.to_dict(),
            "fine_tuned_model": ft_results.to_dict(),
            "improvements": {},
            "regressions": [],
        }

        # Calculate improvements
        for ft_bench in ft_results.benchmarks:
            base_bench = next((b for b in base_results.benchmarks if b.name == ft_bench.name), None)

            if base_bench:
                improvement = ft_bench.overall_score - base_bench.overall_score
                improvement_pct = (improvement / base_bench.overall_score) * 100

                comparison["improvements"][ft_bench.name] = {
                    "base_score": base_bench.overall_score,
                    "ft_score": ft_bench.overall_score,
                    "improvement": improvement,
                    "improvement_pct": improvement_pct,
                }

                # Check for regressions
                if improvement_pct < -self.config.regression_threshold * 100:
                    comparison["regressions"].append(
                        {"benchmark": ft_bench.name, "regression_pct": improvement_pct}
                    )

        # Compare perplexity
        if base_results.perplexity and ft_results.perplexity:
            perplexity_improvement = base_results.perplexity - ft_results.perplexity
            comparison["improvements"]["perplexity"] = {
                "base": base_results.perplexity,
                "fine_tuned": ft_results.perplexity,
                "improvement": perplexity_improvement,
            }

        # Save comparison
        if self.config.save_results:
            comparison_path = (
                self.results_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(comparison_path, "w") as f:
                json.dump(comparison, f, indent=2)
            logger.info(f"Comparison saved to {comparison_path}")

        return comparison

    def detect_regressions(
        self,
        current_results: EvaluationResult,
        baseline_results: EvaluationResult,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Detect performance regressions.

        Args:
            current_results: Current evaluation results
            baseline_results: Baseline results to compare against
            threshold: Regression threshold (percentage)

        Returns:
            List of detected regressions
        """
        threshold = threshold or self.config.regression_threshold
        regressions = []

        # Compare benchmarks
        for current_bench in current_results.benchmarks:
            baseline_bench = next(
                (b for b in baseline_results.benchmarks if b.name == current_bench.name), None
            )

            if baseline_bench:
                score_drop = baseline_bench.overall_score - current_bench.overall_score
                score_drop_pct = (score_drop / baseline_bench.overall_score) * 100

                if score_drop_pct > threshold * 100:
                    regressions.append(
                        {
                            "type": "benchmark",
                            "name": current_bench.name,
                            "baseline_score": baseline_bench.overall_score,
                            "current_score": current_bench.overall_score,
                            "drop_percentage": score_drop_pct,
                        }
                    )

        # Compare perplexity
        if current_results.perplexity and baseline_results.perplexity:
            perplexity_increase = current_results.perplexity - baseline_results.perplexity
            perplexity_increase_pct = (perplexity_increase / baseline_results.perplexity) * 100

            if perplexity_increase_pct > threshold * 100:
                regressions.append(
                    {
                        "type": "perplexity",
                        "baseline": baseline_results.perplexity,
                        "current": current_results.perplexity,
                        "increase_percentage": perplexity_increase_pct,
                    }
                )

        return regressions

    def generate_report(self, results: EvaluationResult, output_format: str = "markdown") -> str:
        """Generate evaluation report.

        Args:
            results: Evaluation results
            output_format: Report format ('markdown' or 'html')

        Returns:
            Formatted report
        """
        if output_format == "markdown":
            return self._generate_markdown_report(results)
        elif output_format == "html":
            return self._generate_html_report(results)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

    def _generate_markdown_report(self, results: EvaluationResult) -> str:
        """Generate markdown evaluation report."""
        report = []
        report.append(f"# Evaluation Report: {results.model_name}")
        report.append(f"\n**Date:** {results.timestamp}")
        report.append(f"**Total Runtime:** {results.total_runtime_seconds:.2f} seconds")

        if results.perplexity:
            report.append(f"\n## Perplexity: {results.perplexity:.2f}")

        report.append("\n## Benchmark Results")

        for benchmark in results.benchmarks:
            report.append(f"\n### {benchmark.name}")
            report.append(f"- **Overall Score:** {benchmark.overall_score:.4f}")
            report.append(f"- **Samples Evaluated:** {benchmark.samples_evaluated}")
            report.append(f"- **Runtime:** {benchmark.runtime_seconds:.2f}s")

            if benchmark.task_scores:
                report.append("\n**Task Scores:**")
                for task, score in benchmark.task_scores.items():
                    report.append(f"- {task}: {score:.4f}")

        if results.custom_metrics:
            report.append("\n## Custom Metrics")
            for metric in results.custom_metrics:
                report.append(f"- **{metric.name}:** {metric.value:.4f}")

        if results.hardware_info:
            report.append("\n## Hardware Information")
            for key, value in results.hardware_info.items():
                report.append(f"- **{key}:** {value}")

        return "\n".join(report)

    def _generate_html_report(self, results: EvaluationResult) -> str:
        """Generate HTML evaluation report."""
        # Convert markdown to HTML
        markdown_report = self._generate_markdown_report(results)

        # Simple HTML wrapper
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evaluation Report: {results.model_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            {markdown_report.replace(chr(10), "<br>" + chr(10))}
        </body>
        </html>
        """

        return html

    def _save_results(self, results: EvaluationResult):
        """Save evaluation results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_path = self.results_dir / f"{results.model_name}_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)

        # Save markdown report
        report = self.generate_report(results)
        report_path = self.results_dir / f"{results.model_name}_{timestamp}.md"
        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"Results saved to {json_path} and {report_path}")

    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        info = {"device": str(self.config.device), "cpu_count": os.cpu_count()}

        if torch.cuda.is_available():
            info.update(
                {
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                }
            )

        return info


# Example usage
if __name__ == "__main__":
    # Initialize evaluation suite
    config = EvaluationConfig(
        benchmarks=["hellaswag", "mmlu"],
        custom_metrics=["perplexity"],
        batch_size=4,
        max_samples=100,  # For quick testing
        save_results=True,
    )

    suite = EvaluationSuite(config)

    # Load model and tokenizer
    model_name = "microsoft/phi-2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Run evaluation
    results = suite.evaluate(model, tokenizer)

    # Print summary
    print(f"\nEvaluation Results for {model_name}:")
    print(f"Perplexity: {results.perplexity:.2f}")

    for benchmark in results.benchmarks:
        print(f"\n{benchmark.name}:")
        print(f"  Overall Score: {benchmark.overall_score:.4f}")
        print(f"  Runtime: {benchmark.runtime_seconds:.2f}s")

    # Generate and save report
    report = suite.generate_report(results)
    print("\nGenerated report:")
    print(report)
