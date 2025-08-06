"""
Local Model-Specific Evaluation Metrics

This module provides evaluation metrics specifically designed for local models,
including performance monitoring, resource utilization tracking, and local model
quality assessments.
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class LocalModelPerformanceMetrics:
    """Performance metrics specific to local model inference."""

    # Timing metrics
    inference_time_ms: float
    tokens_per_second: float
    first_token_latency_ms: float | None = None

    # Resource utilization
    memory_used_mb: float
    memory_peak_mb: float
    gpu_memory_used_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    cpu_utilization_percent: float = 0.0

    # Model-specific metrics
    model_size_mb: float = 0.0
    context_length_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Quality metrics
    response_length: int = 0
    response_coherence_score: float = 0.0
    repetition_penalty_triggered: bool = False


@dataclass
class LocalModelBenchmarkResult:
    """Extended benchmark result with local model-specific data."""

    # Standard benchmark metrics
    model_name: str
    dataset_name: str
    prompt: str
    response: str
    expected_answer: str
    evaluation_score: float
    evaluation_method: str

    # Local model performance metrics
    performance_metrics: LocalModelPerformanceMetrics

    # System information
    backend_used: str
    model_format: str
    quantization_level: str | None = None
    gpu_accelerated: bool = False

    # Timestamp
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class LocalModelEvaluator:
    """Evaluator for local model-specific metrics and performance."""

    def __init__(self, resource_monitor=None):
        """
        Initialize local model evaluator.

        Args:
            resource_monitor: Optional resource monitoring instance
        """
        self.resource_monitor = resource_monitor
        self.logger = logging.getLogger(__name__)

    def evaluate_response_quality(
        self, response: str, expected_answer: str, prompt: str
    ) -> Dict[str, Any]:
        """
        Evaluate response quality with local model-specific considerations.

        Args:
            response: Generated response
            expected_answer: Expected answer
            prompt: Original prompt

        Returns:
            Dictionary with quality metrics
        """
        metrics = {}

        # Basic quality metrics
        metrics["response_length"] = len(response)
        metrics["response_word_count"] = len(response.split())

        # Check for repetition (common issue in local models)
        metrics["repetition_score"] = self._calculate_repetition_score(response)

        # Check for coherence
        metrics["coherence_score"] = self._calculate_coherence_score(response)

        # Check for prompt leakage (model repeating the prompt)
        metrics["prompt_leakage_score"] = self._calculate_prompt_leakage(prompt, response)

        # Token efficiency (response quality vs length)
        metrics["token_efficiency"] = self._calculate_token_efficiency(response, expected_answer)

        return metrics

    def _calculate_repetition_score(self, response: str) -> float:
        """Calculate repetition score (0.0 = no repetition, 1.0 = high repetition)."""
        if not response:
            return 0.0

        words = response.lower().split()
        if len(words) < 2:
            return 0.0

        # Count consecutive repeated words
        repeated_count = 0
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                repeated_count += 1

        # Count repeated phrases (3+ word sequences)
        phrases = []
        for i in range(len(words) - 2):
            phrase = " ".join(words[i : i + 3])
            phrases.append(phrase)

        unique_phrases = len(set(phrases))
        total_phrases = len(phrases)

        if total_phrases == 0:
            phrase_repetition = 0.0
        else:
            phrase_repetition = 1.0 - (unique_phrases / total_phrases)

        # Combine word and phrase repetition
        word_repetition = repeated_count / (len(words) - 1)
        return (word_repetition + phrase_repetition) / 2

    def _calculate_coherence_score(self, response: str) -> float:
        """Calculate coherence score (0.0 = incoherent, 1.0 = coherent)."""
        if not response:
            return 0.0

        sentences = response.split(".")
        if len(sentences) < 2:
            return 1.0  # Single sentence is considered coherent

        # Simple heuristics for coherence
        score = 1.0

        # Check for abrupt topic changes (simplified)
        words_per_sentence = [len(s.split()) for s in sentences if s.strip()]
        if words_per_sentence:
            avg_length = sum(words_per_sentence) / len(words_per_sentence)
            # Penalize very short sentences (might indicate incoherence)
            if avg_length < 3:
                score -= 0.3

        # Check for incomplete sentences
        incomplete_count = sum(1 for s in sentences if s.strip() and not s.strip()[-1:].isalpha())
        if incomplete_count > len(sentences) * 0.3:  # More than 30% incomplete
            score -= 0.2

        return max(0.0, score)

    def _calculate_prompt_leakage(self, prompt: str, response: str) -> float:
        """Calculate how much of the prompt appears in the response (0.0 = no leakage, 1.0 = high leakage)."""
        if not prompt or not response:
            return 0.0

        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())

        if len(prompt_words) == 0:
            return 0.0

        # Calculate overlap
        overlap = len(prompt_words.intersection(response_words))
        return overlap / len(prompt_words)

    def _calculate_token_efficiency(self, response: str, expected_answer: str) -> float:
        """Calculate token efficiency (quality per token used)."""
        if not response:
            return 0.0

        response_tokens = len(response.split())
        expected_tokens = len(expected_answer.split()) if expected_answer else 10

        # Simple efficiency metric: penalize overly long responses
        if response_tokens <= expected_tokens:
            return 1.0
        elif response_tokens <= expected_tokens * 2:
            return 0.8
        elif response_tokens <= expected_tokens * 3:
            return 0.6
        else:
            return 0.4

    def create_performance_metrics(
        self,
        inference_time_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        memory_used_mb: float,
        memory_peak_mb: float,
        model_size_mb: float,
        gpu_memory_used_mb: float = 0.0,
        gpu_utilization_percent: float = 0.0,
        cpu_utilization_percent: float = 0.0,
        first_token_latency_ms: float | None = None,
    ) -> LocalModelPerformanceMetrics:
        """Create performance metrics object."""

        # Calculate tokens per second
        total_tokens = completion_tokens
        if inference_time_ms > 0:
            tokens_per_second = (total_tokens * 1000) / inference_time_ms
        else:
            tokens_per_second = 0.0

        return LocalModelPerformanceMetrics(
            inference_time_ms=inference_time_ms,
            tokens_per_second=tokens_per_second,
            first_token_latency_ms=first_token_latency_ms,
            memory_used_mb=memory_used_mb,
            memory_peak_mb=memory_peak_mb,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_utilization_percent=gpu_utilization_percent,
            cpu_utilization_percent=cpu_utilization_percent,
            model_size_mb=model_size_mb,
            context_length_used=prompt_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            response_length=completion_tokens,
        )


def evaluate_local_model_response(
    model_name: str,
    dataset_name: str,
    prompt: str,
    response: str,
    expected_answer: str,
    performance_metrics: LocalModelPerformanceMetrics,
    backend_used: str,
    model_format: str,
    evaluation_method: str = "local_model_eval",
    quantization_level: str | None = None,
    gpu_accelerated: bool = False,
) -> LocalModelBenchmarkResult:
    """
    Comprehensive evaluation function for local model responses.

    Args:
        model_name: Name of the model
        dataset_name: Name of the benchmark dataset
        prompt: Original prompt
        response: Model's response
        expected_answer: Expected answer
        performance_metrics: Performance metrics from inference
        backend_used: Backend used (transformers, llamacpp, ollama)
        model_format: Model format (GGUF, safetensors, etc.)
        evaluation_method: Evaluation method name
        quantization_level: Quantization level if applicable
        gpu_accelerated: Whether GPU was used

    Returns:
        LocalModelBenchmarkResult with comprehensive metrics
    """
    evaluator = LocalModelEvaluator()

    # Basic evaluation (can be enhanced with standard methods)
    from .improved_evaluation import multi_method_evaluation

    basic_eval = multi_method_evaluation(
        response=response,
        expected_keywords=[expected_answer] if expected_answer else [],
        expected_answer=expected_answer,
    )

    # Get the evaluation score
    evaluation_score = basic_eval.get("overall_score", 0.0)

    # Add local model-specific quality metrics
    quality_metrics = evaluator.evaluate_response_quality(response, expected_answer, prompt)

    # Update performance metrics with quality data
    performance_metrics.response_coherence_score = quality_metrics.get("coherence_score", 0.0)
    performance_metrics.repetition_penalty_triggered = (
        quality_metrics.get("repetition_score", 0.0) > 0.5
    )
    performance_metrics.response_length = quality_metrics.get("response_length", 0)

    return LocalModelBenchmarkResult(
        model_name=model_name,
        dataset_name=dataset_name,
        prompt=prompt,
        response=response,
        expected_answer=expected_answer,
        evaluation_score=evaluation_score,
        evaluation_method=evaluation_method,
        performance_metrics=performance_metrics,
        backend_used=backend_used,
        model_format=model_format,
        quantization_level=quantization_level,
        gpu_accelerated=gpu_accelerated,
    )


def generate_local_model_report(results: List[LocalModelBenchmarkResult]) -> Dict[str, Any]:
    """
    Generate a comprehensive report for local model benchmark results.

    Args:
        results: List of local model benchmark results

    Returns:
        Dictionary with aggregated metrics and analysis
    """
    if not results:
        return {"error": "No results provided"}

    report = {
        "summary": {
            "total_evaluations": len(results),
            "models_tested": len(set(r.model_name for r in results)),
            "datasets_used": len(set(r.dataset_name for r in results)),
            "backends_used": len(set(r.backend_used for r in results)),
        },
        "performance": {},
        "quality": {},
        "efficiency": {},
    }

    # Aggregate performance metrics
    total_inference_time = sum(r.performance_metrics.inference_time_ms for r in results)
    total_tokens = sum(r.performance_metrics.completion_tokens for r in results)

    if results:
        avg_tokens_per_second = sum(r.performance_metrics.tokens_per_second for r in results) / len(
            results
        )
        avg_memory_usage = sum(r.performance_metrics.memory_used_mb for r in results) / len(results)
        avg_gpu_utilization = sum(
            r.performance_metrics.gpu_utilization_percent for r in results
        ) / len(results)

        report["performance"] = {
            "avg_inference_time_ms": total_inference_time / len(results),
            "avg_tokens_per_second": avg_tokens_per_second,
            "avg_memory_usage_mb": avg_memory_usage,
            "avg_gpu_utilization_percent": avg_gpu_utilization,
            "total_tokens_generated": total_tokens,
            "total_time_spent_ms": total_inference_time,
        }

    # Aggregate quality metrics
    avg_evaluation_score = sum(r.evaluation_score for r in results) / len(results)
    avg_coherence_score = sum(
        r.performance_metrics.response_coherence_score for r in results
    ) / len(results)
    repetition_issues = sum(
        1 for r in results if r.performance_metrics.repetition_penalty_triggered
    )

    report["quality"] = {
        "avg_evaluation_score": avg_evaluation_score,
        "avg_coherence_score": avg_coherence_score,
        "repetition_issues_percent": (repetition_issues / len(results)) * 100,
        "avg_response_length": sum(r.performance_metrics.response_length for r in results)
        / len(results),
    }

    # Efficiency analysis
    gpu_accelerated_count = sum(1 for r in results if r.gpu_accelerated)

    report["efficiency"] = {
        "gpu_utilization_percent": (gpu_accelerated_count / len(results)) * 100,
        "tokens_per_mb_memory": total_tokens / avg_memory_usage if avg_memory_usage > 0 else 0,
        "inference_efficiency": total_tokens / (total_inference_time / 1000)
        if total_inference_time > 0
        else 0,  # tokens per second overall
    }

    return report
