"""Evaluation metrics for custom prompt responses.

This module provides comprehensive evaluation metrics beyond simple keyword matching:
- Response length analysis (characters, words, sentences)
- Sentiment analysis
- Coherence measurement
- Response diversity across multiple runs
- Custom metric interface for user-defined evaluations
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import re
import statistics
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MetricResult:
    """Container for a single metric result."""

    name: str
    value: float | int | dict | list
    unit: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str | Any]:
        """Convert to dictionary representation."""
        result = {"name": self.name, "value": self.value}
        if self.unit:
            result["unit"] = self.unit
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate(self, response: str, **kwargs) -> MetricResult:
        """Calculate the metric for a single response.

        Args:
            response: The model's response text
            **kwargs: Additional context (prompt, model_name, etc.)

        Returns:
            MetricResult with the calculated value
        """
        pass

    def calculate_batch(self, responses: List[str], **kwargs) -> List[MetricResult]:
        """Calculate the metric for multiple responses.

        Args:
            responses: List of model responses
            **kwargs: Additional context

        Returns:
            List of MetricResult objects
        """
        return [self.calculate(response, **kwargs) for response in responses]


class ResponseLengthMetric(BaseMetric):
    """Analyzes response length in various units."""

    def __init__(self):
        super().__init__("response_length")

    def calculate(self, response: str, **kwargs) -> MetricResult:
        """Calculate length metrics for the response."""
        # Basic counts
        char_count = len(response)
        word_count = len(response.split())

        # Sentence count (simple heuristic)
        sentence_endings = re.findall(r"[.!?]+", response)
        sentence_count = len(sentence_endings) if sentence_endings else 1

        # Line count
        line_count = len(response.splitlines())

        # Average word length
        words = response.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

        return MetricResult(
            name=self.name,
            value={
                "characters": char_count,
                "words": word_count,
                "sentences": sentence_count,
                "lines": line_count,
                "avg_word_length": round(avg_word_length, 2),
            },
            metadata={"empty_response": char_count == 0, "single_word": word_count == 1},
        )


class SentimentMetric(BaseMetric):
    """Analyzes sentiment of the response using simple heuristics.

    Note: For production use, integrate with TextBlob, VADER, or transformer-based
    sentiment models. This implementation uses keyword-based analysis as a fallback.
    """

    def __init__(self):
        super().__init__("sentiment")

        # Simple keyword lists for demonstration
        self.positive_words = {
            "good",
            "great",
            "excellent",
            "wonderful",
            "fantastic",
            "amazing",
            "positive",
            "helpful",
            "useful",
            "beneficial",
            "successful",
            "happy",
            "pleased",
            "delighted",
            "satisfied",
            "perfect",
            "best",
            "love",
        }

        self.negative_words = {
            "bad",
            "terrible",
            "awful",
            "horrible",
            "poor",
            "negative",
            "harmful",
            "useless",
            "failed",
            "unsuccessful",
            "unhappy",
            "disappointed",
            "dissatisfied",
            "worst",
            "hate",
            "error",
            "problem",
            "issue",
        }

        self.neutral_words = {
            "okay",
            "fine",
            "average",
            "normal",
            "regular",
            "standard",
            "acceptable",
            "adequate",
            "sufficient",
            "moderate",
        }

    def calculate(self, response: str, **kwargs) -> MetricResult:
        """Calculate sentiment scores for the response."""
        # Tokenize and lowercase
        words = response.lower().split()

        # Count sentiment words
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        neutral_count = sum(1 for word in words if word in self.neutral_words)

        total_sentiment_words = positive_count + negative_count + neutral_count

        # Calculate scores
        if total_sentiment_words == 0:
            sentiment_score = 0.0
            sentiment_label = "neutral"
        else:
            # Simple scoring: positive = +1, negative = -1
            sentiment_score = (positive_count - negative_count) / total_sentiment_words

            if sentiment_score > 0.3:
                sentiment_label = "positive"
            elif sentiment_score < -0.3:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"

        return MetricResult(
            name=self.name,
            value={
                "score": round(sentiment_score, 3),
                "label": sentiment_label,
                "positive_words": positive_count,
                "negative_words": negative_count,
                "neutral_words": neutral_count,
            },
            metadata={"method": "keyword_based", "total_words": len(words)},
        )


class CoherenceMetric(BaseMetric):
    """Measures text coherence using various heuristics.

    This implementation uses simple metrics like sentence connectivity
    and repetition. For better results, use embedding-based similarity
    or language model perplexity.
    """

    def __init__(self):
        super().__init__("coherence")

    def calculate(self, response: str, **kwargs) -> MetricResult:
        """Calculate coherence metrics for the response."""
        sentences = re.split(r"[.!?]+", response)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return MetricResult(
                name=self.name,
                value={
                    "score": 1.0,  # Single sentence is perfectly coherent with itself
                    "sentence_connectivity": 1.0,
                    "vocabulary_diversity": 1.0,
                    "repetition_score": 0.0,
                },
                metadata={"sentence_count": len(sentences)},
            )

        # Calculate sentence connectivity (shared words between adjacent sentences)
        connectivity_scores = []
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i + 1].lower().split())

            # Remove common stop words (simplified list)
            stop_words = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "is",
                "was",
                "are",
                "were",
            }
            words1 = words1 - stop_words
            words2 = words2 - stop_words

            if words1 and words2:
                overlap = len(words1 & words2)
                connectivity = overlap / min(len(words1), len(words2))
                connectivity_scores.append(connectivity)

        avg_connectivity = statistics.mean(connectivity_scores) if connectivity_scores else 0

        # Calculate vocabulary diversity
        all_words = response.lower().split()
        unique_words = set(all_words)
        vocabulary_diversity = len(unique_words) / len(all_words) if all_words else 0

        # Calculate repetition score (lower is better)
        word_freq = Counter(all_words)
        repetitions = sum(1 for count in word_freq.values() if count > 3)
        repetition_score = repetitions / len(unique_words) if unique_words else 0

        # Combined coherence score
        coherence_score = (
            0.4 * avg_connectivity
            + 0.4 * vocabulary_diversity
            + 0.2 * (1 - min(repetition_score, 1))
        )

        return MetricResult(
            name=self.name,
            value={
                "score": round(coherence_score, 3),
                "sentence_connectivity": round(avg_connectivity, 3),
                "vocabulary_diversity": round(vocabulary_diversity, 3),
                "repetition_score": round(repetition_score, 3),
            },
            metadata={
                "sentence_count": len(sentences),
                "unique_words": len(unique_words),
                "total_words": len(all_words),
            },
        )


class ResponseDiversityMetric(BaseMetric):
    """Measures diversity across multiple responses to the same prompt.

    Useful for evaluating whether a model produces varied outputs
    or tends to generate similar responses.
    """

    def __init__(self):
        super().__init__("response_diversity")

    def calculate(self, response: str, **kwargs) -> MetricResult:
        """Not applicable for single response. Use calculate_batch instead."""
        raise NotImplementedError(
            "ResponseDiversityMetric requires multiple responses. Use calculate_batch()"
        )

    def calculate_batch(self, responses: List[str], **kwargs) -> List[MetricResult]:
        """Calculate diversity metrics across multiple responses."""
        if len(responses) < 2:
            return [
                MetricResult(
                    name=self.name,
                    value={"error": "Need at least 2 responses for diversity calculation"},
                    metadata={"response_count": len(responses)},
                )
            ]

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                similarity = self._calculate_similarity(responses[i], responses[j])
                similarities.append(similarity)

        avg_similarity = statistics.mean(similarities)
        diversity_score = 1 - avg_similarity  # Higher diversity = lower similarity

        # Calculate lexical diversity across all responses
        all_words = []
        for response in responses:
            all_words.extend(response.lower().split())

        unique_words = set(all_words)
        lexical_diversity = len(unique_words) / len(all_words) if all_words else 0

        # Calculate response length variance
        lengths = [len(r.split()) for r in responses]
        length_variance = statistics.stdev(lengths) if len(lengths) > 1 else 0

        return [
            MetricResult(
                name=self.name,
                value={
                    "diversity_score": round(diversity_score, 3),
                    "avg_pairwise_similarity": round(avg_similarity, 3),
                    "lexical_diversity": round(lexical_diversity, 3),
                    "length_variance": round(length_variance, 2),
                    "response_count": len(responses),
                },
                metadata={
                    "min_similarity": round(min(similarities), 3),
                    "max_similarity": round(max(similarities), 3),
                },
            )
        ]

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using Jaccard similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0


class CustomMetric(BaseMetric):
    """Allows users to define custom evaluation metrics."""

    def __init__(self, name: str, evaluate_fn: callable):
        """Initialize with a custom evaluation function.

        Args:
            name: Name of the metric
            evaluate_fn: Function that takes (response, **kwargs) and returns a value
        """
        super().__init__(name)
        self.evaluate_fn = evaluate_fn

    def calculate(self, response: str, **kwargs) -> MetricResult:
        """Calculate the custom metric."""
        try:
            value = self.evaluate_fn(response, **kwargs)
            return MetricResult(name=self.name, value=value)
        except Exception as e:
            return MetricResult(
                name=self.name, value={"error": str(e)}, metadata={"error_type": type(e).__name__}
            )


class MetricSuite:
    """Manages a collection of metrics and provides aggregation functions."""

    def __init__(self, metrics: Optional[List[BaseMetric]] = None):
        """Initialize with a list of metrics."""
        self.metrics = metrics or self._get_default_metrics()

    def _get_default_metrics(self) -> List[BaseMetric]:
        """Get the default set of metrics."""
        return [ResponseLengthMetric(), SentimentMetric(), CoherenceMetric()]

    def add_metric(self, metric: BaseMetric) -> None:
        """Add a metric to the suite."""
        self.metrics.append(metric)

    def evaluate(self, response: str, **kwargs) -> Dict[str | Any]:
        """Evaluate a single response with all metrics.

        Args:
            response: The model's response
            **kwargs: Additional context

        Returns:
            Dictionary mapping metric names to results
        """
        results = {}
        for metric in self.metrics:
            if isinstance(metric, ResponseDiversityMetric):
                continue  # Skip diversity metric for single response

            result = metric.calculate(response, **kwargs)
            results[result.name] = result.to_dict()

        return results

    def evaluate_batch(self, responses: List[str], **kwargs) -> Dict[str | Any]:
        """Evaluate multiple responses with all metrics.

        Args:
            responses: List of model responses
            **kwargs: Additional context

        Returns:
            Dictionary with individual and aggregated results
        """
        # Individual results for each response
        individual_results = []
        for response in responses:
            individual_results.append(self.evaluate(response, **kwargs))

        # Aggregate results
        aggregated = self._aggregate_results(individual_results)

        # Add diversity metrics if applicable
        if len(responses) > 1:
            diversity_metric = ResponseDiversityMetric()
            diversity_results = diversity_metric.calculate_batch(responses, **kwargs)
            aggregated["diversity"] = diversity_results[0].to_dict()

        return {"individual": individual_results, "aggregated": aggregated}

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str | Any]:
        """Aggregate results across multiple responses."""
        if not results:
            return {}

        aggregated = {}

        # Aggregate each metric
        metric_names = results[0].keys()
        for metric_name in metric_names:
            metric_values = []
            for result in results:
                if metric_name in result and "value" in result[metric_name]:
                    value = result[metric_name]["value"]
                    if isinstance(value, dict):
                        # For dict values, aggregate each sub-value
                        if metric_name not in aggregated:
                            aggregated[metric_name] = {}
                        for key, val in value.items():
                            if isinstance(val, (int, float)):
                                if key not in aggregated[metric_name]:
                                    aggregated[metric_name][key] = []
                                aggregated[metric_name][key].append(val)
                    elif isinstance(value, (int, float)):
                        metric_values.append(value)

            # Calculate statistics for numeric values
            if metric_values:
                aggregated[metric_name] = {
                    "mean": round(statistics.mean(metric_values), 3),
                    "std": round(statistics.stdev(metric_values), 3)
                    if len(metric_values) > 1
                    else 0,
                    "min": min(metric_values),
                    "max": max(metric_values),
                }
            elif metric_name in aggregated and isinstance(aggregated[metric_name], dict):
                # Calculate statistics for each sub-value
                for key, values in list(aggregated[metric_name].items()):
                    if isinstance(values, list) and values:
                        aggregated[metric_name][key] = {
                            "mean": round(statistics.mean(values), 3),
                            "std": round(statistics.stdev(values), 3) if len(values) > 1 else 0,
                            "min": min(values),
                            "max": max(values),
                        }

        return aggregated


# Convenience functions
def evaluate_response(response: str, metrics: Optional[List[BaseMetric]] = None) -> Dict[str | Any]:
    """Evaluate a single response with the specified metrics.

    Args:
        response: The model's response
        metrics: List of metrics to use (defaults to standard suite)

    Returns:
        Dictionary of metric results
    """
    suite = MetricSuite(metrics)
    return suite.evaluate(response)


def evaluate_responses(
    responses: List[str], metrics: Optional[List[BaseMetric]] = None
) -> Dict[str | Any]:
    """Evaluate multiple responses with aggregation.

    Args:
        responses: List of model responses
        metrics: List of metrics to use (defaults to standard suite)

    Returns:
        Dictionary with individual and aggregated results
    """
    suite = MetricSuite(metrics)
    return suite.evaluate_batch(responses)
