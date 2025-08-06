"""
Task Evaluation Panel for Domain-Specific Metrics

This module provides evaluation panels for different task types including
translation (BLEU), classification (F1), and generation (perplexity) with
customizable evaluation sets.

Example:
    evaluator = TaskEvaluationPanel()

    # Evaluate translation task
    results = evaluator.evaluate_translation(
        predictions=["Bonjour le monde"],
        references=["Hello world"]
    )
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# NLP metrics
try:
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
    from nltk.translate.meteor_score import meteor_score

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# ML metrics
try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        mean_absolute_error,
        mean_squared_error,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of evaluation tasks."""

    TRANSLATION = "translation"
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    NAMED_ENTITY_RECOGNITION = "ner"
    SENTIMENT_ANALYSIS = "sentiment"
    CODE_GENERATION = "code_generation"


@dataclass
class EvaluationResult:
    """Result from task evaluation."""

    task_type: TaskType
    timestamp: datetime
    primary_metric: float
    primary_metric_name: str

    # Detailed metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Sample-level scores
    sample_scores: List[float] = field(default_factory=list)

    # Confidence intervals
    confidence_interval: Optional[Tuple[float, float]] = None

    # Metadata
    num_samples: int = 0
    evaluation_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str | Any]:
        """Convert to dictionary."""
        return {
            "task_type": self.task_type.value,
            "timestamp": self.timestamp.isoformat(),
            "primary_metric": self.primary_metric,
            "primary_metric_name": self.primary_metric_name,
            "metrics": self.metrics,
            "num_samples": self.num_samples,
            "confidence_interval": self.confidence_interval,
            "evaluation_time_seconds": self.evaluation_time_seconds,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    task_type: TaskType
    metrics_to_compute: List[str] = field(default_factory=list)
    batch_size: int = 32
    confidence_level: float = 0.95
    use_gpu: bool = False
    cache_results: bool = True
    custom_params: Dict[str, Any] = field(default_factory=dict)


class TaskEvaluationPanel:
    """Panel for task-specific evaluation."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize evaluation panel.

        Args:
            config: Evaluation configuration
        """
        self.config = config

        # Results cache
        self.results_cache: Dict[str, EvaluationResult] = {}

        # Metric functions registry
        self.metric_functions = {
            TaskType.TRANSLATION: {
                "bleu": self._compute_bleu,
                "meteor": self._compute_meteor,
                "ter": self._compute_ter,
                "chrf": self._compute_chrf,
            },
            TaskType.CLASSIFICATION: {
                "accuracy": self._compute_accuracy,
                "f1": self._compute_f1,
                "precision": self._compute_precision,
                "recall": self._compute_recall,
                "auc": self._compute_auc,
            },
            TaskType.GENERATION: {
                "perplexity": self._compute_perplexity,
                "bleu": self._compute_bleu,
                "rouge": self._compute_rouge,
                "distinct": self._compute_distinct,
            },
            TaskType.SUMMARIZATION: {
                "rouge": self._compute_rouge,
                "bert_score": self._compute_bert_score,
                "factuality": self._compute_factuality,
            },
            TaskType.QUESTION_ANSWERING: {
                "exact_match": self._compute_exact_match,
                "f1": self._compute_qa_f1,
                "partial_match": self._compute_partial_match,
            },
            TaskType.CODE_GENERATION: {
                "pass_at_k": self._compute_pass_at_k,
                "syntax_validity": self._compute_syntax_validity,
                "functional_correctness": self._compute_functional_correctness,
            },
        }

        # Download NLTK data if needed
        if NLTK_AVAILABLE:
            try:
                nltk.download("punkt", quiet=True)
                nltk.download("wordnet", quiet=True)
            except:
                pass

    def evaluate(
        self,
        task_type: TaskType,
        predictions: List[Any],
        references: List[Any],
        metrics: Optional[List[str]] = None,
        **kwargs,
    ) -> EvaluationResult:
        """Evaluate predictions for a specific task.

        Args:
            task_type: Type of task
            predictions: Model predictions
            references: Ground truth references
            metrics: Metrics to compute (None for all defaults)
            **kwargs: Additional parameters

        Returns:
            EvaluationResult object
        """
        start_time = datetime.now()

        # Get metrics to compute
        if metrics is None:
            metrics = list(self.metric_functions.get(task_type, {}).keys())

        # Compute metrics
        results = {}
        sample_scores = []

        for metric_name in metrics:
            if (
                task_type in self.metric_functions
                and metric_name in self.metric_functions[task_type]
            ):
                metric_fn = self.metric_functions[task_type][metric_name]
                try:
                    score = metric_fn(predictions, references, **kwargs)
                    if isinstance(score, tuple):
                        results[metric_name] = score[0]
                        if len(score) > 1:
                            sample_scores.extend(score[1])
                    else:
                        results[metric_name] = score
                except Exception as e:
                    logger.warning(f"Failed to compute {metric_name}: {e}")
                    results[metric_name] = None

        # Determine primary metric
        primary_metric_map = {
            TaskType.TRANSLATION: "bleu",
            TaskType.CLASSIFICATION: "f1",
            TaskType.GENERATION: "perplexity",
            TaskType.SUMMARIZATION: "rouge",
            TaskType.QUESTION_ANSWERING: "f1",
            TaskType.CODE_GENERATION: "pass_at_k",
        }

        primary_metric_name = primary_metric_map.get(
            task_type, metrics[0] if metrics else "unknown"
        )
        primary_metric = results.get(primary_metric_name, 0.0)

        # Calculate confidence interval
        confidence_interval = None
        if sample_scores:
            confidence_interval = self._calculate_confidence_interval(
                sample_scores,
                confidence_level=self.config.confidence_level if self.config else 0.95,
            )

        # Create result
        result = EvaluationResult(
            task_type=task_type,
            timestamp=datetime.now(),
            primary_metric=primary_metric,
            primary_metric_name=primary_metric_name,
            metrics=results,
            sample_scores=sample_scores,
            confidence_interval=confidence_interval,
            num_samples=len(predictions),
            evaluation_time_seconds=(datetime.now() - start_time).total_seconds(),
        )

        # Cache result
        cache_key = f"{task_type}_{len(predictions)}_{datetime.now().isoformat()}"
        self.results_cache[cache_key] = result

        return result

    def evaluate_translation(
        self, predictions: List[str], references: List[str], metrics: Optional[List[str]] = None
    ) -> EvaluationResult:
        """Evaluate translation task.

        Args:
            predictions: Predicted translations
            references: Reference translations
            metrics: Metrics to compute

        Returns:
            EvaluationResult
        """
        return self.evaluate(
            TaskType.TRANSLATION, predictions, references, metrics or ["bleu", "meteor"]
        )

    def evaluate_classification(
        self,
        predictions: List[int],
        references: List[int],
        metrics: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Evaluate classification task.

        Args:
            predictions: Predicted labels
            references: True labels
            metrics: Metrics to compute
            labels: Label names

        Returns:
            EvaluationResult
        """
        result = self.evaluate(
            TaskType.CLASSIFICATION,
            predictions,
            references,
            metrics or ["accuracy", "f1", "precision", "recall"],
            labels=labels,
        )

        # Add confusion matrix
        if SKLEARN_AVAILABLE:
            cm = confusion_matrix(references, predictions)
            result.metadata["confusion_matrix"] = cm.tolist()

        return result

    def evaluate_generation(
        self,
        predictions: List[str],
        references: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        model_probs: Optional[List[List[float]]] = None,
    ) -> EvaluationResult:
        """Evaluate text generation task.

        Args:
            predictions: Generated texts
            references: Reference texts (optional)
            metrics: Metrics to compute
            model_probs: Token probabilities for perplexity

        Returns:
            EvaluationResult
        """
        return self.evaluate(
            TaskType.GENERATION,
            predictions,
            references or predictions,
            metrics or ["perplexity", "distinct"],
            model_probs=model_probs,
        )

    # Metric computation functions

    def _compute_bleu(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> Tuple[float | List[float]]:
        """Compute BLEU score."""
        if not NLTK_AVAILABLE:
            return 0.0, []

        # Tokenize
        pred_tokens = [p.split() for p in predictions]
        ref_tokens = [r.split() for r in references]

        # Compute corpus BLEU
        corpus_score = corpus_bleu([[r] for r in ref_tokens], pred_tokens)

        # Compute sentence-level scores
        sentence_scores = []
        for pred, ref in zip(pred_tokens, ref_tokens):
            score = sentence_bleu([ref], pred)
            sentence_scores.append(score)

        return corpus_score, sentence_scores

    def _compute_meteor(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """Compute METEOR score."""
        if not NLTK_AVAILABLE:
            return 0.0

        scores = []
        for pred, ref in zip(predictions, references):
            score = meteor_score([ref], pred)
            scores.append(score)

        return np.mean(scores) if scores else 0.0

    def _compute_ter(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """Compute Translation Edit Rate."""
        # Simplified TER calculation
        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()

            # Calculate edit distance
            edits = self._edit_distance(pred_tokens, ref_tokens)
            ter = edits / len(ref_tokens) if ref_tokens else 0
            scores.append(ter)

        return np.mean(scores) if scores else 0.0

    def _compute_chrf(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """Compute chrF score."""
        # Character n-gram F-score
        scores = []
        for pred, ref in zip(predictions, references):
            # Character bigrams
            pred_bigrams = set(pred[i : i + 2] for i in range(len(pred) - 1))
            ref_bigrams = set(ref[i : i + 2] for i in range(len(ref) - 1))

            if not ref_bigrams:
                scores.append(0)
                continue

            intersection = pred_bigrams & ref_bigrams
            precision = len(intersection) / len(pred_bigrams) if pred_bigrams else 0
            recall = len(intersection) / len(ref_bigrams) if ref_bigrams else 0

            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            scores.append(f1)

        return np.mean(scores) if scores else 0.0

    def _compute_accuracy(self, predictions: List[int], references: List[int], **kwargs) -> float:
        """Compute accuracy."""
        if SKLEARN_AVAILABLE:
            return accuracy_score(references, predictions)
        else:
            correct = sum(1 for p, r in zip(predictions, references) if p == r)
            return correct / len(predictions) if predictions else 0.0

    def _compute_f1(self, predictions: List[int], references: List[int], **kwargs) -> float:
        """Compute F1 score."""
        if SKLEARN_AVAILABLE:
            _, _, f1, _ = precision_recall_fscore_support(
                references, predictions, average="weighted"
            )
            return f1
        else:
            # Simple binary F1
            tp = sum(1 for p, r in zip(predictions, references) if p == 1 and r == 1)
            fp = sum(1 for p, r in zip(predictions, references) if p == 1 and r == 0)
            fn = sum(1 for p, r in zip(predictions, references) if p == 0 and r == 1)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    def _compute_precision(self, predictions: List[int], references: List[int], **kwargs) -> float:
        """Compute precision."""
        if SKLEARN_AVAILABLE:
            precision, _, _, _ = precision_recall_fscore_support(
                references, predictions, average="weighted"
            )
            return precision
        else:
            tp = sum(1 for p, r in zip(predictions, references) if p == 1 and r == 1)
            fp = sum(1 for p, r in zip(predictions, references) if p == 1 and r == 0)
            return tp / (tp + fp) if (tp + fp) > 0 else 0

    def _compute_recall(self, predictions: List[int], references: List[int], **kwargs) -> float:
        """Compute recall."""
        if SKLEARN_AVAILABLE:
            _, recall, _, _ = precision_recall_fscore_support(
                references, predictions, average="weighted"
            )
            return recall
        else:
            tp = sum(1 for p, r in zip(predictions, references) if p == 1 and r == 1)
            fn = sum(1 for p, r in zip(predictions, references) if p == 0 and r == 1)
            return tp / (tp + fn) if (tp + fn) > 0 else 0

    def _compute_auc(self, predictions: List[float], references: List[int], **kwargs) -> float:
        """Compute AUC-ROC."""
        if SKLEARN_AVAILABLE:
            try:
                return roc_auc_score(references, predictions)
            except:
                return 0.0
        else:
            return 0.0

    def _compute_perplexity(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """Compute perplexity."""
        model_probs = kwargs.get("model_probs")

        if model_probs:
            # Calculate from token probabilities
            log_probs = []
            for probs in model_probs:
                if probs:
                    log_probs.extend(np.log(probs))

            if log_probs:
                avg_log_prob = np.mean(log_probs)
                return np.exp(-avg_log_prob)

        # Fallback: estimate from vocabulary coverage
        all_tokens = []
        for text in predictions:
            all_tokens.extend(text.split())

        vocab_size = len(set(all_tokens))
        total_tokens = len(all_tokens)

        # Simplified perplexity estimate
        return vocab_size / total_tokens if total_tokens > 0 else float("inf")

    def _compute_rouge(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> Dict[str | float]:
        """Compute ROUGE scores."""
        # Simplified ROUGE-L calculation
        scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()

            # Longest common subsequence
            lcs_length = self._lcs_length(pred_tokens, ref_tokens)

            precision = lcs_length / len(pred_tokens) if pred_tokens else 0
            recall = lcs_length / len(ref_tokens) if ref_tokens else 0

            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            scores.append(f1)

        return np.mean(scores) if scores else 0.0

    def _compute_distinct(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> Dict[str | float]:
        """Compute distinct n-grams."""
        all_unigrams = []
        all_bigrams = []

        for text in predictions:
            tokens = text.split()
            all_unigrams.extend(tokens)
            all_bigrams.extend(zip(tokens[:-1], tokens[1:]))

        distinct_1 = len(set(all_unigrams)) / len(all_unigrams) if all_unigrams else 0
        distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0

        return (distinct_1 + distinct_2) / 2

    def _compute_bert_score(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """Compute BERTScore."""
        # Placeholder - would require bert_score library
        return 0.0

    def _compute_factuality(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """Compute factuality score."""
        # Placeholder - would require fact-checking model
        return 0.0

    def _compute_exact_match(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> Tuple[float | List[float]]:
        """Compute exact match for QA."""
        scores = []
        for pred, ref in zip(predictions, references):
            score = 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0
            scores.append(score)

        return np.mean(scores), scores

    def _compute_qa_f1(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> Tuple[float | List[float]]:
        """Compute F1 for QA."""
        scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()

            if not pred_tokens or not ref_tokens:
                scores.append(0.0)
                continue

            common = set(pred_tokens) & set(ref_tokens)
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ref_tokens)

            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            scores.append(f1)

        return np.mean(scores), scores

    def _compute_partial_match(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> float:
        """Compute partial match score."""
        scores = []

        for pred, ref in zip(predictions, references):
            # Check if reference is contained in prediction or vice versa
            if ref.lower() in pred.lower() or pred.lower() in ref.lower():
                scores.append(1.0)
            else:
                # Partial credit based on overlap
                pred_tokens = set(pred.lower().split())
                ref_tokens = set(ref.lower().split())

                if ref_tokens:
                    overlap = len(pred_tokens & ref_tokens) / len(ref_tokens)
                    scores.append(overlap)
                else:
                    scores.append(0.0)

        return np.mean(scores) if scores else 0.0

    def _compute_pass_at_k(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """Compute pass@k for code generation."""
        # Simplified - would require code execution
        k = kwargs.get("k", 1)

        # Placeholder: check if code compiles
        passing = 0
        for pred in predictions[:k]:
            try:
                compile(pred, "<string>", "exec")
                passing += 1
            except:
                pass

        return passing / k if k > 0 else 0.0

    def _compute_syntax_validity(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> float:
        """Compute syntax validity for code."""
        valid = 0

        for pred in predictions:
            try:
                compile(pred, "<string>", "exec")
                valid += 1
            except:
                pass

        return valid / len(predictions) if predictions else 0.0

    def _compute_functional_correctness(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> float:
        """Compute functional correctness for code."""
        # Placeholder - would require test execution
        return 0.0

    # Helper functions

    def _edit_distance(self, s1: List[str], s2: List[str]) -> int:
        """Calculate edit distance between sequences."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        return dp[m][n]

    def _lcs_length(self, s1: List[str], s2: List[str]) -> int:
        """Calculate longest common subsequence length."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _calculate_confidence_interval(
        self, scores: List[float], confidence_level: float = 0.95
    ) -> Tuple[float | float]:
        """Calculate confidence interval."""
        if not scores:
            return (0.0, 0.0)

        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        n = len(scores)

        if n < 2:
            return (mean, mean)

        # Calculate confidence interval
        se = std / np.sqrt(n)
        ci = stats.t.interval(confidence_level, n - 1, loc=mean, scale=se)

        return ci


# Example usage
if __name__ == "__main__":
    # Create evaluator
    evaluator = TaskEvaluationPanel()

    # Evaluate translation
    print("Translation Evaluation:")
    translation_result = evaluator.evaluate_translation(
        predictions=["Hello world", "How are you today?"],
        references=["Hello world", "How are you?"],
    )
    print(f"BLEU Score: {translation_result.primary_metric:.4f}")
    print(f"All metrics: {translation_result.metrics}")

    # Evaluate classification
    print("\nClassification Evaluation:")
    classification_result = evaluator.evaluate_classification(
        predictions=[0, 1, 1, 0, 1], references=[0, 1, 0, 0, 1]
    )
    print(f"F1 Score: {classification_result.primary_metric:.4f}")
    print(f"Accuracy: {classification_result.metrics.get('accuracy', 0):.4f}")

    # Evaluate generation
    print("\nGeneration Evaluation:")
    generation_result = evaluator.evaluate_generation(
        predictions=[
            "The quick brown fox jumps over the lazy dog",
            "A beautiful day in the neighborhood",
        ]
    )
    print(f"Perplexity: {generation_result.primary_metric:.2f}")
    print(f"Distinct n-grams: {generation_result.metrics.get('distinct', 0):.4f}")
