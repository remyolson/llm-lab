"""
Custom Metrics for Fine-Tuning Evaluation

This module provides custom metric calculators for evaluating fine-tuned models
with task-specific metrics.

Example:
    calculator = MetricsCalculator()

    # Calculate BLEU score
    bleu = calculator.calculate_bleu(
        predictions=["Hello world"],
        references=["Hello world"]
    )

    # Calculate all metrics
    results = calculator.calculate_all_metrics(
        predictions=predictions,
        references=references,
        metric_names=["bleu", "rouge", "meteor"]
    )
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import logging
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Import metric libraries
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
    from nltk.translate.meteor_score import meteor_score

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer

    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score

    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

try:
    import evaluate

    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result from metric calculation."""

    name: str
    value: float
    details: Dict[str, Any] = None
    confidence_interval: Optional[Tuple[float, float]] = None


class MetricsCalculator:
    """Calculates various evaluation metrics for fine-tuned models."""

    def __init__(self):
        """Initialize metrics calculator."""
        self._check_dependencies()
        self._metric_cache = {}

        # Initialize ROUGE scorer if available
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )

    def _check_dependencies(self):
        """Check and download required dependencies."""
        if NLTK_AVAILABLE:
            try:
                # Download required NLTK data
                nltk.download("punkt", quiet=True)
                nltk.download("wordnet", quiet=True)
                nltk.download("averaged_perceptron_tagger", quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download NLTK data: {e}")

        if not ROUGE_AVAILABLE:
            logger.warning("rouge-score not available. ROUGE metrics will be disabled.")

        if not BERTSCORE_AVAILABLE:
            logger.warning("bert-score not available. BERTScore metric will be disabled.")

    def calculate_all_metrics(
        self,
        predictions: List[str],
        references: List[str | List[str]],
        metric_names: Optional[List[str]] = None,
    ) -> Dict[str | MetricResult]:
        """Calculate all requested metrics.

        Args:
            predictions: List of predicted texts
            references: List of reference texts (or list of lists for multiple refs)
            metric_names: Specific metrics to calculate

        Returns:
            Dictionary of metric results
        """
        if metric_names is None:
            metric_names = ["bleu", "rouge", "meteor", "bertscore", "exact_match", "f1"]

        results = {}

        for metric_name in metric_names:
            try:
                if metric_name == "bleu":
                    result = self.calculate_bleu(predictions, references)
                elif metric_name == "rouge":
                    result = self.calculate_rouge(predictions, references)
                elif metric_name == "meteor":
                    result = self.calculate_meteor(predictions, references)
                elif metric_name == "bertscore":
                    result = self.calculate_bertscore(predictions, references)
                elif metric_name == "exact_match":
                    result = self.calculate_exact_match(predictions, references)
                elif metric_name == "f1":
                    result = self.calculate_f1(predictions, references)
                elif metric_name == "perplexity":
                    # Perplexity requires model outputs, not just text
                    logger.warning("Perplexity calculation requires model outputs")
                    continue
                else:
                    logger.warning(f"Unknown metric: {metric_name}")
                    continue

                results[metric_name] = result
            except Exception as e:
                logger.error(f"Failed to calculate {metric_name}: {e}")

        return results

    def calculate_bleu(
        self,
        predictions: List[str],
        references: List[str | List[str]],
        weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
    ) -> MetricResult:
        """Calculate BLEU score.

        Args:
            predictions: List of predicted texts
            references: List of reference texts
            weights: N-gram weights

        Returns:
            BLEU metric result
        """
        if not NLTK_AVAILABLE:
            return MetricResult(name="bleu", value=0.0, details={"error": "NLTK not available"})

        # Tokenize predictions
        pred_tokens = [word_tokenize(pred.lower()) for pred in predictions]

        # Handle multiple references per prediction
        if isinstance(references[0], list):
            ref_tokens = [[word_tokenize(ref.lower()) for ref in refs] for refs in references]
        else:
            ref_tokens = [[word_tokenize(ref.lower())] for ref in references]

        # Calculate corpus BLEU
        bleu_score = corpus_bleu(ref_tokens, pred_tokens, weights=weights)

        # Calculate individual n-gram scores
        n_gram_scores = []
        for n in range(1, len(weights) + 1):
            n_weights = tuple([1.0 / n if i < n else 0 for i in range(len(weights))])
            n_score = corpus_bleu(ref_tokens, pred_tokens, weights=n_weights)
            n_gram_scores.append(n_score)

        return MetricResult(
            name="bleu",
            value=bleu_score,
            details={
                "bleu_1": n_gram_scores[0] if len(n_gram_scores) > 0 else 0,
                "bleu_2": n_gram_scores[1] if len(n_gram_scores) > 1 else 0,
                "bleu_3": n_gram_scores[2] if len(n_gram_scores) > 2 else 0,
                "bleu_4": n_gram_scores[3] if len(n_gram_scores) > 3 else 0,
            },
        )

    def calculate_rouge(
        self, predictions: List[str], references: List[str | List[str]]
    ) -> MetricResult:
        """Calculate ROUGE scores.

        Args:
            predictions: List of predicted texts
            references: List of reference texts

        Returns:
            ROUGE metric result
        """
        if not ROUGE_AVAILABLE:
            return MetricResult(
                name="rouge", value=0.0, details={"error": "rouge-score not available"}
            )

        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

        for pred, ref in zip(predictions, references):
            # Handle multiple references
            if isinstance(ref, list):
                # Take the best score among multiple references
                best_scores = None
                for r in ref:
                    scores = self.rouge_scorer.score(r, pred)
                    if best_scores is None:
                        best_scores = scores
                    else:
                        for key in scores:
                            if scores[key].fmeasure > best_scores[key].fmeasure:
                                best_scores[key] = scores[key]
                scores = best_scores
            else:
                scores = self.rouge_scorer.score(ref, pred)

            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)

        # Calculate average scores
        avg_scores = {key: np.mean(scores) for key, scores in rouge_scores.items()}

        # Overall ROUGE score (average of ROUGE-1, ROUGE-2, and ROUGE-L)
        overall_score = np.mean(list(avg_scores.values()))

        return MetricResult(name="rouge", value=overall_score, details=avg_scores)

    def calculate_meteor(
        self, predictions: List[str], references: List[str | List[str]]
    ) -> MetricResult:
        """Calculate METEOR score.

        Args:
            predictions: List of predicted texts
            references: List of reference texts

        Returns:
            METEOR metric result
        """
        if not NLTK_AVAILABLE:
            return MetricResult(name="meteor", value=0.0, details={"error": "NLTK not available"})

        meteor_scores = []

        for pred, ref in zip(predictions, references):
            # Tokenize
            pred_tokens = word_tokenize(pred.lower())

            # Handle multiple references
            if isinstance(ref, list):
                # Take the best score among multiple references
                ref_scores = []
                for r in ref:
                    ref_tokens = word_tokenize(r.lower())
                    score = meteor_score([ref_tokens], pred_tokens)
                    ref_scores.append(score)
                score = max(ref_scores)
            else:
                ref_tokens = word_tokenize(ref.lower())
                score = meteor_score([ref_tokens], pred_tokens)

            meteor_scores.append(score)

        avg_meteor = np.mean(meteor_scores)

        return MetricResult(
            name="meteor",
            value=avg_meteor,
            details={"scores": meteor_scores, "std": np.std(meteor_scores)},
        )

    def calculate_bertscore(
        self,
        predictions: List[str],
        references: List[str | List[str]],
        model_type: str = "bert-base-uncased",
    ) -> MetricResult:
        """Calculate BERTScore.

        Args:
            predictions: List of predicted texts
            references: List of reference texts
            model_type: BERT model to use for scoring

        Returns:
            BERTScore metric result
        """
        if not BERTSCORE_AVAILABLE:
            return MetricResult(
                name="bertscore", value=0.0, details={"error": "bert-score not available"}
            )

        # Handle multiple references by flattening
        if isinstance(references[0], list):
            # For multiple references, we'll compare against all and take the best
            all_refs = []
            all_preds = []
            for pred, refs in zip(predictions, references):
                for ref in refs:
                    all_refs.append(ref)
                    all_preds.append(pred)
        else:
            all_preds = predictions
            all_refs = references

        # Calculate BERTScore
        P, R, F1 = bert_score(all_preds, all_refs, model_type=model_type, verbose=False)

        # If we had multiple references, aggregate scores
        if isinstance(references[0], list):
            # Reshape and take max for each prediction
            n_refs = len(references[0])
            P = P.view(-1, n_refs).max(dim=1)[0]
            R = R.view(-1, n_refs).max(dim=1)[0]
            F1 = F1.view(-1, n_refs).max(dim=1)[0]

        return MetricResult(
            name="bertscore",
            value=F1.mean().item(),
            details={
                "precision": P.mean().item(),
                "recall": R.mean().item(),
                "f1": F1.mean().item(),
                "std": F1.std().item(),
            },
        )

    def calculate_exact_match(
        self, predictions: List[str], references: List[str | List[str]]
    ) -> MetricResult:
        """Calculate exact match accuracy.

        Args:
            predictions: List of predicted texts
            references: List of reference texts

        Returns:
            Exact match metric result
        """

        def normalize_text(s):
            """Normalize text for comparison."""
            # Convert to lowercase
            s = s.lower()
            # Remove punctuation
            s = s.translate(str.maketrans("", "", string.punctuation))
            # Remove extra whitespace
            s = " ".join(s.split())
            return s

        exact_matches = 0

        for pred, ref in zip(predictions, references):
            pred_norm = normalize_text(pred)

            # Handle multiple references
            if isinstance(ref, list):
                ref_match = any(pred_norm == normalize_text(r) for r in ref)
            else:
                ref_match = pred_norm == normalize_text(ref)

            if ref_match:
                exact_matches += 1

        accuracy = exact_matches / len(predictions) if predictions else 0

        return MetricResult(
            name="exact_match",
            value=accuracy,
            details={"matches": exact_matches, "total": len(predictions)},
        )

    def calculate_f1(
        self, predictions: List[str], references: List[str | List[str]]
    ) -> MetricResult:
        """Calculate token-level F1 score.

        Args:
            predictions: List of predicted texts
            references: List of reference texts

        Returns:
            F1 metric result
        """

        def get_tokens(s):
            """Get normalized tokens from text."""
            s = s.lower()
            s = s.translate(str.maketrans("", "", string.punctuation))
            return s.split()

        f1_scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = get_tokens(pred)

            # Handle multiple references
            if isinstance(ref, list):
                # Take the best F1 among multiple references
                ref_f1s = []
                for r in ref:
                    ref_tokens = get_tokens(r)
                    f1 = self._calculate_token_f1(pred_tokens, ref_tokens)
                    ref_f1s.append(f1)
                f1 = max(ref_f1s)
            else:
                ref_tokens = get_tokens(ref)
                f1 = self._calculate_token_f1(pred_tokens, ref_tokens)

            f1_scores.append(f1)

        avg_f1 = np.mean(f1_scores)

        return MetricResult(
            name="f1", value=avg_f1, details={"scores": f1_scores, "std": np.std(f1_scores)}
        )

    def _calculate_token_f1(self, pred_tokens: List[str], ref_tokens: List[str]) -> float:
        """Calculate F1 score between two token lists."""
        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0

        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)

        # Calculate overlap
        overlap = sum((pred_counter & ref_counter).values())

        # Calculate precision and recall
        precision = overlap / len(pred_tokens) if pred_tokens else 0
        recall = overlap / len(ref_tokens) if ref_tokens else 0

        # Calculate F1
        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def calculate_perplexity_from_loss(self, loss: float) -> MetricResult:
        """Calculate perplexity from loss value.

        Args:
            loss: Cross-entropy loss value

        Returns:
            Perplexity metric result
        """
        perplexity = torch.exp(torch.tensor(loss)).item()

        return MetricResult(name="perplexity", value=perplexity, details={"loss": loss})

    def calculate_diversity_metrics(
        self, generated_texts: List[str], n_gram_sizes: List[int] = [1, 2, 3]
    ) -> Dict[str | MetricResult]:
        """Calculate diversity metrics for generated texts.

        Args:
            generated_texts: List of generated texts
            n_gram_sizes: N-gram sizes to calculate diversity for

        Returns:
            Dictionary of diversity metrics
        """
        results = {}

        for n in n_gram_sizes:
            # Extract n-grams
            all_ngrams = []
            for text in generated_texts:
                tokens = text.lower().split()
                ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
                all_ngrams.extend(ngrams)

            # Calculate diversity
            unique_ngrams = len(set(all_ngrams))
            total_ngrams = len(all_ngrams)

            diversity = unique_ngrams / total_ngrams if total_ngrams > 0 else 0

            results[f"diversity_{n}gram"] = MetricResult(
                name=f"diversity_{n}gram",
                value=diversity,
                details={"unique": unique_ngrams, "total": total_ngrams},
            )

        # Calculate self-BLEU (measure of diversity)
        if len(generated_texts) > 1:
            self_bleu_scores = []
            for i, text in enumerate(generated_texts):
                other_texts = generated_texts[:i] + generated_texts[i + 1 :]
                bleu_result = self.calculate_bleu([text], [other_texts])
                self_bleu_scores.append(bleu_result.value)

            # Lower self-BLEU means higher diversity
            avg_self_bleu = np.mean(self_bleu_scores)
            diversity_score = 1 - avg_self_bleu

            results["self_bleu_diversity"] = MetricResult(
                name="self_bleu_diversity",
                value=diversity_score,
                details={"self_bleu": avg_self_bleu, "scores": self_bleu_scores},
            )

        return results

    def calculate_task_specific_metrics(
        self, task_type: str, predictions: List[Any], references: List[Any], **kwargs
    ) -> Dict[str | MetricResult]:
        """Calculate task-specific metrics.

        Args:
            task_type: Type of task (e.g., 'classification', 'ner', 'translation')
            predictions: Task predictions
            references: Task references
            **kwargs: Additional task-specific parameters

        Returns:
            Dictionary of task-specific metrics
        """
        results = {}

        if task_type == "classification":
            # Calculate classification metrics
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support

            accuracy = accuracy_score(references, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                references, predictions, average="weighted"
            )

            results["accuracy"] = MetricResult("accuracy", accuracy)
            results["precision"] = MetricResult("precision", precision)
            results["recall"] = MetricResult("recall", recall)
            results["f1"] = MetricResult("f1", f1)

        elif task_type == "ner":
            # Calculate NER metrics
            # Would need specialized NER evaluation logic
            pass

        elif task_type == "translation":
            # Calculate translation metrics
            bleu = self.calculate_bleu(predictions, references)
            results["bleu"] = bleu

            if ROUGE_AVAILABLE:
                rouge = self.calculate_rouge(predictions, references)
                results["rouge"] = rouge

        return results


# Example usage
if __name__ == "__main__":
    # Initialize calculator
    calculator = MetricsCalculator()

    # Example predictions and references
    predictions = [
        "The cat sat on the mat.",
        "I love machine learning.",
        "Natural language processing is amazing.",
    ]

    references = [
        "The cat was sitting on the mat.",
        "I enjoy machine learning.",
        "NLP is wonderful.",
    ]

    # Calculate all metrics
    results = calculator.calculate_all_metrics(
        predictions=predictions,
        references=references,
        metric_names=["bleu", "rouge", "meteor", "exact_match", "f1"],
    )

    # Print results
    print("Metric Results:")
    for metric_name, result in results.items():
        print(f"\n{metric_name}:")
        print(f"  Value: {result.value:.4f}")
        if result.details:
            print(f"  Details: {result.details}")

    # Calculate diversity metrics
    generated = ["Hello world!", "How are you?", "Hello there!", "Good morning!", "How's it going?"]

    diversity_results = calculator.calculate_diversity_metrics(generated)

    print("\n\nDiversity Metrics:")
    for metric_name, result in diversity_results.items():
        print(f"\n{metric_name}:")
        print(f"  Value: {result.value:.4f}")
        print(f"  Details: {result.details}")
