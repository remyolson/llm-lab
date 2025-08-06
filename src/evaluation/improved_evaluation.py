"""
Improved Evaluation Methods for LLM Responses

This module provides more sophisticated evaluation methods beyond simple keyword matching,
including fuzzy matching, semantic similarity, and partial phrase matching.
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import logging
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from ..types.evaluation import EvaluationResult, FuzzyMatchResult
from ..types.generics import GenericEvaluator
from ..types.protocols import ResultType

# Optional imports for enhanced evaluation
try:
    from fuzzywuzzy import fuzz

    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False
    logging.info(
        "fuzzywuzzy not available. Install with: pip install fuzzywuzzy python-levenshtein"
    )


def fuzzy_keyword_match(
    response: str,
    expected_keywords: List[str],
    threshold: float = 0.75,
    partial_threshold: float = 0.60,
) -> "FuzzyMatchResult":
    """
    Enhanced keyword matching using fuzzy string similarity.

    This method orchestrates the fuzzy matching process by delegating to
    specialized helper functions following the Extract Method pattern.

    Args:
        response: The model's generated response text
        expected_keywords: List of keywords/phrases to search for
        threshold: Minimum similarity score for exact matches (0.0-1.0)
        partial_threshold: Minimum similarity score for partial matches (0.0-1.0)

    Returns:
        Dict containing evaluation results with similarity scores

    Examples:
        >>> # Exact match
        >>> result = fuzzy_keyword_match(
        ...     "Machine learning is a subset of artificial intelligence",
        ...     ["machine learning", "artificial intelligence"]
        ... )
        >>> result['success']
        True
        >>> len(result['matched_keywords'])
        2

        >>> # Fuzzy match with typos
        >>> result = fuzzy_keyword_match(
        ...     "Machne learing is powerful",
        ...     ["machine learning"],
        ...     threshold=0.8
        ... )
        >>> len(result['partial_matches']) > 0
        True

        >>> # No match
        >>> result = fuzzy_keyword_match(
        ...     "Completely unrelated text",
        ...     ["quantum computing"],
        ...     threshold=0.9
        ... )
        >>> result['success']
        False
        >>> result['score']
        0.0
    """
    # Input validation
    if not response or not expected_keywords:
        return _create_empty_fuzzy_result("Empty response or keywords")

    # Process keywords and calculate similarities
    response_lower = response.lower().strip()
    matched_keywords, partial_matches, similarity_scores = _process_keywords(
        response_lower, expected_keywords, threshold, partial_threshold
    )

    # Calculate final results
    return _calculate_fuzzy_results(
        matched_keywords, partial_matches, similarity_scores, threshold, partial_threshold
    )


def _create_empty_fuzzy_result(error_message: str) -> Dict[str | Any]:
    """Create an empty result for fuzzy matching when validation fails."""
    return {
        "success": False,
        "score": 0.0,
        "matched_keywords": [],
        "partial_matches": [],
        "method": "fuzzy_keyword_match",
        "details": {"error": error_message},
    }


def _process_keywords(
    response_lower: str, expected_keywords: List[str], threshold: float, partial_threshold: float
) -> tuple:
    """Process each keyword and categorize matches.

    Returns:
        Tuple of (matched_keywords, partial_matches, similarity_scores)
    """
    matched_keywords = []
    partial_matches = []
    similarity_scores = []

    for keyword in expected_keywords:
        if not keyword or not isinstance(keyword, str):
            continue

        keyword_lower = keyword.lower().strip()
        best_score = _calculate_keyword_similarity(keyword_lower, response_lower)

        # Categorize matches based on score
        _categorize_match(
            keyword, best_score, threshold, partial_threshold, matched_keywords, partial_matches
        )

        similarity_scores.append(best_score)

    return matched_keywords, partial_matches, similarity_scores


def _calculate_keyword_similarity(keyword_lower: str, response_lower: str) -> float:
    """Calculate similarity score between keyword and response."""
    # Method 1: Check if keyword appears as substring (high weight)
    if keyword_lower in response_lower:
        return 1.0

    # Method 2: Fuzzy matching
    if FUZZYWUZZY_AVAILABLE:
        scores = [
            fuzz.partial_ratio(keyword_lower, response_lower) / 100.0,
            fuzz.token_sort_ratio(keyword_lower, response_lower) / 100.0,
            fuzz.token_set_ratio(keyword_lower, response_lower) / 100.0,
        ]
        return max(scores)
    else:
        # Fallback: Basic similarity matching
        return SequenceMatcher(None, keyword_lower, response_lower).ratio()


def _categorize_match(
    keyword: str,
    score: float,
    threshold: float,
    partial_threshold: float,
    matched_keywords: List[Dict],
    partial_matches: List[Dict],
) -> None:
    """Categorize a keyword match based on its similarity score."""
    if score >= threshold:
        matched_keywords.append(
            {
                "keyword": keyword,
                "score": score,
                "match_type": "exact" if score >= 0.95 else "fuzzy",
            }
        )
    elif score >= partial_threshold:
        partial_matches.append({"keyword": keyword, "score": score, "match_type": "partial"})


def _calculate_fuzzy_results(
    matched_keywords: List[Dict],
    partial_matches: List[Dict],
    similarity_scores: List[float],
    threshold: float,
    partial_threshold: float,
) -> Dict[str | Any]:
    """Calculate final fuzzy matching results."""
    exact_matches = len(matched_keywords)
    partial_match_count = len(partial_matches)

    # Success if we have exact matches, or multiple good partial matches
    success = exact_matches > 0 or partial_match_count >= 2

    # Score calculation: weighted by match quality
    if exact_matches > 0:
        score = max(m["score"] for m in matched_keywords)
    elif partial_match_count > 0:
        score = max(m["score"] for m in partial_matches) * 0.7  # Penalty for partial
    else:
        score = 0.0

    return {
        "success": success,
        "score": score,
        "matched_keywords": [m["keyword"] for m in matched_keywords],
        "partial_matches": [m["keyword"] for m in partial_matches],
        "method": "fuzzy_keyword_match",
        "details": {
            "exact_matches": exact_matches,
            "partial_matches": partial_match_count,
            "threshold": threshold,
            "partial_threshold": partial_threshold,
            "max_similarity": max(similarity_scores) if similarity_scores else 0.0,
            "all_scores": similarity_scores,
            "full_match_details": matched_keywords + partial_matches,
        },
    }


def semantic_phrase_match(
    response: str, expected_keywords: List[str], similarity_threshold: float = 0.6
) -> Dict[str | Any]:
    """
    Semantic matching that breaks down phrases and looks for conceptual similarity.

    Args:
        response: The model's generated response text
        expected_keywords: List of expected phrases
        similarity_threshold: Minimum similarity for success

    Returns:
        Dict containing evaluation results
    """
    if not response or not expected_keywords:
        return {
            "success": False,
            "score": 0.0,
            "matched_keywords": [],
            "method": "semantic_phrase_match",
            "details": {"error": "Empty response or keywords"},
        }

    response_lower = response.lower()
    matched_concepts = []

    for keyword in expected_keywords:
        if not keyword or not isinstance(keyword, str):
            continue

        keyword_lower = keyword.lower().strip()

        # Break down the expected phrase into key concepts
        key_concepts = extract_key_concepts(keyword_lower)
        concept_matches = 0

        for concept in key_concepts:
            # Look for concept or synonyms in response
            if find_concept_in_text(concept, response_lower):
                concept_matches += 1

        # Calculate concept match ratio
        if key_concepts:
            concept_ratio = concept_matches / len(key_concepts)
            if concept_ratio >= similarity_threshold:
                matched_concepts.append(
                    {
                        "keyword": keyword,
                        "concept_ratio": concept_ratio,
                        "matched_concepts": concept_matches,
                        "total_concepts": len(key_concepts),
                    }
                )

    success = len(matched_concepts) > 0
    score = max([m["concept_ratio"] for m in matched_concepts]) if matched_concepts else 0.0

    return {
        "success": success,
        "score": score,
        "matched_keywords": [m["keyword"] for m in matched_concepts],
        "method": "semantic_phrase_match",
        "details": {
            "matched_concepts": len(matched_concepts),
            "concept_details": matched_concepts,
            "threshold": similarity_threshold,
        },
    }


def extract_key_concepts(phrase: str) -> List[str]:
    """Extract key concepts from a phrase, filtering out common words."""
    # Remove common stop words and extract meaningful terms
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
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
    }

    # Split and clean
    words = re.findall(r"\b\w+\b", phrase.lower())
    concepts = [word for word in words if word not in stop_words and len(word) > 2]

    # Also include important multi-word phrases
    important_phrases = []
    phrase_patterns = [
        r"pass through",
        r"digestive system",
        r"nothing happens",
        r"originated in",
        r"light wavelength",
        r"appear blue",
    ]

    for pattern in phrase_patterns:
        if re.search(pattern, phrase):
            important_phrases.append(pattern.replace(r"\b", "").replace(r"\s+", " "))

    return concepts + important_phrases


def find_concept_in_text(concept: str, text: str) -> bool:
    """Find a concept in text, including synonyms and variations."""
    # Direct match
    if concept in text:
        return True

    # Common synonyms and variations
    synonyms = {
        "pass": ["goes", "moves", "travels", "flows"],
        "through": ["via", "across", "within"],
        "system": ["tract", "pathway", "process"],
        "nothing": ["no", "not", "none"],
        "happens": ["occurs", "takes place", "results"],
        "originated": ["began", "started", "came from", "invented"],
        "blue": ["bluish", "blue-colored"],
    }

    # Check synonyms
    for word, syns in synonyms.items():
        if word in concept:
            for syn in syns:
                if syn in text:
                    return True

    return False


def multi_method_evaluation(
    response: str, expected_keywords: List[str], methods: List[str | None] = None
) -> Dict[str | Any]:
    """
    Combine multiple evaluation methods for more robust assessment.

    This method orchestrates multi-method evaluation by delegating to
    specialized helper functions following the Extract Method pattern.

    Args:
        response: The model's generated response text
        expected_keywords: List of expected keywords/phrases
        methods: List of methods to use ['fuzzy', 'semantic', 'original']

    Returns:
        Dict containing combined evaluation results
    """
    if methods is None:
        methods = ["fuzzy", "semantic"]

    # Execute all evaluation methods
    individual_results = _execute_evaluation_methods(response, expected_keywords, methods)

    # Combine results using weighted voting
    return _combine_method_results(individual_results)


def _execute_evaluation_methods(
    response: str, expected_keywords: List[str], methods: List[str]
) -> Dict[str | Any]:
    """Execute specified evaluation methods and return individual results."""
    results = {}

    if "fuzzy" in methods:
        results["fuzzy"] = fuzzy_keyword_match(response, expected_keywords)

    if "semantic" in methods:
        results["semantic"] = semantic_phrase_match(response, expected_keywords)

    if "original" in methods:
        from .keyword_match import keyword_match

        results["original"] = keyword_match(response, expected_keywords)

    return results


def _combine_method_results(individual_results: Dict[str, Any]) -> Dict[str | Any]:
    """Combine individual method results using weighted voting."""
    method_weights = {"fuzzy": 0.4, "semantic": 0.4, "original": 0.2}

    # Aggregate weighted scores and votes
    aggregated_data = _aggregate_method_scores(individual_results, method_weights)

    # Make final decision
    final_score = aggregated_data["final_score"]
    final_success = _determine_final_success(
        aggregated_data["success_votes"], len(individual_results), final_score
    )

    return {
        "success": final_success,
        "score": final_score,
        "matched_keywords": list(aggregated_data["all_matched"]),
        "method": "multi_method_evaluation",
        "details": {
            "individual_results": individual_results,
            "success_votes": aggregated_data["success_votes"],
            "total_methods": len(individual_results),
            "combined_score": final_score,
            "methods_used": list(individual_results.keys()),
        },
    }


def _aggregate_method_scores(
    individual_results: Dict[str, Any], method_weights: Dict[str, float]
) -> Dict[str | Any]:
    """Aggregate scores and votes from individual method results."""
    total_score = 0.0
    total_weight = 0.0
    success_votes = 0
    all_matched = set()

    for method_name, result in individual_results.items():
        if method_name in method_weights:
            weight = method_weights[method_name]
            total_score += result["score"] * weight
            total_weight += weight

            if result["success"]:
                success_votes += 1

            # Collect all matched keywords
            all_matched.update(result.get("matched_keywords", []))

    final_score = total_score / total_weight if total_weight > 0 else 0.0

    return {
        "final_score": final_score,
        "success_votes": success_votes,
        "all_matched": all_matched,
    }


def _determine_final_success(success_votes: int, total_methods: int, final_score: float) -> bool:
    """Determine final success based on voting and score thresholds."""
    # Success if majority of methods agree OR score is high enough
    return success_votes >= total_methods / 2 or final_score >= 0.6


class FuzzyKeywordEvaluator(GenericEvaluator[Tuple[str, List[str]], EvaluationResult]):
    """Generic fuzzy keyword evaluator with type safety.

    Implements the GenericEvaluator pattern for fuzzy keyword matching
    with proper type constraints.
    """

    def __init__(self, threshold: float = 0.75, partial_threshold: float = 0.60):
        self.threshold = threshold
        self.partial_threshold = partial_threshold

    def evaluate(self, data: Tuple[str, List[str]], **kwargs: Any) -> EvaluationResult:
        """Evaluate using fuzzy keyword matching.

        Args:
            data: Tuple of (response_text, expected_keywords)
            **kwargs: Additional evaluation parameters

        Returns:
            Typed evaluation result
        """
        response, expected_keywords = data

        # Override thresholds if provided
        threshold = kwargs.get("threshold", self.threshold)
        partial_threshold = kwargs.get("partial_threshold", self.partial_threshold)

        # Use existing fuzzy_keyword_match function
        result = fuzzy_keyword_match(response, expected_keywords, threshold, partial_threshold)

        # Convert to EvaluationResult format
        return EvaluationResult(
            {
                "score": result["score"],
                "method": result["method"],
                "timestamp": datetime.now().isoformat(),
                "confidence": result.get("confidence"),
                "details": result.get("details"),
                "metadata": {
                    "threshold": threshold,
                    "partial_threshold": partial_threshold,
                    "matched_keywords": result.get("matched_keywords", []),
                    "partial_matches": result.get("partial_matches", []),
                },
            }
        )

    def get_method_name(self) -> str:
        """Get the name of this evaluation method."""
        return "fuzzy_keyword_match"


class SemanticPhraseEvaluator(GenericEvaluator[Tuple[str, List[str]], EvaluationResult]):
    """Generic semantic phrase evaluator with type safety.

    Implements the GenericEvaluator pattern for semantic phrase matching.
    """

    def __init__(self, similarity_threshold: float = 0.6):
        self.similarity_threshold = similarity_threshold

    def evaluate(self, data: Tuple[str, List[str]], **kwargs: Any) -> EvaluationResult:
        """Evaluate using semantic phrase matching.

        Args:
            data: Tuple of (response_text, expected_keywords)
            **kwargs: Additional evaluation parameters

        Returns:
            Typed evaluation result
        """
        response, expected_keywords = data

        # Override threshold if provided
        similarity_threshold = kwargs.get("similarity_threshold", self.similarity_threshold)

        # Use existing semantic_phrase_match function
        result = semantic_phrase_match(response, expected_keywords, similarity_threshold)

        # Convert to EvaluationResult format
        return EvaluationResult(
            {
                "score": result["score"],
                "method": result["method"],
                "timestamp": datetime.now().isoformat(),
                "details": result.get("details"),
                "metadata": {
                    "similarity_threshold": similarity_threshold,
                    "matched_keywords": result.get("matched_keywords", []),
                },
            }
        )

    def get_method_name(self) -> str:
        """Get the name of this evaluation method."""
        return "semantic_phrase_match"


class MultiMethodEvaluator(GenericEvaluator[Tuple[str, List[str]], EvaluationResult]):
    """Generic multi-method evaluator with configurable evaluation methods.

    Combines multiple evaluation strategies with proper type safety.
    """

    def __init__(self, methods: List[str] | None = None, weights: Dict[str, float] | None = None):
        self.methods = methods or ["fuzzy", "semantic"]
        self.weights = weights or {"fuzzy": 0.4, "semantic": 0.4, "original": 0.2}

        # Initialize sub-evaluators
        self._evaluators = {"fuzzy": FuzzyKeywordEvaluator(), "semantic": SemanticPhraseEvaluator()}

    def evaluate(self, data: Tuple[str, List[str]], **kwargs: Any) -> EvaluationResult:
        """Evaluate using multiple methods with weighted combination.

        Args:
            data: Tuple of (response_text, expected_keywords)
            **kwargs: Additional evaluation parameters

        Returns:
            Combined typed evaluation result
        """
        response, expected_keywords = data

        # Use existing multi_method_evaluation function
        result = multi_method_evaluation(response, expected_keywords, self.methods)

        # Convert to EvaluationResult format
        return EvaluationResult(
            {
                "score": result["score"],
                "method": result["method"],
                "timestamp": datetime.now().isoformat(),
                "details": result.get("details"),
                "metadata": {
                    "methods_used": self.methods,
                    "weights": self.weights,
                    "individual_results": result.get("details", {}).get("individual_results", {}),
                },
            }
        )

    def get_method_name(self) -> str:
        """Get the name of this evaluation method."""
        return "multi_method_evaluation"
