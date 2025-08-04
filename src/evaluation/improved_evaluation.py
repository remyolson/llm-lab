"""
Improved Evaluation Methods for LLM Responses

This module provides more sophisticated evaluation methods beyond simple keyword matching,
including fuzzy matching, semantic similarity, and partial phrase matching.
"""

import re
from typing import Any, Dict, List, Optional
from difflib import SequenceMatcher
import logging

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
) -> Dict[str, Any]:
    """
    Enhanced keyword matching using fuzzy string similarity.

    Args:
        response: The model's generated response text
        expected_keywords: List of keywords/phrases to search for
        threshold: Minimum similarity score for exact matches (0.0-1.0)
        partial_threshold: Minimum similarity score for partial matches (0.0-1.0)

    Returns:
        Dict containing evaluation results with similarity scores
    """
    if not response or not expected_keywords:
        return {
            "success": False,
            "score": 0.0,
            "matched_keywords": [],
            "partial_matches": [],
            "method": "fuzzy_keyword_match",
            "details": {"error": "Empty response or keywords"},
        }

    response_lower = response.lower().strip()
    matched_keywords = []
    partial_matches = []
    similarity_scores = []

    for keyword in expected_keywords:
        if not keyword or not isinstance(keyword, str):
            continue

        keyword_lower = keyword.lower().strip()
        best_score = 0.0
        best_match_text = ""

        # Method 1: Check if keyword appears as substring (high weight)
        if keyword_lower in response_lower:
            best_score = 1.0
            best_match_text = keyword
        else:
            # Method 2: Fuzzy matching with sliding window
            if FUZZYWUZZY_AVAILABLE:
                # Try different fuzzy matching approaches
                scores = [
                    fuzz.partial_ratio(keyword_lower, response_lower) / 100.0,
                    fuzz.token_sort_ratio(keyword_lower, response_lower) / 100.0,
                    fuzz.token_set_ratio(keyword_lower, response_lower) / 100.0,
                ]
                best_score = max(scores)
            else:
                # Fallback: Basic similarity matching
                best_score = SequenceMatcher(None, keyword_lower, response_lower).ratio()

            best_match_text = keyword

        # Categorize matches
        if best_score >= threshold:
            matched_keywords.append(
                {
                    "keyword": keyword,
                    "score": best_score,
                    "match_type": "exact" if best_score >= 0.95 else "fuzzy",
                }
            )
        elif best_score >= partial_threshold:
            partial_matches.append(
                {"keyword": keyword, "score": best_score, "match_type": "partial"}
            )

        similarity_scores.append(best_score)

    # Calculate overall success and score
    exact_matches = len(matched_keywords)
    partial_match_count = len(partial_matches)

    # Success if we have exact matches, or multiple good partial matches
    success = exact_matches > 0 or partial_match_count >= 2

    # Score calculation: weighted by match quality
    if exact_matches > 0:
        score = max([m["score"] for m in matched_keywords])
    elif partial_match_count > 0:
        score = max([m["score"] for m in partial_matches]) * 0.7  # Penalty for partial
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
) -> Dict[str, Any]:
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
    response: str, expected_keywords: List[str], methods: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Combine multiple evaluation methods for more robust assessment.

    Args:
        response: The model's generated response text
        expected_keywords: List of expected keywords/phrases
        methods: List of methods to use ['fuzzy', 'semantic', 'original']

    Returns:
        Dict containing combined evaluation results
    """
    if methods is None:
        methods = ["fuzzy", "semantic"]

    results = {}

    # Run each evaluation method
    if "fuzzy" in methods:
        results["fuzzy"] = fuzzy_keyword_match(response, expected_keywords)

    if "semantic" in methods:
        results["semantic"] = semantic_phrase_match(response, expected_keywords)

    if "original" in methods:
        # Import and use the original keyword match
        from .keyword_match import keyword_match

        results["original"] = keyword_match(response, expected_keywords)

    # Combine results using weighted voting
    method_weights = {"fuzzy": 0.4, "semantic": 0.4, "original": 0.2}

    total_score = 0.0
    total_weight = 0.0
    success_votes = 0
    all_matched = set()

    for method_name, result in results.items():
        if method_name in method_weights:
            weight = method_weights[method_name]
            total_score += result["score"] * weight
            total_weight += weight

            if result["success"]:
                success_votes += 1

            # Collect all matched keywords
            all_matched.update(result.get("matched_keywords", []))

    # Final decision: success if majority of methods agree OR score is high enough
    final_score = total_score / total_weight if total_weight > 0 else 0.0
    final_success = success_votes >= len(results) / 2 or final_score >= 0.6

    return {
        "success": final_success,
        "score": final_score,
        "matched_keywords": list(all_matched),
        "method": "multi_method_evaluation",
        "details": {
            "individual_results": results,
            "success_votes": success_votes,
            "total_methods": len(results),
            "combined_score": final_score,
            "methods_used": list(results.keys()),
        },
    }
