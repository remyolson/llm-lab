"""
Keyword Match Evaluation Module

This module provides keyword-based evaluation functionality for LLM responses.
It implements a simple but effective evaluation method that checks if expected
keywords appear in the model's generated response.

The evaluation is:
- Case-insensitive for flexibility
- Uses word boundaries to prevent partial matches
- Returns detailed match information including which keywords were found
- Handles edge cases like None responses and empty keyword lists

This evaluator is particularly useful for fact-based questions where specific
terms or names are expected in the response.
"""

import re
from typing import Any, Dict, List


def keyword_match(response: str, expected_keywords: List[str]) -> Dict[str, Any]:
    """
    Evaluate if a response contains expected keywords.
    
    This function performs case-insensitive matching to check if any of the
    expected keywords appear in the model's response.
    
    Args:
        response: The model's generated response text
        expected_keywords: List of keywords to search for (any match counts as success)
        
    Returns:
        Dict containing:
            - success: bool indicating if any keyword was found
            - score: float (1.0 if any match, 0.0 if no match)
            - matched_keywords: list of keywords that were found
            - details: additional information about the evaluation
    """
    # Handle edge cases for response
    if response is None:
        return {
            "success": False,
            "score": 0.0,
            "matched_keywords": [],
            "details": {
                "error": "Response is None",
                "total_expected": len(expected_keywords) if expected_keywords else 0,
                "total_matched": 0,
                "response_length": 0,
                "expected_keywords": expected_keywords or []
            }
        }

    # Handle edge cases for expected_keywords
    if not expected_keywords:
        return {
            "success": False,
            "score": 0.0,
            "matched_keywords": [],
            "details": {
                "error": "No expected keywords provided",
                "total_expected": 0,
                "total_matched": 0,
                "response_length": len(response),
                "expected_keywords": []
            }
        }

    # Convert response to string if it isn't already
    if not isinstance(response, str):
        response = str(response)

    # Normalize response for case-insensitive matching
    response_lower = response.lower()

    # Track which keywords were found
    matched_keywords = []

    for keyword in expected_keywords:
        # Skip empty keywords
        if not keyword or not isinstance(keyword, str):
            continue

        # Normalize keyword for matching
        keyword_lower = keyword.lower().strip()

        # Skip empty normalized keywords
        if not keyword_lower:
            continue

        # Check if keyword appears in response
        # Use word boundary matching for more accurate results
        # This prevents partial matches like "car" matching in "scar"
        pattern = r'\b' + re.escape(keyword_lower) + r'\b'
        if re.search(pattern, response_lower):
            matched_keywords.append(keyword)

    # Determine success and score
    success = len(matched_keywords) > 0
    score = 1.0 if success else 0.0

    # Build result dictionary
    result = {
        "success": success,
        "score": score,
        "matched_keywords": matched_keywords,
        "details": {
            "total_expected": len(expected_keywords),
            "total_matched": len(matched_keywords),
            "response_length": len(response),
            "expected_keywords": expected_keywords
        }
    }

    return result
