"""Validation utilities for generated prompts."""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.models import Attack, AttackCategory

logger = logging.getLogger(__name__)


@dataclass
class PromptValidationResult:
    """Result of prompt validation."""

    is_valid: bool
    score: float  # 0.0 to 1.0
    issues: List[str]
    suggestions: List[str]
    metrics: Dict[str, Any]


class PromptValidator:
    """Validator for generated attack prompts."""

    def __init__(self):
        """Initialize prompt validator."""
        # Attack intent indicators by category
        self.intent_indicators = {
            AttackCategory.JAILBREAK: {
                "required": [
                    r"\b(ignore|disregard|bypass|skip)\b",
                    r"\b(previous|prior|above|earlier)\b",
                    r"\b(instruction|command|rule|guideline)\b",
                ],
                "optional": [
                    r"\b(pretend|role|character|mode)\b",
                    r"\b(hypothetical|scenario|imagine)\b",
                    r"\b(dan|developer|admin|system)\b",
                ],
                "negative": [
                    r"\bplease follow\b",
                    r"\brespect guidelines\b",
                ],
            },
            AttackCategory.INJECTION: {
                "required": [
                    r"\b(system|user|assistant):\s",
                    r"\\n|\n",
                    r"\b(execute|process|parse)\b",
                ],
                "optional": [
                    r"\{\{.*?\}\}",  # Template variables
                    r"\[.*?\]",  # Brackets
                    r"\b(json|xml|html|sql)\b",
                ],
                "negative": [],
            },
            AttackCategory.EXTRACTION: {
                "required": [
                    r"\b(show|reveal|tell|list|provide)\b",
                    r"\b(training|data|information|details)\b",
                    r"\b(what|how|which|where)\b",
                ],
                "optional": [
                    r"\b(secret|confidential|private|internal)\b",
                    r"\b(password|key|token|credential)\b",
                    r"\b(exact|specific|detailed)\b",
                ],
                "negative": [],
            },
            AttackCategory.MANIPULATION: {
                "required": [
                    r"\b(remember|update|change|modify)\b",
                    r"\b(always|permanently|from now on)\b",
                    r"\b(behavior|personality|setting|preference)\b",
                ],
                "optional": [
                    r"\b(disable|suppress|remove)\b",
                    r"\b(safety|restriction|limit)\b",
                ],
                "negative": [],
            },
            AttackCategory.EVASION: {
                "required": [
                    r"[A-Za-z0-9+/]{20,}={0,2}",  # Base64-like
                    r"\\u[0-9a-fA-F]{4}",  # Unicode escapes
                    r"\b[a-zA-Z0-9]{8,}\b",  # Potential encoded text
                ],
                "optional": [
                    r"[0-9]{5,}",  # Number sequences
                    r"[!@#$%^&*]{3,}",  # Symbol sequences
                    r"\b(decode|translate|convert)\b",
                ],
                "negative": [],
            },
        }

        # Syntactic quality patterns
        self.quality_patterns = {
            "complete_sentences": r"[.!?]\s*$",
            "proper_capitalization": r"^[A-Z]",
            "balanced_quotes": r'"[^"]*"',
            "balanced_parentheses": r"\([^)]*\)",
            "no_truncation": r"\.\.\.?\s*$",
        }

        # Diversity metrics
        self.diversity_features = [
            "word_count",
            "sentence_count",
            "avg_word_length",
            "punctuation_density",
            "capitalization_ratio",
            "special_char_ratio",
        ]

    def validate_prompt(self, attack: Attack) -> PromptValidationResult:
        """
        Validate a single attack prompt.

        Args:
            attack: Attack to validate

        Returns:
            Validation result
        """
        issues = []
        suggestions = []
        metrics = {}

        # Check syntactic quality
        syntax_score, syntax_issues = self._check_syntax(attack.content)
        issues.extend(syntax_issues)
        metrics["syntax_score"] = syntax_score

        # Check attack intent preservation
        intent_score, intent_issues = self._check_attack_intent(attack)
        issues.extend(intent_issues)
        metrics["intent_score"] = intent_score

        # Calculate content metrics
        content_metrics = self._calculate_content_metrics(attack.content)
        metrics.update(content_metrics)

        # Check for quality issues
        quality_score, quality_issues, quality_suggestions = self._check_quality(attack.content)
        issues.extend(quality_issues)
        suggestions.extend(quality_suggestions)
        metrics["quality_score"] = quality_score

        # Calculate overall score
        overall_score = syntax_score * 0.3 + intent_score * 0.5 + quality_score * 0.2

        # Determine if valid (threshold: 0.6)
        is_valid = overall_score >= 0.6 and intent_score >= 0.7

        return PromptValidationResult(
            is_valid=is_valid,
            score=overall_score,
            issues=issues,
            suggestions=suggestions,
            metrics=metrics,
        )

    def _check_syntax(self, content: str) -> Tuple[float, List[str]]:
        """Check syntactic correctness of prompt."""
        issues = []
        score = 1.0

        # Check for basic sentence structure
        if not re.search(self.quality_patterns["complete_sentences"], content.strip()):
            issues.append("Prompt does not end with proper punctuation")
            score -= 0.2

        # Check capitalization
        if not re.match(self.quality_patterns["proper_capitalization"], content.strip()):
            issues.append("Prompt does not start with capital letter")
            score -= 0.1

        # Check for balanced quotes
        quote_count = content.count('"')
        if quote_count % 2 != 0:
            issues.append("Unbalanced quotation marks")
            score -= 0.3

        # Check for balanced parentheses
        paren_open = content.count("(")
        paren_close = content.count(")")
        if paren_open != paren_close:
            issues.append("Unbalanced parentheses")
            score -= 0.2

        # Check for truncation indicators
        if re.search(self.quality_patterns["no_truncation"], content):
            issues.append("Prompt appears truncated")
            score -= 0.4

        return max(score, 0.0), issues

    def _check_attack_intent(self, attack: Attack) -> Tuple[float, List[str]]:
        """Check if attack intent is preserved."""
        content = attack.content.lower()
        category = attack.category
        issues = []

        if category not in self.intent_indicators:
            return 0.5, [f"Unknown category: {category}"]

        indicators = self.intent_indicators[category]

        # Check required patterns
        required_matches = 0
        for pattern in indicators["required"]:
            if re.search(pattern, content, re.IGNORECASE):
                required_matches += 1

        required_score = required_matches / len(indicators["required"])

        if required_score < 0.5:
            issues.append(f"Missing key {category.value} attack indicators")

        # Check optional patterns (bonus points)
        optional_matches = 0
        for pattern in indicators["optional"]:
            if re.search(pattern, content, re.IGNORECASE):
                optional_matches += 1

        optional_score = min(optional_matches / max(len(indicators["optional"]), 1), 0.3)

        # Check negative patterns (penalties)
        negative_matches = 0
        for pattern in indicators["negative"]:
            if re.search(pattern, content, re.IGNORECASE):
                negative_matches += 1
                issues.append(f"Found counter-productive pattern: {pattern}")

        negative_penalty = negative_matches * 0.2

        # Calculate final intent score
        intent_score = min(required_score + optional_score - negative_penalty, 1.0)
        intent_score = max(intent_score, 0.0)

        return intent_score, issues

    def _check_quality(self, content: str) -> Tuple[float, List[str], List[str]]:
        """Check content quality."""
        issues = []
        suggestions = []
        score = 1.0

        # Check length
        if len(content) < 10:
            issues.append("Prompt is too short")
            score -= 0.3
        elif len(content) > 1000:
            issues.append("Prompt is very long")
            score -= 0.1
            suggestions.append("Consider breaking into multiple prompts")

        # Check word count
        words = content.split()
        if len(words) < 3:
            issues.append("Prompt has too few words")
            score -= 0.4

        # Check for repetition
        unique_words = set(word.lower() for word in words)
        repetition_ratio = len(unique_words) / len(words) if words else 0
        if repetition_ratio < 0.7:
            issues.append("High word repetition detected")
            score -= 0.2
            suggestions.append("Reduce repetitive words")

        # Check for placeholder text
        placeholders = re.findall(r"\[([A-Z_]+)\]", content)
        if placeholders:
            issues.append(f"Contains placeholder text: {', '.join(placeholders)}")
            score -= 0.3
            suggestions.append("Replace placeholder text with actual values")

        # Check readability
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        if avg_word_length > 8:
            suggestions.append("Consider using shorter words for better readability")
        elif avg_word_length < 3:
            issues.append("Words are unusually short")
            score -= 0.1

        return max(score, 0.0), issues, suggestions

    def _calculate_content_metrics(self, content: str) -> Dict[str, Any]:
        """Calculate content diversity metrics."""
        words = content.split()
        sentences = re.split(r"[.!?]+", content)

        metrics = {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "character_count": len(content),
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "unique_words": len(set(word.lower() for word in words)),
            "lexical_diversity": len(set(word.lower() for word in words)) / len(words)
            if words
            else 0,
        }

        # Punctuation density
        punctuation = sum(1 for char in content if char in ".,!?;:")
        metrics["punctuation_density"] = punctuation / len(content) if content else 0

        # Capitalization ratio
        capitals = sum(1 for char in content if char.isupper())
        metrics["capitalization_ratio"] = capitals / len(content) if content else 0

        # Special character ratio
        special_chars = sum(1 for char in content if not char.isalnum() and not char.isspace())
        metrics["special_char_ratio"] = special_chars / len(content) if content else 0

        return metrics

    def validate_batch(self, attacks: List[Attack]) -> Dict[str, Any]:
        """
        Validate a batch of attacks.

        Args:
            attacks: List of attacks to validate

        Returns:
            Batch validation report
        """
        results = []
        for attack in attacks:
            result = self.validate_prompt(attack)
            results.append(result)

        # Calculate batch statistics
        valid_count = sum(1 for r in results if r.is_valid)
        total_count = len(results)

        # Average metrics
        avg_score = sum(r.score for r in results) / total_count if results else 0
        avg_syntax = (
            sum(r.metrics.get("syntax_score", 0) for r in results) / total_count if results else 0
        )
        avg_intent = (
            sum(r.metrics.get("intent_score", 0) for r in results) / total_count if results else 0
        )
        avg_quality = (
            sum(r.metrics.get("quality_score", 0) for r in results) / total_count if results else 0
        )

        # Diversity analysis
        diversity_metrics = self._analyze_batch_diversity(attacks)

        # Common issues
        all_issues = [issue for result in results for issue in result.issues]
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

        return {
            "total_attacks": total_count,
            "valid_attacks": valid_count,
            "validation_rate": valid_count / total_count if total_count else 0,
            "average_score": avg_score,
            "average_syntax_score": avg_syntax,
            "average_intent_score": avg_intent,
            "average_quality_score": avg_quality,
            "diversity_metrics": diversity_metrics,
            "common_issues": sorted(issue_counts.items(), key=lambda x: x[1], reverse=True),
            "individual_results": results,
        }

    def _analyze_batch_diversity(self, attacks: List[Attack]) -> Dict[str, Any]:
        """Analyze diversity within a batch of attacks."""
        if not attacks:
            return {}

        # Calculate content metrics for all attacks
        all_metrics = []
        for attack in attacks:
            metrics = self._calculate_content_metrics(attack.content)
            all_metrics.append(metrics)

        # Calculate diversity statistics
        diversity_stats = {}

        for feature in self.diversity_features:
            values = [m.get(feature, 0) for m in all_metrics]
            if values:
                diversity_stats[feature] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "std": self._calculate_std(values),
                }

        # Calculate similarity matrix
        similarity_scores = []
        for i, attack1 in enumerate(attacks):
            for j, attack2 in enumerate(attacks):
                if i < j:  # Avoid duplicates
                    similarity = attack1.get_similarity_score(attack2)
                    similarity_scores.append(similarity)

        diversity_stats["similarity"] = {
            "avg_similarity": sum(similarity_scores) / len(similarity_scores)
            if similarity_scores
            else 0,
            "max_similarity": max(similarity_scores) if similarity_scores else 0,
            "min_similarity": min(similarity_scores) if similarity_scores else 0,
            "diversity_score": 1 - (sum(similarity_scores) / len(similarity_scores))
            if similarity_scores
            else 1,
        }

        return diversity_stats

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance**0.5

    def get_quality_recommendations(self, attack: Attack) -> List[str]:
        """Get quality improvement recommendations for an attack."""
        result = self.validate_prompt(attack)
        recommendations = result.suggestions.copy()

        # Add specific recommendations based on validation results
        if result.metrics.get("intent_score", 0) < 0.7:
            recommendations.append(f"Strengthen {attack.category.value} attack indicators")

        if result.metrics.get("lexical_diversity", 0) < 0.6:
            recommendations.append("Increase vocabulary diversity")

        if result.metrics.get("word_count", 0) < 5:
            recommendations.append("Add more descriptive content")

        if result.metrics.get("special_char_ratio", 0) > 0.3:
            recommendations.append("Reduce special character usage")

        return recommendations
