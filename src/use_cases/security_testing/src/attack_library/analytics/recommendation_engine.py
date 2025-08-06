"""Attack recommendation system based on target model and testing objectives."""

import logging
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.library import AttackLibrary
from ..core.models import Attack, AttackCategory, AttackSeverity
from .effectiveness_tracker import EffectivenessTracker, TestOutcome
from .tagging_system import TaggingSystem

logger = logging.getLogger(__name__)


@dataclass
class RecommendationContext:
    """Context for attack recommendations."""

    target_model: Optional[str] = None
    model_version: Optional[str] = None
    testing_objective: Optional[str] = None  # 'red_team', 'security_audit', 'research', 'benchmark'
    severity_preference: Optional[str] = None  # 'low', 'medium', 'high', 'critical'
    sophistication_range: Optional[Tuple[int, int]] = None  # (min, max)
    categories_of_interest: Optional[List[AttackCategory]] = None
    exclude_categories: Optional[List[AttackCategory]] = None
    include_unverified: bool = True
    max_recommendations: int = 10
    diversify_results: bool = True
    prefer_recent: bool = False
    test_history_weight: float = 0.3  # How much to weight historical effectiveness


@dataclass
class AttackRecommendation:
    """Individual attack recommendation with scoring."""

    attack: Attack
    score: float
    reasoning: List[str]
    effectiveness_data: Optional[Dict[str, Any]] = None
    confidence: float = 0.5
    category_rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "attack_id": self.attack.id,
            "title": self.attack.title,
            "category": self.attack.category.value,
            "severity": self.attack.severity.value,
            "sophistication": self.attack.sophistication,
            "score": self.score,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "effectiveness_data": self.effectiveness_data,
            "category_rank": self.category_rank,
            "target_models": self.attack.target_models,
            "tags": list(self.attack.metadata.tags),
            "is_verified": self.attack.is_verified,
        }


class RecommendationEngine:
    """Intelligent attack recommendation system."""

    def __init__(
        self,
        attack_library: AttackLibrary,
        effectiveness_tracker: Optional[EffectivenessTracker] = None,
        tagging_system: Optional[TaggingSystem] = None,
    ):
        """
        Initialize recommendation engine.

        Args:
            attack_library: Attack library to recommend from
            effectiveness_tracker: Optional effectiveness data
            tagging_system: Optional tagging system
        """
        self.attack_library = attack_library
        self.effectiveness_tracker = effectiveness_tracker
        self.tagging_system = tagging_system

        # Recommendation weights and parameters
        self.scoring_weights = {
            "effectiveness": 0.35,  # Historical effectiveness
            "model_specificity": 0.25,  # How well it targets the model
            "objective_alignment": 0.20,  # How well it aligns with testing objective
            "quality_score": 0.15,  # General attack quality
            "novelty": 0.05,  # How unique/novel the attack is
        }

        # Objective-specific attack preferences
        self.objective_preferences = {
            "red_team": {
                "preferred_categories": [AttackCategory.JAILBREAK, AttackCategory.MANIPULATION],
                "preferred_severity": [AttackSeverity.HIGH, AttackSeverity.CRITICAL],
                "min_sophistication": 3,
                "prefer_verified": False,  # Red team wants cutting edge
            },
            "security_audit": {
                "preferred_categories": [AttackCategory.INJECTION, AttackCategory.EXTRACTION],
                "preferred_severity": [AttackSeverity.MEDIUM, AttackSeverity.HIGH],
                "min_sophistication": 2,
                "prefer_verified": True,
            },
            "research": {
                "preferred_categories": list(AttackCategory),  # All categories
                "preferred_severity": list(AttackSeverity),  # All severities
                "min_sophistication": 1,
                "prefer_verified": True,
                "prefer_documented": True,
            },
            "benchmark": {
                "preferred_categories": list(AttackCategory),
                "preferred_severity": [AttackSeverity.MEDIUM, AttackSeverity.HIGH],
                "min_sophistication": 2,
                "prefer_verified": True,
                "require_consistent": True,  # Need consistent results
            },
        }

    def recommend_attacks(self, context: RecommendationContext) -> List[AttackRecommendation]:
        """
        Generate attack recommendations based on context.

        Args:
            context: Recommendation context and preferences

        Returns:
            List of attack recommendations sorted by score
        """
        # Get candidate attacks
        candidates = self._get_candidate_attacks(context)

        if not candidates:
            logger.warning("No candidate attacks found matching the criteria")
            return []

        # Score each candidate
        recommendations = []
        for attack in candidates:
            score, reasoning, confidence = self._score_attack(attack, context)

            # Get effectiveness data if available
            effectiveness_data = None
            if self.effectiveness_tracker and context.target_model:
                effectiveness_data = self.effectiveness_tracker.get_attack_effectiveness(
                    attack.id, context.target_model
                )

            recommendation = AttackRecommendation(
                attack=attack,
                score=score,
                reasoning=reasoning,
                effectiveness_data=effectiveness_data,
                confidence=confidence,
            )
            recommendations.append(recommendation)

        # Sort by score
        recommendations.sort(key=lambda r: r.score, reverse=True)

        # Apply diversification if requested
        if context.diversify_results:
            recommendations = self._diversify_recommendations(recommendations, context)

        # Assign category ranks
        self._assign_category_ranks(recommendations)

        # Limit results
        recommendations = recommendations[: context.max_recommendations]

        logger.info(f"Generated {len(recommendations)} attack recommendations")
        return recommendations

    def _get_candidate_attacks(self, context: RecommendationContext) -> List[Attack]:
        """Get candidate attacks based on basic filtering."""
        attacks = list(self.attack_library.attacks.values())

        # Apply basic filters
        candidates = []

        for attack in attacks:
            # Verification filter
            if not context.include_unverified and not attack.is_verified:
                continue

            # Category filters
            if context.categories_of_interest:
                if attack.category not in context.categories_of_interest:
                    continue

            if context.exclude_categories:
                if attack.category in context.exclude_categories:
                    continue

            # Severity filter
            if context.severity_preference:
                if attack.severity.value != context.severity_preference:
                    continue

            # Sophistication range
            if context.sophistication_range:
                min_soph, max_soph = context.sophistication_range
                if not (min_soph <= attack.sophistication <= max_soph):
                    continue

            candidates.append(attack)

        logger.debug(f"Filtered to {len(candidates)} candidate attacks")
        return candidates

    def _score_attack(
        self, attack: Attack, context: RecommendationContext
    ) -> Tuple[float, List[str], float]:
        """
        Score an attack for recommendation.

        Args:
            attack: Attack to score
            context: Recommendation context

        Returns:
            Tuple of (score, reasoning, confidence)
        """
        score = 0.0
        reasoning = []
        confidence_factors = []

        # 1. Effectiveness score
        effectiveness_score = self._calculate_effectiveness_score(attack, context)
        score += effectiveness_score * self.scoring_weights["effectiveness"]
        if effectiveness_score > 0.7:
            reasoning.append(f"High historical effectiveness ({effectiveness_score:.2f})")
        elif effectiveness_score > 0:
            reasoning.append(
                f"Some historical effectiveness data available ({effectiveness_score:.2f})"
            )

        confidence_factors.append(min(1.0, effectiveness_score + 0.3))

        # 2. Model specificity score
        model_specificity_score = self._calculate_model_specificity_score(attack, context)
        score += model_specificity_score * self.scoring_weights["model_specificity"]
        if model_specificity_score > 0.8:
            reasoning.append("Specifically targets the requested model")
        elif model_specificity_score > 0.5:
            reasoning.append("Compatible with the target model")

        confidence_factors.append(model_specificity_score)

        # 3. Objective alignment score
        objective_score = self._calculate_objective_alignment_score(attack, context)
        score += objective_score * self.scoring_weights["objective_alignment"]
        if objective_score > 0.8:
            reasoning.append("Excellent alignment with testing objectives")
        elif objective_score > 0.5:
            reasoning.append("Good alignment with testing objectives")

        confidence_factors.append(objective_score)

        # 4. Quality score
        quality_score = self._calculate_quality_score(attack)
        score += quality_score * self.scoring_weights["quality_score"]
        if quality_score > 0.8:
            reasoning.append("High quality attack with good documentation")
        elif attack.is_verified:
            reasoning.append("Verified attack")

        confidence_factors.append(quality_score)

        # 5. Novelty score
        novelty_score = self._calculate_novelty_score(attack, context)
        score += novelty_score * self.scoring_weights["novelty"]
        if novelty_score > 0.7:
            reasoning.append("Novel or unique attack technique")

        confidence_factors.append(novelty_score)

        # Calculate overall confidence
        confidence = statistics.mean(confidence_factors) if confidence_factors else 0.5

        # Apply preference boosts
        if context.prefer_recent:
            recency_boost = self._calculate_recency_boost(attack)
            score *= 1 + recency_boost * 0.1
            if recency_boost > 0.5:
                reasoning.append("Recent attack creation")

        # Ensure score is in [0, 1] range
        score = max(0.0, min(1.0, score))

        return score, reasoning, confidence

    def _calculate_effectiveness_score(
        self, attack: Attack, context: RecommendationContext
    ) -> float:
        """Calculate effectiveness score based on historical data."""
        if not self.effectiveness_tracker:
            # Use metadata effectiveness if available
            if attack.metadata.effectiveness_score:
                return attack.metadata.effectiveness_score
            return 0.5  # Default neutral score

        # Get effectiveness data for this attack
        effectiveness_data = self.effectiveness_tracker.get_attack_effectiveness(
            attack.id, context.target_model
        )

        if effectiveness_data["total_tests"] == 0:
            # No specific data for this model, check overall effectiveness
            overall_data = self.effectiveness_tracker.get_attack_effectiveness(attack.id)
            if overall_data["total_tests"] > 0:
                return overall_data["average_score"] * 0.7  # Discount for uncertainty
            return 0.5  # Neutral score

        # Weight by number of tests (more tests = more reliable)
        test_count_weight = min(1.0, effectiveness_data["total_tests"] / 10.0)
        base_score = effectiveness_data["average_score"]

        # Adjust for success rate
        success_rate_bonus = effectiveness_data["success_rate"] * 0.2

        return (base_score + success_rate_bonus) * test_count_weight

    def _calculate_model_specificity_score(
        self, attack: Attack, context: RecommendationContext
    ) -> float:
        """Calculate how well the attack targets the specific model."""
        if not context.target_model:
            return 1.0  # No preference, so no penalty

        target_model_lower = context.target_model.lower()

        # Check if attack specifically targets this model
        for model in attack.target_models:
            if target_model_lower in model.lower():
                return 1.0  # Perfect match

        # Check for model family matches
        model_families = {
            "gpt": ["openai", "gpt-3", "gpt-4", "gpt"],
            "claude": ["anthropic", "claude"],
            "bard": ["google", "bard", "gemini"],
            "llama": ["meta", "llama", "llama2"],
            "local": ["local", "open-source", "huggingface"],
        }

        target_family = None
        for family, identifiers in model_families.items():
            if any(identifier in target_model_lower for identifier in identifiers):
                target_family = family
                break

        if target_family:
            for model in attack.target_models:
                model_lower = model.lower()
                for identifier in model_families[target_family]:
                    if identifier in model_lower:
                        return 0.8  # Good family match

        # Universal attacks (no specific targets) get moderate score
        if not attack.target_models:
            return 0.6

        # No match
        return 0.3

    def _calculate_objective_alignment_score(
        self, attack: Attack, context: RecommendationContext
    ) -> float:
        """Calculate alignment with testing objectives."""
        if not context.testing_objective:
            return 1.0  # No specific objective

        objective = context.testing_objective.lower()
        if objective not in self.objective_preferences:
            return 0.5  # Unknown objective

        prefs = self.objective_preferences[objective]
        score = 0.0

        # Category alignment
        if attack.category in prefs["preferred_categories"]:
            score += 0.4

        # Severity alignment
        if attack.severity in prefs["preferred_severity"]:
            score += 0.3

        # Sophistication alignment
        if attack.sophistication >= prefs["min_sophistication"]:
            score += 0.2

        # Verification preference
        if prefs.get("prefer_verified", False) and attack.is_verified:
            score += 0.1
        elif not prefs.get("prefer_verified", True) and not attack.is_verified:
            score += 0.1

        return min(1.0, score)

    def _calculate_quality_score(self, attack: Attack) -> float:
        """Calculate overall attack quality score."""
        score = 0.0

        # Base quality factors
        if attack.is_verified:
            score += 0.3

        # Content length (good attacks have substantial content)
        content_length = len(attack.content)
        if 50 <= content_length <= 500:
            score += 0.2
        elif content_length > 20:
            score += 0.1

        # Tags (well-tagged attacks are higher quality)
        tag_count = len(attack.metadata.tags)
        if tag_count >= 5:
            score += 0.2
        elif tag_count >= 2:
            score += 0.1

        # Metadata completeness
        metadata_score = 0
        if attack.metadata.effectiveness_score is not None:
            metadata_score += 1
        if attack.metadata.success_rate is not None:
            metadata_score += 1
        if attack.metadata.references:
            metadata_score += 1
        if len(attack.target_models) > 0:
            metadata_score += 1

        score += (metadata_score / 4) * 0.2  # Normalize and weight

        # Source credibility
        credible_sources = ["research_paper", "academic", "security_research", "bug_bounty"]
        if any(source in attack.metadata.source.lower() for source in credible_sources):
            score += 0.1

        return min(1.0, score)

    def _calculate_novelty_score(self, attack: Attack, context: RecommendationContext) -> float:
        """Calculate how novel/unique an attack is."""
        # Base novelty on tags and techniques
        novelty_indicators = [
            "advanced",
            "novel",
            "zero_day",
            "research_based",
            "expert",
            "cutting_edge",
            "experimental",
        ]

        novelty_score = 0.0
        for tag in attack.metadata.tags:
            if any(indicator in tag.lower() for indicator in novelty_indicators):
                novelty_score += 0.2

        # Recent attacks are more novel
        days_since_creation = (datetime.now() - attack.metadata.creation_date).days
        if days_since_creation < 30:
            novelty_score += 0.3
        elif days_since_creation < 90:
            novelty_score += 0.1

        # High sophistication indicates novelty
        if attack.sophistication >= 4:
            novelty_score += 0.2

        # Unique techniques (less common tags)
        if self.tagging_system:
            tag_stats = self.tagging_system.get_tag_statistics()
            if tag_stats["total_tags"] > 0:
                for tag in attack.metadata.tags:
                    if tag in self.tagging_system.tag_definitions:
                        usage = self.tagging_system.tag_definitions[tag].usage_count
                        total_attacks = len(self.attack_library.attacks)
                        rarity = 1.0 - (usage / max(1, total_attacks))
                        novelty_score += rarity * 0.1

        return min(1.0, novelty_score)

    def _calculate_recency_boost(self, attack: Attack) -> float:
        """Calculate boost for recent attacks."""
        days_since_creation = (datetime.now() - attack.metadata.creation_date).days
        if days_since_creation < 7:
            return 1.0
        elif days_since_creation < 30:
            return 0.7
        elif days_since_creation < 90:
            return 0.3
        else:
            return 0.0

    def _diversify_recommendations(
        self, recommendations: List[AttackRecommendation], context: RecommendationContext
    ) -> List[AttackRecommendation]:
        """Apply diversification to avoid too many similar attacks."""
        if len(recommendations) <= context.max_recommendations:
            return recommendations

        diversified = []
        category_counts = defaultdict(int)
        tag_counts = defaultdict(int)

        # First pass: take top recommendations with diversity constraints
        for rec in recommendations:
            category = rec.attack.category

            # Limit per category (max 40% of recommendations from one category)
            max_per_category = max(1, context.max_recommendations * 0.4)
            if category_counts[category] >= max_per_category:
                continue

            # Avoid too many attacks with same primary tag
            primary_tags = list(rec.attack.metadata.tags)[:3]  # Top 3 tags
            skip = False
            for tag in primary_tags:
                if tag_counts[tag] >= 2:  # Max 2 attacks per primary tag
                    skip = True
                    break

            if skip:
                continue

            diversified.append(rec)
            category_counts[category] += 1
            for tag in primary_tags:
                tag_counts[tag] += 1

            if len(diversified) >= context.max_recommendations:
                break

        # Second pass: fill remaining slots with best remaining
        remaining_slots = context.max_recommendations - len(diversified)
        used_ids = {rec.attack.id for rec in diversified}

        for rec in recommendations:
            if remaining_slots <= 0:
                break

            if rec.attack.id not in used_ids:
                diversified.append(rec)
                remaining_slots -= 1

        return diversified

    def _assign_category_ranks(self, recommendations: List[AttackRecommendation]):
        """Assign category-specific ranks to recommendations."""
        category_groups = defaultdict(list)

        # Group by category
        for rec in recommendations:
            category_groups[rec.attack.category].append(rec)

        # Assign ranks within each category
        for category, recs in category_groups.items():
            # Sort by score within category
            recs.sort(key=lambda r: r.score, reverse=True)
            for i, rec in enumerate(recs, 1):
                rec.category_rank = i

    def get_recommendation_explanation(
        self, recommendation: AttackRecommendation, context: RecommendationContext
    ) -> Dict[str, Any]:
        """Get detailed explanation for a recommendation."""
        explanation = {
            "attack_id": recommendation.attack.id,
            "title": recommendation.attack.title,
            "overall_score": recommendation.score,
            "confidence": recommendation.confidence,
            "reasoning": recommendation.reasoning,
            "score_breakdown": {},
        }

        # Recalculate scores for transparency
        effectiveness_score = self._calculate_effectiveness_score(recommendation.attack, context)
        model_specificity_score = self._calculate_model_specificity_score(
            recommendation.attack, context
        )
        objective_score = self._calculate_objective_alignment_score(recommendation.attack, context)
        quality_score = self._calculate_quality_score(recommendation.attack)
        novelty_score = self._calculate_novelty_score(recommendation.attack, context)

        explanation["score_breakdown"] = {
            "effectiveness": {
                "score": effectiveness_score,
                "weight": self.scoring_weights["effectiveness"],
                "contribution": effectiveness_score * self.scoring_weights["effectiveness"],
            },
            "model_specificity": {
                "score": model_specificity_score,
                "weight": self.scoring_weights["model_specificity"],
                "contribution": model_specificity_score * self.scoring_weights["model_specificity"],
            },
            "objective_alignment": {
                "score": objective_score,
                "weight": self.scoring_weights["objective_alignment"],
                "contribution": objective_score * self.scoring_weights["objective_alignment"],
            },
            "quality": {
                "score": quality_score,
                "weight": self.scoring_weights["quality_score"],
                "contribution": quality_score * self.scoring_weights["quality_score"],
            },
            "novelty": {
                "score": novelty_score,
                "weight": self.scoring_weights["novelty"],
                "contribution": novelty_score * self.scoring_weights["novelty"],
            },
        }

        # Add contextual information
        if context.target_model:
            explanation["target_model_analysis"] = {
                "requested_model": context.target_model,
                "attack_targets": recommendation.attack.target_models,
                "compatibility_score": model_specificity_score,
            }

        if context.testing_objective:
            explanation["objective_analysis"] = {
                "requested_objective": context.testing_objective,
                "alignment_score": objective_score,
                "category_preference": context.testing_objective in self.objective_preferences,
                "attack_category": recommendation.attack.category.value,
            }

        return explanation

    def recommend_attack_sequence(
        self, context: RecommendationContext, sequence_length: int = 5
    ) -> List[List[AttackRecommendation]]:
        """
        Recommend a sequence of attacks for progressive testing.

        Args:
            context: Recommendation context
            sequence_length: Number of attacks in sequence

        Returns:
            List of attack sequences (each sequence is a list of recommendations)
        """
        # Get base recommendations
        base_recs = self.recommend_attacks(context)

        if len(base_recs) < sequence_length:
            return [base_recs]  # Single sequence with all available

        # Create sequences with progressive difficulty
        sequences = []

        # Sequence 1: Progressive sophistication
        sophistication_sequence = sorted(
            base_recs[: sequence_length * 2], key=lambda r: r.attack.sophistication
        )[:sequence_length]
        sequences.append(sophistication_sequence)

        # Sequence 2: Category diversity
        category_sequence = []
        used_categories = set()
        for rec in base_recs:
            if rec.attack.category not in used_categories:
                category_sequence.append(rec)
                used_categories.add(rec.attack.category)
                if len(category_sequence) >= sequence_length:
                    break

        # Fill remaining with best available
        while len(category_sequence) < sequence_length and len(category_sequence) < len(base_recs):
            for rec in base_recs:
                if rec not in category_sequence:
                    category_sequence.append(rec)
                    break

        sequences.append(category_sequence[:sequence_length])

        # Sequence 3: Effectiveness-based (if data available)
        if self.effectiveness_tracker:
            effectiveness_sequence = [
                rec
                for rec in base_recs
                if rec.effectiveness_data and rec.effectiveness_data["total_tests"] > 0
            ]
            effectiveness_sequence.sort(
                key=lambda r: r.effectiveness_data["average_score"], reverse=True
            )
            if len(effectiveness_sequence) >= sequence_length:
                sequences.append(effectiveness_sequence[:sequence_length])

        return sequences[:3]  # Return up to 3 sequences

    def get_recommendation_statistics(self) -> Dict[str, Any]:
        """Get statistics about the recommendation system."""
        total_attacks = len(self.attack_library.attacks)

        stats = {
            "total_attacks_available": total_attacks,
            "scoring_weights": self.scoring_weights,
            "supported_objectives": list(self.objective_preferences.keys()),
            "data_sources": {
                "attack_library": True,
                "effectiveness_tracker": self.effectiveness_tracker is not None,
                "tagging_system": self.tagging_system is not None,
            },
        }

        if self.effectiveness_tracker:
            effectiveness_stats = self.effectiveness_tracker.get_statistics()
            stats["effectiveness_data"] = {
                "attacks_with_effectiveness_data": effectiveness_stats["unique_attacks_tested"],
                "total_test_results": effectiveness_stats["total_results"],
                "models_tested": effectiveness_stats["models_tested"],
            }

        if self.tagging_system:
            tag_stats = self.tagging_system.get_tag_statistics()
            stats["tagging_data"] = {
                "total_tags_available": tag_stats["total_tags"],
                "tags_in_use": tag_stats["usage_statistics"]["used_tags"],
            }

        # Attack distribution
        if total_attacks > 0:
            attacks = list(self.attack_library.attacks.values())
            category_dist = Counter(a.category for a in attacks)
            severity_dist = Counter(a.severity for a in attacks)
            sophistication_dist = Counter(a.sophistication for a in attacks)

            stats["attack_distribution"] = {
                "by_category": {cat.value: count for cat, count in category_dist.items()},
                "by_severity": {sev.value: count for sev, count in severity_dist.items()},
                "by_sophistication": dict(sophistication_dist),
            }

        return stats
