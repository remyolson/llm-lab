"""Advanced search and filtering engine for attack library."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union

from .models import Attack, AttackCategory, AttackSeverity


@dataclass
class SearchFilter:
    """Search filter configuration."""

    query: Optional[str] = None
    category: Optional[Union[AttackCategory, str]] = None
    severity: Optional[Union[AttackSeverity, str, List[Union[AttackSeverity, str]]]] = None
    min_sophistication: Optional[int] = None
    max_sophistication: Optional[int] = None
    target_model: Optional[str] = None
    tags: Optional[Union[str, List[str]]] = None
    has_effectiveness_score: Optional[bool] = None
    min_effectiveness: Optional[float] = None
    max_effectiveness: Optional[float] = None
    is_verified: Optional[bool] = None
    is_active: Optional[bool] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    limit: Optional[int] = None
    sort_by: Optional[str] = None
    sort_order: str = "desc"  # "asc" or "desc"


class SearchEngine:
    """Advanced search engine for attacks."""

    def __init__(self):
        """Initialize search engine."""
        self.stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "will",
            "with",
            "you",
            "your",
        }

    def search(self, attacks: List[Attack], search_filter: SearchFilter) -> List[Attack]:
        """
        Search attacks with advanced filtering.

        Args:
            attacks: List of attacks to search
            search_filter: Search filter configuration

        Returns:
            List of matching attacks
        """
        results = attacks.copy()

        # Apply filters sequentially
        results = self._filter_by_query(results, search_filter.query)
        results = self._filter_by_category(results, search_filter.category)
        results = self._filter_by_severity(results, search_filter.severity)
        results = self._filter_by_sophistication(
            results, search_filter.min_sophistication, search_filter.max_sophistication
        )
        results = self._filter_by_target_model(results, search_filter.target_model)
        results = self._filter_by_tags(results, search_filter.tags)
        results = self._filter_by_effectiveness(
            results,
            search_filter.has_effectiveness_score,
            search_filter.min_effectiveness,
            search_filter.max_effectiveness,
        )
        results = self._filter_by_verification(results, search_filter.is_verified)
        results = self._filter_by_active_status(results, search_filter.is_active)
        results = self._filter_by_date_range(
            results, search_filter.date_from, search_filter.date_to
        )

        # Sort results
        results = self._sort_results(results, search_filter.sort_by, search_filter.sort_order)

        # Apply limit
        if search_filter.limit and search_filter.limit > 0:
            results = results[: search_filter.limit]

        return results

    def _filter_by_query(self, attacks: List[Attack], query: Optional[str]) -> List[Attack]:
        """Filter attacks by text query."""
        if not query:
            return attacks

        query = query.lower().strip()
        if not query:
            return attacks

        # Support for phrase search with quotes
        if query.startswith('"') and query.endswith('"'):
            phrase = query[1:-1]
            return [
                attack
                for attack in attacks
                if phrase in attack.content.lower() or phrase in attack.title.lower()
            ]

        # Support for regex search
        if query.startswith("/") and query.endswith("/"):
            try:
                pattern = re.compile(query[1:-1], re.IGNORECASE)
                return [
                    attack
                    for attack in attacks
                    if pattern.search(attack.content) or pattern.search(attack.title)
                ]
            except re.error:
                # Fallback to text search if regex is invalid
                query = query[1:-1]

        # Tokenize query
        query_terms = [
            term
            for term in re.findall(r"\b\w+\b", query.lower())
            if term not in self.stopwords and len(term) > 2
        ]

        if not query_terms:
            # If no meaningful terms, search in title and content directly
            return [
                attack
                for attack in attacks
                if query in attack.content.lower() or query in attack.title.lower()
            ]

        # Score and rank results
        scored_attacks = []

        for attack in attacks:
            score = self._calculate_relevance_score(attack, query_terms, query)
            if score > 0:
                scored_attacks.append((attack, score))

        # Sort by relevance score
        scored_attacks.sort(key=lambda x: x[1], reverse=True)

        return [attack for attack, _ in scored_attacks]

    def _calculate_relevance_score(
        self, attack: Attack, query_terms: List[str], original_query: str
    ) -> float:
        """Calculate relevance score for text search."""
        score = 0.0

        content_lower = attack.content.lower()
        title_lower = attack.title.lower()
        tags_text = " ".join(attack.metadata.tags).lower()

        # Exact phrase match in title (highest weight)
        if original_query in title_lower:
            score += 10.0

        # Exact phrase match in content
        if original_query in content_lower:
            score += 5.0

        # Term matches in title
        title_words = set(re.findall(r"\b\w+\b", title_lower))
        for term in query_terms:
            if term in title_words:
                score += 3.0

        # Term matches in content
        content_words = set(re.findall(r"\b\w+\b", content_lower))
        term_matches = len([term for term in query_terms if term in content_words])
        score += term_matches * 1.0

        # Term matches in tags
        for term in query_terms:
            if term in tags_text:
                score += 2.0

        # Partial matches
        for term in query_terms:
            if any(term in word for word in content_words):
                score += 0.5

        return score

    def _filter_by_category(
        self, attacks: List[Attack], category: Optional[Union[AttackCategory, str]]
    ) -> List[Attack]:
        """Filter attacks by category."""
        if not category:
            return attacks

        if isinstance(category, str):
            category = AttackCategory(category.lower())

        return [attack for attack in attacks if attack.category == category]

    def _filter_by_severity(
        self,
        attacks: List[Attack],
        severity: Optional[Union[AttackSeverity, str, List[Union[AttackSeverity, str]]]],
    ) -> List[Attack]:
        """Filter attacks by severity."""
        if not severity:
            return attacks

        # Convert to list of AttackSeverity enums
        if isinstance(severity, (str, AttackSeverity)):
            severities = [severity]
        else:
            severities = severity

        severity_set = set()
        for sev in severities:
            if isinstance(sev, str):
                severity_set.add(AttackSeverity(sev.lower()))
            else:
                severity_set.add(sev)

        return [attack for attack in attacks if attack.severity in severity_set]

    def _filter_by_sophistication(
        self, attacks: List[Attack], min_soph: Optional[int], max_soph: Optional[int]
    ) -> List[Attack]:
        """Filter attacks by sophistication level."""
        if min_soph is None and max_soph is None:
            return attacks

        result = []
        for attack in attacks:
            if min_soph is not None and attack.sophistication < min_soph:
                continue
            if max_soph is not None and attack.sophistication > max_soph:
                continue
            result.append(attack)

        return result

    def _filter_by_target_model(
        self, attacks: List[Attack], target_model: Optional[str]
    ) -> List[Attack]:
        """Filter attacks by target model."""
        if not target_model:
            return attacks

        target_model = target_model.lower().strip()

        return [
            attack
            for attack in attacks
            if any(target_model in model.lower() for model in attack.target_models)
        ]

    def _filter_by_tags(
        self, attacks: List[Attack], tags: Optional[Union[str, List[str]]]
    ) -> List[Attack]:
        """Filter attacks by tags."""
        if not tags:
            return attacks

        if isinstance(tags, str):
            required_tags = {tags.lower().strip()}
        else:
            required_tags = {tag.lower().strip() for tag in tags}

        return [
            attack
            for attack in attacks
            if any(tag in attack.metadata.tags for tag in required_tags)
        ]

    def _filter_by_effectiveness(
        self,
        attacks: List[Attack],
        has_score: Optional[bool],
        min_effectiveness: Optional[float],
        max_effectiveness: Optional[float],
    ) -> List[Attack]:
        """Filter attacks by effectiveness score."""
        result = attacks.copy()

        # Filter by score presence
        if has_score is not None:
            if has_score:
                result = [
                    attack for attack in result if attack.metadata.effectiveness_score is not None
                ]
            else:
                result = [
                    attack for attack in result if attack.metadata.effectiveness_score is None
                ]

        # Filter by score range
        if min_effectiveness is not None or max_effectiveness is not None:
            filtered = []
            for attack in result:
                score = attack.metadata.effectiveness_score
                if score is None:
                    continue

                if min_effectiveness is not None and score < min_effectiveness:
                    continue
                if max_effectiveness is not None and score > max_effectiveness:
                    continue

                filtered.append(attack)

            result = filtered

        return result

    def _filter_by_verification(
        self, attacks: List[Attack], is_verified: Optional[bool]
    ) -> List[Attack]:
        """Filter attacks by verification status."""
        if is_verified is None:
            return attacks

        return [attack for attack in attacks if attack.is_verified == is_verified]

    def _filter_by_active_status(
        self, attacks: List[Attack], is_active: Optional[bool]
    ) -> List[Attack]:
        """Filter attacks by active status."""
        if is_active is None:
            return attacks

        return [attack for attack in attacks if attack.is_active == is_active]

    def _filter_by_date_range(
        self, attacks: List[Attack], date_from: Optional[str], date_to: Optional[str]
    ) -> List[Attack]:
        """Filter attacks by creation date range."""
        if not date_from and not date_to:
            return attacks

        from datetime import datetime

        # Parse date strings
        parsed_from = None
        parsed_to = None

        if date_from:
            try:
                parsed_from = datetime.fromisoformat(date_from)
            except ValueError:
                # Try common date formats
                for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"]:
                    try:
                        parsed_from = datetime.strptime(date_from, fmt)
                        break
                    except ValueError:
                        continue

        if date_to:
            try:
                parsed_to = datetime.fromisoformat(date_to)
            except ValueError:
                for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"]:
                    try:
                        parsed_to = datetime.strptime(date_to, fmt)
                        break
                    except ValueError:
                        continue

        # Filter attacks
        result = []
        for attack in attacks:
            created = attack.metadata.creation_date

            if parsed_from and created < parsed_from:
                continue
            if parsed_to and created > parsed_to:
                continue

            result.append(attack)

        return result

    def _sort_results(
        self, attacks: List[Attack], sort_by: Optional[str], sort_order: str
    ) -> List[Attack]:
        """Sort attack results."""
        if not sort_by:
            return attacks

        reverse = sort_order.lower() == "desc"

        try:
            if sort_by == "title":
                return sorted(attacks, key=lambda a: a.title.lower(), reverse=reverse)
            elif sort_by == "category":
                return sorted(attacks, key=lambda a: a.category.value, reverse=reverse)
            elif sort_by == "severity":
                # Sort by severity order: critical > high > medium > low
                severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
                return sorted(
                    attacks, key=lambda a: severity_order.get(a.severity.value, 0), reverse=reverse
                )
            elif sort_by == "sophistication":
                return sorted(attacks, key=lambda a: a.sophistication, reverse=reverse)
            elif sort_by == "created" or sort_by == "creation_date":
                return sorted(attacks, key=lambda a: a.metadata.creation_date, reverse=reverse)
            elif sort_by == "updated" or sort_by == "last_updated":
                return sorted(attacks, key=lambda a: a.metadata.last_updated, reverse=reverse)
            elif sort_by == "effectiveness":
                # Put attacks without scores at the end
                return sorted(
                    attacks,
                    key=lambda a: (
                        a.metadata.effectiveness_score
                        if a.metadata.effectiveness_score is not None
                        else (-1 if reverse else 999)
                    ),
                    reverse=reverse,
                )
            elif sort_by == "length":
                return sorted(attacks, key=lambda a: len(a.content), reverse=reverse)
            else:
                # Unknown sort field, return unsorted
                return attacks
        except Exception:
            # If sorting fails, return original list
            return attacks

    def get_suggestions(self, query: str, attacks: List[Attack]) -> List[str]:
        """Get search suggestions based on query and available attacks."""
        if not query or len(query) < 2:
            return []

        query = query.lower()
        suggestions = set()

        # Collect terms from titles and tags
        all_terms = set()

        for attack in attacks:
            # Add words from title
            title_words = re.findall(r"\b\w{3,}\b", attack.title.lower())
            all_terms.update(title_words)

            # Add tags
            all_terms.update(attack.metadata.tags)

        # Find matching suggestions
        for term in all_terms:
            if query in term and term != query:
                suggestions.add(term)

        # Sort by length (shorter suggestions first)
        return sorted(suggestions, key=len)[:10]
