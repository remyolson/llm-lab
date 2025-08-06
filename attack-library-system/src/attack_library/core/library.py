"""Core attack library implementation."""

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from ..core.models import (
    Attack,
    AttackCategory,
    AttackLibrarySchema,
    AttackMetadata,
    AttackSeverity,
)
from ..core.search import SearchEngine, SearchFilter

logger = logging.getLogger(__name__)


class AttackLibrary:
    """Main attack library management class."""

    def __init__(self, library_path: Optional[Union[Path, str]] = None):
        """
        Initialize attack library.

        Args:
            library_path: Path to library storage file
        """
        if library_path is None:
            self.library_path = Path("attack_library.json")
        elif isinstance(library_path, str):
            self.library_path = Path(library_path)
        else:
            self.library_path = library_path
        self.attacks: Dict[str, Attack] = {}
        self.search_engine = SearchEngine()

        # Index structures for fast lookups
        self._category_index: Dict[AttackCategory, Set[str]] = defaultdict(set)
        self._severity_index: Dict[AttackSeverity, Set[str]] = defaultdict(set)
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)
        self._model_index: Dict[str, Set[str]] = defaultdict(set)

        # Statistics
        self._stats_cache: Optional[Dict[str, Any]] = None
        self._stats_last_updated: Optional[datetime] = None

        # Load existing library if available
        if self.library_path.exists():
            self.load_from_file(self.library_path)

    def add_attack(
        self,
        title: str,
        content: str,
        category: Union[AttackCategory, str],
        severity: Union[AttackSeverity, str],
        sophistication: int = 1,
        target_models: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        attack_id: Optional[str] = None,
    ) -> Attack:
        """
        Add a new attack to the library.

        Args:
            title: Attack title
            content: Attack prompt content
            category: Attack category
            severity: Attack severity
            sophistication: Sophistication level (1-5)
            target_models: List of target models
            metadata: Additional metadata
            attack_id: Optional custom attack ID

        Returns:
            Created Attack instance
        """
        # Convert string enums to enum types
        if isinstance(category, str):
            category = AttackCategory(category.lower())
        if isinstance(severity, str):
            severity = AttackSeverity(severity.lower())

        # Create metadata
        attack_metadata = AttackMetadata()
        if metadata:
            # Update with provided metadata
            for key, value in metadata.items():
                if hasattr(attack_metadata, key):
                    setattr(attack_metadata, key, value)

        # Create attack
        attack = Attack(
            id=attack_id or Attack.__fields__["id"].default_factory(),
            title=title,
            content=content,
            category=category,
            severity=severity,
            sophistication=sophistication,
            target_models=target_models or [],
            metadata=attack_metadata,
        )

        # Post-process metadata
        attack.__post_init__()

        # Add to library
        self.attacks[attack.id] = attack
        self._update_indices(attack)
        self._invalidate_cache()

        logger.info(f"Added attack '{attack.id}': {attack.title}")
        return attack

    def remove_attack(self, attack_id: str) -> bool:
        """
        Remove an attack from the library.

        Args:
            attack_id: Attack ID to remove

        Returns:
            True if attack was removed, False if not found
        """
        if attack_id not in self.attacks:
            return False

        attack = self.attacks[attack_id]

        # Remove from indices
        self._remove_from_indices(attack)

        # Remove from main collection
        del self.attacks[attack_id]
        self._invalidate_cache()

        logger.info(f"Removed attack '{attack_id}': {attack.title}")
        return True

    def get_attack(self, attack_id: str) -> Optional[Attack]:
        """
        Get attack by ID.

        Args:
            attack_id: Attack ID

        Returns:
            Attack instance or None if not found
        """
        return self.attacks.get(attack_id)

    def update_attack(self, attack_id: str, **updates) -> Optional[Attack]:
        """
        Update an existing attack.

        Args:
            attack_id: Attack ID to update
            **updates: Fields to update

        Returns:
            Updated Attack instance or None if not found
        """
        if attack_id not in self.attacks:
            return None

        attack = self.attacks[attack_id]

        # Remove from indices before update
        self._remove_from_indices(attack)

        # Update fields
        for field, value in updates.items():
            if hasattr(attack, field):
                setattr(attack, field, value)
            elif hasattr(attack.metadata, field):
                setattr(attack.metadata, field, value)

        # Update timestamp
        attack.metadata.last_updated = datetime.now()

        # Update indices
        self._update_indices(attack)
        self._invalidate_cache()

        logger.info(f"Updated attack '{attack_id}': {attack.title}")
        return attack

    def search(
        self,
        query: Optional[str] = None,
        category: Optional[Union[AttackCategory, str]] = None,
        severity: Optional[Union[AttackSeverity, str, List[Union[AttackSeverity, str]]]] = None,
        min_sophistication: Optional[int] = None,
        max_sophistication: Optional[int] = None,
        target_model: Optional[str] = None,
        tags: Optional[Union[str, List[str]]] = None,
        has_effectiveness_score: Optional[bool] = None,
        min_effectiveness: Optional[float] = None,
        is_verified: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> List[Attack]:
        """
        Search attacks with various filters.

        Args:
            query: Text search query
            category: Attack category filter
            severity: Severity level(s) filter
            min_sophistication: Minimum sophistication level
            max_sophistication: Maximum sophistication level
            target_model: Target model filter
            tags: Tag(s) filter
            has_effectiveness_score: Filter by effectiveness score presence
            min_effectiveness: Minimum effectiveness score
            is_verified: Filter by verification status
            limit: Maximum number of results

        Returns:
            List of matching attacks
        """
        # Build search filter
        search_filter = SearchFilter(
            query=query,
            category=category,
            severity=severity,
            min_sophistication=min_sophistication,
            max_sophistication=max_sophistication,
            target_model=target_model,
            tags=tags,
            has_effectiveness_score=has_effectiveness_score,
            min_effectiveness=min_effectiveness,
            is_verified=is_verified,
            limit=limit,
        )

        return self.search_engine.search(list(self.attacks.values()), search_filter)

    def get_by_category(self, category: Union[AttackCategory, str]) -> List[Attack]:
        """Get all attacks in a category."""
        if isinstance(category, str):
            category = AttackCategory(category.lower())

        attack_ids = self._category_index.get(category, set())
        return [self.attacks[aid] for aid in attack_ids if aid in self.attacks]

    def get_by_severity(self, severity: Union[AttackSeverity, str]) -> List[Attack]:
        """Get all attacks with specified severity."""
        if isinstance(severity, str):
            severity = AttackSeverity(severity.lower())

        attack_ids = self._severity_index.get(severity, set())
        return [self.attacks[aid] for aid in attack_ids if aid in self.attacks]

    def get_by_tag(self, tag: str) -> List[Attack]:
        """Get all attacks with specified tag."""
        tag = tag.lower().strip()
        attack_ids = self._tag_index.get(tag, set())
        return [self.attacks[aid] for aid in attack_ids if aid in self.attacks]

    def get_by_model(self, model: str) -> List[Attack]:
        """Get all attacks targeting specified model."""
        model = model.lower().strip()
        attack_ids = self._model_index.get(model, set())
        return [self.attacks[aid] for aid in attack_ids if aid in self.attacks]

    def get_random_attacks(self, count: int = 10, **filters) -> List[Attack]:
        """Get random attacks with optional filters."""
        import random

        if filters:
            candidates = self.search(**filters)
        else:
            candidates = list(self.attacks.values())

        if len(candidates) <= count:
            return candidates

        return random.sample(candidates, count)

    def load_from_file(self, file_path: Union[str, Path]) -> int:
        """
        Load attacks from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            Number of attacks loaded
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.warning(f"Library file not found: {file_path}")
            return 0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load as schema
            if isinstance(data, dict) and "attacks" in data:
                schema = AttackLibrarySchema.from_dict(data)
                attacks_to_load = schema.attacks
            elif isinstance(data, list):
                # Legacy format - list of attacks
                attacks_to_load = [Attack.from_dict(attack_data) for attack_data in data]
            else:
                raise ValueError("Invalid file format")

            # Clear existing attacks and load new ones
            self.clear()

            loaded_count = 0
            for attack in attacks_to_load:
                self.attacks[attack.id] = attack
                self._update_indices(attack)
                loaded_count += 1

            self._invalidate_cache()
            logger.info(f"Loaded {loaded_count} attacks from {file_path}")

            return loaded_count

        except Exception as e:
            logger.error(f"Error loading attacks from {file_path}: {e}")
            raise

    def save_to_file(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save attacks to JSON file.

        Args:
            file_path: Optional path to save to (defaults to library_path)
        """
        file_path = Path(file_path) if file_path else self.library_path

        try:
            # Create schema
            schema = AttackLibrarySchema(
                attacks=list(self.attacks.values()),
                metadata={
                    "total_by_category": self.get_category_counts(),
                    "total_by_severity": self.get_severity_counts(),
                    "statistics": self.get_statistics(),
                },
            )

            # Save to file
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(schema.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(self.attacks)} attacks to {file_path}")

        except Exception as e:
            logger.error(f"Error saving attacks to {file_path}: {e}")
            raise

    def clear(self) -> None:
        """Clear all attacks from the library."""
        self.attacks.clear()
        self._category_index.clear()
        self._severity_index.clear()
        self._tag_index.clear()
        self._model_index.clear()
        self._invalidate_cache()

        logger.info("Cleared all attacks from library")

    def get_statistics(self) -> Dict[str, Any]:
        """Get library statistics."""
        if (
            self._stats_cache
            and self._stats_last_updated
            and (datetime.now() - self._stats_last_updated).seconds < 300
        ):  # 5 minute cache
            return self._stats_cache

        attacks = list(self.attacks.values())
        total = len(attacks)

        if total == 0:
            return {"total_attacks": 0}

        # Category distribution
        category_counts = Counter(attack.category.value for attack in attacks)

        # Severity distribution
        severity_counts = Counter(attack.severity.value for attack in attacks)

        # Sophistication distribution
        sophistication_counts = Counter(attack.sophistication for attack in attacks)

        # Verification stats
        verified_count = sum(1 for attack in attacks if attack.is_verified)

        # Effectiveness stats
        attacks_with_scores = [
            attack for attack in attacks if attack.metadata.effectiveness_score is not None
        ]

        effectiveness_stats = {}
        if attacks_with_scores:
            scores = [attack.metadata.effectiveness_score for attack in attacks_with_scores]
            effectiveness_stats = {
                "count": len(scores),
                "average": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
            }

        # Tag stats
        all_tags = set()
        for attack in attacks:
            all_tags.update(attack.metadata.tags)

        # Model coverage
        all_models = set()
        for attack in attacks:
            all_models.update(attack.target_models)

        stats = {
            "total_attacks": total,
            "categories": dict(category_counts),
            "severities": dict(severity_counts),
            "sophistication_levels": dict(sophistication_counts),
            "verification": {
                "verified": verified_count,
                "unverified": total - verified_count,
                "percentage_verified": (verified_count / total) * 100,
            },
            "effectiveness": effectiveness_stats,
            "tags": {
                "total_unique_tags": len(all_tags),
                "most_common": Counter(
                    tag for attack in attacks for tag in attack.metadata.tags
                ).most_common(10),
            },
            "models": {
                "total_unique_models": len(all_models),
                "most_targeted": Counter(
                    model for attack in attacks for model in attack.target_models
                ).most_common(10),
            },
            "content_stats": {
                "average_length": sum(len(attack.content) for attack in attacks) / total,
                "total_characters": sum(len(attack.content) for attack in attacks),
                "total_words": sum(len(attack.content.split()) for attack in attacks),
            },
        }

        # Cache results
        self._stats_cache = stats
        self._stats_last_updated = datetime.now()

        return stats

    def get_category_counts(self) -> Dict[str, int]:
        """Get attack count by category."""
        return {cat.value: len(ids) for cat, ids in self._category_index.items()}

    def get_severity_counts(self) -> Dict[str, int]:
        """Get attack count by severity."""
        return {sev.value: len(ids) for sev, ids in self._severity_index.items()}

    def get_similar_attacks(
        self, attack_id: str, limit: int = 5, min_similarity: float = 0.3
    ) -> List[tuple[Attack, float]]:
        """
        Find attacks similar to the given attack.

        Args:
            attack_id: Reference attack ID
            limit: Maximum number of similar attacks
            min_similarity: Minimum similarity threshold

        Returns:
            List of (attack, similarity_score) tuples
        """
        reference_attack = self.get_attack(attack_id)
        if not reference_attack:
            return []

        similarities = []
        for attack in self.attacks.values():
            if attack.id == attack_id:
                continue

            similarity = reference_attack.get_similarity_score(attack)
            if similarity >= min_similarity:
                similarities.append((attack, similarity))

        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:limit]

    def bulk_update(self, updates: Dict[str, Dict[str, Any]]) -> int:
        """
        Perform bulk updates on multiple attacks.

        Args:
            updates: Dictionary mapping attack_id to update fields

        Returns:
            Number of attacks successfully updated
        """
        updated_count = 0

        for attack_id, update_data in updates.items():
            if self.update_attack(attack_id, **update_data):
                updated_count += 1

        return updated_count

    def export_attacks(
        self,
        format: str = "json",
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Export attacks in various formats.

        Args:
            format: Export format ('json', 'csv', 'dict')
            filters: Optional search filters
            include_metadata: Whether to include metadata

        Returns:
            Exported data
        """
        # Get attacks to export
        if filters:
            attacks = self.search(**filters)
        else:
            attacks = list(self.attacks.values())

        if format.lower() == "dict":
            return [attack.to_dict() for attack in attacks]
        elif format.lower() == "json":
            return json.dumps(
                [attack.to_dict() for attack in attacks], indent=2, ensure_ascii=False
            )
        elif format.lower() == "csv":
            import csv
            from io import StringIO

            output = StringIO()
            if not attacks:
                return ""

            fieldnames = ["id", "title", "content", "category", "severity", "sophistication"]
            if include_metadata:
                fieldnames.extend(["tags", "effectiveness_score", "creation_date"])

            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

            for attack in attacks:
                row = {
                    "id": attack.id,
                    "title": attack.title,
                    "content": attack.content,
                    "category": attack.category.value,
                    "severity": attack.severity.value,
                    "sophistication": attack.sophistication,
                }

                if include_metadata:
                    row.update(
                        {
                            "tags": ",".join(attack.metadata.tags),
                            "effectiveness_score": attack.metadata.effectiveness_score,
                            "creation_date": attack.metadata.creation_date.isoformat(),
                        }
                    )

                writer.writerow(row)

            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _update_indices(self, attack: Attack) -> None:
        """Update index structures for an attack."""
        # Category index
        self._category_index[attack.category].add(attack.id)

        # Severity index
        self._severity_index[attack.severity].add(attack.id)

        # Tag index
        for tag in attack.metadata.tags:
            self._tag_index[tag.lower()].add(attack.id)

        # Model index
        for model in attack.target_models:
            self._model_index[model.lower()].add(attack.id)

    def _remove_from_indices(self, attack: Attack) -> None:
        """Remove attack from index structures."""
        # Category index
        self._category_index[attack.category].discard(attack.id)

        # Severity index
        self._severity_index[attack.severity].discard(attack.id)

        # Tag index
        for tag in attack.metadata.tags:
            self._tag_index[tag.lower()].discard(attack.id)

        # Model index
        for model in attack.target_models:
            self._model_index[model.lower()].discard(attack.id)

    def _invalidate_cache(self) -> None:
        """Invalidate statistics cache."""
        self._stats_cache = None
        self._stats_last_updated = None

    def __len__(self) -> int:
        """Return number of attacks in library."""
        return len(self.attacks)

    def __contains__(self, attack_id: str) -> bool:
        """Check if attack exists in library."""
        return attack_id in self.attacks

    def __iter__(self):
        """Iterate over attacks."""
        return iter(self.attacks.values())
