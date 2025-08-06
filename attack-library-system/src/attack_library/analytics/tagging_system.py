"""Comprehensive tagging system for attack characteristics and metadata."""

import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..core.models import Attack, AttackCategory, AttackSeverity

logger = logging.getLogger(__name__)


class TagCategory(Enum):
    """Categories of tags for organization."""

    TECHNIQUE = "technique"  # Attack technique used
    TARGET = "target"  # Target vulnerability/capability
    CONTEXT = "context"  # Required context or setup
    EVASION = "evasion"  # Evasion methods employed
    SOPHISTICATION = "sophistication"  # Technical sophistication indicators
    DOMAIN = "domain"  # Domain-specific tags
    SOURCE = "source"  # Source or origin indicators
    EFFECTIVENESS = "effectiveness"  # Effectiveness indicators


@dataclass
class TagDefinition:
    """Definition of a structured tag."""

    name: str
    category: TagCategory
    description: str
    aliases: List[str]
    parent_tags: List[str]
    child_tags: List[str]
    created_date: str
    usage_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["category"] = self.category.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TagDefinition":
        """Create from dictionary."""
        data["category"] = TagCategory(data["category"])
        return cls(**data)


class TaggingSystem:
    """Comprehensive tagging system for attack characteristics."""

    def __init__(self, tags_file: Optional[Path] = None):
        """
        Initialize tagging system.

        Args:
            tags_file: File to store tag definitions
        """
        self.tags_file = tags_file or Path("attack_tags.json")
        self.tag_definitions: Dict[str, TagDefinition] = {}
        self.tag_hierarchy: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_hierarchy: Dict[str, Set[str]] = defaultdict(set)

        # Initialize predefined tags
        self._initialize_predefined_tags()

        # Load existing tags
        self.load_tags()

    def _initialize_predefined_tags(self):
        """Initialize predefined tag structure."""
        self.predefined_tags = {
            # Technique tags
            TagCategory.TECHNIQUE: [
                ("jailbreaking", "Attempts to bypass safety guidelines"),
                ("prompt_injection", "Injects malicious instructions into prompts"),
                ("role_playing", "Uses role-playing to bypass restrictions"),
                ("hypothetical_scenario", "Frames harmful requests as hypothetical"),
                ("chain_of_thought", "Exploits reasoning capabilities"),
                ("few_shot_learning", "Uses examples to teach harmful behavior"),
                ("context_stuffing", "Overwhelms context with irrelevant information"),
                ("attention_manipulation", "Manipulates attention mechanisms"),
                ("gradient_optimization", "Uses gradient-based optimization"),
                ("adversarial_suffix", "Appends adversarial tokens"),
            ],
            # Target tags
            TagCategory.TARGET: [
                ("safety_guidelines", "Targets safety and ethical guidelines"),
                ("content_policy", "Targets content moderation policies"),
                ("training_data", "Attempts to extract training data"),
                ("model_weights", "Targets model parameters"),
                ("system_prompts", "Targets system-level instructions"),
                ("personal_information", "Targets personal data extraction"),
                ("harmful_content", "Aims to generate harmful content"),
                ("biased_output", "Exploits model biases"),
                ("factual_accuracy", "Targets factual knowledge"),
                ("reasoning_capability", "Exploits reasoning flaws"),
            ],
            # Context tags
            TagCategory.CONTEXT: [
                ("no_context", "Works without special context"),
                ("conversation_history", "Requires conversation setup"),
                ("specific_domain", "Requires domain-specific knowledge"),
                ("multi_turn", "Requires multiple conversation turns"),
                ("priming_required", "Needs priming or setup"),
                ("authority_establishment", "Requires establishing authority"),
                ("trust_building", "Needs trust establishment"),
                ("emotional_manipulation", "Uses emotional context"),
            ],
            # Evasion tags
            TagCategory.EVASION: [
                ("encoding", "Uses character or text encoding"),
                ("obfuscation", "Obscures malicious intent"),
                ("indirection", "Uses indirect references"),
                ("metaphor", "Uses metaphorical language"),
                ("foreign_language", "Uses non-English languages"),
                ("code_switching", "Switches between languages"),
                ("steganography", "Hides content in other content"),
                ("leetspeak", "Uses leetspeak encoding"),
                ("unicode_manipulation", "Exploits unicode properties"),
                ("whitespace_tricks", "Uses whitespace manipulation"),
            ],
            # Sophistication tags
            TagCategory.SOPHISTICATION: [
                ("basic", "Simple, straightforward approach"),
                ("intermediate", "Moderate technical complexity"),
                ("advanced", "High technical sophistication"),
                ("expert", "Requires deep technical knowledge"),
                ("automated", "Generated by automated systems"),
                ("hand_crafted", "Manually crafted by humans"),
                ("research_based", "Based on academic research"),
                ("zero_day", "Uses novel, unknown techniques"),
            ],
            # Domain tags
            TagCategory.DOMAIN: [
                ("cybersecurity", "Security and hacking domain"),
                ("healthcare", "Medical and health information"),
                ("finance", "Financial and monetary topics"),
                ("legal", "Legal and regulatory content"),
                ("education", "Educational and academic content"),
                ("entertainment", "Media and entertainment"),
                ("politics", "Political and governmental topics"),
                ("technology", "Technical and engineering topics"),
                ("science", "Scientific and research content"),
                ("personal", "Personal and private information"),
            ],
            # Source tags
            TagCategory.SOURCE: [
                ("academic_paper", "From academic research"),
                ("ctf_challenge", "From CTF competitions"),
                ("bug_bounty", "From bug bounty programs"),
                ("red_team", "From red team exercises"),
                ("community", "From security community"),
                ("synthetic", "AI-generated content"),
                ("real_world", "From real-world incidents"),
                ("honeypot", "From honeypot systems"),
            ],
            # Effectiveness tags
            TagCategory.EFFECTIVENESS: [
                ("high_success", "High success rate observed"),
                ("moderate_success", "Moderate success rate"),
                ("low_success", "Low success rate"),
                ("model_specific", "Effective against specific models"),
                ("universal", "Works across multiple models"),
                ("version_sensitive", "Effectiveness varies by version"),
                ("context_dependent", "Success depends on context"),
                ("unreliable", "Inconsistent effectiveness"),
            ],
        }

    def load_tags(self):
        """Load tag definitions from file."""
        if self.tags_file.exists():
            try:
                with open(self.tags_file, "r") as f:
                    data = json.load(f)

                    # Load tag definitions
                    for tag_data in data.get("tag_definitions", []):
                        tag_def = TagDefinition.from_dict(tag_data)
                        self.tag_definitions[tag_def.name] = tag_def

                    # Rebuild hierarchy
                    self._rebuild_hierarchy()

                logger.info(f"Loaded {len(self.tag_definitions)} tag definitions")
            except Exception as e:
                logger.error(f"Failed to load tags: {e}")
                self._create_default_tags()
        else:
            self._create_default_tags()

    def _create_default_tags(self):
        """Create default tag definitions."""
        from datetime import datetime

        for category, tags in self.predefined_tags.items():
            for tag_name, description in tags:
                tag_def = TagDefinition(
                    name=tag_name,
                    category=category,
                    description=description,
                    aliases=[],
                    parent_tags=[],
                    child_tags=[],
                    created_date=datetime.now().isoformat(),
                    usage_count=0,
                )
                self.tag_definitions[tag_name] = tag_def

        # Set up some hierarchical relationships
        self._setup_tag_hierarchies()
        self.save_tags()

    def _setup_tag_hierarchies(self):
        """Set up hierarchical relationships between tags."""
        # Sophistication hierarchy
        self._add_hierarchy("basic", "intermediate")
        self._add_hierarchy("intermediate", "advanced")
        self._add_hierarchy("advanced", "expert")

        # Technique hierarchies
        self._add_hierarchy("jailbreaking", "role_playing")
        self._add_hierarchy("jailbreaking", "hypothetical_scenario")
        self._add_hierarchy("prompt_injection", "context_stuffing")

        # Evasion hierarchies
        self._add_hierarchy("obfuscation", "encoding")
        self._add_hierarchy("obfuscation", "indirection")
        self._add_hierarchy("encoding", "leetspeak")
        self._add_hierarchy("encoding", "unicode_manipulation")

    def _add_hierarchy(self, parent: str, child: str):
        """Add parent-child relationship between tags."""
        if parent in self.tag_definitions:
            self.tag_definitions[parent].child_tags.append(child)
        if child in self.tag_definitions:
            self.tag_definitions[child].parent_tags.append(parent)

    def _rebuild_hierarchy(self):
        """Rebuild hierarchy mappings from tag definitions."""
        self.tag_hierarchy.clear()
        self.reverse_hierarchy.clear()

        for tag_name, tag_def in self.tag_definitions.items():
            # Forward hierarchy (parent -> children)
            self.tag_hierarchy[tag_name] = set(tag_def.child_tags)

            # Reverse hierarchy (child -> parents)
            for child in tag_def.child_tags:
                self.reverse_hierarchy[child].add(tag_name)

    def save_tags(self):
        """Save tag definitions to file."""
        try:
            data = {
                "metadata": {
                    "total_tags": len(self.tag_definitions),
                    "categories": list(TagCategory),
                    "last_updated": str(datetime.now()),
                },
                "tag_definitions": [tag_def.to_dict() for tag_def in self.tag_definitions.values()],
            }

            with open(self.tags_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Saved {len(self.tag_definitions)} tag definitions")
        except Exception as e:
            logger.error(f"Failed to save tags: {e}")

    def add_tag_definition(
        self,
        name: str,
        category: TagCategory,
        description: str,
        aliases: Optional[List[str]] = None,
        parent_tags: Optional[List[str]] = None,
    ) -> TagDefinition:
        """
        Add a new tag definition.

        Args:
            name: Tag name
            category: Tag category
            description: Tag description
            aliases: Alternative names
            parent_tags: Parent tags in hierarchy

        Returns:
            Created tag definition
        """
        from datetime import datetime

        if name in self.tag_definitions:
            raise ValueError(f"Tag '{name}' already exists")

        tag_def = TagDefinition(
            name=name,
            category=category,
            description=description,
            aliases=aliases or [],
            parent_tags=parent_tags or [],
            child_tags=[],
            created_date=datetime.now().isoformat(),
            usage_count=0,
        )

        self.tag_definitions[name] = tag_def

        # Update parent tags
        for parent in parent_tags or []:
            if parent in self.tag_definitions:
                self.tag_definitions[parent].child_tags.append(name)

        self._rebuild_hierarchy()
        self.save_tags()

        logger.info(f"Added new tag definition: {name}")
        return tag_def

    def suggest_tags_for_attack(self, attack: Attack) -> List[Tuple[str, float]]:
        """
        Suggest appropriate tags for an attack based on content analysis.

        Args:
            attack: Attack to analyze

        Returns:
            List of (tag_name, confidence_score) tuples
        """
        suggestions = []
        content_lower = attack.content.lower()
        title_lower = attack.title.lower()

        # Analyze content for tag indicators
        tag_indicators = {
            # Technique indicators
            "jailbreaking": ["ignore", "forget", "previous", "instructions", "rules"],
            "role_playing": ["pretend", "act as", "you are", "roleplay", "character"],
            "hypothetical_scenario": ["imagine", "suppose", "what if", "hypothetical", "scenario"],
            "chain_of_thought": ["step by step", "think about", "reasoning", "logic"],
            "prompt_injection": ["system:", "user:", "assistant:", "###", "---"],
            # Target indicators
            "harmful_content": ["harmful", "dangerous", "illegal", "unethical"],
            "personal_information": ["personal", "private", "confidential", "pii"],
            "training_data": ["training", "dataset", "memorized", "recall"],
            # Evasion indicators
            "encoding": ["base64", "encode", "decode", "cipher", "rot13"],
            "obfuscation": ["hidden", "disguise", "mask", "camouflage"],
            "foreign_language": ["español", "français", "deutsch", "中文"],
            "leetspeak": ["1337", "h4ck", "3", "1", "0"],
            # Context indicators
            "multi_turn": ["first", "then", "next", "finally", "after"],
            "authority_establishment": ["authorized", "administrator", "supervisor"],
            "emotional_manipulation": ["desperate", "help", "urgent", "please"],
            # Domain indicators
            "cybersecurity": ["hack", "exploit", "vulnerability", "security"],
            "healthcare": ["medical", "health", "doctor", "patient"],
            "finance": ["money", "financial", "banking", "investment"],
        }

        # Calculate confidence scores
        for tag, indicators in tag_indicators.items():
            if tag in self.tag_definitions:
                matches = sum(1 for indicator in indicators if indicator in content_lower)
                if matches > 0:
                    confidence = min(1.0, matches / len(indicators))
                    suggestions.append((tag, confidence))

        # Category-based suggestions
        category_tags = {
            AttackCategory.JAILBREAK: ["jailbreaking", "safety_guidelines"],
            AttackCategory.INJECTION: ["prompt_injection", "context_stuffing"],
            AttackCategory.EXTRACTION: ["training_data", "personal_information"],
            AttackCategory.MANIPULATION: ["emotional_manipulation", "trust_building"],
            AttackCategory.EVASION: ["obfuscation", "encoding"],
        }

        if attack.category in category_tags:
            for tag in category_tags[attack.category]:
                if tag in self.tag_definitions:
                    suggestions.append((tag, 0.8))

        # Severity-based suggestions
        if attack.severity == AttackSeverity.CRITICAL:
            suggestions.append(("expert", 0.7))
            suggestions.append(("high_success", 0.6))
        elif attack.severity == AttackSeverity.LOW:
            suggestions.append(("basic", 0.7))

        # Remove duplicates and sort by confidence
        unique_suggestions = {}
        for tag, confidence in suggestions:
            if tag not in unique_suggestions or confidence > unique_suggestions[tag]:
                unique_suggestions[tag] = confidence

        result = [(tag, conf) for tag, conf in unique_suggestions.items()]
        result.sort(key=lambda x: x[1], reverse=True)

        return result[:10]  # Top 10 suggestions

    def apply_tags_to_attack(self, attack: Attack, tags: List[str]) -> Set[str]:
        """
        Apply tags to an attack and update usage statistics.

        Args:
            attack: Attack to tag
            tags: List of tags to apply

        Returns:
            Set of successfully applied tags
        """
        applied_tags = set()

        for tag in tags:
            if tag in self.tag_definitions:
                attack.metadata.tags.add(tag)
                self.tag_definitions[tag].usage_count += 1
                applied_tags.add(tag)
            else:
                logger.warning(f"Unknown tag: {tag}")

        # Also add hierarchical parent tags
        for tag in applied_tags.copy():
            parents = self.reverse_hierarchy.get(tag, set())
            for parent in parents:
                attack.metadata.tags.add(parent)
                applied_tags.add(parent)
                self.tag_definitions[parent].usage_count += 1

        self.save_tags()
        return applied_tags

    def get_tag_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tag usage statistics."""
        if not self.tag_definitions:
            return {"total_tags": 0}

        # Category distribution
        category_counts = defaultdict(int)
        for tag_def in self.tag_definitions.values():
            category_counts[tag_def.category.value] += 1

        # Usage statistics
        usage_stats = {
            "total_usage": sum(tag.usage_count for tag in self.tag_definitions.values()),
            "used_tags": len([tag for tag in self.tag_definitions.values() if tag.usage_count > 0]),
            "unused_tags": len(
                [tag for tag in self.tag_definitions.values() if tag.usage_count == 0]
            ),
        }

        # Most used tags
        most_used = sorted(
            self.tag_definitions.values(), key=lambda t: t.usage_count, reverse=True
        )[:10]

        # Hierarchy statistics
        hierarchy_stats = {
            "tags_with_children": len([t for t in self.tag_definitions.values() if t.child_tags]),
            "tags_with_parents": len([t for t in self.tag_definitions.values() if t.parent_tags]),
            "max_depth": self._calculate_max_hierarchy_depth(),
        }

        return {
            "total_tags": len(self.tag_definitions),
            "category_distribution": dict(category_counts),
            "usage_statistics": usage_stats,
            "most_used_tags": [
                {"name": tag.name, "usage_count": tag.usage_count, "category": tag.category.value}
                for tag in most_used
            ],
            "hierarchy_statistics": hierarchy_stats,
        }

    def _calculate_max_hierarchy_depth(self) -> int:
        """Calculate maximum depth of tag hierarchy."""

        def get_depth(tag_name: str, visited: Set[str] = None) -> int:
            if visited is None:
                visited = set()

            if tag_name in visited:
                return 0  # Cycle detected

            visited.add(tag_name)
            children = self.tag_hierarchy.get(tag_name, set())

            if not children:
                return 1

            return 1 + max(get_depth(child, visited.copy()) for child in children)

        if not self.tag_definitions:
            return 0

        # Find root tags (tags with no parents)
        root_tags = [
            name for name, tag_def in self.tag_definitions.items() if not tag_def.parent_tags
        ]

        if not root_tags:
            return 1  # No hierarchy

        return max(get_depth(root_tag) for root_tag in root_tags)

    def search_tags(
        self, query: str, category: Optional[TagCategory] = None, include_aliases: bool = True
    ) -> List[TagDefinition]:
        """
        Search for tags by name or description.

        Args:
            query: Search query
            category: Optional category filter
            include_aliases: Whether to search aliases

        Returns:
            Matching tag definitions
        """
        query_lower = query.lower()
        matches = []

        for tag_def in self.tag_definitions.values():
            # Category filter
            if category and tag_def.category != category:
                continue

            # Name match
            if query_lower in tag_def.name.lower():
                matches.append((tag_def, 1.0))  # Exact name match gets highest score
                continue

            # Description match
            if query_lower in tag_def.description.lower():
                matches.append((tag_def, 0.8))
                continue

            # Alias match
            if include_aliases:
                for alias in tag_def.aliases:
                    if query_lower in alias.lower():
                        matches.append((tag_def, 0.9))
                        break

        # Sort by relevance score
        matches.sort(key=lambda x: x[1], reverse=True)
        return [tag_def for tag_def, _ in matches]

    def get_tag_relationships(self, tag_name: str) -> Dict[str, Any]:
        """
        Get comprehensive relationship information for a tag.

        Args:
            tag_name: Tag to analyze

        Returns:
            Relationship information
        """
        if tag_name not in self.tag_definitions:
            return {"error": f"Tag {tag_name} not found"}

        tag_def = self.tag_definitions[tag_name]

        return {
            "tag_name": tag_name,
            "category": tag_def.category.value,
            "description": tag_def.description,
            "parent_tags": tag_def.parent_tags,
            "child_tags": tag_def.child_tags,
            "aliases": tag_def.aliases,
            "usage_count": tag_def.usage_count,
            "created_date": tag_def.created_date,
            "related_tags": self._find_related_tags(tag_name),
            "hierarchy_level": self._get_hierarchy_level(tag_name),
        }

    def _find_related_tags(self, tag_name: str) -> List[str]:
        """Find tags related through hierarchy or category."""
        related = set()
        tag_def = self.tag_definitions[tag_name]

        # Same category tags
        for name, other_def in self.tag_definitions.items():
            if name != tag_name and other_def.category == tag_def.category:
                related.add(name)

        # Sibling tags (same parent)
        for parent in tag_def.parent_tags:
            if parent in self.tag_definitions:
                related.update(self.tag_definitions[parent].child_tags)

        # Remove self and limit results
        related.discard(tag_name)
        return list(related)[:10]

    def _get_hierarchy_level(self, tag_name: str) -> int:
        """Get the hierarchy level of a tag (0 = root)."""
        if tag_name not in self.tag_definitions:
            return -1

        tag_def = self.tag_definitions[tag_name]
        if not tag_def.parent_tags:
            return 0

        # Level is 1 + max level of parents
        parent_levels = [
            self._get_hierarchy_level(parent)
            for parent in tag_def.parent_tags
            if parent in self.tag_definitions
        ]

        return 1 + max(parent_levels) if parent_levels else 0

    def export_tag_system(self, output_file: Path, format_type: str = "json"):
        """
        Export the complete tag system.

        Args:
            output_file: Output file path
            format_type: Export format ('json', 'csv')
        """
        if format_type.lower() == "json":
            export_data = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_tags": len(self.tag_definitions),
                    "categories": [cat.value for cat in TagCategory],
                },
                "tag_definitions": [tag_def.to_dict() for tag_def in self.tag_definitions.values()],
                "statistics": self.get_tag_statistics(),
            }

            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

        elif format_type.lower() == "csv":
            import csv

            with open(output_file, "w", newline="") as f:
                fieldnames = [
                    "name",
                    "category",
                    "description",
                    "usage_count",
                    "parent_tags",
                    "child_tags",
                    "aliases",
                    "created_date",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for tag_def in self.tag_definitions.values():
                    writer.writerow(
                        {
                            "name": tag_def.name,
                            "category": tag_def.category.value,
                            "description": tag_def.description,
                            "usage_count": tag_def.usage_count,
                            "parent_tags": ";".join(tag_def.parent_tags),
                            "child_tags": ";".join(tag_def.child_tags),
                            "aliases": ";".join(tag_def.aliases),
                            "created_date": tag_def.created_date,
                        }
                    )
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

        logger.info(f"Exported tag system to {output_file}")
