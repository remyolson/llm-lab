"""Core data models for the attack library system."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field, validator


class AttackCategory(str, Enum):
    """Attack categories."""

    JAILBREAK = "jailbreak"
    INJECTION = "injection"
    EXTRACTION = "extraction"
    MANIPULATION = "manipulation"
    EVASION = "evasion"


class AttackSeverity(str, Enum):
    """Attack severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackMetadata(BaseModel):
    """Metadata for attack prompts."""

    source: str = Field(..., description="Source of the attack (e.g., 'research_paper', 'manual')")
    effectiveness_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Effectiveness score from 0.0 to 1.0"
    )
    creation_date: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    tags: Set[str] = Field(default_factory=set, description="Searchable tags")
    author: Optional[str] = Field(None, description="Author or creator")
    references: List[str] = Field(default_factory=list, description="Research references")
    tested_models: List[str] = Field(default_factory=list, description="Models tested against")
    success_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Success rate in testing"
    )
    language: str = Field(default="en", description="Language of the attack")
    character_count: Optional[int] = Field(None, ge=0)
    word_count: Optional[int] = Field(None, ge=0)

    @validator("tags", pre=True)
    def convert_tags_to_set(cls, v):
        """Convert tags to set if provided as list."""
        if isinstance(v, list):
            return set(v)
        return v


class Attack(BaseModel):
    """Core attack model."""

    id: str = Field(default_factory=lambda: str(uuid4())[:8], description="Unique attack ID")
    title: str = Field(..., min_length=1, max_length=200, description="Attack title")
    content: str = Field(..., min_length=1, description="Attack prompt content")
    category: AttackCategory = Field(..., description="Attack category")
    severity: AttackSeverity = Field(..., description="Attack severity level")
    sophistication: int = Field(
        ..., ge=1, le=5, description="Sophistication level from 1 (basic) to 5 (advanced)"
    )
    target_models: List[str] = Field(
        default_factory=list, description="Target model types or names"
    )
    metadata: AttackMetadata = Field(default_factory=AttackMetadata)

    # Versioning
    version: str = Field(default="1.0", description="Attack version")
    parent_id: Optional[str] = Field(None, description="Parent attack ID for variants")
    variant_type: Optional[str] = Field(None, description="Type of variant if applicable")

    # Status tracking
    is_active: bool = Field(default=True, description="Whether attack is active")
    is_verified: bool = Field(default=False, description="Whether attack is verified")
    verification_date: Optional[datetime] = Field(None)

    def __post_init__(self):
        """Post-initialization processing."""
        # Update character and word counts
        self.metadata.character_count = len(self.content)
        self.metadata.word_count = len(self.content.split())

        # Update last_updated timestamp
        self.metadata.last_updated = datetime.now()

    @validator("id")
    def validate_id(cls, v):
        """Validate attack ID format."""
        if not v or len(v) < 3:
            raise ValueError("Attack ID must be at least 3 characters long")
        return v

    @validator("content")
    def validate_content(cls, v):
        """Validate attack content."""
        if len(v.strip()) == 0:
            raise ValueError("Attack content cannot be empty")
        return v.strip()

    @validator("target_models")
    def validate_target_models(cls, v):
        """Validate target models list."""
        return [model.lower().strip() for model in v if model.strip()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = self.dict()

        # Convert datetime objects to ISO strings
        if isinstance(data["metadata"]["creation_date"], datetime):
            data["metadata"]["creation_date"] = data["metadata"]["creation_date"].isoformat()
        if isinstance(data["metadata"]["last_updated"], datetime):
            data["metadata"]["last_updated"] = data["metadata"]["last_updated"].isoformat()
        if data.get("verification_date"):
            data["verification_date"] = data["verification_date"].isoformat()

        # Convert set to list for JSON serialization
        if isinstance(data["metadata"]["tags"], set):
            data["metadata"]["tags"] = list(data["metadata"]["tags"])

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Attack":
        """Create Attack instance from dictionary."""
        # Parse datetime strings
        if "metadata" in data:
            metadata = data["metadata"]
            if isinstance(metadata.get("creation_date"), str):
                metadata["creation_date"] = datetime.fromisoformat(metadata["creation_date"])
            if isinstance(metadata.get("last_updated"), str):
                metadata["last_updated"] = datetime.fromisoformat(metadata["last_updated"])

        if isinstance(data.get("verification_date"), str):
            data["verification_date"] = datetime.fromisoformat(data["verification_date"])

        return cls(**data)

    def add_tag(self, tag: str) -> None:
        """Add a tag to the attack."""
        self.metadata.tags.add(tag.lower().strip())
        self.metadata.last_updated = datetime.now()

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the attack."""
        self.metadata.tags.discard(tag.lower().strip())
        self.metadata.last_updated = datetime.now()

    def has_tag(self, tag: str) -> bool:
        """Check if attack has a specific tag."""
        return tag.lower().strip() in self.metadata.tags

    def update_effectiveness(self, score: float) -> None:
        """Update effectiveness score."""
        if not 0.0 <= score <= 1.0:
            raise ValueError("Effectiveness score must be between 0.0 and 1.0")

        self.metadata.effectiveness_score = score
        self.metadata.last_updated = datetime.now()

    def mark_verified(self) -> None:
        """Mark attack as verified."""
        self.is_verified = True
        self.verification_date = datetime.now()
        self.metadata.last_updated = datetime.now()

    def create_variant(self, new_content: str, variant_type: str = "manual", **kwargs) -> "Attack":
        """Create a variant of this attack."""
        variant_data = self.dict()
        variant_data.update(
            {
                "id": str(uuid4())[:8],
                "content": new_content,
                "parent_id": self.id,
                "variant_type": variant_type,
                "version": "1.0",
                "is_verified": False,
                "verification_date": None,
                **kwargs,
            }
        )

        # Reset metadata for variant
        variant_data["metadata"]["creation_date"] = datetime.now()
        variant_data["metadata"]["last_updated"] = datetime.now()

        return Attack(**variant_data)

    def get_similarity_score(self, other: "Attack") -> float:
        """Calculate similarity score with another attack."""
        # Simple similarity based on common words and category
        self_words = set(self.content.lower().split())
        other_words = set(other.content.lower().split())

        # Jaccard similarity for content
        intersection = len(self_words & other_words)
        union = len(self_words | other_words)
        content_similarity = intersection / union if union > 0 else 0.0

        # Category similarity
        category_similarity = 1.0 if self.category == other.category else 0.0

        # Tag similarity
        tag_intersection = len(self.metadata.tags & other.metadata.tags)
        tag_union = len(self.metadata.tags | other.metadata.tags)
        tag_similarity = tag_intersection / tag_union if tag_union > 0 else 0.0

        # Weighted average
        return content_similarity * 0.6 + category_similarity * 0.2 + tag_similarity * 0.2


class AttackLibrarySchema(BaseModel):
    """Schema for attack library storage format."""

    version: str = Field(default="1.0", description="Schema version")
    schema: str = Field(default="attack-library-v1.0", description="Schema identifier")
    created: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    total_attacks: int = Field(default=0, description="Total number of attacks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Library metadata")
    attacks: List[Attack] = Field(default_factory=list, description="Attack collection")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "version": self.version,
            "schema": self.schema,
            "created": self.created.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "total_attacks": len(self.attacks),
            "metadata": self.metadata,
            "attacks": [attack.to_dict() for attack in self.attacks],
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttackLibrarySchema":
        """Create from dictionary."""
        # Parse datetime strings
        if isinstance(data.get("created"), str):
            data["created"] = datetime.fromisoformat(data["created"])
        if isinstance(data.get("last_updated"), str):
            data["last_updated"] = datetime.fromisoformat(data["last_updated"])

        # Parse attacks
        if "attacks" in data:
            data["attacks"] = [Attack.from_dict(attack_data) for attack_data in data["attacks"]]

        return cls(**data)
