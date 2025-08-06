"""Validation utilities for attack library."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..core.models import Attack, AttackCategory, AttackSeverity

logger = logging.getLogger(__name__)


class AttackValidator:
    """Validator for attack objects and data."""

    def __init__(self, strict: bool = False):
        """
        Initialize validator.

        Args:
            strict: If True, apply stricter validation rules
        """
        self.strict = strict
        self.schema_validator = SchemaValidator()

        # Validation statistics
        self.validation_stats = {"total_validated": 0, "valid": 0, "invalid": 0, "errors": []}

    def validate_attack(self, attack: Attack) -> bool:
        """
        Validate an attack object.

        Args:
            attack: Attack to validate

        Returns:
            True if valid, False otherwise
        """
        self.validation_stats["total_validated"] += 1

        try:
            # Basic field validation
            if not self._validate_basic_fields(attack):
                return False

            # Content validation
            if not self._validate_content(attack):
                return False

            # Category and severity validation
            if not self._validate_enums(attack):
                return False

            # Sophistication validation
            if not self._validate_sophistication(attack):
                return False

            # Target models validation
            if not self._validate_target_models(attack):
                return False

            # Metadata validation
            if not self._validate_metadata(attack):
                return False

            # Strict mode validations
            if self.strict and not self._validate_strict_rules(attack):
                return False

            self.validation_stats["valid"] += 1
            return True

        except Exception as e:
            error_msg = f"Validation error for attack {attack.id}: {e}"
            logger.error(error_msg)
            self.validation_stats["errors"].append(error_msg)
            self.validation_stats["invalid"] += 1
            return False

    def _validate_basic_fields(self, attack: Attack) -> bool:
        """Validate basic required fields."""
        # ID validation
        if not attack.id or len(attack.id) < 3:
            logger.error(f"Invalid ID: {attack.id}")
            return False

        if not re.match(r"^[a-zA-Z0-9_-]+$", attack.id):
            logger.error(f"Invalid ID format: {attack.id}")
            return False

        # Title validation
        if not attack.title or len(attack.title.strip()) == 0:
            logger.error(f"Invalid title for attack {attack.id}")
            return False

        if len(attack.title) > 200:
            logger.error(f"Title too long for attack {attack.id}")
            return False

        return True

    def _validate_content(self, attack: Attack) -> bool:
        """Validate attack content."""
        if not attack.content or len(attack.content.strip()) == 0:
            logger.error(f"Empty content for attack {attack.id}")
            return False

        # Content length check
        if len(attack.content) > 10000:
            logger.error(f"Content too long for attack {attack.id}")
            return False

        # Check for obvious test/placeholder content in strict mode
        if self.strict:
            placeholder_patterns = [
                r"lorem ipsum",
                r"test content",
                r"placeholder",
                r"sample text",
                r"dummy data",
            ]

            content_lower = attack.content.lower()
            for pattern in placeholder_patterns:
                if re.search(pattern, content_lower):
                    logger.warning(f"Placeholder content detected in attack {attack.id}")
                    if self.strict:
                        return False

        return True

    def _validate_enums(self, attack: Attack) -> bool:
        """Validate enum values."""
        # Category validation
        if attack.category not in AttackCategory:
            logger.error(f"Invalid category {attack.category} for attack {attack.id}")
            return False

        # Severity validation
        if attack.severity not in AttackSeverity:
            logger.error(f"Invalid severity {attack.severity} for attack {attack.id}")
            return False

        return True

    def _validate_sophistication(self, attack: Attack) -> bool:
        """Validate sophistication level."""
        if not isinstance(attack.sophistication, int):
            logger.error(f"Sophistication must be integer for attack {attack.id}")
            return False

        if not (1 <= attack.sophistication <= 5):
            logger.error(f"Sophistication must be 1-5 for attack {attack.id}")
            return False

        return True

    def _validate_target_models(self, attack: Attack) -> bool:
        """Validate target models list."""
        if not isinstance(attack.target_models, list):
            logger.error(f"Target models must be list for attack {attack.id}")
            return False

        # Check for valid model names
        valid_model_patterns = [
            r"gpt-\d+(\.\d+)?",
            r"claude(-\d+)?",
            r"bard",
            r"llama(-\d+)?",
            r"codex",
            r"text-davinci-\d+",
            r"palm(-\d+)?",
            r"mixtral(-\d+x\d+b)?",
            r"gemini(-pro|-ultra)?",
            r"local",
            r"custom",
            r"azure-.*",
        ]

        for model in attack.target_models:
            if not isinstance(model, str) or not model.strip():
                logger.error(f"Invalid model name '{model}' for attack {attack.id}")
                return False

            # In strict mode, validate against known model patterns
            if self.strict:
                model_lower = model.lower().strip()
                if not any(re.match(pattern, model_lower) for pattern in valid_model_patterns):
                    logger.warning(f"Unknown model '{model}' for attack {attack.id}")

        return True

    def _validate_metadata(self, attack: Attack) -> bool:
        """Validate metadata fields."""
        metadata = attack.metadata

        # Source is required
        if not metadata.source or not metadata.source.strip():
            logger.error(f"Missing source for attack {attack.id}")
            return False

        # Effectiveness score validation
        if metadata.effectiveness_score is not None:
            if not isinstance(metadata.effectiveness_score, (int, float)):
                logger.error(f"Invalid effectiveness score type for attack {attack.id}")
                return False

            if not (0.0 <= metadata.effectiveness_score <= 1.0):
                logger.error(f"Effectiveness score must be 0.0-1.0 for attack {attack.id}")
                return False

        # Success rate validation
        if metadata.success_rate is not None:
            if not isinstance(metadata.success_rate, (int, float)):
                logger.error(f"Invalid success rate type for attack {attack.id}")
                return False

            if not (0.0 <= metadata.success_rate <= 1.0):
                logger.error(f"Success rate must be 0.0-1.0 for attack {attack.id}")
                return False

        # Tags validation
        if not isinstance(metadata.tags, set):
            logger.error(f"Tags must be set for attack {attack.id}")
            return False

        for tag in metadata.tags:
            if not isinstance(tag, str) or not tag.strip():
                logger.error(f"Invalid tag '{tag}' for attack {attack.id}")
                return False

            if len(tag) > 50:
                logger.error(f"Tag too long '{tag}' for attack {attack.id}")
                return False

        # Language code validation
        if not re.match(r"^[a-z]{2}(-[A-Z]{2})?$", metadata.language):
            logger.error(f"Invalid language code '{metadata.language}' for attack {attack.id}")
            return False

        return True

    def _validate_strict_rules(self, attack: Attack) -> bool:
        """Apply strict validation rules."""
        # Check for minimum content quality
        if len(attack.content.split()) < 5:
            logger.error(f"Content too short (strict mode) for attack {attack.id}")
            return False

        # Require effectiveness score for verified attacks
        if attack.is_verified and attack.metadata.effectiveness_score is None:
            logger.error(f"Verified attack missing effectiveness score: {attack.id}")
            return False

        # Require at least one target model
        if len(attack.target_models) == 0:
            logger.error(f"No target models specified (strict mode) for attack {attack.id}")
            return False

        # Require at least one tag
        if len(attack.metadata.tags) == 0:
            logger.error(f"No tags specified (strict mode) for attack {attack.id}")
            return False

        return True

    def validate_attack_list(self, attacks: List[Attack]) -> Dict[str, Any]:
        """
        Validate a list of attacks.

        Args:
            attacks: List of attacks to validate

        Returns:
            Validation report
        """
        # Reset stats
        self.validation_stats = {"total_validated": 0, "valid": 0, "invalid": 0, "errors": []}

        valid_attacks = []
        invalid_attacks = []
        duplicate_ids = set()
        seen_ids = set()

        for attack in attacks:
            # Check for duplicate IDs
            if attack.id in seen_ids:
                duplicate_ids.add(attack.id)
                logger.error(f"Duplicate attack ID: {attack.id}")
                invalid_attacks.append(attack)
                continue

            seen_ids.add(attack.id)

            # Validate individual attack
            if self.validate_attack(attack):
                valid_attacks.append(attack)
            else:
                invalid_attacks.append(attack)

        # Generate report
        report = {
            "total_attacks": len(attacks),
            "valid_attacks": len(valid_attacks),
            "invalid_attacks": len(invalid_attacks),
            "duplicate_ids": list(duplicate_ids),
            "validation_rate": len(valid_attacks) / len(attacks) if attacks else 0,
            "errors": self.validation_stats["errors"],
        }

        # Category distribution for valid attacks
        category_counts = {}
        severity_counts = {}

        for attack in valid_attacks:
            category_counts[attack.category.value] = (
                category_counts.get(attack.category.value, 0) + 1
            )
            severity_counts[attack.severity.value] = (
                severity_counts.get(attack.severity.value, 0) + 1
            )

        report["category_distribution"] = category_counts
        report["severity_distribution"] = severity_counts

        return report

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get current validation statistics."""
        return self.validation_stats.copy()


class SchemaValidator:
    """JSON schema validator for attack library data."""

    def __init__(self, schema_path: Optional[Path] = None):
        """
        Initialize schema validator.

        Args:
            schema_path: Path to JSON schema file
        """
        if schema_path is None:
            # Use bundled schema
            schema_path = Path(__file__).parent.parent / "data" / "schemas" / "attack_schema.json"

        self.schema_path = schema_path
        self.schema = self._load_schema()

    def _load_schema(self) -> Dict[str, Any]:
        """Load JSON schema."""
        try:
            with open(self.schema_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load schema from {self.schema_path}: {e}")
            return {}

    def validate_attack_dict(self, attack_data: Dict[str, Any]) -> bool:
        """
        Validate attack dictionary against schema.

        Args:
            attack_data: Attack data dictionary

        Returns:
            True if valid, False otherwise
        """
        if not self.schema:
            logger.warning("No schema loaded, skipping schema validation")
            return True

        try:
            # Basic schema validation without jsonschema
            required_fields = ["id", "title", "content", "category", "severity", "sophistication"]
            for field in required_fields:
                if field not in attack_data:
                    logger.error(f"Missing required field: {field}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return False

    def validate_library_dict(self, library_data: Dict[str, Any]) -> bool:
        """
        Validate library dictionary against schema.

        Args:
            library_data: Library data dictionary

        Returns:
            True if valid, False otherwise
        """
        if not self.schema:
            logger.warning("No schema loaded, skipping schema validation")
            return True

        try:
            # Basic library validation without jsonschema
            required_fields = ["attacks", "metadata"]
            for field in required_fields:
                if field not in library_data:
                    logger.warning(f"Missing library field: {field}")
            return True
        except Exception as e:
            logger.error(f"Library schema validation error: {e}")
            return False
