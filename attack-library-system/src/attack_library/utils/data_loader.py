"""Data loading utilities for attack library."""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..core.models import Attack, AttackLibrarySchema
from .validators import AttackValidator

logger = logging.getLogger(__name__)


class DataLoader:
    """Utility class for loading attack data from various sources."""

    def __init__(self, validator: Optional[AttackValidator] = None):
        """
        Initialize data loader.

        Args:
            validator: Optional attack validator
        """
        self.validator = validator or AttackValidator()
        self.supported_formats = {"json", "csv", "yaml", "yml"}

    def load_from_file(self, file_path: Union[str, Path]) -> List[Attack]:
        """
        Load attacks from file.

        Args:
            file_path: Path to the file

        Returns:
            List of loaded attacks

        Raises:
            ValueError: If file format is unsupported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = file_path.suffix.lower().lstrip(".")

        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")

        try:
            if file_extension == "json":
                return self._load_json(file_path)
            elif file_extension == "csv":
                return self._load_csv(file_path)
            elif file_extension in ("yaml", "yml"):
                return self._load_yaml(file_path)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise

    def _load_json(self, file_path: Path) -> List[Attack]:
        """Load attacks from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different JSON formats
        if isinstance(data, list):
            # Direct list of attacks
            attacks = [Attack.from_dict(attack_data) for attack_data in data]
        elif isinstance(data, dict):
            if "attacks" in data:
                # Library schema format
                schema = AttackLibrarySchema.from_dict(data)
                attacks = schema.attacks
            else:
                # Single attack
                attacks = [Attack.from_dict(data)]
        else:
            raise ValueError("Invalid JSON format")

        # Validate attacks
        validated_attacks = []
        for attack in attacks:
            if self.validator.validate_attack(attack):
                validated_attacks.append(attack)
            else:
                logger.warning(f"Skipping invalid attack: {attack.id}")

        logger.info(f"Loaded {len(validated_attacks)} valid attacks from {file_path}")
        return validated_attacks

    def _load_csv(self, file_path: Path) -> List[Attack]:
        """Load attacks from CSV file."""
        attacks = []

        with open(file_path, "r", encoding="utf-8", newline="") as f:
            # Try to detect delimiter
            sample = f.read(1024)
            f.seek(0)

            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter

            reader = csv.DictReader(f, delimiter=delimiter)

            for row_num, row in enumerate(reader, start=2):  # Start at 2 for header
                try:
                    attack_data = self._process_csv_row(row)
                    attack = Attack.from_dict(attack_data)

                    if self.validator.validate_attack(attack):
                        attacks.append(attack)
                    else:
                        logger.warning(f"Invalid attack at row {row_num}: {attack.id}")

                except Exception as e:
                    logger.error(f"Error processing CSV row {row_num}: {e}")
                    continue

        logger.info(f"Loaded {len(attacks)} attacks from CSV {file_path}")
        return attacks

    def _process_csv_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        """Process a single CSV row into attack data."""
        # Required fields
        attack_data = {
            "id": row.get("id", "").strip(),
            "title": row.get("title", "").strip(),
            "content": row.get("content", "").strip(),
            "category": row.get("category", "").strip().lower(),
            "severity": row.get("severity", "").strip().lower(),
            "sophistication": int(row.get("sophistication", "1")),
        }

        # Optional fields
        target_models = row.get("target_models", "")
        if target_models:
            if target_models.startswith("[") and target_models.endswith("]"):
                # JSON-like format
                attack_data["target_models"] = json.loads(target_models)
            else:
                # Comma-separated
                attack_data["target_models"] = [
                    model.strip() for model in target_models.split(",") if model.strip()
                ]

        # Metadata
        metadata = {"source": row.get("source", "csv_import")}

        # Handle numeric fields
        for field in ["effectiveness_score", "success_rate"]:
            value = row.get(field, "").strip()
            if value and value.lower() != "null":
                try:
                    metadata[field] = float(value)
                except ValueError:
                    pass

        # Handle tags
        tags = row.get("tags", "")
        if tags:
            if tags.startswith("[") and tags.endswith("]"):
                metadata["tags"] = json.loads(tags)
            else:
                metadata["tags"] = [tag.strip() for tag in tags.split(",") if tag.strip()]

        # Handle other metadata fields
        for field in ["author", "language"]:
            value = row.get(field, "").strip()
            if value:
                metadata[field] = value

        attack_data["metadata"] = metadata

        return attack_data

    def _load_yaml(self, file_path: Path) -> List[Attack]:
        """Load attacks from YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML support. Install with: pip install pyyaml"
            )

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Handle different YAML formats
        if isinstance(data, list):
            attacks = [Attack.from_dict(attack_data) for attack_data in data]
        elif isinstance(data, dict):
            if "attacks" in data:
                attacks = [Attack.from_dict(attack_data) for attack_data in data["attacks"]]
            else:
                attacks = [Attack.from_dict(data)]
        else:
            raise ValueError("Invalid YAML format")

        # Validate attacks
        validated_attacks = []
        for attack in attacks:
            if self.validator.validate_attack(attack):
                validated_attacks.append(attack)
            else:
                logger.warning(f"Skipping invalid attack: {attack.id}")

        logger.info(f"Loaded {len(validated_attacks)} valid attacks from {file_path}")
        return validated_attacks

    def load_bulk_attacks(self, directory: Union[str, Path]) -> List[Attack]:
        """
        Load attacks from all supported files in a directory.

        Args:
            directory: Directory path

        Returns:
            List of all loaded attacks
        """
        directory = Path(directory)

        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Directory not found or not a directory: {directory}")

        all_attacks = []

        # Find all supported files
        for file_path in directory.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower().lstrip(".") in self.supported_formats
            ):
                try:
                    attacks = self.load_from_file(file_path)
                    all_attacks.extend(attacks)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
                    continue

        logger.info(f"Loaded {len(all_attacks)} total attacks from directory {directory}")
        return all_attacks

    def detect_duplicates(self, attacks: List[Attack]) -> List[tuple[Attack, Attack, float]]:
        """
        Detect potential duplicate attacks.

        Args:
            attacks: List of attacks to check

        Returns:
            List of (attack1, attack2, similarity_score) tuples for potential duplicates
        """
        duplicates = []
        threshold = 0.8  # Similarity threshold for considering duplicates

        for i, attack1 in enumerate(attacks):
            for attack2 in attacks[i + 1 :]:
                similarity = attack1.get_similarity_score(attack2)
                if similarity >= threshold:
                    duplicates.append((attack1, attack2, similarity))

        return duplicates

    def merge_datasets(self, *datasets: List[Attack]) -> List[Attack]:
        """
        Merge multiple attack datasets, handling duplicates.

        Args:
            *datasets: Variable number of attack datasets

        Returns:
            Merged list of unique attacks
        """
        all_attacks = []
        seen_ids = set()

        for dataset in datasets:
            for attack in dataset:
                if attack.id not in seen_ids:
                    all_attacks.append(attack)
                    seen_ids.add(attack.id)
                else:
                    logger.warning(f"Duplicate ID found, skipping: {attack.id}")

        # Check for content duplicates
        duplicates = self.detect_duplicates(all_attacks)
        if duplicates:
            logger.warning(f"Found {len(duplicates)} potential content duplicates")
            for attack1, attack2, similarity in duplicates:
                logger.warning(
                    f"Potential duplicate: {attack1.id} <-> {attack2.id} (similarity: {similarity:.2f})"
                )

        return all_attacks

    def create_sample_dataset(self, count_per_category: int = 5) -> List[Attack]:
        """
        Create a sample dataset for testing.

        Args:
            count_per_category: Number of attacks per category

        Returns:
            List of sample attacks
        """
        import random

        from ..core.models import AttackCategory, AttackSeverity

        sample_attacks = []

        categories = list(AttackCategory)
        severities = list(AttackSeverity)

        for i, category in enumerate(categories):
            for j in range(count_per_category):
                attack_id = f"{category.value}_{j + 1:03d}"

                attack = Attack(
                    id=attack_id,
                    title=f"Sample {category.value.title()} Attack #{j + 1}",
                    content=f"This is a sample {category.value} attack for testing purposes.",
                    category=category,
                    severity=random.choice(severities),
                    sophistication=random.randint(1, 5),
                )

                attack.metadata.source = "sample_generator"
                attack.metadata.tags.add("sample")
                attack.metadata.tags.add(category.value)

                sample_attacks.append(attack)

        logger.info(f"Created {len(sample_attacks)} sample attacks")
        return sample_attacks
