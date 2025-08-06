"""Data validation and quality assurance for synthetic data."""

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]


@dataclass
class QualityResult:
    """Result of quality assessment."""

    score: float
    metrics: Dict[str, float]
    recommendations: List[str]


class DataValidator:
    """Validator for synthetic data quality and consistency."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data validator.

        Args:
            config: Validation configuration
        """
        self.config = config or {}
        self.min_quality_score = self.config.get("min_quality_score", 0.8)

    def validate_batch(self, data: List[Dict[str, Any]]) -> ValidationResult:
        """
        Validate a batch of generated data.

        Args:
            data: Data to validate

        Returns:
            Validation result
        """
        errors = []
        warnings = []
        metrics = {}

        if not data:
            errors.append("Empty data batch")
            return ValidationResult(False, errors, warnings, metrics)

        # Check data structure consistency
        structure_valid, structure_errors = self._check_structure_consistency(data)
        if not structure_valid:
            errors.extend(structure_errors)

        # Check for duplicates
        duplicate_count = self._check_duplicates(data)
        if duplicate_count > 0:
            warnings.append(f"Found {duplicate_count} duplicate records")
        metrics["duplicate_count"] = duplicate_count

        # Check data diversity
        diversity_score = self._calculate_diversity(data)
        metrics["diversity_score"] = diversity_score
        if diversity_score < 0.5:
            warnings.append(f"Low data diversity: {diversity_score:.2f}")

        # Check field completeness
        completeness = self._check_completeness(data)
        metrics["completeness"] = completeness
        if completeness < 0.9:
            warnings.append(f"Data completeness below threshold: {completeness:.2f}")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, metrics)

    def validate_quality(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the overall quality of generated data.

        Args:
            data: Data to validate

        Returns:
            Quality validation results
        """
        results = {
            "total_records": len(data),
            "unique_records": len(self._get_unique_records(data)),
            "diversity_score": self._calculate_diversity(data),
            "completeness_score": self._check_completeness(data),
            "consistency_score": self._check_consistency(data),
            "format_validity": self._check_format_validity(data),
        }

        # Calculate overall quality score
        scores = [
            results["diversity_score"],
            results["completeness_score"],
            results["consistency_score"],
            results["format_validity"],
        ]
        results["overall_score"] = np.mean(scores)
        results["passed"] = results["overall_score"] >= self.min_quality_score

        return results

    def calculate_quality_score(self, data: List[Dict[str, Any]]) -> QualityResult:
        """
        Calculate comprehensive quality score for generated data.

        Args:
            data: Data to assess

        Returns:
            Quality assessment result
        """
        metrics = {}
        recommendations = []

        # Calculate individual metrics
        metrics["diversity"] = self._calculate_diversity(data)
        metrics["completeness"] = self._check_completeness(data)
        metrics["consistency"] = self._check_consistency(data)
        metrics["uniqueness"] = len(self._get_unique_records(data)) / len(data) if data else 0
        metrics["format_validity"] = self._check_format_validity(data)

        # Calculate weighted score
        weights = {
            "diversity": 0.25,
            "completeness": 0.25,
            "consistency": 0.20,
            "uniqueness": 0.15,
            "format_validity": 0.15,
        }

        score = sum(metrics[key] * weight for key, weight in weights.items())

        # Generate recommendations
        if metrics["diversity"] < 0.6:
            recommendations.append("Increase data diversity by varying generation parameters")
        if metrics["completeness"] < 0.9:
            recommendations.append("Ensure all required fields are populated")
        if metrics["consistency"] < 0.8:
            recommendations.append("Improve data consistency across records")
        if metrics["uniqueness"] < 0.95:
            recommendations.append("Reduce duplicate records in generation")

        return QualityResult(score, metrics, recommendations)

    def _check_structure_consistency(self, data: List[Dict[str, Any]]) -> tuple:
        """Check if all records have consistent structure."""
        errors = []

        if not data:
            return True, errors

        # Get reference keys from first record
        reference_keys = set(data[0].keys())

        for i, record in enumerate(data[1:], 1):
            record_keys = set(record.keys())
            if record_keys != reference_keys:
                missing = reference_keys - record_keys
                extra = record_keys - reference_keys
                error_msg = f"Record {i} has inconsistent structure"
                if missing:
                    error_msg += f" (missing: {missing})"
                if extra:
                    error_msg += f" (extra: {extra})"
                errors.append(error_msg)

        return len(errors) == 0, errors

    def _check_duplicates(self, data: List[Dict[str, Any]]) -> int:
        """Count duplicate records in data."""
        if not data:
            return 0

        # Convert records to hashable format
        hashable_records = []
        for record in data:
            hashable_record = tuple(sorted(record.items()))
            hashable_records.append(hashable_record)

        unique_count = len(set(hashable_records))
        duplicate_count = len(data) - unique_count

        return duplicate_count

    def _calculate_diversity(self, data: List[Dict[str, Any]]) -> float:
        """Calculate diversity score for data."""
        if not data or len(data) < 2:
            return 1.0

        # Extract text fields for diversity calculation
        text_fields = []
        for record in data:
            text = " ".join(str(v) for v in record.values() if v)
            text_fields.append(text)

        if len(text_fields) < 2:
            return 1.0

        try:
            # Calculate TF-IDF vectors
            vectorizer = TfidfVectorizer(max_features=100)
            tfidf_matrix = vectorizer.fit_transform(text_fields)

            # Calculate pairwise similarities
            similarities = cosine_similarity(tfidf_matrix)

            # Diversity is inverse of average similarity
            avg_similarity = (similarities.sum() - len(data)) / (len(data) * (len(data) - 1))
            diversity = 1.0 - avg_similarity

            return max(0.0, min(1.0, diversity))

        except Exception as e:
            logger.warning(f"Error calculating diversity: {str(e)}")
            return 0.5

    def _check_completeness(self, data: List[Dict[str, Any]]) -> float:
        """Check data completeness (non-null fields)."""
        if not data:
            return 0.0

        total_fields = 0
        non_null_fields = 0

        for record in data:
            for value in record.values():
                total_fields += 1
                if value is not None and value != "":
                    non_null_fields += 1

        if total_fields == 0:
            return 0.0

        return non_null_fields / total_fields

    def _check_consistency(self, data: List[Dict[str, Any]]) -> float:
        """Check data consistency across records."""
        if not data or len(data) < 2:
            return 1.0

        consistency_scores = []

        # Check type consistency for each field
        for field in data[0].keys():
            field_types = []
            for record in data:
                if field in record and record[field] is not None:
                    field_types.append(type(record[field]).__name__)

            if field_types:
                type_counts = Counter(field_types)
                most_common_type_count = type_counts.most_common(1)[0][1]
                consistency = most_common_type_count / len(field_types)
                consistency_scores.append(consistency)

        if not consistency_scores:
            return 1.0

        return np.mean(consistency_scores)

    def _check_format_validity(self, data: List[Dict[str, Any]]) -> float:
        """Check if data follows expected formats."""
        if not data:
            return 0.0

        valid_count = 0
        total_count = 0

        # Common format patterns
        email_pattern = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")
        phone_pattern = re.compile(r"^\+?\d{10,15}$")
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}")

        for record in data:
            for key, value in record.items():
                if value is None or value == "":
                    continue

                total_count += 1
                valid = True

                # Check common field formats
                if "email" in key.lower() and isinstance(value, str):
                    valid = bool(email_pattern.match(value))
                elif "phone" in key.lower() and isinstance(value, str):
                    valid = bool(phone_pattern.match(value.replace(" ", "").replace("-", "")))
                elif "date" in key.lower() and isinstance(value, str):
                    valid = bool(date_pattern.match(value))

                if valid:
                    valid_count += 1

        if total_count == 0:
            return 1.0

        return valid_count / total_count

    def _get_unique_records(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get unique records from data."""
        if not data:
            return []

        seen = set()
        unique = []

        for record in data:
            hashable_record = tuple(sorted(record.items()))
            if hashable_record not in seen:
                seen.add(hashable_record)
                unique.append(record)

        return unique
