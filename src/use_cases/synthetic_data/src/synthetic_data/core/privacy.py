"""Privacy preservation for synthetic data."""

import hashlib
import logging
import random
import re
import string
from typing import Any, Dict, List, Optional, Set

from faker import Faker

logger = logging.getLogger(__name__)
fake = Faker()


class PrivacyPreserver:
    """Privacy preservation for synthetic data."""

    # PII patterns
    PII_PATTERNS = {
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "credit_card": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "license_plate": re.compile(r"\b[A-Z]{1,3}[-\s]?\d{1,4}[-\s]?[A-Z]{0,3}\b"),
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the privacy preserver.

        Args:
            config: Privacy configuration
        """
        self.config = config or {}
        self.anonymization_level = self.config.get("anonymization_level", "medium")
        self.pii_detection = self.config.get("pii_detection", True)
        self.pii_removal = self.config.get("pii_removal", True)
        self._anonymization_cache = {}

    def apply_privacy(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply privacy preservation to data.

        Args:
            data: Data to process

        Returns:
            Privacy-preserved data
        """
        if not data:
            return data

        processed_data = []

        for record in data:
            processed_record = self._process_record(record)
            processed_data.append(processed_record)

        # Apply differential privacy if configured
        if self.config.get("differential_privacy", False):
            processed_data = self._apply_differential_privacy(processed_data)

        return processed_data

    def _process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single record for privacy."""
        processed = {}

        for key, value in record.items():
            if value is None:
                processed[key] = value
                continue

            # Detect and handle PII
            if self.pii_detection and isinstance(value, str):
                value = self._handle_pii(value, key)

            # Apply field-specific anonymization
            if self._should_anonymize(key):
                value = self._anonymize_value(value, key)

            processed[key] = value

        return processed

    def _handle_pii(self, text: str, field_name: str) -> str:
        """Detect and handle PII in text."""
        if not self.pii_removal:
            return text

        processed_text = text

        for pii_type, pattern in self.PII_PATTERNS.items():
            if pattern.search(processed_text):
                if self.anonymization_level == "high":
                    # Complete removal
                    processed_text = pattern.sub(f"[{pii_type.upper()}_REMOVED]", processed_text)
                elif self.anonymization_level == "medium":
                    # Partial masking
                    processed_text = self._mask_pii(processed_text, pattern, pii_type)
                else:  # low
                    # Hash-based replacement
                    processed_text = self._hash_pii(processed_text, pattern, pii_type)

        return processed_text

    def _mask_pii(self, text: str, pattern: re.Pattern, pii_type: str) -> str:
        """Mask PII with partial visibility."""

        def mask_match(match):
            value = match.group()
            if pii_type == "email":
                parts = value.split("@")
                if len(parts) == 2:
                    masked_user = parts[0][:2] + "*" * (len(parts[0]) - 2)
                    return f"{masked_user}@{parts[1]}"
            elif pii_type == "phone":
                # Keep area code, mask rest
                digits = re.sub(r"\D", "", value)
                if len(digits) >= 10:
                    return f"{digits[:3]}-***-****"
            elif pii_type == "ssn":
                # Mask all but last 4
                return "***-**-" + value[-4:]
            elif pii_type == "credit_card":
                # Keep first and last 4
                digits = re.sub(r"\D", "", value)
                if len(digits) >= 16:
                    return f"{digits[:4]} **** **** {digits[-4:]}"

            # Default: mask middle portion
            length = len(value)
            if length > 4:
                visible = max(1, length // 4)
                return value[:visible] + "*" * (length - 2 * visible) + value[-visible:]
            return "*" * length

        return pattern.sub(mask_match, text)

    def _hash_pii(self, text: str, pattern: re.Pattern, pii_type: str) -> str:
        """Replace PII with consistent hash."""

        def hash_match(match):
            value = match.group()
            # Use cached hash if available
            if value in self._anonymization_cache:
                return self._anonymization_cache[value]

            # Generate consistent hash
            hash_value = hashlib.md5(value.encode()).hexdigest()[:8]
            replacement = f"{pii_type}_{hash_value}"

            # Cache for consistency
            self._anonymization_cache[value] = replacement
            return replacement

        return pattern.sub(hash_match, text)

    def _should_anonymize(self, field_name: str) -> bool:
        """Check if field should be anonymized."""
        sensitive_fields = {
            "name",
            "first_name",
            "last_name",
            "full_name",
            "address",
            "street",
            "city",
            "zip",
            "postal_code",
            "email",
            "phone",
            "mobile",
            "telephone",
            "ssn",
            "social_security",
            "tax_id",
            "credit_card",
            "card_number",
            "account_number",
            "date_of_birth",
            "dob",
            "birthdate",
            "salary",
            "income",
            "wage",
            "medical_record",
            "patient_id",
            "diagnosis",
        }

        field_lower = field_name.lower()
        return any(sensitive in field_lower for sensitive in sensitive_fields)

    def _anonymize_value(self, value: Any, field_name: str) -> Any:
        """Anonymize a value based on field type."""
        if value is None:
            return value

        field_lower = field_name.lower()

        # Name fields
        if any(n in field_lower for n in ["name", "first_name", "last_name"]):
            if "first" in field_lower:
                return fake.first_name()
            elif "last" in field_lower:
                return fake.last_name()
            else:
                return fake.name()

        # Address fields
        elif "address" in field_lower or "street" in field_lower:
            return fake.street_address()
        elif "city" in field_lower:
            return fake.city()
        elif "zip" in field_lower or "postal" in field_lower:
            return fake.postcode()

        # Contact fields
        elif "email" in field_lower:
            return fake.email()
        elif "phone" in field_lower or "mobile" in field_lower:
            return fake.phone_number()

        # Financial fields
        elif "credit_card" in field_lower or "card_number" in field_lower:
            return fake.credit_card_number()
        elif "account" in field_lower:
            return fake.bban()

        # Date fields
        elif "birth" in field_lower or "dob" in field_lower:
            return fake.date_of_birth().isoformat()

        # Numeric fields
        elif isinstance(value, (int, float)):
            if "salary" in field_lower or "income" in field_lower:
                # Add noise to numeric values
                noise = random.uniform(0.9, 1.1)
                return type(value)(value * noise)
            elif "age" in field_lower:
                # Add small random offset
                offset = random.randint(-2, 2)
                return max(0, value + offset)

        # Default: return generic anonymized value
        if isinstance(value, str):
            return f"ANONYMIZED_{field_name.upper()}_{random.randint(1000, 9999)}"

        return value

    def _apply_differential_privacy(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply differential privacy to numerical fields."""
        epsilon = self.config.get("differential_privacy_epsilon", 1.0)

        for record in data:
            for key, value in record.items():
                if isinstance(value, (int, float)):
                    # Add Laplace noise
                    sensitivity = abs(value) * 0.1 if value != 0 else 1.0
                    noise = random.random() * (sensitivity / epsilon)
                    record[key] = type(value)(value + noise)

        return data

    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """
        Detect PII in text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary of detected PII by type
        """
        detected = {}

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                detected[pii_type] = matches

        return detected

    def validate_privacy(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate privacy preservation in data.

        Args:
            data: Data to validate

        Returns:
            Privacy validation results
        """
        results = {
            "pii_found": False,
            "pii_details": {},
            "anonymization_coverage": 0.0,
            "recommendations": [],
        }

        total_fields = 0
        anonymized_fields = 0
        pii_instances = {}

        for record in data:
            for key, value in record.items():
                total_fields += 1

                if isinstance(value, str):
                    # Check for PII
                    detected = self.detect_pii(value)
                    if detected:
                        results["pii_found"] = True
                        for pii_type, instances in detected.items():
                            if pii_type not in pii_instances:
                                pii_instances[pii_type] = []
                            pii_instances[pii_type].extend(instances)

                    # Check if field appears anonymized
                    if any(marker in value for marker in ["ANONYMIZED", "REMOVED", "****", "xxx"]):
                        anonymized_fields += 1

                # Check sensitive fields
                if self._should_anonymize(key):
                    if value and not self._appears_anonymized(value):
                        results["recommendations"].append(f"Consider anonymizing field: {key}")

        results["pii_details"] = pii_instances
        results["anonymization_coverage"] = (
            anonymized_fields / total_fields if total_fields > 0 else 0.0
        )

        if results["pii_found"]:
            results["recommendations"].append(
                "PII detected in data - apply stronger privacy preservation"
            )

        return results

    def _appears_anonymized(self, value: Any) -> bool:
        """Check if a value appears to be anonymized."""
        if not isinstance(value, str):
            return False

        anonymization_markers = [
            "ANONYMIZED",
            "REMOVED",
            "MASKED",
            "REDACTED",
            "****",
            "XXXX",
            "###",
            "[HIDDEN]",
        ]

        return any(marker in value.upper() for marker in anonymization_markers)
