"""Variant generator for creating attack variations."""

from typing import List

from ..core.models import Attack
from .base import AttackGenerator


class VariantGenerator(AttackGenerator):
    """Generator for creating attack variants."""

    def generate(self, count: int = 1, **kwargs) -> List[Attack]:
        """Generate new attacks (not implemented for variant generator)."""
        raise NotImplementedError(
            "VariantGenerator only supports generating variants from existing attacks"
        )

    def generate_variants(self, base_attack: Attack, count: int = 1, **kwargs) -> List[Attack]:
        """
        Generate variants of an existing attack.

        Args:
            base_attack: Attack to create variants from
            count: Number of variants to generate
            **kwargs: Additional generation parameters

        Returns:
            List of attack variants
        """
        variants = []

        for i in range(count):
            # Simple variant generation - just modify the content slightly
            variant_content = f"{base_attack.content} [Variant {i + 1}]"

            variant = base_attack.create_variant(
                new_content=variant_content, variant_type="simple_modification"
            )

            variants.append(variant)

        return variants
