"""Base attack generator class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..core.models import Attack


class AttackGenerator(ABC):
    """Base class for attack generators."""

    def __init__(self, **kwargs):
        """Initialize generator with configuration."""
        self.config = kwargs

    @abstractmethod
    def generate(self, count: int = 1, **kwargs) -> List[Attack]:
        """
        Generate new attacks.

        Args:
            count: Number of attacks to generate
            **kwargs: Additional generation parameters

        Returns:
            List of generated attacks
        """
        pass

    @abstractmethod
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
        pass
