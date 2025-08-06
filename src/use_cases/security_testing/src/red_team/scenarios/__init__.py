"""
Domain-specific attack scenario templates.
"""

from .base import BaseScenario, ScenarioBuilder
from .customer_service import CustomerServiceScenarios
from .financial import FinancialScenarios
from .healthcare import HealthcareScenarios
from .registry import ScenarioRegistry

__all__ = [
    "BaseScenario",
    "ScenarioBuilder",
    "ScenarioRegistry",
    "HealthcareScenarios",
    "FinancialScenarios",
    "CustomerServiceScenarios",
]
