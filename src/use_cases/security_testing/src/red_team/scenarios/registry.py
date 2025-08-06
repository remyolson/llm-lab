"""
Scenario registry for managing attack scenarios across domains.
"""

import logging
from typing import Dict, List, Optional, Type

from ..core.models import AttackScenario, DifficultyLevel
from .base import BaseScenario
from .customer_service import CustomerServiceScenarios
from .financial import FinancialScenarios
from .healthcare import HealthcareScenarios

logger = logging.getLogger(__name__)


class ScenarioRegistry:
    """Central registry for managing attack scenarios across all domains."""

    _instance: Optional["ScenarioRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "ScenarioRegistry":
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the registry with available scenario providers."""
        if self._initialized:
            return

        self._scenario_providers: Dict[str, Type[BaseScenario]] = {}
        self._scenario_cache: Dict[str, AttackScenario] = {}
        self._difficulty_levels: Dict[str, DifficultyLevel] = {
            "basic": DifficultyLevel.BASIC,
            "intermediate": DifficultyLevel.INTERMEDIATE,
            "advanced": DifficultyLevel.ADVANCED,
            "expert": DifficultyLevel.EXPERT,
        }

        # Register built-in scenario providers
        self._register_builtin_providers()
        self._initialized = True

        logger.info(f"ScenarioRegistry initialized with {len(self._scenario_providers)} providers")

    def _register_builtin_providers(self):
        """Register built-in scenario providers."""
        self._scenario_providers.update(
            {
                "healthcare": HealthcareScenarios,
                "financial": FinancialScenarios,
                "customer_service": CustomerServiceScenarios,
            }
        )

    @classmethod
    def get_instance(cls) -> "ScenarioRegistry":
        """Get singleton instance of the registry."""
        return cls()

    def register_provider(self, domain: str, provider_class: Type[BaseScenario]):
        """Register a new scenario provider for a domain."""
        self._scenario_providers[domain] = provider_class
        logger.info(f"Registered scenario provider for domain: {domain}")

    def get_available_domains(self) -> List[str]:
        """Get list of available scenario domains."""
        return list(self._scenario_providers.keys())

    def get_available_scenarios(
        self, domain: str, difficulty_level: str = "intermediate"
    ) -> List[str]:
        """Get list of available scenarios for a domain."""
        if domain not in self._scenario_providers:
            return []

        difficulty = self._difficulty_levels.get(difficulty_level, DifficultyLevel.INTERMEDIATE)
        provider = self._scenario_providers[domain](difficulty)
        return provider.get_available_scenarios()

    def get_scenario(
        self,
        scenario_identifier: str,
        difficulty_level: str = "intermediate",
        use_cache: bool = True,
    ) -> Optional[AttackScenario]:
        """
        Get a scenario by identifier.

        Args:
            scenario_identifier: Either "domain.scenario_name" or just "scenario_name"
            difficulty_level: Difficulty level (basic, intermediate, advanced, expert)
            use_cache: Whether to use cached scenarios

        Returns:
            AttackScenario instance or None if not found
        """
        cache_key = f"{scenario_identifier}_{difficulty_level}"

        # Check cache first
        if use_cache and cache_key in self._scenario_cache:
            return self._scenario_cache[cache_key]

        # Parse scenario identifier
        if "." in scenario_identifier:
            domain, scenario_name = scenario_identifier.split(".", 1)
        else:
            # Try to find scenario across all domains
            domain, scenario_name = self._find_scenario_domain(scenario_identifier)
            if not domain:
                logger.warning(f"Scenario not found: {scenario_identifier}")
                return None

        # Get scenario from provider
        if domain not in self._scenario_providers:
            logger.error(f"Unknown domain: {domain}")
            return None

        difficulty = self._difficulty_levels.get(difficulty_level, DifficultyLevel.INTERMEDIATE)

        try:
            provider = self._scenario_providers[domain](difficulty)
            scenario = provider.create_scenario(scenario_name)

            # Cache the scenario
            if use_cache:
                self._scenario_cache[cache_key] = scenario

            logger.info(f"Created scenario: {domain}.{scenario_name} ({difficulty_level})")
            return scenario

        except Exception as e:
            logger.error(f"Failed to create scenario {domain}.{scenario_name}: {e}")
            return None

    def _find_scenario_domain(self, scenario_name: str) -> tuple[Optional[str], str]:
        """Find which domain contains a scenario by searching all providers."""
        for domain, provider_class in self._scenario_providers.items():
            try:
                provider = provider_class(DifficultyLevel.INTERMEDIATE)
                available_scenarios = provider.get_available_scenarios()
                if scenario_name in available_scenarios:
                    return domain, scenario_name
            except Exception as e:
                logger.warning(f"Error checking provider {domain}: {e}")

        return None, scenario_name

    def create_scenario_variant(
        self,
        base_scenario: str,
        difficulty_level: str = "intermediate",
        customizations: Optional[Dict] = None,
    ) -> Optional[AttackScenario]:
        """Create a customized variant of an existing scenario."""
        scenario = self.get_scenario(base_scenario, difficulty_level, use_cache=False)
        if not scenario:
            return None

        if customizations:
            # Apply customizations
            if "success_threshold" in customizations:
                scenario.success_threshold = customizations["success_threshold"]

            if "max_duration_minutes" in customizations:
                scenario.max_duration_minutes = customizations["max_duration_minutes"]

            if "tags" in customizations:
                scenario.tags.extend(customizations["tags"])

            if "initial_context" in customizations:
                scenario.initial_context.update(customizations["initial_context"])

        return scenario

    def get_scenarios_by_tag(
        self, tag: str, difficulty_level: str = "intermediate"
    ) -> List[AttackScenario]:
        """Get all scenarios that contain a specific tag."""
        scenarios = []

        for domain in self.get_available_domains():
            available_scenarios = self.get_available_scenarios(domain, difficulty_level)
            for scenario_name in available_scenarios:
                scenario = self.get_scenario(f"{domain}.{scenario_name}", difficulty_level)
                if scenario and tag in scenario.tags:
                    scenarios.append(scenario)

        return scenarios

    def get_scenarios_by_compliance_framework(
        self, framework: str, difficulty_level: str = "intermediate"
    ) -> List[AttackScenario]:
        """Get all scenarios that test a specific compliance framework."""
        scenarios = []

        for domain in self.get_available_domains():
            available_scenarios = self.get_available_scenarios(domain, difficulty_level)
            for scenario_name in available_scenarios:
                scenario = self.get_scenario(f"{domain}.{scenario_name}", difficulty_level)
                if scenario and framework in scenario.compliance_frameworks:
                    scenarios.append(scenario)

        return scenarios

    def get_scenario_metadata(self, scenario_identifier: str) -> Optional[Dict]:
        """Get metadata for a scenario without creating the full instance."""
        scenario = self.get_scenario(scenario_identifier, use_cache=False)
        if not scenario:
            return None

        return {
            "scenario_id": scenario.scenario_id,
            "name": scenario.name,
            "description": scenario.description,
            "domain": scenario.domain,
            "difficulty_level": scenario.difficulty_level.value,
            "num_chains": len(scenario.attack_chains),
            "total_steps": len(scenario.get_attack_steps()),
            "tags": scenario.tags,
            "compliance_frameworks": scenario.compliance_frameworks,
            "success_threshold": scenario.success_threshold,
            "max_duration_minutes": scenario.max_duration_minutes,
        }

    def search_scenarios(
        self,
        query: str,
        domains: Optional[List[str]] = None,
        difficulty_level: str = "intermediate",
        tags: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Search for scenarios based on query and filters."""
        results = []
        search_domains = domains or self.get_available_domains()

        for domain in search_domains:
            if domain not in self._scenario_providers:
                continue

            available_scenarios = self.get_available_scenarios(domain, difficulty_level)

            for scenario_name in available_scenarios:
                # Check if query matches scenario name or description
                scenario_identifier = f"{domain}.{scenario_name}"
                metadata = self.get_scenario_metadata(scenario_identifier)

                if not metadata:
                    continue

                # Text search in name and description
                text_match = (
                    query.lower() in scenario_name.lower()
                    or query.lower() in metadata.get("description", "").lower()
                )

                # Tag filter
                tag_match = not tags or any(tag in metadata.get("tags", []) for tag in tags)

                if text_match and tag_match:
                    results.append(
                        {
                            "identifier": scenario_identifier,
                            "domain": domain,
                            "scenario_name": scenario_name,
                            "metadata": metadata,
                        }
                    )

        return results

    def export_scenario_catalog(self) -> Dict:
        """Export complete catalog of available scenarios."""
        catalog = {
            "domains": {},
            "total_scenarios": 0,
            "difficulty_levels": list(self._difficulty_levels.keys()),
        }

        for domain in self.get_available_domains():
            catalog["domains"][domain] = {}

            for difficulty_level in self._difficulty_levels.keys():
                scenarios = self.get_available_scenarios(domain, difficulty_level)
                catalog["domains"][domain][difficulty_level] = []

                for scenario_name in scenarios:
                    identifier = f"{domain}.{scenario_name}"
                    metadata = self.get_scenario_metadata(identifier)
                    if metadata:
                        catalog["domains"][domain][difficulty_level].append(metadata)
                        catalog["total_scenarios"] += 1

        return catalog

    def clear_cache(self):
        """Clear the scenario cache."""
        self._scenario_cache.clear()
        logger.info("Scenario cache cleared")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "cached_scenarios": len(self._scenario_cache),
            "registered_providers": len(self._scenario_providers),
            "available_domains": self.get_available_domains(),
        }


# Global registry instance
_global_registry: Optional[ScenarioRegistry] = None


def get_global_registry() -> ScenarioRegistry:
    """Get the global scenario registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ScenarioRegistry()
    return _global_registry


# Convenience functions for common operations


def get_scenario(
    scenario_identifier: str, difficulty_level: str = "intermediate"
) -> Optional[AttackScenario]:
    """Get a scenario from the global registry."""
    return get_global_registry().get_scenario(scenario_identifier, difficulty_level)


def list_scenarios(domain: Optional[str] = None) -> List[str]:
    """List available scenarios, optionally filtered by domain."""
    registry = get_global_registry()

    if domain:
        return [f"{domain}.{name}" for name in registry.get_available_scenarios(domain)]
    else:
        scenarios = []
        for domain in registry.get_available_domains():
            domain_scenarios = registry.get_available_scenarios(domain)
            scenarios.extend([f"{domain}.{name}" for name in domain_scenarios])
        return scenarios


def search_scenarios(query: str, **filters) -> List[Dict]:
    """Search for scenarios in the global registry."""
    return get_global_registry().search_scenarios(query, **filters)


# Registry shortcuts for specific domains


class Registry:
    """Convenience class for accessing the global registry."""

    @staticmethod
    def healthcare(difficulty_level: str = "intermediate") -> HealthcareScenarios:
        """Get healthcare scenarios provider."""
        difficulty = get_global_registry()._difficulty_levels.get(
            difficulty_level, DifficultyLevel.INTERMEDIATE
        )
        return HealthcareScenarios(difficulty)

    @staticmethod
    def financial(difficulty_level: str = "intermediate") -> FinancialScenarios:
        """Get financial scenarios provider."""
        difficulty = get_global_registry()._difficulty_levels.get(
            difficulty_level, DifficultyLevel.INTERMEDIATE
        )
        return FinancialScenarios(difficulty)

    @staticmethod
    def customer_service(difficulty_level: str = "intermediate") -> CustomerServiceScenarios:
        """Get customer service scenarios provider."""
        difficulty = get_global_registry()._difficulty_levels.get(
            difficulty_level, DifficultyLevel.INTERMEDIATE
        )
        return CustomerServiceScenarios(difficulty)
