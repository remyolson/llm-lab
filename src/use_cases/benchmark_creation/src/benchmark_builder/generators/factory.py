"""Factory for creating appropriate generators based on configuration."""

import logging
from typing import Any, Dict, Optional, Type

from .base import BaseGenerator, GeneratorConfig
from .config import DomainConfig, GenerationStrategy, GeneratorConfigManager
from .text_generator import TextGenerator

logger = logging.getLogger(__name__)


class GeneratorFactory:
    """Factory for creating test case generators."""

    # Registry of available generators
    _generators: Dict[str, Type[BaseGenerator]] = {
        "text": TextGenerator,
        "default": TextGenerator,
    }

    def __init__(self, config_manager: Optional[GeneratorConfigManager] = None):
        """
        Initialize generator factory.

        Args:
            config_manager: Configuration manager
        """
        self.config_manager = config_manager or GeneratorConfigManager()

    @classmethod
    def register_generator(cls, name: str, generator_class: Type[BaseGenerator]) -> None:
        """
        Register a new generator type.

        Args:
            name: Name for the generator type
            generator_class: Generator class
        """
        cls._generators[name] = generator_class
        logger.info(f"Registered generator: {name}")

    def create_generator(
        self,
        generator_type: str = "default",
        domain: Optional[str] = None,
        strategy: Optional[str] = None,
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> BaseGenerator:
        """
        Create a generator instance.

        Args:
            generator_type: Type of generator to create
            domain: Domain for generation
            strategy: Generation strategy
            custom_config: Custom configuration overrides

        Returns:
            Generator instance
        """
        # Get generator class
        generator_class = self._generators.get(generator_type)
        if not generator_class:
            logger.warning(f"Unknown generator type: {generator_type}, using default")
            generator_class = self._generators["default"]

        # Build configuration
        config = self._build_config(domain, strategy, custom_config)

        # Create generator
        generator = generator_class(config)

        logger.info(f"Created {generator_type} generator for domain: {domain}")
        return generator

    def _build_config(
        self,
        domain: Optional[str],
        strategy: Optional[str],
        custom_config: Optional[Dict[str, Any]],
    ) -> GeneratorConfig:
        """
        Build generator configuration.

        Args:
            domain: Domain name
            strategy: Strategy name
            custom_config: Custom configuration

        Returns:
            Generator configuration
        """
        config = GeneratorConfig()

        # Apply domain configuration
        if domain:
            domain_config = self.config_manager.get_domain_config(domain)
            if domain_config:
                config.domain = domain
                config.difficulty_distribution = domain_config.difficulty_weights
                config.task_types = domain_config.question_types
                config.metadata["domain_config"] = domain_config

        # Apply strategy configuration
        if strategy:
            strategy_config = self.config_manager.get_strategy_config(strategy)
            if strategy_config:
                config.metadata["strategy"] = strategy_config.name
                config.metadata["strategy_type"] = strategy_config.type
                config.metadata["strategy_params"] = strategy_config.parameters

        # Apply global configuration
        global_config = self.config_manager.global_config
        config.batch_size = global_config.get("batch_size", 10)
        config.metadata["global_config"] = global_config

        # Apply custom configuration
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    config.metadata[key] = value

        return config

    def create_multi_domain_generator(
        self, domains: list[str], generator_type: str = "default"
    ) -> "MultiDomainGenerator":
        """
        Create a generator that handles multiple domains.

        Args:
            domains: List of domain names
            generator_type: Type of generator

        Returns:
            Multi-domain generator
        """
        generators = {}
        for domain in domains:
            generators[domain] = self.create_generator(generator_type=generator_type, domain=domain)

        return MultiDomainGenerator(generators)

    def create_hybrid_generator(
        self, strategies: list[str], domain: Optional[str] = None
    ) -> "HybridGenerator":
        """
        Create a generator that combines multiple strategies.

        Args:
            strategies: List of strategy names
            domain: Domain for generation

        Returns:
            Hybrid generator
        """
        generators = []
        weights = []

        for strategy in strategies:
            strategy_config = self.config_manager.get_strategy_config(strategy)
            if strategy_config:
                generator = self.create_generator(domain=domain, strategy=strategy)
                generators.append(generator)
                weights.append(strategy_config.parameters.get("weight", 1.0))

        return HybridGenerator(generators, weights)


class MultiDomainGenerator(BaseGenerator):
    """Generator that handles multiple domains."""

    def __init__(self, generators: Dict[str, BaseGenerator]):
        """
        Initialize multi-domain generator.

        Args:
            generators: Dictionary of domain-specific generators
        """
        super().__init__()
        self.generators = generators
        self.current_domain = None

    def generate_single(self) -> Any:
        """Generate a single test case from a random domain."""
        import random

        domain = random.choice(list(self.generators.keys()))
        self.current_domain = domain
        test_case = self.generators[domain].generate_single()
        test_case.domain = domain
        return test_case

    def validate_case(self, test_case: Any) -> bool:
        """Validate test case using appropriate domain generator."""
        domain = test_case.domain or self.current_domain
        if domain in self.generators:
            return self.generators[domain].validate_case(test_case)
        return False

    def generate_balanced(self, count_per_domain: int) -> list:
        """
        Generate balanced test cases across domains.

        Args:
            count_per_domain: Number of cases per domain

        Returns:
            List of test cases
        """
        all_cases = []
        for domain, generator in self.generators.items():
            cases = generator.generate(count_per_domain)
            for case in cases:
                case.domain = domain
            all_cases.extend(cases)

        # Shuffle to mix domains
        import random

        random.shuffle(all_cases)

        return all_cases


class HybridGenerator(BaseGenerator):
    """Generator that combines multiple generation strategies."""

    def __init__(self, generators: list[BaseGenerator], weights: Optional[list[float]] = None):
        """
        Initialize hybrid generator.

        Args:
            generators: List of generators
            weights: Weights for each generator
        """
        super().__init__()
        self.generators = generators
        self.weights = weights or [1.0] * len(generators)

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def generate_single(self) -> Any:
        """Generate a single test case using weighted random selection."""
        import random

        import numpy as np

        # Select generator based on weights
        generator = np.random.choice(self.generators, p=self.weights)

        return generator.generate_single()

    def validate_case(self, test_case: Any) -> bool:
        """Validate test case using all generators."""
        # All generators must validate the case
        return all(gen.validate_case(test_case) for gen in self.generators)

    def generate_ensemble(self, count: int) -> list:
        """
        Generate test cases using ensemble approach.

        Args:
            count: Total number of cases

        Returns:
            List of test cases
        """
        all_cases = []

        # Calculate count for each generator
        counts = [int(count * w) for w in self.weights]

        # Adjust for rounding
        diff = count - sum(counts)
        if diff > 0:
            counts[0] += diff

        # Generate from each generator
        for generator, gen_count in zip(self.generators, counts):
            cases = generator.generate(gen_count)
            all_cases.extend(cases)

        # Shuffle to mix strategies
        import random

        random.shuffle(all_cases)

        return all_cases[:count]  # Ensure exact count
