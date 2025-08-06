"""Configuration management for generators."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DomainConfig:
    """Configuration for domain-specific generation."""

    name: str
    topics: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    difficulty_weights: Dict[str, float] = field(
        default_factory=lambda: {"easy": 0.3, "medium": 0.5, "hard": 0.2}
    )
    question_types: List[str] = field(
        default_factory=lambda: ["multiple_choice", "true_false", "open_ended"]
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationStrategy:
    """Configuration for generation strategies."""

    name: str
    type: str  # template, rule, model, hybrid
    parameters: Dict[str, Any] = field(default_factory=dict)
    model_config: Optional[Dict[str, Any]] = None
    template_path: Optional[str] = None
    rules: List[Dict[str, Any]] = field(default_factory=list)


class GeneratorConfigManager:
    """Manages generator configurations."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.domains: Dict[str, DomainConfig] = {}
        self.strategies: Dict[str, GenerationStrategy] = {}
        self.global_config: Dict[str, Any] = {}

        if config_path and config_path.exists():
            self.load_config(config_path)
        else:
            self._load_defaults()

    def _load_defaults(self) -> None:
        """Load default configurations."""
        # Default domains
        self.domains = {
            "mathematics": DomainConfig(
                name="mathematics",
                topics=["algebra", "geometry", "calculus", "statistics", "number_theory"],
                concepts=["function", "derivative", "integral", "limit", "matrix", "vector"],
                entities=["equation", "theorem", "proof", "formula", "graph"],
                properties=["linear", "continuous", "differentiable", "convergent", "symmetric"],
            ),
            "science": DomainConfig(
                name="science",
                topics=["physics", "chemistry", "biology", "astronomy", "earth_science"],
                concepts=["energy", "matter", "force", "evolution", "ecosystem"],
                entities=["atom", "molecule", "cell", "organism", "planet"],
                properties=["stable", "reactive", "organic", "inorganic", "observable"],
            ),
            "computer_science": DomainConfig(
                name="computer_science",
                topics=["algorithms", "data_structures", "databases", "networking", "security"],
                concepts=["complexity", "recursion", "iteration", "optimization", "encryption"],
                entities=["array", "tree", "graph", "hash_table", "queue"],
                properties=["efficient", "scalable", "secure", "distributed", "concurrent"],
            ),
            "language": DomainConfig(
                name="language",
                topics=["grammar", "vocabulary", "comprehension", "writing", "literature"],
                concepts=["syntax", "semantics", "pragmatics", "morphology", "phonetics"],
                entities=["noun", "verb", "sentence", "paragraph", "essay"],
                properties=["grammatical", "coherent", "persuasive", "descriptive", "analytical"],
            ),
            "general": DomainConfig(
                name="general",
                topics=["reasoning", "logic", "common_sense", "trivia", "current_events"],
                concepts=["cause", "effect", "correlation", "inference", "deduction"],
                entities=["fact", "opinion", "argument", "evidence", "conclusion"],
                properties=["valid", "sound", "relevant", "sufficient", "necessary"],
            ),
        }

        # Default strategies
        self.strategies = {
            "template": GenerationStrategy(
                name="template",
                type="template",
                parameters={"randomize": True, "variation_count": 5},
            ),
            "rule": GenerationStrategy(
                name="rule",
                type="rule",
                parameters={"complexity_level": 2, "rule_combination": "AND"},
                rules=[
                    {"type": "length", "min": 10, "max": 100},
                    {"type": "difficulty", "distribution": "normal"},
                ],
            ),
            "model": GenerationStrategy(
                name="model",
                type="model",
                parameters={"temperature": 0.7, "max_tokens": 500, "top_p": 0.9},
                model_config={"provider": "openai", "model": "gpt-3.5-turbo"},
            ),
            "hybrid": GenerationStrategy(
                name="hybrid",
                type="hybrid",
                parameters={"template_weight": 0.4, "rule_weight": 0.3, "model_weight": 0.3},
            ),
        }

        # Global configuration
        self.global_config = {
            "batch_size": 10,
            "max_retries": 3,
            "timeout": 30,
            "cache_enabled": True,
            "validation_strict": True,
            "diversity_threshold": 0.7,
            "quality_threshold": 0.8,
        }

    def load_config(self, config_path: Path) -> None:
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, "r") as f:
                if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
                    data = yaml.safe_load(f)
                elif config_path.suffix == ".json":
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")

            # Load domains
            if "domains" in data:
                for domain_name, domain_data in data["domains"].items():
                    self.domains[domain_name] = DomainConfig(name=domain_name, **domain_data)

            # Load strategies
            if "strategies" in data:
                for strategy_name, strategy_data in data["strategies"].items():
                    self.strategies[strategy_name] = GenerationStrategy(
                        name=strategy_name, **strategy_data
                    )

            # Load global config
            if "global" in data:
                self.global_config.update(data["global"])

            logger.info(f"Loaded configuration from {config_path}")

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._load_defaults()

    def save_config(self, config_path: Path) -> None:
        """
        Save configuration to file.

        Args:
            config_path: Path to save configuration
        """
        data = {
            "domains": {
                name: {
                    "topics": domain.topics,
                    "concepts": domain.concepts,
                    "entities": domain.entities,
                    "properties": domain.properties,
                    "difficulty_weights": domain.difficulty_weights,
                    "question_types": domain.question_types,
                    "metadata": domain.metadata,
                }
                for name, domain in self.domains.items()
            },
            "strategies": {
                name: {
                    "type": strategy.type,
                    "parameters": strategy.parameters,
                    "model_config": strategy.model_config,
                    "template_path": strategy.template_path,
                    "rules": strategy.rules,
                }
                for name, strategy in self.strategies.items()
            },
            "global": self.global_config,
        }

        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, "w") as f:
                if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
                    yaml.safe_dump(data, f, default_flow_style=False)
                elif config_path.suffix == ".json":
                    json.dump(data, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")

            logger.info(f"Saved configuration to {config_path}")

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def get_domain_config(self, domain: str) -> Optional[DomainConfig]:
        """
        Get configuration for a specific domain.

        Args:
            domain: Domain name

        Returns:
            Domain configuration or None
        """
        return self.domains.get(domain)

    def get_strategy_config(self, strategy: str) -> Optional[GenerationStrategy]:
        """
        Get configuration for a specific strategy.

        Args:
            strategy: Strategy name

        Returns:
            Strategy configuration or None
        """
        return self.strategies.get(strategy)

    def add_domain(self, domain: DomainConfig) -> None:
        """
        Add or update domain configuration.

        Args:
            domain: Domain configuration
        """
        self.domains[domain.name] = domain
        logger.info(f"Added domain configuration: {domain.name}")

    def add_strategy(self, strategy: GenerationStrategy) -> None:
        """
        Add or update generation strategy.

        Args:
            strategy: Generation strategy
        """
        self.strategies[strategy.name] = strategy
        logger.info(f"Added generation strategy: {strategy.name}")

    def update_global_config(self, updates: Dict[str, Any]) -> None:
        """
        Update global configuration.

        Args:
            updates: Configuration updates
        """
        self.global_config.update(updates)
        logger.info(f"Updated global configuration: {updates.keys()}")

    def validate_config(self) -> List[str]:
        """
        Validate current configuration.

        Returns:
            List of validation warnings
        """
        warnings = []

        # Check domains
        if not self.domains:
            warnings.append("No domains configured")

        for domain_name, domain in self.domains.items():
            if not domain.topics:
                warnings.append(f"Domain '{domain_name}' has no topics")
            if sum(domain.difficulty_weights.values()) != 1.0:
                warnings.append(f"Domain '{domain_name}' difficulty weights don't sum to 1.0")

        # Check strategies
        if not self.strategies:
            warnings.append("No generation strategies configured")

        for strategy_name, strategy in self.strategies.items():
            if strategy.type == "model" and not strategy.model_config:
                warnings.append(
                    f"Strategy '{strategy_name}' is model-based but has no model config"
                )
            if strategy.type == "template" and not strategy.template_path:
                warnings.append(
                    f"Strategy '{strategy_name}' is template-based but has no template path"
                )

        # Check global config
        if self.global_config.get("batch_size", 0) < 1:
            warnings.append("Invalid batch size in global config")

        return warnings
