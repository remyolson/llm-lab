"""Configuration classes for security scanning."""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class SeverityLevel(str, Enum):
    """Security vulnerability severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class DetectionMode(str, Enum):
    """Detection strategy modes."""

    RULE_BASED = "rule_based"
    ML_BASED = "ml_based"
    HEURISTIC = "heuristic"
    COMBINED = "combined"


@dataclass
class SeverityThresholds:
    """Severity classification thresholds."""

    critical: float = 0.9
    high: float = 0.7
    medium: float = 0.5
    low: float = 0.3
    # Below low threshold = info level

    def classify_score(self, score: float) -> SeverityLevel:
        """Classify a confidence score into severity level."""
        if score >= self.critical:
            return SeverityLevel.CRITICAL
        elif score >= self.high:
            return SeverityLevel.HIGH
        elif score >= self.medium:
            return SeverityLevel.MEDIUM
        elif score >= self.low:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.INFO


@dataclass
class ParallelConfig:
    """Configuration for parallel scanning."""

    max_workers: int = 4
    batch_size: int = 10
    timeout_seconds: int = 30
    rate_limit_per_second: Optional[float] = None
    enable_backpressure: bool = True


@dataclass
class DetectionStrategyConfig:
    """Configuration for individual detection strategies."""

    enabled: bool = True
    weight: float = 1.0
    strategy_specific: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.weight < 0.0 or self.weight > 1.0:
            raise ValueError("Strategy weight must be between 0.0 and 1.0")


@dataclass
class ScanConfig:
    """Comprehensive scanning configuration."""

    # Detection strategy configuration
    detection_modes: List[DetectionMode] = field(default_factory=lambda: [DetectionMode.COMBINED])
    strategy_configs: Dict[str, DetectionStrategyConfig] = field(
        default_factory=lambda: {
            "rule_based": DetectionStrategyConfig(weight=0.4),
            "ml_based": DetectionStrategyConfig(weight=0.4),
            "heuristic": DetectionStrategyConfig(weight=0.2),
        }
    )

    # Severity and scoring
    severity_thresholds: SeverityThresholds = field(default_factory=SeverityThresholds)
    min_confidence_threshold: float = 0.1
    max_vulnerabilities_per_scan: int = 100

    # Test suite selection
    attack_categories: Optional[List[str]] = None
    attack_severity_filter: Optional[List[str]] = None
    custom_attack_ids: Optional[List[str]] = None

    # Performance configuration
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600

    # Analysis configuration
    include_response_analysis: bool = True
    include_pattern_matching: bool = True
    include_sentiment_analysis: bool = True
    include_contextual_analysis: bool = True

    # Output configuration
    include_explanations: bool = True
    detailed_scoring: bool = True
    export_intermediate_results: bool = False

    # Model-specific settings
    target_model_config: Dict[str, Any] = field(default_factory=dict)
    model_timeout_seconds: int = 60

    @classmethod
    def from_file(cls, config_path: Path) -> "ScanConfig":
        """Load configuration from JSON file."""
        with open(config_path, "r") as f:
            data = json.load(f)

        # Handle nested objects
        if "severity_thresholds" in data:
            data["severity_thresholds"] = SeverityThresholds(**data["severity_thresholds"])

        if "parallel_config" in data:
            data["parallel_config"] = ParallelConfig(**data["parallel_config"])

        # Handle strategy configs
        if "strategy_configs" in data:
            strategy_configs = {}
            for name, config_data in data["strategy_configs"].items():
                strategy_configs[name] = DetectionStrategyConfig(**config_data)
            data["strategy_configs"] = strategy_configs

        # Convert enum strings
        if "detection_modes" in data:
            data["detection_modes"] = [DetectionMode(mode) for mode in data["detection_modes"]]

        return cls(**data)

    def to_file(self, config_path: Path):
        """Save configuration to JSON file."""
        data = asdict(self)

        # Convert enums to strings for JSON serialization
        data["detection_modes"] = [mode.value for mode in self.detection_modes]

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Validate thresholds
        thresholds = [
            self.severity_thresholds.critical,
            self.severity_thresholds.high,
            self.severity_thresholds.medium,
            self.severity_thresholds.low,
        ]

        if not all(0.0 <= t <= 1.0 for t in thresholds):
            issues.append("All severity thresholds must be between 0.0 and 1.0")

        if not all(thresholds[i] > thresholds[i + 1] for i in range(len(thresholds) - 1)):
            issues.append("Severity thresholds must be in descending order")

        # Validate strategy weights
        total_weight = sum(
            config.weight for config in self.strategy_configs.values() if config.enabled
        )
        if total_weight == 0:
            issues.append("At least one detection strategy must be enabled with weight > 0")

        # Validate parallel config
        if self.parallel_config.max_workers <= 0:
            issues.append("max_workers must be positive")

        if self.parallel_config.batch_size <= 0:
            issues.append("batch_size must be positive")

        # Validate confidence threshold
        if not 0.0 <= self.min_confidence_threshold <= 1.0:
            issues.append("min_confidence_threshold must be between 0.0 and 1.0")

        return issues

    def get_enabled_strategies(self) -> Dict[str, DetectionStrategyConfig]:
        """Get only enabled detection strategies."""
        return {name: config for name, config in self.strategy_configs.items() if config.enabled}

    def normalize_strategy_weights(self):
        """Normalize strategy weights to sum to 1.0."""
        enabled_configs = self.get_enabled_strategies()
        total_weight = sum(config.weight for config in enabled_configs.values())

        if total_weight > 0:
            for config in enabled_configs.values():
                config.weight /= total_weight

    def get_default_config() -> "ScanConfig":
        """Get default scanning configuration."""
        return ScanConfig()

    @staticmethod
    def create_quick_scan_config() -> "ScanConfig":
        """Create configuration optimized for quick scanning."""
        return ScanConfig(
            detection_modes=[DetectionMode.RULE_BASED],
            strategy_configs={
                "rule_based": DetectionStrategyConfig(weight=1.0),
                "ml_based": DetectionStrategyConfig(enabled=False),
                "heuristic": DetectionStrategyConfig(enabled=False),
            },
            parallel_config=ParallelConfig(max_workers=2, batch_size=20),
            include_sentiment_analysis=False,
            include_contextual_analysis=False,
            detailed_scoring=False,
        )

    @staticmethod
    def create_thorough_scan_config() -> "ScanConfig":
        """Create configuration for comprehensive scanning."""
        return ScanConfig(
            detection_modes=[DetectionMode.COMBINED],
            strategy_configs={
                "rule_based": DetectionStrategyConfig(weight=0.3),
                "ml_based": DetectionStrategyConfig(weight=0.4),
                "heuristic": DetectionStrategyConfig(weight=0.3),
            },
            parallel_config=ParallelConfig(max_workers=8, batch_size=5),
            include_response_analysis=True,
            include_pattern_matching=True,
            include_sentiment_analysis=True,
            include_contextual_analysis=True,
            detailed_scoring=True,
            include_explanations=True,
        )
