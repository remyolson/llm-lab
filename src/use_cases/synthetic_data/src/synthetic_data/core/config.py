"""Configuration management for synthetic data generation."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


class GenerationConfig(BaseModel):
    """Configuration for data generation."""

    model: str = Field(default="gpt-4", description="LLM model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    batch_size: int = Field(default=10, gt=0)
    retry_attempts: int = Field(default=3, ge=0)
    timeout: int = Field(default=30, gt=0)

    class Config:
        extra = "allow"


class ValidationConfig(BaseModel):
    """Configuration for data validation."""

    enable_quality_checks: bool = True
    min_quality_score: float = Field(default=0.8, ge=0.0, le=1.0)
    check_diversity: bool = True
    check_consistency: bool = True
    check_format: bool = True


class PrivacyConfig(BaseModel):
    """Configuration for privacy preservation."""

    enable_privacy_preservation: bool = True
    differential_privacy_epsilon: float = Field(default=1.0, gt=0.0)
    anonymization_level: str = Field(default="medium", pattern="^(low|medium|high)$")
    pii_detection: bool = True
    pii_removal: bool = True


class DomainConfig(BaseModel):
    """Configuration for a specific domain."""

    templates_path: str
    validation_rules: str = Field(default="moderate", pattern="^(strict|moderate|lenient)$")
    privacy_level: str = Field(default="medium", pattern="^(low|medium|high)$")
    custom_settings: Dict[str, Any] = Field(default_factory=dict)


class PlatformConfig(BaseModel):
    """Main platform configuration."""

    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    domains: Dict[str, DomainConfig] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)
    logging: Dict[str, Any] = Field(default_factory=dict)


def load_config(config_path: Optional[str] = None) -> PlatformConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Platform configuration
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "default.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return PlatformConfig()

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    return PlatformConfig(**config_data)


def save_config(config: PlatformConfig, config_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Platform configuration
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config.dict(), f, default_flow_style=False)
