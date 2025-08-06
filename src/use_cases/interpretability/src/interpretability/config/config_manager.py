"""Configuration management system for interpretability methods."""

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class VisualizationStyle(Enum):
    """Available visualization styles."""

    DEFAULT = "default"
    DARK = "dark"
    PAPER = "paper"
    MINIMAL = "minimal"


class AttributionMethod(Enum):
    """Available attribution methods."""

    INTEGRATED_GRADIENTS = "integrated_gradients"
    GRADIENT_SHAP = "gradient_shap"
    DEEPLIFT = "deeplift"
    OCCLUSION = "occlusion"
    LIME = "lime"
    ATTENTION_ROLLOUT = "attention_rollout"


class ExplanationStyle(Enum):
    """Available explanation styles."""

    TECHNICAL = "technical"
    SIMPLE = "simple"
    DETAILED = "detailed"


@dataclass
class HookConfig:
    """Configuration for hook management."""

    enabled: bool = True
    target_layers: List[str] = field(default_factory=list)
    layer_types: List[str] = field(
        default_factory=lambda: ["Linear", "Conv2d", "MultiheadAttention", "LayerNorm"]
    )
    collect_gradients: bool = True
    collect_activations: bool = True
    collect_attention: bool = True
    max_stored_batches: int = 10


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""

    style: VisualizationStyle = VisualizationStyle.DEFAULT
    figure_size: Tuple[int, int] = (10, 8)
    dpi: int = 150
    color_scheme: str = "Blues"
    interactive: bool = True
    save_format: str = "png"
    animation_fps: int = 10
    max_tokens_display: int = 50


@dataclass
class AttributionConfig:
    """Configuration for attribution methods."""

    default_method: AttributionMethod = AttributionMethod.INTEGRATED_GRADIENTS
    integrated_gradients_steps: int = 50
    gradient_shap_samples: int = 50
    gradient_shap_stdev: float = 0.1
    occlusion_window_size: int = 3
    occlusion_stride: int = 1
    lime_samples: int = 1000
    lime_feature_mask_prob: float = 0.5
    use_baseline: bool = True
    baseline_type: str = "zero"  # "zero", "mean", "random", "custom"


@dataclass
class ExplanationConfig:
    """Configuration for explanation generation."""

    style: ExplanationStyle = ExplanationStyle.TECHNICAL
    include_confidence: bool = True
    include_top_features: bool = True
    top_features_count: int = 10
    include_attention_summary: bool = True
    include_gradient_summary: bool = True
    include_activation_summary: bool = True
    contrastive_explanations: bool = False
    natural_language: bool = True


@dataclass
class DashboardConfig:
    """Configuration for dashboard settings."""

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8050
    auto_refresh: bool = True
    refresh_interval: int = 5000  # milliseconds
    theme: str = "bootstrap"
    debug: bool = False


@dataclass
class ModelConfig:
    """Configuration for model-specific settings."""

    model_type: str = "transformer"  # "transformer", "cnn", "rnn", "custom"
    framework: str = "pytorch"  # "pytorch", "tensorflow", "jax"
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    mixed_precision: bool = False
    batch_size: int = 32
    max_sequence_length: Optional[int] = 512


@dataclass
class OutputConfig:
    """Configuration for output settings."""

    output_dir: str = "./interpretability_outputs"
    save_raw_data: bool = True
    save_visualizations: bool = True
    save_explanations: bool = True
    compression: bool = True
    timestamp_outputs: bool = True
    create_report: bool = True
    report_format: str = "html"  # "html", "pdf", "markdown"


@dataclass
class InterpretabilityConfig:
    """Main configuration for interpretability suite."""

    hook: HookConfig = field(default_factory=HookConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
    explanation: ExplanationConfig = field(default_factory=ExplanationConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Advanced settings
    cache_enabled: bool = True
    cache_size_mb: int = 500
    parallel_processing: bool = True
    num_workers: int = 4
    logging_level: str = "INFO"
    random_seed: Optional[int] = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "InterpretabilityConfig":
        """Create configuration from dictionary."""
        # Handle nested configurations
        if "hook" in config_dict and isinstance(config_dict["hook"], dict):
            config_dict["hook"] = HookConfig(**config_dict["hook"])
        if "visualization" in config_dict and isinstance(config_dict["visualization"], dict):
            viz_dict = config_dict["visualization"]
            if "style" in viz_dict and isinstance(viz_dict["style"], str):
                viz_dict["style"] = VisualizationStyle(viz_dict["style"])
            config_dict["visualization"] = VisualizationConfig(**viz_dict)
        if "attribution" in config_dict and isinstance(config_dict["attribution"], dict):
            attr_dict = config_dict["attribution"]
            if "default_method" in attr_dict and isinstance(attr_dict["default_method"], str):
                attr_dict["default_method"] = AttributionMethod(attr_dict["default_method"])
            config_dict["attribution"] = AttributionConfig(**attr_dict)
        if "explanation" in config_dict and isinstance(config_dict["explanation"], dict):
            exp_dict = config_dict["explanation"]
            if "style" in exp_dict and isinstance(exp_dict["style"], str):
                exp_dict["style"] = ExplanationStyle(exp_dict["style"])
            config_dict["explanation"] = ExplanationConfig(**exp_dict)
        if "dashboard" in config_dict and isinstance(config_dict["dashboard"], dict):
            config_dict["dashboard"] = DashboardConfig(**config_dict["dashboard"])
        if "model" in config_dict and isinstance(config_dict["model"], dict):
            config_dict["model"] = ModelConfig(**config_dict["model"])
        if "output" in config_dict and isinstance(config_dict["output"], dict):
            config_dict["output"] = OutputConfig(**config_dict["output"])

        return cls(**config_dict)


class ConfigManager:
    """Manages configuration for interpretability suite."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path
        self.config = InterpretabilityConfig()

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return

        # Determine format from extension
        if config_path.suffix == ".json":
            with open(config_path, "r") as f:
                config_dict = json.load(f)
        elif config_path.suffix in [".yaml", ".yml"]:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

        self.config = InterpretabilityConfig.from_dict(config_dict)
        logger.info(f"Loaded configuration from {config_path}")

    def save_config(self, config_path: Optional[str] = None) -> None:
        """
        Save configuration to file.

        Args:
            config_path: Path to save configuration
        """
        if config_path is None:
            config_path = self.config_path

        if config_path is None:
            raise ValueError("No config path specified")

        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.config.to_dict()

        # Convert enums to strings for serialization
        def convert_enums(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums(item) for item in obj]
            return obj

        config_dict = convert_enums(config_dict)

        # Save based on extension
        if config_path.suffix == ".json":
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
        elif config_path.suffix in [".yaml", ".yml"]:
            with open(config_path, "w") as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

        logger.info(f"Saved configuration to {config_path}")

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of updates
        """
        for key, value in updates.items():
            if hasattr(self.config, key):
                if isinstance(value, dict):
                    # Handle nested updates
                    current = getattr(self.config, key)
                    if hasattr(current, "__dict__"):
                        for sub_key, sub_value in value.items():
                            if hasattr(current, sub_key):
                                setattr(current, sub_key, sub_value)
                else:
                    setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown config key: {key}")

    def get_preset(self, preset_name: str) -> InterpretabilityConfig:
        """
        Get a preset configuration.

        Args:
            preset_name: Name of the preset

        Returns:
            Preset configuration
        """
        presets = {
            "minimal": self._get_minimal_preset(),
            "comprehensive": self._get_comprehensive_preset(),
            "fast": self._get_fast_preset(),
            "research": self._get_research_preset(),
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}")

        return presets[preset_name]

    def _get_minimal_preset(self) -> InterpretabilityConfig:
        """Get minimal configuration preset."""
        config = InterpretabilityConfig()
        config.hook.collect_gradients = False
        config.visualization.interactive = False
        config.attribution.integrated_gradients_steps = 20
        config.explanation.style = ExplanationStyle.SIMPLE
        config.dashboard.enabled = False
        config.output.save_raw_data = False
        return config

    def _get_comprehensive_preset(self) -> InterpretabilityConfig:
        """Get comprehensive configuration preset."""
        config = InterpretabilityConfig()
        config.hook.collect_gradients = True
        config.hook.collect_activations = True
        config.hook.collect_attention = True
        config.visualization.interactive = True
        config.attribution.integrated_gradients_steps = 100
        config.explanation.style = ExplanationStyle.DETAILED
        config.explanation.contrastive_explanations = True
        config.dashboard.enabled = True
        config.output.save_raw_data = True
        config.output.create_report = True
        return config

    def _get_fast_preset(self) -> InterpretabilityConfig:
        """Get fast configuration preset."""
        config = InterpretabilityConfig()
        config.hook.max_stored_batches = 5
        config.visualization.interactive = False
        config.visualization.dpi = 100
        config.attribution.integrated_gradients_steps = 25
        config.attribution.gradient_shap_samples = 25
        config.attribution.lime_samples = 500
        config.explanation.style = ExplanationStyle.SIMPLE
        config.dashboard.auto_refresh = False
        config.parallel_processing = True
        config.cache_enabled = True
        return config

    def _get_research_preset(self) -> InterpretabilityConfig:
        """Get research configuration preset."""
        config = InterpretabilityConfig()
        config.hook.collect_gradients = True
        config.hook.collect_activations = True
        config.hook.collect_attention = True
        config.hook.max_stored_batches = 50
        config.visualization.style = VisualizationStyle.PAPER
        config.visualization.dpi = 300
        config.visualization.save_format = "pdf"
        config.attribution.integrated_gradients_steps = 200
        config.attribution.gradient_shap_samples = 100
        config.explanation.style = ExplanationStyle.DETAILED
        config.explanation.contrastive_explanations = True
        config.output.save_raw_data = True
        config.output.compression = False
        config.output.create_report = True
        config.output.report_format = "pdf"
        return config

    def validate_config(self) -> List[str]:
        """
        Validate current configuration.

        Returns:
            List of validation warnings
        """
        warnings = []

        # Check for conflicting settings
        if self.config.dashboard.enabled and not self.config.visualization.interactive:
            warnings.append("Dashboard enabled but interactive visualization disabled")

        if self.config.attribution.integrated_gradients_steps < 10:
            warnings.append("Very low integration steps may reduce attribution quality")

        if self.config.model.batch_size > 256:
            warnings.append("Large batch size may cause memory issues")

        if self.config.cache_size_mb > 2000:
            warnings.append("Large cache size may impact system memory")

        if self.config.parallel_processing and self.config.num_workers > 16:
            warnings.append("High number of workers may not improve performance")

        return warnings

    def export_cli_args(self) -> Dict[str, Any]:
        """
        Export configuration as CLI arguments.

        Returns:
            Dictionary of CLI arguments
        """
        args = {}

        # Flatten configuration for CLI
        for section_name in [
            "hook",
            "visualization",
            "attribution",
            "explanation",
            "dashboard",
            "model",
            "output",
        ]:
            section = getattr(self.config, section_name)
            for key, value in section.__dict__.items():
                cli_key = f"--{section_name}-{key}".replace("_", "-")
                args[cli_key] = value

        # Add top-level settings
        args["--cache-enabled"] = self.config.cache_enabled
        args["--cache-size-mb"] = self.config.cache_size_mb
        args["--parallel-processing"] = self.config.parallel_processing
        args["--num-workers"] = self.config.num_workers
        args["--logging-level"] = self.config.logging_level
        args["--random-seed"] = self.config.random_seed

        return args
