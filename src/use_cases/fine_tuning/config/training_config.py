"""
Training configuration management for fine-tuning.

This module provides configuration classes with validation for
managing fine-tuning hyperparameters and settings.
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class OptimizerType(str, Enum):
    """Supported optimizer types."""

    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"
    ADAFACTOR = "adafactor"


class SchedulerType(str, Enum):
    """Supported scheduler types."""

    NONE = "none"
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class LoggingBackend(str, Enum):
    """Supported logging backends."""

    NONE = "none"
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    BOTH = "both"


@dataclass
class LoRAConfig:
    """Configuration for LoRA parameters."""

    r: int = 8  # Rank
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    bias: str = "none"  # none, all, lora_only
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    fan_in_fan_out: bool = False
    modules_to_save: Optional[List[str]] = None

    def validate(self):
        """Validate LoRA configuration."""
        if self.r <= 0:
            raise ValueError("LoRA rank (r) must be positive")

        if self.lora_alpha <= 0:
            raise ValueError("LoRA alpha must be positive")

        if not 0 <= self.lora_dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")

        if self.bias not in ["none", "all", "lora_only"]:
            raise ValueError(f"Invalid bias type: {self.bias}")


@dataclass
class DataConfig:
    """Configuration for dataset parameters."""

    dataset_path: str
    validation_split: float = 0.1
    test_split: float = 0.0
    max_seq_length: int = 512
    input_format: str = "auto"  # auto, instruction, prompt_completion, text
    num_workers: int = 0
    preprocessing_num_workers: Optional[int] = None

    def validate(self):
        """Validate data configuration."""
        if not self.dataset_path:
            raise ValueError("Dataset path must be provided")

        if not 0 <= self.validation_split < 1:
            raise ValueError("Validation split must be in [0, 1)")

        if not 0 <= self.test_split < 1:
            raise ValueError("Test split must be in [0, 1)")

        if self.validation_split + self.test_split >= 1:
            raise ValueError("Sum of validation and test splits must be < 1")

        if self.max_seq_length <= 0:
            raise ValueError("Max sequence length must be positive")


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    # Model settings
    model_name: str
    use_qlora: bool = False
    bits: int = 4  # For QLoRA
    trust_remote_code: bool = False

    # Training hyperparameters
    learning_rate: float = 5e-5
    num_epochs: int = 3
    batch_size: int = 4
    eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Optimizer and scheduler
    optimizer: OptimizerType = OptimizerType.ADAMW
    scheduler: SchedulerType = SchedulerType.LINEAR
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # LoRA configuration
    lora_config: Optional[LoRAConfig] = None

    # Data configuration
    data_config: Optional[DataConfig] = None

    # Training control
    seed: int = 42
    device: str = "auto"  # auto, cuda, mps, cpu
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False

    # Checkpointing
    output_dir: str = "./outputs"
    save_strategy: str = "epoch"  # epoch, steps, best
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True

    # Evaluation
    evaluation_strategy: str = "epoch"  # epoch, steps, no
    eval_steps: int = 500
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False

    # Early stopping
    early_stopping_patience: int = 0
    early_stopping_threshold: float = 0.0

    # Logging
    logging_backend: LoggingBackend = LoggingBackend.TENSORBOARD
    logging_dir: Optional[str] = None
    logging_steps: int = 10
    log_level: str = "INFO"
    show_progress: bool = True

    # W&B specific
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None

    # Benchmark integration
    run_benchmark_after: bool = False
    benchmark_dataset: Optional[str] = None
    benchmark_tasks: Optional[List[str]] = None

    def __post_init__(self):
        """Initialize nested configs if provided as dicts."""
        if isinstance(self.lora_config, dict):
            self.lora_config = LoRAConfig(**self.lora_config)

        if isinstance(self.data_config, dict):
            self.data_config = DataConfig(**self.data_config)

        # Set default eval batch size
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size

        # Set default logging dir
        if self.logging_dir is None:
            self.logging_dir = os.path.join(self.output_dir, "logs")

    def validate(self):
        """Validate the complete configuration."""
        # Basic validation
        if not self.model_name:
            raise ValueError("Model name must be provided")

        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")

        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        if self.gradient_accumulation_steps <= 0:
            raise ValueError("Gradient accumulation steps must be positive")

        # Validate precision settings
        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both fp16 and bf16")

        # Validate warmup settings
        if self.warmup_steps > 0 and self.warmup_ratio > 0:
            raise ValueError("Cannot specify both warmup_steps and warmup_ratio")

        # Validate save/eval strategies
        if self.save_strategy not in ["epoch", "steps", "best", "no"]:
            raise ValueError(f"Invalid save strategy: {self.save_strategy}")

        if self.evaluation_strategy not in ["epoch", "steps", "no"]:
            raise ValueError(f"Invalid evaluation strategy: {self.evaluation_strategy}")

        # Validate nested configs
        if self.lora_config:
            self.lora_config.validate()

        if self.data_config:
            self.data_config.validate()

        # QLoRA validation
        if self.use_qlora:
            if self.bits not in [4, 8]:
                raise ValueError("QLoRA only supports 4-bit or 8-bit quantization")

            if not self.lora_config:
                raise ValueError("QLoRA requires LoRA configuration")

    def to_dict(self) -> Dict[str | Any]:
        """Convert configuration to dictionary."""
        config_dict = asdict(self)

        # Convert enums to strings
        config_dict["optimizer"] = self.optimizer.value
        config_dict["scheduler"] = self.scheduler.value
        config_dict["logging_backend"] = self.logging_backend.value

        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create configuration from dictionary."""
        # Convert string enums back
        if "optimizer" in config_dict:
            config_dict["optimizer"] = OptimizerType(config_dict["optimizer"])

        if "scheduler" in config_dict:
            config_dict["scheduler"] = SchedulerType(config_dict["scheduler"])

        if "logging_backend" in config_dict:
            config_dict["logging_backend"] = LoggingBackend(config_dict["logging_backend"])

        return cls(**config_dict)

    def save(self, path: str | Path):
        """Save configuration to file."""
        path = Path(path)
        config_dict = self.to_dict()

        if path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)
        elif path.suffix in [".yaml", ".yml"]:
            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        logger.info(f"Saved configuration to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        """Load configuration from file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        if path.suffix == ".json":
            with open(path) as f:
                config_dict = json.load(f)
        elif path.suffix in [".yaml", ".yml"]:
            with open(path) as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return cls.from_dict(config_dict)

    def setup_logging(self):
        """Set up logging backends."""
        if self.logging_backend in [LoggingBackend.NONE, LoggingBackend.NONE.value]:
            return None

        loggers = {}

        # TensorBoard setup
        if self.logging_backend in [LoggingBackend.TENSORBOARD, LoggingBackend.BOTH]:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_dir = os.path.join(self.logging_dir, "tensorboard")
                os.makedirs(tb_dir, exist_ok=True)
                loggers["tensorboard"] = SummaryWriter(tb_dir)
                logger.info(f"TensorBoard logging enabled: {tb_dir}")
            except ImportError:
                logger.warning("TensorBoard not available. Install with: pip install tensorboard")

        # Weights & Biases setup
        if self.logging_backend in [LoggingBackend.WANDB, LoggingBackend.BOTH]:
            try:
                import wandb

                wandb_config = {
                    "learning_rate": self.learning_rate,
                    "epochs": self.num_epochs,
                    "batch_size": self.batch_size,
                    "model": self.model_name,
                }

                if self.lora_config:
                    wandb_config.update(
                        {
                            "lora_r": self.lora_config.r,
                            "lora_alpha": self.lora_config.lora_alpha,
                            "lora_dropout": self.lora_config.lora_dropout,
                        }
                    )

                wandb.init(
                    project=self.wandb_project or "llm-lab-finetuning",
                    entity=self.wandb_entity,
                    name=self.wandb_name,
                    tags=self.wandb_tags,
                    config=wandb_config,
                )

                loggers["wandb"] = wandb
                logger.info("Weights & Biases logging enabled")

            except ImportError:
                logger.warning("W&B not available. Install with: pip install wandb")

        return loggers


# Pre-defined configuration templates
TRAINING_TEMPLATES = {
    "lora_standard": TrainingConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        learning_rate=1e-4,
        num_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=4,
        lora_config=LoRAConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        ),
        data_config=DataConfig(
            dataset_path="",  # To be filled
            max_seq_length=512,
        ),
    ),
    "qlora_efficient": TrainingConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        use_qlora=True,
        bits=4,
        learning_rate=2e-4,
        num_epochs=3,
        batch_size=8,
        gradient_accumulation_steps=2,
        fp16=True,
        gradient_checkpointing=True,
        lora_config=LoRAConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
        data_config=DataConfig(
            dataset_path="",  # To be filled
            max_seq_length=1024,
        ),
    ),
    "instruction_tuning": TrainingConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        learning_rate=5e-5,
        num_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
        scheduler=SchedulerType.COSINE,
        lora_config=LoRAConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        ),
        data_config=DataConfig(
            dataset_path="",  # To be filled
            max_seq_length=2048,
            input_format="instruction",
        ),
        early_stopping_patience=3,
        save_strategy="best",
        evaluation_strategy="steps",
        eval_steps=100,
    ),
}


def get_template(template_name: str) -> TrainingConfig:
    """
    Get a pre-defined configuration template.

    Args:
        template_name: Name of the template

    Returns:
        TrainingConfig instance
    """
    if template_name not in TRAINING_TEMPLATES:
        available = ", ".join(TRAINING_TEMPLATES.keys())
        raise ValueError(f"Unknown template: {template_name}. Available: {available}")

    # Return a copy to avoid modifying the template
    template = TRAINING_TEMPLATES[template_name]
    return TrainingConfig.from_dict(template.to_dict())
