"""
Base trainer class for fine-tuning models.

This module provides the abstract base class that all trainer implementations
must extend, defining the common interface and shared functionality.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    epoch: int
    step: int
    loss: float
    learning_rate: float
    gradient_norm: Optional[float] = None
    memory_used_gb: Optional[float] = None
    examples_per_second: Optional[float] = None
    additional_metrics: Dict[str, float] = field(default_factory=dict)


class BaseTrainer(ABC):
    """
    Abstract base class for model trainers.

    This class provides the common functionality needed for training models,
    including device management, logging, checkpointing, and the training loop
    structure. Subclasses must implement the model-specific methods.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base trainer.

        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.validate_config(config)

        # Set up logging
        self._setup_logging()

        # Device management
        self.device = self._get_device()
        logger.info(f"Using device: {self.device}")

        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = None
        self.metrics_history: List[TrainingMetrics] = []

        # Paths
        self.output_dir = Path(config.get("output_dir", "./outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Early stopping
        self.early_stopping_patience = config.get("early_stopping_patience", 0)
        self.early_stopping_counter = 0
        self.early_stopping_metric = config.get("early_stopping_metric", "loss")
        self.early_stopping_mode = config.get("early_stopping_mode", "min")

    def _setup_logging(self):
        """Set up logging configuration."""
        log_level = self.config.get("log_level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def _get_device(self) -> torch.device:
        """
        Get the device to use for training.

        Returns:
            torch.device: The device (cuda/mps/cpu)
        """
        device_str = self.config.get("device", "auto")

        if device_str == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("MPS available. Using Apple Silicon GPU")
            else:
                device = torch.device("cpu")
                logger.info("No GPU available. Using CPU")
        else:
            device = torch.device(device_str)

        return device

    def validate_config(self, config: Dict[str, Any]):
        """
        Validate the training configuration.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ["model_name", "learning_rate", "num_epochs"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Required configuration field missing: {field}")

        # Validate numeric fields
        if config["learning_rate"] <= 0:
            raise ValueError("Learning rate must be positive")

        if config["num_epochs"] <= 0:
            raise ValueError("Number of epochs must be positive")

        # Validate batch size
        batch_size = config.get("batch_size", 1)
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")

    @abstractmethod
    def load_model(self) -> torch.nn.Module:
        """
        Load and prepare the model for training.

        Returns:
            The model ready for training
        """
        pass

    @abstractmethod
    def prepare_dataset(self, dataset_path: str) -> Tuple[DataLoader | Optional[DataLoader]]:
        """
        Prepare the dataset for training.

        Args:
            dataset_path: Path to the dataset

        Returns:
            Tuple of (train_dataloader, eval_dataloader)
        """
        pass

    @abstractmethod
    def compute_loss(self, model: torch.nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute the loss for a batch.

        Args:
            model: The model
            batch: Batch of data

        Returns:
            Loss tensor
        """
        pass

    def train(self, dataset_path: str):
        """
        Main training loop.

        Args:
            dataset_path: Path to the training dataset
        """
        logger.info("Starting training...")
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")

        # Load model
        self.model = self.load_model()
        self.model.to(self.device)

        # Prepare dataset
        self.train_dataloader, self.eval_dataloader = self.prepare_dataset(dataset_path)

        # Set up optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Training loop
        for epoch in range(self.config["num_epochs"]):
            self.current_epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")

            # Train for one epoch
            train_metrics = self._train_epoch()

            # Evaluate if eval dataset is provided
            if self.eval_dataloader:
                eval_metrics = self._evaluate()
                train_metrics.additional_metrics.update(
                    {f"eval_{k}": v for k, v in eval_metrics.items()}
                )

            # Log metrics
            self._log_metrics(train_metrics)
            self.metrics_history.append(train_metrics)

            # Save checkpoint
            if self._should_save_checkpoint(epoch):
                self.save_checkpoint(epoch, train_metrics)

            # Early stopping check
            if self._check_early_stopping(train_metrics):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        logger.info("Training completed!")
        self._save_final_model()
        self._save_training_summary()

    def _train_epoch(self) -> TrainingMetrics:
        """
        Train for one epoch.

        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_dataloader)
        start_time = time.time()

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch + 1}",
            disable=not self.config.get("show_progress", True),
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Forward pass
            loss = self.compute_loss(self.model, batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if "max_grad_norm" in self.config:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["max_grad_norm"]
                )
            else:
                grad_norm = None

            # Optimizer step
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"}
            )

        # Calculate epoch metrics
        epoch_time = time.time() - start_time
        avg_loss = total_loss / num_batches
        examples_per_second = len(self.train_dataloader.dataset) / epoch_time

        # Get memory usage
        memory_used_gb = None
        if self.device.type == "cuda":
            memory_used_gb = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.reset_peak_memory_stats()

        return TrainingMetrics(
            epoch=self.current_epoch,
            step=self.global_step,
            loss=avg_loss,
            learning_rate=self.optimizer.param_groups[0]["lr"],
            gradient_norm=grad_norm.item() if grad_norm else None,
            memory_used_gb=memory_used_gb,
            examples_per_second=examples_per_second,
        )

    def _evaluate(self) -> Dict[str | float]:
        """
        Evaluate the model on the evaluation dataset.

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        num_batches = len(self.eval_dataloader)

        with torch.no_grad():
            for batch in tqdm(
                self.eval_dataloader,
                desc="Evaluating",
                disable=not self.config.get("show_progress", True),
            ):
                batch = self._move_batch_to_device(batch)
                loss = self.compute_loss(self.model, batch)
                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str | Any]:
        """Move batch tensors to the training device."""
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            else:
                moved_batch[key] = value
        return moved_batch

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer."""
        optimizer_type = self.config.get("optimizer", "adamw")
        learning_rate = self.config["learning_rate"]
        weight_decay = self.config.get("weight_decay", 0.01)

        if optimizer_type.lower() == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_type.lower() == "adam":
            return torch.optim.Adam(
                self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_type.lower() == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=self.config.get("momentum", 0.9),
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create the learning rate scheduler."""
        scheduler_type = self.config.get("scheduler", "none")

        if scheduler_type == "none":
            return None
        elif scheduler_type == "linear":
            num_training_steps = len(self.train_dataloader) * self.config["num_epochs"]
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps
            )
        elif scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["num_epochs"],
                eta_min=self.config.get("min_lr", 0),
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

    def _should_save_checkpoint(self, epoch: int) -> bool:
        """Determine if checkpoint should be saved."""
        save_strategy = self.config.get("save_strategy", "epoch")

        if save_strategy == "epoch":
            return True
        elif save_strategy == "steps":
            save_steps = self.config.get("save_steps", 1000)
            return self.global_step % save_steps == 0
        elif save_strategy == "best":
            return self._is_best_model()
        else:
            return False

    def _is_best_model(self) -> bool:
        """Check if current model is the best so far."""
        if not self.metrics_history:
            return True

        current_metric = self.metrics_history[-1].additional_metrics.get(
            f"eval_{self.early_stopping_metric}", self.metrics_history[-1].loss
        )

        if self.best_metric is None:
            self.best_metric = current_metric
            return True

        if self.early_stopping_mode == "min":
            is_best = current_metric < self.best_metric
        else:
            is_best = current_metric > self.best_metric

        if is_best:
            self.best_metric = current_metric

        return is_best

    def _check_early_stopping(self, metrics: TrainingMetrics) -> bool:
        """Check if early stopping criteria is met."""
        if self.early_stopping_patience <= 0:
            return False

        current_metric = metrics.additional_metrics.get(
            f"eval_{self.early_stopping_metric}", metrics.loss
        )

        if self._is_best_model():
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        return self.early_stopping_counter >= self.early_stopping_patience

    def save_checkpoint(self, epoch: int, metrics: TrainingMetrics):
        """
        Save training checkpoint.

        Args:
            epoch: Current epoch
            metrics: Training metrics
        """
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": metrics,
            "config": self.config,
            "best_metric": self.best_metric,
        }

        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save as best model if applicable
        if self._is_best_model():
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_metric = checkpoint.get("best_metric")

        logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")

    def _log_metrics(self, metrics: TrainingMetrics):
        """Log training metrics."""
        log_str = (
            f"Epoch {metrics.epoch + 1} - "
            f"Loss: {metrics.loss:.4f} - "
            f"LR: {metrics.learning_rate:.2e}"
        )

        if metrics.gradient_norm:
            log_str += f" - Grad Norm: {metrics.gradient_norm:.2f}"

        if metrics.memory_used_gb:
            log_str += f" - Memory: {metrics.memory_used_gb:.2f} GB"

        if metrics.examples_per_second:
            log_str += f" - Speed: {metrics.examples_per_second:.1f} ex/s"

        for key, value in metrics.additional_metrics.items():
            log_str += f" - {key}: {value:.4f}"

        logger.info(log_str)

    def _save_final_model(self):
        """Save the final trained model."""
        final_model_path = self.output_dir / "final_model.pt"
        torch.save(self.model.state_dict(), final_model_path)
        logger.info(f"Saved final model: {final_model_path}")

    def _save_training_summary(self):
        """Save training summary and metrics history."""
        summary = {
            "config": self.config,
            "final_epoch": self.current_epoch,
            "total_steps": self.global_step,
            "best_metric": self.best_metric,
            "metrics_history": [
                {
                    "epoch": m.epoch,
                    "step": m.step,
                    "loss": m.loss,
                    "learning_rate": m.learning_rate,
                    "gradient_norm": m.gradient_norm,
                    "memory_used_gb": m.memory_used_gb,
                    "examples_per_second": m.examples_per_second,
                    **m.additional_metrics,
                }
                for m in self.metrics_history
            ],
        }

        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved training summary: {summary_path}")

    def get_memory_usage(self) -> Dict[str | float]:
        """
        Get current memory usage statistics.

        Returns:
            Dictionary with memory statistics in GB
        """
        stats = {}

        if self.device.type == "cuda":
            stats["allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            stats["reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            stats["max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3

        return stats
