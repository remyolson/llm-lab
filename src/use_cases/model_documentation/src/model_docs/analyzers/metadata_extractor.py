"""Extract metadata from models and training artifacts."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..models import ModelMetadata, PerformanceMetrics, TrainingConfig

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extract metadata from models and associated files."""

    def extract_metadata(
        self, model: Any, model_path: Optional[Path] = None, config_path: Optional[Path] = None
    ) -> ModelMetadata:
        """
        Extract metadata from a model.

        Args:
            model: The model object
            model_path: Path to saved model
            config_path: Path to configuration file

        Returns:
            ModelMetadata object
        """
        from .model_inspector import ModelInspector

        inspector = ModelInspector()
        model_info = inspector.inspect_model(model)

        # Extract basic metadata
        metadata = ModelMetadata(
            name=self._extract_model_name(model, model_path),
            version="1.0.0",
            architecture=model_info.get("architecture", "Unknown"),
            framework=model_info.get("framework", "Unknown"),
            total_parameters=model_info.get("total_parameters", 0),
            trainable_parameters=model_info.get("trainable_parameters", 0),
        )

        # Add configuration if available
        if config_path and config_path.exists():
            config = self._load_config(config_path)
            metadata.description = config.get("description", metadata.description)
            metadata.tags = config.get("tags", metadata.tags)

        # Extract shapes if available
        if "input_shape" in model_info:
            metadata.input_shape = model_info["input_shape"]
        if "output_shape" in model_info:
            metadata.output_shape = model_info["output_shape"]

        return metadata

    def extract_training_config(
        self, config_path: Optional[Path] = None, training_log: Optional[Path] = None
    ) -> TrainingConfig:
        """
        Extract training configuration.

        Args:
            config_path: Path to training configuration
            training_log: Path to training logs

        Returns:
            TrainingConfig object
        """
        config = TrainingConfig()

        if config_path and config_path.exists():
            training_data = self._load_config(config_path)

            config.dataset_name = training_data.get("dataset")
            config.batch_size = training_data.get("batch_size")
            config.learning_rate = training_data.get("learning_rate")
            config.epochs = training_data.get("epochs")
            config.optimizer = training_data.get("optimizer")
            config.loss_function = training_data.get("loss")
            config.hyperparameters = training_data.get("hyperparameters", {})

        if training_log and training_log.exists():
            log_data = self._parse_training_log(training_log)
            config.training_time_hours = log_data.get("training_time")
            config.hardware = log_data.get("hardware")

        return config

    def extract_performance_metrics(
        self, metrics_path: Optional[Path] = None, evaluation_results: Optional[Dict] = None
    ) -> PerformanceMetrics:
        """
        Extract performance metrics.

        Args:
            metrics_path: Path to metrics file
            evaluation_results: Direct evaluation results

        Returns:
            PerformanceMetrics object
        """
        metrics = PerformanceMetrics()

        if evaluation_results:
            metrics.accuracy = evaluation_results.get("accuracy")
            metrics.precision = evaluation_results.get("precision")
            metrics.recall = evaluation_results.get("recall")
            metrics.f1_score = evaluation_results.get("f1_score")
            metrics.loss = evaluation_results.get("loss")
            metrics.custom_metrics = {
                k: v
                for k, v in evaluation_results.items()
                if k not in ["accuracy", "precision", "recall", "f1_score", "loss"]
            }

        if metrics_path and metrics_path.exists():
            saved_metrics = self._load_config(metrics_path)
            for key, value in saved_metrics.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)

        return metrics

    def _extract_model_name(self, model: Any, model_path: Optional[Path]) -> str:
        """Extract or generate model name."""
        if hasattr(model, "name"):
            return model.name
        elif hasattr(model, "__name__"):
            return model.__name__
        elif model_path:
            return model_path.stem
        else:
            return f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _load_config(self, path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        if path.suffix == ".json":
            with open(path, "r") as f:
                return json.load(f)
        elif path.suffix in [".yaml", ".yml"]:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Unsupported config format: {path.suffix}")
            return {}

    def _parse_training_log(self, log_path: Path) -> Dict[str, Any]:
        """Parse training log file."""
        log_data = {}

        try:
            with open(log_path, "r") as f:
                lines = f.readlines()

            # Simple parsing - can be extended based on log format
            for line in lines:
                if "training_time" in line.lower():
                    # Extract training time
                    pass
                elif "gpu" in line.lower() or "cuda" in line.lower():
                    # Extract hardware info
                    pass
        except Exception as e:
            logger.error(f"Error parsing training log: {e}")

        return log_data
