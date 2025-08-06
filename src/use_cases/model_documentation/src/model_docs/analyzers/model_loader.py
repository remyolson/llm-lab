"""Unified model loading interface."""

import logging
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load models from various formats and frameworks."""

    @staticmethod
    def load_model(model_path: Union[str, Path], framework: Optional[str] = None) -> Any:
        """
        Load a model from file.

        Args:
            model_path: Path to the model file
            framework: Framework to use (auto-detected if None)

        Returns:
            Loaded model object
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Auto-detect framework if not specified
        if framework is None:
            framework = ModelLoader._detect_framework_from_path(model_path)

        if framework == "pytorch":
            return ModelLoader._load_pytorch_model(model_path)
        elif framework == "tensorflow":
            return ModelLoader._load_tensorflow_model(model_path)
        elif framework == "onnx":
            return ModelLoader._load_onnx_model(model_path)
        elif framework == "sklearn":
            return ModelLoader._load_sklearn_model(model_path)
        elif framework == "transformers":
            return ModelLoader._load_transformers_model(model_path)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    @staticmethod
    def _detect_framework_from_path(model_path: Path) -> str:
        """Detect framework from file extension or structure."""
        suffix = model_path.suffix.lower()

        if suffix in [".pt", ".pth"]:
            return "pytorch"
        elif suffix in [".h5", ".keras"]:
            return "tensorflow"
        elif suffix == ".onnx":
            return "onnx"
        elif suffix in [".pkl", ".pickle", ".joblib"]:
            return "sklearn"
        elif model_path.is_dir():
            # Check for transformers model structure
            if (model_path / "config.json").exists():
                return "transformers"
            elif (model_path / "saved_model.pb").exists():
                return "tensorflow"

        raise ValueError(f"Cannot detect framework from path: {model_path}")

    @staticmethod
    def _load_pytorch_model(model_path: Path) -> Any:
        """Load a PyTorch model."""
        try:
            import torch

            model = torch.load(model_path, map_location="cpu")
            logger.info(f"Loaded PyTorch model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            raise

    @staticmethod
    def _load_tensorflow_model(model_path: Path) -> Any:
        """Load a TensorFlow/Keras model."""
        try:
            import tensorflow as tf

            if model_path.suffix in [".h5", ".keras"]:
                model = tf.keras.models.load_model(model_path)
            else:
                model = tf.saved_model.load(str(model_path))

            logger.info(f"Loaded TensorFlow model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading TensorFlow model: {e}")
            raise

    @staticmethod
    def _load_onnx_model(model_path: Path) -> Any:
        """Load an ONNX model."""
        try:
            import onnx

            model = onnx.load(str(model_path))
            logger.info(f"Loaded ONNX model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
            raise

    @staticmethod
    def _load_sklearn_model(model_path: Path) -> Any:
        """Load a scikit-learn model."""
        try:
            import joblib

            model = joblib.load(model_path)
            logger.info(f"Loaded scikit-learn model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading scikit-learn model: {e}")
            raise

    @staticmethod
    def _load_transformers_model(model_path: Path) -> Any:
        """Load a Hugging Face Transformers model."""
        try:
            from transformers import AutoModel, AutoTokenizer

            model = AutoModel.from_pretrained(str(model_path))
            logger.info(f"Loaded Transformers model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading Transformers model: {e}")
            raise
