"""Model inspection and analysis utilities."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class ModelInspector:
    """Base class for model inspection and analysis."""

    def __init__(self):
        """Initialize the model inspector."""
        self.supported_frameworks = ["pytorch", "tensorflow", "onnx", "sklearn", "transformers"]

    def inspect_model(self, model: Any, framework: Optional[str] = None) -> Dict[str, Any]:
        """
        Inspect a model and extract its properties.

        Args:
            model: The model to inspect
            framework: The framework used (auto-detected if None)

        Returns:
            Dictionary containing model properties
        """
        if framework is None:
            framework = self.detect_framework(model)

        if framework == "pytorch":
            return self._inspect_pytorch_model(model)
        elif framework == "tensorflow":
            return self._inspect_tensorflow_model(model)
        elif framework == "onnx":
            return self._inspect_onnx_model(model)
        elif framework == "sklearn":
            return self._inspect_sklearn_model(model)
        elif framework == "transformers":
            return self._inspect_transformers_model(model)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    def detect_framework(self, model: Any) -> str:
        """
        Detect the framework of a model.

        Args:
            model: The model to analyze

        Returns:
            Framework name
        """
        model_type = type(model).__module__

        if "torch" in model_type:
            return "pytorch"
        elif "tensorflow" in model_type or "keras" in model_type:
            return "tensorflow"
        elif "onnx" in model_type:
            return "onnx"
        elif "sklearn" in model_type:
            return "sklearn"
        elif "transformers" in model_type:
            return "transformers"
        else:
            raise ValueError(f"Cannot detect framework for model type: {model_type}")

    def _inspect_pytorch_model(self, model: Any) -> Dict[str, Any]:
        """Inspect a PyTorch model."""
        import torch

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        layers = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                layers.append(
                    {
                        "name": name,
                        "type": module.__class__.__name__,
                        "parameters": sum(p.numel() for p in module.parameters()),
                    }
                )

        return {
            "framework": "pytorch",
            "architecture": model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "layers": layers,
            "device": str(next(model.parameters()).device) if total_params > 0 else "cpu",
        }

    def _inspect_tensorflow_model(self, model: Any) -> Dict[str, Any]:
        """Inspect a TensorFlow/Keras model."""
        import tensorflow as tf

        if hasattr(model, "summary"):
            # Keras model
            total_params = model.count_params()
            trainable_params = sum(
                [tf.keras.backend.count_params(w) for w in model.trainable_weights]
            )

            layers = []
            for layer in model.layers:
                layers.append(
                    {
                        "name": layer.name,
                        "type": layer.__class__.__name__,
                        "parameters": layer.count_params(),
                        "output_shape": str(layer.output_shape),
                    }
                )

            return {
                "framework": "tensorflow",
                "architecture": model.__class__.__name__,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "layers": layers,
                "input_shape": str(model.input_shape) if hasattr(model, "input_shape") else None,
                "output_shape": str(model.output_shape) if hasattr(model, "output_shape") else None,
            }
        else:
            # TensorFlow model
            return {
                "framework": "tensorflow",
                "architecture": model.__class__.__name__,
                "type": "TensorFlow Model",
            }

    def _inspect_onnx_model(self, model: Any) -> Dict[str, Any]:
        """Inspect an ONNX model."""
        import onnx

        graph = model.graph

        # Count parameters
        total_params = 0
        for initializer in graph.initializer:
            shape = [dim for dim in initializer.dims]
            total_params += np.prod(shape) if shape else 1

        # Get input/output info
        inputs = [
            {"name": inp.name, "shape": [dim.dim_value for dim in inp.type.tensor_type.shape.dim]}
            for inp in graph.input
        ]

        outputs = [
            {"name": out.name, "shape": [dim.dim_value for dim in out.type.tensor_type.shape.dim]}
            for out in graph.output
        ]

        return {
            "framework": "onnx",
            "architecture": "ONNX Model",
            "total_parameters": total_params,
            "inputs": inputs,
            "outputs": outputs,
            "nodes": len(graph.node),
            "initializers": len(graph.initializer),
        }

    def _inspect_sklearn_model(self, model: Any) -> Dict[str, Any]:
        """Inspect a scikit-learn model."""
        model_type = model.__class__.__name__

        info = {
            "framework": "sklearn",
            "architecture": model_type,
            "parameters": model.get_params() if hasattr(model, "get_params") else {},
        }

        # Add specific attributes based on model type
        if hasattr(model, "n_features_in_"):
            info["n_features"] = model.n_features_in_
        if hasattr(model, "n_classes_"):
            info["n_classes"] = model.n_classes_
        if hasattr(model, "feature_importances_"):
            info["feature_importances"] = model.feature_importances_.tolist()

        return info

    def _inspect_transformers_model(self, model: Any) -> Dict[str, Any]:
        """Inspect a Hugging Face Transformers model."""
        from transformers import PreTrainedModel

        config = model.config if hasattr(model, "config") else {}

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "framework": "transformers",
            "architecture": model.__class__.__name__,
            "model_type": config.model_type if hasattr(config, "model_type") else None,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "hidden_size": config.hidden_size if hasattr(config, "hidden_size") else None,
            "num_layers": config.num_hidden_layers
            if hasattr(config, "num_hidden_layers")
            else None,
            "num_attention_heads": config.num_attention_heads
            if hasattr(config, "num_attention_heads")
            else None,
            "vocab_size": config.vocab_size if hasattr(config, "vocab_size") else None,
        }

    def count_parameters(self, model: Any, framework: Optional[str] = None) -> Dict[str, int]:
        """
        Count model parameters.

        Args:
            model: The model to analyze
            framework: The framework used

        Returns:
            Dictionary with parameter counts
        """
        info = self.inspect_model(model, framework)
        return {
            "total_parameters": info.get("total_parameters", 0),
            "trainable_parameters": info.get("trainable_parameters", 0),
            "non_trainable_parameters": info.get("total_parameters", 0)
            - info.get("trainable_parameters", 0),
        }

    def analyze_architecture(self, model: Any, framework: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze model architecture in detail.

        Args:
            model: The model to analyze
            framework: The framework used

        Returns:
            Detailed architecture analysis
        """
        info = self.inspect_model(model, framework)

        analysis = {
            "framework": info.get("framework"),
            "architecture_type": info.get("architecture"),
            "depth": len(info.get("layers", [])),
            "parameter_distribution": self._analyze_parameter_distribution(info),
        }

        return analysis

    def _analyze_parameter_distribution(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how parameters are distributed across layers."""
        layers = model_info.get("layers", [])
        if not layers:
            return {}

        layer_types = {}
        for layer in layers:
            layer_type = layer.get("type", "unknown")
            if layer_type not in layer_types:
                layer_types[layer_type] = {"count": 0, "parameters": 0}
            layer_types[layer_type]["count"] += 1
            layer_types[layer_type]["parameters"] += layer.get("parameters", 0)

        return layer_types
