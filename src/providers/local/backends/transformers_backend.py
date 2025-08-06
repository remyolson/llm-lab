"""
Transformers Backend for Hugging Face Models

This backend provides support for loading and running models through the
Hugging Face Transformers library, supporting various model formats including
safetensors and PyTorch models.
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import logging
import os
import platform
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .base import (
    BackendCapabilities,
    GenerationConfig,
    LocalBackend,
    ModelFormat,
    ModelInfo,
)

logger = logging.getLogger(__name__)


class TransformersBackend(LocalBackend):
    """Backend for Hugging Face Transformers models."""

    def __init__(self):
        super().__init__("transformers")
        self._transformers = None
        self._torch = None
        self._tokenizers = None

    def is_available(self) -> bool:
        """Check if Transformers is available."""
        try:
            import torch
            import transformers

            self._transformers = transformers
            self._torch = torch
            return True
        except ImportError:
            return False

    def get_capabilities(self) -> BackendCapabilities:
        """Get Transformers backend capabilities."""
        return BackendCapabilities(
            streaming=True,
            embeddings=True,
            function_calling=False,  # Model-dependent
            batch_generation=True,
            gpu_acceleration=self._has_gpu_support(),
            quantization=True,
            concurrent_models=True,  # Limited by memory
        )

    def _has_gpu_support(self) -> bool:
        """Check if GPU acceleration is available."""
        if not self._torch:
            return False

        # Check CUDA
        if self._torch.cuda.is_available():
            return True

        # Check Metal (Apple Silicon)
        if hasattr(self._torch.backends, "mps") and self._torch.backends.mps.is_available():
            return True

        return False

    def discover_models(self, search_paths: List[str, Path]) -> List[ModelInfo]:
        """Discover Transformers-compatible models."""
        discovered = []

        for search_path in search_paths:
            path = Path(search_path)
            if not path.exists():
                continue

            # Look for model directories
            for model_dir in path.iterdir():
                if not model_dir.is_dir():
                    continue

                model_info = self._analyze_model_directory(model_dir)
                if model_info:
                    discovered.append(model_info)

        return discovered

    def _analyze_model_directory(self, model_dir: Path) -> ModelInfo | None:
        """Analyze a directory to determine if it contains a Transformers model."""
        # Look for key files that indicate a Transformers model
        config_file = model_dir / "config.json"
        if not config_file.exists():
            return None

        # Check for model files
        model_files = []
        safetensors_files = list(model_dir.glob("*.safetensors"))
        pytorch_files = list(model_dir.glob("*.bin")) + list(model_dir.glob("pytorch_model*.bin"))

        if safetensors_files:
            model_files.extend(safetensors_files)
            format_type = ModelFormat.SAFETENSORS
        elif pytorch_files:
            model_files.extend(pytorch_files)
            format_type = ModelFormat.PYTORCH
        else:
            return None

        # Get model size
        total_size = sum(f.stat().st_size for f in model_files if f.exists())
        size_mb = total_size / (1024 * 1024)

        # Try to read config for more info
        context_length = None
        parameters = None
        try:
            import json

            with open(config_file, "r") as f:
                config = json.load(f)
                context_length = config.get("max_position_embeddings") or config.get("n_positions")
                # Try to estimate parameters from config
                if "n_embd" in config and "n_layer" in config:
                    # Rough estimation for transformer models
                    parameters = config["n_embd"] * config["n_layer"] * 12  # Approximate
        except Exception as e:
            logger.debug(f"Could not read config for {model_dir}: {e}")

        return ModelInfo(
            name=model_dir.name,
            path=model_dir,
            format=format_type,
            size_mb=size_mb,
            parameters=parameters,
            context_length=context_length,
            description=f"Transformers model in {format_type.value} format",
            capabilities=self.get_capabilities(),
        )

    def can_load_model(self, model_info: ModelInfo) -> bool:
        """Check if this backend can load the given model."""
        return model_info.format in [ModelFormat.SAFETENSORS, ModelFormat.PYTORCH]

    def load_model(self, model_info: ModelInfo, **kwargs) -> str:
        """Load a Transformers model."""
        if not self.is_available():
            raise RuntimeError("Transformers backend is not available")

        model_id = f"transformers_{model_info.name}"

        if model_id in self._loaded_models:
            logger.info(f"Model {model_id} already loaded")
            return model_id

        try:
            # Determine device
            device = self._get_best_device(**kwargs)

            # Load tokenizer
            tokenizer = self._transformers.AutoTokenizer.from_pretrained(
                str(model_info.path), trust_remote_code=kwargs.get("trust_remote_code", False)
            )

            # Load model with appropriate settings
            model_kwargs = {
                "trust_remote_code": kwargs.get("trust_remote_code", False),
                "torch_dtype": kwargs.get("torch_dtype", "auto"),
                "device_map": kwargs.get("device_map", "auto") if device != "cpu" else None,
            }

            # Handle quantization
            if kwargs.get("quantization"):
                model_kwargs.update(self._get_quantization_config(kwargs["quantization"]))

            model = self._transformers.AutoModelForCausalLM.from_pretrained(
                str(model_info.path), **model_kwargs
            )

            # Move to device if not using device_map
            if model_kwargs.get("device_map") is None:
                model = model.to(device)

            # Store loaded model
            self._loaded_models[model_id] = {
                "model": model,
                "tokenizer": tokenizer,
                "device": device,
                "config": model_kwargs,
            }
            self._model_metadata[model_id] = model_info

            logger.info(f"Successfully loaded {model_info.name} as {model_id}")
            return model_id

        except Exception as e:
            logger.error(f"Failed to load model {model_info.name}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def _get_best_device(self, **kwargs) -> str:
        """Determine the best device for model loading."""
        if "device" in kwargs:
            return kwargs["device"]

        # Auto-detect best device
        if self._torch.cuda.is_available():
            return "cuda"
        elif hasattr(self._torch.backends, "mps") and self._torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _get_quantization_config(self, quantization: str) -> Dict[str, Any]:
        """Get quantization configuration."""
        config = {}

        if quantization == "int8":
            try:
                from transformers import BitsAndBytesConfig

                config["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                logger.warning("BitsAndBytesConfig not available, skipping 8-bit quantization")

        elif quantization == "int4":
            try:
                from transformers import BitsAndBytesConfig

                config["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self._torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                logger.warning("BitsAndBytesConfig not available, skipping 4-bit quantization")

        return config

    def unload_model(self, model_id: str) -> None:
        """Unload a model from memory."""
        if model_id not in self._loaded_models:
            return

        try:
            model_data = self._loaded_models[model_id]

            # Clear model from memory
            if "model" in model_data:
                del model_data["model"]
            if "tokenizer" in model_data:
                del model_data["tokenizer"]

            # Clear CUDA cache if using GPU
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()

            # Remove from loaded models
            del self._loaded_models[model_id]
            if model_id in self._model_metadata:
                del self._model_metadata[model_id]

            logger.info(f"Unloaded model {model_id}")

        except Exception as e:
            logger.error(f"Error unloading model {model_id}: {e}")

    def generate(self, model_id: str, prompt: str, config: GenerationConfig) -> str | Iterator[str]:
        """Generate text using the loaded model."""
        if model_id not in self._loaded_models:
            raise ValueError(f"Model {model_id} is not loaded")

        self.validate_generation_config(config)
        model_data = self._loaded_models[model_id]
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        device = model_data["device"]

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Prepare generation arguments
        gen_kwargs = {
            "max_new_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "do_sample": config.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }

        if config.top_k is not None:
            gen_kwargs["top_k"] = config.top_k

        if config.stop:
            # Add stop sequences
            stop_token_ids = []
            for stop_seq in config.stop:
                tokens = tokenizer.encode(stop_seq, add_special_tokens=False)
                stop_token_ids.extend(tokens)
            if stop_token_ids:
                gen_kwargs["eos_token_id"] = stop_token_ids

        if config.seed is not None:
            self._torch.manual_seed(config.seed)

        # Add any extra parameters
        gen_kwargs.update(config.extra_params)

        try:
            if config.stream:
                return self._generate_stream(model, tokenizer, inputs, gen_kwargs)
            else:
                return self._generate_sync(model, tokenizer, inputs, gen_kwargs)

        except Exception as e:
            logger.error(f"Generation failed for model {model_id}: {e}")
            raise RuntimeError(f"Generation failed: {e}")

    def _generate_sync(self, model, tokenizer, inputs, gen_kwargs) -> str:
        """Generate text synchronously."""
        with self._torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response

    def _generate_stream(self, model, tokenizer, inputs, gen_kwargs) -> Iterator[str]:
        """Generate text with streaming."""
        # Remove max_new_tokens for streaming (use max_length instead)
        max_new_tokens = gen_kwargs.pop("max_new_tokens", 100)
        input_length = inputs["input_ids"].shape[1]
        gen_kwargs["max_length"] = input_length + max_new_tokens

        # Create streamer
        try:
            import threading

            from transformers import TextIteratorStreamer

            streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
            gen_kwargs["streamer"] = streamer

            # Start generation in a separate thread
            generation_thread = threading.Thread(
                target=model.generate, kwargs={**inputs, **gen_kwargs}
            )
            generation_thread.start()

            # Yield tokens as they're generated
            for token in streamer:
                yield token

            generation_thread.join()

        except ImportError:
            # Fallback to non-streaming if TextIteratorStreamer not available
            logger.warning("TextIteratorStreamer not available, falling back to sync generation")
            response = self._generate_sync(model, tokenizer, inputs, gen_kwargs)
            yield response

    def get_model_memory_usage(self, model_id: str) -> Dict[str, float]:
        """Get memory usage of a loaded model."""
        if model_id not in self._loaded_models:
            return {"ram": 0.0, "vram": 0.0}

        usage = {"ram": 0.0, "vram": 0.0}

        try:
            model_data = self._loaded_models[model_id]
            device = model_data["device"]

            if device == "cpu":
                # Estimate RAM usage (rough approximation)
                model = model_data["model"]
                param_count = sum(p.numel() for p in model.parameters())
                # Assume 4 bytes per parameter (float32)
                usage["ram"] = param_count * 4 / (1024 * 1024)

            elif device in ["cuda", "mps"]:
                if self._torch.cuda.is_available() and device == "cuda":
                    # Get GPU memory usage
                    usage["vram"] = self._torch.cuda.memory_allocated() / (1024 * 1024)
                else:
                    # For MPS or other devices, estimate based on parameters
                    model = model_data["model"]
                    param_count = sum(p.numel() for p in model.parameters())
                    usage["vram"] = param_count * 4 / (1024 * 1024)

        except Exception as e:
            logger.warning(f"Could not get memory usage for {model_id}: {e}")

        return usage
