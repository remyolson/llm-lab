"""
Llama.cpp Backend for GGUF Models

This backend provides support for loading and running GGUF/GGML format models
through the llama-cpp-python library, with efficient quantization and
GPU acceleration support.
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


class LlamaCppBackend(LocalBackend):
    """Backend for GGUF/GGML models using llama-cpp-python."""

    def __init__(self):
        super().__init__("llamacpp")
        self._llama_cpp = None

    def is_available(self) -> bool:
        """Check if llama-cpp-python is available."""
        try:
            import llama_cpp

            self._llama_cpp = llama_cpp
            return True
        except ImportError:
            return False

    def get_capabilities(self) -> BackendCapabilities:
        """Get llama.cpp backend capabilities."""
        return BackendCapabilities(
            streaming=True,
            embeddings=True,
            function_calling=False,
            batch_generation=False,  # llama.cpp typically processes one at a time
            gpu_acceleration=self._has_gpu_support(),
            quantization=True,  # GGUF format is inherently quantized
            concurrent_models=False,  # llama.cpp typically loads one model at a time
        )

    def _has_gpu_support(self) -> bool:
        """Check if GPU acceleration is available."""
        # Check CUDA
        try:
            import torch

            if torch.cuda.is_available():
                return True
        except ImportError:
            pass

        # Check Metal (Apple Silicon)
        if platform.system() == "Darwin" and platform.processor() == "arm":
            return True

        return False

    def discover_models(self, search_paths: List[str, Path]) -> List[ModelInfo]:
        """Discover GGUF/GGML models."""
        discovered = []

        for search_path in search_paths:
            path = Path(search_path)
            if not path.exists():
                continue

            # Look for GGUF files
            for gguf_file in path.rglob("*.gguf"):
                model_info = self._analyze_gguf_file(gguf_file)
                if model_info:
                    discovered.append(model_info)

            # Look for GGML files (legacy)
            for ggml_file in path.rglob("*.ggml"):
                model_info = self._analyze_ggml_file(ggml_file)
                if model_info:
                    discovered.append(model_info)

        return discovered

    def _analyze_gguf_file(self, gguf_file: Path) -> ModelInfo | None:
        """Analyze a GGUF file to extract model information."""
        try:
            size_mb = gguf_file.stat().st_size / (1024 * 1024)

            # Try to extract model name from filename
            name = gguf_file.stem
            # Remove common suffixes
            for suffix in [".Q4_K_M", ".Q4_K_S", ".Q5_K_M", ".Q5_K_S", ".Q8_0", ".f16"]:
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    break

            # Estimate parameters from file size (rough approximation)
            parameters = None
            if size_mb < 500:  # < 500MB likely small model
                parameters = int(size_mb * 0.4 * 1_000_000)  # Very rough estimate
            elif size_mb < 2000:  # 500MB-2GB likely 7B class
                parameters = 7_000_000_000
            elif size_mb < 8000:  # 2GB-8GB likely 13B class
                parameters = 13_000_000_000
            else:  # > 8GB likely larger
                parameters = int(size_mb * 0.5 * 1_000_000)

            return ModelInfo(
                name=name,
                path=gguf_file,
                format=ModelFormat.GGUF,
                size_mb=size_mb,
                parameters=parameters,
                context_length=None,  # Will be detected at load time
                description=f"GGUF quantized model ({size_mb:.0f}MB)",
                capabilities=self.get_capabilities(),
            )

        except Exception as e:
            logger.debug(f"Could not analyze GGUF file {gguf_file}: {e}")
            return None

    def _analyze_ggml_file(self, ggml_file: Path) -> ModelInfo | None:
        """Analyze a GGML file to extract model information."""
        try:
            size_mb = ggml_file.stat().st_size / (1024 * 1024)
            name = ggml_file.stem

            return ModelInfo(
                name=name,
                path=ggml_file,
                format=ModelFormat.GGML,
                size_mb=size_mb,
                parameters=None,
                context_length=None,
                description=f"GGML quantized model ({size_mb:.0f}MB)",
                capabilities=self.get_capabilities(),
            )

        except Exception as e:
            logger.debug(f"Could not analyze GGML file {ggml_file}: {e}")
            return None

    def can_load_model(self, model_info: ModelInfo) -> bool:
        """Check if this backend can load the given model."""
        return model_info.format in [ModelFormat.GGUF, ModelFormat.GGML]

    def load_model(self, model_info: ModelInfo, **kwargs) -> str:
        """Load a GGUF/GGML model."""
        if not self.is_available():
            raise RuntimeError("llama-cpp-python backend is not available")

        model_id = f"llamacpp_{model_info.name}"

        if model_id in self._loaded_models:
            logger.info(f"Model {model_id} already loaded")
            return model_id

        try:
            # Prepare llama.cpp parameters
            llama_params = {
                "model_path": str(model_info.path),
                "n_ctx": kwargs.get("n_ctx", 2048),
                "n_batch": kwargs.get("n_batch", 512),
                "n_threads": kwargs.get("n_threads", None),
                "n_gpu_layers": kwargs.get("n_gpu_layers", self._detect_gpu_layers()),
                "use_mmap": kwargs.get("use_mmap", True),
                "use_mlock": kwargs.get("use_mlock", False),
                "f16_kv": kwargs.get("f16_kv", True),
                "verbose": kwargs.get("verbose", False),
                "seed": kwargs.get("seed", -1),
            }

            # Load model
            llama_model = self._llama_cpp.Llama(**llama_params)

            # Store loaded model
            self._loaded_models[model_id] = {
                "model": llama_model,
                "params": llama_params,
            }
            self._model_metadata[model_id] = model_info

            logger.info(f"Successfully loaded {model_info.name} as {model_id}")
            return model_id

        except Exception as e:
            logger.error(f"Failed to load model {model_info.name}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def _detect_gpu_layers(self) -> int:
        """Detect optimal number of GPU layers."""
        try:
            # Try to detect CUDA availability
            import torch

            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory
                # Estimate layers based on memory
                if gpu_mem > 16 * 1024**3:  # 16GB+
                    return 35
                elif gpu_mem > 8 * 1024**3:  # 8GB+
                    return 20
                elif gpu_mem > 4 * 1024**3:  # 4GB+
                    return 10
                else:
                    return 5
        except ImportError:
            pass

        # Check for Metal (Apple Silicon)
        if platform.system() == "Darwin" and platform.processor() == "arm":
            return 1  # Enable Metal acceleration

        # Default to CPU only
        return 0

    def unload_model(self, model_id: str) -> None:
        """Unload a model from memory."""
        if model_id not in self._loaded_models:
            return

        try:
            model_data = self._loaded_models[model_id]

            # llama.cpp models should be garbage collected automatically
            if "model" in model_data:
                del model_data["model"]

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
        llama_model = model_data["model"]

        # Prepare generation arguments
        gen_kwargs = {
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "repeat_penalty": config.repeat_penalty,
            "stream": config.stream,
        }

        if config.top_k is not None:
            gen_kwargs["top_k"] = config.top_k

        if config.stop:
            gen_kwargs["stop"] = config.stop

        if config.seed is not None and config.seed >= 0:
            gen_kwargs["seed"] = config.seed

        # Add any extra parameters
        gen_kwargs.update(config.extra_params)

        try:
            if config.stream:
                return self._generate_stream(llama_model, prompt, gen_kwargs)
            else:
                return self._generate_sync(llama_model, prompt, gen_kwargs)

        except Exception as e:
            logger.error(f"Generation failed for model {model_id}: {e}")
            raise RuntimeError(f"Generation failed: {e}")

    def _generate_sync(self, llama_model, prompt: str, gen_kwargs: Dict[str, Any]) -> str:
        """Generate text synchronously."""
        # Remove stream parameter for sync generation
        gen_kwargs = gen_kwargs.copy()
        gen_kwargs.pop("stream", None)

        response = llama_model(prompt, **gen_kwargs)

        # Extract text from response
        if isinstance(response, dict) and "choices" in response:
            return response["choices"][0]["text"]
        else:
            return str(response)

    def _generate_stream(
        self, llama_model, prompt: str, gen_kwargs: Dict[str, Any]
    ) -> Iterator[str]:
        """Generate text with streaming."""
        # Ensure streaming is enabled
        gen_kwargs = gen_kwargs.copy()
        gen_kwargs["stream"] = True

        try:
            stream = llama_model(prompt, **gen_kwargs)

            for chunk in stream:
                if isinstance(chunk, dict) and "choices" in chunk:
                    token = chunk["choices"][0]["text"]
                    yield token
                else:
                    yield str(chunk)

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise RuntimeError(f"Streaming failed: {e}")

    def get_model_memory_usage(self, model_id: str) -> Dict[str, float]:
        """Get memory usage of a loaded model."""
        if model_id not in self._loaded_models:
            return {"ram": 0.0, "vram": 0.0}

        usage = {"ram": 0.0, "vram": 0.0}

        try:
            # Get model info to estimate memory usage
            model_info = self._model_metadata.get(model_id)
            if model_info and model_info.size_mb:
                # llama.cpp loads models efficiently, estimate based on file size
                model_data = self._loaded_models[model_id]
                n_gpu_layers = model_data["params"].get("n_gpu_layers", 0)

                if n_gpu_layers > 0:
                    # Estimate VRAM usage (rough approximation)
                    gpu_ratio = min(1.0, n_gpu_layers / 35)  # Assume 35 layers for 7B model
                    usage["vram"] = model_info.size_mb * gpu_ratio
                    usage["ram"] = model_info.size_mb * (1 - gpu_ratio) + 100  # + overhead
                else:
                    # CPU only
                    usage["ram"] = model_info.size_mb + 100  # + overhead

        except Exception as e:
            logger.warning(f"Could not get memory usage for {model_id}: {e}")

        return usage
