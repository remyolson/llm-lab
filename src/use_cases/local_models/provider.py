"""
Local Model Provider for LLM Lab

This module implements a provider for running local models using llama-cpp-python.
It supports GGUF format models with configurable quantization and hardware acceleration.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .providers.base import LLMProvider, ProviderConfig
from .providers.exceptions import (
    ModelNotFoundError,
    ProviderConfigurationError,
    ProviderError,
)

logger = logging.getLogger(__name__)


@dataclass
class LocalModelConfig(ProviderConfig):
    """Extended configuration for local models."""

    model_path: str = ""
    n_ctx: int = 2048  # Context window size
    n_batch: int = 512  # Batch size for prompt processing
    n_threads: Optional[int] = None  # CPU threads (None = auto)
    n_gpu_layers: int = -1  # GPU layers (-1 = auto, 0 = CPU only)
    use_mmap: bool = True  # Memory-mapped files
    use_mlock: bool = False  # Lock model in RAM
    f16_kv: bool = True  # Use half-precision for key/value cache
    verbose: bool = False
    seed: int = -1  # Random seed (-1 = random)


class LocalModelProvider(LLMProvider):
    """
    Provider for local models using llama-cpp-python.

    Supports popular open-source models in GGUF format including:
    - Llama 2 (7B, 13B)
    - Mistral 7B
    - Phi-2
    - And other GGUF-compatible models
    """

    # Popular model configurations
    SUPPORTED_MODELS = [
        "llama-2-7b",
        "llama-2-13b",
        "mistral-7b",
        "phi-2",
        "custom",  # Allow custom model paths
    ]

    # Default model paths (can be overridden)
    DEFAULT_MODEL_PATHS = {
        "llama-2-7b": "models/llama-2-7b-chat.Q4_K_M.gguf",
        "llama-2-13b": "models/llama-2-13b-chat.Q4_K_M.gguf",
        "mistral-7b": "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "phi-2": "models/phi-2.Q4_K_M.gguf",
    }

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the local model provider.

        Args:
            model_name: Name of the model or "custom" for custom paths
            model_path: Path to the GGUF model file (required if model_name is "custom")
            n_gpu_layers: Number of layers to offload to GPU (-1 for auto, 0 for CPU only)
            n_ctx: Context window size
            **kwargs: Additional configuration parameters
        """
        # Initialize llama_cpp instance variable
        self._llama = None
        self._llama_cpp = None

        # Call parent constructor
        super().__init__(model_name, **kwargs)

        # Set up model path
        if model_name == "custom":
            if "model_path" not in kwargs:
                raise ProviderConfigurationError(
                    provider_name="localmodel",
                    config_issue="model_path is required when using custom model",
                )
            self.model_path = kwargs["model_path"]
        else:
            # Use default path or provided override
            self.model_path = kwargs.get("model_path", self.DEFAULT_MODEL_PATHS.get(model_name, ""))

        # Initialize the model
        self._initialize_model()

    def _parse_config(self, kwargs: Dict[str, Any]) -> LocalModelConfig:
        """Parse and validate local model configuration."""
        # Extract known local model parameters
        local_params = {
            "model_path",
            "n_ctx",
            "n_batch",
            "n_threads",
            "n_gpu_layers",
            "use_mmap",
            "use_mlock",
            "f16_kv",
            "verbose",
            "seed",
        }

        config_params = {}
        additional_params = {}

        # Also include base parameters
        base_params = {
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "timeout",
            "max_retries",
            "retry_delay",
        }

        for key, value in kwargs.items():
            if key in local_params or key in base_params:
                config_params[key] = value
            else:
                additional_params[key] = value

        if additional_params:
            config_params["additional_params"] = additional_params

        return LocalModelConfig(**config_params)

    def _initialize_model(self):
        """Initialize the local model."""
        try:
            # Import llama-cpp-python
            import llama_cpp

            self._llama_cpp = llama_cpp

        except ImportError:
            raise ProviderConfigurationError(
                provider_name="localmodel",
                config_issue="llama-cpp-python is not installed. Run: pip install llama-cpp-python",
            )

        # Check if model file exists
        if not os.path.exists(self.model_path):
            raise ModelNotFoundError(
                model_name=self.model_name,
                provider_name="localmodel",
                details=f"Model file not found at: {self.model_path}",
            )

        try:
            # Initialize model with configuration
            config = self.config

            # Detect GPU availability
            n_gpu_layers = config.n_gpu_layers
            if n_gpu_layers == -1:
                # Auto-detect GPU layers based on available memory
                n_gpu_layers = self._detect_gpu_layers()

            logger.info(f"Loading model from {self.model_path}")
            logger.info(f"Using {n_gpu_layers} GPU layers")

            self._llama = self._llama_cpp.Llama(
                model_path=self.model_path,
                n_ctx=config.n_ctx,
                n_batch=config.n_batch,
                n_threads=config.n_threads,
                n_gpu_layers=n_gpu_layers,
                use_mmap=config.use_mmap,
                use_mlock=config.use_mlock,
                f16_kv=config.f16_kv,
                verbose=config.verbose,
                seed=config.seed if config.seed != -1 else None,
            )

            self._initialized = True
            logger.info(f"Successfully loaded {self.model_name} model")

        except Exception as e:
            raise ProviderError(
                provider_name="localmodel",
                error_type="initialization_error",
                details=f"Failed to initialize model: {e!s}",
            )

    def _detect_gpu_layers(self) -> int:
        """
        Detect optimal number of GPU layers based on available hardware.

        Returns:
            Number of layers to offload to GPU
        """
        try:
            # Try to detect CUDA availability
            import torch

            if torch.cuda.is_available():
                # Get available GPU memory
                gpu_mem = torch.cuda.get_device_properties(0).total_memory
                # Estimate layers based on memory (rough estimate)
                if gpu_mem > 16 * 1024**3:  # 16GB+
                    return 35  # Most layers for 7B models
                elif gpu_mem > 8 * 1024**3:  # 8GB+
                    return 20
                elif gpu_mem > 4 * 1024**3:  # 4GB+
                    return 10
                else:
                    return 5
        except ImportError:
            pass

        # Check for Metal (Apple Silicon)
        try:
            import platform

            if platform.system() == "Darwin" and platform.processor() == "arm":
                return 1  # Enable Metal acceleration
        except:
            pass

        # Default to CPU only
        return 0

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using the local model.

        Args:
            prompt: The input prompt
            stream: Whether to stream the response
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        if not self._initialized or self._llama is None:
            raise ProviderError(
                provider_name="localmodel",
                error_type="not_initialized",
                details="Model not initialized. Call initialize() first.",
            )

        # Merge kwargs with config
        generation_params = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k) if self.config.top_k else 40,
            "stream": kwargs.get("stream", False),
            "stop": kwargs.get("stop", None),
            "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
        }

        try:
            start_time = time.time()

            if generation_params["stream"]:
                # Handle streaming response
                return self._generate_stream(prompt, generation_params)
            else:
                # Regular generation
                response = self._llama(prompt, **generation_params)

                generation_time = time.time() - start_time
                logger.debug(f"Generation completed in {generation_time:.2f}s")

                # Extract text from response
                if isinstance(response, dict) and "choices" in response:
                    return response["choices"][0]["text"]
                else:
                    return str(response)

        except Exception as e:
            logger.error(f"Generation error: {e!s}")
            raise ProviderError(
                provider_name="localmodel",
                error_type="generation_error",
                details=f"Failed to generate response: {e!s}",
            )

    def _generate_stream(self, prompt: str, params: Dict[str, Any]) -> str:
        """
        Generate a streaming response.

        Args:
            prompt: The input prompt
            params: Generation parameters

        Returns:
            Complete generated text
        """
        # Remove stream parameter for the actual call
        params = params.copy()
        params.pop("stream")

        # Collect streamed tokens
        tokens = []

        try:
            # Create stream
            stream = self._llama(prompt, stream=True, **params)

            # Collect tokens
            for output in stream:
                if isinstance(output, dict) and "choices" in output:
                    token = output["choices"][0]["text"]
                    tokens.append(token)
                    # Could yield token here for real-time streaming

            return "".join(tokens)

        except Exception as e:
            logger.error(f"Streaming error: {e!s}")
            raise ProviderError(
                provider_name="localmodel",
                error_type="streaming_error",
                details=f"Failed during streaming: {e!s}",
            )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            "model_name": self.model_name,
            "provider": "localmodel",
            "model_path": self.model_path,
            "context_length": self.config.n_ctx,
            "gpu_layers": self.config.n_gpu_layers,
        }

        if self._llama is not None:
            # Add runtime information
            info.update(
                {
                    "model_loaded": True,
                    "n_ctx": self._llama.n_ctx(),
                    "n_embd": self._llama.n_embd(),
                    "n_vocab": self._llama.n_vocab(),
                }
            )
        else:
            info["model_loaded"] = False

        # Add hardware info
        info["hardware"] = self._get_hardware_info()

        return info

    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware acceleration information."""
        hardware = {
            "cpu_threads": self.config.n_threads or "auto",
            "gpu_acceleration": False,
            "gpu_type": "none",
        }

        # Check for CUDA
        try:
            import torch

            if torch.cuda.is_available():
                hardware["gpu_acceleration"] = True
                hardware["gpu_type"] = "cuda"
                hardware["gpu_name"] = torch.cuda.get_device_name(0)
                hardware["gpu_memory"] = (
                    f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
                )
        except ImportError:
            pass

        # Check for Metal
        try:
            import platform

            if platform.system() == "Darwin" and platform.processor() == "arm":
                hardware["gpu_acceleration"] = True
                hardware["gpu_type"] = "metal"
                hardware["gpu_name"] = "Apple Silicon"
        except:
            pass

        return hardware

    def validate_credentials(self) -> bool:
        """
        Validate that the model can be loaded.

        For local models, this checks that the model file exists
        and can be accessed.
        """
        # Check model file exists
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            return False

        # Check file is readable
        if not os.access(self.model_path, os.R_OK):
            logger.error(f"Model file not readable: {self.model_path}")
            return False

        # Check llama-cpp-python is installed
        try:
            import llama_cpp

            return True
        except ImportError:
            logger.error("llama-cpp-python is not installed")
            return False

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage of the model.

        Returns:
            Dictionary with memory usage in MB
        """
        memory = {"ram_used_mb": 0, "vram_used_mb": 0}

        if self._llama is None:
            return memory

        try:
            import psutil

            process = psutil.Process()
            memory["ram_used_mb"] = process.memory_info().rss / 1024 / 1024
        except ImportError:
            pass

        # Try to get GPU memory usage
        try:
            import torch

            if torch.cuda.is_available():
                memory["vram_used_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        except ImportError:
            pass

        return memory

    def unload_model(self):
        """Unload the model from memory."""
        if self._llama is not None:
            del self._llama
            self._llama = None
            self._initialized = False
            logger.info(f"Unloaded {self.model_name} model")

    def __del__(self):
        """Cleanup when provider is destroyed."""
        self.unload_model()
