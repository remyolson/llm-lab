"""
Ollama Backend for Local Models

This backend provides support for models served through Ollama,
a local model serving system that handles model management and inference.
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import requests

from .base import (
    BackendCapabilities,
    GenerationConfig,
    LocalBackend,
    ModelFormat,
    ModelInfo,
)

logger = logging.getLogger(__name__)


class OllamaBackend(LocalBackend):
    """Backend for Ollama-served models."""

    def __init__(self, base_url: str | None = None):
        super().__init__("ollama")

        # Use configuration system for base URL
        if base_url is None:
            try:
                from ...config.settings import get_settings

                settings = get_settings()
                base_url = settings.network.ollama_base_url
            except ImportError:
                # Fallback to hardcoded default for backward compatibility
                base_url = "http://localhost:11434"

        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def is_available(self) -> bool:
        """Check if Ollama is available and running."""
        try:
            # Get version check timeout from configuration
            try:
                from ...config.settings import get_settings

                settings = get_settings()
                timeout = 5  # Use a fixed short timeout for version check (not configurable)
            except ImportError:
                timeout = 5  # Fallback for backward compatibility

            response = self._session.get(f"{self.base_url}/api/version", timeout=timeout)
            return response.status_code == 200
        except Exception:
            return False

    def get_capabilities(self) -> BackendCapabilities:
        """Get Ollama backend capabilities."""
        return BackendCapabilities(
            streaming=True,
            embeddings=True,
            function_calling=False,
            batch_generation=False,  # Ollama processes one request at a time
            gpu_acceleration=True,  # Ollama handles GPU automatically
            quantization=True,  # Ollama models are typically quantized
            concurrent_models=True,  # Ollama can keep multiple models loaded
        )

    def discover_models(self, search_paths: List[str, Path]) -> List[ModelInfo]:
        """Discover models available through Ollama."""
        discovered = []

        try:
            response = self._session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()

                for model in data.get("models", []):
                    model_info = self._parse_ollama_model(model)
                    if model_info:
                        discovered.append(model_info)

        except Exception as e:
            logger.debug(f"Could not discover Ollama models: {e}")

        return discovered

    def _parse_ollama_model(self, model_data: Dict[str, Any]) -> ModelInfo | None:
        """Parse Ollama model data into ModelInfo."""
        try:
            name = model_data.get("name", "")
            if not name:
                return None

            # Extract size information
            size_mb = None
            if "size" in model_data:
                size_mb = model_data["size"] / (1024 * 1024)

            # Try to parse model details
            details = model_data.get("details", {})

            # Estimate parameters from model name or details
            parameters = None
            if "parameters" in details:
                parameters = int(details["parameters"])
            else:
                # Try to guess from model name
                name_lower = name.lower()
                if "7b" in name_lower:
                    parameters = 7_000_000_000
                elif "13b" in name_lower:
                    parameters = 13_000_000_000
                elif "70b" in name_lower:
                    parameters = 70_000_000_000
                elif "1b" in name_lower:
                    parameters = 1_000_000_000
                elif "3b" in name_lower:
                    parameters = 3_000_000_000

            return ModelInfo(
                name=name,
                path=name,  # For Ollama, the name is the "path"
                format=ModelFormat.OLLAMA,
                size_mb=size_mb,
                parameters=parameters,
                context_length=details.get("context_length"),
                description=f"Ollama model: {name}",
                capabilities=self.get_capabilities(),
                metadata={
                    "digest": model_data.get("digest"),
                    "modified_at": model_data.get("modified_at"),
                    "details": details,
                },
            )

        except Exception as e:
            logger.debug(f"Could not parse Ollama model data: {e}")
            return None

    def can_load_model(self, model_info: ModelInfo) -> bool:
        """Check if this backend can load the given model."""
        return model_info.format == ModelFormat.OLLAMA

    def load_model(self, model_info: ModelInfo, **kwargs) -> str:
        """Load a model through Ollama (just verify it exists)."""
        if not self.is_available():
            raise RuntimeError("Ollama backend is not available")

        model_id = f"ollama_{model_info.name.replace(':', '_')}"

        if model_id in self._loaded_models:
            logger.info(f"Model {model_id} already loaded")
            return model_id

        try:
            # Check if model exists by trying to show it
            # Get API request timeout from configuration
            try:
                from ...config.settings import get_settings

                settings = get_settings()
                timeout = settings.network.api_request_timeout
            except ImportError:
                timeout = 30  # Fallback for backward compatibility

            response = self._session.post(
                f"{self.base_url}/api/show", json={"name": model_info.name}, timeout=timeout
            )

            if response.status_code != 200:
                raise RuntimeError(f"Model {model_info.name} not found in Ollama")

            model_details = response.json()

            # Store model reference (Ollama handles actual loading)
            self._loaded_models[model_id] = {
                "name": model_info.name,
                "details": model_details,
            }
            self._model_metadata[model_id] = model_info

            logger.info(f"Successfully registered {model_info.name} as {model_id}")
            return model_id

        except Exception as e:
            logger.error(f"Failed to load model {model_info.name}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def unload_model(self, model_id: str) -> None:
        """Unload a model (just remove from tracking)."""
        if model_id not in self._loaded_models:
            return

        try:
            # Remove from loaded models (Ollama handles actual unloading)
            del self._loaded_models[model_id]
            if model_id in self._model_metadata:
                del self._model_metadata[model_id]

            logger.info(f"Unloaded model {model_id}")

        except Exception as e:
            logger.error(f"Error unloading model {model_id}: {e}")

    def generate(self, model_id: str, prompt: str, config: GenerationConfig) -> str | Iterator[str]:
        """Generate text using Ollama."""
        if model_id not in self._loaded_models:
            raise ValueError(f"Model {model_id} is not loaded")

        self.validate_generation_config(config)
        model_data = self._loaded_models[model_id]
        model_name = model_data["name"]

        # Prepare request payload
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": config.stream,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "num_predict": config.max_tokens,
            },
        }

        if config.top_k is not None:
            payload["options"]["top_k"] = config.top_k

        if config.stop:
            payload["options"]["stop"] = config.stop

        if config.seed is not None:
            payload["options"]["seed"] = config.seed

        # Add repeat penalty
        payload["options"]["repeat_penalty"] = config.repeat_penalty

        # Add any extra parameters
        payload["options"].update(config.extra_params)

        try:
            if config.stream:
                return self._generate_stream(payload)
            else:
                return self._generate_sync(payload)

        except Exception as e:
            logger.error(f"Generation failed for model {model_id}: {e}")
            raise RuntimeError(f"Generation failed: {e}")

    def _generate_sync(self, payload: Dict[str, Any]) -> str:
        """Generate text synchronously."""
        # Remove stream for sync generation
        payload = payload.copy()
        payload["stream"] = False

        # Get generation timeout from configuration
        try:
            from ...config.settings import get_settings

            settings = get_settings()
            timeout = settings.network.generation_timeout
        except ImportError:
            timeout = 300  # Fallback to 5 minute timeout for backward compatibility

        response = self._session.post(
            f"{self.base_url}/api/generate", json=payload, timeout=timeout
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama generation failed: {response.text}")

        result = response.json()
        return result.get("response", "")

    def _generate_stream(self, payload: Dict[str, Any]) -> Iterator[str]:
        """Generate text with streaming."""
        payload = payload.copy()
        payload["stream"] = True

        try:
            # Get generation timeout from configuration (same as non-streaming)
            try:
                from ...config.settings import get_settings

                settings = get_settings()
                timeout = settings.network.generation_timeout
            except ImportError:
                timeout = 300  # Fallback to 5 minute timeout for backward compatibility

            response = self._session.post(
                f"{self.base_url}/api/generate", json=payload, stream=True, timeout=timeout
            )

            if response.status_code != 200:
                raise RuntimeError(f"Ollama streaming failed: {response.text}")

            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode("utf-8"))

                        if "response" in chunk:
                            yield chunk["response"]

                        # Check if generation is done
                        if chunk.get("done", False):
                            break

                    except json.JSONDecodeError:
                        # Skip invalid JSON lines
                        continue

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise RuntimeError(f"Streaming failed: {e}")

    def get_model_memory_usage(self, model_id: str) -> Dict[str, float]:
        """Get memory usage of a loaded model."""
        if model_id not in self._loaded_models:
            return {"ram": 0.0, "vram": 0.0}

        usage = {"ram": 0.0, "vram": 0.0}

        try:
            # Try to get running models from Ollama
            response = self._session.get(f"{self.base_url}/api/ps")
            if response.status_code == 200:
                running_models = response.json().get("models", [])

                model_data = self._loaded_models[model_id]
                model_name = model_data["name"]

                for running_model in running_models:
                    if running_model.get("name") == model_name:
                        # Get size from running model info
                        size_mb = running_model.get("size", 0) / (1024 * 1024)
                        # Ollama typically uses VRAM when available
                        usage["vram"] = size_mb
                        break
                else:
                    # Model not currently running, estimate from metadata
                    model_info = self._model_metadata.get(model_id)
                    if model_info and model_info.size_mb:
                        usage["ram"] = model_info.size_mb

        except Exception as e:
            logger.warning(f"Could not get memory usage for {model_id}: {e}")

        return usage

    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            # Get model pull timeout from configuration
            try:
                from ...config.settings import get_settings

                settings = get_settings()
                timeout = settings.network.model_pull_timeout
            except ImportError:
                timeout = 1800  # Fallback to 30 minute timeout for backward compatibility

            response = self._session.post(
                f"{self.base_url}/api/pull", json={"name": model_name}, stream=True, timeout=timeout
            )

            if response.status_code != 200:
                logger.error(f"Failed to pull model {model_name}: {response.text}")
                return False

            # Monitor pull progress
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode("utf-8"))

                        if "status" in chunk:
                            logger.info(f"Pull {model_name}: {chunk['status']}")

                        if chunk.get("error"):
                            logger.error(f"Pull error: {chunk['error']}")
                            return False

                    except json.JSONDecodeError:
                        continue

            return True

        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False

    def list_available_models(self) -> List[str]:
        """List models available in Ollama registry."""
        try:
            # This would require access to Ollama's model registry
            # For now, return some common models
            return [
                "llama3.2:1b",
                "llama3.2:3b",
                "llama3.2:8b",
                "mistral:7b",
                "codellama:7b",
                "phi3:mini",
                "gemma:2b",
                "gemma:7b",
            ]
        except Exception as e:
            logger.debug(f"Could not list available models: {e}")
            return []
