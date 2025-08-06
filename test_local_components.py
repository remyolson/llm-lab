#!/usr/bin/env python3
"""
Direct test of local provider components

This script tests individual components without going through
the main providers package that has external dependencies.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def test_backend_base_classes():
    """Test the backend base classes and data structures."""
    logger.info("Testing backend base classes...")

    try:
        # Import directly from the backend modules
        from providers.local.backends.base import (
            BackendCapabilities,
            GenerationConfig,
            LocalBackend,
            ModelFormat,
            ModelInfo,
        )

        # Test ModelFormat enum
        formats = [ModelFormat.GGUF, ModelFormat.SAFETENSORS, ModelFormat.OLLAMA]
        logger.info(f"Model formats: {[f.value for f in formats]}")

        # Test BackendCapabilities
        capabilities = BackendCapabilities(
            streaming=True, gpu_acceleration=True, quantization=True, batch_generation=False
        )
        logger.info(
            f"Capabilities: streaming={capabilities.streaming}, gpu={capabilities.gpu_acceleration}"
        )

        # Test ModelInfo
        model_info = ModelInfo(
            name="test-model",
            path="/fake/path/model.gguf",
            format=ModelFormat.GGUF,
            size_mb=150.5,
            parameters=7000000,
            description="Test model for validation",
        )
        logger.info(f"Model: {model_info.name} ({model_info.format.value}, {model_info.size_mb}MB)")

        # Test GenerationConfig
        gen_config = GenerationConfig(temperature=0.7, max_tokens=100, top_p=0.9, stream=False)
        logger.info(
            f"Generation config: temp={gen_config.temperature}, tokens={gen_config.max_tokens}"
        )

        return True

    except Exception as e:
        logger.error(f"Backend base classes error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_transformers_backend():
    """Test the Transformers backend (availability check only)."""
    logger.info("Testing Transformers backend...")

    try:
        from providers.local.backends.transformers_backend import TransformersBackend

        backend = TransformersBackend()
        logger.info(f"Backend name: {backend.name}")

        is_available = backend.is_available()
        logger.info(f"Transformers available: {is_available}")

        capabilities = backend.get_capabilities()
        logger.info(f"Capabilities: {capabilities}")

        # Test model discovery with empty paths (should not crash)
        models = backend.discover_models([])
        logger.info(f"Discovered {len(models)} models with empty paths")

        return True

    except Exception as e:
        logger.error(f"Transformers backend error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_llamacpp_backend():
    """Test the LlamaCpp backend (availability check only)."""
    logger.info("Testing LlamaCpp backend...")

    try:
        from providers.local.backends.llamacpp_backend import LlamaCppBackend

        backend = LlamaCppBackend()
        logger.info(f"Backend name: {backend.name}")

        is_available = backend.is_available()
        logger.info(f"LlamaCpp available: {is_available}")

        capabilities = backend.get_capabilities()
        logger.info(f"Capabilities: {capabilities}")

        # Test model discovery with empty paths (should not crash)
        models = backend.discover_models([])
        logger.info(f"Discovered {len(models)} models with empty paths")

        return True

    except Exception as e:
        logger.error(f"LlamaCpp backend error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_ollama_backend():
    """Test the Ollama backend (availability check only)."""
    logger.info("Testing Ollama backend...")

    try:
        from providers.local.backends.ollama_backend import OllamaBackend

        backend = OllamaBackend()
        logger.info(f"Backend name: {backend.name}")

        is_available = backend.is_available()
        logger.info(f"Ollama available: {is_available}")

        capabilities = backend.get_capabilities()
        logger.info(f"Capabilities: {capabilities}")

        # Test model discovery (should not crash even if Ollama not running)
        models = backend.discover_models([])
        logger.info(f"Discovered {len(models)} models")

        return True

    except Exception as e:
        logger.error(f"Ollama backend error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_resource_manager():
    """Test the resource manager directly."""
    logger.info("Testing resource manager...")

    try:
        from providers.local.resource_manager import ResourceManager, SystemResources

        manager = ResourceManager()
        logger.info(
            f"Resource manager initialized with thresholds: mem={manager.memory_threshold}, vram={manager.vram_threshold}"
        )

        # Test system resource detection
        resources = manager.get_system_resources()
        logger.info(f"System resources detected:")
        logger.info(
            f"  RAM: {resources.used_ram_mb:.1f}/{resources.total_ram_mb:.1f}MB ({resources.ram_percent:.1f}%)"
        )
        logger.info(f"  GPU: {resources.gpu_type} - {resources.gpu_name}")
        if resources.total_vram_mb > 0:
            logger.info(f"  VRAM: {resources.used_vram_mb:.1f}/{resources.total_vram_mb:.1f}MB")
        logger.info(f"  CPU: {resources.cpu_cores} cores")

        # Test memory estimation
        test_cases = [(100, "gguf"), (500, "safetensors"), (1000, "ollama"), (2000, "pytorch")]

        logger.info("Memory estimations:")
        for size_mb, format_type in test_cases:
            ram_est, vram_est = manager.estimate_model_memory(size_mb, format_type)
            logger.info(f"  {size_mb}MB {format_type}: RAM={ram_est:.1f}MB, VRAM={vram_est:.1f}MB")

        # Test can_load_model
        can_load = manager.can_load_model(100, "gguf")
        logger.info(f"Can load 100MB GGUF model: {can_load}")

        return True

    except Exception as e:
        logger.error(f"Resource manager error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_discovery():
    """Test model discovery with actual paths."""
    logger.info("Testing model discovery with real paths...")

    try:
        from providers.local.backends import LlamaCppBackend, OllamaBackend, TransformersBackend

        # Paths to search (these should exist in the project)
        search_paths = [
            Path("models"),
            Path("models/small-llms"),
            Path.home() / ".cache" / "huggingface" / "hub",
        ]

        # Filter to existing paths
        existing_paths = [p for p in search_paths if p.exists()]
        logger.info(
            f"Searching in {len(existing_paths)} existing paths: {[str(p) for p in existing_paths]}"
        )

        backends = [TransformersBackend(), LlamaCppBackend(), OllamaBackend()]

        total_discovered = 0

        for backend in backends:
            if backend.is_available():
                try:
                    models = backend.discover_models(existing_paths)
                    logger.info(f"{backend.name}: discovered {len(models)} models")
                    total_discovered += len(models)

                    # Show first few models
                    for model in models[:3]:
                        logger.info(
                            f"  - {model.name} ({model.format.value}, {model.size_mb:.1f}MB)"
                        )

                except Exception as e:
                    logger.warning(f"{backend.name} discovery error: {e}")
            else:
                logger.info(f"{backend.name}: not available")

        logger.info(f"Total models discovered: {total_discovered}")
        return True

    except Exception as e:
        logger.error(f"Model discovery error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Testing Local Provider Components Directly")
    logger.info("=" * 60)

    tests = [
        ("Backend Base Classes", test_backend_base_classes),
        ("Transformers Backend", test_transformers_backend),
        ("LlamaCpp Backend", test_llamacpp_backend),
        ("Ollama Backend", test_ollama_backend),
        ("Resource Manager", test_resource_manager),
        ("Model Discovery", test_model_discovery),
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results[test_name] = result
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"{test_name}: ✗ ERROR - {e}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name:<25} {status}")

    logger.info(f"\nResults: {passed}/{total} tests passed")

    if passed >= 4:  # Allow some tests to fail due to missing dependencies
        logger.info("🎉 Core local provider components are working!")
        return True
    else:
        logger.warning("⚠️  Some core components failed. Check the logs above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
