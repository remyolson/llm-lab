#!/usr/bin/env python3
"""
Isolated test script for the Unified Local Provider

This script tests the local provider functionality without importing
the main providers package that has dependencies we might not have.
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


def test_backend_availability():
    """Test which backends are available."""
    logger.info("Testing backend availability...")

    try:
        from providers.local.backends import LlamaCppBackend, OllamaBackend, TransformersBackend

        backends = [
            ("Transformers", TransformersBackend),
            ("LlamaCpp", LlamaCppBackend),
            ("Ollama", OllamaBackend),
        ]

        available_backends = []

        for name, backend_class in backends:
            try:
                backend = backend_class()
                is_available = backend.is_available()
                logger.info(
                    f"{name} Backend: {'✓ Available' if is_available else '✗ Not available'}"
                )

                if is_available:
                    available_backends.append(name)
                    capabilities = backend.get_capabilities()
                    logger.info(
                        f"  Capabilities: Streaming={capabilities.streaming}, "
                        f"GPU={capabilities.gpu_acceleration}, "
                        f"Batch={capabilities.batch_generation}"
                    )

            except Exception as e:
                logger.error(f"{name} Backend error: {e}")

        return available_backends

    except ImportError as e:
        logger.error(f"Import error: {e}")
        return []


def test_model_registry():
    """Test the model registry functionality."""
    logger.info("Testing model registry...")

    try:
        from providers.local.registry import ModelRegistry

        registry = ModelRegistry()

        # Test backend initialization
        available_backends = registry.get_available_backends()
        logger.info(f"Registry backends: {available_backends}")

        # Test model discovery (even if no models found, should not crash)
        discovered = registry.discover_models()
        total_models = sum(len(models) for models in discovered.values())
        logger.info(f"Discovered {total_models} models across {len(discovered)} backends")

        for backend_name, models in discovered.items():
            logger.info(f"  {backend_name}: {len(models)} models")
            for model in models[:3]:  # Show first 3 models
                logger.info(f"    - {model.name} ({model.format.value}, {model.size_mb:.1f}MB)")

        # Test summary
        summary = registry.get_models_summary()
        logger.info(f"Registry summary: {summary}")

        return True

    except Exception as e:
        logger.error(f"Registry error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_resource_manager():
    """Test the resource manager."""
    logger.info("Testing resource manager...")

    try:
        from providers.local.resource_manager import ResourceManager

        manager = ResourceManager()

        # Test system resource detection
        resources = manager.get_system_resources()
        logger.info(f"System Resources:")
        logger.info(
            f"  RAM: {resources.used_ram_mb:.1f}/{resources.total_ram_mb:.1f}MB ({resources.ram_percent:.1f}%)"
        )
        logger.info(f"  GPU: {resources.gpu_type} ({resources.gpu_name})")
        if resources.total_vram_mb > 0:
            logger.info(
                f"  VRAM: {resources.used_vram_mb:.1f}/{resources.total_vram_mb:.1f}MB ({resources.vram_percent:.1f}%)"
            )
        logger.info(f"  CPU: {resources.cpu_cores} cores ({resources.cpu_percent:.1f}% usage)")

        # Test model memory estimation
        test_cases = [
            (100, "gguf"),
            (500, "safetensors"),
            (1000, "ollama"),
        ]

        for size_mb, format_type in test_cases:
            ram_est, vram_est = manager.estimate_model_memory(size_mb, format_type)
            logger.info(
                f"  {size_mb}MB {format_type} model: RAM={ram_est:.1f}MB, VRAM={vram_est:.1f}MB"
            )

        return True

    except Exception as e:
        logger.error(f"Resource manager error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_backend_interfaces():
    """Test the backend interface implementations."""
    logger.info("Testing backend interfaces...")

    try:
        from providers.local.backends.base import (
            BackendCapabilities,
            GenerationConfig,
            ModelFormat,
            ModelInfo,
        )

        # Test data classes
        capabilities = BackendCapabilities(streaming=True, gpu_acceleration=True, quantization=True)
        logger.info(f"BackendCapabilities: {capabilities}")

        model_info = ModelInfo(
            name="test-model",
            path="/fake/path",
            format=ModelFormat.GGUF,
            size_mb=100.0,
            parameters=7000000,
        )
        logger.info(f"ModelInfo: {model_info.name} ({model_info.format.value})")

        gen_config = GenerationConfig(temperature=0.7, max_tokens=100, top_p=0.9)
        logger.info(
            f"GenerationConfig: temp={gen_config.temperature}, max_tokens={gen_config.max_tokens}"
        )

        return True

    except Exception as e:
        logger.error(f"Backend interface error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Testing Unified Local Provider System (Isolated)")
    logger.info("=" * 60)

    tests = [
        ("Backend Interface", test_backend_interfaces),
        ("Backend Availability", test_backend_availability),
        ("Resource Manager", test_resource_manager),
        ("Model Registry", test_model_registry),
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
            import traceback

            traceback.print_exc()

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

    if passed >= 3:  # Allow some tests to fail due to missing dependencies
        logger.info("🎉 Core functionality tests passed! The unified provider system is working.")
    else:
        logger.warning("⚠️  Core tests failed. Check the logs above for details.")

    return passed >= 3


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
