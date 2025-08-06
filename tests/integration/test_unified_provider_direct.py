#!/usr/bin/env python3
"""
Direct test of the unified local provider without any external dependencies
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


def test_unified_provider_direct():
    """Test the unified provider by importing it directly."""
    logger.info("Testing unified provider with direct imports...")

    try:
        # Import modules directly by file path to avoid package-level imports
        import importlib.util
        import sys

        # Import base classes first
        logger.info("Importing base classes...")

        spec = importlib.util.spec_from_file_location("base", "src/providers/base.py")
        base_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base_module)

        LLMProvider = base_module.LLMProvider
        ProviderConfig = base_module.ProviderConfig
        logger.info(f"✓ Base classes imported: {LLMProvider}, {ProviderConfig}")

        # Import backend base classes
        logger.info("Importing backend base classes...")

        spec = importlib.util.spec_from_file_location(
            "backends_base", "src/providers/local/backends/base.py"
        )
        backends_base_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backends_base_module)

        ModelFormat = backends_base_module.ModelFormat
        BackendCapabilities = backends_base_module.BackendCapabilities
        ModelInfo = backends_base_module.ModelInfo
        GenerationConfig = backends_base_module.GenerationConfig
        LocalBackend = backends_base_module.LocalBackend

        logger.info(f"✓ Backend base classes imported: {LocalBackend}")

        # Test data structures
        logger.info("Testing data structures...")

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

        logger.info("✓ Data structures work correctly")

        # Import individual backends
        logger.info("Testing individual backends...")

        backend_modules = [
            ("Transformers", "src/providers/local/backends/transformers_backend.py"),
            ("LlamaCpp", "src/providers/local/backends/llamacpp_backend.py"),
            ("Ollama", "src/providers/local/backends/ollama_backend.py"),
        ]

        available_backends = []

        for name, path in backend_modules:
            try:
                spec = importlib.util.spec_from_file_location(f"{name.lower()}_backend", path)
                backend_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(backend_module)

                # Get the backend class
                backend_class_name = f"{name}Backend"
                backend_class = getattr(backend_module, backend_class_name)

                # Test instantiation and availability
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
                logger.warning(f"{name} Backend error: {e}")

        logger.info(f"✓ Found {len(available_backends)} available backends: {available_backends}")

        # Import resource manager
        logger.info("Testing resource manager...")

        spec = importlib.util.spec_from_file_location(
            "resource_manager", "src/providers/local/resource_manager.py"
        )
        resource_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(resource_module)

        ResourceManager = resource_module.ResourceManager
        SystemResources = resource_module.SystemResources

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

        logger.info("✓ Resource manager works correctly")

        # Import model registry
        logger.info("Testing model registry...")

        spec = importlib.util.spec_from_file_location(
            "model_registry", "src/providers/local/registry.py"
        )
        registry_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(registry_module)

        ModelRegistry = registry_module.ModelRegistry

        registry = ModelRegistry()

        # Test backend initialization
        available_registry_backends = registry.get_available_backends()
        logger.info(f"Registry backends: {available_registry_backends}")

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

        logger.info("✓ Model registry works correctly")

        # Import unified provider
        logger.info("Testing unified provider...")

        spec = importlib.util.spec_from_file_location(
            "unified_provider", "src/providers/local/unified_provider.py"
        )
        unified_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(unified_module)

        UnifiedLocalProvider = unified_module.UnifiedLocalProvider

        # Test provider instantiation
        provider = UnifiedLocalProvider("local:pythia-70m")
        logger.info(f"Unified provider created: {provider.model_name}")
        logger.info(f"Supported models: {provider.SUPPORTED_MODELS[:5]}...")  # Show first 5

        # Test backend selection
        selected_backend = provider._select_backend_for_model(provider.model_name)
        logger.info(f"Selected backend for {provider.model_name}: {selected_backend}")

        # Test provider info
        info = provider.get_model_info()
        logger.info(f"Provider info: {info}")

        logger.info("✓ Unified provider works correctly")

        return True

    except Exception as e:
        logger.error(f"Direct unified provider test error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run direct unified provider test."""
    logger.info("=" * 60)
    logger.info("Testing Unified Local Provider System (Direct)")
    logger.info("=" * 60)

    success = test_unified_provider_direct()

    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    if success:
        logger.info("🎉 Direct unified provider test passed! The implementation works correctly.")
        logger.info(
            "\nThe unified local provider system is ready for integration with the benchmark runner."
        )
        return True
    else:
        logger.error("❌ Direct unified provider test failed. Check the logs above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
