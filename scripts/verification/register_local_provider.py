#!/usr/bin/env python3
"""
Register the Unified Local Provider with the main registry

This script demonstrates how to register our unified local provider
with the main LLM provider registry system.
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


def register_unified_local_provider():
    """Register the unified local provider with the provider registry."""
    try:
        # Import the unified provider directly (this avoids the providers package)
        import importlib
        import sys

        # Import the unified provider module directly
        spec = importlib.util.spec_from_file_location(
            "unified_provider", "src/providers/local/unified_provider.py"
        )
        unified_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(unified_module)

        UnifiedLocalProvider = unified_module.UnifiedLocalProvider

        # Now import the registry (this may trigger the Google import issue)
        from providers.registry import registry

        # Register the provider with its supported models
        registry.register(UnifiedLocalProvider, UnifiedLocalProvider.SUPPORTED_MODELS)

        logger.info("Successfully registered UnifiedLocalProvider")
        return True

    except Exception as e:
        logger.error(f"Failed to register UnifiedLocalProvider: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_provider_usage():
    """Test using the registered provider."""
    try:
        from providers.registry import get_provider_for_model

        # Test getting the provider for a local model
        test_models = ["local:pythia-70m", "ollama:llama3.2:1b", "transformers:smollm-135m"]

        for model_name in test_models:
            try:
                provider_class = get_provider_for_model(model_name)
                logger.info(f"✓ Found provider for {model_name}: {provider_class.__name__}")
            except Exception as e:
                logger.info(f"✗ No provider for {model_name}: {e}")

        return True

    except Exception as e:
        logger.error(f"Provider usage test failed: {e}")
        return False


def main():
    """Main registration and testing."""
    logger.info("=" * 60)
    logger.info("Registering Unified Local Provider")
    logger.info("=" * 60)

    # Step 1: Register the provider
    logger.info("Step 1: Registering provider...")
    success = register_unified_local_provider()

    if not success:
        logger.error("Failed to register provider. Exiting.")
        return False

    # Step 2: Test provider usage
    logger.info("\nStep 2: Testing provider usage...")
    test_success = test_provider_usage()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("REGISTRATION SUMMARY")
    logger.info("=" * 60)

    if success and test_success:
        logger.info("🎉 Unified Local Provider successfully registered and tested!")
        logger.info("\nYou can now use local models in benchmarks with commands like:")
        logger.info(
            "  python run_benchmarks.py --model local:pythia-70m --dataset truthfulness --limit 10"
        )
        logger.info(
            "  python run_benchmarks.py --model ollama:llama3.2:1b --dataset gsm8k --limit 5"
        )
        return True
    else:
        logger.error("❌ Registration or testing failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
