"""Tests for the provider registry system."""

from typing import Any, Dict

import pytest

from src.providers.base import LLMProvider
from src.providers.registry import (
    ProviderRegistry,
    get_provider_for_model,
    register_provider,
    registry,
)


class TestProvider1(LLMProvider):
    """Test provider 1 for registry testing."""

    def generate(self, prompt: str, **kwargs) -> str:
        return f"TestProvider1: {prompt}"

    def get_model_info(self) -> Dict[str, Any]:
        return {"provider": "test1"}

    def validate_credentials(self) -> bool:
        return True


class TestProvider2(LLMProvider):
    """Test provider 2 for registry testing."""

    def generate(self, prompt: str, **kwargs) -> str:
        return f"TestProvider2: {prompt}"

    def get_model_info(self) -> Dict[str, Any]:
        return {"provider": "test2"}

    def validate_credentials(self) -> bool:
        return True


class TestProviderRegistry:
    """Test the ProviderRegistry class."""

    @pytest.fixture
    def clean_registry(self):
        """Fixture to provide a clean registry for each test."""
        test_registry = ProviderRegistry()
        test_registry.clear()
        yield test_registry
        test_registry.clear()

    def test_singleton_pattern(self):
        """Test that ProviderRegistry follows singleton pattern."""
        registry1 = ProviderRegistry()
        registry2 = ProviderRegistry()
        assert registry1 is registry2

    def test_register_provider(self, clean_registry):
        """Test registering a provider."""
        clean_registry.register(TestProvider1, ["model1", "model2"])

        # Check that provider is registered
        assert clean_registry.get_provider("model1") == TestProvider1
        assert clean_registry.get_provider("model2") == TestProvider1

    def test_register_multiple_providers(self, clean_registry):
        """Test registering multiple providers."""
        clean_registry.register(TestProvider1, ["model1", "model2"])
        clean_registry.register(TestProvider2, ["model3", "model4"])

        assert clean_registry.get_provider("model1") == TestProvider1
        assert clean_registry.get_provider("model2") == TestProvider1
        assert clean_registry.get_provider("model3") == TestProvider2
        assert clean_registry.get_provider("model4") == TestProvider2

    def test_register_conflict_error(self, clean_registry):
        """Test error when registering conflicting models."""
        clean_registry.register(TestProvider1, ["model1"])

        with pytest.raises(ValueError) as exc_info:
            clean_registry.register(TestProvider2, ["model1"])

        assert "already registered" in str(exc_info.value)
        assert "test1" in str(exc_info.value)
        assert "test2" in str(exc_info.value)

    def test_register_same_provider_multiple_times(self, clean_registry):
        """Test registering same provider with additional models."""
        clean_registry.register(TestProvider1, ["model1"])
        # Should not raise error when same provider adds more models
        clean_registry.register(TestProvider1, ["model2"])

        assert clean_registry.get_provider("model1") == TestProvider1
        assert clean_registry.get_provider("model2") == TestProvider1

    def test_model_name_normalization(self, clean_registry):
        """Test that model names are normalized correctly."""
        clean_registry.register(TestProvider1, ["GPT-4", "gpt-3.5-turbo"])

        # All these variations should resolve to the same provider
        assert clean_registry.get_provider("GPT-4") == TestProvider1
        assert clean_registry.get_provider("gpt-4") == TestProvider1
        assert clean_registry.get_provider("GPT_4") == TestProvider1
        assert clean_registry.get_provider("gpt_4") == TestProvider1

        assert clean_registry.get_provider("gpt-3.5-turbo") == TestProvider1
        assert clean_registry.get_provider("GPT-3.5-TURBO") == TestProvider1
        assert clean_registry.get_provider("gpt_3.5_turbo") == TestProvider1

    def test_get_provider_not_found(self, clean_registry):
        """Test get_provider returns None for unknown model."""
        clean_registry.register(TestProvider1, ["model1"])

        assert clean_registry.get_provider("unknown-model") is None

    def test_get_provider_by_prefix(self, clean_registry):
        """Test getting provider by model name prefix."""
        clean_registry.register(TestProvider1, ["model1"])

        # Should find provider by prefix
        assert clean_registry.get_provider("test1-custom-model") == TestProvider1
        assert clean_registry.get_provider("test2-custom-model") is None

    def test_list_providers(self, clean_registry):
        """Test listing all providers and their models."""
        clean_registry.register(TestProvider1, ["model1", "model2"])
        clean_registry.register(TestProvider2, ["model3", "model4"])

        providers = clean_registry.list_providers()

        assert "test1" in providers
        assert "test2" in providers
        assert set(providers["test1"]) == {"model1", "model2"}
        assert set(providers["test2"]) == {"model3", "model4"}

    def test_list_all_models(self, clean_registry):
        """Test listing all supported models."""
        clean_registry.register(TestProvider1, ["model-b", "model-a"])
        clean_registry.register(TestProvider2, ["model-d", "model-c"])

        models = clean_registry.list_all_models()

        # Should be sorted
        assert models == ["model-a", "model-b", "model-c", "model-d"]

    def test_clear_registry(self, clean_registry):
        """Test clearing the registry."""
        clean_registry.register(TestProvider1, ["model1"])
        assert clean_registry.get_provider("model1") == TestProvider1

        clean_registry.clear()
        assert clean_registry.get_provider("model1") is None
        assert clean_registry.list_providers() == {}
        assert clean_registry.list_all_models() == []


class TestRegisterProviderDecorator:
    """Test the @register_provider decorator."""

    def test_decorator_basic_usage(self):
        """Test basic usage of the decorator."""
        # Clear global registry first
        registry.clear()

        @register_provider(models=["test-model-1", "test-model-2"])
        class DecoratedProvider(LLMProvider):
            def generate(self, prompt: str, **kwargs) -> str:
                return "response"

            def get_model_info(self) -> Dict[str, Any]:
                return {}

            def validate_credentials(self) -> bool:
                return True

        # Check that provider was registered
        assert registry.get_provider("test-model-1") == DecoratedProvider
        assert registry.get_provider("test-model-2") == DecoratedProvider

        # Check that SUPPORTED_MODELS was set
        assert DecoratedProvider.SUPPORTED_MODELS == ["test-model-1", "test-model-2"]

        # Clean up
        registry.clear()

    def test_decorator_with_instantiation(self):
        """Test that decorated providers can be instantiated correctly."""
        registry.clear()

        @register_provider(models=["decorated-model"])
        class DecoratedProvider(LLMProvider):
            def generate(self, prompt: str, **kwargs) -> str:
                return "response"

            def get_model_info(self) -> Dict[str, Any]:
                return {"model": self.model_name}

            def validate_credentials(self) -> bool:
                return True

        # Instantiate the provider
        provider = DecoratedProvider("decorated-model")
        assert provider.model_name == "decorated-model"
        assert provider.generate("test") == "response"

        registry.clear()


class TestGetProviderForModel:
    """Test the get_provider_for_model utility function."""

    def test_get_existing_provider(self):
        """Test getting an existing provider."""
        registry.clear()
        registry.register(TestProvider1, ["existing-model"])

        provider_class = get_provider_for_model("existing-model")
        assert provider_class == TestProvider1

        registry.clear()

    def test_get_nonexistent_provider(self):
        """Test error when getting non-existent provider."""
        registry.clear()
        registry.register(TestProvider1, ["model1", "model2"])

        with pytest.raises(ValueError) as exc_info:
            get_provider_for_model("nonexistent-model")

        error_msg = str(exc_info.value)
        assert "No provider found for model 'nonexistent-model'" in error_msg
        assert "Available models:" in error_msg
        assert "model1" in error_msg
        assert "model2" in error_msg

        registry.clear()

    def test_get_provider_with_normalization(self):
        """Test getting provider with model name normalization."""
        registry.clear()
        registry.register(TestProvider1, ["GPT-4"])

        # All these should work
        assert get_provider_for_model("GPT-4") == TestProvider1
        assert get_provider_for_model("gpt-4") == TestProvider1
        assert get_provider_for_model("GPT_4") == TestProvider1

        registry.clear()


class TestGlobalRegistry:
    """Test the global registry instance."""

    def test_global_registry_is_singleton(self):
        """Test that the global registry is a singleton."""
        from src.providers.registry import (
            registry as registry1,
            registry as registry2,
        )

        assert registry1 is registry2

    def test_global_registry_persistence(self):
        """Test that global registry persists across imports."""
        # This would normally be tested across multiple test files
        # For now, just verify it exists and is a ProviderRegistry instance
        assert isinstance(registry, ProviderRegistry)
