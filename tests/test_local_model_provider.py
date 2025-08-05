"""
Integration tests for LocalModelProvider.

These tests verify that the local model provider integrates correctly
with the LLM Lab benchmark framework.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.use_cases.local_models.provider import LocalModelProvider
from src.use_cases.local_models.download_helper import ModelDownloader
from src.providers.exceptions import (
    ProviderError,
    ModelNotFoundError,
    ProviderConfigurationError
)


class TestLocalModelProvider:
    """Test suite for LocalModelProvider."""
    
    @pytest.fixture
    def mock_llama_cpp(self):
        """Mock llama-cpp-python module."""
        with patch('src.use_cases.local_models.provider.llama_cpp') as mock:
            # Mock the Llama class
            mock_llama_instance = MagicMock()
            mock_llama_instance.n_ctx.return_value = 2048
            mock_llama_instance.n_embd.return_value = 4096
            mock_llama_instance.n_vocab.return_value = 32000
            
            # Mock generation response
            mock_llama_instance.return_value = {
                "choices": [{"text": "Test response"}]
            }
            
            mock.Llama.return_value = mock_llama_instance
            yield mock
    
    @pytest.fixture
    def temp_model_file(self, tmp_path):
        """Create a temporary model file."""
        model_file = tmp_path / "test-model.gguf"
        model_file.write_bytes(b"dummy model content")
        return str(model_file)
    
    def test_provider_initialization(self, mock_llama_cpp, temp_model_file):
        """Test provider can be initialized with valid model path."""
        provider = LocalModelProvider(
            model_name="custom",
            model_path=temp_model_file
        )
        
        assert provider.model_name == "custom"
        assert provider.model_path == temp_model_file
        assert provider._initialized
    
    def test_provider_without_model_path(self):
        """Test provider raises error when custom model lacks path."""
        with pytest.raises(ProviderConfigurationError) as exc_info:
            LocalModelProvider(model_name="custom")
        
        assert "model_path is required" in str(exc_info.value)
    
    def test_provider_with_invalid_model_path(self, mock_llama_cpp):
        """Test provider raises error for non-existent model file."""
        with pytest.raises(ModelNotFoundError) as exc_info:
            LocalModelProvider(
                model_name="custom",
                model_path="/non/existent/model.gguf"
            )
        
        assert "Model file not found" in str(exc_info.value)
    
    def test_generate_text(self, mock_llama_cpp, temp_model_file):
        """Test text generation."""
        provider = LocalModelProvider(
            model_name="custom",
            model_path=temp_model_file
        )
        
        response = provider.generate("Test prompt")
        
        assert response == "Test response"
        mock_llama_cpp.Llama.return_value.assert_called_once()
    
    def test_generate_with_parameters(self, mock_llama_cpp, temp_model_file):
        """Test generation with custom parameters."""
        provider = LocalModelProvider(
            model_name="custom",
            model_path=temp_model_file,
            temperature=0.5,
            max_tokens=100
        )
        
        response = provider.generate(
            "Test prompt",
            temperature=0.8,
            max_tokens=200
        )
        
        # Verify parameters were passed
        call_args = mock_llama_cpp.Llama.return_value.call_args
        assert call_args[1]["temperature"] == 0.8
        assert call_args[1]["max_tokens"] == 200
    
    def test_streaming_generation(self, mock_llama_cpp, temp_model_file):
        """Test streaming text generation."""
        # Mock streaming response
        mock_stream = [
            {"choices": [{"text": "Hello"}]},
            {"choices": [{"text": " world"}]},
            {"choices": [{"text": "!"}]}
        ]
        mock_llama_cpp.Llama.return_value.return_value = mock_stream
        
        provider = LocalModelProvider(
            model_name="custom",
            model_path=temp_model_file
        )
        
        response = provider.generate("Test prompt", stream=True)
        
        assert response == "Hello world!"
    
    def test_get_model_info(self, mock_llama_cpp, temp_model_file):
        """Test getting model information."""
        provider = LocalModelProvider(
            model_name="custom",
            model_path=temp_model_file,
            n_ctx=4096,
            n_gpu_layers=20
        )
        
        info = provider.get_model_info()
        
        assert info["model_name"] == "custom"
        assert info["provider"] == "localmodel"
        assert info["model_path"] == temp_model_file
        assert info["context_length"] == 4096
        assert info["gpu_layers"] == 20
        assert info["model_loaded"] is True
        assert "hardware" in info
    
    def test_gpu_detection(self, mock_llama_cpp, temp_model_file):
        """Test GPU detection functionality."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value.total_memory = 8 * 1024**3  # 8GB
                
                provider = LocalModelProvider(
                    model_name="custom",
                    model_path=temp_model_file,
                    n_gpu_layers=-1  # Auto-detect
                )
                
                # Check that GPU layers were set based on memory
                init_call = mock_llama_cpp.Llama.call_args
                assert init_call[1]["n_gpu_layers"] == 20  # Expected for 8GB
    
    def test_memory_usage_tracking(self, mock_llama_cpp, temp_model_file):
        """Test memory usage tracking."""
        provider = LocalModelProvider(
            model_name="custom",
            model_path=temp_model_file
        )
        
        memory = provider.get_memory_usage()
        
        assert "ram_used_mb" in memory
        assert "vram_used_mb" in memory
        assert memory["ram_used_mb"] >= 0
        assert memory["vram_used_mb"] >= 0
    
    def test_model_unloading(self, mock_llama_cpp, temp_model_file):
        """Test model can be unloaded from memory."""
        provider = LocalModelProvider(
            model_name="custom",
            model_path=temp_model_file
        )
        
        assert provider._initialized
        assert provider._llama is not None
        
        provider.unload_model()
        
        assert not provider._initialized
        assert provider._llama is None
    
    def test_validate_credentials(self, mock_llama_cpp, temp_model_file):
        """Test credential validation (model file existence)."""
        provider = LocalModelProvider(
            model_name="custom",
            model_path=temp_model_file
        )
        
        assert provider.validate_credentials() is True
        
        # Test with non-existent file
        provider.model_path = "/non/existent/file.gguf"
        assert provider.validate_credentials() is False
    
    def test_batch_generation(self, mock_llama_cpp, temp_model_file):
        """Test batch generation of multiple prompts."""
        provider = LocalModelProvider(
            model_name="custom",
            model_path=temp_model_file
        )
        
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = provider.batch_generate(prompts)
        
        assert len(responses) == 3
        assert all(r == "Test response" for r in responses)
    
    def test_error_handling(self, mock_llama_cpp, temp_model_file):
        """Test error handling during generation."""
        provider = LocalModelProvider(
            model_name="custom",
            model_path=temp_model_file
        )
        
        # Mock generation error
        mock_llama_cpp.Llama.return_value.side_effect = Exception("Generation failed")
        
        with pytest.raises(ProviderError) as exc_info:
            provider.generate("Test prompt")
        
        assert "generation_error" in str(exc_info.value)
        assert "Generation failed" in str(exc_info.value)


class TestModelDownloader:
    """Test suite for ModelDownloader."""
    
    @pytest.fixture
    def downloader(self, tmp_path):
        """Create a downloader with temporary cache directory."""
        return ModelDownloader(cache_dir=str(tmp_path))
    
    def test_downloader_initialization(self, downloader, tmp_path):
        """Test downloader creates cache directory."""
        assert downloader.cache_dir == tmp_path
        assert downloader.cache_dir.exists()
        assert downloader.metadata_file.exists()
    
    def test_model_path_resolution(self, downloader):
        """Test getting model paths."""
        # Test known model
        path = downloader.get_model_path("phi-2")
        assert path is None  # Not downloaded yet
        
        # Test unknown model
        path = downloader.get_model_path("unknown-model")
        assert path is None
    
    def test_is_model_downloaded(self, downloader):
        """Test checking if model is downloaded."""
        assert not downloader.is_model_downloaded("phi-2")
        
        # Create dummy model file
        model_file = downloader.cache_dir / "phi-2.Q4_K_M.gguf"
        model_file.write_bytes(b"dummy")
        
        assert downloader.is_model_downloaded("phi-2")
    
    def test_list_downloaded_models(self, downloader):
        """Test listing downloaded models."""
        assert downloader.list_downloaded_models() == []
        
        # Create dummy model files
        (downloader.cache_dir / "phi-2.Q4_K_M.gguf").write_bytes(b"dummy")
        (downloader.cache_dir / "mistral-7b-instruct-v0.2.Q4_K_M.gguf").write_bytes(b"dummy")
        
        downloaded = downloader.list_downloaded_models()
        assert "phi-2" in downloaded
        assert "mistral-7b" in downloaded
    
    def test_cache_size_calculation(self, downloader):
        """Test calculating total cache size."""
        assert downloader.get_total_cache_size() == 0
        
        # Create dummy model files
        (downloader.cache_dir / "model1.gguf").write_bytes(b"x" * 1000)
        (downloader.cache_dir / "model2.gguf").write_bytes(b"x" * 2000)
        
        assert downloader.get_total_cache_size() == 3000
    
    def test_clear_cache(self, downloader):
        """Test clearing model cache."""
        # Create dummy model files
        model1 = downloader.cache_dir / "phi-2.Q4_K_M.gguf"
        model2 = downloader.cache_dir / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        model1.write_bytes(b"dummy")
        model2.write_bytes(b"dummy")
        
        # Clear specific model
        downloader.clear_cache("phi-2")
        assert not model1.exists()
        assert model2.exists()
        
        # Clear all
        downloader.clear_cache()
        assert not model2.exists()
    
    @patch('requests.get')
    def test_download_with_progress(self, mock_get, downloader):
        """Test downloading with progress tracking."""
        # Mock response
        mock_response = Mock()
        mock_response.headers = {'content-length': '1000'}
        mock_response.iter_content = Mock(return_value=[b"x" * 100] * 10)
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Mock model registry
        with patch.dict('src.use_cases.local_models.download_helper.MODEL_REGISTRY', {
            "test-model": {
                "name": "Test Model",
                "filename": "test.gguf",
                "url": "http://example.com/test.gguf",
                "size": "1KB"
            }
        }):
            path = downloader.download_model("test-model")
            
            assert path.exists()
            assert path.stat().st_size == 1000


class TestIntegrationWithBenchmark:
    """Test integration with the benchmark framework."""
    
    @pytest.fixture
    def mock_provider(self, mock_llama_cpp, temp_model_file):
        """Create a mock local model provider."""
        return LocalModelProvider(
            model_name="custom",
            model_path=temp_model_file
        )
    
    def test_provider_interface_compliance(self, mock_provider):
        """Test provider implements all required methods."""
        # Check required methods exist
        assert hasattr(mock_provider, 'generate')
        assert hasattr(mock_provider, 'get_model_info')
        assert hasattr(mock_provider, 'validate_credentials')
        assert hasattr(mock_provider, 'batch_generate')
        assert hasattr(mock_provider, 'get_default_parameters')
        
        # Check methods are callable
        assert callable(mock_provider.generate)
        assert callable(mock_provider.get_model_info)
        assert callable(mock_provider.validate_credentials)
    
    def test_provider_registration(self):
        """Test provider can be registered with the framework."""
        from src.providers.registry import ProviderRegistry
        
        # Register the local model provider
        ProviderRegistry.register("localmodel", LocalModelProvider)
        
        # Verify registration
        assert "localmodel" in ProviderRegistry.list_providers()
        
        # Test getting provider class
        provider_class = ProviderRegistry.get_provider("localmodel")
        assert provider_class == LocalModelProvider
    
    @patch('src.use_cases.local_models.provider.llama_cpp')
    def test_benchmark_compatibility(self, mock_llama_cpp, temp_model_file):
        """Test provider works with benchmark runner."""
        # Mock llama-cpp
        mock_llama = MagicMock()
        mock_llama.return_value = {"choices": [{"text": "42"}]}
        mock_llama_cpp.Llama.return_value = mock_llama
        
        # Create provider
        provider = LocalModelProvider(
            model_name="custom",
            model_path=temp_model_file
        )
        
        # Simulate benchmark test
        test_prompts = [
            "What is 2+2?",
            "What is the capital of France?",
            "Translate 'hello' to Spanish"
        ]
        
        results = []
        for prompt in test_prompts:
            response = provider.generate(prompt, max_tokens=100)
            results.append({
                "prompt": prompt,
                "response": response,
                "model": provider.model_name
            })
        
        # Verify results
        assert len(results) == 3
        assert all(r["response"] == "42" for r in results)
        assert all(r["model"] == "custom" for r in results)