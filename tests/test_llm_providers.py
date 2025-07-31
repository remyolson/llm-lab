"""Tests for LLM providers."""

from unittest.mock import Mock, patch

import pytest

from config import ConfigurationError
from llm_providers.google import GeminiProvider


class TestGeminiProvider:
    """Test GeminiProvider class."""

    @patch('llm_providers.google.genai')
    def test_init_success(self, mock_genai):
        """Test successful initialization."""
        api_key = "test-api-key"
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider(api_key)

        mock_genai.configure.assert_called_once_with(api_key=api_key)
        mock_genai.GenerativeModel.assert_called_once_with('gemini-1.5-flash')
        assert provider.model == mock_model

    def test_init_no_api_key(self):
        """Test initialization without API key."""
        with pytest.raises(ConfigurationError) as exc_info:
            GeminiProvider("")
        assert "API key is required" in str(exc_info.value)

    @patch('llm_providers.google.genai')
    def test_init_genai_error(self, mock_genai):
        """Test initialization with genai error."""
        mock_genai.configure.side_effect = Exception("Invalid API key")

        with pytest.raises(ConfigurationError) as exc_info:
            GeminiProvider("test-key")
        assert "Failed to initialize Gemini provider" in str(exc_info.value)

    @patch('llm_providers.google.genai')
    def test_generate_success(self, mock_genai):
        """Test successful generation."""
        # Setup
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Generated response"
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider("test-key")
        result = provider.generate("Test prompt")

        assert result == "Generated response"
        mock_model.generate_content.assert_called_once_with("Test prompt")

    @patch('llm_providers.google.genai')
    def test_generate_empty_prompt(self, mock_genai):
        """Test generation with empty prompt."""
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider("test-key")
        result = provider.generate("")

        assert result == "Error: Empty prompt provided"
        mock_model.generate_content.assert_not_called()

    @patch('llm_providers.google.genai')
    def test_generate_no_response_text(self, mock_genai):
        """Test generation when response has no text."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = None
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider("test-key")
        result = provider.generate("Test prompt")

        assert result == "Error: No text in model response"

    @patch('llm_providers.google.genai')
    def test_generate_api_error(self, mock_genai):
        """Test generation with API error."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("Invalid API_KEY")
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider("test-key")
        result = provider.generate("Test prompt")

        assert "Error: Invalid API key" in result

    @patch('llm_providers.google.genai')
    def test_generate_quota_error(self, mock_genai):
        """Test generation with quota exceeded error."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("Quota exceeded")
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider("test-key")
        result = provider.generate("Test prompt")

        assert "Error: API quota exceeded" in result

    @patch('llm_providers.google.genai')
    def test_generate_timeout_error(self, mock_genai):
        """Test generation with timeout error."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("Request timeout")
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider("test-key")
        result = provider.generate("Test prompt")

        assert "Error: Network connection failed" in result

    @patch('llm_providers.google.genai')
    def test_generate_safety_filter(self, mock_genai):
        """Test generation blocked by safety filter."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("Safety filter blocked")
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider("test-key")
        result = provider.generate("Test prompt")

        assert "Error: Content was blocked by safety filters" in result
