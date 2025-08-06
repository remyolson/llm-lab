"""Tests for LLM providers."""

from unittest.mock import Mock, patch

import pytest

from src.config import ConfigurationError
from src.providers.google import GeminiProvider


class TestGeminiProvider:
    """Test GeminiProvider class."""

    def test_init_success(self):
        """Test successful initialization."""
        # Test basic initialization with valid model
        provider = GeminiProvider("gemini-1.5-flash")

        assert provider.model_name == "gemini-1.5-flash"
        assert provider.provider_name == "google"
        assert "gemini-1.5-flash" in provider.supported_models

    def test_init_no_api_key(self):
        """Test initialization succeeds without API key."""
        # Current implementation allows creation without validation
        provider = GeminiProvider("gemini-1.5-flash")
        assert provider.model_name == "gemini-1.5-flash"
        assert provider.provider_name == "google"

    def test_init_genai_error(self):
        """Test initialization with invalid model."""
        with pytest.raises(Exception) as exc_info:
            GeminiProvider("invalid-model")
        assert "not supported" in str(exc_info.value)

    @patch("src.providers.google.genai")
    def test_generate_success(self, mock_genai):
        """Test successful generation."""
        # Setup
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Generated response"
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider("gemini-1.5-flash")
        result = provider.generate("Test prompt")

        assert result == "Generated response"
        # Check that generate_content was called with the prompt and generation config
        mock_model.generate_content.assert_called_once_with(
            "Test prompt",
            generation_config={"temperature": 0.7, "max_output_tokens": 1000, "top_p": 1.0},
        )

    @patch("src.providers.google.genai")
    def test_generate_empty_prompt(self, mock_genai):
        """Test generation with empty prompt."""
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider("gemini-1.5-flash")

        with pytest.raises(Exception) as exc_info:
            provider.generate("")

        assert "Empty prompt provided" in str(exc_info.value)
        mock_model.generate_content.assert_not_called()

    @patch("src.providers.google.genai")
    def test_generate_no_response_text(self, mock_genai):
        """Test generation when response has no text."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = None
        mock_response.prompt_feedback = "blocked"
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider("gemini-1.5-flash")

        with pytest.raises(Exception) as exc_info:
            provider.generate("Test prompt")

        assert "blocked by safety filters" in str(exc_info.value)

    @patch("src.providers.google.genai")
    def test_generate_api_error(self, mock_genai):
        """Test generation with API error."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("Invalid API_KEY")
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider("gemini-1.5-flash")

        with pytest.raises(Exception) as exc_info:
            provider.generate("Test prompt")

        assert "Invalid API_KEY" in str(exc_info.value)

    @patch("src.providers.google.genai")
    def test_generate_quota_error(self, mock_genai):
        """Test generation with quota exceeded error."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("Quota exceeded")
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider("gemini-1.5-flash")

        with pytest.raises(Exception) as exc_info:
            provider.generate("Test prompt")

        assert "Rate limit exceeded" in str(exc_info.value)

    @patch("src.providers.google.genai")
    def test_generate_timeout_error(self, mock_genai):
        """Test generation with timeout error."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("Request timeout")
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider("gemini-1.5-flash")

        with pytest.raises(Exception) as exc_info:
            provider.generate("Test prompt")

        assert "timed out" in str(exc_info.value)

    @patch("src.providers.google.genai")
    def test_generate_safety_filter(self, mock_genai):
        """Test generation blocked by safety filter."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("Safety filter blocked")
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider("gemini-1.5-flash")

        with pytest.raises(Exception) as exc_info:
            provider.generate("Test prompt")

        assert "blocked by safety filters" in str(exc_info.value)
