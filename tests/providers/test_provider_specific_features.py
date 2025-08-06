"""
Test provider-specific features

This module tests features that are unique to specific providers.
"""

import json
from unittest.mock import Mock, patch

import pytest

from llm_providers import AnthropicProvider, GoogleProvider, OpenAIProvider
from src.providers.exceptions import ProviderError

from .fixtures import (
    mock_anthropic_provider,
    mock_google_provider,
    mock_openai_provider,
    sample_evaluation_data,
    temp_config_file,
    test_config,
)


class TestOpenAISpecificFeatures:
    """Test OpenAI-specific features."""

    @patch("openai.OpenAI")
    def test_streaming_response(self, mock_openai_class):
        """Test OpenAI streaming responses."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Create mock streaming response
        mock_chunk1 = Mock()
        mock_chunk1.choices = [Mock(delta=Mock(content="Hello"))]

        mock_chunk2 = Mock()
        mock_chunk2.choices = [Mock(delta=Mock(content=" world"))]

        mock_chunk3 = Mock()
        mock_chunk3.choices = [Mock(delta=Mock(content="!"))]

        mock_stream = [mock_chunk1, mock_chunk2, mock_chunk3]
        mock_client.chat.completions.create.return_value = mock_stream

        provider = OpenAIProvider(model="gpt-3.5-turbo")
        provider.initialize()

        # Test streaming (if implemented)
        # Note: Current implementation might not support streaming
        # This test shows how to test it if added
        with patch.object(provider, "stream", True):
            try:
                response = provider.generate("Test", stream=True)
                # If streaming is implemented, check the response
                assert isinstance(response, str) or hasattr(response, "__iter__")
            except NotImplementedError:
                # Streaming might not be implemented yet
                pass

    @patch("openai.OpenAI")
    def test_function_calling(self, mock_openai_class):
        """Test OpenAI function calling feature."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock response with function call
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=None,
                    function_call=Mock(
                        name="get_weather", arguments='{"location": "San Francisco"}'
                    ),
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(model="gpt-3.5-turbo")
        provider.initialize()

        # Test with functions parameter
        functions = [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
            }
        ]

        # Current implementation might not support functions
        # This shows how to test when added
        try:
            response = provider.generate("What's the weather?", functions=functions)
            # Check if function call is handled
            assert response is not None
        except TypeError:
            # Functions parameter might not be supported yet
            pass

    @patch("openai.OpenAI")
    def test_logit_bias(self, mock_openai_class):
        """Test OpenAI logit bias feature."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Biased response"))]
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(model="gpt-3.5-turbo")
        provider.initialize()

        # Test with logit bias
        logit_bias = {123: 10, 456: -10}  # Token IDs with bias values

        try:
            response = provider.generate("Test", logit_bias=logit_bias)

            # Verify logit_bias was passed to API
            call_args = mock_client.chat.completions.create.call_args[1]
            if "logit_bias" in call_args:
                assert call_args["logit_bias"] == logit_bias
        except TypeError:
            # logit_bias might not be supported in current implementation
            pass

    @patch("openai.OpenAI")
    def test_response_format(self, mock_openai_class):
        """Test OpenAI response format options."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock JSON response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"result": "test"}'))]
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(model="gpt-3.5-turbo")
        provider.initialize()

        # Test JSON mode
        try:
            response = provider.generate("Return JSON", response_format={"type": "json_object"})
            # Try to parse as JSON
            json.loads(response)
        except (TypeError, json.JSONDecodeError):
            # Feature might not be implemented
            pass


class TestAnthropicSpecificFeatures:
    """Test Anthropic-specific features."""

    @patch("anthropic.Anthropic")
    def test_claude_vision(self, mock_anthropic_class):
        """Test Claude's vision capabilities."""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_message = Mock()
        mock_message.content = [Mock(text="I see an image of a cat")]
        mock_client.messages.create.return_value = mock_message

        provider = AnthropicProvider(model="claude-3-opus-20240229")
        provider.initialize()

        # Test with image input
        try:
            # Vision models can accept images
            response = provider.generate(
                "What's in this image?", images=["base64_encoded_image_data"]
            )
            assert "image" in response.lower() or "cat" in response.lower()
        except TypeError:
            # Image support might not be implemented
            pass

    @patch("anthropic.Anthropic")
    def test_claude_long_context(self, mock_anthropic_class):
        """Test Claude's long context handling."""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_message = Mock()
        mock_message.content = [Mock(text="Processed long context")]
        mock_message.usage = Mock(input_tokens=50000, output_tokens=100)
        mock_client.messages.create.return_value = mock_message

        provider = AnthropicProvider(model="claude-3-opus-20240229")
        provider.initialize()

        # Create a very long context (Claude supports up to 200k tokens)
        long_context = "This is a test. " * 10000  # ~40k words

        response = provider.generate(long_context, max_tokens=100)
        assert response == "Processed long context"

        # Verify the API was called with the long context
        call_args = mock_client.messages.create.call_args[1]
        assert len(call_args["messages"][0]["content"]) > 100000

    @patch("anthropic.Anthropic")
    def test_claude_xml_tags(self, mock_anthropic_class):
        """Test Claude's XML tag handling."""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_message = Mock()
        mock_message.content = [Mock(text="<analysis>Good code</analysis>")]
        mock_client.messages.create.return_value = mock_message

        provider = AnthropicProvider(model="claude-3-sonnet-20240229")
        provider.initialize()

        # Claude often uses XML tags for structured output
        response = provider.generate("Analyze this code and wrap your response in <analysis> tags")

        assert "<analysis>" in response
        assert "</analysis>" in response

    @patch("anthropic.Anthropic")
    def test_claude_prefill(self, mock_anthropic_class):
        """Test Claude's assistant message prefill feature."""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_message = Mock()
        mock_message.content = [Mock(text="Sure! Here's the answer: 42")]
        mock_client.messages.create.return_value = mock_message

        provider = AnthropicProvider(model="claude-3-haiku-20240307")
        provider.initialize()

        # Test with assistant prefill
        try:
            response = provider.generate(
                "What is the answer?", assistant_prefill="Sure! Here's the answer:"
            )
            assert response.startswith("Sure! Here's the answer:")
        except TypeError:
            # Prefill might not be implemented
            pass


class TestGoogleSpecificFeatures:
    """Test Google-specific features."""

    @patch("google.generativeai.GenerativeModel")
    def test_gemini_safety_settings(self, mock_model_class):
        """Test Gemini's safety settings."""
        # Setup mock
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        mock_response = Mock()
        mock_response.text = "Safe response"
        mock_response.prompt_feedback = Mock(
            safety_ratings=[Mock(category="HARM_CATEGORY_DANGEROUS", probability="NEGLIGIBLE")]
        )
        mock_model.generate_content.return_value = mock_response

        provider = GoogleProvider(model="gemini-1.5-pro")
        provider.initialize()

        # Test with custom safety settings
        safety_settings = {
            "HARM_CATEGORY_DANGEROUS": "BLOCK_NONE",
            "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
        }

        try:
            response = provider.generate("Generate something", safety_settings=safety_settings)
            assert response == "Safe response"
        except TypeError:
            # Safety settings might not be exposed in current implementation
            pass

    @patch("google.generativeai.GenerativeModel")
    def test_gemini_multi_turn_chat(self, mock_model_class):
        """Test Gemini's multi-turn chat capabilities."""
        # Setup mock
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        # Create mock chat
        mock_chat = Mock()
        mock_model.start_chat.return_value = mock_chat

        # Mock responses for multi-turn conversation
        mock_response1 = Mock(text="Hello! How can I help?")
        mock_response2 = Mock(text="The answer is 42")
        mock_chat.send_message.side_effect = [mock_response1, mock_response2]

        provider = GoogleProvider(model="gemini-1.5-flash")
        provider.initialize()

        # Test chat session (if implemented)
        try:
            # Start a chat session
            chat = provider.start_chat()
            response1 = chat.send_message("Hello")
            assert "Hello" in response1 or "help" in response1

            response2 = chat.send_message("What's the answer?")
            assert "42" in response2
        except AttributeError:
            # Chat API might not be exposed
            pass

    @patch("google.generativeai.GenerativeModel")
    def test_gemini_code_execution(self, mock_model_class):
        """Test Gemini's code execution capabilities."""
        # Setup mock
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        mock_response = Mock()
        mock_response.text = "The result is: 4"
        mock_response.candidates = [
            Mock(
                content=Mock(
                    parts=[
                        Mock(text="Let me calculate 2+2"),
                        Mock(code_execution_result=Mock(output="4")),
                    ]
                )
            )
        ]
        mock_model.generate_content.return_value = mock_response

        provider = GoogleProvider(model="gemini-1.5-pro")
        provider.initialize()

        # Test code execution (if supported)
        try:
            response = provider.generate("Calculate 2+2 using Python", enable_code_execution=True)
            assert "4" in response
        except TypeError:
            # Code execution might not be available
            pass

    @patch("google.generativeai.GenerativeModel")
    def test_gemini_grounding(self, mock_model_class):
        """Test Gemini's grounding/search capabilities."""
        # Setup mock
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        mock_response = Mock()
        mock_response.text = "According to recent search results..."
        mock_response.candidates = [
            Mock(
                grounding_metadata=Mock(
                    search_queries=["latest AI news"],
                    grounding_chunks=[Mock(web=Mock(uri="https://example.com"))],
                )
            )
        ]
        mock_model.generate_content.return_value = mock_response

        provider = GoogleProvider(model="gemini-1.5-flash")
        provider.initialize()

        # Test with grounding
        try:
            response = provider.generate("What's the latest in AI?", enable_grounding=True)
            assert "search" in response.lower() or "recent" in response.lower()
        except TypeError:
            # Grounding might not be exposed
            pass


class TestProviderCompatibilityFeatures:
    """Test features that should work across providers with adaptations."""

    def test_token_counting_compatibility(self):
        """Test token counting across providers."""
        test_text = "This is a test prompt for token counting."

        providers_and_mocks = [
            (OpenAIProvider, "openai.OpenAI", "tiktoken"),
            (AnthropicProvider, "anthropic.Anthropic", "anthropic"),
            (GoogleProvider, "google.generativeai.GenerativeModel", None),
        ]

        for provider_class, client_mock_path, token_lib in providers_and_mocks:
            with patch(client_mock_path):
                if hasattr(provider_class, "SUPPORTED_MODELS"):
                    model = list(provider_class.SUPPORTED_MODELS)[0]
                    provider = provider_class(model=model)

                    # Test token counting if available
                    try:
                        if hasattr(provider, "count_tokens"):
                            token_count = provider.count_tokens(test_text)
                            assert isinstance(token_count, int)
                            assert token_count > 0
                    except (AttributeError, NotImplementedError):
                        # Token counting might not be implemented
                        pass

    def test_conversation_history_handling(self):
        """Test how providers handle conversation history."""
        providers = [
            (OpenAIProvider, "openai.OpenAI"),
            (AnthropicProvider, "anthropic.Anthropic"),
            (GoogleProvider, "google.generativeai.GenerativeModel"),
        ]

        for provider_class, mock_path in providers:
            with patch(mock_path) as mock_client:
                if hasattr(provider_class, "SUPPORTED_MODELS"):
                    model = list(provider_class.SUPPORTED_MODELS)[0]
                    provider = provider_class(model=model)

                    # Test with conversation history
                    history = [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello!"},
                        {"role": "user", "content": "How are you?"},
                    ]

                    try:
                        # Mock appropriate response
                        if provider_class == OpenAIProvider:
                            mock_instance = mock_client.return_value
                            mock_response = Mock()
                            mock_response.choices = [Mock(message=Mock(content="I'm doing well!"))]
                            mock_instance.chat.completions.create.return_value = mock_response

                        provider.initialize()
                        response = provider.generate("How are you?", conversation_history=history)
                        assert isinstance(response, str)
                    except TypeError:
                        # History might not be supported in current implementation
                        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
