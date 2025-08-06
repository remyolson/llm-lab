#!/usr/bin/env python3
"""
Cross-LLM Testing Examples for Use Case 3 & 4

This module provides comprehensive testing examples demonstrating cross-LLM testing
patterns including unit tests, regression testing, and performance benchmarking.

Example 1: Unit Tests for Chatbot with Pytest Fixtures
Example 2: Regression Test Suite for Model Performance Monitoring
Example 3: Performance Benchmark Suite for Latency/Throughput/Cost Analysis
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from providers.anthropic import AnthropicProvider

# Import existing provider infrastructure
from providers.base import LLMProvider
from providers.google import GoogleProvider
from providers.openai import OpenAIProvider

# ===============================================================================
# EXAMPLE 1: UNIT TESTS FOR CHATBOT WITH PYTEST FIXTURES
# ===============================================================================


@dataclass
class ChatbotMessage:
    """Represents a chatbot message."""

    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatbotResponse:
    """Represents a chatbot response with metrics."""

    message: str
    response_time: float
    token_count: int
    cost: float
    provider: str
    model: str
    confidence: float = 0.0


class SimpleChatbot:
    """
    Simple chatbot implementation for testing different LLM providers.

    This serves as the system under test for our unit testing examples.
    """

    def __init__(self, provider: LLMProvider, system_prompt: str = None):
        """Initialize chatbot with a provider."""
        self.provider = provider
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.conversation_history: List[ChatbotMessage] = []
        self._total_cost = 0.0
        self._total_tokens = 0

    def chat(self, user_message: str, max_tokens: int = 150) -> ChatbotResponse:
        """Send a message and get a response."""
        # Add user message to history
        user_msg = ChatbotMessage(role="user", content=user_message)
        self.conversation_history.append(user_msg)

        # Prepare context
        context = self._build_context()

        # Get response from provider
        start_time = time.time()
        try:
            response = self.provider.generate(
                prompt=context, max_tokens=max_tokens, temperature=0.7
            )
            response_time = time.time() - start_time

            # Add response to history
            assistant_msg = ChatbotMessage(role="assistant", content=response)
            self.conversation_history.append(assistant_msg)

            # Calculate metrics (simplified)
            token_count = len(response.split()) * 2  # Rough estimate
            cost = self._estimate_cost(token_count)

            self._total_tokens += token_count
            self._total_cost += cost

            return ChatbotResponse(
                message=response,
                response_time=response_time,
                token_count=token_count,
                cost=cost,
                provider=self.provider.__class__.__name__,
                model=self.provider.model,
                confidence=0.95,  # Mock confidence score
            )

        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"Error: {e!s}"

            assistant_msg = ChatbotMessage(
                role="assistant", content=error_msg, metadata={"error": True}
            )
            self.conversation_history.append(assistant_msg)

            return ChatbotResponse(
                message=error_msg,
                response_time=response_time,
                token_count=0,
                cost=0.0,
                provider=self.provider.__class__.__name__,
                model=self.provider.model,
                confidence=0.0,
            )

    def _build_context(self) -> str:
        """Build context from conversation history."""
        context_parts = [self.system_prompt]

        # Include recent conversation history (last 5 messages)
        recent_messages = self.conversation_history[-5:]
        for msg in recent_messages:
            context_parts.append(f"{msg.role}: {msg.content}")

        return "\n".join(context_parts)

    def _estimate_cost(self, token_count: int) -> float:
        """Estimate cost based on provider and token count."""
        # Simplified cost estimation
        cost_per_token = {
            "OpenAIProvider": 0.00002,
            "AnthropicProvider": 0.00001,
            "GoogleProvider": 0.000005,
        }

        provider_name = self.provider.__class__.__name__
        return token_count * cost_per_token.get(provider_name, 0.00001)

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation metrics."""
        return {
            "total_messages": len(self.conversation_history),
            "total_tokens": self._total_tokens,
            "total_cost": self._total_cost,
            "conversation_duration": (
                self.conversation_history[-1].timestamp - self.conversation_history[0].timestamp
            ).total_seconds()
            if self.conversation_history
            else 0,
        }

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        self._total_cost = 0.0
        self._total_tokens = 0


# ===============================================================================
# PYTEST FIXTURES FOR CROSS-PROVIDER TESTING
# ===============================================================================


@pytest.fixture
def mock_openai_provider():
    """Mock OpenAI provider for testing."""
    provider = Mock(spec=OpenAIProvider)
    provider.model = "gpt-4o-mini"
    provider.__class__.__name__ = "OpenAIProvider"
    provider.generate.return_value = "This is a test response from OpenAI."
    return provider


@pytest.fixture
def mock_anthropic_provider():
    """Mock Anthropic provider for testing."""
    provider = Mock(spec=AnthropicProvider)
    provider.model = "claude-3-haiku-20240307"
    provider.__class__.__name__ = "AnthropicProvider"
    provider.generate.return_value = "This is a test response from Anthropic."
    return provider


@pytest.fixture
def mock_google_provider():
    """Mock Google provider for testing."""
    provider = Mock(spec=GoogleProvider)
    provider.model = "gemini-1.5-flash"
    provider.__class__.__name__ = "GoogleProvider"
    provider.generate.return_value = "This is a test response from Google."
    return provider


@pytest.fixture(params=["openai", "anthropic", "google"])
def provider_fixture(request, mock_openai_provider, mock_anthropic_provider, mock_google_provider):
    """Parametrized fixture that provides all provider types."""
    providers = {
        "openai": mock_openai_provider,
        "anthropic": mock_anthropic_provider,
        "google": mock_google_provider,
    }
    return providers[request.param]


@pytest.fixture
def chatbot_factory():
    """Factory for creating chatbot instances."""

    def _create_chatbot(provider, system_prompt=None):
        return SimpleChatbot(provider, system_prompt)

    return _create_chatbot


@pytest.fixture
def test_prompts():
    """Collection of test prompts for comprehensive testing."""
    return {
        "simple": "Hello, how are you?",
        "math": "What is 2 + 2?",
        "code": "Write a Python function to reverse a string.",
        "creative": "Write a short poem about testing.",
        "reasoning": "Explain why testing is important in software development.",
        "long_context": "A" * 1000 + " Please summarize this.",
        "empty": "",
        "special_chars": "Test with symbols: @#$%^&*()",
        "multilingual": "Hola, ¿cómo estás? 你好吗？",
        "edge_case": None,
    }


# ===============================================================================
# UNIT TEST EXAMPLES
# ===============================================================================


class TestChatbotCrossProvider:
    """Cross-provider unit tests for the chatbot."""

    def test_basic_chat_functionality(self, provider_fixture, chatbot_factory):
        """Test basic chat functionality across all providers."""
        chatbot = chatbot_factory(provider_fixture)

        response = chatbot.chat("Hello!")

        assert response.message is not None
        assert len(response.message) > 0
        assert response.response_time >= 0
        assert response.provider == provider_fixture.__class__.__name__
        assert response.model == provider_fixture.model
        assert len(chatbot.conversation_history) == 2  # User + Assistant

    def test_conversation_history_tracking(self, provider_fixture, chatbot_factory):
        """Test conversation history is properly tracked."""
        chatbot = chatbot_factory(provider_fixture)

        # Send multiple messages
        chatbot.chat("First message")
        chatbot.chat("Second message")
        chatbot.chat("Third message")

        assert len(chatbot.conversation_history) == 6  # 3 user + 3 assistant
        assert chatbot.conversation_history[0].role == "user"
        assert chatbot.conversation_history[1].role == "assistant"
        assert chatbot.conversation_history[0].content == "First message"

    def test_cost_tracking(self, provider_fixture, chatbot_factory):
        """Test that cost tracking works across providers."""
        chatbot = chatbot_factory(provider_fixture)

        response = chatbot.chat("Test message")
        summary = chatbot.get_conversation_summary()

        assert response.cost > 0
        assert summary["total_cost"] > 0
        assert summary["total_tokens"] > 0

    def test_different_prompts(self, provider_fixture, chatbot_factory, test_prompts):
        """Test various prompt types with each provider."""
        chatbot = chatbot_factory(provider_fixture)

        # Test with different prompt types
        for prompt_type, prompt in test_prompts.items():
            if prompt is None or prompt == "":
                continue  # Skip edge cases for this test

            response = chatbot.chat(prompt)

            assert response.message is not None
            assert response.provider == provider_fixture.__class__.__name__
            # Each provider should handle different prompt types

    def test_error_handling(self, provider_fixture, chatbot_factory):
        """Test error handling across providers."""
        chatbot = chatbot_factory(provider_fixture)

        # Mock provider to raise an exception
        provider_fixture.generate.side_effect = Exception("API Error")

        response = chatbot.chat("This should fail")

        assert "Error:" in response.message
        assert response.confidence == 0.0
        assert response.cost == 0.0
        assert response.token_count == 0

    def test_conversation_reset(self, provider_fixture, chatbot_factory):
        """Test conversation reset functionality."""
        chatbot = chatbot_factory(provider_fixture)

        # Build up conversation
        chatbot.chat("Message 1")
        chatbot.chat("Message 2")

        initial_summary = chatbot.get_conversation_summary()
        assert initial_summary["total_messages"] == 4

        # Reset and verify
        chatbot.reset_conversation()
        reset_summary = chatbot.get_conversation_summary()

        assert reset_summary["total_messages"] == 0
        assert reset_summary["total_cost"] == 0.0
        assert reset_summary["total_tokens"] == 0

    def test_system_prompt_usage(self, provider_fixture, chatbot_factory):
        """Test that system prompts are properly used."""
        system_prompt = "You are a math tutor. Only answer math questions."
        chatbot = chatbot_factory(provider_fixture, system_prompt)

        response = chatbot.chat("What is 5 + 3?")

        # Verify the provider was called with the context including system prompt
        call_args = provider_fixture.generate.call_args
        assert system_prompt in call_args[1]["prompt"]

    @pytest.mark.parametrize("max_tokens", [50, 100, 200])
    def test_token_limits(self, provider_fixture, chatbot_factory, max_tokens):
        """Test token limit handling across providers."""
        chatbot = chatbot_factory(provider_fixture)

        response = chatbot.chat("Tell me a long story", max_tokens=max_tokens)

        # Verify max_tokens was passed to provider
        call_args = provider_fixture.generate.call_args
        assert call_args[1]["max_tokens"] == max_tokens

    def test_concurrent_chatbots(self, provider_fixture, chatbot_factory):
        """Test multiple chatbot instances don't interfere."""
        chatbot1 = chatbot_factory(provider_fixture)
        chatbot2 = chatbot_factory(provider_fixture)

        chatbot1.chat("Chatbot 1 message")
        chatbot2.chat("Chatbot 2 message")

        assert len(chatbot1.conversation_history) == 2
        assert len(chatbot2.conversation_history) == 2
        assert chatbot1.conversation_history[0].content != chatbot2.conversation_history[0].content


# ===============================================================================
# INTEGRATION TEST EXAMPLES
# ===============================================================================


class TestChatbotIntegration:
    """Integration tests for real provider interactions."""

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("INTEGRATION_TESTS"), reason="Integration tests disabled")
    def test_real_provider_integration(self, chatbot_factory):
        """Test with real providers (requires API keys)."""
        # This would test with real providers if API keys are available
        providers_to_test = []

        # Try to initialize real providers
        if os.getenv("OPENAI_API_KEY"):
            try:
                provider = OpenAIProvider(model="gpt-4o-mini")
                provider.initialize()
                providers_to_test.append(("OpenAI", provider))
            except Exception:
                pass

        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                provider = AnthropicProvider(model="claude-3-haiku-20240307")
                provider.initialize()
                providers_to_test.append(("Anthropic", provider))
            except Exception:
                pass

        if os.getenv("GOOGLE_API_KEY"):
            try:
                provider = GoogleProvider(model="gemini-1.5-flash")
                provider.initialize()
                providers_to_test.append(("Google", provider))
            except Exception:
                pass

        # Test each available provider
        for provider_name, provider in providers_to_test:
            chatbot = chatbot_factory(provider)

            response = chatbot.chat("Hello! Can you help me test this chatbot?")

            assert response.message is not None
            assert len(response.message) > 0
            assert response.response_time > 0
            assert response.cost >= 0
            assert response.provider == provider.__class__.__name__

            print(f"\n{provider_name} Integration Test:")
            print(f"  Response: {response.message[:100]}...")
            print(f"  Response Time: {response.response_time:.3f}s")
            print(f"  Cost: ${response.cost:.6f}")


if __name__ == "__main__":
    # Example usage
    print("Cross-LLM Testing Examples - Unit Tests")
    print("=" * 50)

    # Run a simple example
    mock_provider = Mock(spec=LLMProvider)
    mock_provider.model = "test-model"
    mock_provider.__class__.__name__ = "TestProvider"
    mock_provider.generate.return_value = "Hello! I'm a test chatbot."

    chatbot = SimpleChatbot(mock_provider)
    response = chatbot.chat("Hello!")

    print(f"Chatbot Response: {response.message}")
    print(f"Response Time: {response.response_time:.3f}s")
    print(f"Cost: ${response.cost:.6f}")
    print(f"Provider: {response.provider}")

    print("\nTo run the full test suite:")
    print("pytest examples/use_cases/cross_llm_testing_examples.py -v")
    print("\nTo run with coverage:")
    print("pytest examples/use_cases/cross_llm_testing_examples.py --cov=. -v")
