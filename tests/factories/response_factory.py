"""
Response Factory

Factory for creating test API responses and completions.
"""

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


@dataclass
class ResponseFactory:
    """Factory for creating API response objects."""

    @classmethod
    def create_completion(
        cls,
        text: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        tokens_used: Optional[int] = None,
        finish_reason: str = "stop",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a completion response object.

        Args:
            text: Response text
            model: Model name
            tokens_used: Number of tokens used
            finish_reason: Reason for completion end
            **kwargs: Additional response fields

        Returns:
            Completion response dictionary
        """
        if text is None:
            text = cls._generate_text()

        if tokens_used is None:
            tokens_used = len(text.split()) * 2  # Rough estimate

        response = {
            "id": f"completion-{random.randint(1000, 9999)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": random.randint(10, 100),
                "completion_tokens": tokens_used,
                "total_tokens": tokens_used + random.randint(10, 100),
            },
        }

        response.update(kwargs)
        return response

    @classmethod
    def create_chat_completion(
        cls,
        content: Optional[str] = None,
        role: str = "assistant",
        model: str = "gpt-3.5-turbo",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a chat completion response object.

        Args:
            content: Message content
            role: Message role
            model: Model name
            **kwargs: Additional response fields

        Returns:
            Chat completion response dictionary
        """
        if content is None:
            content = cls._generate_text()

        response = {
            "id": f"chatcmpl-{random.randint(1000, 9999)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": role,
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": random.randint(10, 100),
                "completion_tokens": len(content.split()) * 2,
                "total_tokens": len(content.split()) * 2 + random.randint(10, 100),
            },
        }

        response.update(kwargs)
        return response

    @classmethod
    def create_error_response(
        cls,
        error_type: str = "invalid_request_error",
        message: Optional[str] = None,
        status_code: int = 400,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create an error response object.

        Args:
            error_type: Type of error
            message: Error message
            status_code: HTTP status code
            **kwargs: Additional error fields

        Returns:
            Error response dictionary
        """
        if message is None:
            message = cls._generate_error_message(error_type)

        response = {
            "error": {
                "type": error_type,
                "message": message,
                "param": kwargs.get("param"),
                "code": kwargs.get("code", status_code),
            }
        }

        return response

    @classmethod
    def create_streaming_response(
        cls, chunks: Optional[List[str]] = None, model: str = "gpt-3.5-turbo", **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Create a streaming response as a list of chunks.

        Args:
            chunks: List of text chunks
            model: Model name
            **kwargs: Additional chunk fields

        Returns:
            List of streaming chunk dictionaries
        """
        if chunks is None:
            text = cls._generate_text()
            chunks = text.split()

        response_chunks = []

        for i, chunk in enumerate(chunks):
            chunk_data = {
                "id": f"chunk-{random.randint(1000, 9999)}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": chunk + " " if i < len(chunks) - 1 else chunk,
                        },
                        "finish_reason": None if i < len(chunks) - 1 else "stop",
                    }
                ],
            }
            chunk_data.update(kwargs)
            response_chunks.append(chunk_data)

        return response_chunks

    @staticmethod
    def _generate_text() -> str:
        """Generate random response text."""
        templates = [
            "The answer to your question is: {}",
            "Based on my analysis, I can conclude that: {}",
            "Here's what I found: {}",
            "Let me explain this concept: {}",
            "The solution involves: {}",
        ]

        content = [
            "this is a comprehensive response with multiple important points",
            "the implementation requires careful consideration of various factors",
            "we should analyze the problem from different perspectives",
            "the optimal approach would be to use established best practices",
            "there are several viable solutions to this challenge",
        ]

        template = random.choice(templates)
        text = random.choice(content)
        return template.format(text)

    @staticmethod
    def _generate_error_message(error_type: str) -> str:
        """Generate error message based on type."""
        messages = {
            "invalid_request_error": "The request was invalid or cannot be processed",
            "authentication_error": "Invalid authentication credentials",
            "rate_limit_error": "Rate limit exceeded. Please retry after some time",
            "server_error": "An error occurred on the server",
            "timeout_error": "The request timed out",
        }
        return messages.get(error_type, "An error occurred")


class CompletionFactory:
    """Factory specifically for text completions."""

    @staticmethod
    def create_code_completion(
        language: str = "python", code: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a code completion response."""
        if code is None:
            code_samples = {
                "python": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
                "javascript": "function fibonacci(n) {\n    if (n <= 1) return n;\n    return fibonacci(n - 1) + fibonacci(n - 2);\n}",
                "java": "public static int sum(int[] arr) {\n    int total = 0;\n    for (int num : arr) {\n        total += num;\n    }\n    return total;\n}",
            }
            code = code_samples.get(language, code_samples["python"])

        return ResponseFactory.create_completion(
            text=f"```{language}\n{code}\n```",
            model="code-davinci-002",
        )

    @staticmethod
    def create_translation_completion(
        source_lang: str = "English",
        target_lang: str = "French",
        source_text: str = "Hello, world!",
        translation: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a translation completion response."""
        if translation is None:
            translations = {
                "French": "Bonjour, le monde!",
                "Spanish": "¡Hola, mundo!",
                "German": "Hallo, Welt!",
                "Italian": "Ciao, mondo!",
                "Japanese": "こんにちは、世界！",
            }
            translation = translations.get(target_lang, f"[Translation to {target_lang}]")

        return ResponseFactory.create_completion(
            text=translation,
            model="text-davinci-003",
        )

    @staticmethod
    def create_summary_completion(
        text: Optional[str] = None, summary_length: str = "short"
    ) -> Dict[str, Any]:
        """Create a text summarization completion."""
        if text is None:
            summaries = {
                "short": "This text discusses the main points briefly.",
                "medium": "This text provides a comprehensive overview of the topic, covering the key aspects and important details that readers should know.",
                "long": "This detailed summary encompasses all the major points discussed in the original text, including the background context, main arguments, supporting evidence, and conclusions drawn. It preserves the essential information while condensing the content.",
            }
            text = summaries.get(summary_length, summaries["short"])

        return ResponseFactory.create_completion(text=text)


class ChatFactory:
    """Factory specifically for chat completions."""

    @staticmethod
    def create_conversation(
        messages: Optional[List[Dict[str, str]]] = None, model: str = "gpt-3.5-turbo"
    ) -> List[Dict[str, Any]]:
        """
        Create a series of chat messages.

        Args:
            messages: List of message dictionaries with role and content
            model: Model name

        Returns:
            List of chat completion responses
        """
        if messages is None:
            messages = [
                {"role": "user", "content": "What is machine learning?"},
                {
                    "role": "assistant",
                    "content": "Machine learning is a subset of AI that enables systems to learn from data.",
                },
                {"role": "user", "content": "Can you give me an example?"},
                {
                    "role": "assistant",
                    "content": "Sure! Email spam filters use machine learning to identify spam messages.",
                },
            ]

        responses = []
        for msg in messages:
            if msg["role"] == "assistant":
                responses.append(
                    ResponseFactory.create_chat_completion(
                        content=msg["content"],
                        role=msg["role"],
                        model=model,
                    )
                )

        return responses

    @staticmethod
    def create_function_call_response(
        function_name: str = "get_weather",
        arguments: Optional[Dict[str, Any]] = None,
        model: str = "gpt-3.5-turbo-0613",
    ) -> Dict[str, Any]:
        """Create a chat completion with function calling."""
        if arguments is None:
            arguments = {"location": "San Francisco", "unit": "celsius"}

        return {
            "id": f"chatcmpl-{random.randint(1000, 9999)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": function_name,
                            "arguments": json.dumps(arguments),
                        },
                    },
                    "finish_reason": "function_call",
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "total_tokens": 70,
            },
        }
