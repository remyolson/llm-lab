"""
Custom Evaluation Functions for Recipe-Based Fine-Tuning

This module provides a collection of custom evaluation functions that can be
referenced in recipe configurations for domain-specific model assessment.

Example:
    # In recipe YAML/JSON:
    evaluation:
        custom_eval_function: "domain_specific_medical"
        eval_function_config:
            min_accuracy: 0.85
            test_cases: "path/to/medical_cases.json"
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class CustomEvaluationRegistry:
    """Registry for custom evaluation functions."""

    _evaluations: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a custom evaluation function."""

        def decorator(func: Callable):
            cls._evaluations[name] = func
            return func

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Callable]:
        """Get a registered evaluation function by name."""
        return cls._evaluations.get(name)

    @classmethod
    def list_available(cls) -> List[str]:
        """List all available custom evaluation functions."""
        return list(cls._evaluations.keys())


@CustomEvaluationRegistry.register("instruction_following")
def evaluate_instruction_following(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Evaluate model's ability to follow instructions.

    Args:
        model: Fine-tuned model
        tokenizer: Model tokenizer
        config: Evaluation configuration

    Returns:
        List of metric results
    """
    config = config or {}
    test_instructions = config.get(
        "test_instructions",
        [
            "Write a haiku about machine learning",
            "List 5 benefits of exercise",
            "Explain quantum computing in simple terms",
        ],
    )

    results = []
    for instruction in test_instructions:
        prompt = f"### Instruction: {instruction}\n### Response:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # Basic evaluation criteria
        follows_format = "###" not in response  # Shouldn't repeat format markers
        has_content = len(response.strip()) > 10

        results.append(
            {
                "name": f"instruction_following_{instruction[:20]}",
                "value": float(follows_format and has_content),
                "metadata": {
                    "instruction": instruction,
                    "response_length": len(response),
                    "follows_format": follows_format,
                },
            }
        )

    # Aggregate score
    avg_score = np.mean([r["value"] for r in results])
    results.append(
        {
            "name": "instruction_following_overall",
            "value": avg_score,
            "metadata": {"num_tests": len(test_instructions)},
        }
    )

    return results


@CustomEvaluationRegistry.register("code_generation")
def evaluate_code_generation(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Evaluate model's code generation capabilities.

    Args:
        model: Fine-tuned model
        tokenizer: Model tokenizer
        config: Evaluation configuration

    Returns:
        List of metric results
    """
    config = config or {}

    test_problems = config.get(
        "test_problems",
        [
            {
                "prompt": "Write a Python function to calculate factorial",
                "test": "assert factorial(5) == 120",
            },
            {
                "prompt": "Write a Python function to check if a number is prime",
                "test": "assert is_prime(7) == True",
            },
        ],
    )

    results = []
    for problem in test_problems:
        prompt = f"```python\n# {problem['prompt']}\ndef "
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.2, do_sample=True)

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # Extract function and check syntax
        try:
            # Simple syntax check
            compile(f"def {generated}", "<string>", "exec")
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False

        results.append(
            {
                "name": f"code_gen_{problem['prompt'][:30]}",
                "value": float(syntax_valid),
                "metadata": {
                    "prompt": problem["prompt"],
                    "syntax_valid": syntax_valid,
                    "generated_length": len(generated),
                },
            }
        )

    avg_score = np.mean([r["value"] for r in results])
    results.append(
        {
            "name": "code_generation_overall",
            "value": avg_score,
            "metadata": {"num_problems": len(test_problems)},
        }
    )

    return results


@CustomEvaluationRegistry.register("domain_specific_medical")
def evaluate_medical_domain(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Evaluate model on medical domain knowledge.

    Args:
        model: Fine-tuned model
        tokenizer: Model tokenizer
        config: Evaluation configuration

    Returns:
        List of metric results
    """
    config = config or {}

    # Medical terminology and reasoning tests
    test_cases = [
        {
            "question": "What is the primary function of antibodies?",
            "keywords": ["immune", "antigen", "protein", "defense"],
        },
        {
            "question": "Describe the symptoms of Type 2 diabetes",
            "keywords": ["glucose", "insulin", "thirst", "fatigue"],
        },
    ]

    results = []
    for case in test_cases:
        prompt = f"Q: {case['question']}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.3, do_sample=True)

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).lower()

        # Check for key medical terms
        keywords_found = sum(1 for kw in case["keywords"] if kw.lower() in response)
        accuracy = keywords_found / len(case["keywords"])

        results.append(
            {
                "name": f"medical_{case['question'][:30]}",
                "value": accuracy,
                "metadata": {
                    "question": case["question"],
                    "keywords_found": keywords_found,
                    "total_keywords": len(case["keywords"]),
                },
            }
        )

    avg_accuracy = np.mean([r["value"] for r in results])
    results.append(
        {
            "name": "medical_domain_overall",
            "value": avg_accuracy,
            "metadata": {"num_tests": len(test_cases)},
        }
    )

    return results


@CustomEvaluationRegistry.register("chat_coherence")
def evaluate_chat_coherence(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Evaluate model's ability to maintain coherent conversations.

    Args:
        model: Fine-tuned model
        tokenizer: Model tokenizer
        config: Evaluation configuration

    Returns:
        List of metric results
    """
    config = config or {}

    # Multi-turn conversation test
    conversation = [
        "Hello! How are you today?",
        "I'm doing well, thank you! How can I help you?",
        "Can you tell me about the weather?",
    ]

    results = []
    context = ""

    for i, turn in enumerate(conversation[:-1]):
        context += f"User: {turn}\n"
        if i > 0:
            context += f"Assistant: {conversation[i]}\n"

        prompt = context + "Assistant:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7, do_sample=True)

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # Evaluate coherence metrics
        is_relevant = len(response) > 10 and not response.startswith(turn)
        no_repetition = turn.lower() not in response.lower()

        results.append(
            {
                "name": f"chat_turn_{i + 1}",
                "value": float(is_relevant and no_repetition),
                "metadata": {
                    "turn": i + 1,
                    "response_length": len(response),
                    "is_relevant": is_relevant,
                },
            }
        )

    coherence_score = np.mean([r["value"] for r in results])
    results.append(
        {
            "name": "chat_coherence_overall",
            "value": coherence_score,
            "metadata": {"num_turns": len(conversation) - 1},
        }
    )

    return results


@CustomEvaluationRegistry.register("summarization_quality")
def evaluate_summarization(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Evaluate model's summarization capabilities.

    Args:
        model: Fine-tuned model
        tokenizer: Model tokenizer
        config: Evaluation configuration

    Returns:
        List of metric results
    """
    config = config or {}

    test_text = config.get(
        "test_text",
        "Machine learning is a subset of artificial intelligence that enables "
        "computers to learn and improve from experience without being explicitly "
        "programmed. It focuses on developing algorithms that can analyze data, "
        "identify patterns, and make decisions with minimal human intervention.",
    )

    prompt = f"Summarize the following text:\n{test_text}\n\nSummary:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.3, do_sample=True)

    summary = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    # Basic quality metrics
    compression_ratio = len(summary) / len(test_text)
    has_key_terms = any(
        term in summary.lower() for term in ["machine learning", "ai", "artificial intelligence"]
    )
    appropriate_length = 0.1 < compression_ratio < 0.5

    return [
        {
            "name": "summarization_quality",
            "value": float(has_key_terms and appropriate_length),
            "metadata": {
                "compression_ratio": compression_ratio,
                "summary_length": len(summary),
                "original_length": len(test_text),
            },
        }
    ]


def get_custom_evaluation_function(
    name: str, config: Optional[Dict[str, Any]] = None
) -> Optional[Callable]:
    """
    Get a custom evaluation function by name with configuration.

    Args:
        name: Name of the evaluation function
        config: Configuration for the evaluation

    Returns:
        Configured evaluation function or None
    """
    eval_fn = CustomEvaluationRegistry.get(name)

    if eval_fn is None:
        logger.warning(f"Custom evaluation function '{name}' not found")
        return None

    # Return a wrapped function that includes the config
    def wrapped_eval(model, tokenizer):
        return eval_fn(model, tokenizer, config)

    return wrapped_eval


def create_recipe_evaluation_function(recipe: Dict[str, Any]) -> Optional[Callable]:
    """
    Create an evaluation function from a recipe configuration.

    Args:
        recipe: Recipe dictionary containing evaluation config

    Returns:
        Evaluation function or None
    """
    eval_config = recipe.get("evaluation", {})
    eval_name = eval_config.get("custom_eval_function")

    if not eval_name:
        return None

    eval_fn_config = eval_config.get("eval_function_config", {})
    return get_custom_evaluation_function(eval_name, eval_fn_config)


# Export main components
__all__ = [
    "CustomEvaluationRegistry",
    "create_recipe_evaluation_function",
    "evaluate_chat_coherence",
    "evaluate_code_generation",
    "evaluate_instruction_following",
    "evaluate_medical_domain",
    "evaluate_summarization",
    "get_custom_evaluation_function",
]
