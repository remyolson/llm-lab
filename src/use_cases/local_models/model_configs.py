"""
Model configurations for popular local models.

This module provides default configurations and download information
for commonly used open-source models.
"""

from typing import Any, Dict

# Model download URLs and information
MODEL_REGISTRY = {
    "llama-2-7b": {
        "name": "Llama 2 7B Chat",
        "filename": "llama-2-7b-chat.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
        "size": "3.83 GB",
        "quantization": "Q4_K_M",
        "context_length": 4096,
        "recommended_gpu_layers": 35,
        "min_ram": "6 GB",
        "description": "Meta's Llama 2 7B model optimized for chat, 4-bit quantized",
    },
    "llama-2-13b": {
        "name": "Llama 2 13B Chat",
        "filename": "llama-2-13b-chat.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf",
        "size": "7.37 GB",
        "quantization": "Q4_K_M",
        "context_length": 4096,
        "recommended_gpu_layers": 43,
        "min_ram": "10 GB",
        "description": "Meta's Llama 2 13B model optimized for chat, 4-bit quantized",
    },
    "mistral-7b": {
        "name": "Mistral 7B Instruct v0.2",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size": "4.07 GB",
        "quantization": "Q4_K_M",
        "context_length": 32768,
        "recommended_gpu_layers": 35,
        "min_ram": "6 GB",
        "description": "Mistral AI's 7B instruct model v0.2, 4-bit quantized",
    },
    "phi-2": {
        "name": "Microsoft Phi-2",
        "filename": "phi-2.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
        "size": "1.45 GB",
        "quantization": "Q4_K_M",
        "context_length": 2048,
        "recommended_gpu_layers": 33,
        "min_ram": "3 GB",
        "description": "Microsoft's Phi-2 2.7B model, compact but capable",
    },
}

# Prompt templates for different models
PROMPT_TEMPLATES = {
    "llama-2": {
        "system": "<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n",
        "user": "{prompt} [/INST]",
        "assistant": " {response} </s><s>[INST] ",
        "default_system": "You are a helpful, respectful and honest assistant.",
    },
    "mistral": {
        "system": "<s>",
        "user": "[INST] {prompt} [/INST]",
        "assistant": "{response}</s> ",
        "default_system": "",
    },
    "phi": {
        "system": "",
        "user": "Instruct: {prompt}\nOutput:",
        "assistant": " {response}\n",
        "default_system": "",
    },
    "default": {
        "system": "",
        "user": "User: {prompt}\nAssistant:",
        "assistant": " {response}\n",
        "default_system": "",
    },
}


def get_model_config(model_name: str) -> Dict[str | Any]:
    """
    Get configuration for a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Model configuration dictionary
    """
    return MODEL_REGISTRY.get(model_name, {})


def get_prompt_template(model_name: str) -> Dict[str | str]:
    """
    Get the appropriate prompt template for a model.

    Args:
        model_name: Name of the model

    Returns:
        Prompt template dictionary
    """
    # Map model names to template types
    if "llama-2" in model_name:
        return PROMPT_TEMPLATES["llama-2"]
    elif "mistral" in model_name:
        return PROMPT_TEMPLATES["mistral"]
    elif "phi" in model_name:
        return PROMPT_TEMPLATES["phi"]
    else:
        return PROMPT_TEMPLATES["default"]


def format_prompt(prompt: str, model_name: str, system_prompt: str = None) -> str:
    """
    Format a prompt according to the model's expected template.

    Args:
        prompt: The user prompt
        model_name: Name of the model
        system_prompt: Optional system prompt

    Returns:
        Formatted prompt string
    """
    template = get_prompt_template(model_name)

    formatted = ""

    # Add system prompt if provided
    if system_prompt and template["system"]:
        formatted += template["system"].format(system=system_prompt)
    elif template.get("default_system"):
        formatted += template["system"].format(system=template["default_system"])

    # Add user prompt
    formatted += template["user"].format(prompt=prompt)

    return formatted
