"""
Dataset Registry for Fine-Tuning

This module provides a centralized registry of available datasets for fine-tuning,
including HuggingFace datasets and custom datasets.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class DatasetType(Enum):
    """Types of datasets available."""
    INSTRUCTION = "instruction"
    CONVERSATION = "conversation"
    COMPLETION = "completion"
    CLASSIFICATION = "classification"
    QA = "qa"
    MATH = "math"
    CODE = "code"
    STEM = "stem"
    TOOL_CALLING = "tool_calling"
    MIXED = "mixed"


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    hf_path: str
    description: str
    dataset_type: DatasetType
    size: str  # e.g., "25.7M rows", "100GB"
    splits: List[str]
    license: str
    language: str = "en"
    tasks: List[str] = field(default_factory=list)
    format: str = "parquet"
    citation: Optional[str] = None
    url: Optional[str] = None
    special_features: List[str] = field(default_factory=list)
    recommended_models: List[str] = field(default_factory=list)
    loading_args: Dict[str, Any] = field(default_factory=dict)


class DatasetRegistry:
    """Registry of available datasets for fine-tuning."""

    DATASETS = {
        # Nemotron Post-Training Dataset
        "nemotron-post-training": DatasetInfo(
            name="Nemotron Post-Training Dataset v1",
            hf_path="nvidia/Nemotron-Post-Training-Dataset-v1",
            description="A comprehensive SFT dataset from NVIDIA designed to improve capabilities in math, code, STEM, reasoning, and tool calling. Used for training Llama-3.3-Nemotron models.",
            dataset_type=DatasetType.MIXED,
            size="25.7M rows",
            splits=["chat", "code", "math", "stem", "tool_calling"],
            license="CC BY 4.0",
            language="en",
            tasks=["instruction-following", "mathematical-reasoning", "code-generation", "stem-qa", "tool-use"],
            format="parquet",
            special_features=[
                "Multi-domain coverage",
                "High-quality synthetic data",
                "Optimized for post-training",
                "Includes tool-calling examples",
                "Balanced across different capabilities"
            ],
            recommended_models=[
                "llama-3.3-70b",
                "llama-3.1-70b",
                "qwen3-4b",
                "mistral-7b"
            ],
            loading_args={
                "trust_remote_code": False,
                "streaming": True  # Recommended due to large size
            },
            url="https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1"
        ),

        # Popular instruction datasets
        "alpaca": DatasetInfo(
            name="Stanford Alpaca",
            hf_path="tatsu-lab/alpaca",
            description="52K instruction-following demonstrations generated using OpenAI's text-davinci-003.",
            dataset_type=DatasetType.INSTRUCTION,
            size="52K rows",
            splits=["train"],
            license="CC BY-NC 4.0",
            tasks=["instruction-following"],
            format="json"
        ),

        "dolly-15k": DatasetInfo(
            name="Databricks Dolly 15k",
            hf_path="databricks/databricks-dolly-15k",
            description="15K human-generated instruction-response pairs covering various categories.",
            dataset_type=DatasetType.INSTRUCTION,
            size="15K rows",
            splits=["train"],
            license="CC BY-SA 3.0",
            tasks=["instruction-following", "qa", "summarization", "creative-writing"]
        ),

        "oasst1": DatasetInfo(
            name="OpenAssistant Conversations",
            hf_path="OpenAssistant/oasst1",
            description="Human-generated, human-annotated assistant-style conversation corpus.",
            dataset_type=DatasetType.CONVERSATION,
            size="161K messages",
            splits=["train", "validation"],
            license="Apache 2.0",
            tasks=["conversation", "instruction-following"]
        ),

        # Math datasets
        "gsm8k": DatasetInfo(
            name="GSM8K",
            hf_path="gsm8k",
            description="Grade school math problems with chain-of-thought solutions.",
            dataset_type=DatasetType.MATH,
            size="8.5K problems",
            splits=["train", "test"],
            license="MIT",
            tasks=["mathematical-reasoning", "word-problems"]
        ),

        "math": DatasetInfo(
            name="MATH",
            hf_path="hendrycks/competition_math",
            description="Competition mathematics problems from AMC, AIME, and more.",
            dataset_type=DatasetType.MATH,
            size="12.5K problems",
            splits=["train", "test"],
            license="MIT",
            tasks=["mathematical-reasoning", "competition-math"]
        ),

        # Code datasets
        "codeparrot": DatasetInfo(
            name="CodeParrot Clean",
            hf_path="codeparrot/codeparrot-clean",
            description="Cleaned and deduplicated Python code from GitHub.",
            dataset_type=DatasetType.CODE,
            size="100GB",
            splits=["train", "valid"],
            license="Various (GitHub)",
            tasks=["code-generation", "code-completion"],
            format="parquet"
        ),

        "humaneval": DatasetInfo(
            name="HumanEval",
            hf_path="openai_humaneval",
            description="Hand-written Python programming problems with test cases.",
            dataset_type=DatasetType.CODE,
            size="164 problems",
            splits=["test"],
            license="MIT",
            tasks=["code-generation", "function-synthesis"]
        ),

        # QA datasets
        "squad_v2": DatasetInfo(
            name="SQuAD 2.0",
            hf_path="squad_v2",
            description="Reading comprehension dataset with unanswerable questions.",
            dataset_type=DatasetType.QA,
            size="150K questions",
            splits=["train", "validation"],
            license="CC BY-SA 4.0",
            tasks=["reading-comprehension", "extractive-qa"]
        ),

        # Tool calling datasets
        "toolbench": DatasetInfo(
            name="ToolBench",
            hf_path="toolbench/toolbench",
            description="Dataset for training models to use external tools and APIs.",
            dataset_type=DatasetType.TOOL_CALLING,
            size="16K examples",
            splits=["train", "test"],
            license="Apache 2.0",
            tasks=["tool-use", "api-calling", "function-calling"],
            special_features=["Real API schemas", "Multi-step reasoning"]
        ),

        # Mixed/comprehensive datasets
        "flan_v2": DatasetInfo(
            name="FLAN v2",
            hf_path="conceptofmind/flan2_v2",
            description="Collection of tasks formatted with instructions from FLAN.",
            dataset_type=DatasetType.MIXED,
            size="15M examples",
            splits=["train"],
            license="Apache 2.0",
            tasks=["instruction-following", "qa", "reasoning", "classification"]
        ),
    }

    @classmethod
    def get_dataset(cls, name: str) -> Optional[DatasetInfo]:
        """Get dataset information by name."""
        return cls.DATASETS.get(name)

    @classmethod
    def list_datasets(cls, dataset_type: Optional[DatasetType] = None) -> List[DatasetInfo]:
        """List all datasets, optionally filtered by type."""
        datasets = list(cls.DATASETS.values())

        if dataset_type:
            datasets = [d for d in datasets if d.dataset_type == dataset_type]

        return datasets

    @classmethod
    def search_datasets(cls, query: str) -> List[DatasetInfo]:
        """Search datasets by name or description."""
        query_lower = query.lower()
        results = []

        for dataset in cls.DATASETS.values():
            if (query_lower in dataset.name.lower() or
                query_lower in dataset.description.lower() or
                any(query_lower in task for task in dataset.tasks)):
                results.append(dataset)

        return results

    @classmethod
    def get_datasets_by_task(cls, task: str) -> List[DatasetInfo]:
        """Get datasets that support a specific task."""
        return [d for d in cls.DATASETS.values() if task in d.tasks]

    @classmethod
    def get_dataset_loading_code(cls, name: str) -> str:
        """Generate code snippet for loading a dataset."""
        dataset = cls.get_dataset(name)
        if not dataset:
            return f"# Dataset '{name}' not found in registry"

        code = f"""# Load {dataset.name}
from datasets import load_dataset

# Load the dataset
dataset = load_dataset(
    "{dataset.hf_path}","""

        if dataset.loading_args:
            for key, value in dataset.loading_args.items():
                code += f"\n    {key}={value},"

        code += "\n)\n"

        if dataset.splits:
            code += f"\n# Available splits: {', '.join(dataset.splits)}"

        if dataset.dataset_type == DatasetType.MIXED and "nemotron" in name:
            code += """

# For Nemotron, you can load specific splits:
# chat_data = load_dataset("{}", split="chat")
# code_data = load_dataset("{}", split="code")
# math_data = load_dataset("{}", split="math")
# stem_data = load_dataset("{}", split="stem")
# tool_data = load_dataset("{}", split="tool_calling")
""".format(dataset.hf_path, dataset.hf_path, dataset.hf_path, dataset.hf_path, dataset.hf_path)

        return code


# Convenience functions
def get_nemotron_dataset() -> DatasetInfo:
    """Get the Nemotron Post-Training Dataset information."""
    return DatasetRegistry.get_dataset("nemotron-post-training")


def list_available_datasets() -> List[str]:
    """List all available dataset names."""
    return list(DatasetRegistry.DATASETS.keys())
