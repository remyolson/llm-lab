"""
Dataset Preparation Pipeline for Fine-Tuning

This module provides automatic dataset preprocessing with format detection,
validation, splitting, and augmentation capabilities. It supports multiple
input formats and integrates with the recipe system for custom preprocessing.

Example:
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.prepare_dataset(
        "path/to/data.jsonl",
        recipe_config=dataset_config
    )
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer

import datasets
from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Report containing dataset quality metrics and issues."""

    total_samples: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0
    avg_text_length: float = 0.0
    min_text_length: int = 0
    max_text_length: int = 0
    empty_samples: int = 0
    duplicate_samples: int = 0
    issues: List[Dict[str, Any]] = field(default_factory=list)
    token_statistics: Dict[str, Any] = field(default_factory=dict)
    label_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str | Any]:
        """Convert report to dictionary."""
        return {
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "invalid_samples": self.invalid_samples,
            "avg_text_length": self.avg_text_length,
            "min_text_length": self.min_text_length,
            "max_text_length": self.max_text_length,
            "empty_samples": self.empty_samples,
            "duplicate_samples": self.duplicate_samples,
            "issues": self.issues,
            "token_statistics": self.token_statistics,
            "label_distribution": self.label_distribution,
        }


class DataPreprocessor:
    """Handles dataset preparation for fine-tuning workflows."""

    # Supported file formats
    SUPPORTED_FORMATS = {
        ".json": "json",
        ".jsonl": "jsonl",
        ".csv": "csv",
        ".tsv": "tsv",
        ".txt": "text",
        ".parquet": "parquet",
    }

    # Common dataset formats and their expected fields
    KNOWN_FORMATS = {
        "alpaca": ["instruction", "input", "output"],
        "chattml": ["messages"],
        "openai": ["prompt", "completion"],
        "dolly": ["instruction", "context", "response"],
        "vicuna": ["conversations"],
        "plain": ["text"],
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize DataPreprocessor.

        Args:
            cache_dir: Directory for caching processed datasets
        """
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path.home() / ".cache" / "lllm_lab" / "datasets"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._format_detectors = self._setup_format_detectors()

    def _setup_format_detectors(self) -> Dict[str | Callable]:
        """Setup format detection functions."""
        return {
            "alpaca": self._detect_alpaca_format,
            "chattml": self._detect_chattml_format,
            "openai": self._detect_openai_format,
            "dolly": self._detect_dolly_format,
            "vicuna": self._detect_vicuna_format,
            "plain": self._detect_plain_format,
        }

    def detect_format(self, data_path: str | Path) -> Tuple[str | str]:
        """Detect the format of the dataset.

        Args:
            data_path: Path to the dataset file

        Returns:
            Tuple of (file_format, data_format)
        """
        data_path = Path(data_path)

        # Detect file format
        file_ext = data_path.suffix.lower()
        file_format = self.SUPPORTED_FORMATS.get(file_ext, "unknown")

        if file_format == "unknown":
            raise ValueError(f"Unsupported file format: {file_ext}")

        # Load sample data to detect data format
        sample_data = self._load_sample(data_path, file_format)

        # Detect data format
        data_format = "unknown"
        for format_name, detector in self._format_detectors.items():
            if detector(sample_data):
                data_format = format_name
                break

        logger.info(f"Detected formats - File: {file_format}, Data: {data_format}")
        return file_format, data_format

    def _load_sample(self, data_path: Path, file_format: str, n_samples: int = 5) -> List[Dict]:
        """Load sample data for format detection."""
        samples = []

        try:
            if file_format == "jsonl":
                with open(data_path, encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= n_samples:
                            break
                        samples.append(json.loads(line.strip()))

            elif file_format == "json":
                with open(data_path, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples = data[:n_samples]
                    else:
                        samples = [data]

            elif file_format in ["csv", "tsv"]:
                delimiter = "\t" if file_format == "tsv" else ","
                df = pd.read_csv(data_path, delimiter=delimiter, nrows=n_samples)
                samples = df.to_dict("records")

            elif file_format == "parquet":
                df = pd.read_parquet(data_path).head(n_samples)
                samples = df.to_dict("records")

            elif file_format == "text":
                with open(data_path, encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= n_samples:
                            break
                        samples.append({"text": line.strip()})

        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
            raise

        return samples

    def _detect_alpaca_format(self, samples: List[Dict]) -> bool:
        """Detect Alpaca-style format."""
        if not samples:
            return False

        required_fields = {"instruction", "output"}
        optional_fields = {"input"}

        for sample in samples:
            if not required_fields.issubset(sample.keys()):
                return False

        return True

    def _detect_chattml_format(self, samples: List[Dict]) -> bool:
        """Detect ChatML format."""
        if not samples:
            return False

        for sample in samples:
            if "messages" not in sample:
                return False

            messages = sample["messages"]
            if not isinstance(messages, list) or not messages:
                return False

            # Check message structure
            for msg in messages:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    return False

        return True

    def _detect_openai_format(self, samples: List[Dict]) -> bool:
        """Detect OpenAI fine-tuning format."""
        if not samples:
            return False

        required_fields = {"prompt", "completion"}

        for sample in samples:
            if not required_fields.issubset(sample.keys()):
                return False

        return True

    def _detect_dolly_format(self, samples: List[Dict]) -> bool:
        """Detect Dolly-style format."""
        if not samples:
            return False

        required_fields = {"instruction", "response"}

        for sample in samples:
            if not required_fields.issubset(sample.keys()):
                return False

        return True

    def _detect_vicuna_format(self, samples: List[Dict]) -> bool:
        """Detect Vicuna conversation format."""
        if not samples:
            return False

        for sample in samples:
            if "conversations" not in sample:
                return False

            conversations = sample["conversations"]
            if not isinstance(conversations, list) or not conversations:
                return False

        return True

    def _detect_plain_format(self, samples: List[Dict]) -> bool:
        """Detect plain text format."""
        if not samples:
            return False

        for sample in samples:
            if "text" not in sample:
                return False

        return True

    def load_dataset(
        self,
        data_path: str | Path,
        file_format: Optional[str] = None,
        data_format: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> Dataset:
        """Load dataset from file.

        Args:
            data_path: Path to dataset file
            file_format: File format (auto-detected if None)
            data_format: Data format (auto-detected if None)
            max_samples: Maximum number of samples to load

        Returns:
            Loaded dataset
        """
        data_path = Path(data_path)

        # Auto-detect formats if not provided
        if file_format is None or data_format is None:
            detected_file_format, detected_data_format = self.detect_format(data_path)
            file_format = file_format or detected_file_format
            data_format = data_format or detected_data_format

        logger.info(f"Loading dataset from {data_path} (format: {file_format})")

        # Load based on file format
        if file_format == "jsonl":
            dataset = load_dataset("json", data_files=str(data_path), split="train")
        elif file_format == "json":
            dataset = load_dataset("json", data_files=str(data_path), split="train")
        elif file_format == "csv":
            dataset = load_dataset("csv", data_files=str(data_path), split="train")
        elif file_format == "tsv":
            dataset = load_dataset("csv", data_files=str(data_path), delimiter="\t", split="train")
        elif file_format == "parquet":
            dataset = load_dataset("parquet", data_files=str(data_path), split="train")
        elif file_format == "text":
            dataset = load_dataset("text", data_files=str(data_path), split="train")
        else:
            # Try HuggingFace dataset
            dataset = load_dataset(str(data_path), split="train")

        # Limit samples if requested
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        return dataset

    def validate_dataset(
        self, dataset: Dataset, data_format: str, tokenizer: Optional[PreTrainedTokenizer] = None
    ) -> DataQualityReport:
        """Validate dataset quality and generate report.

        Args:
            dataset: Dataset to validate
            data_format: Format of the data
            tokenizer: Optional tokenizer for token statistics

        Returns:
            Data quality report
        """
        report = DataQualityReport()
        report.total_samples = len(dataset)

        # Track issues and statistics
        text_lengths = []
        seen_hashes = set()

        for idx, sample in enumerate(dataset):
            try:
                # Extract text based on format
                text = self._extract_text(sample, data_format)

                if not text or not text.strip():
                    report.empty_samples += 1
                    report.issues.append({"index": idx, "type": "empty_text", "sample": sample})
                    continue

                # Check for duplicates
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in seen_hashes:
                    report.duplicate_samples += 1
                    report.issues.append({"index": idx, "type": "duplicate", "hash": text_hash})
                seen_hashes.add(text_hash)

                # Track text length
                text_lengths.append(len(text))

                # Validate format-specific requirements
                if not self._validate_sample_format(sample, data_format):
                    report.invalid_samples += 1
                    report.issues.append({"index": idx, "type": "invalid_format", "sample": sample})
                else:
                    report.valid_samples += 1

            except Exception as e:
                report.invalid_samples += 1
                report.issues.append(
                    {"index": idx, "type": "processing_error", "error": str(e), "sample": sample}
                )

        # Calculate statistics
        if text_lengths:
            report.avg_text_length = np.mean(text_lengths)
            report.min_text_length = min(text_lengths)
            report.max_text_length = max(text_lengths)

        # Token statistics if tokenizer provided
        if tokenizer and text_lengths:
            token_counts = []
            for sample in dataset.select(range(min(100, len(dataset)))):  # Sample for efficiency
                text = self._extract_text(sample, data_format)
                if text:
                    tokens = tokenizer.encode(text)
                    token_counts.append(len(tokens))

            if token_counts:
                report.token_statistics = {
                    "avg_tokens": np.mean(token_counts),
                    "min_tokens": min(token_counts),
                    "max_tokens": max(token_counts),
                    "std_tokens": np.std(token_counts),
                }

        logger.info(
            f"Validation complete: {report.valid_samples}/{report.total_samples} valid samples"
        )
        return report

    def _extract_text(self, sample: Dict[str, Any], data_format: str) -> str:
        """Extract text from sample based on format."""
        if data_format == "alpaca":
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")
            output = sample.get("output", "")
            if input_text:
                return f"{instruction}\n{input_text}\n{output}"
            return f"{instruction}\n{output}"

        elif data_format == "chattml":
            messages = sample.get("messages", [])
            text_parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                text_parts.append(f"{role}: {content}")
            return "\n".join(text_parts)

        elif data_format == "openai":
            prompt = sample.get("prompt", "")
            completion = sample.get("completion", "")
            return f"{prompt}{completion}"

        elif data_format == "dolly":
            instruction = sample.get("instruction", "")
            context = sample.get("context", "")
            response = sample.get("response", "")
            if context:
                return f"{instruction}\n{context}\n{response}"
            return f"{instruction}\n{response}"

        elif data_format == "vicuna":
            conversations = sample.get("conversations", [])
            text_parts = []
            for conv in conversations:
                if isinstance(conv, dict):
                    from_role = conv.get("from", "")
                    value = conv.get("value", "")
                    text_parts.append(f"{from_role}: {value}")
            return "\n".join(text_parts)

        elif data_format == "plain":
            return sample.get("text", "")

        return str(sample)

    def _validate_sample_format(self, sample: Dict[str, Any], data_format: str) -> bool:
        """Validate sample against expected format."""
        if data_format in self.KNOWN_FORMATS:
            required_fields = self.KNOWN_FORMATS[data_format]

            # Special handling for nested structures
            if data_format == "chattml" and "messages" in sample:
                messages = sample["messages"]
                if not isinstance(messages, list) or not messages:
                    return False
                for msg in messages:
                    if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                        return False
                return True

            elif data_format == "vicuna" and "conversations" in sample:
                conversations = sample["conversations"]
                if not isinstance(conversations, list) or not conversations:
                    return False
                return True

            # Check required fields
            for field in required_fields:
                if field not in sample:
                    return False

        return True

    def split_dataset(
        self,
        dataset: Dataset,
        split_ratios: Dict[str, float],
        stratify_column: Optional[str] = None,
        seed: int = 42,
    ) -> DatasetDict:
        """Split dataset into train/validation/test sets.

        Args:
            dataset: Dataset to split
            split_ratios: Dictionary with 'train', 'validation', 'test' ratios
            stratify_column: Optional column name for stratified splitting
            seed: Random seed

        Returns:
            DatasetDict with splits
        """
        # Validate split ratios
        total_ratio = sum(split_ratios.values())
        if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

        # Convert to pandas for splitting
        df = dataset.to_pandas()

        # Calculate split sizes
        n_samples = len(df)
        train_size = int(n_samples * split_ratios.get("train", 0.8))
        val_size = int(n_samples * split_ratios.get("validation", 0.1))
        test_size = n_samples - train_size - val_size

        # Stratify if column specified
        stratify = (
            df[stratify_column] if stratify_column and stratify_column in df.columns else None
        )

        # First split: train+val vs test
        if test_size > 0:
            train_val_df, test_df = train_test_split(
                df, test_size=test_size, random_state=seed, stratify=stratify
            )
        else:
            train_val_df = df
            test_df = pd.DataFrame()

        # Second split: train vs val
        if val_size > 0 and len(train_val_df) > 0:
            val_ratio = val_size / (train_size + val_size)
            stratify_tv = (
                train_val_df[stratify_column]
                if stratify_column and stratify_column in train_val_df.columns
                else None
            )

            train_df, val_df = train_test_split(
                train_val_df, test_size=val_ratio, random_state=seed, stratify=stratify_tv
            )
        else:
            train_df = train_val_df
            val_df = pd.DataFrame()

        # Convert back to datasets
        splits = {}
        if not train_df.empty:
            splits["train"] = Dataset.from_pandas(train_df, preserve_index=False)
        if not val_df.empty:
            splits["validation"] = Dataset.from_pandas(val_df, preserve_index=False)
        if not test_df.empty:
            splits["test"] = Dataset.from_pandas(test_df, preserve_index=False)

        dataset_dict = DatasetDict(splits)

        logger.info(
            f"Dataset split: Train={len(splits.get('train', []))}, "
            f"Val={len(splits.get('validation', []))}, "
            f"Test={len(splits.get('test', []))}"
        )

        return dataset_dict

    def apply_preprocessing(
        self,
        dataset: Dataset,
        data_format: str,
        preprocessing_config: Dict[str, Any],
        custom_preprocessor: Optional[Callable] = None,
    ) -> Dataset:
        """Apply preprocessing transformations to dataset.

        Args:
            dataset: Dataset to preprocess
            data_format: Format of the data
            preprocessing_config: Preprocessing configuration
            custom_preprocessor: Optional custom preprocessing function

        Returns:
            Preprocessed dataset
        """
        # Apply custom preprocessor if provided
        if custom_preprocessor:
            logger.info("Applying custom preprocessing")
            dataset = dataset.map(custom_preprocessor, batched=True)

        # Apply format-specific preprocessing
        if data_format == "alpaca":
            dataset = self._preprocess_alpaca(dataset, preprocessing_config)
        elif data_format == "chattml":
            dataset = self._preprocess_chattml(dataset, preprocessing_config)
        elif data_format == "openai":
            dataset = self._preprocess_openai(dataset, preprocessing_config)

        # Apply common preprocessing
        if preprocessing_config.get("lowercase", False):
            logger.info("Converting to lowercase")
            dataset = dataset.map(
                lambda x: {k: v.lower() if isinstance(v, str) else v for k, v in x.items()}
            )

        if preprocessing_config.get("remove_html", False):
            logger.info("Removing HTML tags")
            dataset = dataset.map(self._remove_html_tags)

        if preprocessing_config.get("normalize_whitespace", False):
            logger.info("Normalizing whitespace")
            dataset = dataset.map(self._normalize_whitespace)

        return dataset

    def _preprocess_alpaca(self, dataset: Dataset, config: Dict[str, Any]) -> Dataset:
        """Preprocess Alpaca format dataset."""

        def format_alpaca(examples):
            # Combine instruction and input
            prompts = []
            for i in range(len(examples["instruction"])):
                instruction = examples["instruction"][i]
                input_text = examples.get("input", [""] * len(examples["instruction"]))[i]

                if input_text:
                    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
                else:
                    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

                prompts.append(prompt)

            return {"prompt": prompts}

        if config.get("add_prompt_template", True):
            dataset = dataset.map(format_alpaca, batched=True)

        return dataset

    def _preprocess_chattml(self, dataset: Dataset, config: Dict[str, Any]) -> Dataset:
        """Preprocess ChatML format dataset."""

        def format_chattml(examples):
            formatted_texts = []

            for messages in examples["messages"]:
                text_parts = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")

                    if role == "system":
                        text_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
                    elif role == "user":
                        text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
                    elif role == "assistant":
                        text_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

                formatted_texts.append("\n".join(text_parts))

            return {"text": formatted_texts}

        if config.get("add_special_tokens", True):
            dataset = dataset.map(format_chattml, batched=True)

        return dataset

    def _preprocess_openai(self, dataset: Dataset, config: Dict[str, Any]) -> Dataset:
        """Preprocess OpenAI format dataset."""

        def format_openai(examples):
            texts = []
            for i in range(len(examples["prompt"])):
                prompt = examples["prompt"][i]
                completion = examples["completion"][i]

                # Ensure completion starts with a space
                if not completion.startswith(" "):
                    completion = " " + completion

                texts.append(prompt + completion)

            return {"text": texts}

        if config.get("combine_prompt_completion", True):
            dataset = dataset.map(format_openai, batched=True)

        return dataset

    def _remove_html_tags(self, example: Dict[str, Any]) -> Dict[str | Any]:
        """Remove HTML tags from text fields."""
        html_pattern = re.compile("<.*?>")

        for key, value in example.items():
            if isinstance(value, str):
                example[key] = html_pattern.sub("", value)

        return example

    def _normalize_whitespace(self, example: Dict[str, Any]) -> Dict[str | Any]:
        """Normalize whitespace in text fields."""
        for key, value in example.items():
            if isinstance(value, str):
                # Replace multiple spaces with single space
                value = re.sub(r"\s+", " ", value)
                # Strip leading/trailing whitespace
                example[key] = value.strip()

        return example

    def tokenize_dataset(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        tokenizer_config: Dict[str, Any],
        text_column: str = "text",
    ) -> Dataset:
        """Tokenize dataset using provided tokenizer.

        Args:
            dataset: Dataset to tokenize
            tokenizer: Tokenizer to use
            tokenizer_config: Tokenization configuration
            text_column: Name of text column

        Returns:
            Tokenized dataset
        """
        max_length = tokenizer_config.get("max_length", 512)
        padding = tokenizer_config.get("padding", "max_length")
        truncation = tokenizer_config.get("truncation", True)

        def tokenize_function(examples):
            return tokenizer(
                examples[text_column],
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=None,
            )

        # Remove columns that will be replaced by tokenization
        columns_to_remove = dataset.column_names
        if "labels" in columns_to_remove:
            columns_to_remove.remove("labels")

        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=columns_to_remove
        )

        # Add labels for causal LM if not present
        if "labels" not in tokenized_dataset.column_names:
            tokenized_dataset = tokenized_dataset.map(
                lambda x: {"labels": x["input_ids"]}, batched=True
            )

        return tokenized_dataset

    def augment_dataset(self, dataset: Dataset, augmentation_config: Dict[str, Any]) -> Dataset:
        """Apply data augmentation techniques.

        Args:
            dataset: Dataset to augment
            augmentation_config: Augmentation configuration

        Returns:
            Augmented dataset
        """
        augmented_samples = []

        if augmentation_config.get("paraphrase", False):
            # This would require a paraphrasing model
            logger.info("Paraphrase augmentation not yet implemented")

        if augmentation_config.get("back_translation", False):
            # This would require translation models
            logger.info("Back-translation augmentation not yet implemented")

        if augmentation_config.get("synonym_replacement", False):
            # Simple synonym replacement
            logger.info("Applying synonym replacement augmentation")
            # Implementation would go here

        # Add augmented samples to dataset
        if augmented_samples:
            augmented_dataset = Dataset.from_list(augmented_samples)
            dataset = datasets.concatenate_datasets([dataset, augmented_dataset])

        return dataset

    def prepare_dataset(
        self,
        data_path: str | Path,
        recipe_config: Dict[str, Any],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        custom_preprocessor: Optional[Callable] = None,
        cache: bool = True,
    ) -> Dict[str | Any]:
        """Complete dataset preparation pipeline.

        Args:
            data_path: Path to dataset
            recipe_config: Dataset configuration from recipe
            tokenizer: Optional tokenizer
            custom_preprocessor: Optional custom preprocessing function
            cache: Whether to cache processed dataset

        Returns:
            Dictionary with processed dataset and metadata
        """
        # Extract configuration
        file_format = recipe_config.get("format")
        data_format = recipe_config.get("data_format")
        max_samples = recipe_config.get("max_samples")
        split_ratios = recipe_config.get(
            "split_ratios", {"train": 0.8, "validation": 0.1, "test": 0.1}
        )
        preprocessing = recipe_config.get("preprocessing", {})
        tokenizer_config = recipe_config.get("tokenizer_config", {})

        # Generate cache key
        cache_key = None
        if cache:
            config_str = json.dumps(recipe_config, sort_keys=True)
            cache_key = hashlib.md5(f"{data_path}{config_str}".encode()).hexdigest()
            cache_path = self.cache_dir / f"{cache_key}.dataset"

            if cache_path.exists():
                logger.info(f"Loading cached dataset from {cache_path}")
                return datasets.load_from_disk(str(cache_path))

        # Load dataset
        dataset = self.load_dataset(
            data_path, file_format=file_format, data_format=data_format, max_samples=max_samples
        )

        # Auto-detect format if not specified
        if not data_format:
            _, data_format = self.detect_format(data_path)

        # Validate dataset
        quality_report = self.validate_dataset(dataset, data_format, tokenizer)

        # Apply preprocessing
        dataset = self.apply_preprocessing(dataset, data_format, preprocessing, custom_preprocessor)

        # Split dataset
        dataset_dict = self.split_dataset(dataset, split_ratios)

        # Tokenize if tokenizer provided
        if tokenizer:
            logger.info("Tokenizing dataset")
            for split_name in dataset_dict:
                dataset_dict[split_name] = self.tokenize_dataset(
                    dataset_dict[split_name],
                    tokenizer,
                    tokenizer_config,
                    text_column="text"
                    if "text" in dataset_dict[split_name].column_names
                    else "prompt",
                )

        # Apply augmentation if configured
        if preprocessing.get("augmentation"):
            train_dataset = dataset_dict.get("train")
            if train_dataset:
                dataset_dict["train"] = self.augment_dataset(
                    train_dataset, preprocessing["augmentation"]
                )

        # Cache processed dataset
        if cache and cache_key:
            logger.info(f"Caching processed dataset to {cache_path}")
            dataset_dict.save_to_disk(str(cache_path))

        # Prepare result
        result = {
            "dataset": dataset_dict,
            "quality_report": quality_report.to_dict(),
            "metadata": {
                "file_format": file_format,
                "data_format": data_format,
                "total_samples": sum(len(split) for split in dataset_dict.values()),
                "splits": {name: len(split) for name, split in dataset_dict.items()},
                "cache_key": cache_key,
            },
        }

        return result


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Example recipe configuration
    dataset_config = {
        "name": "alpaca_cleaned",
        "path": "path/to/alpaca_data.jsonl",
        "format": "jsonl",
        "data_format": "alpaca",
        "split_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "max_samples": 1000,
        "preprocessing": {
            "lowercase": False,
            "remove_html": True,
            "normalize_whitespace": True,
            "add_prompt_template": True,
        },
        "tokenizer_config": {"max_length": 512, "padding": "max_length", "truncation": True},
    }

    # Load tokenizer (example)
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    # Prepare dataset
    # result = preprocessor.prepare_dataset(
    #     dataset_config["path"],
    #     dataset_config,
    #     tokenizer=tokenizer
    # )

    # print(f"Dataset prepared: {result['metadata']}")
    # print(f"Quality report: {result['quality_report']}")
