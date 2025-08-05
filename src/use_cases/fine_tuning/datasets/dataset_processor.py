"""
Dataset processor for fine-tuning datasets.

This module handles loading, preprocessing, and formatting datasets
for fine-tuning language models.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
import numpy as np

logger = logging.getLogger(__name__)


class DatasetProcessor:
    """
    Processor for preparing datasets for fine-tuning.
    
    Supports multiple formats (JSONL, CSV, Parquet) and various
    data schemas (instruction-following, prompt-completion, etc.).
    """
    
    SUPPORTED_FORMATS = [".jsonl", ".json", ".csv", ".parquet"]
    
    def __init__(self, 
                 tokenizer=None,
                 max_length: int = 512,
                 validation_split: float = 0.1,
                 test_split: float = 0.0,
                 seed: int = 42):
        """
        Initialize dataset processor.
        
        Args:
            tokenizer: Tokenizer to use (optional)
            max_length: Maximum sequence length
            validation_split: Fraction of data for validation
            test_split: Fraction of data for testing
            seed: Random seed for splitting
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.validation_split = validation_split
        self.test_split = test_split
        self.seed = seed
        
    def load_dataset_from_file(self, file_path: str) -> Dataset:
        """
        Load dataset from a file.
        
        Args:
            file_path: Path to dataset file
            
        Returns:
            Loaded dataset
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        logger.info(f"Loading dataset from {file_path}")
        
        if suffix in [".jsonl", ".json"]:
            data = self._load_json_data(file_path)
        elif suffix == ".csv":
            data = self._load_csv_data(file_path)
        elif suffix == ".parquet":
            data = self._load_parquet_data(file_path)
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_pandas(pd.DataFrame(data))
        
        logger.info(f"Loaded {len(dataset)} examples")
        
        return dataset
    
    def _load_json_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from JSON/JSONL file."""
        data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix == ".jsonl":
                # JSONL format: one JSON object per line
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            else:
                # Regular JSON format
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
        
        return data
    
    def _load_csv_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from CSV file."""
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    
    def _load_parquet_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from Parquet file."""
        df = pd.read_parquet(file_path)
        return df.to_dict('records')
    
    def format_dataset(self, 
                      dataset: Dataset,
                      input_format: str = "auto") -> Dataset:
        """
        Format dataset for training.
        
        Args:
            dataset: Input dataset
            input_format: Format of input data ("auto", "instruction", "prompt_completion", "text")
            
        Returns:
            Formatted dataset
        """
        if input_format == "auto":
            input_format = self._detect_format(dataset)
        
        logger.info(f"Formatting dataset with format: {input_format}")
        
        if input_format == "instruction":
            dataset = self._format_instruction_dataset(dataset)
        elif input_format == "prompt_completion":
            dataset = self._format_prompt_completion_dataset(dataset)
        elif input_format == "text":
            # Already in correct format
            pass
        else:
            raise ValueError(f"Unknown input format: {input_format}")
        
        return dataset
    
    def _detect_format(self, dataset: Dataset) -> str:
        """Detect the format of the dataset."""
        if len(dataset) == 0:
            return "text"
        
        columns = dataset.column_names
        
        if "instruction" in columns or "input" in columns:
            return "instruction"
        elif "prompt" in columns and "completion" in columns:
            return "prompt_completion"
        elif "text" in columns:
            return "text"
        else:
            # Default to treating first column as text
            logger.warning(f"Could not detect format from columns: {columns}. Using first column as text.")
            return "text"
    
    def _format_instruction_dataset(self, dataset: Dataset) -> Dataset:
        """Format instruction-following dataset."""
        
        def format_example(example):
            # Handle different instruction formats
            instruction = example.get("instruction", example.get("input", ""))
            response = example.get("response", example.get("output", ""))
            
            # Optional context/system prompt
            context = example.get("context", "")
            
            # Format as conversation
            if context:
                text = f"### Context:\n{context}\n\n"
            else:
                text = ""
            
            text += f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            
            return {"text": text}
        
        return dataset.map(format_example, remove_columns=dataset.column_names)
    
    def _format_prompt_completion_dataset(self, dataset: Dataset) -> Dataset:
        """Format prompt-completion dataset."""
        
        def format_example(example):
            prompt = example["prompt"]
            completion = example["completion"]
            
            # Ensure completion starts with a space for better generation
            if not completion.startswith(" "):
                completion = " " + completion
            
            text = f"{prompt}{completion}"
            
            return {"text": text}
        
        return dataset.map(format_example, remove_columns=dataset.column_names)
    
    def split_dataset(self, 
                     dataset: Dataset,
                     validation_split: Optional[float] = None,
                     test_split: Optional[float] = None) -> DatasetDict:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            dataset: Input dataset
            validation_split: Validation split ratio (uses self.validation_split if None)
            test_split: Test split ratio (uses self.test_split if None)
            
        Returns:
            DatasetDict with splits
        """
        validation_split = validation_split or self.validation_split
        test_split = test_split or self.test_split
        
        # Calculate split sizes
        n_samples = len(dataset)
        n_test = int(n_samples * test_split)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_test - n_val
        
        logger.info(f"Splitting dataset: train={n_train}, val={n_val}, test={n_test}")
        
        # Create indices
        indices = np.arange(n_samples)
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        
        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Create splits
        splits = {"train": dataset.select(train_indices)}
        
        if n_val > 0:
            splits["validation"] = dataset.select(val_indices)
        
        if n_test > 0:
            splits["test"] = dataset.select(test_indices)
        
        return DatasetDict(splits)
    
    def tokenize_dataset(self,
                        dataset: Dataset,
                        tokenizer=None,
                        max_length: Optional[int] = None) -> Dataset:
        """
        Tokenize dataset.
        
        Args:
            dataset: Input dataset
            tokenizer: Tokenizer to use (uses self.tokenizer if None)
            max_length: Maximum sequence length (uses self.max_length if None)
            
        Returns:
            Tokenized dataset
        """
        tokenizer = tokenizer or self.tokenizer
        if tokenizer is None:
            raise ValueError("No tokenizer provided")
        
        max_length = max_length or self.max_length
        
        def tokenize_function(examples):
            # Tokenize the text
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors=None
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Tokenize in batches
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        return tokenized_dataset
    
    def validate_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Validate dataset and return statistics.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            Dictionary with validation results and statistics
        """
        stats = {
            "num_examples": len(dataset),
            "columns": dataset.column_names,
            "issues": []
        }
        
        if len(dataset) == 0:
            stats["issues"].append("Dataset is empty")
            return stats
        
        # Check for required columns
        if "text" not in dataset.column_names:
            stats["issues"].append("Missing 'text' column")
        
        # Analyze text lengths
        if "text" in dataset.column_names:
            texts = dataset["text"]
            lengths = [len(text) for text in texts]
            
            stats["text_stats"] = {
                "min_length": min(lengths),
                "max_length": max(lengths),
                "mean_length": np.mean(lengths),
                "std_length": np.std(lengths),
                "empty_texts": sum(1 for l in lengths if l == 0)
            }
            
            # Check for issues
            if stats["text_stats"]["empty_texts"] > 0:
                stats["issues"].append(f"{stats['text_stats']['empty_texts']} empty texts found")
            
            if stats["text_stats"]["max_length"] > 10000:
                stats["issues"].append("Very long texts found (>10k chars)")
        
        # If tokenizer is available, check token counts
        if self.tokenizer and "text" in dataset.column_names:
            sample_texts = dataset["text"][:100]  # Check first 100
            token_counts = []
            
            for text in sample_texts:
                tokens = self.tokenizer.encode(text)
                token_counts.append(len(tokens))
            
            stats["token_stats"] = {
                "min_tokens": min(token_counts),
                "max_tokens": max(token_counts),
                "mean_tokens": np.mean(token_counts),
                "over_max_length": sum(1 for c in token_counts if c > self.max_length)
            }
            
            if stats["token_stats"]["over_max_length"] > 0:
                pct = stats["token_stats"]["over_max_length"] / len(sample_texts) * 100
                stats["issues"].append(f"{pct:.1f}% of samples exceed max_length ({self.max_length})")
        
        return stats
    
    def clean_dataset(self, dataset: Dataset) -> Dataset:
        """
        Clean dataset by removing problematic examples.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Cleaned dataset
        """
        initial_size = len(dataset)
        
        # Remove empty texts
        if "text" in dataset.column_names:
            dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
        
        # Remove duplicates
        if "text" in dataset.column_names:
            seen = set()
            def is_unique(example):
                text = example["text"]
                if text in seen:
                    return False
                seen.add(text)
                return True
            
            dataset = dataset.filter(is_unique)
        
        final_size = len(dataset)
        if final_size < initial_size:
            logger.info(f"Cleaned dataset: {initial_size} -> {final_size} examples")
        
        return dataset
    
    def save_dataset(self, 
                    dataset: Union[Dataset, DatasetDict],
                    output_dir: str,
                    format: str = "jsonl"):
        """
        Save dataset to disk.
        
        Args:
            dataset: Dataset to save
            output_dir: Output directory
            format: Output format ("jsonl", "json", "csv", "parquet")
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if isinstance(dataset, DatasetDict):
            # Save each split
            for split_name, split_data in dataset.items():
                output_file = output_dir / f"{split_name}.{format}"
                self._save_single_dataset(split_data, output_file, format)
        else:
            # Save single dataset
            output_file = output_dir / f"dataset.{format}"
            self._save_single_dataset(dataset, output_file, format)
    
    def _save_single_dataset(self, dataset: Dataset, output_file: Path, format: str):
        """Save a single dataset split."""
        logger.info(f"Saving dataset to {output_file}")
        
        if format == "jsonl":
            dataset.to_json(output_file, lines=True)
        elif format == "json":
            dataset.to_json(output_file)
        elif format == "csv":
            dataset.to_csv(output_file)
        elif format == "parquet":
            dataset.to_parquet(output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def prepare_for_training(self,
                           file_path: str,
                           tokenizer=None,
                           input_format: str = "auto",
                           clean: bool = True,
                           validate: bool = True) -> DatasetDict:
        """
        Complete pipeline to prepare dataset for training.
        
        Args:
            file_path: Path to dataset file
            tokenizer: Tokenizer to use
            input_format: Format of input data
            clean: Whether to clean dataset
            validate: Whether to validate dataset
            
        Returns:
            DatasetDict ready for training
        """
        # Load dataset
        dataset = self.load_dataset_from_file(file_path)
        
        # Format dataset
        dataset = self.format_dataset(dataset, input_format)
        
        # Clean if requested
        if clean:
            dataset = self.clean_dataset(dataset)
        
        # Validate if requested
        if validate:
            stats = self.validate_dataset(dataset)
            if stats["issues"]:
                logger.warning(f"Dataset issues found: {stats['issues']}")
        
        # Split dataset
        dataset_dict = self.split_dataset(dataset)
        
        # Tokenize if tokenizer provided
        if tokenizer:
            self.tokenizer = tokenizer
            for split_name in dataset_dict:
                dataset_dict[split_name] = self.tokenize_dataset(dataset_dict[split_name])
        
        return dataset_dict