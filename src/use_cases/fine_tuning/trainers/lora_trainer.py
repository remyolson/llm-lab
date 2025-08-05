"""
LoRA trainer implementation for parameter-efficient fine-tuning.

This module implements LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA)
training using the PEFT library.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
from datasets import load_dataset, Dataset as HFDataset

from .base_trainer import BaseTrainer, TrainingMetrics

logger = logging.getLogger(__name__)


class LoRATrainer(BaseTrainer):
    """
    Trainer for LoRA/QLoRA fine-tuning.
    
    This trainer implements parameter-efficient fine-tuning using Low-Rank
    Adaptation (LoRA) with optional quantization support (QLoRA).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LoRA trainer.
        
        Args:
            config: Training configuration including LoRA-specific parameters
        """
        super().__init__(config)
        
        # LoRA-specific config
        self.lora_config = self._create_lora_config()
        self.tokenizer = None
        self.use_qlora = config.get("use_qlora", False)
        self.bits = config.get("bits", 4) if self.use_qlora else None
        
        # Model and data parameters
        self.max_seq_length = config.get("max_seq_length", 512)
        self.model_name = config["model_name"]
        
    def _create_lora_config(self) -> LoraConfig:
        """Create LoRA configuration from training config."""
        lora_params = self.config.get("lora_params", {})
        
        # Default LoRA parameters
        default_params = {
            "r": 8,  # Rank
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
        }
        
        # Merge with user config
        lora_params = {**default_params, **lora_params}
        
        logger.info(f"LoRA configuration: {lora_params}")
        
        return LoraConfig(**lora_params)
    
    def load_model(self) -> torch.nn.Module:
        """
        Load and prepare model with LoRA adapters.
        
        Returns:
            Model with LoRA adapters attached
        """
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization for QLoRA
        bnb_config = None
        if self.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True if self.bits == 4 else False,
                load_in_8bit=True if self.bits == 8 else False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            logger.info(f"Using QLoRA with {self.bits}-bit quantization")
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto" if self.use_qlora else None,
            torch_dtype=torch.float16 if self.use_qlora else torch.float32,
            trust_remote_code=self.config.get("trust_remote_code", False)
        )
        
        # Prepare model for k-bit training if using QLoRA
        if self.use_qlora:
            model = prepare_model_for_kbit_training(model)
        
        # Add LoRA adapters
        model = get_peft_model(model, self.lora_config)
        
        # Print trainable parameters
        self._print_trainable_parameters(model)
        
        return model
    
    def _print_trainable_parameters(self, model):
        """Print the number of trainable parameters."""
        trainable_params = 0
        all_param = 0
        
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        percentage = 100 * trainable_params / all_param
        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable%: {percentage:.2f}%"
        )
    
    def prepare_dataset(self, dataset_path: str) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Prepare dataset for training.
        
        Args:
            dataset_path: Path to dataset file or HuggingFace dataset name
            
        Returns:
            Tuple of (train_dataloader, eval_dataloader)
        """
        logger.info(f"Loading dataset from: {dataset_path}")
        
        # Load dataset
        if os.path.exists(dataset_path):
            # Load from file
            if dataset_path.endswith(".jsonl"):
                dataset = load_dataset("json", data_files=dataset_path)["train"]
            elif dataset_path.endswith(".csv"):
                dataset = load_dataset("csv", data_files=dataset_path)["train"]
            elif dataset_path.endswith(".parquet"):
                dataset = load_dataset("parquet", data_files=dataset_path)["train"]
            else:
                raise ValueError(f"Unsupported file format: {dataset_path}")
        else:
            # Try loading from HuggingFace Hub
            dataset = load_dataset(dataset_path)["train"]
        
        # Split into train/eval if needed
        if self.config.get("validation_split", 0.1) > 0:
            split_dataset = dataset.train_test_split(
                test_size=self.config["validation_split"],
                seed=self.config.get("seed", 42)
            )
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
        else:
            train_dataset = dataset
            eval_dataset = None
        
        # Tokenize datasets
        train_dataset = self._tokenize_dataset(train_dataset)
        if eval_dataset:
            eval_dataset = self._tokenize_dataset(eval_dataset)
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.get("batch_size", 4),
            shuffle=True,
            collate_fn=data_collator,
            num_workers=self.config.get("num_workers", 0),
            pin_memory=True if self.device.type == "cuda" else False
        )
        
        eval_dataloader = None
        if eval_dataset:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.config.get("eval_batch_size", self.config.get("batch_size", 4)),
                shuffle=False,
                collate_fn=data_collator,
                num_workers=self.config.get("num_workers", 0),
                pin_memory=True if self.device.type == "cuda" else False
            )
        
        logger.info(f"Train samples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Eval samples: {len(eval_dataset)}")
        
        return train_dataloader, eval_dataloader
    
    def _tokenize_dataset(self, dataset: HFDataset) -> HFDataset:
        """Tokenize a dataset."""
        
        def tokenize_function(examples):
            # Handle different input formats
            if "text" in examples:
                texts = examples["text"]
            elif "instruction" in examples and "response" in examples:
                # Instruction-following format
                texts = []
                for inst, resp in zip(examples["instruction"], examples["response"]):
                    text = f"### Instruction:\n{inst}\n\n### Response:\n{resp}"
                    texts.append(text)
            elif "prompt" in examples and "completion" in examples:
                # Prompt-completion format
                texts = []
                for prompt, completion in zip(examples["prompt"], examples["completion"]):
                    text = f"{prompt}{completion}"
                    texts.append(text)
            else:
                raise ValueError("Dataset must have 'text', 'instruction'/'response', or 'prompt'/'completion' fields")
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_length,
                return_tensors=None
            )
            
            # Set labels (same as input_ids for causal LM)
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Tokenize in batches
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset
    
    def compute_loss(self, model: torch.nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute loss for a batch.
        
        Args:
            model: The model
            batch: Batch containing input_ids, attention_mask, labels
            
        Returns:
            Loss tensor
        """
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        return outputs.loss
    
    def save_checkpoint(self, epoch: int, metrics: TrainingMetrics):
        """
        Save LoRA checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Training metrics
        """
        # Save LoRA adapters only
        adapter_path = self.checkpoint_dir / f"adapter_epoch_{epoch}"
        self.model.save_pretrained(adapter_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(adapter_path)
        
        # Save training state using parent method
        super().save_checkpoint(epoch, metrics)
        
        logger.info(f"Saved LoRA adapters to: {adapter_path}")
    
    def merge_and_save(self, output_path: str):
        """
        Merge LoRA adapters with base model and save.
        
        Args:
            output_path: Path to save merged model
        """
        logger.info("Merging LoRA adapters with base model...")
        
        # Merge adapters
        merged_model = self.model.merge_and_unload()
        
        # Save merged model
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        logger.info(f"Saved merged model to: {output_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load LoRA checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        # Load training state
        super().load_checkpoint(checkpoint_path)
        
        # Load LoRA adapters
        checkpoint_dir = Path(checkpoint_path).parent
        adapter_dir = None
        
        # Find adapter directory for the epoch
        for item in checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("adapter_epoch_"):
                epoch_num = int(item.name.split("_")[-1])
                if epoch_num == self.current_epoch:
                    adapter_dir = item
                    break
        
        if adapter_dir and adapter_dir.exists():
            logger.info(f"Loading LoRA adapters from: {adapter_dir}")
            self.model = PeftModel.from_pretrained(
                self.model.get_base_model(),
                adapter_dir,
                is_trainable=True
            )
    
    def estimate_memory_usage(self) -> Dict[str, float]:
        """
        Estimate memory usage for LoRA training.
        
        Returns:
            Dictionary with memory estimates in GB
        """
        # Count parameters
        base_params = sum(p.numel() for p in self.model.get_base_model().parameters())
        lora_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate memory
        if self.use_qlora:
            # QLoRA: base model in 4/8-bit, LoRA in fp16
            base_memory_gb = base_params * (self.bits / 8) / 1024**3
            lora_memory_gb = lora_params * 2 / 1024**3  # fp16
        else:
            # Regular LoRA: base model frozen in fp32, LoRA in fp32
            base_memory_gb = base_params * 4 / 1024**3
            lora_memory_gb = lora_params * 4 / 1024**3
        
        # Optimizer states (Adam has 2 states per parameter)
        optimizer_memory_gb = lora_params * 4 * 2 / 1024**3
        
        # Gradients
        gradient_memory_gb = lora_params * 4 / 1024**3
        
        # Activations (rough estimate)
        batch_size = self.config.get("batch_size", 4)
        seq_length = self.max_seq_length
        hidden_size = 768  # Rough estimate, varies by model
        activation_memory_gb = batch_size * seq_length * hidden_size * 4 * 24 / 1024**3
        
        return {
            "base_model_gb": base_memory_gb,
            "lora_adapters_gb": lora_memory_gb,
            "optimizer_states_gb": optimizer_memory_gb,
            "gradients_gb": gradient_memory_gb,
            "activations_gb": activation_memory_gb,
            "total_estimated_gb": (
                base_memory_gb + lora_memory_gb + optimizer_memory_gb +
                gradient_memory_gb + activation_memory_gb
            )
        }