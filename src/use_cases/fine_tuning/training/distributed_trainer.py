"""
Distributed Training Setup for Fine-Tuning

This module provides distributed training capabilities supporting multiple
backends including PyTorch DDP, FSDP, and DeepSpeed for efficient multi-GPU
and multi-node training.

Example:
    trainer = DistributedTrainer(
        model=model,
        training_args=training_args,
        backend="fsdp"
    )

    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
"""

import logging
import os
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Import transformers components
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
    get_scheduler,
)

# FSDP support
try:
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        CPUOffload,
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        FullOptimStateDictConfig,
        StateDictType,
    )

    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False

# DeepSpeed support
try:
    import deepspeed
    from deepspeed import DeepSpeedConfig

    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

# Accelerate support
try:
    from accelerate import Accelerator, DistributedType
    from accelerate.utils import set_seed

    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DistributedTrainingConfig:
    """Configuration for distributed training."""

    backend: str = "ddp"  # ddp, fsdp, deepspeed, accelerate
    world_size: int = -1  # -1 for auto-detection
    rank: int = -1
    local_rank: int = -1
    master_addr: str = "localhost"
    master_port: str = "29500"

    # DDP specific
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True

    # FSDP specific
    fsdp_sharding_strategy: str = "full_shard"  # full_shard, shard_grad_op, no_shard
    fsdp_cpu_offload: bool = False
    fsdp_auto_wrap_policy: Optional[str] = None
    fsdp_min_num_params: int = 1e8
    fsdp_transformer_layer_cls_to_wrap: Optional[List[str]] = None

    # DeepSpeed specific
    deepspeed_config: Optional[Dict[str, Any]] = None
    zero_stage: int = 2  # ZeRO optimization stage (0, 1, 2, 3)

    # Mixed precision
    fp16: bool = True
    bf16: bool = False
    fp16_opt_level: str = "O1"  # O0, O1, O2, O3

    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0

    # Communication
    backend_comm: str = "nccl"  # nccl, gloo, mpi
    init_method: str = "env://"
    timeout: int = 1800  # 30 minutes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class DistributedTrainer:
    """Handles distributed training across multiple GPUs/nodes."""

    def __init__(
        self,
        model: PreTrainedModel,
        training_args: TrainingArguments,
        config: Optional[DistributedTrainingConfig] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        """Initialize distributed trainer.

        Args:
            model: Model to train
            training_args: Training arguments
            config: Distributed training configuration
            tokenizer: Optional tokenizer
        """
        self.model = model
        self.training_args = training_args
        self.config = config or DistributedTrainingConfig()
        self.tokenizer = tokenizer

        # Initialize distributed environment
        self._setup_distributed()

        # Wrap model based on backend
        self.wrapped_model = self._wrap_model()

        # Setup optimizer and scheduler
        self.optimizer = None
        self.scheduler = None

    def _setup_distributed(self):
        """Setup distributed training environment."""
        if self.config.backend == "accelerate" and ACCELERATE_AVAILABLE:
            self._setup_accelerate()
            return

        # Standard PyTorch distributed setup
        if "LOCAL_RANK" in os.environ:
            self.config.local_rank = int(os.environ["LOCAL_RANK"])
            self.config.rank = int(os.environ.get("RANK", 0))
            self.config.world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Initialize process group if not already initialized
        if not dist.is_initialized() and self.config.world_size > 1:
            dist.init_process_group(
                backend=self.config.backend_comm,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank,
                timeout=timedelta(seconds=self.config.timeout),
            )

            # Set device
            if torch.cuda.is_available():
                torch.cuda.set_device(self.config.local_rank)

        # Set seed for reproducibility
        if self.training_args.seed is not None:
            set_seed(self.training_args.seed + self.config.rank)

    def _setup_accelerate(self):
        """Setup Accelerate for distributed training."""
        self.accelerator = Accelerator(
            mixed_precision="fp16" if self.config.fp16 else "bf16" if self.config.bf16 else "no",
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            cpu=not torch.cuda.is_available(),
        )

        # Update config from accelerator
        self.config.local_rank = self.accelerator.local_process_index
        self.config.rank = self.accelerator.process_index
        self.config.world_size = self.accelerator.num_processes

    def _wrap_model(self) -> nn.Module:
        """Wrap model for distributed training based on backend."""
        if self.config.backend == "ddp":
            return self._wrap_ddp()
        elif self.config.backend == "fsdp":
            return self._wrap_fsdp()
        elif self.config.backend == "deepspeed":
            return self._wrap_deepspeed()
        elif self.config.backend == "accelerate":
            return self._wrap_accelerate()
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

    def _wrap_ddp(self) -> nn.Module:
        """Wrap model with DistributedDataParallel."""
        if torch.cuda.is_available():
            self.model = self.model.cuda(self.config.local_rank)

        model = DDP(
            self.model,
            device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
            output_device=self.config.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=self.config.find_unused_parameters,
            gradient_as_bucket_view=self.config.gradient_as_bucket_view,
        )

        logger.info(f"Model wrapped with DDP on rank {self.config.rank}")
        return model

    def _wrap_fsdp(self) -> nn.Module:
        """Wrap model with FullyShardedDataParallel."""
        if not FSDP_AVAILABLE:
            raise ImportError("FSDP requires PyTorch >= 1.12")

        # Configure mixed precision
        mixed_precision_config = None
        if self.config.fp16 or self.config.bf16:
            dtype = torch.float16 if self.config.fp16 else torch.bfloat16
            mixed_precision_config = MixedPrecision(
                param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype
            )

        # Configure CPU offload
        cpu_offload_config = None
        if self.config.fsdp_cpu_offload:
            cpu_offload_config = CPUOffload(offload_params=True)

        # Configure sharding strategy
        sharding_strategy_map = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
            "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
        }
        sharding_strategy = sharding_strategy_map.get(
            self.config.fsdp_sharding_strategy, ShardingStrategy.FULL_SHARD
        )

        # Auto wrap policy
        auto_wrap_policy = None
        if self.config.fsdp_transformer_layer_cls_to_wrap:
            # This would need to be implemented based on model architecture
            pass

        # Wrap model
        model = FSDP(
            self.model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload_config,
            mixed_precision=mixed_precision_config,
            auto_wrap_policy=auto_wrap_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        )

        logger.info(f"Model wrapped with FSDP on rank {self.config.rank}")
        return model

    def _wrap_deepspeed(self) -> nn.Module:
        """Wrap model with DeepSpeed."""
        if not DEEPSPEED_AVAILABLE:
            raise ImportError("DeepSpeed is not installed")

        # Create DeepSpeed config if not provided
        if self.config.deepspeed_config is None:
            self.config.deepspeed_config = self._create_deepspeed_config()

        # Initialize DeepSpeed
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config=self.config.deepspeed_config,
            dist_init_required=not dist.is_initialized(),
        )

        self.optimizer = optimizer  # DeepSpeed manages optimizer

        logger.info(f"Model wrapped with DeepSpeed on rank {self.config.rank}")
        return model_engine

    def _wrap_accelerate(self) -> nn.Module:
        """Wrap model with Accelerate."""
        if not hasattr(self, "accelerator"):
            raise RuntimeError("Accelerator not initialized")

        # Prepare model with accelerator
        self.model = self.accelerator.prepare_model(self.model)

        logger.info(f"Model wrapped with Accelerate on rank {self.config.rank}")
        return self.model

    def _create_deepspeed_config(self) -> Dict[str, Any]:
        """Create default DeepSpeed configuration."""
        config = {
            "train_batch_size": self.training_args.per_device_train_batch_size
            * self.config.world_size,
            "train_micro_batch_size_per_gpu": self.training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "gradient_clipping": self.config.gradient_clipping,
            "fp16": {
                "enabled": self.config.fp16,
                "auto_cast": True,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "bf16": {"enabled": self.config.bf16},
            "zero_optimization": {
                "stage": self.config.zero_stage,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "overlap_comm": True,
                "contiguous_gradients": True,
            },
        }

        # Add CPU offload for ZeRO-3
        if self.config.zero_stage == 3:
            config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
            config["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": True}

        return config

    def create_optimizer_and_scheduler(self):
        """Create optimizer and learning rate scheduler."""
        if self.optimizer is not None:
            return  # Already created (e.g., by DeepSpeed)

        # Create optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.wrapped_model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.training_args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.wrapped_model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.training_args.learning_rate,
            betas=(self.training_args.adam_beta1, self.training_args.adam_beta2),
            eps=self.training_args.adam_epsilon,
        )

        # Create scheduler
        num_training_steps = self.training_args.max_steps
        if num_training_steps == -1:
            num_training_steps = (
                len(self.train_dataloader)
                // self.config.gradient_accumulation_steps
                * self.training_args.num_train_epochs
            )

        self.scheduler = get_scheduler(
            name=self.training_args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.training_args.warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Prepare with accelerator if using
        if self.config.backend == "accelerate":
            self.optimizer, self.scheduler = self.accelerator.prepare(
                self.optimizer, self.scheduler
            )

    def create_dataloaders(
        self,
        train_dataset,
        eval_dataset=None,
        train_batch_size: Optional[int] = None,
        eval_batch_size: Optional[int] = None,
    ) -> Tuple[DataLoader | Optional[DataLoader]]:
        """Create distributed dataloaders.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            train_batch_size: Override training batch size
            eval_batch_size: Override evaluation batch size

        Returns:
            Tuple of (train_dataloader, eval_dataloader)
        """
        train_batch_size = train_batch_size or self.training_args.per_device_train_batch_size
        eval_batch_size = eval_batch_size or self.training_args.per_device_eval_batch_size

        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=True,
            seed=self.training_args.seed,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=train_sampler,
            num_workers=self.training_args.dataloader_num_workers,
            pin_memory=True,
            drop_last=self.training_args.dataloader_drop_last,
        )

        eval_dataloader = None
        if eval_dataset is not None:
            eval_sampler = DistributedSampler(
                eval_dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=False,
            )

            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=eval_batch_size,
                sampler=eval_sampler,
                num_workers=self.training_args.dataloader_num_workers,
                pin_memory=True,
                drop_last=False,
            )

        # Store for later use
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Prepare with accelerator if using
        if self.config.backend == "accelerate":
            if eval_dataloader:
                self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
                    train_dataloader, eval_dataloader
                )
            else:
                self.train_dataloader = self.accelerator.prepare(train_dataloader)

        return self.train_dataloader, self.eval_dataloader

    def save_checkpoint(self, output_dir: str, epoch: int, step: int):
        """Save distributed checkpoint.

        Args:
            output_dir: Directory to save checkpoint
            epoch: Current epoch
            step: Current step
        """
        if self.config.rank != 0:
            return  # Only save on main process

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save based on backend
        if self.config.backend == "fsdp":
            self._save_fsdp_checkpoint(output_dir, epoch, step)
        elif self.config.backend == "deepspeed":
            self._save_deepspeed_checkpoint(output_dir, epoch, step)
        else:
            self._save_standard_checkpoint(output_dir, epoch, step)

    def _save_standard_checkpoint(self, output_dir: Path, epoch: int, step: int):
        """Save standard PyTorch checkpoint."""
        # Unwrap model if needed
        model_to_save = self.wrapped_model
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module

        # Save model state
        torch.save(model_to_save.state_dict(), output_dir / "pytorch_model.bin")

        # Save optimizer and scheduler
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "epoch": epoch,
                "step": step,
                "config": self.config.to_dict(),
            },
            output_dir / "training_state.pt",
        )

        # Save config
        model_to_save.config.save_pretrained(output_dir)

        # Save tokenizer if available
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Checkpoint saved to {output_dir}")

    def _save_fsdp_checkpoint(self, output_dir: Path, epoch: int, step: int):
        """Save FSDP checkpoint."""
        if not FSDP_AVAILABLE:
            return self._save_standard_checkpoint(output_dir, epoch, step)

        # Configure state dict type
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        with FSDP.state_dict_type(self.wrapped_model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = self.wrapped_model.state_dict()

            if self.config.rank == 0:
                # Save model state
                torch.save(state_dict, output_dir / "pytorch_model.bin")

                # Save optimizer state
                optim_state = FSDP.full_optim_state_dict(self.wrapped_model, self.optimizer)
                torch.save(
                    {
                        "optimizer": optim_state,
                        "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                        "epoch": epoch,
                        "step": step,
                        "config": self.config.to_dict(),
                    },
                    output_dir / "training_state.pt",
                )

                logger.info(f"FSDP checkpoint saved to {output_dir}")

    def _save_deepspeed_checkpoint(self, output_dir: Path, epoch: int, step: int):
        """Save DeepSpeed checkpoint."""
        # DeepSpeed handles its own checkpointing
        self.wrapped_model.save_checkpoint(output_dir, tag=f"epoch{epoch}_step{step}")
        logger.info(f"DeepSpeed checkpoint saved to {output_dir}")

    def load_checkpoint(self, checkpoint_dir: str):
        """Load distributed checkpoint.

        Args:
            checkpoint_dir: Directory containing checkpoint
        """
        checkpoint_dir = Path(checkpoint_dir)

        if self.config.backend == "deepspeed":
            # DeepSpeed handles its own loading
            _, client_state = self.wrapped_model.load_checkpoint(checkpoint_dir)
            return client_state

        # Load model state
        if (checkpoint_dir / "pytorch_model.bin").exists():
            state_dict = torch.load(
                checkpoint_dir / "pytorch_model.bin",
                map_location=f"cuda:{self.config.local_rank}"
                if torch.cuda.is_available()
                else "cpu",
            )

            # Handle FSDP state dict
            if self.config.backend == "fsdp" and FSDP_AVAILABLE:
                with FSDP.state_dict_type(self.wrapped_model, StateDictType.FULL_STATE_DICT):
                    self.wrapped_model.load_state_dict(state_dict)
            else:
                # Unwrap model if needed
                model = self.wrapped_model
                if hasattr(model, "module"):
                    model = model.module
                model.load_state_dict(state_dict)

        # Load training state
        training_state = {}
        if (checkpoint_dir / "training_state.pt").exists():
            training_state = torch.load(
                checkpoint_dir / "training_state.pt",
                map_location=f"cuda:{self.config.local_rank}"
                if torch.cuda.is_available()
                else "cpu",
            )

            if self.optimizer and "optimizer" in training_state:
                self.optimizer.load_state_dict(training_state["optimizer"])

            if self.scheduler and "scheduler" in training_state:
                self.scheduler.load_state_dict(training_state["scheduler"])

        logger.info(f"Checkpoint loaded from {checkpoint_dir}")
        return training_state

    def cleanup(self):
        """Cleanup distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        if self.config.backend == "accelerate" and hasattr(self, "accelerator"):
            return self.accelerator.is_main_process
        return self.config.rank == 0

    @property
    def device(self):
        """Get current device."""
        if self.config.backend == "accelerate" and hasattr(self, "accelerator"):
            return self.accelerator.device

        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.config.local_rank}")
        return torch.device("cpu")

    def print_summary(self):
        """Print distributed training summary."""
        if self.is_main_process:
            logger.info("=" * 60)
            logger.info("Distributed Training Configuration")
            logger.info("=" * 60)
            logger.info(f"Backend: {self.config.backend}")
            logger.info(f"World size: {self.config.world_size}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Mixed precision: fp16={self.config.fp16}, bf16={self.config.bf16}")
            logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")

            if self.config.backend == "fsdp":
                logger.info(f"FSDP sharding strategy: {self.config.fsdp_sharding_strategy}")
                logger.info(f"FSDP CPU offload: {self.config.fsdp_cpu_offload}")
            elif self.config.backend == "deepspeed":
                logger.info(f"DeepSpeed ZeRO stage: {self.config.zero_stage}")

            logger.info("=" * 60)


# Utility functions
def estimate_memory_usage(
    model: nn.Module, batch_size: int, sequence_length: int, precision: str = "fp16"
) -> Dict[str, float]:
    """Estimate memory usage for distributed training.

    Args:
        model: Model to estimate
        batch_size: Batch size per GPU
        sequence_length: Sequence length
        precision: Training precision (fp32, fp16, bf16)

    Returns:
        Dictionary with memory estimates in GB
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Bytes per parameter based on precision
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}.get(precision, 4)

    # Model memory
    model_memory = (total_params * bytes_per_param) / (1024**3)  # GB

    # Optimizer memory (Adam uses 2x model memory for momentum and variance)
    optimizer_memory = (trainable_params * bytes_per_param * 2) / (1024**3)

    # Gradient memory
    gradient_memory = (trainable_params * bytes_per_param) / (1024**3)

    # Activation memory (rough estimate)
    # Assumes transformer with hidden size from model config
    if hasattr(model.config, "hidden_size"):
        hidden_size = model.config.hidden_size
        num_layers = model.config.num_hidden_layers

        # Approximate activation memory per sample
        activation_per_sample = (
            sequence_length * hidden_size * num_layers * 4 * bytes_per_param
        ) / (1024**3)

        activation_memory = activation_per_sample * batch_size
    else:
        activation_memory = model_memory * 0.5  # Rough estimate

    # Total memory
    total_memory = model_memory + optimizer_memory + gradient_memory + activation_memory

    return {
        "model_memory_gb": model_memory,
        "optimizer_memory_gb": optimizer_memory,
        "gradient_memory_gb": gradient_memory,
        "activation_memory_gb": activation_memory,
        "total_memory_gb": total_memory,
        "total_params_millions": total_params / 1e6,
        "trainable_params_millions": trainable_params / 1e6,
    }


def get_distributed_backend_recommendation(model_size: int, num_gpus: int, gpu_memory: int) -> str:
    """Get recommended distributed backend based on model and hardware.

    Args:
        model_size: Number of model parameters
        num_gpus: Number of available GPUs
        gpu_memory: GPU memory in GB

    Returns:
        Recommended backend name
    """
    model_size_gb = (model_size * 2) / (1024**3)  # Assuming fp16
    memory_per_gpu_needed = model_size_gb * 4  # Model + optimizer + gradients + activations

    if num_gpus == 1:
        return "none"  # Single GPU training

    if memory_per_gpu_needed <= gpu_memory * 0.8:  # 80% to leave headroom
        return "ddp"  # Model fits on single GPU

    if memory_per_gpu_needed <= gpu_memory * num_gpus * 0.8:
        return "fsdp"  # Model fits with sharding

    # Very large model, need aggressive memory optimization
    return "deepspeed"  # ZeRO-3 with CPU offload


# Example usage
if __name__ == "__main__":
    # Example configuration
    from transformers import AutoModel, AutoTokenizer

    # Load a small model for testing
    model_name = "microsoft/deberta-v3-small"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./test_distributed",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        learning_rate=2e-5,
        warmup_steps=500,
        logging_steps=10,
        save_steps=1000,
        eval_steps=500,
        save_total_limit=2,
        fp16=True,
        dataloader_num_workers=4,
        seed=42,
    )

    # Distributed config
    distributed_config = DistributedTrainingConfig(
        backend="ddp", fp16=True, gradient_accumulation_steps=2
    )

    # Create trainer
    trainer = DistributedTrainer(
        model=model, training_args=training_args, config=distributed_config, tokenizer=tokenizer
    )

    # Print configuration
    trainer.print_summary()

    # Estimate memory usage
    memory_estimate = estimate_memory_usage(
        model=model, batch_size=8, sequence_length=512, precision="fp16"
    )

    print("\nMemory Usage Estimate:")
    for key, value in memory_estimate.items():
        print(f"  {key}: {value:.2f}")
