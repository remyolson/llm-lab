"""
Checkpoint Management System for Fine-Tuning

This module provides comprehensive checkpoint management including automatic
saving, versioning, pruning strategies, and support for both local and cloud
storage backends.

Example:
    manager = CheckpointManager(
        checkpoint_dir="./checkpoints",
        keep_best_n=3,
        keep_recent_m=2
    )
    
    # Save checkpoint
    manager.save_checkpoint(model, optimizer, metrics, epoch=5, step=1000)
    
    # Resume from best checkpoint
    checkpoint = manager.load_best_checkpoint()
"""

import os
import json
import shutil
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import torch
import numpy as np
from collections import deque
import hashlib
import pickle

# Cloud storage support
try:
    import boto3
    from botocore.exceptions import NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

try:
    from google.cloud import storage as gcs_storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    checkpoint_id: str
    epoch: int
    global_step: int
    timestamp: str
    metrics: Dict[str, float]
    model_hash: str
    training_args: Dict[str, Any]
    recipe_name: Optional[str] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create from dictionary."""
        return cls(**data)


class StorageBackend:
    """Abstract base class for storage backends."""
    
    def save(self, local_path: Path, remote_path: str):
        """Save file to storage."""
        raise NotImplementedError
    
    def load(self, remote_path: str, local_path: Path):
        """Load file from storage."""
        raise NotImplementedError
    
    def delete(self, remote_path: str):
        """Delete file from storage."""
        raise NotImplementedError
    
    def exists(self, remote_path: str) -> bool:
        """Check if file exists."""
        raise NotImplementedError
    
    def list_files(self, prefix: str) -> List[str]:
        """List files with given prefix."""
        raise NotImplementedError


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""
    
    def __init__(self, base_dir: Path):
        """Initialize local storage backend."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, local_path: Path, remote_path: str):
        """Copy file to storage location."""
        dest_path = self.base_dir / remote_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if local_path != dest_path:
            shutil.copy2(local_path, dest_path)
    
    def load(self, remote_path: str, local_path: Path):
        """Load file from storage."""
        src_path = self.base_dir / remote_path
        if not src_path.exists():
            raise FileNotFoundError(f"File not found: {src_path}")
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path != local_path:
            shutil.copy2(src_path, local_path)
    
    def delete(self, remote_path: str):
        """Delete file from storage."""
        file_path = self.base_dir / remote_path
        if file_path.exists():
            file_path.unlink()
    
    def exists(self, remote_path: str) -> bool:
        """Check if file exists."""
        return (self.base_dir / remote_path).exists()
    
    def list_files(self, prefix: str) -> List[str]:
        """List files with given prefix."""
        prefix_path = self.base_dir / prefix
        if not prefix_path.exists():
            return []
        
        files = []
        for path in prefix_path.rglob("*"):
            if path.is_file():
                relative_path = path.relative_to(self.base_dir)
                files.append(str(relative_path))
        
        return sorted(files)


class S3StorageBackend(StorageBackend):
    """AWS S3 storage backend."""
    
    def __init__(self, bucket_name: str, prefix: str = ""):
        """Initialize S3 storage backend."""
        if not S3_AVAILABLE:
            raise ImportError("boto3 is required for S3 storage backend")
        
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip('/')
        self.s3_client = boto3.client('s3')
    
    def _get_key(self, remote_path: str) -> str:
        """Get S3 key with prefix."""
        if self.prefix:
            return f"{self.prefix}/{remote_path}"
        return remote_path
    
    def save(self, local_path: Path, remote_path: str):
        """Upload file to S3."""
        key = self._get_key(remote_path)
        try:
            self.s3_client.upload_file(str(local_path), self.bucket_name, key)
            logger.info(f"Uploaded {local_path} to s3://{self.bucket_name}/{key}")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            raise
    
    def load(self, remote_path: str, local_path: Path):
        """Download file from S3."""
        key = self._get_key(remote_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.s3_client.download_file(self.bucket_name, key, str(local_path))
            logger.info(f"Downloaded s3://{self.bucket_name}/{key} to {local_path}")
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            raise
    
    def delete(self, remote_path: str):
        """Delete file from S3."""
        key = self._get_key(remote_path)
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            logger.info(f"Deleted s3://{self.bucket_name}/{key}")
        except Exception as e:
            logger.error(f"Failed to delete from S3: {e}")
            raise
    
    def exists(self, remote_path: str) -> bool:
        """Check if file exists in S3."""
        key = self._get_key(remote_path)
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except:
            return False
    
    def list_files(self, prefix: str) -> List[str]:
        """List files with given prefix in S3."""
        full_prefix = self._get_key(prefix)
        
        files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=full_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Remove the backend prefix to get relative path
                    key = obj['Key']
                    if self.prefix and key.startswith(self.prefix + '/'):
                        key = key[len(self.prefix) + 1:]
                    files.append(key)
        
        return sorted(files)


class GCSStorageBackend(StorageBackend):
    """Google Cloud Storage backend."""
    
    def __init__(self, bucket_name: str, prefix: str = ""):
        """Initialize GCS storage backend."""
        if not GCS_AVAILABLE:
            raise ImportError("google-cloud-storage is required for GCS backend")
        
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip('/')
        self.client = gcs_storage.Client()
        self.bucket = self.client.bucket(bucket_name)
    
    def _get_blob_name(self, remote_path: str) -> str:
        """Get blob name with prefix."""
        if self.prefix:
            return f"{self.prefix}/{remote_path}"
        return remote_path
    
    def save(self, local_path: Path, remote_path: str):
        """Upload file to GCS."""
        blob_name = self._get_blob_name(remote_path)
        blob = self.bucket.blob(blob_name)
        
        try:
            blob.upload_from_filename(str(local_path))
            logger.info(f"Uploaded {local_path} to gs://{self.bucket_name}/{blob_name}")
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {e}")
            raise
    
    def load(self, remote_path: str, local_path: Path):
        """Download file from GCS."""
        blob_name = self._get_blob_name(remote_path)
        blob = self.bucket.blob(blob_name)
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            blob.download_to_filename(str(local_path))
            logger.info(f"Downloaded gs://{self.bucket_name}/{blob_name} to {local_path}")
        except Exception as e:
            logger.error(f"Failed to download from GCS: {e}")
            raise
    
    def delete(self, remote_path: str):
        """Delete file from GCS."""
        blob_name = self._get_blob_name(remote_path)
        blob = self.bucket.blob(blob_name)
        
        try:
            blob.delete()
            logger.info(f"Deleted gs://{self.bucket_name}/{blob_name}")
        except Exception as e:
            logger.error(f"Failed to delete from GCS: {e}")
            raise
    
    def exists(self, remote_path: str) -> bool:
        """Check if file exists in GCS."""
        blob_name = self._get_blob_name(remote_path)
        blob = self.bucket.blob(blob_name)
        return blob.exists()
    
    def list_files(self, prefix: str) -> List[str]:
        """List files with given prefix in GCS."""
        full_prefix = self._get_blob_name(prefix)
        
        files = []
        for blob in self.bucket.list_blobs(prefix=full_prefix):
            # Remove the backend prefix to get relative path
            name = blob.name
            if self.prefix and name.startswith(self.prefix + '/'):
                name = name[len(self.prefix) + 1:]
            files.append(name)
        
        return sorted(files)


class CheckpointManager:
    """Manages model checkpoints with versioning and pruning."""
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "./checkpoints",
        keep_best_n: int = 3,
        keep_recent_m: int = 2,
        metric_name: str = "eval_loss",
        metric_mode: str = "min",
        storage_backend: Optional[str] = None,
        storage_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize CheckpointManager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            keep_best_n: Number of best checkpoints to keep
            keep_recent_m: Number of recent checkpoints to keep
            metric_name: Metric name for best checkpoint selection
            metric_mode: 'min' or 'max' for metric comparison
            storage_backend: Storage backend type ('local', 's3', 'gcs')
            storage_config: Configuration for storage backend
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_best_n = keep_best_n
        self.keep_recent_m = keep_recent_m
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        
        # Initialize storage backend
        self.storage = self._init_storage_backend(storage_backend, storage_config)
        
        # Track checkpoints
        self.checkpoint_history = deque(maxlen=100)
        self.best_checkpoints = []
        self.recent_checkpoints = deque(maxlen=keep_recent_m)
        
        # Load existing checkpoint metadata
        self._load_checkpoint_registry()
    
    def _init_storage_backend(
        self,
        backend_type: Optional[str],
        config: Optional[Dict[str, Any]]
    ) -> StorageBackend:
        """Initialize storage backend."""
        if backend_type == "s3":
            if not config or "bucket_name" not in config:
                raise ValueError("S3 backend requires bucket_name in config")
            return S3StorageBackend(
                bucket_name=config["bucket_name"],
                prefix=config.get("prefix", "")
            )
        elif backend_type == "gcs":
            if not config or "bucket_name" not in config:
                raise ValueError("GCS backend requires bucket_name in config")
            return GCSStorageBackend(
                bucket_name=config["bucket_name"],
                prefix=config.get("prefix", "")
            )
        else:
            # Default to local storage
            return LocalStorageBackend(self.checkpoint_dir)
    
    def _load_checkpoint_registry(self):
        """Load checkpoint registry from disk."""
        registry_path = self.checkpoint_dir / "checkpoint_registry.json"
        
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
                
                # Restore checkpoint history
                for checkpoint_data in registry.get("history", []):
                    metadata = CheckpointMetadata.from_dict(checkpoint_data)
                    self.checkpoint_history.append(metadata)
                
                # Restore best checkpoints
                self.best_checkpoints = [
                    CheckpointMetadata.from_dict(data)
                    for data in registry.get("best", [])
                ]
                
                # Restore recent checkpoints
                for checkpoint_data in registry.get("recent", []):
                    metadata = CheckpointMetadata.from_dict(checkpoint_data)
                    self.recent_checkpoints.append(metadata)
                
                logger.info(f"Loaded {len(self.checkpoint_history)} checkpoints from registry")
            
            except Exception as e:
                logger.error(f"Failed to load checkpoint registry: {e}")
    
    def _save_checkpoint_registry(self):
        """Save checkpoint registry to disk."""
        registry = {
            "history": [cp.to_dict() for cp in self.checkpoint_history],
            "best": [cp.to_dict() for cp in self.best_checkpoints],
            "recent": [cp.to_dict() for cp in self.recent_checkpoints]
        }
        
        registry_path = self.checkpoint_dir / "checkpoint_registry.json"
        
        try:
            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint registry: {e}")
    
    def _compute_model_hash(self, model_state_dict: Dict[str, Any]) -> str:
        """Compute hash of model state for versioning."""
        # Use a subset of weights to compute hash (for efficiency)
        sample_keys = sorted(model_state_dict.keys())[:10]
        
        hasher = hashlib.md5()
        for key in sample_keys:
            if key in model_state_dict:
                tensor = model_state_dict[key]
                if torch.is_tensor(tensor):
                    hasher.update(tensor.cpu().numpy().tobytes())
        
        return hasher.hexdigest()[:8]
    
    def save_checkpoint(
        self,
        model,
        optimizer,
        metrics: Dict[str, float],
        epoch: int,
        global_step: int,
        scheduler=None,
        training_args: Optional[Dict[str, Any]] = None,
        recipe_name: Optional[str] = None,
        notes: Optional[str] = None,
        additional_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            metrics: Current metrics
            epoch: Current epoch
            global_step: Current global step
            scheduler: Optional learning rate scheduler
            training_args: Training arguments
            recipe_name: Recipe name
            notes: Optional notes
            additional_state: Additional state to save
            
        Returns:
            Checkpoint ID
        """
        # Generate checkpoint ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_id = f"checkpoint_epoch{epoch}_step{global_step}_{timestamp}"
        
        # Create checkpoint directory
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Get model state dict
        if hasattr(model, 'module'):
            # Handle DataParallel/DistributedDataParallel
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        
        # Compute model hash
        model_hash = self._compute_model_hash(model_state_dict)
        
        # Save model state
        model_path = checkpoint_path / "model.pt"
        torch.save(model_state_dict, model_path)
        
        # Save optimizer state
        optimizer_path = checkpoint_path / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_path)
        
        # Save scheduler state if provided
        if scheduler is not None:
            scheduler_path = checkpoint_path / "scheduler.pt"
            torch.save(scheduler.state_dict(), scheduler_path)
        
        # Save additional state if provided
        if additional_state:
            additional_path = checkpoint_path / "additional_state.pt"
            torch.save(additional_state, additional_path)
        
        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            epoch=epoch,
            global_step=global_step,
            timestamp=timestamp,
            metrics=metrics,
            model_hash=model_hash,
            training_args=training_args or {},
            recipe_name=recipe_name,
            notes=notes
        )
        
        # Save metadata
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Upload to cloud storage if configured
        if not isinstance(self.storage, LocalStorageBackend):
            for file_path in checkpoint_path.glob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.checkpoint_dir)
                    self.storage.save(file_path, str(relative_path))
        
        # Update checkpoint tracking
        self.checkpoint_history.append(metadata)
        self.recent_checkpoints.append(metadata)
        
        # Update best checkpoints if applicable
        if self.metric_name in metrics:
            self._update_best_checkpoints(metadata)
        
        # Apply pruning strategies
        self._prune_checkpoints()
        
        # Save registry
        self._save_checkpoint_registry()
        
        logger.info(f"Saved checkpoint: {checkpoint_id}")
        logger.info(f"Metrics: {metrics}")
        
        return checkpoint_id
    
    def _update_best_checkpoints(self, metadata: CheckpointMetadata):
        """Update list of best checkpoints."""
        metric_value = metadata.metrics.get(self.metric_name)
        if metric_value is None:
            return
        
        # Add to best checkpoints
        self.best_checkpoints.append(metadata)
        
        # Sort by metric
        reverse = (self.metric_mode == "max")
        self.best_checkpoints.sort(
            key=lambda x: x.metrics.get(self.metric_name, float('inf')),
            reverse=reverse
        )
        
        # Keep only top N
        if len(self.best_checkpoints) > self.keep_best_n:
            # Remove checkpoints that are no longer in top N
            removed = self.best_checkpoints[self.keep_best_n:]
            self.best_checkpoints = self.best_checkpoints[:self.keep_best_n]
            
            # Delete removed checkpoints if they're not in recent
            for checkpoint in removed:
                if checkpoint not in self.recent_checkpoints:
                    self._delete_checkpoint(checkpoint.checkpoint_id)
    
    def _prune_checkpoints(self):
        """Apply pruning strategies to limit checkpoint storage."""
        # Get all checkpoints that should be kept
        keep_ids = set()
        
        # Keep best checkpoints
        for checkpoint in self.best_checkpoints:
            keep_ids.add(checkpoint.checkpoint_id)
        
        # Keep recent checkpoints
        for checkpoint in self.recent_checkpoints:
            keep_ids.add(checkpoint.checkpoint_id)
        
        # Delete checkpoints not in keep list
        all_checkpoint_dirs = list(self.checkpoint_dir.glob("checkpoint_*"))
        
        for checkpoint_dir in all_checkpoint_dirs:
            checkpoint_id = checkpoint_dir.name
            if checkpoint_id not in keep_ids and checkpoint_dir.is_dir():
                self._delete_checkpoint(checkpoint_id)
    
    def _delete_checkpoint(self, checkpoint_id: str):
        """Delete a checkpoint."""
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        
        logger.info(f"Deleting checkpoint: {checkpoint_id}")
        
        # Delete from cloud storage if configured
        if not isinstance(self.storage, LocalStorageBackend):
            # List and delete all files for this checkpoint
            files = self.storage.list_files(checkpoint_id)
            for file_path in files:
                self.storage.delete(file_path)
        
        # Delete local files
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_id: str,
        model=None,
        optimizer=None,
        scheduler=None,
        map_location=None
    ) -> Dict[str, Any]:
        """Load a specific checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID to load
            model: Model to load state into (optional)
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            map_location: Device mapping for loading
            
        Returns:
            Dictionary with checkpoint data
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        
        # Download from cloud storage if needed
        if not isinstance(self.storage, LocalStorageBackend):
            # Ensure local directory exists
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            # Download all files for this checkpoint
            files = self.storage.list_files(checkpoint_id)
            for file_path in files:
                local_file = self.checkpoint_dir / file_path
                if not local_file.exists():
                    self.storage.load(file_path, local_file)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")
        
        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = CheckpointMetadata.from_dict(json.load(f))
        
        # Load model state
        model_path = checkpoint_path / "model.pt"
        model_state_dict = torch.load(model_path, map_location=map_location)
        
        if model is not None:
            if hasattr(model, 'module'):
                model.module.load_state_dict(model_state_dict)
            else:
                model.load_state_dict(model_state_dict)
        
        # Load optimizer state
        optimizer_state_dict = None
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=map_location)
            if optimizer is not None:
                optimizer.load_state_dict(optimizer_state_dict)
        
        # Load scheduler state
        scheduler_state_dict = None
        scheduler_path = checkpoint_path / "scheduler.pt"
        if scheduler_path.exists():
            scheduler_state_dict = torch.load(scheduler_path, map_location=map_location)
            if scheduler is not None:
                scheduler.load_state_dict(scheduler_state_dict)
        
        # Load additional state
        additional_state = None
        additional_path = checkpoint_path / "additional_state.pt"
        if additional_path.exists():
            additional_state = torch.load(additional_path, map_location=map_location)
        
        logger.info(f"Loaded checkpoint: {checkpoint_id}")
        logger.info(f"Epoch: {metadata.epoch}, Step: {metadata.global_step}")
        logger.info(f"Metrics: {metadata.metrics}")
        
        return {
            "metadata": metadata,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "scheduler_state_dict": scheduler_state_dict,
            "additional_state": additional_state,
            "epoch": metadata.epoch,
            "global_step": metadata.global_step
        }
    
    def load_best_checkpoint(self, **kwargs) -> Dict[str, Any]:
        """Load the best checkpoint based on metric.
        
        Args:
            **kwargs: Arguments passed to load_checkpoint
            
        Returns:
            Dictionary with checkpoint data
        """
        if not self.best_checkpoints:
            raise ValueError("No best checkpoints available")
        
        best_checkpoint = self.best_checkpoints[0]
        logger.info(f"Loading best checkpoint: {best_checkpoint.checkpoint_id}")
        logger.info(f"{self.metric_name}: {best_checkpoint.metrics.get(self.metric_name)}")
        
        return self.load_checkpoint(best_checkpoint.checkpoint_id, **kwargs)
    
    def load_latest_checkpoint(self, **kwargs) -> Dict[str, Any]:
        """Load the most recent checkpoint.
        
        Args:
            **kwargs: Arguments passed to load_checkpoint
            
        Returns:
            Dictionary with checkpoint data
        """
        if not self.recent_checkpoints:
            raise ValueError("No recent checkpoints available")
        
        latest_checkpoint = self.recent_checkpoints[-1]
        logger.info(f"Loading latest checkpoint: {latest_checkpoint.checkpoint_id}")
        
        return self.load_checkpoint(latest_checkpoint.checkpoint_id, **kwargs)
    
    def get_checkpoint_info(self, checkpoint_id: str) -> CheckpointMetadata:
        """Get metadata for a specific checkpoint."""
        # Check in history
        for checkpoint in self.checkpoint_history:
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        
        # Load from disk if not in memory
        metadata_path = self.checkpoint_dir / checkpoint_id / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return CheckpointMetadata.from_dict(json.load(f))
        
        raise ValueError(f"Checkpoint not found: {checkpoint_id}")
    
    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """List all available checkpoints."""
        checkpoints = []
        
        # List from registry
        for checkpoint in self.checkpoint_history:
            checkpoints.append(checkpoint)
        
        # Also check disk for any not in registry
        for checkpoint_dir in self.checkpoint_dir.glob("checkpoint_*"):
            if checkpoint_dir.is_dir():
                checkpoint_id = checkpoint_dir.name
                
                # Skip if already in list
                if any(cp.checkpoint_id == checkpoint_id for cp in checkpoints):
                    continue
                
                # Try to load metadata
                try:
                    metadata = self.get_checkpoint_info(checkpoint_id)
                    checkpoints.append(metadata)
                except:
                    logger.warning(f"Failed to load metadata for {checkpoint_id}")
        
        # Sort by timestamp
        checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
        
        return checkpoints
    
    def compare_checkpoints(
        self,
        checkpoint_id1: str,
        checkpoint_id2: str
    ) -> Dict[str, Any]:
        """Compare two checkpoints.
        
        Args:
            checkpoint_id1: First checkpoint ID
            checkpoint_id2: Second checkpoint ID
            
        Returns:
            Comparison results
        """
        # Load metadata
        metadata1 = self.get_checkpoint_info(checkpoint_id1)
        metadata2 = self.get_checkpoint_info(checkpoint_id2)
        
        # Compare metrics
        metric_comparison = {}
        all_metrics = set(metadata1.metrics.keys()) | set(metadata2.metrics.keys())
        
        for metric in all_metrics:
            val1 = metadata1.metrics.get(metric)
            val2 = metadata2.metrics.get(metric)
            
            if val1 is not None and val2 is not None:
                diff = val2 - val1
                pct_change = (diff / val1 * 100) if val1 != 0 else 0
                metric_comparison[metric] = {
                    "checkpoint1": val1,
                    "checkpoint2": val2,
                    "difference": diff,
                    "pct_change": pct_change
                }
        
        # Compare training progress
        comparison = {
            "checkpoint1": {
                "id": checkpoint_id1,
                "epoch": metadata1.epoch,
                "step": metadata1.global_step,
                "timestamp": metadata1.timestamp
            },
            "checkpoint2": {
                "id": checkpoint_id2,
                "epoch": metadata2.epoch,
                "step": metadata2.global_step,
                "timestamp": metadata2.timestamp
            },
            "metric_comparison": metric_comparison,
            "model_changed": metadata1.model_hash != metadata2.model_hash
        }
        
        return comparison
    
    def export_checkpoint(
        self,
        checkpoint_id: str,
        export_path: Union[str, Path],
        include_optimizer: bool = True,
        include_training_state: bool = True
    ):
        """Export checkpoint to a portable format.
        
        Args:
            checkpoint_id: Checkpoint to export
            export_path: Path to export to
            include_optimizer: Include optimizer state
            include_training_state: Include training state
        """
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load checkpoint
        checkpoint_data = self.load_checkpoint(checkpoint_id)
        
        # Prepare export data
        export_data = {
            "model_state_dict": checkpoint_data["model_state_dict"],
            "metadata": checkpoint_data["metadata"].to_dict()
        }
        
        if include_optimizer and checkpoint_data["optimizer_state_dict"]:
            export_data["optimizer_state_dict"] = checkpoint_data["optimizer_state_dict"]
        
        if include_training_state:
            if checkpoint_data["scheduler_state_dict"]:
                export_data["scheduler_state_dict"] = checkpoint_data["scheduler_state_dict"]
            if checkpoint_data["additional_state"]:
                export_data["additional_state"] = checkpoint_data["additional_state"]
        
        # Save as single file
        torch.save(export_data, export_path)
        
        logger.info(f"Exported checkpoint {checkpoint_id} to {export_path}")


# Example usage
if __name__ == "__main__":
    # Create checkpoint manager
    manager = CheckpointManager(
        checkpoint_dir="./test_checkpoints",
        keep_best_n=3,
        keep_recent_m=2,
        metric_name="eval_loss",
        metric_mode="min"
    )
    
    # Example: Save checkpoint
    import torch.nn as nn
    import torch.optim as optim
    
    # Dummy model and optimizer
    model = nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters())
    
    # Save checkpoint
    checkpoint_id = manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        metrics={"loss": 0.5, "eval_loss": 0.6, "accuracy": 0.85},
        epoch=5,
        global_step=1000,
        recipe_name="test_recipe",
        notes="Test checkpoint"
    )
    
    print(f"Saved checkpoint: {checkpoint_id}")
    
    # List checkpoints
    checkpoints = manager.list_checkpoints()
    print(f"\nAvailable checkpoints: {len(checkpoints)}")
    for cp in checkpoints:
        print(f"  - {cp.checkpoint_id}: {cp.metrics}")
    
    # Load best checkpoint
    # best_data = manager.load_best_checkpoint(model=model, optimizer=optimizer)