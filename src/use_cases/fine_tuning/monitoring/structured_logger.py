"""
Structured Logging for Fine-Tuning Pipeline

This module provides structured logging capabilities specifically designed
for the fine-tuning pipeline, with support for JSON logs, metric tracking,
and integration with monitoring platforms.

Example:
    logger = StructuredLogger("fine_tuning_experiment")
    
    logger.log_training_step(
        step=100,
        loss=0.5,
        metrics={"accuracy": 0.95},
        duration=1.23
    )
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
from contextlib import contextmanager
import traceback


class LogLevel(Enum):
    """Log levels for structured logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventType(Enum):
    """Event types for structured logging."""
    TRAINING_START = "training_start"
    TRAINING_END = "training_end"
    TRAINING_STEP = "training_step"
    EVALUATION_START = "evaluation_start"
    EVALUATION_END = "evaluation_end"
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_LOADED = "checkpoint_loaded"
    DATASET_LOADED = "dataset_loaded"
    MODEL_LOADED = "model_loaded"
    ERROR = "error"
    WARNING = "warning"
    METRIC = "metric"
    HYPERPARAMETER = "hyperparameter"
    RESOURCE = "resource"
    CUSTOM = "custom"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    event_type: str
    message: str
    context: Dict[str, Any]
    duration_ms: Optional[float] = None
    error_info: Optional[Dict[str, Any]] = None
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self))


class StructuredLogger:
    """Structured logger for fine-tuning pipeline."""
    
    def __init__(
        self,
        name: str,
        log_dir: str = "./logs",
        log_file: Optional[str] = None,
        console_output: bool = True,
        json_output: bool = True,
        buffer_size: int = 100
    ):
        """Initialize structured logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            log_file: Specific log file name
            console_output: Enable console output
            json_output: Use JSON format for logs
            buffer_size: Size of log buffer
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup log file
        if log_file:
            self.log_file = self.log_dir / log_file
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = self.log_dir / f"{name}_{timestamp}.log"
        
        self.console_output = console_output
        self.json_output = json_output
        
        # Setup standard logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        
        if json_output:
            # JSON formatter
            file_handler.setFormatter(logging.Formatter('%(message)s'))
        else:
            # Standard formatter
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
        
        self.logger.addHandler(file_handler)
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(console_handler)
        
        # Log buffer for async writing
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.buffer_thread = threading.Thread(target=self._buffer_writer, daemon=True)
        self.buffer_thread.start()
        
        # Timing context
        self.timers = {}
    
    def _buffer_writer(self):
        """Background thread for writing buffered logs."""
        while True:
            try:
                entry = self.buffer.get(timeout=1)
                if entry is None:  # Shutdown signal
                    break
                
                if self.json_output:
                    self.logger.info(entry.to_json())
                else:
                    self.logger.log(
                        getattr(logging, entry.level),
                        f"[{entry.event_type}] {entry.message}"
                    )
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in log buffer writer: {e}")
    
    def _log(
        self,
        level: LogLevel,
        event_type: EventType,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        error: Optional[Exception] = None
    ):
        """Internal logging method."""
        # Create log entry
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.value,
            event_type=event_type.value,
            message=message,
            context=context or {},
            duration_ms=duration_ms
        )
        
        # Add error info if present
        if error:
            entry.error_info = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc()
            }
        
        # Add to buffer
        try:
            self.buffer.put_nowait(entry)
        except queue.Full:
            # If buffer is full, log directly
            if self.json_output:
                self.logger.info(entry.to_json())
            else:
                self.logger.log(
                    getattr(logging, entry.level),
                    f"[{entry.event_type}] {entry.message}"
                )
    
    @contextmanager
    def timer(self, name: str):
        """Context manager for timing operations.
        
        Args:
            name: Timer name
            
        Yields:
            Timer name
        """
        start_time = time.time()
        self.timers[name] = start_time
        
        try:
            yield name
        finally:
            duration = (time.time() - start_time) * 1000  # Convert to ms
            self.timers.pop(name, None)
            
            # Log timing
            self._log(
                LogLevel.DEBUG,
                EventType.CUSTOM,
                f"Timer '{name}' completed",
                {"timer_name": name},
                duration_ms=duration
            )
    
    def log_training_start(
        self,
        model_name: str,
        dataset_name: str,
        config: Dict[str, Any]
    ):
        """Log training start event."""
        self._log(
            LogLevel.INFO,
            EventType.TRAINING_START,
            f"Starting training: {model_name} on {dataset_name}",
            {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "config": config
            }
        )
    
    def log_training_step(
        self,
        step: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
        learning_rate: Optional[float] = None,
        duration: Optional[float] = None
    ):
        """Log training step."""
        context = {
            "step": step,
            "loss": loss
        }
        
        if metrics:
            context["metrics"] = metrics
        if learning_rate is not None:
            context["learning_rate"] = learning_rate
        
        self._log(
            LogLevel.DEBUG,
            EventType.TRAINING_STEP,
            f"Step {step}: loss={loss:.4f}",
            context,
            duration_ms=duration * 1000 if duration else None
        )
    
    def log_evaluation(
        self,
        step: int,
        metrics: Dict[str, float],
        dataset: str = "validation"
    ):
        """Log evaluation results."""
        self._log(
            LogLevel.INFO,
            EventType.METRIC,
            f"Evaluation at step {step}",
            {
                "step": step,
                "dataset": dataset,
                "metrics": metrics
            }
        )
    
    def log_checkpoint_saved(
        self,
        checkpoint_path: str,
        step: int,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Log checkpoint save event."""
        context = {
            "checkpoint_path": checkpoint_path,
            "step": step
        }
        
        if metrics:
            context["metrics"] = metrics
        
        self._log(
            LogLevel.INFO,
            EventType.CHECKPOINT_SAVED,
            f"Checkpoint saved at step {step}",
            context
        )
    
    def log_checkpoint_loaded(
        self,
        checkpoint_path: str,
        step: Optional[int] = None
    ):
        """Log checkpoint load event."""
        self._log(
            LogLevel.INFO,
            EventType.CHECKPOINT_LOADED,
            f"Checkpoint loaded from {checkpoint_path}",
            {
                "checkpoint_path": checkpoint_path,
                "step": step
            }
        )
    
    def log_dataset_loaded(
        self,
        dataset_name: str,
        num_samples: int,
        splits: Optional[Dict[str, int]] = None
    ):
        """Log dataset loading event."""
        context = {
            "dataset_name": dataset_name,
            "num_samples": num_samples
        }
        
        if splits:
            context["splits"] = splits
        
        self._log(
            LogLevel.INFO,
            EventType.DATASET_LOADED,
            f"Dataset loaded: {dataset_name} ({num_samples} samples)",
            context
        )
    
    def log_model_loaded(
        self,
        model_name: str,
        num_parameters: Optional[int] = None,
        model_info: Optional[Dict[str, Any]] = None
    ):
        """Log model loading event."""
        context = {
            "model_name": model_name
        }
        
        if num_parameters:
            context["num_parameters"] = num_parameters
        if model_info:
            context["model_info"] = model_info
        
        self._log(
            LogLevel.INFO,
            EventType.MODEL_LOADED,
            f"Model loaded: {model_name}",
            context
        )
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters."""
        self._log(
            LogLevel.INFO,
            EventType.HYPERPARAMETER,
            "Hyperparameters configured",
            {"hyperparameters": hyperparams}
        )
    
    def log_resource_usage(
        self,
        cpu_percent: float,
        memory_percent: float,
        gpu_info: Optional[List[Dict[str, Any]]] = None
    ):
        """Log resource usage."""
        context = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent
        }
        
        if gpu_info:
            context["gpu_info"] = gpu_info
        
        self._log(
            LogLevel.DEBUG,
            EventType.RESOURCE,
            "Resource usage",
            context
        )
    
    def log_error(
        self,
        message: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log an error."""
        self._log(
            LogLevel.ERROR,
            EventType.ERROR,
            message,
            context,
            error=error
        )
    
    def log_warning(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log a warning."""
        self._log(
            LogLevel.WARNING,
            EventType.WARNING,
            message,
            context
        )
    
    def log_custom(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        level: LogLevel = LogLevel.INFO
    ):
        """Log a custom event."""
        self._log(
            level,
            EventType.CUSTOM,
            message,
            context
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get logging summary statistics."""
        # Read log file and analyze
        if not self.log_file.exists():
            return {}
        
        summary = {
            "total_entries": 0,
            "by_level": {},
            "by_event_type": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        if self.json_output:
                            entry = json.loads(line)
                            summary["total_entries"] += 1
                            
                            # Count by level
                            level = entry.get("level", "UNKNOWN")
                            summary["by_level"][level] = summary["by_level"].get(level, 0) + 1
                            
                            # Count by event type
                            event_type = entry.get("event_type", "UNKNOWN")
                            summary["by_event_type"][event_type] = \
                                summary["by_event_type"].get(event_type, 0) + 1
                            
                            # Collect errors and warnings
                            if level == "ERROR":
                                summary["errors"].append({
                                    "timestamp": entry.get("timestamp"),
                                    "message": entry.get("message"),
                                    "error_info": entry.get("error_info")
                                })
                            elif level == "WARNING":
                                summary["warnings"].append({
                                    "timestamp": entry.get("timestamp"),
                                    "message": entry.get("message")
                                })
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            summary["error"] = str(e)
        
        return summary
    
    def close(self):
        """Close the logger and flush buffers."""
        # Signal buffer thread to stop
        self.buffer.put(None)
        self.buffer_thread.join(timeout=5)
        
        # Log closing
        self._log(
            LogLevel.INFO,
            EventType.CUSTOM,
            "Logger closed",
            {"name": self.name}
        )


# Example usage
if __name__ == "__main__":
    # Create logger
    logger = StructuredLogger(
        "test_training",
        log_dir="./test_logs",
        console_output=True,
        json_output=True
    )
    
    # Log training start
    logger.log_training_start(
        model_name="bert-base-uncased",
        dataset_name="squad",
        config={
            "batch_size": 32,
            "learning_rate": 2e-5,
            "num_epochs": 3
        }
    )
    
    # Log hyperparameters
    logger.log_hyperparameters({
        "optimizer": "AdamW",
        "scheduler": "linear",
        "warmup_steps": 500,
        "max_grad_norm": 1.0
    })
    
    # Simulate training steps
    import random
    
    for step in range(10):
        with logger.timer(f"step_{step}"):
            # Simulate training
            time.sleep(0.1)
            
            # Log step
            logger.log_training_step(
                step=step,
                loss=2.5 - step * 0.2 + random.random() * 0.1,
                metrics={
                    "accuracy": 0.7 + step * 0.02,
                    "f1": 0.65 + step * 0.025
                },
                learning_rate=2e-5 * (0.9 ** step)
            )
            
            # Log evaluation every 5 steps
            if step % 5 == 0:
                logger.log_evaluation(
                    step=step,
                    metrics={
                        "eval_loss": 2.3 - step * 0.15,
                        "eval_accuracy": 0.75 + step * 0.015
                    }
                )
                
                # Save checkpoint
                logger.log_checkpoint_saved(
                    f"./checkpoints/step_{step}.pt",
                    step=step,
                    metrics={"loss": 2.5 - step * 0.2}
                )
    
    # Simulate error
    try:
        raise ValueError("Simulated training error")
    except Exception as e:
        logger.log_error("Training failed", e, {"step": 10})
    
    # Log warning
    logger.log_warning("GPU memory usage high", {"usage_percent": 95})
    
    # Get summary
    print("\nLogging Summary:")
    print(json.dumps(logger.get_summary(), indent=2))
    
    # Close logger
    logger.close()