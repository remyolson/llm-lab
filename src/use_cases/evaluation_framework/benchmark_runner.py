"""
AutoBenchmarkRunner Core Infrastructure

This module provides the main orchestration class for automated before/after
evaluations, including model version detection, automatic benchmark scheduling,
state management, and monitoring integration.

Example:
    runner = AutoBenchmarkRunner()
    
    # Run before/after evaluation
    results = runner.evaluate_fine_tuning(
        base_model="gpt2",
        fine_tuned_model="./fine_tuned/gpt2-custom",
        benchmarks=["hellaswag", "mmlu", "custom_eval"]
    )
    
    # Generate comparison report
    report = runner.generate_comparison_report(results)
"""

import os
import json
import time
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import yaml
from collections import defaultdict
import numpy as np

# Import evaluation components
from ..fine_tuning.evaluation.suite import EvaluationSuite, EvaluationConfig, EvaluationResult
from ..fine_tuning.evaluation.benchmarks import BenchmarkRunner
from ..fine_tuning.monitoring.integrations import MonitoringIntegration
from ..fine_tuning.monitoring.structured_logger import StructuredLogger

# Import model loading
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        PreTrainedModel,
        PreTrainedTokenizer
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EvaluationState(Enum):
    """Evaluation job states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelVersionType(Enum):
    """Model version types."""
    BASE = "base"
    FINE_TUNED = "fine_tuned"
    CHECKPOINT = "checkpoint"
    VARIANT = "variant"


@dataclass
class ModelVersion:
    """Model version information."""
    version_id: str
    model_path: str
    version_type: ModelVersionType
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Convert string types if needed."""
        if isinstance(self.version_type, str):
            self.version_type = ModelVersionType(self.version_type)
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "model_path": self.model_path,
            "version_type": self.version_type.value,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "metrics": self.metrics
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a model."""
    model_version: ModelVersion
    evaluation_results: EvaluationResult
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    system_info: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_version": self.model_version.to_dict(),
            "evaluation_results": self.evaluation_results.to_dict(),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "system_info": self.system_info,
            "errors": self.errors
        }


@dataclass
class ComparisonResult:
    """Comparison result between two models."""
    base_result: BenchmarkResult
    fine_tuned_result: BenchmarkResult
    improvements: Dict[str, Any] = field(default_factory=dict)
    regressions: List[Dict[str, Any]] = field(default_factory=list)
    statistical_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_result": self.base_result.to_dict(),
            "fine_tuned_result": self.fine_tuned_result.to_dict(),
            "improvements": self.improvements,
            "regressions": self.regressions,
            "statistical_analysis": self.statistical_analysis
        }


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    benchmarks: List[str] = field(default_factory=lambda: ["hellaswag", "mmlu"])
    custom_evaluations: List[str] = field(default_factory=list)
    batch_size: int = 8
    max_samples: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    cache_results: bool = True
    cache_dir: str = "./benchmark_cache"
    monitoring_enabled: bool = True
    monitoring_platforms: List[str] = field(default_factory=lambda: ["tensorboard"])
    parallel_benchmarks: bool = True
    max_parallel_jobs: int = 2
    timeout_minutes: int = 60
    save_intermediate: bool = True
    
    # Hooks
    pre_evaluation_hook: Optional[Callable] = None
    post_evaluation_hook: Optional[Callable] = None
    on_error_hook: Optional[Callable] = None


class AutoBenchmarkRunner:
    """Automated benchmark runner for before/after evaluations."""
    
    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        state_dir: str = "./evaluation_state"
    ):
        """Initialize benchmark runner.
        
        Args:
            config: Benchmark configuration
            state_dir: Directory for storing evaluation state
        """
        self.config = config or BenchmarkConfig()
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.evaluation_suite = None
        self.benchmark_runner = BenchmarkRunner()
        self.monitoring = None
        self.logger = StructuredLogger(
            "benchmark_runner",
            log_dir=str(self.state_dir / "logs")
        )
        
        # State tracking
        self.running_evaluations = {}
        self.evaluation_history = []
        
        # Initialize monitoring if enabled
        if self.config.monitoring_enabled:
            self._init_monitoring()
        
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_parallel_jobs
        )
    
    def _init_monitoring(self):
        """Initialize monitoring integration."""
        try:
            self.monitoring = MonitoringIntegration(
                platforms=self.config.monitoring_platforms,
                project_name="benchmark_evaluation",
                config={
                    "log_file": str(self.state_dir / "monitoring.log")
                }
            )
            
            # Add alerts
            self.monitoring.add_alert(
                metric="evaluation_duration",
                threshold=self.config.timeout_minutes * 60,
                condition="greater_than",
                message="Evaluation taking too long"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize monitoring: {e}")
            self.monitoring = None
    
    def detect_model_version(self, model_path: str) -> ModelVersion:
        """Detect model version information.
        
        Args:
            model_path: Path to model
            
        Returns:
            ModelVersion object
        """
        model_path = Path(model_path)
        
        # Generate version ID
        if model_path.is_dir():
            # Hash directory contents
            hasher = hashlib.sha256()
            for file in sorted(model_path.rglob("*.bin")) + sorted(model_path.rglob("*.safetensors")):
                hasher.update(str(file).encode())
                hasher.update(str(file.stat().st_mtime).encode())
            version_id = hasher.hexdigest()[:12]
        else:
            # Hash file
            hasher = hashlib.sha256()
            with open(model_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            version_id = hasher.hexdigest()[:12]
        
        # Detect version type
        version_type = ModelVersionType.BASE
        if "fine_tuned" in str(model_path) or "ft_" in str(model_path):
            version_type = ModelVersionType.FINE_TUNED
        elif "checkpoint" in str(model_path) or "ckpt" in str(model_path):
            version_type = ModelVersionType.CHECKPOINT
        
        # Get metadata
        metadata = {}
        config_path = model_path / "config.json" if model_path.is_dir() else None
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                metadata["model_type"] = config.get("model_type", "unknown")
                metadata["num_parameters"] = config.get("num_parameters")
        
        # Get creation time
        created_at = datetime.fromtimestamp(
            model_path.stat().st_mtime
        )
        
        return ModelVersion(
            version_id=version_id,
            model_path=str(model_path),
            version_type=version_type,
            created_at=created_at,
            metadata=metadata
        )
    
    def evaluate_model(
        self,
        model_path: str,
        model_version: Optional[ModelVersion] = None,
        benchmarks: Optional[List[str]] = None,
        custom_evaluations: Optional[List[Callable]] = None
    ) -> BenchmarkResult:
        """Evaluate a single model.
        
        Args:
            model_path: Path to model
            model_version: Optional model version info
            benchmarks: List of benchmarks to run
            custom_evaluations: Custom evaluation functions
            
        Returns:
            BenchmarkResult
        """
        start_time = datetime.now()
        
        # Detect model version if not provided
        if model_version is None:
            model_version = self.detect_model_version(model_path)
        
        # Log evaluation start
        self.logger.log_training_start(
            model_name=model_version.model_path,
            dataset_name="benchmark_suite",
            config={"benchmarks": benchmarks or self.config.benchmarks}
        )
        
        # Check cache
        cache_key = f"{model_version.version_id}_{'-'.join(benchmarks or self.config.benchmarks)}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result and self.config.cache_results:
            logger.info(f"Using cached result for {model_version.version_id}")
            return cached_result
        
        errors = []
        
        try:
            # Load model and tokenizer
            logger.info(f"Loading model from {model_path}")
            model, tokenizer = self._load_model_and_tokenizer(model_path)
            
            # Initialize evaluation suite
            eval_config = EvaluationConfig(
                benchmarks=benchmarks or self.config.benchmarks,
                batch_size=self.config.batch_size,
                max_samples=self.config.max_samples,
                device=self.config.device,
                save_results=self.config.save_intermediate,
                results_dir=str(self.state_dir / "results")
            )
            
            evaluation_suite = EvaluationSuite(eval_config)
            
            # Run pre-evaluation hook
            if self.config.pre_evaluation_hook:
                self.config.pre_evaluation_hook(model_version, model, tokenizer)
            
            # Run evaluation
            evaluation_results = evaluation_suite.evaluate(
                model=model,
                tokenizer=tokenizer,
                custom_eval_fn=self._create_custom_eval_fn(custom_evaluations)
            )
            
            # Update model version metrics
            model_version.metrics = {
                "perplexity": evaluation_results.perplexity,
                "benchmarks": {
                    b.name: b.overall_score
                    for b in evaluation_results.benchmarks
                }
            }
            
            # Run post-evaluation hook
            if self.config.post_evaluation_hook:
                self.config.post_evaluation_hook(model_version, evaluation_results)
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            errors.append(str(e))
            
            # Run error hook
            if self.config.on_error_hook:
                self.config.on_error_hook(model_version, e)
            
            # Create empty evaluation result
            evaluation_results = EvaluationResult(
                model_name=model_version.model_path,
                benchmarks=[],
                custom_metrics=[]
            )
        
        finally:
            # Clean up model from memory
            if 'model' in locals():
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Log completion
        self.logger.log_custom(
            f"Evaluation completed for {model_version.version_id}",
            context={
                "duration_seconds": duration,
                "errors": errors
            }
        )
        
        # Create result
        result = BenchmarkResult(
            model_version=model_version,
            evaluation_results=evaluation_results,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            system_info=self._get_system_info(),
            errors=errors
        )
        
        # Cache result
        if self.config.cache_results:
            self._cache_result(cache_key, result)
        
        # Log to monitoring
        if self.monitoring:
            self.monitoring.log_metrics({
                "evaluation_duration": duration,
                "model_version": model_version.version_id,
                "num_benchmarks": len(evaluation_results.benchmarks)
            })
        
        return result
    
    def evaluate_fine_tuning(
        self,
        base_model: str,
        fine_tuned_model: str,
        benchmarks: Optional[List[str]] = None,
        custom_evaluations: Optional[List[Callable]] = None,
        parallel: Optional[bool] = None
    ) -> ComparisonResult:
        """Evaluate models before and after fine-tuning.
        
        Args:
            base_model: Path to base model
            fine_tuned_model: Path to fine-tuned model
            benchmarks: List of benchmarks to run
            custom_evaluations: Custom evaluation functions
            parallel: Run evaluations in parallel
            
        Returns:
            ComparisonResult
        """
        parallel = parallel if parallel is not None else self.config.parallel_benchmarks
        
        # Create evaluation ID
        eval_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.running_evaluations[eval_id] = {
            "state": EvaluationState.RUNNING,
            "start_time": datetime.now()
        }
        
        try:
            if parallel and self.config.max_parallel_jobs > 1:
                # Run evaluations in parallel
                logger.info("Running evaluations in parallel")
                
                future_base = self.executor.submit(
                    self.evaluate_model,
                    base_model,
                    benchmarks=benchmarks,
                    custom_evaluations=custom_evaluations
                )
                
                future_ft = self.executor.submit(
                    self.evaluate_model,
                    fine_tuned_model,
                    benchmarks=benchmarks,
                    custom_evaluations=custom_evaluations
                )
                
                # Wait for results with timeout
                timeout = self.config.timeout_minutes * 60
                base_result = future_base.result(timeout=timeout)
                ft_result = future_ft.result(timeout=timeout)
                
            else:
                # Run evaluations sequentially
                logger.info("Running evaluations sequentially")
                base_result = self.evaluate_model(
                    base_model,
                    benchmarks=benchmarks,
                    custom_evaluations=custom_evaluations
                )
                
                ft_result = self.evaluate_model(
                    fine_tuned_model,
                    benchmarks=benchmarks,
                    custom_evaluations=custom_evaluations
                )
            
            # Analyze comparison
            comparison = self._analyze_comparison(base_result, ft_result)
            
            # Update state
            self.running_evaluations[eval_id]["state"] = EvaluationState.COMPLETED
            self.running_evaluations[eval_id]["end_time"] = datetime.now()
            
            # Add to history
            self.evaluation_history.append({
                "eval_id": eval_id,
                "timestamp": datetime.now(),
                "comparison": comparison
            })
            
            # Save state
            self._save_state()
            
            return comparison
            
        except Exception as e:
            logger.error(f"Fine-tuning evaluation failed: {e}")
            self.running_evaluations[eval_id]["state"] = EvaluationState.FAILED
            self.running_evaluations[eval_id]["error"] = str(e)
            raise
    
    def _analyze_comparison(
        self,
        base_result: BenchmarkResult,
        ft_result: BenchmarkResult
    ) -> ComparisonResult:
        """Analyze comparison between base and fine-tuned models.
        
        Args:
            base_result: Base model results
            ft_result: Fine-tuned model results
            
        Returns:
            ComparisonResult
        """
        improvements = {}
        regressions = []
        
        # Compare benchmarks
        base_benchmarks = {b.name: b for b in base_result.evaluation_results.benchmarks}
        ft_benchmarks = {b.name: b for b in ft_result.evaluation_results.benchmarks}
        
        for benchmark_name in base_benchmarks:
            if benchmark_name in ft_benchmarks:
                base_score = base_benchmarks[benchmark_name].overall_score
                ft_score = ft_benchmarks[benchmark_name].overall_score
                
                improvement = ft_score - base_score
                improvement_pct = (improvement / base_score * 100) if base_score > 0 else 0
                
                improvements[benchmark_name] = {
                    "base_score": base_score,
                    "ft_score": ft_score,
                    "improvement": improvement,
                    "improvement_pct": improvement_pct
                }
                
                # Check for regression
                if improvement_pct < -5:  # 5% regression threshold
                    regressions.append({
                        "benchmark": benchmark_name,
                        "regression_pct": improvement_pct,
                        "base_score": base_score,
                        "ft_score": ft_score
                    })
        
        # Compare perplexity
        if base_result.evaluation_results.perplexity and ft_result.evaluation_results.perplexity:
            base_ppl = base_result.evaluation_results.perplexity
            ft_ppl = ft_result.evaluation_results.perplexity
            
            improvements["perplexity"] = {
                "base": base_ppl,
                "fine_tuned": ft_ppl,
                "improvement": base_ppl - ft_ppl,
                "improvement_pct": ((base_ppl - ft_ppl) / base_ppl * 100) if base_ppl > 0 else 0
            }
        
        # Statistical analysis
        statistical_analysis = self._perform_statistical_analysis(
            base_result, ft_result, improvements
        )
        
        return ComparisonResult(
            base_result=base_result,
            fine_tuned_result=ft_result,
            improvements=improvements,
            regressions=regressions,
            statistical_analysis=statistical_analysis
        )
    
    def _perform_statistical_analysis(
        self,
        base_result: BenchmarkResult,
        ft_result: BenchmarkResult,
        improvements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform statistical analysis on results.
        
        Args:
            base_result: Base model results
            ft_result: Fine-tuned model results
            improvements: Calculated improvements
            
        Returns:
            Statistical analysis results
        """
        analysis = {
            "summary": {},
            "significance": {}
        }
        
        # Calculate summary statistics
        improvement_values = [
            imp["improvement_pct"]
            for imp in improvements.values()
            if isinstance(imp, dict) and "improvement_pct" in imp
        ]
        
        if improvement_values:
            analysis["summary"] = {
                "mean_improvement_pct": np.mean(improvement_values),
                "median_improvement_pct": np.median(improvement_values),
                "std_improvement_pct": np.std(improvement_values),
                "min_improvement_pct": np.min(improvement_values),
                "max_improvement_pct": np.max(improvement_values)
            }
        
        # Perform significance tests (simplified)
        # In practice, would use proper statistical tests
        for benchmark, imp in improvements.items():
            if isinstance(imp, dict) and "improvement_pct" in imp:
                # Simple threshold-based significance
                is_significant = abs(imp["improvement_pct"]) > 5.0
                analysis["significance"][benchmark] = {
                    "is_significant": is_significant,
                    "confidence": 0.95 if is_significant else 0.5
                }
        
        return analysis
    
    def schedule_evaluation(
        self,
        base_model: str,
        fine_tuned_model: str,
        schedule_time: datetime,
        **kwargs
    ) -> str:
        """Schedule an evaluation for later execution.
        
        Args:
            base_model: Path to base model
            fine_tuned_model: Path to fine-tuned model
            schedule_time: When to run evaluation
            **kwargs: Additional evaluation parameters
            
        Returns:
            Schedule ID
        """
        schedule_id = f"schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save scheduled job
        scheduled_job = {
            "schedule_id": schedule_id,
            "base_model": base_model,
            "fine_tuned_model": fine_tuned_model,
            "schedule_time": schedule_time.isoformat(),
            "created_at": datetime.now().isoformat(),
            "status": "scheduled",
            "kwargs": kwargs
        }
        
        schedule_file = self.state_dir / f"scheduled_{schedule_id}.json"
        with open(schedule_file, 'w') as f:
            json.dump(scheduled_job, f, indent=2)
        
        logger.info(f"Scheduled evaluation {schedule_id} for {schedule_time}")
        
        return schedule_id
    
    def get_evaluation_state(self, eval_id: str) -> Dict[str, Any]:
        """Get current state of an evaluation.
        
        Args:
            eval_id: Evaluation ID
            
        Returns:
            Evaluation state
        """
        if eval_id in self.running_evaluations:
            return self.running_evaluations[eval_id]
        
        # Check history
        for hist in self.evaluation_history:
            if hist["eval_id"] == eval_id:
                return {
                    "state": EvaluationState.COMPLETED,
                    "comparison": hist["comparison"]
                }
        
        return {"state": EvaluationState.PENDING}
    
    def _load_model_and_tokenizer(
        self,
        model_path: str
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer.
        
        Args:
            model_path: Path to model
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model with appropriate class
        # Try to detect model type from config
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check for task-specific model
            if "num_labels" in config and config["num_labels"] > 1:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    device_map=self.config.device
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=self.config.device
                )
        else:
            # Default to causal LM
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=self.config.device
            )
        
        return model, tokenizer
    
    def _create_custom_eval_fn(
        self,
        custom_evaluations: Optional[List[Callable]]
    ) -> Optional[Callable]:
        """Create combined custom evaluation function.
        
        Args:
            custom_evaluations: List of custom evaluation functions
            
        Returns:
            Combined evaluation function
        """
        if not custom_evaluations:
            return None
        
        def combined_eval(model, tokenizer):
            results = []
            for eval_fn in custom_evaluations:
                try:
                    result = eval_fn(model, tokenizer)
                    if isinstance(result, list):
                        results.extend(result)
                    else:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Custom evaluation failed: {e}")
            return results
        
        return combined_eval
    
    def _get_cached_result(self, cache_key: str) -> Optional[BenchmarkResult]:
        """Get cached benchmark result.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached result or None
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        return None
    
    def _cache_result(self, cache_key: str, result: BenchmarkResult):
        """Cache benchmark result.
        
        Args:
            cache_key: Cache key
            result: Result to cache
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            "platform": os.uname().sysname,
            "python_version": os.sys.version,
            "device": self.config.device
        }
        
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return info
    
    def _save_state(self):
        """Save current state to disk."""
        state = {
            "running_evaluations": self.running_evaluations,
            "evaluation_history": self.evaluation_history[-100:]  # Keep last 100
        }
        
        state_file = self.state_dir / "runner_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def _load_state(self):
        """Load state from disk."""
        state_file = self.state_dir / "runner_state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.running_evaluations = state.get("running_evaluations", {})
                self.evaluation_history = state.get("evaluation_history", [])
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
    
    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
        
        if self.monitoring:
            self.monitoring.finish()
        
        self.logger.close()


# Example usage
if __name__ == "__main__":
    # Configure runner
    config = BenchmarkConfig(
        benchmarks=["hellaswag", "mmlu"],
        batch_size=8,
        max_samples=100,  # For testing
        cache_results=True,
        monitoring_enabled=True,
        parallel_benchmarks=True
    )
    
    # Create runner
    runner = AutoBenchmarkRunner(config)
    
    # Example: Evaluate fine-tuning
    try:
        comparison = runner.evaluate_fine_tuning(
            base_model="gpt2",
            fine_tuned_model="./fine_tuned/gpt2-custom",  # Example path
            benchmarks=["hellaswag"]
        )
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Base model performance:")
        for benchmark in comparison.base_result.evaluation_results.benchmarks:
            print(f"  {benchmark.name}: {benchmark.overall_score:.4f}")
        
        print(f"\nFine-tuned model performance:")
        for benchmark in comparison.fine_tuned_result.evaluation_results.benchmarks:
            print(f"  {benchmark.name}: {benchmark.overall_score:.4f}")
        
        print(f"\nImprovements:")
        for name, imp in comparison.improvements.items():
            if isinstance(imp, dict) and "improvement_pct" in imp:
                print(f"  {name}: {imp['improvement_pct']:.2f}%")
        
        if comparison.regressions:
            print(f"\nRegressions detected:")
            for reg in comparison.regressions:
                print(f"  {reg['benchmark']}: {reg['regression_pct']:.2f}%")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
    
    finally:
        runner.cleanup()