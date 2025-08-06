"""
Hyperparameter Optimization Engine for Fine-Tuning

This module provides automated hyperparameter search and optimization using
Optuna as the primary backend. It supports multiple search strategies and
integrates with the recipe system for search space definition.

Example:
    optimizer = HyperparameterOptimizer(
        objective_metric="eval_loss",
        direction="minimize"
    )

    # Define search space
    search_space = {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
        "batch_size": {"type": "categorical", "choices": [4, 8, 16]}
    }

    # Run optimization
    best_params = optimizer.optimize(
        train_function=train_model,
        search_space=search_space,
        n_trials=20
    )
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Configure logging first
logger = logging.getLogger(__name__)

# Optuna for hyperparameter optimization
try:
    import optuna
    from optuna import Study, Trial
    from optuna.pruners import HyperbandPruner, MedianPruner
    from optuna.samplers import GridSampler, RandomSampler, TPESampler
    from optuna.visualization import (
        plot_contour,
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
    )

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")

# Ray Tune as alternative
try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""

    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    optimization_history: List[Dict[str, Any]]
    param_importance: Optional[Dict[str, float]] = None
    study_name: Optional[str] = None
    duration_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, path: str | Path):
        """Save results to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "OptimizationResult":
        """Load results from file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class SearchSpace:
    """Defines hyperparameter search space."""

    def __init__(self, space_config: Dict[str, Dict[str, Any]]):
        """Initialize search space.

        Args:
            space_config: Dictionary defining search space where keys are
                         parameter names and values are parameter configs
        """
        self.space_config = space_config
        self._validate_space()

    def _validate_space(self):
        """Validate search space configuration."""
        valid_types = {"float", "int", "categorical", "discrete_uniform"}

        for param_name, param_config in self.space_config.items():
            if "type" not in param_config:
                raise ValueError(f"Parameter '{param_name}' missing 'type'")

            param_type = param_config["type"]
            if param_type not in valid_types:
                raise ValueError(f"Invalid type '{param_type}' for parameter '{param_name}'")

            # Validate type-specific requirements
            if param_type in ["float", "int"]:
                if "low" not in param_config or "high" not in param_config:
                    raise ValueError(f"Parameter '{param_name}' requires 'low' and 'high'")

            elif param_type == "categorical":
                if "choices" not in param_config:
                    raise ValueError(f"Parameter '{param_name}' requires 'choices'")

            elif param_type == "discrete_uniform":
                if (
                    "low" not in param_config
                    or "high" not in param_config
                    or "q" not in param_config
                ):
                    raise ValueError(f"Parameter '{param_name}' requires 'low', 'high', and 'q'")

    def sample_optuna(self, trial: Trial) -> Dict[str, Any]:
        """Sample parameters using Optuna trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of sampled parameters
        """
        params = {}

        for param_name, param_config in self.space_config.items():
            param_type = param_config["type"]

            if param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )

            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )

            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, param_config["choices"])

            elif param_type == "discrete_uniform":
                params[param_name] = trial.suggest_discrete_uniform(
                    param_name, param_config["low"], param_config["high"], param_config["q"]
                )

        return params

    def get_grid(self) -> List[Dict[str, Any]]:
        """Get all parameter combinations for grid search.

        Returns:
            List of parameter dictionaries
        """
        import itertools

        # Extract discrete values for each parameter
        param_values = {}

        for param_name, param_config in self.space_config.items():
            param_type = param_config["type"]

            if param_type == "categorical":
                param_values[param_name] = param_config["choices"]

            elif param_type in ["float", "int"]:
                # For continuous parameters, use grid points if specified
                if "grid_points" in param_config:
                    param_values[param_name] = param_config["grid_points"]
                else:
                    # Default to 3 points
                    low = param_config["low"]
                    high = param_config["high"]

                    if param_config.get("log", False):
                        points = np.logspace(np.log10(low), np.log10(high), 3)
                    else:
                        points = np.linspace(low, high, 3)

                    if param_type == "int":
                        points = [int(p) for p in points]

                    param_values[param_name] = points.tolist()

            elif param_type == "discrete_uniform":
                low = param_config["low"]
                high = param_config["high"]
                q = param_config["q"]
                param_values[param_name] = list(np.arange(low, high + q, q))

        # Generate all combinations
        keys = param_values.keys()
        values = param_values.values()

        grid = []
        for combination in itertools.product(*values):
            grid.append(dict(zip(keys, combination)))

        return grid


class HyperparameterOptimizer:
    """Main hyperparameter optimization engine."""

    def __init__(
        self,
        objective_metric: str,
        direction: str = "minimize",
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        backend: str = "optuna",
    ):
        """Initialize hyperparameter optimizer.

        Args:
            objective_metric: Name of the metric to optimize
            direction: Optimization direction ('minimize' or 'maximize')
            study_name: Name for the optimization study
            storage: Storage URL for distributed optimization
            backend: Optimization backend ('optuna' or 'ray')
        """
        self.objective_metric = objective_metric
        self.direction = direction
        self.study_name = study_name or f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage = storage
        self.backend = backend

        if backend == "optuna" and not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required but not installed")
        elif backend == "ray" and not RAY_AVAILABLE:
            raise ImportError("Ray is required but not installed")

        self.study = None
        self.optimization_history = []

    def _create_optuna_study(
        self,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ) -> Study:
        """Create Optuna study.

        Args:
            sampler: Optuna sampler for search strategy
            pruner: Optuna pruner for early stopping

        Returns:
            Optuna study object
        """
        if sampler is None:
            sampler = TPESampler(seed=42)

        if pruner is None:
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
            storage=self.storage,
            load_if_exists=True,
        )

        return study

    def optimize(
        self,
        train_function: Callable,
        search_space: SearchSpace | Dict[str | Dict[str, Any]],
        n_trials: int = 20,
        n_jobs: int = 1,
        timeout: Optional[float] = None,
        strategy: str = "tpe",
        callbacks: Optional[List[Callable]] = None,
        catch_exceptions: bool = True,
        **kwargs,
    ) -> OptimizationResult:
        """Run hyperparameter optimization.

        Args:
            train_function: Function that trains model and returns metric
            search_space: Search space definition
            n_trials: Number of trials to run
            n_jobs: Number of parallel jobs
            timeout: Timeout in seconds
            strategy: Search strategy ('tpe', 'random', 'grid')
            callbacks: List of callback functions
            catch_exceptions: Whether to catch exceptions in trials
            **kwargs: Additional arguments passed to train_function

        Returns:
            OptimizationResult object
        """
        # Convert dict to SearchSpace if needed
        if isinstance(search_space, dict):
            search_space = SearchSpace(search_space)

        start_time = datetime.now()

        if self.backend == "optuna":
            result = self._optimize_optuna(
                train_function=train_function,
                search_space=search_space,
                n_trials=n_trials,
                n_jobs=n_jobs,
                timeout=timeout,
                strategy=strategy,
                callbacks=callbacks,
                catch_exceptions=catch_exceptions,
                **kwargs,
            )
        else:
            result = self._optimize_ray(
                train_function=train_function,
                search_space=search_space,
                n_trials=n_trials,
                n_jobs=n_jobs,
                timeout=timeout,
                strategy=strategy,
                **kwargs,
            )

        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        result.duration_seconds = duration

        return result

    def _optimize_optuna(
        self,
        train_function: Callable,
        search_space: SearchSpace,
        n_trials: int,
        n_jobs: int,
        timeout: Optional[float],
        strategy: str,
        callbacks: Optional[List[Callable]],
        catch_exceptions: bool,
        **kwargs,
    ) -> OptimizationResult:
        """Run optimization using Optuna."""
        # Create sampler based on strategy
        if strategy == "tpe":
            sampler = TPESampler(seed=42)
        elif strategy == "random":
            sampler = RandomSampler(seed=42)
        elif strategy == "grid":
            # Grid search requires explicit search space
            grid = search_space.get_grid()
            sampler = GridSampler(search_space=grid)
            n_trials = len(grid)  # Override n_trials for grid search
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Create study
        self.study = self._create_optuna_study(sampler=sampler)

        # Define objective function
        def objective(trial: Trial) -> float:
            # Sample parameters
            params = search_space.sample_optuna(trial)

            # Log trial start
            logger.info(f"Trial {trial.number}: {params}")

            try:
                # Run training with sampled parameters
                result = train_function(params, trial=trial, **kwargs)

                # Extract metric value
                if isinstance(result, dict):
                    value = result[self.objective_metric]
                else:
                    value = float(result)

                # Record in history
                self.optimization_history.append(
                    {"trial": trial.number, "params": params, "value": value, "state": "COMPLETE"}
                )

                return value

            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")

                if catch_exceptions:
                    # Record failure
                    self.optimization_history.append(
                        {
                            "trial": trial.number,
                            "params": params,
                            "value": None,
                            "state": "FAIL",
                            "error": str(e),
                        }
                    )

                    # Optuna will handle the pruned trial
                    raise optuna.TrialPruned()
                else:
                    raise

        # Add callbacks
        if callbacks:
            for callback in callbacks:
                self.study.optimize(objective, n_trials=1, n_jobs=1, callbacks=[callback])

        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            catch=(Exception,) if catch_exceptions else (),
        )

        # Get results
        best_trial = self.study.best_trial

        # Calculate parameter importance
        try:
            importance = optuna.importance.get_param_importances(self.study)
        except:
            importance = None

        result = OptimizationResult(
            best_params=best_trial.params,
            best_value=best_trial.value,
            n_trials=len(self.study.trials),
            optimization_history=self.optimization_history,
            param_importance=importance,
            study_name=self.study_name,
        )

        return result

    def _optimize_ray(
        self,
        train_function: Callable,
        search_space: SearchSpace,
        n_trials: int,
        n_jobs: int,
        timeout: Optional[float],
        strategy: str,
        **kwargs,
    ) -> OptimizationResult:
        """Run optimization using Ray Tune."""
        # Convert search space to Ray format
        ray_search_space = {}

        for param_name, param_config in search_space.space_config.items():
            param_type = param_config["type"]

            if param_type == "float":
                if param_config.get("log", False):
                    ray_search_space[param_name] = tune.loguniform(
                        param_config["low"], param_config["high"]
                    )
                else:
                    ray_search_space[param_name] = tune.uniform(
                        param_config["low"], param_config["high"]
                    )

            elif param_type == "int":
                ray_search_space[param_name] = tune.randint(
                    param_config["low"], param_config["high"] + 1
                )

            elif param_type == "categorical":
                ray_search_space[param_name] = tune.choice(param_config["choices"])

        # Configure scheduler
        if strategy == "asha":
            scheduler = ASHAScheduler(
                metric=self.objective_metric,
                mode="min" if self.direction == "minimize" else "max",
                max_t=100,
                grace_period=10,
            )
        else:
            scheduler = None

        # Wrap train function for Ray
        def ray_train_wrapper(config):
            result = train_function(config, **kwargs)

            if isinstance(result, dict):
                tune.report(**result)
            else:
                tune.report(**{self.objective_metric: result})

        # Run optimization
        analysis = tune.run(
            ray_train_wrapper,
            config=ray_search_space,
            num_samples=n_trials,
            scheduler=scheduler,
            resources_per_trial={"cpu": n_jobs},
            verbose=1,
        )

        # Get best result
        best_config = analysis.get_best_config(
            metric=self.objective_metric, mode="min" if self.direction == "minimize" else "max"
        )

        best_result = analysis.get_best_trial(
            metric=self.objective_metric, mode="min" if self.direction == "minimize" else "max"
        )

        # Build optimization history
        optimization_history = []
        for trial in analysis.trials:
            optimization_history.append(
                {
                    "trial": trial.trial_id,
                    "params": trial.config,
                    "value": trial.last_result.get(self.objective_metric),
                    "state": trial.status,
                }
            )

        result = OptimizationResult(
            best_params=best_config,
            best_value=best_result.last_result[self.objective_metric],
            n_trials=len(analysis.trials),
            optimization_history=optimization_history,
            study_name=self.study_name,
        )

        return result

    def visualize_optimization(self, result: OptimizationResult, output_dir: str | Path):
        """Generate visualization plots for optimization results.

        Args:
            result: OptimizationResult object
            output_dir: Directory to save plots
        """
        if self.backend != "optuna" or self.study is None:
            logger.warning("Visualization only available for Optuna backend")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Optimization history
        try:
            fig = plot_optimization_history(self.study)
            fig.write_html(output_dir / "optimization_history.html")
            fig.write_image(output_dir / "optimization_history.png")
        except Exception as e:
            logger.error(f"Failed to create optimization history plot: {e}")

        # Parameter importance
        if result.param_importance:
            try:
                fig = plot_param_importances(self.study)
                fig.write_html(output_dir / "param_importance.html")
                fig.write_image(output_dir / "param_importance.png")
            except Exception as e:
                logger.error(f"Failed to create parameter importance plot: {e}")

        # Contour plot for 2D relationships
        if len(result.best_params) >= 2:
            try:
                params = list(result.best_params.keys())[:2]
                fig = plot_contour(self.study, params=params)
                fig.write_html(output_dir / "contour_plot.html")
                fig.write_image(output_dir / "contour_plot.png")
            except Exception as e:
                logger.error(f"Failed to create contour plot: {e}")

        # Parallel coordinate plot
        try:
            fig = plot_parallel_coordinate(self.study)
            fig.write_html(output_dir / "parallel_coordinate.html")
            fig.write_image(output_dir / "parallel_coordinate.png")
        except Exception as e:
            logger.error(f"Failed to create parallel coordinate plot: {e}")

    def resume_optimization(
        self,
        train_function: Callable,
        search_space: SearchSpace | Dict[str | Dict[str, Any]],
        n_additional_trials: int = 10,
        **kwargs,
    ) -> OptimizationResult:
        """Resume a previous optimization study.

        Args:
            train_function: Training function
            search_space: Search space definition
            n_additional_trials: Number of additional trials to run
            **kwargs: Additional arguments

        Returns:
            Updated OptimizationResult
        """
        if self.backend != "optuna":
            raise NotImplementedError("Resume only supported for Optuna backend")

        if self.study is None:
            # Load existing study
            self.study = optuna.load_study(study_name=self.study_name, storage=self.storage)

        # Continue optimization
        return self.optimize(
            train_function=train_function,
            search_space=search_space,
            n_trials=n_additional_trials,
            **kwargs,
        )

    @staticmethod
    def create_from_recipe(recipe_config: Dict[str, Any]) -> "HyperparameterOptimizer":
        """Create optimizer from recipe configuration.

        Args:
            recipe_config: Recipe configuration dictionary

        Returns:
            Configured HyperparameterOptimizer
        """
        optimization_config = recipe_config.get("optimization", {})

        return HyperparameterOptimizer(
            objective_metric=optimization_config.get("metric", "eval_loss"),
            direction=optimization_config.get("direction", "minimize"),
            study_name=optimization_config.get("study_name"),
            backend=optimization_config.get("backend", "optuna"),
        )


# Utility functions for common search spaces
def get_default_search_space(model_type: str = "transformer") -> Dict[str | Dict[str, Any]]:
    """Get default search space for common model types.

    Args:
        model_type: Type of model ('transformer', 'lora', etc.)

    Returns:
        Search space configuration
    """
    if model_type == "transformer":
        return {
            "learning_rate": {"type": "float", "low": 1e-5, "high": 5e-4, "log": True},
            "per_device_train_batch_size": {"type": "categorical", "choices": [4, 8, 16]},
            "gradient_accumulation_steps": {"type": "categorical", "choices": [1, 2, 4]},
            "warmup_ratio": {"type": "float", "low": 0.0, "high": 0.1},
            "weight_decay": {"type": "float", "low": 0.0, "high": 0.1},
        }

    elif model_type == "lora":
        return {
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-3, "log": True},
            "lora_rank": {"type": "categorical", "choices": [8, 16, 32, 64]},
            "lora_alpha": {"type": "categorical", "choices": [16, 32, 64]},
            "lora_dropout": {"type": "float", "low": 0.0, "high": 0.2},
        }

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Example usage
if __name__ == "__main__":
    # Create optimizer
    optimizer = HyperparameterOptimizer(objective_metric="eval_loss", direction="minimize")

    # Define search space
    search_space = get_default_search_space("lora")

    # Mock training function
    def train_model(params: Dict[str, Any], trial=None) -> Dict[str, float]:
        """Mock training function for testing."""
        # Simulate training
        import random
        import time

        time.sleep(0.1)  # Simulate training time

        # Mock metrics based on hyperparameters
        lr = params["learning_rate"]
        rank = params["lora_rank"]

        # Simulate that lower learning rate and moderate rank are better
        loss = abs(lr - 3e-4) * 1000 + abs(rank - 16) * 0.01 + random.random() * 0.1

        # Report intermediate values for pruning
        if trial is not None:
            for step in range(10):
                intermediate_loss = loss + (10 - step) * 0.1
                trial.report(intermediate_loss, step)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return {"eval_loss": loss, "train_loss": loss * 0.9, "accuracy": 0.9 - loss * 0.1}

    # Run optimization
    print("Running hyperparameter optimization...")
    result = optimizer.optimize(
        train_function=train_model, search_space=search_space, n_trials=10, strategy="tpe"
    )

    print(f"\nBest parameters: {result.best_params}")
    print(f"Best value: {result.best_value:.4f}")
    print(f"Total trials: {result.n_trials}")
