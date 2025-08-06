"""
Enhanced Fine-Tuning System for Local LLMs

This module provides a comprehensive fine-tuning framework with visual performance
tracking, recipe management, MacBook Pro optimization, and integrated API services.
"""

# Core fine-tuning functionality (optional if dependencies not available)
try:
    from .checkpoints import CheckpointManager
except ImportError:
    CheckpointManager = None

try:
    from .cli import main as cli_main
except ImportError:
    cli_main = None

try:
    from .evaluation import EvaluationSuite
except ImportError:
    EvaluationSuite = None

try:
    from .optimization import HyperparameterOptimizer
except ImportError:
    HyperparameterOptimizer = None

try:
    from .pipelines import DataPreprocessor, DataQualityReport
except ImportError:
    DataPreprocessor = None
    DataQualityReport = None

try:
    from .recipes import Recipe, RecipeManager
except ImportError:
    Recipe = None
    RecipeManager = None

try:
    from .training import DistributedTrainer
except ImportError:
    DistributedTrainer = None

try:
    from .visualization import TrainingDashboard
except ImportError:
    TrainingDashboard = None

# New integrated API and deployment services
try:
    from .api import (
        AuthHandler,
        CollaborationManager,
        Dataset,
        Deployment,
        Experiment,
        ExperimentCreate,
        ModelCardGenerator,
        VersioningSystem,
        api_app,
    )
except ImportError:
    api_app = None
    AuthHandler = None
    CollaborationManager = None
    VersioningSystem = None
    ModelCardGenerator = None
    ExperimentCreate = None
    Experiment = None
    Dataset = None
    Deployment = None

try:
    from .deployment import DeploymentPipeline
except ImportError:
    DeploymentPipeline = None

__all__ = [
    # Core fine-tuning
    "CheckpointManager",
    "DataPreprocessor",
    "DataQualityReport",
    "DistributedTrainer",
    "EvaluationSuite",
    "HyperparameterOptimizer",
    "Recipe",
    "RecipeManager",
    "TrainingDashboard",
    "cli_main",
    # API and services
    "api_app",
    "AuthHandler",
    "CollaborationManager",
    "VersioningSystem",
    "ModelCardGenerator",
    "ExperimentCreate",
    "Experiment",
    "Dataset",
    "Deployment",
    "DeploymentPipeline",
]
