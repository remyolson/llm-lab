"""
Fine-Tuning Studio API

This module provides a comprehensive API for managing fine-tuning experiments,
including experiment management, dataset handling, model deployment, and A/B testing.
"""

# Import core app (optional if FastAPI not available)
try:
    from .main import app as api_app
except ImportError:
    api_app = None

from .models import (
    ABTest,
    # A/B Testing Models
    ABTestConfig,
    # Response Models
    APIResponse,
    Dataset,
    DatasetConfig,
    # Dataset Models
    DatasetCreate,
    DatasetValidation,
    Deployment,
    # Deployment Models
    DeploymentConfig,
    DeploymentTarget,
    EvaluationConfig,
    Experiment,
    # Experiment Models
    ExperimentCreate,
    ExperimentStatus,
    ExperimentUpdate,
    ListResponse,
    # Configuration Models
    ModelConfig,
    MonitoringConfig,
    Recipe,
    TrainingConfig,
)

# Import core services (optional if dependencies not available)
try:
    from .auth import AuthHandler, setup_auth_routes
except ImportError:
    AuthHandler = None
    setup_auth_routes = None

try:
    from .websocket import ConnectionManager, setup_websocket_routes
except ImportError:
    ConnectionManager = None
    setup_websocket_routes = None

try:
    from .collaboration import CollaborationManager
except ImportError:
    CollaborationManager = None

try:
    from .versioning import VersioningSystem
except ImportError:
    VersioningSystem = None

try:
    from .model_cards import ModelCardData, ModelCardGenerator
except ImportError:
    ModelCardGenerator = None
    ModelCardData = None

# Import deployment services
try:
    from ..deployment.deploy import DeploymentPipeline
except ImportError:
    DeploymentPipeline = None

__version__ = "1.0.0"

__all__ = [
    # FastAPI app
    "api_app",
    # Configuration Models
    "ModelConfig",
    "DatasetConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "MonitoringConfig",
    "Recipe",
    # Experiment Models
    "ExperimentCreate",
    "ExperimentUpdate",
    "Experiment",
    "ExperimentStatus",
    # Dataset Models
    "DatasetCreate",
    "Dataset",
    "DatasetValidation",
    # Deployment Models
    "DeploymentConfig",
    "Deployment",
    "DeploymentTarget",
    # A/B Testing Models
    "ABTestConfig",
    "ABTest",
    # Response Models
    "APIResponse",
    "ListResponse",
    # Core Services
    "AuthHandler",
    "setup_auth_routes",
    "ConnectionManager",
    "setup_websocket_routes",
    "CollaborationManager",
    "VersioningSystem",
    "ModelCardGenerator",
    "ModelCardData",
    # Deployment Services
    "DeploymentPipeline",
]


def create_app(config=None):
    """Create and configure the FastAPI application"""
    return api_app


def get_version():
    """Get the API version"""
    return __version__
