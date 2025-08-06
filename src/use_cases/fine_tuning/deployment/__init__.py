"""
Fine-Tuning Model Deployment

This module provides deployment capabilities for fine-tuned models across
multiple platforms including HuggingFace Hub, local serving, and cloud providers.
"""

from .deploy import (
    DEPLOYMENT_TEMPLATES,
    AWSSageMakerDeployer,
    DeploymentConfig,
    DeploymentPipeline,
    DeploymentProvider,
    HuggingFaceDeployer,
    LocalVLLMDeployer,
    ModelDeployer,
)

__version__ = "1.0.0"

__all__ = [
    "DeploymentPipeline",
    "ModelDeployer",
    "HuggingFaceDeployer",
    "LocalVLLMDeployer",
    "AWSSageMakerDeployer",
    "DeploymentProvider",
    "DeploymentConfig",
    "DEPLOYMENT_TEMPLATES",
]


def get_deployment_pipeline():
    """Get a configured deployment pipeline instance"""
    return DeploymentPipeline()


def get_supported_providers():
    """Get list of supported deployment providers"""
    return list(DeploymentProvider)


def get_deployment_template(name: str):
    """Get a deployment template by name"""
    return DEPLOYMENT_TEMPLATES.get(name)
