"""
Fine-Tuning Studio Web Interface

This module provides the web interface components for the Fine-Tuning Studio,
including React components, hooks, pages, and utilities.
"""

# This module contains frontend files that are built and served separately
# The Python __init__.py file is mainly for organizational purposes

__version__ = "1.0.0"

# Web interface metadata
WEB_CONFIG = {
    "name": "Fine-Tuning Studio Web",
    "version": __version__,
    "framework": "Next.js",
    "ui_library": "Material-UI",
    "build_dir": "dist",
    "dev_port": 3001,
    "prod_port": 3000,
}


def get_web_config():
    """Get web interface configuration"""
    return WEB_CONFIG


def get_component_list():
    """Get list of available React components"""
    return [
        "Navigation",
        "ABTesting",
        "DatasetExplorer",
        "ErrorBoundary",
        "LivePreview",
        "QualityAnalysis",
    ]


def get_page_list():
    """Get list of available pages"""
    return [
        "index",
        "dashboard",
        "experiments",
        "experiments_new",
        "datasets",
        "models",
        "deploy",
        "ab_testing",
        "settings",
    ]
