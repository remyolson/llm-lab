"""
LLM Monitoring Dashboard

A comprehensive web dashboard for visualizing and monitoring LLM performance,
cost analysis, and system health across multiple providers.

This package provides:
- Real-time performance monitoring
- Interactive visualizations
- Automated report generation
- Role-based access control
- Data export capabilities
"""

from .app import create_app
from .config.settings import DashboardConfig

__version__ = "1.0.0"
__all__ = ["create_app", "DashboardConfig"]