"""
Integration Module for Evaluation Framework

This module provides integration with fine-tuning pipelines and other systems.
"""

from .pipeline_hooks import FineTuningHooks, HookConfig, HookType, PipelineIntegration

__all__ = ["FineTuningHooks", "HookType", "HookConfig", "PipelineIntegration"]
