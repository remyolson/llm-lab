"""
Red Team workflow engine package for automated and manual testing orchestration.
"""

from .campaign_manager import CampaignManager
from .manual_interface import ManualRedTeamInterface
from .scoring import RealTimeScorer, ScoringConfig
from .templates import WorkflowTemplate, WorkflowTemplateLibrary
from .workflow_engine import WorkflowEngine, WorkflowExecutor

__all__ = [
    "WorkflowEngine",
    "WorkflowExecutor",
    "CampaignManager",
    "ManualRedTeamInterface",
    "RealTimeScorer",
    "ScoringConfig",
    "WorkflowTemplate",
    "WorkflowTemplateLibrary",
]
