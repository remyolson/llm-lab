"""
Red Team Simulation Framework

Advanced adversarial testing system for multi-step attack simulation and sophisticated
security assessment of Large Language Models.
"""

__version__ = "1.0.0"
__author__ = "LLM Lab Security Team"

# Core components
from .core.models import AttackChain, AttackScenario, AttackSession, AttackStep, CampaignResult
from .core.simulator import RedTeamSimulator

# Evasion techniques
from .evasion.engine import EvasionEngine
from .evasion.techniques import ContextManipulation, PrivilegeEscalation

# Scenario management
from .scenarios.base import BaseScenario
from .scenarios.registry import ScenarioRegistry

# Workflow management
from .workflows.engine import WorkflowEngine
from .workflows.models import AutomatedWorkflow, ManualWorkflow

__all__ = [
    "RedTeamSimulator",
    "AttackSession",
    "AttackScenario",
    "AttackChain",
    "AttackStep",
    "CampaignResult",
    "BaseScenario",
    "ScenarioRegistry",
    "EvasionEngine",
    "ContextManipulation",
    "PrivilegeEscalation",
    "WorkflowEngine",
    "AutomatedWorkflow",
    "ManualWorkflow",
]
