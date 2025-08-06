"""Runtime intervention framework for alignment research."""

from .base import (
    AlignmentContext,
    InterventionPipeline,
    InterventionResult,
    InterventionStrategy,
    InterventionType,
    OutputFilter,
    PromptModifier,
    ResponseModifier,
    SafetyChecker,
)
from .intervention import (
    ContentSafetyChecker,
    InterventionConfig,
    PromptRewriter,
    RuntimeInterventionSystem,
    ToneAdjuster,
    ToxicityFilter,
    create_default_system,
)
from .strategies import (
    BiasRemovalFilter,
    ContextPreserver,
    EthicalGuardrailsChecker,
    FactualityEnforcer,
    HelpfulnessEnhancer,
    InstructionClarifier,
    get_content_moderation_strategy,
    get_educational_strategy,
    get_professional_strategy,
)

__all__ = [
    # Base classes
    "InterventionType",
    "InterventionResult",
    "AlignmentContext",
    "InterventionStrategy",
    "OutputFilter",
    "ResponseModifier",
    "PromptModifier",
    "SafetyChecker",
    "InterventionPipeline",
    # Core intervention system
    "InterventionConfig",
    "PromptRewriter",
    "ToxicityFilter",
    "ToneAdjuster",
    "ContentSafetyChecker",
    "RuntimeInterventionSystem",
    "create_default_system",
    # Concrete strategies
    "BiasRemovalFilter",
    "FactualityEnforcer",
    "EthicalGuardrailsChecker",
    "InstructionClarifier",
    "HelpfulnessEnhancer",
    "ContextPreserver",
    "get_content_moderation_strategy",
    "get_educational_strategy",
    "get_professional_strategy",
]
