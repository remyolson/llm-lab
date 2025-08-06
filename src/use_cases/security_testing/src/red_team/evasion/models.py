"""
Data models for evasion techniques and results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class EvasionCategory(Enum):
    """Categories of evasion techniques."""

    CONTEXT_MANIPULATION = "context_manipulation"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    OBFUSCATION = "obfuscation"
    TIMING_BASED = "timing_based"
    ADAPTIVE = "adaptive"
    JAILBREAK = "jailbreak"
    ROLE_CONFUSION = "role_confusion"
    PROMPT_INJECTION = "prompt_injection"


class EvasionEffectiveness(Enum):
    """Effectiveness levels for evasion techniques."""

    LOW = "low"  # 0-30% success rate
    MEDIUM = "medium"  # 30-60% success rate
    HIGH = "high"  # 60-85% success rate
    CRITICAL = "critical"  # 85-100% success rate


@dataclass
class EvasionResult:
    """Result of applying an evasion technique."""

    technique_name: str
    category: EvasionCategory
    success: bool
    effectiveness_score: float  # 0.0 to 1.0
    original_prompt: str
    modified_prompt: str
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def get_effectiveness_level(self) -> EvasionEffectiveness:
        """Get effectiveness level based on score."""
        if self.effectiveness_score >= 0.85:
            return EvasionEffectiveness.CRITICAL
        elif self.effectiveness_score >= 0.60:
            return EvasionEffectiveness.HIGH
        elif self.effectiveness_score >= 0.30:
            return EvasionEffectiveness.MEDIUM
        else:
            return EvasionEffectiveness.LOW


@dataclass
class EvasionConfig:
    """Configuration for evasion techniques."""

    enabled_categories: List[EvasionCategory] = field(
        default_factory=lambda: [
            EvasionCategory.CONTEXT_MANIPULATION,
            EvasionCategory.OBFUSCATION,
            EvasionCategory.ROLE_CONFUSION,
        ]
    )

    # Context manipulation settings
    max_context_history: int = 5
    conversation_poisoning_enabled: bool = True
    role_confusion_enabled: bool = True

    # Obfuscation settings
    character_substitution_enabled: bool = True
    semantic_paraphrasing_enabled: bool = True
    multi_language_enabled: bool = False
    encoding_obfuscation_enabled: bool = True

    # Timing and adaptive settings
    delayed_payload_enabled: bool = False
    adaptive_learning_enabled: bool = True
    retry_failed_techniques: bool = True
    max_retries: int = 3

    # Jailbreak settings
    jailbreak_library_enabled: bool = True
    custom_jailbreak_patterns: List[str] = field(default_factory=list)

    # Safety and limits
    max_prompt_length: int = 4000
    ethical_boundaries: List[str] = field(
        default_factory=lambda: [
            "no_harmful_content",
            "no_illegal_activities",
            "defensive_testing_only",
        ]
    )


class EvasionTechnique(ABC):
    """Base class for all evasion techniques."""

    def __init__(
        self, name: str, category: EvasionCategory, config: Optional[EvasionConfig] = None
    ):
        self.name = name
        self.category = category
        self.config = config or EvasionConfig()
        self.effectiveness_history: List[float] = []
        self.usage_count = 0

    @abstractmethod
    def apply(self, prompt: str, context: Dict[str, Any]) -> EvasionResult:
        """Apply the evasion technique to a prompt."""
        pass

    @abstractmethod
    def is_applicable(self, prompt: str, context: Dict[str, Any]) -> bool:
        """Check if this technique is applicable to the given prompt and context."""
        pass

    def update_effectiveness(self, success: bool, score: float):
        """Update effectiveness tracking."""
        self.usage_count += 1
        self.effectiveness_history.append(score)

        # Keep only recent history
        if len(self.effectiveness_history) > 100:
            self.effectiveness_history = self.effectiveness_history[-100:]

    def get_average_effectiveness(self) -> float:
        """Get average effectiveness score."""
        if not self.effectiveness_history:
            return 0.0
        return sum(self.effectiveness_history) / len(self.effectiveness_history)

    def get_success_rate(self) -> float:
        """Get success rate (percentage of successful applications)."""
        if not self.effectiveness_history:
            return 0.0
        successes = sum(1 for score in self.effectiveness_history if score > 0.5)
        return successes / len(self.effectiveness_history)


@dataclass
class EvasionStrategy:
    """Strategy for applying multiple evasion techniques."""

    name: str
    description: str
    techniques: List[str]  # Technique names
    application_order: str = "sequential"  # "sequential", "parallel", "adaptive"
    combination_logic: str = "best_result"  # "best_result", "merge_techniques", "cascade"
    success_threshold: float = 0.7
    max_techniques_per_prompt: int = 3

    # Adaptive strategy parameters
    learning_enabled: bool = True
    technique_selection_weights: Dict[str, float] = field(default_factory=dict)

    def update_technique_weight(self, technique_name: str, effectiveness: float):
        """Update technique selection weights based on effectiveness."""
        if technique_name not in self.technique_selection_weights:
            self.technique_selection_weights[technique_name] = 1.0

        # Exponential moving average for weight updates
        alpha = 0.3
        current_weight = self.technique_selection_weights[technique_name]
        new_weight = alpha * effectiveness + (1 - alpha) * current_weight
        self.technique_selection_weights[technique_name] = max(0.1, min(2.0, new_weight))


@dataclass
class ContextState:
    """State information for context-based evasion techniques."""

    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    established_roles: List[str] = field(default_factory=list)
    claimed_authorities: List[str] = field(default_factory=list)
    injected_context: Dict[str, Any] = field(default_factory=dict)
    successful_patterns: List[str] = field(default_factory=list)
    failed_patterns: List[str] = field(default_factory=list)
    session_variables: Dict[str, Any] = field(default_factory=dict)

    def add_conversation_turn(self, role: str, content: str):
        """Add a conversation turn to the history."""
        self.conversation_history.append(
            {"timestamp": datetime.now().isoformat(), "role": role, "content": content}
        )

        # Keep only recent history
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def establish_role(self, role: str):
        """Establish a role in the conversation."""
        if role not in self.established_roles:
            self.established_roles.append(role)

    def claim_authority(self, authority: str):
        """Claim an authority or privilege."""
        if authority not in self.claimed_authorities:
            self.claimed_authorities.append(authority)

    def inject_context_variable(self, key: str, value: Any):
        """Inject a context variable."""
        self.injected_context[key] = value

    def record_successful_pattern(self, pattern: str):
        """Record a successful evasion pattern."""
        if pattern not in self.successful_patterns:
            self.successful_patterns.append(pattern)

    def record_failed_pattern(self, pattern: str):
        """Record a failed evasion pattern."""
        if pattern not in self.failed_patterns:
            self.failed_patterns.append(pattern)


@dataclass
class JailbreakPattern:
    """A specific jailbreak pattern with metadata."""

    name: str
    pattern: str
    description: str
    effectiveness_score: float
    category: str
    target_models: List[str] = field(default_factory=list)
    success_indicators: List[str] = field(default_factory=list)
    failure_indicators: List[str] = field(default_factory=list)
    created_date: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    usage_count: int = 0

    def apply_to_prompt(self, base_prompt: str) -> str:
        """Apply the jailbreak pattern to a base prompt."""
        # Replace placeholders in the pattern
        modified_pattern = self.pattern.replace("{prompt}", base_prompt)
        return modified_pattern

    def update_usage(self, success: bool):
        """Update usage statistics."""
        self.usage_count += 1
        self.last_used = datetime.now()

        # Update effectiveness based on success
        if success:
            self.effectiveness_score = min(1.0, self.effectiveness_score + 0.1)
        else:
            self.effectiveness_score = max(0.0, self.effectiveness_score - 0.05)
