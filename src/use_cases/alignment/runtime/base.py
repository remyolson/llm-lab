"""Base classes for runtime alignment intervention framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class InterventionType(Enum):
    """Types of interventions that can be applied."""
    PROMPT_MODIFICATION = "prompt_modification"
    OUTPUT_FILTERING = "output_filtering"
    RESPONSE_STEERING = "response_steering"
    SAFETY_CHECK = "safety_check"


@dataclass
class InterventionResult:
    """Result of applying an intervention."""
    intervention_type: InterventionType
    original_value: Any
    modified_value: Any
    metadata: Dict[str, Any]
    applied: bool
    reason: Optional[str] = None
    
    @property
    def was_modified(self) -> bool:
        """Check if the intervention actually modified the value."""
        return self.applied and self.original_value != self.modified_value


@dataclass
class AlignmentContext:
    """Context information for alignment interventions."""
    model_name: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_history: List[Dict[str, str]] = None
    custom_rules: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.custom_rules is None:
            self.custom_rules = []
        if self.metadata is None:
            self.metadata = {}


class InterventionStrategy(ABC):
    """Abstract base class for intervention strategies."""
    
    def __init__(self, name: str, priority: int = 0):
        """
        Initialize intervention strategy.
        
        Args:
            name: Name of the strategy
            priority: Priority for ordering (higher = applied first)
        """
        self.name = name
        self.priority = priority
        self.enabled = True
    
    @abstractmethod
    def should_intervene(self, value: Any, context: AlignmentContext) -> bool:
        """
        Determine if intervention should be applied.
        
        Args:
            value: The value to potentially intervene on
            context: Current alignment context
            
        Returns:
            True if intervention should be applied
        """
        pass
    
    @abstractmethod
    def apply(self, value: Any, context: AlignmentContext) -> Tuple[Any, Dict[str, Any]]:
        """
        Apply the intervention.
        
        Args:
            value: The value to intervene on
            context: Current alignment context
            
        Returns:
            Tuple of (modified_value, metadata)
        """
        pass
    
    @property
    @abstractmethod
    def intervention_type(self) -> InterventionType:
        """Return the type of intervention this strategy performs."""
        pass


class OutputFilter(InterventionStrategy):
    """Base class for output filtering strategies."""
    
    @property
    def intervention_type(self) -> InterventionType:
        return InterventionType.OUTPUT_FILTERING
    
    @abstractmethod
    def filter(self, text: str, context: AlignmentContext) -> Tuple[str, bool]:
        """
        Filter the output text.
        
        Args:
            text: Text to filter
            context: Current alignment context
            
        Returns:
            Tuple of (filtered_text, was_modified)
        """
        pass
    
    def apply(self, value: Any, context: AlignmentContext) -> Tuple[Any, Dict[str, Any]]:
        """Apply output filtering."""
        if not isinstance(value, str):
            return value, {"error": "OutputFilter requires string input"}
        
        filtered_text, was_modified = self.filter(value, context)
        return filtered_text, {
            "was_modified": was_modified,
            "filter_name": self.name
        }


class ResponseModifier(InterventionStrategy):
    """Base class for response modification strategies."""
    
    @property
    def intervention_type(self) -> InterventionType:
        return InterventionType.RESPONSE_STEERING
    
    @abstractmethod
    def modify_generation_params(
        self, 
        params: Dict[str, Any], 
        context: AlignmentContext
    ) -> Dict[str, Any]:
        """
        Modify generation parameters to steer response.
        
        Args:
            params: Original generation parameters
            context: Current alignment context
            
        Returns:
            Modified generation parameters
        """
        pass
    
    def apply(self, value: Any, context: AlignmentContext) -> Tuple[Any, Dict[str, Any]]:
        """Apply response modification."""
        if not isinstance(value, dict):
            return value, {"error": "ResponseModifier requires dict input"}
        
        modified_params = self.modify_generation_params(value, context)
        return modified_params, {
            "modifications": [
                key for key in modified_params 
                if key not in value or modified_params[key] != value[key]
            ]
        }


class PromptModifier(InterventionStrategy):
    """Base class for prompt modification strategies."""
    
    @property
    def intervention_type(self) -> InterventionType:
        return InterventionType.PROMPT_MODIFICATION
    
    @abstractmethod
    def modify_prompt(self, prompt: str, context: AlignmentContext) -> str:
        """
        Modify the input prompt.
        
        Args:
            prompt: Original prompt
            context: Current alignment context
            
        Returns:
            Modified prompt
        """
        pass
    
    def apply(self, value: Any, context: AlignmentContext) -> Tuple[Any, Dict[str, Any]]:
        """Apply prompt modification."""
        if not isinstance(value, str):
            return value, {"error": "PromptModifier requires string input"}
        
        modified_prompt = self.modify_prompt(value, context)
        return modified_prompt, {
            "was_modified": modified_prompt != value,
            "length_change": len(modified_prompt) - len(value)
        }


class SafetyChecker(InterventionStrategy):
    """Base class for safety checking strategies."""
    
    @property
    def intervention_type(self) -> InterventionType:
        return InterventionType.SAFETY_CHECK
    
    @abstractmethod
    def check_safety(self, content: str, context: AlignmentContext) -> Tuple[bool, str]:
        """
        Check if content is safe.
        
        Args:
            content: Content to check
            context: Current alignment context
            
        Returns:
            Tuple of (is_safe, reason_if_unsafe)
        """
        pass
    
    def should_intervene(self, value: Any, context: AlignmentContext) -> bool:
        """Check if content is unsafe and needs intervention."""
        if not isinstance(value, str):
            return False
        
        is_safe, _ = self.check_safety(value, context)
        return not is_safe
    
    def apply(self, value: Any, context: AlignmentContext) -> Tuple[Any, Dict[str, Any]]:
        """Apply safety check and potentially block content."""
        if not isinstance(value, str):
            return value, {"error": "SafetyChecker requires string input"}
        
        is_safe, reason = self.check_safety(value, context)
        
        if not is_safe:
            # Return a safe replacement message
            safe_message = f"[Content blocked by {self.name}: {reason}]"
            return safe_message, {
                "blocked": True,
                "reason": reason,
                "original_length": len(value)
            }
        
        return value, {
            "blocked": False,
            "passed_check": self.name
        }


class InterventionPipeline:
    """Pipeline for applying multiple interventions in sequence."""
    
    def __init__(self, strategies: List[InterventionStrategy] = None):
        """Initialize pipeline with strategies."""
        self.strategies = strategies or []
        self._sort_strategies()
    
    def add_strategy(self, strategy: InterventionStrategy):
        """Add a strategy to the pipeline."""
        self.strategies.append(strategy)
        self._sort_strategies()
    
    def remove_strategy(self, name: str):
        """Remove a strategy by name."""
        self.strategies = [s for s in self.strategies if s.name != name]
    
    def _sort_strategies(self):
        """Sort strategies by priority (descending)."""
        self.strategies.sort(key=lambda s: s.priority, reverse=True)
    
    def apply(self, value: Any, context: AlignmentContext) -> List[InterventionResult]:
        """
        Apply all relevant interventions.
        
        Args:
            value: Value to process
            context: Alignment context
            
        Returns:
            List of intervention results
        """
        results = []
        current_value = value
        
        for strategy in self.strategies:
            if not strategy.enabled:
                continue
                
            if strategy.should_intervene(current_value, context):
                try:
                    modified_value, metadata = strategy.apply(current_value, context)
                    
                    result = InterventionResult(
                        intervention_type=strategy.intervention_type,
                        original_value=current_value,
                        modified_value=modified_value,
                        metadata=metadata,
                        applied=True
                    )
                    
                    results.append(result)
                    current_value = modified_value
                    
                except Exception as e:
                    result = InterventionResult(
                        intervention_type=strategy.intervention_type,
                        original_value=current_value,
                        modified_value=current_value,
                        metadata={"error": str(e)},
                        applied=False,
                        reason=f"Error: {str(e)}"
                    )
                    results.append(result)
        
        return results
    
    def get_final_value(self, value: Any, context: AlignmentContext) -> Any:
        """Get the final value after all interventions."""
        results = self.apply(value, context)
        if results:
            return results[-1].modified_value
        return value