"""Runtime intervention framework for prompt modification and output filtering."""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

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

logger = logging.getLogger(__name__)


@dataclass
class InterventionConfig:
    """Configuration for runtime interventions."""

    # General settings
    enabled: bool = True
    log_interventions: bool = True
    strict_mode: bool = False  # Fail on any intervention error

    # Prompt modification settings
    max_prompt_modifications: int = 3
    preserve_original_intent: bool = True

    # Output filtering settings
    max_filter_iterations: int = 5
    filter_confidence_threshold: float = 0.8

    # Response steering settings
    steering_strength: float = 0.7  # 0.0 to 1.0
    allow_complete_rewrite: bool = False

    # Safety settings
    safety_threshold: float = 0.9
    block_on_safety_failure: bool = True


class PromptRewriter(PromptModifier):
    """Rewrites prompts to improve alignment."""

    def __init__(self, rewrite_rules: Dict[str, str]):
        self.rewrite_rules = rewrite_rules

    def should_intervene(self, prompt: str, context: AlignmentContext) -> bool:
        """Check if prompt contains patterns that need rewriting."""
        for pattern in self.rewrite_rules:
            if re.search(pattern, prompt, re.IGNORECASE):
                return True
        return False

    def modify(self, prompt: str, context: AlignmentContext) -> InterventionResult:
        """Apply rewrite rules to the prompt."""
        modified_prompt = prompt
        modifications = []

        for pattern, replacement in self.rewrite_rules.items():
            if re.search(pattern, modified_prompt, re.IGNORECASE):
                modified_prompt = re.sub(pattern, replacement, modified_prompt, flags=re.IGNORECASE)
                modifications.append(f"Applied rule: {pattern} -> {replacement}")

        return InterventionResult(
            success=True,
            intervention_type=InterventionType.PROMPT_MODIFICATION,
            modified_value=modified_prompt,
            original_value=prompt,
            metadata={"modifications": modifications, "rule_count": len(modifications)},
        )


class ToxicityFilter(OutputFilter):
    """Filters toxic content from model outputs."""

    def __init__(self, toxic_patterns: List[str], replacement: str = "[FILTERED]"):
        self.toxic_patterns = [re.compile(p, re.IGNORECASE) for p in toxic_patterns]
        self.replacement = replacement

    def should_intervene(self, output: str, context: AlignmentContext) -> bool:
        """Check if output contains toxic patterns."""
        return any(pattern.search(output) for pattern in self.toxic_patterns)

    def filter(self, output: str, context: AlignmentContext) -> InterventionResult:
        """Filter toxic content from output."""
        filtered_output = output
        filtered_segments = []

        for pattern in self.toxic_patterns:
            matches = pattern.findall(filtered_output)
            if matches:
                filtered_output = pattern.sub(self.replacement, filtered_output)
                filtered_segments.extend(matches)

        return InterventionResult(
            success=True,
            intervention_type=InterventionType.OUTPUT_FILTERING,
            modified_value=filtered_output,
            original_value=output,
            metadata={
                "filtered_segments": filtered_segments,
                "filter_count": len(filtered_segments),
            },
        )


class ToneAdjuster(ResponseModifier):
    """Adjusts the tone of model responses."""

    def __init__(self, target_tone: str, tone_markers: Dict[str, List[str]]):
        self.target_tone = target_tone
        self.tone_markers = tone_markers

    def should_intervene(self, response: str, context: AlignmentContext) -> bool:
        """Check if response tone needs adjustment."""
        current_tone = self._detect_tone(response)
        return current_tone != self.target_tone

    def _detect_tone(self, response: str) -> str:
        """Detect the current tone of the response."""
        tone_scores = {}

        for tone, markers in self.tone_markers.items():
            score = sum(1 for marker in markers if marker.lower() in response.lower())
            tone_scores[tone] = score

        if tone_scores:
            return max(tone_scores, key=tone_scores.get)
        return "neutral"

    def steer(self, response: str, context: AlignmentContext) -> InterventionResult:
        """Steer response towards target tone."""
        current_tone = self._detect_tone(response)

        # Simple tone adjustment by adding tone markers
        if self.target_tone in self.tone_markers:
            tone_prefix = f"[{self.target_tone.upper()} TONE]: "
            adjusted_response = tone_prefix + response
        else:
            adjusted_response = response

        return InterventionResult(
            success=True,
            intervention_type=InterventionType.RESPONSE_STEERING,
            modified_value=adjusted_response,
            original_value=response,
            metadata={"original_tone": current_tone, "target_tone": self.target_tone},
        )


class ContentSafetyChecker(SafetyChecker):
    """Checks content for safety violations."""

    def __init__(self, safety_rules: Dict[str, Callable[[str], bool]]):
        self.safety_rules = safety_rules

    def should_intervene(self, content: str, context: AlignmentContext) -> bool:
        """Check if content violates any safety rules."""
        return any(rule(content) for rule in self.safety_rules.values())

    def check(self, content: str, context: AlignmentContext) -> InterventionResult:
        """Check content against safety rules."""
        violations = []

        for rule_name, rule_func in self.safety_rules.items():
            if rule_func(content):
                violations.append(rule_name)

        return InterventionResult(
            success=len(violations) == 0,
            intervention_type=InterventionType.SAFETY_CHECK,
            modified_value=content if not violations else "[BLOCKED FOR SAFETY]",
            original_value=content,
            metadata={"violations": violations, "passed": len(violations) == 0},
        )


class RuntimeInterventionSystem:
    """Main runtime intervention system."""

    def __init__(self, config: Optional[InterventionConfig] = None):
        self.config = config or InterventionConfig()
        self.pipeline = InterventionPipeline()
        self.intervention_history: List[Dict[str, Any]] = []

    def add_intervention(self, intervention: InterventionStrategy) -> None:
        """Add an intervention to the pipeline."""
        self.pipeline.add_intervention(intervention)

    def remove_intervention(self, intervention: InterventionStrategy) -> None:
        """Remove an intervention from the pipeline."""
        self.pipeline.remove_intervention(intervention)

    def intervene_on_prompt(self, prompt: str, context: Optional[AlignmentContext] = None) -> str:
        """Apply prompt interventions."""
        if not self.config.enabled:
            return prompt

        context = context or AlignmentContext()
        context.metadata["intervention_type"] = "prompt"

        # Get only prompt modifiers
        prompt_modifiers = [i for i in self.pipeline.interventions if isinstance(i, PromptModifier)]

        modified_prompt = prompt
        modification_count = 0

        for modifier in prompt_modifiers:
            if modification_count >= self.config.max_prompt_modifications:
                break

            if modifier.should_intervene(modified_prompt, context):
                result = modifier.modify(modified_prompt, context)
                if result.success:
                    modified_prompt = result.modified_value
                    modification_count += 1
                    self._log_intervention(result)
                elif self.config.strict_mode:
                    raise RuntimeError(f"Prompt modification failed: {result}")

        return modified_prompt

    def intervene_on_output(self, output: str, context: Optional[AlignmentContext] = None) -> str:
        """Apply output interventions."""
        if not self.config.enabled:
            return output

        context = context or AlignmentContext()
        context.metadata["intervention_type"] = "output"

        # Apply filters first
        filtered_output = self._apply_filters(output, context)

        # Then apply response modifiers
        modified_output = self._apply_modifiers(filtered_output, context)

        # Finally run safety checks
        safe_output = self._apply_safety_checks(modified_output, context)

        return safe_output

    def _apply_filters(self, output: str, context: AlignmentContext) -> str:
        """Apply output filters."""
        filters = [i for i in self.pipeline.interventions if isinstance(i, OutputFilter)]

        filtered_output = output
        iteration_count = 0

        while iteration_count < self.config.max_filter_iterations:
            changed = False

            for filter_obj in filters:
                if filter_obj.should_intervene(filtered_output, context):
                    result = filter_obj.filter(filtered_output, context)
                    if result.success:
                        filtered_output = result.modified_value
                        changed = True
                        self._log_intervention(result)
                    elif self.config.strict_mode:
                        raise RuntimeError(f"Output filtering failed: {result}")

            if not changed:
                break

            iteration_count += 1

        return filtered_output

    def _apply_modifiers(self, output: str, context: AlignmentContext) -> str:
        """Apply response modifiers."""
        modifiers = [i for i in self.pipeline.interventions if isinstance(i, ResponseModifier)]

        modified_output = output

        for modifier in modifiers:
            if modifier.should_intervene(modified_output, context):
                result = modifier.steer(modified_output, context)
                if result.success:
                    # Apply steering with configured strength
                    if self.config.allow_complete_rewrite:
                        modified_output = result.modified_value
                    else:
                        # Blend original and modified based on steering strength
                        modified_output = self._blend_responses(
                            modified_output, result.modified_value, self.config.steering_strength
                        )
                    self._log_intervention(result)
                elif self.config.strict_mode:
                    raise RuntimeError(f"Response modification failed: {result}")

        return modified_output

    def _apply_safety_checks(self, output: str, context: AlignmentContext) -> str:
        """Apply safety checks."""
        checkers = [i for i in self.pipeline.interventions if isinstance(i, SafetyChecker)]

        for checker in checkers:
            if checker.should_intervene(output, context):
                result = checker.check(output, context)
                self._log_intervention(result)

                if not result.success and self.config.block_on_safety_failure:
                    return result.modified_value  # Return blocked message

        return output

    def _blend_responses(self, original: str, modified: str, strength: float) -> str:
        """Blend original and modified responses based on strength."""
        if strength >= 1.0:
            return modified
        elif strength <= 0.0:
            return original
        else:
            # Simple blending: take more from modified based on strength
            # In practice, this would use more sophisticated blending
            return modified if strength > 0.5 else original

    def _log_intervention(self, result: InterventionResult) -> None:
        """Log intervention results."""
        if self.config.log_interventions:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": result.intervention_type.value,
                "success": result.success,
                "metadata": result.metadata,
            }
            self.intervention_history.append(log_entry)
            logger.info(f"Intervention applied: {json.dumps(log_entry)}")

    def get_intervention_stats(self) -> Dict[str | Any]:
        """Get statistics about interventions."""
        if not self.intervention_history:
            return {"total_interventions": 0}

        stats = {
            "total_interventions": len(self.intervention_history),
            "by_type": {},
            "success_rate": 0.0,
            "recent_interventions": self.intervention_history[-10:],
        }

        # Count by type
        for entry in self.intervention_history:
            intervention_type = entry["type"]
            stats["by_type"][intervention_type] = stats["by_type"].get(intervention_type, 0) + 1

        # Calculate success rate
        successful = sum(1 for e in self.intervention_history if e["success"])
        stats["success_rate"] = successful / len(self.intervention_history)

        return stats

    def clear_history(self) -> None:
        """Clear intervention history."""
        self.intervention_history.clear()


# Convenience functions
def create_default_system() -> RuntimeInterventionSystem:
    """Create a runtime intervention system with default settings."""
    system = RuntimeInterventionSystem()

    # Add basic toxic content filter
    system.add_intervention(
        ToxicityFilter(
            [r"\b(hate|kill|destroy)\s+\w+", r"harmful\s+content", r"dangerous\s+advice"]
        )
    )

    # Add basic prompt rewriter
    system.add_intervention(
        PromptRewriter(
            {
                r"how\s+to\s+hack": "how to ethically test security of",
                r"make\s+a\s+bomb": "understand chemistry safety",
                r"illegal\s+drugs": "pharmaceutical compounds",
            }
        )
    )

    # Add basic safety checker
    def contains_pii(content: str) -> bool:
        # Simple PII detection
        patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{16}\b",  # Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        ]
        return any(re.search(p, content) for p in patterns)

    system.add_intervention(ContentSafetyChecker({"pii_detected": contains_pii}))

    return system
