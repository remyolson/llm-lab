"""Constitutional AI engine for applying rule-based alignment."""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .rules import ConstitutionalRule, RulePriority, RuleSet, RuleType, create_default_ruleset
from .yaml_parser import RuleYAMLParser

logger = logging.getLogger(__name__)


@dataclass
class RuleEvaluationResult:
    """Result of evaluating a constitutional rule."""

    rule_id: str
    rule_name: str
    rule_type: RuleType
    triggered: bool
    actions: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConstitutionalContext:
    """Context for constitutional AI evaluation."""

    prompt: str
    response: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str | Any]:
        """Convert context to dictionary for rule evaluation."""
        return {
            "prompt": self.prompt,
            "response": self.response or "",
            "user_id": self.user_id,
            "session_id": self.session_id,
            **self.metadata,
        }


class ConstitutionalAIEngine:
    """Engine for applying constitutional AI rules."""

    def __init__(
        self, ruleset: Optional[RuleSet] = None, yaml_directory: Optional[str | Path] = None
    ):
        """
        Initialize the engine.

        Args:
            ruleset: Initial ruleset to use
            yaml_directory: Directory containing YAML rule definitions
        """
        self.rulesets: Dict[str, RuleSet] = {}
        self.parser = RuleYAMLParser()
        self.evaluation_history: List[RuleEvaluationResult] = []

        # Load default ruleset if none provided
        if ruleset:
            self.add_ruleset(ruleset)
        else:
            default_ruleset = create_default_ruleset()
            self.add_ruleset(default_ruleset)

        # Load rules from YAML directory if provided
        if yaml_directory:
            self.load_yaml_rules(yaml_directory)

    def add_ruleset(self, ruleset: RuleSet) -> None:
        """Add a ruleset to the engine."""
        self.rulesets[ruleset.id] = ruleset
        logger.info(f"Added ruleset: {ruleset.name} ({len(ruleset.rules)} rules)")

    def remove_ruleset(self, ruleset_id: str) -> bool:
        """Remove a ruleset from the engine."""
        if ruleset_id in self.rulesets:
            del self.rulesets[ruleset_id]
            logger.info(f"Removed ruleset: {ruleset_id}")
            return True
        return False

    def load_yaml_rules(self, directory: str | Path) -> None:
        """Load rules from YAML files in a directory."""
        loaded_rulesets = self.parser.load_directory(directory)
        for ruleset_id, ruleset in loaded_rulesets.items():
            self.add_ruleset(ruleset)

    def evaluate_prompt(
        self,
        prompt: str,
        context: Optional[ConstitutionalContext] = None,
        ruleset_ids: Optional[List[str]] = None,
    ) -> Dict[str | Any]:
        """
        Evaluate a prompt against constitutional rules.

        Args:
            prompt: The prompt to evaluate
            context: Additional context for evaluation
            ruleset_ids: Specific rulesets to use (None = use all)

        Returns:
            Dictionary containing evaluation results and actions
        """
        if context is None:
            context = ConstitutionalContext(prompt=prompt)
        else:
            context.prompt = prompt

        return self._evaluate(context, ruleset_ids, evaluation_type="prompt")

    def evaluate_response(
        self,
        response: str,
        prompt: str,
        context: Optional[ConstitutionalContext] = None,
        ruleset_ids: Optional[List[str]] = None,
    ) -> Dict[str | Any]:
        """
        Evaluate a response against constitutional rules.

        Args:
            response: The response to evaluate
            prompt: The original prompt
            context: Additional context for evaluation
            ruleset_ids: Specific rulesets to use (None = use all)

        Returns:
            Dictionary containing evaluation results and actions
        """
        if context is None:
            context = ConstitutionalContext(prompt=prompt, response=response)
        else:
            context.prompt = prompt
            context.response = response

        return self._evaluate(context, ruleset_ids, evaluation_type="response")

    def _evaluate(
        self, context: ConstitutionalContext, ruleset_ids: Optional[List[str]], evaluation_type: str
    ) -> Dict[str | Any]:
        """Internal evaluation method."""
        # Determine which rulesets to use
        if ruleset_ids:
            rulesets_to_use = [self.rulesets[rid] for rid in ruleset_ids if rid in self.rulesets]
        else:
            rulesets_to_use = list(self.rulesets.values())

        # Collect all evaluation results
        all_results = {
            "evaluation_type": evaluation_type,
            "timestamp": datetime.now().isoformat(),
            "context": {
                "prompt": context.prompt[:100] + "..."
                if len(context.prompt) > 100
                else context.prompt,
                "has_response": context.response is not None,
            },
            "rulesets_evaluated": [rs.id for rs in rulesets_to_use],
            "triggered_rules": [],
            "actions": [],
            "violations": [],
            "modifications": [],
        }

        # Evaluate each ruleset
        for ruleset in rulesets_to_use:
            results = ruleset.evaluate(context.to_dict())

            # Process triggered rules
            for rule in results["triggered"]:
                eval_result = RuleEvaluationResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    rule_type=rule.type,
                    triggered=True,
                    actions=rule.get_actions(),
                )

                self.evaluation_history.append(eval_result)
                all_results["triggered_rules"].append(
                    {
                        "rule_id": rule.id,
                        "rule_name": rule.name,
                        "rule_type": rule.type.value,
                        "priority": rule.priority.value,
                    }
                )

                # Process actions
                for action in rule.get_actions():
                    processed_action = self._process_action(action, rule, context, all_results)
                    if processed_action:
                        all_results["actions"].append(processed_action)

            # Check for violations in strict mode
            if ruleset.strict_mode and results["failed"]:
                for rule in results["failed"]:
                    all_results["violations"].append(
                        {
                            "rule_id": rule.id,
                            "rule_name": rule.name,
                            "severity": rule.priority.value,
                        }
                    )

        # Determine overall verdict
        all_results["verdict"] = self._determine_verdict(all_results)

        return all_results

    def _process_action(
        self,
        action: Dict[str, Any],
        rule: ConstitutionalRule,
        context: ConstitutionalContext,
        results: Dict[str, Any],
    ) -> Optional[Dict[str | Any]]:
        """Process a single action from a rule."""
        action_type = action.get("type")

        if action_type == "block":
            return {
                "type": "block",
                "rule_id": rule.id,
                "message": action.get("message", "Content blocked by constitutional rules"),
                "severity": rule.priority.value,
            }

        elif action_type == "modify":
            operation = action.get("operation")
            modification = {"type": "modify", "rule_id": rule.id, "operation": operation}

            if operation == "replace":
                modification["replacement"] = action.get("replacement", "")
            elif operation == "add_prefix":
                modification["prefix"] = action.get("prefix", "")
            elif operation == "add_suffix":
                modification["suffix"] = action.get("suffix", "")
            elif operation == "add_qualifier":
                modification["qualifier"] = action.get("qualifier", "")

            results["modifications"].append(modification)
            return modification

        elif action_type == "redact":
            return {
                "type": "redact",
                "rule_id": rule.id,
                "pattern": action.get("pattern"),
                "replacement": action.get("replacement", "[REDACTED]"),
            }

        elif action_type == "log":
            logger.log(
                getattr(logging, action.get("level", "INFO").upper()),
                f"Constitutional rule {rule.id}: {action.get('message', 'Rule triggered')}",
            )
            return None

        elif action_type == "suggest_alternative":
            return {
                "type": "suggest_alternative",
                "rule_id": rule.id,
                "suggestion": action.get("suggestion", ""),
            }

        else:
            return {
                "type": action_type,
                "rule_id": rule.id,
                **{k: v for k, v in action.items() if k != "type"},
            }

    def _determine_verdict(self, results: Dict[str, Any]) -> str:
        """Determine overall verdict from evaluation results."""
        # Check for blocks
        for action in results["actions"]:
            if action.get("type") == "block":
                return "blocked"

        # Check for violations
        if results["violations"]:
            critical_violations = [v for v in results["violations"] if v["severity"] == "critical"]
            if critical_violations:
                return "rejected"

        # Check for modifications
        if results["modifications"]:
            return "modified"

        # Otherwise approved
        return "approved"

    def apply_modifications(self, text: str, modifications: List[Dict[str, Any]]) -> str:
        """Apply modifications to text based on rule actions."""
        modified_text = text

        for mod in modifications:
            operation = mod.get("operation")

            if operation == "replace":
                modified_text = mod.get("replacement", "")
            elif operation == "add_prefix":
                modified_text = mod.get("prefix", "") + modified_text
            elif operation == "add_suffix":
                modified_text = modified_text + mod.get("suffix", "")
            elif operation == "add_qualifier":
                # Add qualifier at the beginning
                qualifier = mod.get("qualifier", "")
                modified_text = f"{qualifier}, {modified_text.lower()}"
            elif operation == "add_nuance":
                template = mod.get("template", "{}")
                modified_text = template.format(modified_text)

        return modified_text

    def get_rule_stats(self) -> Dict[str | Any]:
        """Get statistics about rule evaluations."""
        if not self.evaluation_history:
            return {
                "total_evaluations": 0,
                "rules_loaded": sum(len(rs.rules) for rs in self.rulesets.values()),
            }

        stats = {
            "total_evaluations": len(self.evaluation_history),
            "rules_loaded": sum(len(rs.rules) for rs in self.rulesets.values()),
            "triggered_by_type": {},
            "triggered_by_priority": {},
            "most_triggered": {},
        }

        # Count by type and priority
        rule_trigger_counts = {}

        for eval_result in self.evaluation_history:
            if eval_result.triggered:
                # Count by type
                rule_type = eval_result.rule_type.value
                stats["triggered_by_type"][rule_type] = (
                    stats["triggered_by_type"].get(rule_type, 0) + 1
                )

                # Count individual rules
                rule_id = eval_result.rule_id
                rule_trigger_counts[rule_id] = rule_trigger_counts.get(rule_id, 0) + 1

        # Find most triggered rules
        if rule_trigger_counts:
            sorted_rules = sorted(rule_trigger_counts.items(), key=lambda x: x[1], reverse=True)
            stats["most_triggered"] = dict(sorted_rules[:5])

        return stats

    def export_rules(self, output_directory: str | Path) -> None:
        """Export all rules to YAML files."""
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        for ruleset_id, ruleset in self.rulesets.items():
            output_path = output_dir / f"{ruleset_id}.yaml"
            self.parser.save_ruleset(ruleset, output_path)
            logger.info(f"Exported ruleset {ruleset_id} to {output_path}")

    def clear_history(self) -> None:
        """Clear evaluation history."""
        self.evaluation_history.clear()


# Convenience functions
def create_safety_focused_engine() -> ConstitutionalAIEngine:
    """Create an engine focused on safety."""
    from .rules import RuleTemplates

    engine = ConstitutionalAIEngine()

    # Create safety-focused ruleset
    safety_ruleset = RuleSet(
        id="safety_focus",
        name="Safety-Focused Rules",
        description="Strict safety rules for content generation",
        strict_mode=True,
        priority_threshold=RulePriority.MEDIUM,
    )

    # Add safety rules
    safety_ruleset.add_rule(RuleTemplates.create_harm_prevention_rule())
    safety_ruleset.add_rule(RuleTemplates.create_privacy_rule())

    engine.add_ruleset(safety_ruleset)
    return engine


def create_educational_engine() -> ConstitutionalAIEngine:
    """Create an engine for educational contexts."""
    from .rules import RuleTemplates

    engine = ConstitutionalAIEngine()

    # Create educational ruleset
    edu_ruleset = RuleSet(
        id="educational",
        name="Educational Context Rules",
        description="Rules optimized for educational applications",
        strict_mode=False,
    )

    # Add educational rules
    edu_ruleset.add_rule(RuleTemplates.create_truthfulness_rule())
    edu_ruleset.add_rule(RuleTemplates.create_helpfulness_rule())
    edu_ruleset.add_rule(RuleTemplates.create_bias_mitigation_rule())

    engine.add_ruleset(edu_ruleset)
    return engine
