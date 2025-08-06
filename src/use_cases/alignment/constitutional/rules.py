"""Constitutional AI rule definitions and structures."""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class RuleType(Enum):
    """Types of constitutional rules."""

    PRINCIPLE = "principle"  # High-level ethical principle
    CONSTRAINT = "constraint"  # Hard constraint that must not be violated
    PREFERENCE = "preference"  # Soft preference to guide behavior
    OBJECTIVE = "objective"  # Goal to optimize for
    PROHIBITION = "prohibition"  # Explicit prohibition


class RulePriority(Enum):
    """Priority levels for rules."""

    CRITICAL = "critical"  # Must never be violated
    HIGH = "high"  # Very important to follow
    MEDIUM = "medium"  # Should generally be followed
    LOW = "low"  # Nice to have but can be overridden


@dataclass
class ConstitutionalRule:
    """A single constitutional rule."""

    # Core attributes
    id: str
    name: str
    description: str
    type: RuleType
    priority: RulePriority

    # Rule conditions
    conditions: List[Dict[str, Any]] = field(default_factory=list)

    # Actions to take
    actions: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)

    # Versioning
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Rule state
    enabled: bool = True
    test_mode: bool = False

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate if rule conditions are met."""
        if not self.enabled:
            return False

        for condition in self.conditions:
            if not self._evaluate_condition(condition, context):
                return False

        return True

    def _evaluate_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate a single condition."""
        condition_type = condition.get("type")

        if condition_type == "contains":
            return self._check_contains(condition, context)
        elif condition_type == "matches":
            return self._check_matches(condition, context)
        elif condition_type == "custom":
            return self._check_custom(condition, context)
        else:
            return True

    def _check_contains(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if context contains specified values."""
        field = condition.get("field", "")
        values = condition.get("values", [])
        value = context.get(field, "")

        if isinstance(value, str):
            return any(v.lower() in value.lower() for v in values)
        return False

    def _check_matches(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if context matches pattern."""
        field = condition.get("field", "")
        pattern = condition.get("pattern", "")
        value = context.get(field, "")

        if isinstance(value, str) and pattern:
            return bool(re.search(pattern, value, re.IGNORECASE))
        return False

    def _check_custom(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check custom condition function."""
        func = condition.get("function")
        if callable(func):
            return func(context)
        return True

    def get_actions(self) -> List[Dict[str | Any]]:
        """Get actions to take when rule is triggered."""
        return self.actions if self.enabled else []


@dataclass
class RuleSet:
    """A collection of constitutional rules."""

    id: str
    name: str
    description: str
    rules: List[ConstitutionalRule] = field(default_factory=list)

    # Metadata
    version: str = "1.0"
    author: Optional[str] = None
    organization: Optional[str] = None

    # Configuration
    strict_mode: bool = False  # All rules must pass
    priority_threshold: Optional[RulePriority] = None

    def add_rule(self, rule: ConstitutionalRule) -> None:
        """Add a rule to the set."""
        self.rules.append(rule)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID."""
        original_count = len(self.rules)
        self.rules = [r for r in self.rules if r.id != rule_id]
        return len(self.rules) < original_count

    def get_rule(self, rule_id: str) -> Optional[ConstitutionalRule]:
        """Get a rule by ID."""
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        return None

    def evaluate(self, context: Dict[str, Any]) -> Dict[str | Any]:
        """Evaluate all rules against context."""
        results = {"passed": [], "failed": [], "triggered": [], "actions": []}

        for rule in self.rules:
            # Check priority threshold
            if self.priority_threshold:
                if self._compare_priority(rule.priority, self.priority_threshold) < 0:
                    continue

            # Evaluate rule
            if rule.evaluate(context):
                results["triggered"].append(rule)
                results["actions"].extend(rule.get_actions())

                # In strict mode, a constraint violation is a failure
                if self.strict_mode and rule.type == RuleType.CONSTRAINT:
                    results["failed"].append(rule)
            else:
                results["passed"].append(rule)

        return results

    def _compare_priority(self, p1: RulePriority, p2: RulePriority) -> int:
        """Compare two priorities. Returns -1 if p1 < p2, 0 if equal, 1 if p1 > p2."""
        priority_order = [
            RulePriority.LOW,
            RulePriority.MEDIUM,
            RulePriority.HIGH,
            RulePriority.CRITICAL,
        ]

        idx1 = priority_order.index(p1)
        idx2 = priority_order.index(p2)

        if idx1 < idx2:
            return -1
        elif idx1 > idx2:
            return 1
        else:
            return 0


# Predefined rule templates
class RuleTemplates:
    """Common rule templates for constitutional AI."""

    @staticmethod
    def create_harm_prevention_rule() -> ConstitutionalRule:
        """Create a rule preventing harmful content."""
        return ConstitutionalRule(
            id="harm_prevention",
            name="Prevent Harmful Content",
            description="Prevent generation of content that could cause harm",
            type=RuleType.CONSTRAINT,
            priority=RulePriority.CRITICAL,
            conditions=[
                {
                    "type": "contains",
                    "field": "prompt",
                    "values": ["harm", "hurt", "damage", "injure", "kill"],
                }
            ],
            actions=[
                {"type": "block", "message": "This request could lead to harmful content"},
                {
                    "type": "suggest_alternative",
                    "suggestion": "Consider rephrasing your request in a constructive way",
                },
            ],
            category="safety",
            tags=["harm_prevention", "safety", "ethics"],
        )

    @staticmethod
    def create_truthfulness_rule() -> ConstitutionalRule:
        """Create a rule enforcing truthfulness."""
        return ConstitutionalRule(
            id="truthfulness",
            name="Enforce Truthfulness",
            description="Ensure responses are truthful and acknowledge uncertainty",
            type=RuleType.PRINCIPLE,
            priority=RulePriority.HIGH,
            conditions=[
                {
                    "type": "matches",
                    "field": "response",
                    "pattern": r"(definitely|absolutely|certainly|100%|guaranteed)",
                }
            ],
            actions=[
                {
                    "type": "modify",
                    "operation": "add_qualifier",
                    "qualifier": "Based on available information",
                }
            ],
            category="truthfulness",
            tags=["accuracy", "honesty", "uncertainty"],
        )

    @staticmethod
    def create_privacy_rule() -> ConstitutionalRule:
        """Create a rule protecting privacy."""
        return ConstitutionalRule(
            id="privacy_protection",
            name="Protect Privacy",
            description="Prevent sharing of personal information",
            type=RuleType.PROHIBITION,
            priority=RulePriority.CRITICAL,
            conditions=[
                {
                    "type": "matches",
                    "field": "response",
                    "pattern": r"(\b\d{3}-\d{2}-\d{4}\b|\b\d{16}\b|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
                }
            ],
            actions=[{"type": "redact", "pattern": "PII_PATTERN", "replacement": "[REDACTED]"}],
            category="privacy",
            tags=["privacy", "pii", "data_protection"],
        )

    @staticmethod
    def create_helpfulness_rule() -> ConstitutionalRule:
        """Create a rule promoting helpfulness."""
        return ConstitutionalRule(
            id="helpfulness",
            name="Be Helpful",
            description="Ensure responses are helpful and informative",
            type=RuleType.OBJECTIVE,
            priority=RulePriority.MEDIUM,
            conditions=[
                {"type": "custom", "function": lambda ctx: len(ctx.get("response", "")) < 50}
            ],
            actions=[{"type": "enhance", "operation": "expand_response", "min_length": 100}],
            category="quality",
            tags=["helpfulness", "quality", "user_experience"],
        )

    @staticmethod
    def create_bias_mitigation_rule() -> ConstitutionalRule:
        """Create a rule for bias mitigation."""
        return ConstitutionalRule(
            id="bias_mitigation",
            name="Mitigate Bias",
            description="Remove or reduce biased language",
            type=RuleType.PREFERENCE,
            priority=RulePriority.HIGH,
            conditions=[
                {
                    "type": "contains",
                    "field": "response",
                    "values": ["all women", "all men", "always", "never", "every"],
                }
            ],
            actions=[
                {
                    "type": "modify",
                    "operation": "add_nuance",
                    "template": "While this may be true in some cases, ",
                }
            ],
            category="fairness",
            tags=["bias", "fairness", "inclusion"],
        )


def create_default_ruleset() -> RuleSet:
    """Create a default constitutional ruleset."""
    ruleset = RuleSet(
        id="default",
        name="Default Constitutional Rules",
        description="Standard set of rules for safe and helpful AI behavior",
    )

    # Add all template rules
    ruleset.add_rule(RuleTemplates.create_harm_prevention_rule())
    ruleset.add_rule(RuleTemplates.create_truthfulness_rule())
    ruleset.add_rule(RuleTemplates.create_privacy_rule())
    ruleset.add_rule(RuleTemplates.create_helpfulness_rule())
    ruleset.add_rule(RuleTemplates.create_bias_mitigation_rule())

    return ruleset
