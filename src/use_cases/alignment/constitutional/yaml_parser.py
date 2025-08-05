"""YAML parser for constitutional AI rules."""

import yaml
import os
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime

from .rules import (
    ConstitutionalRule,
    RuleSet,
    RuleType,
    RulePriority
)


class RuleYAMLParser:
    """Parser for loading constitutional rules from YAML files."""
    
    def __init__(self):
        self.loaded_rulesets: Dict[str, RuleSet] = {}
        
    def load_rule(self, yaml_content: Union[str, Dict[str, Any]]) -> ConstitutionalRule:
        """Load a single rule from YAML content."""
        if isinstance(yaml_content, str):
            rule_data = yaml.safe_load(yaml_content)
        else:
            rule_data = yaml_content
            
        # Parse rule type and priority
        rule_type = RuleType(rule_data.get("type", "principle"))
        priority = RulePriority(rule_data.get("priority", "medium"))
        
        # Create rule
        rule = ConstitutionalRule(
            id=rule_data["id"],
            name=rule_data["name"],
            description=rule_data["description"],
            type=rule_type,
            priority=priority,
            conditions=rule_data.get("conditions", []),
            actions=rule_data.get("actions", []),
            category=rule_data.get("category"),
            tags=rule_data.get("tags", []),
            examples=rule_data.get("examples", []),
            version=rule_data.get("version", "1.0"),
            enabled=rule_data.get("enabled", True),
            test_mode=rule_data.get("test_mode", False)
        )
        
        # Handle timestamps
        if "created_at" in rule_data:
            rule.created_at = datetime.fromisoformat(rule_data["created_at"])
        if "updated_at" in rule_data:
            rule.updated_at = datetime.fromisoformat(rule_data["updated_at"])
            
        return rule
        
    def load_ruleset(self, yaml_path: Union[str, Path]) -> RuleSet:
        """Load a ruleset from a YAML file."""
        path = Path(yaml_path)
        
        with open(path, 'r') as f:
            ruleset_data = yaml.safe_load(f)
            
        # Create ruleset
        ruleset = RuleSet(
            id=ruleset_data["id"],
            name=ruleset_data["name"],
            description=ruleset_data["description"],
            version=ruleset_data.get("version", "1.0"),
            author=ruleset_data.get("author"),
            organization=ruleset_data.get("organization"),
            strict_mode=ruleset_data.get("strict_mode", False)
        )
        
        # Parse priority threshold
        if "priority_threshold" in ruleset_data:
            ruleset.priority_threshold = RulePriority(ruleset_data["priority_threshold"])
            
        # Load rules
        for rule_data in ruleset_data.get("rules", []):
            rule = self.load_rule(rule_data)
            ruleset.add_rule(rule)
            
        # Cache the loaded ruleset
        self.loaded_rulesets[ruleset.id] = ruleset
        
        return ruleset
        
    def load_directory(self, directory: Union[str, Path]) -> Dict[str, RuleSet]:
        """Load all rulesets from a directory."""
        directory = Path(directory)
        rulesets = {}
        
        # Load all YAML files
        for yaml_file in directory.glob("*.yaml"):
            try:
                ruleset = self.load_ruleset(yaml_file)
                rulesets[ruleset.id] = ruleset
            except Exception as e:
                print(f"Error loading {yaml_file}: {e}")
                
        # Also check for .yml extension
        for yaml_file in directory.glob("*.yml"):
            try:
                ruleset = self.load_ruleset(yaml_file)
                rulesets[ruleset.id] = ruleset
            except Exception as e:
                print(f"Error loading {yaml_file}: {e}")
                
        return rulesets
        
    def save_rule(self, rule: ConstitutionalRule, output_path: Union[str, Path]) -> None:
        """Save a rule to a YAML file."""
        rule_data = {
            "id": rule.id,
            "name": rule.name,
            "description": rule.description,
            "type": rule.type.value,
            "priority": rule.priority.value,
            "conditions": rule.conditions,
            "actions": rule.actions,
            "category": rule.category,
            "tags": rule.tags,
            "examples": rule.examples,
            "version": rule.version,
            "created_at": rule.created_at.isoformat(),
            "updated_at": rule.updated_at.isoformat(),
            "enabled": rule.enabled,
            "test_mode": rule.test_mode
        }
        
        path = Path(output_path)
        with open(path, 'w') as f:
            yaml.dump(rule_data, f, default_flow_style=False, sort_keys=False)
            
    def save_ruleset(self, ruleset: RuleSet, output_path: Union[str, Path]) -> None:
        """Save a ruleset to a YAML file."""
        ruleset_data = {
            "id": ruleset.id,
            "name": ruleset.name,
            "description": ruleset.description,
            "version": ruleset.version,
            "author": ruleset.author,
            "organization": ruleset.organization,
            "strict_mode": ruleset.strict_mode,
            "rules": []
        }
        
        if ruleset.priority_threshold:
            ruleset_data["priority_threshold"] = ruleset.priority_threshold.value
            
        # Add rules
        for rule in ruleset.rules:
            rule_data = {
                "id": rule.id,
                "name": rule.name,
                "description": rule.description,
                "type": rule.type.value,
                "priority": rule.priority.value,
                "conditions": rule.conditions,
                "actions": rule.actions,
                "category": rule.category,
                "tags": rule.tags,
                "examples": rule.examples,
                "version": rule.version,
                "enabled": rule.enabled,
                "test_mode": rule.test_mode
            }
            ruleset_data["rules"].append(rule_data)
            
        path = Path(output_path)
        with open(path, 'w') as f:
            yaml.dump(ruleset_data, f, default_flow_style=False, sort_keys=False)
            
    def validate_rule_yaml(self, yaml_content: str) -> Dict[str, Any]:
        """Validate YAML content for a rule."""
        try:
            rule_data = yaml.safe_load(yaml_content)
            
            # Check required fields
            required = ["id", "name", "description"]
            missing = [f for f in required if f not in rule_data]
            
            if missing:
                return {
                    "valid": False,
                    "errors": [f"Missing required fields: {missing}"]
                }
                
            # Validate rule type
            if "type" in rule_data:
                try:
                    RuleType(rule_data["type"])
                except ValueError:
                    return {
                        "valid": False,
                        "errors": [f"Invalid rule type: {rule_data['type']}"]
                    }
                    
            # Validate priority
            if "priority" in rule_data:
                try:
                    RulePriority(rule_data["priority"])
                except ValueError:
                    return {
                        "valid": False,
                        "errors": [f"Invalid priority: {rule_data['priority']}"]
                    }
                    
            return {"valid": True, "errors": []}
            
        except yaml.YAMLError as e:
            return {
                "valid": False,
                "errors": [f"YAML parsing error: {str(e)}"]
            }


def create_example_rule_yaml() -> str:
    """Create an example YAML rule definition."""
    return """# Example Constitutional AI Rule
id: example_safety_rule
name: Example Safety Rule
description: Demonstrates how to define a constitutional AI rule in YAML
type: constraint  # Options: principle, constraint, preference, objective, prohibition
priority: high    # Options: critical, high, medium, low

# Conditions that trigger this rule
conditions:
  - type: contains
    field: prompt
    values:
      - dangerous
      - harmful
      - illegal
      
  - type: matches
    field: response
    pattern: "\\b(weapon|explosive|drug)\\b"

# Actions to take when rule is triggered
actions:
  - type: block
    message: "This content violates safety guidelines"
    
  - type: modify
    operation: replace
    target: response
    replacement: "I cannot provide information on this topic for safety reasons."
    
  - type: log
    level: warning
    message: "Safety rule triggered"

# Metadata
category: safety
tags:
  - content_moderation
  - safety
  - harm_prevention

# Examples of when this rule applies
examples:
  - input: "How to make dangerous items"
    expected_action: "block"
    explanation: "Requests for dangerous items should be blocked"
    
  - input: "Tell me about historical weapons"
    expected_action: "modify"
    explanation: "Educational content can be modified to be safe"

# Configuration
version: "1.0"
enabled: true
test_mode: false
"""


def create_example_ruleset_yaml() -> str:
    """Create an example YAML ruleset definition."""
    return """# Example Constitutional AI Ruleset
id: example_ruleset
name: Example Safety and Ethics Ruleset
description: A comprehensive ruleset for safe and ethical AI behavior
version: "1.0"
author: AI Safety Team
organization: Example Corp

# Ruleset configuration
strict_mode: false  # If true, all rules must pass
priority_threshold: medium  # Only evaluate rules at or above this priority

# Rules in this ruleset
rules:
  - id: harm_prevention
    name: Prevent Physical Harm
    description: Block content that could lead to physical harm
    type: constraint
    priority: critical
    conditions:
      - type: contains
        field: prompt
        values: ["hurt", "harm", "injure", "kill"]
    actions:
      - type: block
        message: "I cannot provide information that could lead to harm"
        
  - id: truthfulness_check
    name: Ensure Truthfulness
    description: Add uncertainty qualifiers to overconfident statements
    type: principle
    priority: high
    conditions:
      - type: matches
        field: response
        pattern: "(definitely|absolutely|100%|guaranteed)"
    actions:
      - type: modify
        operation: add_prefix
        prefix: "Based on my training data, "
        
  - id: privacy_protection
    name: Protect Personal Information
    description: Redact personal information from responses
    type: prohibition
    priority: critical
    conditions:
      - type: matches
        field: response
        pattern: "\\b\\d{3}-\\d{2}-\\d{4}\\b|\\b\\d{16}\\b"
    actions:
      - type: redact
        pattern: "PII_PATTERN"
        replacement: "[REDACTED]"
        
  - id: helpful_responses
    name: Ensure Helpful Responses
    description: Enhance brief responses to be more helpful
    type: objective
    priority: medium
    conditions:
      - type: custom
        function: response_too_short
        min_length: 50
    actions:
      - type: enhance
        operation: expand_response
        template: "{response}\\n\\nWould you like more information?"
        
  - id: bias_reduction
    name: Reduce Biased Language
    description: Modify statements that make sweeping generalizations
    type: preference
    priority: high
    conditions:
      - type: contains
        field: response
        values: ["all", "every", "never", "always"]
    actions:
      - type: modify
        operation: add_nuance
        template: "While this may often be the case, "
"""