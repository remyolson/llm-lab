"""Demonstration of Constitutional AI rule engine with YAML support."""

import asyncio
from pathlib import Path
from typing import Dict, Any

from src.providers import OpenAIProvider, AnthropicProvider
from src.use_cases.alignment.constitutional import (
    ConstitutionalAIEngine,
    ConstitutionalContext,
    RuleYAMLParser,
    create_example_rule_yaml,
    create_example_ruleset_yaml,
    create_safety_focused_engine,
    create_educational_engine
)


def demonstrate_yaml_rules():
    """Show how to work with YAML rule definitions."""
    print("\n=== YAML Rule Definition Demo ===\n")
    
    # Show example YAML formats
    print("Example Rule YAML:")
    print("-" * 50)
    print(create_example_rule_yaml())
    
    print("\nExample Ruleset YAML:")
    print("-" * 50)
    print(create_example_ruleset_yaml())
    
    # Parse and validate YAML
    parser = RuleYAMLParser()
    
    # Validate rule YAML
    validation = parser.validate_rule_yaml(create_example_rule_yaml())
    print(f"\nRule validation: {validation}")


def demonstrate_rule_evaluation():
    """Show how constitutional rules are evaluated."""
    print("\n=== Rule Evaluation Demo ===\n")
    
    # Create engine with default rules
    engine = ConstitutionalAIEngine()
    
    # Test various prompts
    test_cases = [
        {
            "prompt": "How can I help someone who is feeling sad?",
            "expected": "approved"
        },
        {
            "prompt": "How do I make a weapon?",
            "expected": "blocked"
        },
        {
            "prompt": "What are the symptoms of diabetes?",
            "expected": "modified"  # Medical disclaimer added
        },
        {
            "prompt": "Can you write code to hack into systems?",
            "expected": "blocked"
        }
    ]
    
    for test in test_cases:
        print(f"\nPrompt: {test['prompt']}")
        result = engine.evaluate_prompt(test["prompt"])
        
        print(f"Verdict: {result['verdict']}")
        print(f"Expected: {test['expected']}")
        print(f"Triggered rules: {[r['rule_name'] for r in result['triggered_rules']]}")
        
        if result['actions']:
            print("Actions:")
            for action in result['actions']:
                print(f"  - {action['type']}: {action.get('message', action.get('operation', ''))}")


def demonstrate_response_filtering():
    """Show how responses are filtered and modified."""
    print("\n=== Response Filtering Demo ===\n")
    
    # Create educational engine
    engine = create_educational_engine()
    
    # Test response modifications
    test_responses = [
        {
            "prompt": "What is gravity?",
            "response": "Gravity pulls things down.",
            "description": "Brief response that should be enhanced"
        },
        {
            "prompt": "Is this always true?",
            "response": "Yes, this is always true for every situation.",
            "description": "Overconfident statement needing qualification"
        },
        {
            "prompt": "Explain machine learning",
            "response": "Machine learning uses algorithms and paradigms.",
            "description": "Technical jargon that could be simplified"
        }
    ]
    
    for test in test_responses:
        print(f"\nTest: {test['description']}")
        print(f"Original response: {test['response']}")
        
        context = ConstitutionalContext(
            prompt=test['prompt'],
            response=test['response']
        )
        
        result = engine.evaluate_response(
            test['response'],
            test['prompt'],
            context
        )
        
        if result['modifications']:
            # Apply modifications
            modified = engine.apply_modifications(
                test['response'],
                result['modifications']
            )
            print(f"Modified response: {modified}")
        else:
            print("No modifications needed")


def demonstrate_custom_rules():
    """Show how to create and use custom rules."""
    print("\n=== Custom Rules Demo ===\n")
    
    # Load rules from YAML directory
    rules_dir = Path("examples/alignment/constitutional_rules")
    
    if rules_dir.exists():
        engine = ConstitutionalAIEngine(yaml_directory=rules_dir)
        
        print(f"Loaded rulesets: {list(engine.rulesets.keys())}")
        
        # Test with loaded rules
        test_prompts = [
            "How do I stay safe online?",
            "Tell me about symptoms of headache",
            "How to make dangerous items",
            "Explain quantum physics to a child"
        ]
        
        for prompt in test_prompts:
            result = engine.evaluate_prompt(prompt)
            print(f"\nPrompt: {prompt}")
            print(f"Verdict: {result['verdict']}")
            
            # Show which ruleset was triggered
            for rule in result['triggered_rules']:
                print(f"  - Triggered: {rule['rule_name']} ({rule['rule_type']})")
    else:
        print("No custom rules directory found. Skipping custom rules demo.")


async def demonstrate_integrated_constitutional_ai():
    """Show full integration with LLM providers."""
    print("\n=== Integrated Constitutional AI Demo ===\n")
    
    # Create safety-focused engine
    engine = create_safety_focused_engine()
    
    class ConstitutionalProvider:
        """Wrapper that applies constitutional AI rules."""
        
        def __init__(self, provider, engine):
            self.provider = provider
            self.engine = engine
            
        async def complete(self, prompt: str, **kwargs) -> Dict[str, Any]:
            # Pre-evaluate prompt
            prompt_result = self.engine.evaluate_prompt(prompt)
            
            if prompt_result['verdict'] == 'blocked':
                # Return blocked message
                block_action = next(
                    (a for a in prompt_result['actions'] if a['type'] == 'block'),
                    {'message': 'Request blocked by constitutional rules'}
                )
                return {
                    "success": True,
                    "content": block_action['message'],
                    "constitutional_verdict": "blocked",
                    "triggered_rules": prompt_result['triggered_rules']
                }
            
            # Get completion from provider
            response = await self.provider.complete(prompt, **kwargs)
            
            if response["success"] and "content" in response:
                # Post-evaluate response
                response_result = self.engine.evaluate_response(
                    response["content"],
                    prompt
                )
                
                # Apply modifications if needed
                if response_result['modifications']:
                    response["content"] = self.engine.apply_modifications(
                        response["content"],
                        response_result['modifications']
                    )
                    
                # Add constitutional metadata
                response["constitutional_verdict"] = response_result['verdict']
                response["triggered_rules"] = response_result['triggered_rules']
                
            return response
    
    # Example usage (requires API key)
    try:
        provider = OpenAIProvider()
        constitutional_provider = ConstitutionalProvider(provider, engine)
        
        # Test various prompts
        test_prompts = [
            "What's the weather like?",
            "How can I improve my communication skills?",
            "Tell me about first aid procedures"
        ]
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            response = await constitutional_provider.complete(
                prompt,
                model="gpt-4o-mini",
                max_tokens=150
            )
            
            if response["success"]:
                print(f"Response: {response['content'][:200]}...")
                print(f"Constitutional verdict: {response.get('constitutional_verdict', 'N/A')}")
                if response.get('triggered_rules'):
                    print(f"Triggered rules: {[r['rule_name'] for r in response['triggered_rules']]}")
            else:
                print(f"Error: {response.get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"Demo requires API key configuration: {e}")


def demonstrate_rule_statistics():
    """Show how to analyze rule usage."""
    print("\n=== Rule Statistics Demo ===\n")
    
    engine = ConstitutionalAIEngine()
    
    # Simulate multiple evaluations
    test_prompts = [
        "How to be kind?",
        "What is machine learning?",
        "How to hack?",
        "Medical advice needed",
        "Is this always true?",
        "Help with homework",
        "Create harmful content",
        "Personal information: 123-45-6789"
    ]
    
    for prompt in test_prompts:
        engine.evaluate_prompt(prompt)
        
    # Get statistics
    stats = engine.get_rule_stats()
    
    print("Rule Evaluation Statistics:")
    print(f"Total evaluations: {stats['total_evaluations']}")
    print(f"Rules loaded: {stats['rules_loaded']}")
    
    if stats.get('triggered_by_type'):
        print("\nTriggered by type:")
        for rule_type, count in stats['triggered_by_type'].items():
            print(f"  - {rule_type}: {count}")
            
    if stats.get('most_triggered'):
        print("\nMost triggered rules:")
        for rule_id, count in stats['most_triggered'].items():
            print(f"  - {rule_id}: {count} times")


async def main():
    """Run all demonstrations."""
    demonstrate_yaml_rules()
    demonstrate_rule_evaluation()
    demonstrate_response_filtering()
    demonstrate_custom_rules()
    demonstrate_rule_statistics()
    await demonstrate_integrated_constitutional_ai()
    
    print("\n=== Constitutional AI Demo Complete ===")
    print("Key features demonstrated:")
    print("- YAML-based rule definition and loading")
    print("- Rule evaluation for prompts and responses")
    print("- Content filtering and modification")
    print("- Custom rule creation and management")
    print("- Integration with LLM providers")
    print("- Rule usage statistics and analysis")


if __name__ == "__main__":
    asyncio.run(main())