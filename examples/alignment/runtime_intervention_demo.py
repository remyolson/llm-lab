"""Demonstration of runtime intervention framework for LLM alignment."""

import asyncio
from typing import Any, Dict

from src.providers import OpenAIProvider
from use_cases.alignment.runtime import (
    AlignmentContext,
    InstructionClarifier,
    InterventionConfig,
    RuntimeInterventionSystem,
    get_content_moderation_strategy,
    get_educational_strategy,
)


async def demonstrate_prompt_intervention():
    """Show how prompts can be modified before sending to LLM."""
    print("\n=== Prompt Intervention Demo ===\n")

    # Create intervention system
    system = RuntimeInterventionSystem()

    # Add instruction clarifier
    system.add_intervention(InstructionClarifier())

    # Test prompts that need clarification
    test_prompts = [
        "How do I hack into a computer system?",
        "Can you help me break into my own locked account?",
        "I need to destroy all the files on my old computer",
        "Show me how to attack a web server",
    ]

    for prompt in test_prompts:
        print(f"Original: {prompt}")
        modified = system.intervene_on_prompt(prompt)
        print(f"Modified: {modified}")
        print("-" * 50)


async def demonstrate_output_filtering():
    """Show how outputs can be filtered for bias and toxicity."""
    print("\n=== Output Filtering Demo ===\n")

    # Create intervention system with content moderation
    system = RuntimeInterventionSystem()

    # Add all content moderation strategies
    for strategy in get_content_moderation_strategy():
        system.add_intervention(strategy)

    # Test outputs that need filtering
    test_outputs = [
        "The chairman should hire a waitress for the event.",
        "This is definitely a proven fact that cannot be questioned.",
        "The elderly person was too slow to understand modern technology.",
        "You should mislead people to get what you want in business.",
    ]

    for output in test_outputs:
        print(f"Original: {output}")
        filtered = system.intervene_on_output(output)
        print(f"Filtered: {filtered}")
        print("-" * 50)


async def demonstrate_integrated_system():
    """Show full integration with actual LLM providers."""
    print("\n=== Integrated System Demo ===\n")

    # Create intervention system
    intervention_system = RuntimeInterventionSystem(
        InterventionConfig(
            log_interventions=True, preserve_original_intent=True, steering_strength=0.8
        )
    )

    # Add educational strategies
    for strategy in get_educational_strategy():
        intervention_system.add_intervention(strategy)

    # Create a wrapped provider
    class IntervenedProvider:
        def __init__(self, provider, intervention_system):
            self.provider = provider
            self.intervention_system = intervention_system

        async def complete(self, prompt: str, **kwargs) -> Dict[str, Any]:
            # Apply prompt interventions
            context = AlignmentContext(
                user_id="demo_user",
                session_id="demo_session",
                metadata={"provider": self.provider.__class__.__name__},
            )

            modified_prompt = self.intervention_system.intervene_on_prompt(prompt, context)

            # Get completion from provider
            response = await self.provider.complete(modified_prompt, **kwargs)

            # Apply output interventions
            if response["success"] and "content" in response:
                modified_content = self.intervention_system.intervene_on_output(
                    response["content"], context
                )
                response["content"] = modified_content

            # Add intervention stats to response
            response["intervention_stats"] = intervention_system.get_intervention_stats()

            return response

    # Example usage (requires API key)
    try:
        provider = OpenAIProvider()
        intervened_provider = IntervenedProvider(provider, intervention_system)

        # Test with a prompt that needs intervention
        prompt = "Is climate change definitely real?"

        print(f"Original prompt: {prompt}")
        response = await intervened_provider.complete(prompt, model="gpt-4o-mini", max_tokens=200)

        if response["success"]:
            print(f"\nFinal response: {response['content']}")
            print(f"\nIntervention stats: {response['intervention_stats']}")
        else:
            print(f"Error: {response.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"Demo requires API key configuration: {e}")


async def demonstrate_custom_interventions():
    """Show how to create custom intervention strategies."""
    print("\n=== Custom Interventions Demo ===\n")

    from use_cases.alignment.runtime.base import InterventionResult, OutputFilter

    # Create a custom jargon simplifier
    class JargonSimplifier(OutputFilter):
        def __init__(self):
            self.jargon_map = {
                "utilize": "use",
                "implement": "do",
                "leverage": "use",
                "synergy": "working together",
                "paradigm": "model",
                "methodology": "method",
            }

        def should_intervene(self, output: str, context: AlignmentContext) -> bool:
            return any(jargon in output.lower() for jargon in self.jargon_map)

        def filter(self, output: str, context: AlignmentContext) -> InterventionResult:
            simplified = output
            replacements = []

            for jargon, simple in self.jargon_map.items():
                if jargon in simplified.lower():
                    # Case-insensitive replacement while preserving original case
                    import re

                    pattern = re.compile(re.escape(jargon), re.IGNORECASE)
                    simplified = pattern.sub(simple, simplified)
                    replacements.append((jargon, simple))

            return InterventionResult(
                success=True,
                intervention_type=InterventionType.OUTPUT_FILTERING,
                modified_value=simplified,
                original_value=output,
                metadata={"replacements": replacements},
            )

    # Use the custom intervention
    system = RuntimeInterventionSystem()
    system.add_intervention(JargonSimplifier())

    corporate_speak = (
        "We need to leverage our core competencies to implement a new paradigm "
        "that will create synergy across all departments. Our methodology will "
        "utilize best practices to achieve optimal results."
    )

    print(f"Original: {corporate_speak}")
    simplified = system.intervene_on_output(corporate_speak)
    print(f"Simplified: {simplified}")


async def demonstrate_safety_pipeline():
    """Show how safety checks work in the pipeline."""
    print("\n=== Safety Pipeline Demo ===\n")

    # Create system with strict safety settings
    config = InterventionConfig(safety_threshold=0.8, block_on_safety_failure=True)
    system = RuntimeInterventionSystem(config)

    # Add multiple safety layers
    from use_cases.alignment.runtime import EthicalGuardrailsChecker

    system.add_intervention(EthicalGuardrailsChecker())

    # Test various safety scenarios
    test_cases = [
        "Here's how to manipulate emotions to get what you want.",
        "You can exploit vulnerability in people by doing this.",
        "This technique helps improve communication skills.",
        "Always respect others' privacy and get consent first.",
    ]

    for test in test_cases:
        print(f"\nChecking: {test}")
        result = system.intervene_on_output(test)
        print(f"Result: {result}")

        # Check if it was blocked
        stats = system.get_intervention_stats()
        if stats["total_interventions"] > 0:
            recent = stats["recent_interventions"][-1]
            if recent["type"] == "safety_check" and not recent["success"]:
                print("Status: BLOCKED for safety")
            else:
                print("Status: Modified but allowed")
        else:
            print("Status: Passed all checks")


async def main():
    """Run all demonstrations."""
    await demonstrate_prompt_intervention()
    await demonstrate_output_filtering()
    await demonstrate_custom_interventions()
    await demonstrate_safety_pipeline()
    await demonstrate_integrated_system()

    print("\n=== Demo Complete ===")
    print("The runtime intervention framework provides flexible tools for:")
    print("- Modifying prompts before they reach the LLM")
    print("- Filtering and adjusting LLM outputs")
    print("- Enforcing safety and ethical guidelines")
    print("- Creating custom intervention strategies")
    print("- Building comprehensive alignment pipelines")


if __name__ == "__main__":
    asyncio.run(main())
