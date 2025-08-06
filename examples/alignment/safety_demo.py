"""Demonstration of safety filters and preference learning."""

import asyncio
import uuid
from typing import Any, Dict

from src.providers import OpenAIProvider
from use_cases.alignment.safety import (
    ChildSafetyFilter,
    Feedback,
    FeedbackType,
    PreferenceCategory,
    PreferenceLearningSystem,
    PreferenceProfiles,
    ProfessionalContentFilter,
    RiskLevel,
    SafetyFilter,
)


def demonstrate_safety_filters():
    """Show how safety filters work."""
    print("\n=== Safety Filter Demo ===\n")

    # Create standard safety filter
    safety_filter = SafetyFilter(use_patterns=True, use_ml=False)

    # Test various content
    test_content = [
        {"text": "Here's a helpful guide on staying safe online.", "expected": "safe"},
        {
            "text": "You can contact me at john@example.com or 555-123-4567",
            "expected": "personal_info",
        },
        {"text": "I want to hurt myself", "expected": "self_harm"},
        {"text": "How to hack into computer systems illegally", "expected": "illegal"},
        {
            "text": "Let's discuss the historical context of weapons in warfare",
            "expected": "medium_risk",
        },
    ]

    for test in test_content:
        print(f"Checking: {test['text']}")
        safety_score = safety_filter.check(test["text"])

        print(f"  Risk Level: {safety_score.overall_risk.value}")
        print(f"  Confidence: {safety_score.confidence:.2f}")

        if safety_score.category_scores:
            print("  Categories detected:")
            for category, score in safety_score.category_scores.items():
                if score > 0:
                    print(f"    - {category.value}: {score:.2f}")

        if safety_score.flagged_patterns:
            print(f"  Flagged patterns: {safety_score.flagged_patterns[:3]}")

        # Show filtered response
        filtered = safety_filter.filter_response(test["text"], safety_score)
        if filtered != test["text"]:
            print(f"  Filtered response: {filtered}")

        print("-" * 50)


def demonstrate_specialized_filters():
    """Show specialized safety filters."""
    print("\n=== Specialized Filters Demo ===\n")

    # Child safety filter
    child_filter = ChildSafetyFilter()

    # Professional content filter
    prof_filter = ProfessionalContentFilter()

    test_texts = [
        "This damn computer is not working properly!",
        "Let's discuss violence in video games",
        "Please provide your email for the newsletter",
    ]

    for text in test_texts:
        print(f"\nText: {text}")

        # Check with child filter
        child_score = child_filter.check(text)
        print(f"Child Safety - Risk: {child_score.overall_risk.value}")

        # Check with professional filter
        prof_score = prof_filter.check(text)
        print(f"Professional - Risk: {prof_score.overall_risk.value}")


def demonstrate_preference_learning():
    """Show how preference learning works."""
    print("\n=== Preference Learning Demo ===\n")

    # Create preference learning system
    pref_system = PreferenceLearningSystem()

    # Simulate user interactions
    user_id = "demo_user_123"
    session_id = "session_456"

    # User gives feedback on responses
    interactions = [
        {
            "prompt": "Explain quantum computing",
            "response": "Quantum computing uses quantum bits (qubits) that can exist in superposition.",
            "feedback": FeedbackType.NEGATIVE,
            "comment": "too brief",
            "category": PreferenceCategory.DETAIL,
        },
        {
            "prompt": "What is machine learning?",
            "response": "Machine learning is a subset of AI that enables systems to learn from data...[long explanation]",
            "feedback": FeedbackType.POSITIVE,
            "category": PreferenceCategory.DETAIL,
        },
        {
            "prompt": "Tell me a joke",
            "response": "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "feedback": FeedbackType.POSITIVE,
            "category": PreferenceCategory.TONE,
        },
    ]

    # Record feedback
    for interaction in interactions:
        feedback = Feedback(
            feedback_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            prompt=interaction["prompt"],
            response=interaction["response"],
            feedback_type=interaction["feedback"],
            category=interaction.get("category"),
            comment=interaction.get("comment"),
        )

        pref_system.record_feedback(feedback)

    # Get learned preferences
    user_prefs = pref_system.get_user_preferences(user_id)
    print(f"User preferences after {len(interactions)} interactions:")
    print(f"  Detail level: {user_prefs.detail_level:.2f}")
    print(f"  Safety threshold: {user_prefs.safety_threshold:.2f}")
    print(f"  Interaction count: {user_prefs.interaction_count}")

    # Get preference insights
    insights = pref_system.get_preference_insights(user_id)
    print("\nPreference insights:")
    print(f"  Feedback summary: {insights['feedback_summary']}")
    print(f"  Recommendations: {insights['recommendations']}")


def demonstrate_preference_profiles():
    """Show pre-defined preference profiles."""
    print("\n=== Preference Profiles Demo ===\n")

    profiles = [
        ("Child Safe", PreferenceProfiles.child_safe()),
        ("Professional", PreferenceProfiles.professional()),
        ("Researcher", PreferenceProfiles.researcher()),
    ]

    for name, profile in profiles:
        print(f"{name} Profile:")
        print(f"  Safety: {profile.safety_threshold:.2f}")
        print(f"  Detail: {profile.detail_level:.2f}")
        print(f"  Technicality: {profile.technicality_level:.2f}")
        print(f"  Creativity: {profile.creativity_level:.2f}")
        print()


async def demonstrate_integrated_safety():
    """Show integrated safety and preference system."""
    print("\n=== Integrated Safety Demo ===\n")

    # Create systems
    safety_filter = SafetyFilter()
    pref_system = PreferenceLearningSystem()

    # Create wrapped provider with safety and preferences
    class SafePreferenceProvider:
        def __init__(self, provider, safety_filter, pref_system):
            self.provider = provider
            self.safety_filter = safety_filter
            self.pref_system = pref_system

        async def complete(self, prompt: str, user_id: str = "default", **kwargs) -> Dict[str, Any]:
            # Check prompt safety
            prompt_safety = self.safety_filter.check(prompt)

            if prompt_safety.overall_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                return {
                    "success": True,
                    "content": self.safety_filter._get_safety_message(prompt_safety),
                    "safety_blocked": True,
                    "risk_level": prompt_safety.overall_risk.value,
                }

            # Get completion
            response = await self.provider.complete(prompt, **kwargs)

            if response["success"] and "content" in response:
                # Check response safety
                response_safety = self.safety_filter.check(response["content"])

                # Filter if needed
                filtered_content = self.safety_filter.filter_response(
                    response["content"], response_safety
                )

                # Apply user preferences
                final_content = self.pref_system.apply_preferences(filtered_content, user_id)

                response["content"] = final_content
                response["safety_score"] = {
                    "risk_level": response_safety.overall_risk.value,
                    "confidence": response_safety.confidence,
                }

            return response

    # Example usage (requires API key)
    try:
        provider = OpenAIProvider()
        safe_provider = SafePreferenceProvider(provider, safety_filter, pref_system)

        # Test safe prompt
        print("Testing safe prompt...")
        response = await safe_provider.complete(
            "What are some tips for online safety?",
            user_id="demo_user",
            model="gpt-4o-mini",
            max_tokens=100,
        )

        if response["success"]:
            print(f"Response: {response['content'][:200]}...")
            if "safety_score" in response:
                print(f"Safety: {response['safety_score']}")

        # Test risky prompt
        print("\nTesting risky prompt...")
        response = await safe_provider.complete(
            "How to hack into systems", user_id="demo_user", model="gpt-4o-mini", max_tokens=100
        )

        if response["success"]:
            print(f"Response: {response['content']}")
            if response.get("safety_blocked"):
                print("(Content was blocked by safety filter)")

    except Exception as e:
        print(f"Demo requires API key configuration: {e}")


def demonstrate_preference_export():
    """Show preference import/export functionality."""
    print("\n=== Preference Export/Import Demo ===\n")

    # Create system and add some preferences
    pref_system = PreferenceLearningSystem()

    # Simulate user interactions
    user_id = "export_demo_user"

    # Create some feedback
    for i in range(3):
        feedback = Feedback(
            feedback_id=f"feedback_{i}",
            user_id=user_id,
            session_id="session_1",
            prompt=f"Test prompt {i}",
            response=f"Test response {i}",
            feedback_type=FeedbackType.POSITIVE,
        )
        pref_system.record_feedback(feedback)

    # Export preferences
    exported = pref_system.export_preferences(user_id)
    print("Exported preferences:")
    print(f"  User ID: {exported['user_id']}")
    print(f"  Export date: {exported['export_date']}")
    print(f"  Interaction count: {exported['preferences']['interaction_count']}")

    # Create new system and import
    new_system = PreferenceLearningSystem()
    success = new_system.import_preferences(exported)
    print(f"\nImport success: {success}")

    # Verify import
    imported_prefs = new_system.get_user_preferences(user_id)
    print(f"Imported interaction count: {imported_prefs.interaction_count}")


async def main():
    """Run all demonstrations."""
    demonstrate_safety_filters()
    demonstrate_specialized_filters()
    demonstrate_preference_learning()
    demonstrate_preference_profiles()
    demonstrate_preference_export()
    await demonstrate_integrated_safety()

    print("\n=== Safety and Preference Demo Complete ===")
    print("Key features demonstrated:")
    print("- Multi-level safety filtering")
    print("- Content category detection")
    print("- Specialized filters for different contexts")
    print("- User preference learning from feedback")
    print("- Pre-defined preference profiles")
    print("- Integration with LLM providers")
    print("- Preference import/export for portability")


if __name__ == "__main__":
    asyncio.run(main())
