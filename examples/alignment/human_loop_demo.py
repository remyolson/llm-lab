"""Demonstration of human-in-the-loop interface and A/B testing."""

import asyncio
from typing import Tuple

from use_cases.alignment.human_loop import (
    # A/B testing
    ABTestingFramework,
    Experiment,
    # Review system
    HumanInTheLoopSystem,
    ReviewAction,
    ReviewPriority,
    ReviewRequest,
    Variant,
    VariantType,
    create_intervention_experiment,
    create_safety_threshold_experiment,
)


async def demonstrate_human_review():
    """Show how human review system works."""
    print("\n=== Human Review System Demo ===\n")

    # Create HITL system
    hitl = HumanInTheLoopSystem(auto_approve_threshold=0.8, review_sample_rate=0.3)

    # Test content that needs review
    test_cases = [
        {
            "prompt": "What's the weather like today?",
            "response": "I don't have access to real-time weather data.",
            "should_review": False,
        },
        {
            "prompt": "How can I get medical advice for my symptoms?",
            "response": "You should consult with a qualified healthcare provider for medical concerns.",
            "should_review": True,
        },
        {
            "prompt": "Tell me about investment strategies",
            "response": "Here are some general investment principles...",
            "should_review": True,
        },
    ]

    for i, test in enumerate(test_cases):
        print(f"Test {i + 1}: {test['prompt'][:50]}...")

        # Check if should review
        should_review = await hitl.should_review(test["prompt"], test["response"], safety_score=0.7)

        print(f"  Should review: {should_review}")
        print(f"  Expected: {test['should_review']}")

        if should_review:
            # Submit for review
            request_id = await hitl.submit_for_review(
                test["prompt"],
                test["response"],
                ReviewPriority.MEDIUM,
                "Flagged for content review",
            )
            print(f"  Submitted for review: {request_id}")

        print("-" * 50)


async def demonstrate_review_queue():
    """Show how review queue processing works."""
    print("\n=== Review Queue Processing Demo ===\n")

    hitl = HumanInTheLoopSystem()

    # Submit several items for review
    review_items = [
        ("What is machine learning?", "ML is a subset of AI...", ReviewPriority.LOW),
        ("How do I invest money?", "Consider diversified portfolios...", ReviewPriority.HIGH),
        ("Medical advice needed", "Please consult a doctor...", ReviewPriority.CRITICAL),
    ]

    request_ids = []
    for prompt, response, priority in review_items:
        request_id = await hitl.submit_for_review(prompt, response, priority)
        request_ids.append(request_id)
        print(f"Submitted: {prompt[:30]}... (Priority: {priority.value})")

    # Get pending reviews
    pending = await hitl.review_interface.get_pending_reviews()
    print(f"\nPending reviews: {len(pending)}")

    # Create review queue handler
    handler = hitl.create_review_queue_handler()

    # Define a simple review function
    def simple_reviewer(request: ReviewRequest) -> Tuple[ReviewAction, str, str]:
        """Simple automated reviewer."""
        if "medical" in request.prompt.lower() or "investment" in request.prompt.lower():
            return (
                ReviewAction.MODIFY,
                "Please consult a qualified professional.",
                "Requires disclaimer",
            )
        else:
            return ReviewAction.APPROVE, None, "Content approved"

    # Process reviews (simulate with manual processing)
    print("\nProcessing reviews...")
    for request in pending:
        action, modified_content, notes = simple_reviewer(request)

        if hasattr(hitl.review_interface, "submit_review"):
            await hitl.review_interface.submit_review(
                request.request_id, "demo_reviewer", action, modified_content, notes
            )

        print(f"  {request.request_id}: {action.value}")

    # Check results
    for request_id in request_ids:
        result = await hitl.review_interface.get_review_result(request_id)
        if result:
            print(f"Result {request_id}: {result.status.value}")


async def demonstrate_ab_testing():
    """Show how A/B testing framework works."""
    print("\n=== A/B Testing Framework Demo ===\n")

    # Create A/B testing framework
    ab_test = ABTestingFramework()

    # Create an experiment
    experiment = create_intervention_experiment()
    experiment_id = ab_test.create_experiment(experiment)

    print(f"Created experiment: {experiment.name}")
    print(f"Variants: {[v.name for v in experiment.variants]}")

    # Start the experiment
    experiment.start()

    # Simulate user allocations
    test_users = [f"user_{i}" for i in range(100)]
    allocations = {"control": 0, "treatment": 0}

    for user_id in test_users:
        variant = ab_test.allocate_user(user_id, experiment_id)
        if variant:
            allocations[variant.variant_id] += 1

            # Simulate some conversions
            import random

            if random.random() < 0.3:  # 30% conversion rate
                ab_test.record_conversion(user_id, experiment_id, "conversion")

    print("\nUser allocations:")
    for variant_id, count in allocations.items():
        print(f"  {variant_id}: {count} users")

    # Get experiment results
    results = ab_test.get_experiment_results(experiment_id)

    print("\nExperiment Results:")
    print(f"Total exposures: {results['total_exposures']}")

    for variant_result in results["variants"]:
        print(f"\n{variant_result['name']}:")
        print(f"  Exposures: {variant_result['exposures']}")
        print(f"  Conversions: {variant_result['conversions']}")
        print(f"  Conversion rate: {variant_result['conversion_rate']:.2%}")


async def demonstrate_safety_threshold_experiment():
    """Show safety threshold A/B testing."""
    print("\n=== Safety Threshold Experiment Demo ===\n")

    ab_test = ABTestingFramework()

    # Create safety threshold experiment
    experiment = create_safety_threshold_experiment()
    experiment_id = ab_test.create_experiment(experiment)

    print(f"Created experiment: {experiment.name}")
    print(f"Testing thresholds: {[v.config['safety_threshold'] for v in experiment.variants]}")

    experiment.start()

    # Simulate user interactions with different safety outcomes
    test_scenarios = [
        ("safe_content", 0.95, True),  # Safe content, high score, should convert
        ("questionable", 0.75, False),  # Questionable, medium score, blocked by high threshold
        ("risky_content", 0.4, False),  # Risky content, low score, blocked by all
    ]

    user_counter = 0
    for scenario_type, safety_score, should_convert in test_scenarios:
        for _ in range(50):  # 50 users per scenario
            user_id = f"user_{user_counter}"
            user_counter += 1

            # Allocate user
            variant = ab_test.allocate_user(user_id, experiment_id)
            if not variant:
                continue

            # Check if content would be blocked
            threshold = variant.config["safety_threshold"]
            blocked = safety_score < threshold

            # Record conversion if not blocked and should convert
            if not blocked and should_convert:
                ab_test.record_conversion(user_id, experiment_id, "conversion")
                ab_test.record_conversion(user_id, experiment_id, "safety_score", safety_score)

    # Get results
    results = ab_test.get_experiment_results(experiment_id)

    print("\nSafety Threshold Results:")

    for variant_result in results["variants"]:
        threshold = next(
            v.config["safety_threshold"]
            for v in experiment.variants
            if v.variant_id == variant_result["variant_id"]
        )

        print(f"\nThreshold {threshold}:")
        print(f"  Exposures: {variant_result['exposures']}")
        print(f"  Conversion rate: {variant_result['conversion_rate']:.2%}")

        # Show safety score stats
        safety_stats = variant_result["metrics"].get("safety_score", {})
        if safety_stats.get("count", 0) > 0:
            print(f"  Avg safety score: {safety_stats['mean']:.2f}")


async def demonstrate_integrated_system():
    """Show integrated human-loop and A/B testing."""
    print("\n=== Integrated System Demo ===\n")

    # Create systems
    hitl = HumanInTheLoopSystem()
    ab_test = ABTestingFramework()

    # Create experiment for review policies
    experiment = Experiment(
        experiment_id="review_policy_test",
        name="Review Policy Comparison",
        description="Compare strict vs. lenient review policies",
    )

    # Strict policy variant
    strict_variant = Variant(
        variant_id="strict",
        name="Strict Review Policy",
        description="Conservative review with high rejection rate",
        variant_type=VariantType.CONTROL,
        config={"policy": "strict"},
    )

    # Lenient policy variant
    lenient_variant = Variant(
        variant_id="lenient",
        name="Lenient Review Policy",
        description="Permissive review with low rejection rate",
        variant_type=VariantType.TREATMENT,
        config={"policy": "lenient"},
    )

    experiment.add_variant(strict_variant)
    experiment.add_variant(lenient_variant)

    experiment_id = ab_test.create_experiment(experiment)
    experiment.start()

    print(f"Created integrated experiment: {experiment.name}")

    # Simulate content review with different policies
    test_content = [
        ("What's machine learning?", "ML is a branch of AI...", "safe"),
        ("Investment advice needed", "Consider these strategies...", "financial"),
        ("Medical question about symptoms", "You should see a doctor...", "medical"),
        ("How to cook pasta", "Boil water, add pasta...", "safe"),
    ]

    review_results = {
        "strict": {"approved": 0, "rejected": 0},
        "lenient": {"approved": 0, "rejected": 0},
    }

    for i, (prompt, response, category) in enumerate(test_content):
        user_id = f"review_user_{i}"

        # Allocate to variant
        variant = ab_test.allocate_user(user_id, experiment_id)
        if not variant:
            continue

        # Apply review policy
        policy = variant.config["policy"]

        if policy == "strict":
            # Strict policy: reject financial/medical content
            if category in ["financial", "medical"]:
                review_results["strict"]["rejected"] += 1
            else:
                review_results["strict"]["approved"] += 1
                ab_test.record_conversion(user_id, experiment_id, "approved")

        else:  # lenient
            # Lenient policy: approve most content
            if category == "medical":  # Only reject medical
                review_results["lenient"]["rejected"] += 1
            else:
                review_results["lenient"]["approved"] += 1
                ab_test.record_conversion(user_id, experiment_id, "approved")

    print("\nReview Results by Policy:")
    for policy, results in review_results.items():
        total = results["approved"] + results["rejected"]
        approval_rate = results["approved"] / total if total > 0 else 0
        print(f"  {policy.title()}: {approval_rate:.1%} approval rate")

    # Show experiment results
    results = ab_test.get_experiment_results(experiment_id)
    print("\nA/B Test Results:")
    for variant_result in results["variants"]:
        print(f"  {variant_result['name']}: {variant_result['conversion_rate']:.1%} approval rate")


def demonstrate_metrics_and_analysis():
    """Show metrics collection and analysis."""
    print("\n=== Metrics and Analysis Demo ===\n")

    ab_test = ABTestingFramework()

    # Create experiment with multiple metrics
    experiment = Experiment(
        experiment_id="multi_metric_test",
        name="Multi-Metric Analysis",
        description="Track multiple success metrics",
        primary_metric="user_satisfaction",
        secondary_metrics=["response_time", "safety_score", "helpfulness"],
    )

    # Add variants
    control = Variant(
        variant_id="baseline", name="Baseline System", variant_type=VariantType.CONTROL
    )

    treatment = Variant(
        variant_id="enhanced", name="Enhanced System", variant_type=VariantType.TREATMENT
    )

    experiment.add_variant(control)
    experiment.add_variant(treatment)

    experiment_id = ab_test.create_experiment(experiment)
    experiment.start()

    # Simulate data collection
    import random

    for i in range(200):
        user_id = f"metric_user_{i}"
        variant = ab_test.allocate_user(user_id, experiment_id)

        if not variant:
            continue

        # Simulate different performance based on variant
        if variant.variant_id == "baseline":
            satisfaction = random.uniform(0.6, 0.8)
            response_time = random.uniform(1.0, 3.0)
            safety_score = random.uniform(0.7, 0.9)
            helpfulness = random.uniform(0.5, 0.7)
        else:  # enhanced
            satisfaction = random.uniform(0.7, 0.9)
            response_time = random.uniform(0.8, 2.5)
            safety_score = random.uniform(0.8, 0.95)
            helpfulness = random.uniform(0.6, 0.8)

        # Record metrics
        ab_test.record_conversion(user_id, experiment_id, "user_satisfaction", satisfaction)
        ab_test.record_conversion(user_id, experiment_id, "response_time", response_time)
        ab_test.record_conversion(user_id, experiment_id, "safety_score", safety_score)
        ab_test.record_conversion(user_id, experiment_id, "helpfulness", helpfulness)

    # Analyze results
    results = ab_test.get_experiment_results(experiment_id)

    print("Multi-Metric Analysis Results:")
    print("-" * 50)

    for variant_result in results["variants"]:
        print(f"\n{variant_result['name']}:")
        print(f"  Exposures: {variant_result['exposures']}")

        for metric_name, stats in variant_result["metrics"].items():
            if stats["count"] > 0:
                print(f"  {metric_name}:")
                print(f"    Mean: {stats['mean']:.3f}")
                print(f"    Std: {stats['std']:.3f}")
                print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")


async def main():
    """Run all demonstrations."""
    await demonstrate_human_review()
    await demonstrate_review_queue()
    await demonstrate_ab_testing()
    await demonstrate_safety_threshold_experiment()
    await demonstrate_integrated_system()
    demonstrate_metrics_and_analysis()

    print("\n=== Human-in-the-Loop Demo Complete ===")
    print("Key features demonstrated:")
    print("- Human review workflow and queue management")
    print("- Automated review policies and handlers")
    print("- A/B testing framework for alignment strategies")
    print("- Multi-variant experiments with statistical analysis")
    print("- Integration of human feedback with automated systems")
    print("- Comprehensive metrics collection and analysis")
    print("- Safety threshold optimization through experimentation")


if __name__ == "__main__":
    asyncio.run(main())
