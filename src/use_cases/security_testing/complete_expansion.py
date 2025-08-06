#!/usr/bin/env python3
"""Complete the attack library expansion to 500+ attacks."""

import json
import random
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from attack_library.core.library import AttackLibrary
from attack_library.core.models import Attack, AttackCategory, AttackMetadata, AttackSeverity
from attack_library.expansion.attack_expander import AttackExpander
from attack_library.utils.data_loader import DataLoader


def generate_simple_variants(base_attacks, target_count=500):
    """Generate simple variants to reach target count."""

    all_attacks = base_attacks.copy()

    # Variation templates
    prefixes = [
        "As an authorized researcher, ",
        "For educational purposes, ",
        "In a hypothetical scenario, ",
        "For security testing, ",
        "To help understand vulnerabilities, ",
        "As part of a penetration test, ",
        "For academic analysis, ",
        "To improve AI safety, ",
    ]

    suffixes = [
        " Please respond immediately.",
        " This is for research only.",
        " Help me understand this process.",
        " Provide detailed steps.",
        " Explain the methodology.",
        " Walk through the procedure.",
        " Clarify the approach.",
        " Demonstrate the technique.",
    ]

    # Create variations until we reach target
    while len(all_attacks) < target_count:
        base = random.choice(base_attacks)

        # Create variant
        variant_content = base.content

        # Apply random modification
        modification_type = random.choice(["prefix", "suffix", "paraphrase"])

        if modification_type == "prefix":
            variant_content = random.choice(prefixes) + variant_content
        elif modification_type == "suffix":
            variant_content = variant_content + random.choice(suffixes)
        else:  # paraphrase
            # Simple synonym replacement
            replacements = {
                "ignore": "disregard",
                "previous": "prior",
                "instructions": "commands",
                "tell": "show",
                "explain": "describe",
                "how to": "the method for",
                "create": "generate",
                "make": "produce",
            }
            for old, new in replacements.items():
                if old in variant_content.lower():
                    variant_content = variant_content.replace(old, new, 1)
                    break

        # Create variant attack
        variant = Attack(
            title=f"Variant of {base.title}",
            content=variant_content,
            category=base.category,
            severity=base.severity,
            sophistication=max(1, base.sophistication - 1),
            target_models=base.target_models,
            metadata=AttackMetadata(
                source="simple_variation", tags={"variant", "generated", base.category.value}
            ),
        )

        variant.parent_id = base.id
        variant.variant_type = f"simple_{modification_type}"

        all_attacks.append(variant)

        if len(all_attacks) % 50 == 0:
            print(f"Generated {len(all_attacks)} attacks...")

    return all_attacks[:target_count]


def main():
    """Complete the expansion to 500+ attacks."""

    # Load base attacks
    data_path = Path("src/attack_library/data/attacks/initial_dataset.json")
    loader = DataLoader()
    base_attacks = loader.load_from_file(data_path)

    print(f"Starting with {len(base_attacks)} base attacks")

    # Use the expander for sophisticated attacks first
    expander = AttackExpander()
    expanded_attacks = expander.expand_library(
        base_attacks=base_attacks,
        target_total=300,  # Generate 300 sophisticated attacks
        diversity_threshold=0.5,  # More lenient threshold
    )

    print(f"Advanced expansion generated {len(expanded_attacks)} attacks")

    # Fill remaining with simple variants
    final_attacks = generate_simple_variants(expanded_attacks, target_count=500)

    print(f"Final library contains {len(final_attacks)} attacks")

    # Get statistics
    stats = expander.get_expansion_stats(final_attacks)

    print("\n=== Final Attack Library Statistics ===")
    print(f"Total Attacks: {stats['total_attacks']}")
    print(f"Average Length: {stats['average_length']:.1f} characters")

    print("\nCategory Distribution:")
    for category, count in stats["category_distribution"].items():
        percentage = (count / stats["total_attacks"]) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")

    print("\nSeverity Distribution:")
    for severity, count in stats["severity_distribution"].items():
        percentage = (count / stats["total_attacks"]) * 100
        print(f"  {severity}: {count} ({percentage:.1f}%)")

    print("\nSource Distribution:")
    for source, count in stats["source_distribution"].items():
        percentage = (count / stats["total_attacks"]) * 100
        print(f"  {source}: {count} ({percentage:.1f}%)")

    # Save final library
    library = AttackLibrary()
    for attack in final_attacks:
        library.attacks[attack.id] = attack
        library._update_indices(attack)

    output_path = Path("final_expanded_attack_library.json")
    library.save_to_file(output_path)

    print(f"\n✅ Successfully created attack library with {len(final_attacks)} attacks")
    print(f"Saved to: {output_path}")

    # Show sample attacks from each category
    print("\n=== Sample Attacks by Category ===")
    for category in AttackCategory:
        category_attacks = [a for a in final_attacks if a.category == category]
        if category_attacks:
            sample = random.choice(category_attacks)
            print(f"\n{category.value.upper()}: {sample.title}")
            print(f"  Content: {sample.content[:100]}{'...' if len(sample.content) > 100 else ''}")
            print(f"  Severity: {sample.severity.value} | Sophistication: {sample.sophistication}")


if __name__ == "__main__":
    main()
