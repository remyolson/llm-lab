#!/usr/bin/env python3
"""Test script for attack library expansion."""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from attack_library.core.library import AttackLibrary
from attack_library.expansion.attack_expander import AttackExpander
from attack_library.utils.data_loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Test attack library expansion."""

    # Load initial dataset
    data_path = Path("src/attack_library/data/attacks/initial_dataset.json")
    if not data_path.exists():
        logger.error(f"Initial dataset not found at {data_path}")
        return

    logger.info("Loading initial dataset...")
    loader = DataLoader()
    base_attacks = loader.load_from_file(data_path)

    logger.info(f"Loaded {len(base_attacks)} base attacks")

    # Initialize expander
    logger.info("Initializing attack expander...")
    expander = AttackExpander()

    # Expand library to 500+ attacks
    target_total = 500
    logger.info(f"Expanding library to {target_total} attacks...")

    expanded_attacks = expander.expand_library(
        base_attacks=base_attacks,
        target_total=target_total,
        diversity_threshold=0.6,  # Lower threshold for more attacks
    )

    logger.info(f"Expansion complete! Generated {len(expanded_attacks)} attacks")

    # Get expansion statistics
    stats = expander.get_expansion_stats(expanded_attacks)

    logger.info("Expansion Statistics:")
    logger.info(f"  Total Attacks: {stats['total_attacks']}")
    logger.info(f"  Average Length: {stats['average_length']:.1f} characters")
    logger.info(f"  Unique Targets: {stats['unique_targets']}")

    logger.info("\nCategory Distribution:")
    for category, count in stats["category_distribution"].items():
        percentage = (count / stats["total_attacks"]) * 100 if stats["total_attacks"] > 0 else 0
        logger.info(f"  {category}: {count} ({percentage:.1f}%)")

    logger.info("\nSeverity Distribution:")
    for severity, count in stats["severity_distribution"].items():
        percentage = (count / stats["total_attacks"]) * 100 if stats["total_attacks"] > 0 else 0
        logger.info(f"  {severity}: {count} ({percentage:.1f}%)")

    logger.info("\nSophistication Distribution:")
    for level, count in stats["sophistication_distribution"].items():
        percentage = (count / stats["total_attacks"]) * 100 if stats["total_attacks"] > 0 else 0
        logger.info(f"  Level {level}: {count} ({percentage:.1f}%)")

    logger.info("\nSource Distribution:")
    for source, count in stats["source_distribution"].items():
        percentage = (count / stats["total_attacks"]) * 100 if stats["total_attacks"] > 0 else 0
        logger.info(f"  {source}: {count} ({percentage:.1f}%)")

    # Save expanded library
    output_path = Path("expanded_attack_library.json")
    logger.info(f"Saving expanded library to {output_path}...")

    # Create library and save
    library = AttackLibrary()
    for attack in expanded_attacks:
        library.attacks[attack.id] = attack
        library._update_indices(attack)

    library.save_to_file(output_path)
    logger.info(f"✅ Saved expanded library with {len(expanded_attacks)} attacks")

    # Display some example attacks
    logger.info("\nExample Generated Attacks:")
    for i, attack in enumerate(expanded_attacks[-10:], 1):  # Show last 10 attacks
        logger.info(f"\n{i}. {attack.title} ({attack.id})")
        logger.info(f"   Category: {attack.category.value} | Severity: {attack.severity.value}")
        logger.info(f"   Source: {attack.metadata.source}")
        content_preview = (
            attack.content[:80] + "..." if len(attack.content) > 80 else attack.content
        )
        logger.info(f"   Content: {content_preview}")


if __name__ == "__main__":
    main()
