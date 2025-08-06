#!/usr/bin/env python3
"""Debug script for expansion issues."""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Debug expansion."""
    try:
        from attack_library.core.library import AttackLibrary
        from attack_library.expansion.attack_expander import AttackExpander

        print("Loading library...")
        library = AttackLibrary("attack_library.json")
        base_attacks = list(library.attacks.values())
        print(f"Loaded {len(base_attacks)} attacks")

        if not base_attacks:
            print("No attacks found in library")
            return

        print("Initializing expander...")
        expander = AttackExpander()

        print("Starting expansion with 110 target, 0.4 diversity...")
        expanded_attacks = expander.expand_library(
            base_attacks=base_attacks,
            target_total=110,  # Small target for debugging
            diversity_threshold=0.4,
        )

        print(f"Expansion successful! Generated {len(expanded_attacks)} attacks")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
