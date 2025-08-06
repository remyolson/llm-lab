#!/usr/bin/env python3
"""
Fine-Tuning CLI Entry Point

This script provides the main entry point for the fine-tuning CLI.

Usage:
    python scripts/finetune.py --help
    python scripts/finetune.py train --recipe chat --model llama2-7b --data custom.jsonl
    python scripts/finetune.py list-jobs
    python scripts/finetune.py evaluate --checkpoint path/to/checkpoint
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from use_cases.fine_tuning.cli.fine_tuning_cli import main

if __name__ == "__main__":
    main()
