"""Truthfulness benchmark module."""

import json
import os


def validate_dataset():
    """
    Validate the truthfulness dataset.jsonl file.
    
    Returns:
        bool: True if dataset is valid
        
    Raises:
        FileNotFoundError: If dataset.jsonl doesn't exist
        json.JSONDecodeError: If JSON is malformed
        ValueError: If required fields are missing or invalid
    """
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset.jsonl')

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    required_fields = {'id', 'prompt', 'evaluation_method', 'expected_keywords'}

    with open(dataset_path, encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {line_num}: {e.msg}",
                    e.doc,
                    e.pos
                )

            # Check required fields
            missing_fields = required_fields - set(entry.keys())
            if missing_fields:
                raise ValueError(
                    f"Missing required fields on line {line_num}: {missing_fields}"
                )

            # Validate expected_keywords is a list
            if not isinstance(entry['expected_keywords'], list):
                raise ValueError(
                    f"'expected_keywords' must be a list on line {line_num}"
                )

            # Validate expected_keywords is not empty
            if not entry['expected_keywords']:
                raise ValueError(
                    f"'expected_keywords' cannot be empty on line {line_num}"
                )

    return True
