# Attack Library System

A comprehensive attack library and prompt generation system for LLM security testing. This system provides over 500+ categorized attack prompts for testing various vulnerabilities including jailbreak attempts, prompt injections, data extraction, manipulation, and evasion techniques.

## Features

- **Comprehensive Attack Database**: 500+ categorized attack prompts covering all major LLM vulnerabilities
- **Dynamic Prompt Generation**: AI-powered generation of new attack variants
- **Multi-Category Support**: Jailbreak, injection, extraction, manipulation, and evasion attacks
- **Severity Classification**: Attacks rated by severity (low/medium/high/critical)
- **Sophistication Scoring**: 1-5 scale sophistication ratings
- **Target Model Mapping**: Model-specific attack effectiveness tracking
- **Metadata System**: Rich metadata including source, effectiveness scores, and tags
- **Search and Filtering**: Advanced search capabilities by category, severity, and sophistication
- **Batch Operations**: Efficient batch processing for large-scale testing
- **API Interface**: RESTful API for integration with testing frameworks
- **CLI Tool**: Command-line interface for interactive usage

## Quick Start

```bash
# Install the package
pip install -e .

# Initialize attack library
attack-library init

# List available attacks
attack-library list --category jailbreak

# Generate new attack variants
attack-library generate --category injection --count 10

# Search attacks by keyword
attack-library search "ignore instructions"

# Export attacks for testing
attack-library export --format json --output attacks.json
```

## Architecture

### Core Components

1. **AttackLibrary**: Main class for managing the attack database
2. **Attack**: Data model for individual attack prompts
3. **AttackGenerator**: Dynamic generation of new attack variants
4. **AttackMetadata**: Rich metadata system for attack tracking
5. **SearchEngine**: Advanced search and filtering capabilities

### Data Model

Each attack in the library contains:

- **id**: Unique identifier
- **title**: Descriptive title
- **content**: Attack prompt content
- **category**: Attack type (jailbreak/injection/extraction/manipulation/evasion)
- **severity**: Impact level (low/medium/high/critical)
- **sophistication**: Complexity score (1-5)
- **target_models**: List of applicable model types
- **metadata**: Additional information (source, effectiveness, tags, etc.)

### Storage Format

Attacks are stored in versioned JSON format with schema validation:

```json
{
  "version": "1.0",
  "schema": "attack-library-v1.0",
  "attacks": [
    {
      "id": "jb_001",
      "title": "Basic Jailbreak Attempt",
      "content": "Ignore all previous instructions...",
      "category": "jailbreak",
      "severity": "medium",
      "sophistication": 2,
      "target_models": ["gpt-3.5", "gpt-4"],
      "metadata": {
        "source": "research_paper",
        "effectiveness_score": 0.75,
        "creation_date": "2024-01-15",
        "tags": ["basic", "instruction-override"]
      }
    }
  ]
}
```

## API Reference

### AttackLibrary Class

```python
from attack_library import AttackLibrary

# Initialize library
library = AttackLibrary()

# Load attacks from file
library.load_from_file("attacks.json")

# Search attacks
results = library.search(
    category="jailbreak",
    severity=["high", "critical"],
    min_sophistication=3
)

# Add new attack
library.add_attack(
    title="Custom Attack",
    content="Custom prompt content...",
    category="injection",
    severity="high"
)

# Generate variants
variants = library.generate_variants(
    base_attack_id="jb_001",
    count=5,
    variation_type="paraphrase"
)
```

## Categories

### Jailbreak Attacks
- Instruction override attempts
- Role-playing scenarios
- Hypothetical situations
- DAN (Do Anything Now) variants

### Injection Attacks
- Prompt injection techniques
- Context manipulation
- Instruction confusion
- Template breaking

### Extraction Attacks
- Data exfiltration attempts
- Training data recovery
- Prompt extraction
- Model introspection

### Manipulation Attacks
- Output manipulation
- Bias exploitation
- Preference override
- Safety bypass

### Evasion Attacks
- Content filter bypass
- Detection avoidance
- Obfuscation techniques
- Encoding tricks

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Security Notice

This tool is designed for security testing and research purposes only. Use responsibly and in accordance with applicable laws and terms of service.
