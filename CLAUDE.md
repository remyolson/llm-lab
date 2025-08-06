# Claude Code Instructions

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md

## Codebase Structure Standards

**CRITICAL: All application source code lives in `src/` and nowhere else.**

### Root Directory Structure
```
llm-lab/
├── src/                    # ALL SOURCE CODE (Single source of truth)
│   ├── providers/          # LLM provider integrations
│   ├── use_cases/          # Feature modules (13 use cases)
│   ├── evaluation/         # Core evaluation logic
│   ├── config/             # Configuration management
│   ├── types/              # Type definitions
│   └── utils/              # Shared utilities
├── docs/                   # Documentation and guides
├── tests/                  # All test code
├── examples/               # Usage examples and demos
├── scripts/                # Utility and automation scripts
├── datasets/               # Data files and management
├── models/                 # Model files and configurations
├── templates/              # Reusable templates
├── results/                # Benchmark results
└── .taskmaster/            # Project management
```

### Use Cases (Features) - All in `src/use_cases/`
1. **security_testing** - LLM Security Testing Framework (500+ attack patterns)
2. **synthetic_data** - Synthetic Data Generation Platform
3. **model_documentation** - Automated Model Documentation System
4. **interpretability** - LLM Interpretability Suite
5. **benchmark_creation** - Benchmark Creation Tool
6. **fine_tuning** - Fine-tuning workflows
7. **monitoring** - Continuous monitoring
8. **custom_prompts** - Custom prompt engineering
9. **local_models** - Local model management
10. **alignment** - AI alignment tools
11. **evaluation_framework** - Evaluation framework
12. **visual_analytics** - Visual analytics dashboard
13. **fine_tuning_studio** - Interactive fine-tuning

### Anti-Patterns (NEVER CREATE)
❌ **Root-level feature directories**: `my-feature/`, `feature-system/`, `standalone-module/`
❌ **Hyphenated directories at root**: `security-testing/`, `model-docs-system/`
❌ **Mixed locations**: Some features in `src/`, others at root

### Use Case Module Structure Standard
```
src/use_cases/example_feature/
├── __init__.py             # Main exports
├── setup.py                # Package config (name: llm-{feature})
├── requirements.txt        # Feature-specific dependencies
├── README.md              # Overview and setup
├── src/example_feature/   # Implementation
│   ├── cli.py
│   ├── core/
│   └── models.py
└── tests/                 # Feature tests
```

### Import Standards
```python
# ✅ Always use absolute imports from src/
from src.providers.openai import OpenAIProvider
from src.use_cases.security_testing import SecurityScanner

# ❌ Never use relative imports or old paths
from ..providers.openai import OpenAIProvider
from attack_library_system import SecurityScanner  # Old structure
```

### Naming Conventions
- **Directories**: `snake_case` (security_testing, model_documentation)
- **Packages**: `llm-{use_case}` (llm-security-testing, llm-synthetic-data)
- **Files**: `snake_case.py` following PEP 8

### Development Commands
```bash
# Navigate to use case
cd src/use_cases/security_testing

# Install use case dependencies
pip install -r requirements.txt

# Run use case CLI
python -m attack_library.cli scan --model gpt-4o-mini
```

**📖 Full Standards**: See `.taskmaster/docs/codebase_structure_standards.md` for comprehensive guidelines.

## Code Quality Requirements
- **Ruff**: Linting and formatting must pass
- **MyPy**: Type checking (where configured)
- **Pytest**: All tests must pass
- **Structure**: Follow standards above
