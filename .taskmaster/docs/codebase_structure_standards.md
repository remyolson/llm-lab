# LLM Lab Codebase Structure Standards

## Philosophy and Principles

The LLM Lab codebase follows a **domain-driven, hierarchical structure** designed for:

- **Clarity**: Each directory has a single, clear purpose
- **Maintainability**: Easy to navigate, modify, and extend
- **Scalability**: Structure grows logically with new features
- **Discoverability**: Users and contributors can easily find what they need
- **Consistency**: All similar components follow the same patterns

## Root Directory Structure

```
llm-lab/
├── src/                    # All source code (SINGLE SOURCE OF TRUTH)
├── docs/                   # Documentation and guides
├── tests/                  # All test code
├── examples/               # Usage examples and demos
├── scripts/                # Utility and automation scripts
├── datasets/               # Data files and dataset management
├── models/                 # Model files and configurations
├── templates/              # Reusable templates (prompts, configs)
├── results/                # Benchmark and evaluation results
├── benchmarks/             # Benchmark definitions and configs
├── .taskmaster/            # Project management and planning
├── .claude/                # Claude Code configuration
└── [config files]         # Root-level configuration (pyproject.toml, etc.)
```

## Core Principle: Single Source Directory

**CRITICAL RULE**: All application source code lives in `src/` and nowhere else.

❌ **Never create these patterns:**
```
my-feature/                 # Root-level feature directories
feature-name-system/        # Hyphenated feature directories at root
standalone-module/          # Independent modules at root level
```

✅ **Always use this pattern:**
```
src/use_cases/my_feature/   # Features go in use_cases
src/providers/my_provider/  # Providers go in providers
src/utils/my_utility/       # Utilities go in utils
```

## Source Code Organization (`src/`)

```
src/
├── __init__.py
├── providers/              # LLM provider integrations
│   ├── openai.py
│   ├── anthropic.py
│   ├── google.py
│   ├── local/
│   └── __init__.py
├── use_cases/              # Feature modules (USE CASES)
│   ├── security_testing/
│   ├── synthetic_data/
│   ├── model_documentation/
│   ├── interpretability/
│   ├── benchmark_creation/
│   ├── fine_tuning/
│   ├── monitoring/
│   ├── custom_prompts/
│   └── __init__.py
├── evaluation/             # Core evaluation logic
├── config/                 # Configuration management
├── types/                  # Type definitions
├── utils/                  # Shared utilities
├── logging/                # Logging infrastructure
├── analysis/               # Analysis and comparison tools
└── di/                     # Dependency injection system
```

## Use Case Module Structure

Every use case in `src/use_cases/` follows this standard structure:

```
src/use_cases/example_feature/
├── __init__.py             # Main exports and module interface
├── setup.py                # Installable package configuration
├── requirements.txt        # Dependencies specific to this use case
├── README.md              # Use case overview and basic setup
├── src/                   # Implementation code
│   └── example_feature/
│       ├── __init__.py
│       ├── cli.py         # Command-line interface (if applicable)
│       ├── core/          # Core business logic
│       ├── models.py      # Data models and schemas
│       ├── config/        # Configuration handling
│       └── utils/         # Use case specific utilities
├── tests/                 # Use case specific tests
├── docs/                  # Detailed documentation (optional)
├── config/                # Configuration files and examples
└── examples/              # Usage examples (optional)
```

## Documentation Structure (`docs/`)

```
docs/
├── guides/                 # User guides and how-tos
│   ├── USE_CASES_OVERVIEW.md
│   ├── USE_CASE_1_HOW_TO.md
│   ├── USE_CASE_2_HOW_TO.md
│   └── ...
├── api/                    # API documentation
├── architecture/           # Architecture decisions and design
├── development/            # Development guides and standards
├── examples/               # Documentation examples
└── getting_started/        # New user onboarding
```

## Testing Structure (`tests/`)

```
tests/
├── unit/                   # Unit tests
├── integration/            # Integration tests
├── e2e/                    # End-to-end tests
├── performance/            # Performance tests
├── use_cases/              # Use case specific tests
│   ├── security_testing/
│   ├── synthetic_data/
│   └── ...
├── providers/              # Provider-specific tests
├── fixtures/               # Test data and fixtures
├── utils/                  # Testing utilities
└── conftest.py            # Pytest configuration
```

## Naming Conventions

### Directory Names
- **Root level**: `lowercase` (e.g., `docs`, `tests`, `scripts`)
- **Use cases**: `snake_case` (e.g., `security_testing`, `model_documentation`)
- **Python packages**: `snake_case` following PEP 8

### File Names
- **Python files**: `snake_case.py` (e.g., `model_analyzer.py`)
- **Configuration**: `lowercase.ext` (e.g., `requirements.txt`, `setup.py`)
- **Documentation**: `UPPERCASE.md` for important docs, `lowercase.md` for others

### Package Names (setup.py)
- **Use case packages**: `llm-{use_case}` (e.g., `llm-security-testing`)
- **Core packages**: `llm-lab-{component}` (e.g., `llm-lab-core`)

## Configuration Management

### Hierarchy of Configuration
1. **Global**: Root-level config files (`pyproject.toml`, `requirements.txt`)
2. **Use Case**: Use case specific config in their directories
3. **Environment**: `.env` files for secrets and environment-specific settings
4. **User**: `.claude/`, `.taskmaster/` for user/tool specific settings

### Configuration Files Location
```
llm-lab/
├── pyproject.toml          # Project metadata and build config
├── requirements.txt        # Core dependencies
├── mypy.ini               # Type checking configuration
├── pytest.ini            # Testing configuration
├── ruff.toml              # Linting configuration
├── .gitignore             # Version control exclusions
├── .env.example           # Environment variable template
└── src/use_cases/*/
    ├── requirements.txt    # Use case specific dependencies
    ├── setup.py           # Use case package configuration
    └── config/            # Use case configuration files
```

## Import Standards

### Absolute Imports Only
```python
# ✅ Correct
from src.providers.openai import OpenAIProvider
from src.use_cases.security_testing import SecurityScanner
from src.utils.validation import validate_response

# ❌ Incorrect
from ..providers.openai import OpenAIProvider
from .security_testing import SecurityScanner
```

### Use Case Module Exports
Each use case `__init__.py` should export main classes:

```python
# src/use_cases/security_testing/__init__.py
"""
LLM Security Testing Framework - Use Case 9
"""

from .src.attack_library.security.scanner import SecurityScanner
from .src.attack_library.core.library import AttackLibrary

__all__ = ["SecurityScanner", "AttackLibrary"]
```

## Development Workflow Integration

### Adding New Use Cases

1. **Create use case directory**: `src/use_cases/new_feature/`
2. **Follow standard structure**: See "Use Case Module Structure" above
3. **Create documentation**: `docs/guides/USE_CASE_N_HOW_TO.md`
4. **Update overview**: Add to `docs/guides/USE_CASES_OVERVIEW.md`
5. **Add to Task Master**: Create PRD and tasks in `.taskmaster/`

### Code Quality Standards

All code must pass:
- **Ruff**: Linting and formatting
- **MyPy**: Type checking (where configured)
- **Pytest**: All tests passing
- **Pre-commit hooks**: Automated quality checks

## Integration Points

### With Task Master
- Use cases documented in `.taskmaster/docs/prd.txt`
- Implementation tracked in `.taskmaster/tasks/`
- Progress monitored through Task Master commands

### With Claude Code
- Structure documented in `CLAUDE.md`
- Custom commands in `.claude/commands/`
- Tool allowlist in `.claude/settings.json`

### With CI/CD
- All tests run against `src/` structure
- Documentation built from `docs/` structure
- Quality gates enforce structure standards

## Anti-Patterns to Avoid

❌ **Root-level feature directories**
```
my-feature-system/          # Never create these
another-module/
standalone-component/
```

❌ **Inconsistent naming**
```
src/use_cases/SecurityTesting/     # PascalCase - wrong
src/use_cases/security-testing/    # kebab-case - wrong
src/use_cases/securityTesting/     # camelCase - wrong
```

❌ **Mixed import styles**
```python
from src.providers import openai                    # inconsistent
from src.use_cases.security_testing.scanner import  # missing module
from .relative.imports import something             # relative imports
```

❌ **Configuration scattered everywhere**
```
src/use_cases/feature/my_config.ini
another_feature/config.yaml
random_config_file.json                            # at root
```

## Benefits of This Structure

1. **Clear Separation**: Each directory has one clear purpose
2. **Easy Navigation**: Follow the hierarchy to find anything
3. **Scalable Growth**: Add new use cases without restructuring
4. **Tool Integration**: Works seamlessly with Task Master, Claude Code, CI/CD
5. **Documentation Alignment**: Structure matches user journey
6. **Testing Clarity**: Tests mirror source structure
7. **Dependency Management**: Clear separation of dependencies
8. **Import Simplicity**: Predictable import paths

## Maintenance and Evolution

### Regular Structure Audits
- Monthly review for new root-level directories
- Quarterly review of use case organization
- Annual review of overall structure effectiveness

### Structure Enforcement
- Pre-commit hooks validate structure
- CI/CD fails on structure violations
- Code review checklist includes structure compliance

### Evolution Process
1. **Identify Need**: Structure limitation or improvement opportunity
2. **Propose Change**: Document proposed structure modification
3. **Discuss Impact**: Review with team and stakeholders
4. **Implement Migration**: Update structure with proper migration
5. **Update Documentation**: Revise this document and related guides

---

*This document serves as the definitive guide for LLM Lab codebase structure. All contributors should follow these standards to maintain code quality and project maintainability.*
