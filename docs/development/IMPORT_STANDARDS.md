# Import Standards and Guidelines

This document outlines the import standards and conventions used in the LLM Lab project.

## Overview

The LLM Lab project follows strict import standards to ensure code consistency, maintainability, and readability. These standards are enforced through automated tools and pre-commit hooks.

## Import Ordering (PEP 8)

All import statements must follow PEP 8 ordering:

1. **Standard library imports** - Python built-in modules
2. **Third-party imports** - External packages
3. **First-party imports** - Project modules (src, tests, benchmarks, scripts)
4. **Local imports** - Relative imports within the same package

### Example

```python
# Standard library
import os
import sys
from typing import Dict, List, Optional

# Third-party
import click
import pydantic
import requests

# First-party
from src.providers.base import LLMProvider
from src.config.settings import Settings

# Local (relative)
from .exceptions import ProviderError
from .utils import format_response
```

## Import Patterns

### ✅ Preferred Patterns

**Within `src/` directory - Use relative imports:**
```python
# Good - relative imports within src/
from .base import LLMProvider
from .exceptions import ProviderError
from ..config import Settings
```

**Outside `src/` directory - Use absolute imports:**
```python
# Good - absolute imports from outside src/
from src.providers.anthropic import AnthropicProvider
from src.evaluation.metrics import calculate_accuracy
```

**Explicit imports:**
```python
# Good - explicit imports
from typing import Dict, List, Optional
from .fixtures import (
    mock_anthropic_provider,
    mock_google_provider,
    sample_evaluation_data,
)
```

### ❌ Patterns to Avoid

**src. prefix imports (deprecated):**
```python
# Bad - avoid src. prefix
from src.providers.base import LLMProvider  # Within src/
```

**Wildcard imports:**
```python
# Bad - wildcard imports
from .fixtures import *
from typing import *
```

**Mixed import styles:**
```python
# Bad - inconsistent import styles
import src.config
from src.providers.base import LLMProvider
from .exceptions import ProviderError
```

## Configuration

### pyproject.toml

The project uses ruff for import formatting with isort compatibility:

```toml
[tool.ruff.lint.isort]
known-first-party = ["src", "tests", "benchmarks", "scripts", "llm_providers"]
sections = ["FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"]
multi-line-output = 3
include-trailing-comma = true
force-grid-wrap = 0
combine-as-imports = true
split-on-trailing-comma = true
```

### Pre-commit Hooks

Import standards are enforced through pre-commit hooks:

```yaml
- repo: https://github.com/astral-sh/ruff-pre-commit
  hooks:
    - id: ruff
      args: [--fix, --select=I,F401]  # Focus on imports and unused imports
    - id: ruff-format
```

## Automated Tools

### Import Refactoring Script

Use `scripts/fix_imports.py` to automatically fix import patterns:

```bash
# Fix all files
python scripts/fix_imports.py

# Dry run to preview changes
python scripts/fix_imports.py --dry-run

# Fix specific file
python scripts/fix_imports.py --file src/providers/base.py
```

### Import Audit Script

Use `scripts/audit_imports.py` to audit import patterns:

```bash
# Run audit
python scripts/audit_imports.py

# View summary
cat import_audit_report_summary.md
```

## Guidelines by File Location

### Within `src/` Directory

- **Use relative imports** for same-package references
- **Avoid `src.` prefixes** completely
- **Use `..` for parent package imports**

```python
# In src/providers/anthropic.py
from .base import LLMProvider          # Same package
from .exceptions import ProviderError  # Same package
from ..config import Settings          # Parent package
```

### In `tests/` Directory

- **Use absolute imports** for src modules
- **Use relative imports** for test utilities
- **Explicit imports** for fixtures

```python
# In tests/test_providers.py
from src.providers.anthropic import AnthropicProvider  # Absolute
from .fixtures import mock_anthropic_provider          # Relative test util
```

### In `examples/` Directory

- **Use absolute imports** for all src modules
- **Prefer explicit imports** for clarity

```python
# In examples/quick_start.py
from src.providers import AnthropicProvider, GoogleProvider
from src.config.settings import Settings
```

### In `scripts/` Directory

- **Use absolute imports** for src modules
- **Import only what's needed**

```python
# In scripts/run_benchmarks.py
from src.benchmarks.runner import BenchmarkRunner
from src.providers.registry import get_provider_for_model
```

## Import Validation

### Automated Checks

1. **Pre-commit hooks** - Run on every commit
2. **Import audit** - Run periodically to catch issues
3. **CI/CD validation** - Ensures standards in pull requests

### Manual Validation

Run these commands to validate imports manually:

```bash
# Check import formatting
python -m ruff check --select I src/ tests/

# Fix import issues automatically
python -m ruff check --fix --select I src/ tests/

# Run comprehensive audit
python scripts/audit_imports.py
```

## Migration Guide

### From Legacy Import Patterns

If you encounter old import patterns, use the automated tools:

1. **Run the fix script:**
   ```bash
   python scripts/fix_imports.py
   ```

2. **Verify changes:**
   ```bash
   python scripts/audit_imports.py
   ```

3. **Test imports work:**
   ```bash
   python -c "import src.providers.base; print('Imports OK')"
   ```

### Common Fixes

| Old Pattern | New Pattern | Context |
|-------------|-------------|---------|
| `from src.providers.base import LLMProvider` | `from .base import LLMProvider` | Within src/ |
| `from .fixtures import *` | `from .fixtures import mock_provider, test_data` | Anywhere |
| `import src.config` | `from .config import Settings` | Within src/ |

## Troubleshooting

### Common Issues

**ImportError: No module named 'src'**
- Solution: Use relative imports within src/, absolute imports outside

**Circular import errors**
- Solution: Restructure modules or use late imports

**Pre-commit hook failures**
- Solution: Run `python scripts/fix_imports.py` before committing

### Getting Help

1. Run the import audit: `python scripts/audit_imports.py`
2. Check the generated report: `import_audit_report_summary.md`
3. Use the fix script: `python scripts/fix_imports.py`
4. Refer to PEP 8 import guidelines

## Current Status

As of the last audit (August 2025):

- **✅ 276 src. prefix imports** - Fixed
- **✅ 5 wildcard imports** - Fixed
- **✅ Import ordering** - Configured in tools
- **✅ Pre-commit hooks** - Active
- **📊 Remaining issues:** 8 (down from 289)

## References

- [PEP 8 - Import Guidelines](https://pep8.org/#imports)
- [Ruff Import Rules (I)](https://docs.astral.sh/ruff/rules/#isort-i)
- [Python Import System](https://docs.python.org/3/reference/import.html)
