# Standardized Use Case Structure for LLM Lab

## Overview

This document defines the standardized organizational structure for all use cases in the LLM Lab project to maintain consistency, reduce technical debt, and improve user experience.

## Directory Structure Standard

All use case implementations MUST follow this structure:

```
llm-lab/
├── src/use_cases/                    # All use case modules go here
│   ├── {use_case_name}/              # Individual use case module
│   │   ├── __init__.py               # Module exports and documentation
│   │   ├── src/                      # Source code implementation
│   │   ├── setup.py                  # Package configuration
│   │   ├── requirements.txt          # Dependencies
│   │   ├── README.md                 # Module-specific documentation
│   │   ├── .gitignore               # Module-specific git ignore
│   │   └── tests/                    # Unit tests
│   └── ...
├── docs/guides/                      # User-facing how-to guides
│   ├── USE_CASE_{N}_HOW_TO.md        # Numbered use case guides
│   └── USE_CASES_OVERVIEW.md         # Complete overview
└── .taskmaster/docs/                 # Task Master PRDs and planning
    └── use_case_{N}_prd.md           # PRD documents
```

## Use Case Module Organization

### 1. Module Location
- **Location**: `src/use_cases/{use_case_name}/`
- **Naming**: Use snake_case for directory names
- **Examples**:
  - `security_testing` (not `attack-library-system`)
  - `synthetic_data` (not `synthetic-data-platform`)
  - `model_documentation` (not `model-documentation-system`)

### 2. Module Structure
Each use case module MUST contain:

```
use_case_module/
├── __init__.py                       # Main exports and module documentation
├── src/                              # Implementation source code
│   └── {module_name}/                # Main package
│       ├── __init__.py
│       ├── cli.py                    # Command-line interface
│       ├── core/                     # Core functionality
│       ├── models/                   # Data models and types
│       ├── utils/                    # Utility functions
│       └── ...                       # Domain-specific modules
├── setup.py                          # Package setup with standardized naming
├── requirements.txt                  # Python dependencies
├── README.md                         # Module documentation
├── .gitignore                        # Module-specific ignores
└── tests/                            # Unit and integration tests
    ├── __init__.py
    └── test_*.py
```

### 3. Standardized Naming Convention

#### Package Names (setup.py):
- Format: `llm-{use_case_name}`
- Examples:
  - `llm-security-testing`
  - `llm-synthetic-data`
  - `llm-interpretability`
  - `llm-benchmark-creation`

#### CLI Commands:
- Format: `python -m {module_name}.cli {command}`
- Examples:
  - `python -m attack_library.cli scan`
  - `python -m synthetic_data.cli generate`
  - `python -m model_docs.cli generate-card`

## Documentation Structure

### 1. How-To Guides
- **Location**: `docs/guides/USE_CASE_{N}_HOW_TO.md`
- **Numbering**: Sequential numbering starting from USE_CASE_1_HOW_TO.md
- **Format**: Comprehensive guides with examples, costs, troubleshooting

#### Required Sections:
1. **What You'll Accomplish** - Clear outcomes
2. **Before You Begin** - Prerequisites and cost estimates
3. **Setup and Installation** - Quick start instructions
4. **Usage Examples** - Multiple complexity levels
5. **Advanced Features** - Extended functionality
6. **Integration** - Connection with other use cases
7. **Troubleshooting** - Common issues and solutions

### 2. Overview Documentation
- **Location**: `docs/guides/USE_CASES_OVERVIEW.md`
- **Content**: Complete summary of all use cases with status
- **Updates**: Must be updated when new use cases are added

## Implementation Status Classification

### Status Levels:
1. **✅ Implemented**: Complete implementation with documentation
2. **⚠️ Partial**: Basic functionality exists, needs enhancement
3. **❌ Not Started**: Planned but not yet implemented

### Status Requirements:
- **Implemented**: Code, tests, documentation, how-to guide complete
- **Partial**: Basic code exists, documentation may be incomplete
- **Not Started**: Only PRD exists, no implementation

## Migration Guidelines

### For Existing Modules:
1. **Move** from root level to `src/use_cases/{use_case_name}/`
2. **Update** import statements and references
3. **Standardize** package names in setup.py
4. **Create** comprehensive how-to guide
5. **Update** USE_CASES_OVERVIEW.md

### For New Modules:
1. **Start** in correct location: `src/use_cases/{use_case_name}/`
2. **Follow** standardized structure from the beginning
3. **Create** documentation alongside implementation
4. **Test** integration with existing framework

## Quality Standards

### Code Quality:
- Follow existing code style (black, ruff formatting)
- Include comprehensive type hints
- Write unit tests for core functionality
- Include CLI interfaces for main features

### Documentation Quality:
- Clear, actionable how-to guides
- Code examples that actually work
- Cost estimates and prerequisites
- Troubleshooting sections

### Integration Quality:
- Compatible with existing use cases
- Follows established patterns
- Minimal dependencies conflicts
- Standard CLI interface patterns

## Future Development

### New Use Case Checklist:
- [ ] PRD created in `.taskmaster/docs/`
- [ ] Implementation in `src/use_cases/{use_case_name}/`
- [ ] Standardized package naming
- [ ] CLI interface following patterns
- [ ] Comprehensive how-to guide created
- [ ] USE_CASES_OVERVIEW.md updated
- [ ] Integration tested with existing use cases
- [ ] Documentation reviewed and approved

### Maintenance Guidelines:
- Regular review of use case structure adherence
- Update documentation when features are added
- Maintain backward compatibility when possible
- Follow semantic versioning for major changes

## Benefits of This Structure

1. **User Experience**: Consistent interface across all use cases
2. **Maintainability**: Clear organization reduces technical debt
3. **Documentation**: Standardized guides improve usability
4. **Development**: Predictable structure accelerates development
5. **Testing**: Uniform structure simplifies testing strategies
6. **Deployment**: Consistent packaging enables automation

## Enforcement

This structure is enforced through:
- Code review requirements
- Automated testing of structure adherence
- Documentation quality gates
- Task Master integration for new use cases

---

*This document serves as the authoritative guide for use case organization in LLM Lab. All new development must follow these standards, and existing modules should be migrated to comply.*
