# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-31

### Added
- Initial release of LLM Lab benchmark framework
- Core architecture with modular provider system
- Google Gemini 1.5 Flash provider implementation
- Keyword match evaluation method for truthfulness benchmarks
- CSV results logging with comprehensive tracking
- Example truthfulness benchmark using Cervantes "Don Quixote" dataset
- Comprehensive test suite with 83.61% coverage
- pytest and ruff configuration for code quality
- GitHub Actions CI/CD pipeline
- Full documentation including README and module docstrings
- Configuration management for API keys and settings
- Error handling and graceful degradation
- Asynchronous processing support ready for future providers

### Technical Details
- Python 3.8+ support
- Type hints throughout codebase
- Modular design for easy extension
- Provider abstraction for multiple LLM integrations
- Flexible evaluation framework
- Structured logging and results tracking

### Known Limitations
- Single provider (Google Gemini) implemented in v0.1
- Basic keyword matching evaluation only
- Limited benchmark datasets included

[0.1.0]: https://github.com/username/llm-lab/releases/tag/v0.1
