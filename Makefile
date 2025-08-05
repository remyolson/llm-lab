# LLM Lab Testing Suite Makefile

.PHONY: help install install-dev install-all test test-unit test-integration test-compatibility test-performance test-security test-all coverage lint format type-check security clean setup docs

# Default target
help:
	@echo "LLM Lab Testing Suite - Available Commands:"
	@echo ""
	@echo "  Setup & Installation:"
	@echo "    install         Install core dependencies"
	@echo "    install-dev     Install development dependencies"
	@echo "    install-all     Install all dependencies (dev, benchmarks, security)"
	@echo "    setup           Setup development environment (install + pre-commit)"
	@echo ""
	@echo "  Testing:"
	@echo "    test            Run all available tests"
	@echo "    test-unit       Run unit tests only"
	@echo "    test-integration Run integration tests (requires API keys)"
	@echo "    test-compatibility Run compatibility tests (requires API keys)"
	@echo "    test-performance Run performance benchmarks (requires API keys)"
	@echo "    test-security   Run security and code quality checks"
	@echo ""
	@echo "  Code Quality:"
	@echo "    lint            Run all linters (flake8, mypy, bandit)"
	@echo "    format          Format code with black and isort"
	@echo "    type-check      Run mypy type checking"
	@echo "    security        Run security analysis"
	@echo ""
	@echo "  Coverage & Reports:"
	@echo "    coverage        Generate coverage report"
	@echo "    coverage-html   Generate HTML coverage report"
	@echo ""
	@echo "  Utilities:"
	@echo "    clean           Clean up generated files"
	@echo "    docs            Generate documentation"
	@echo ""
	@echo "  Environment Variables:"
	@echo "    Set API keys for integration tests:"
	@echo "      export OPENAI_API_KEY=your_key_here"
	@echo "      export ANTHROPIC_API_KEY=your_key_here"
	@echo "      export GOOGLE_API_KEY=your_key_here"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,test]"

install-all:
	pip install -e ".[all]"

setup: install-all
	pre-commit install
	@echo "Development environment setup complete!"

# Testing targets
test:
	pytest tests/unit_providers/ tests/providers/ -v

test-unit:
	pytest tests/unit_providers/ tests/providers/ \
		--cov=llm_providers \
		--cov-report=term-missing \
		--cov-report=html \
		-v

test-integration:
	@echo "Running integration tests (requires API keys)..."
	@if [ -z "$$OPENAI_API_KEY" ] && [ -z "$$ANTHROPIC_API_KEY" ] && [ -z "$$GOOGLE_API_KEY" ]; then \
		echo "Warning: No API keys found. Set at least one API key:"; \
		echo "  export OPENAI_API_KEY=your_key_here"; \
		echo "  export ANTHROPIC_API_KEY=your_key_here"; \
		echo "  export GOOGLE_API_KEY=your_key_here"; \
	fi
	pytest tests/integration/ -v -s --tb=short

test-compatibility:
	@echo "Running compatibility tests (requires API keys)..."
	pytest tests/compatibility/ -v --tb=short

test-performance:
	@echo "Running performance benchmarks (requires API keys)..."
	python tests/performance/demo_performance_suite.py --mode quick

test-security:
	@echo "Running security and code quality checks..."
	bandit -r llm_providers/ -f json -o reports/bandit-report.json || true
	safety check --json --output reports/safety-report.json || true
	semgrep --config=auto --json --output=reports/semgrep-report.json llm_providers/ || true

test-all: test-unit test-integration test-compatibility test-security
	@echo "All tests completed!"

# Code quality targets
lint: 
	@echo "Running linters..."
	ruff check .
	mypy src/ tests/ --ignore-missing-imports --no-strict-optional
	bandit -r src/ --skip B101,B601

format:
	@echo "Formatting code..."
	ruff format .
	ruff check . --fix

type-check:
	mypy src/ tests/ --ignore-missing-imports --no-strict-optional

security:
	@echo "Running security analysis..."
	bandit -r src/ --skip B101,B601
	safety check

# Coverage targets
coverage:
	pytest tests/unit_providers/ tests/providers/ \
		--cov=llm_providers \
		--cov=tests \
		--cov-report=term-missing \
		--cov-report=xml \
		--cov-fail-under=75

coverage-html:
	pytest tests/unit_providers/ tests/providers/ \
		--cov=llm_providers \
		--cov=tests \
		--cov-report=html \
		--cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

# Utility targets
clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage coverage.xml htmlcov/ .pytest_cache/
	rm -rf reports/ benchmark_reports/ compatibility_reports/
	rm -rf .mypy_cache/ .tox/

docs:
	@echo "Documentation generation not implemented yet"
	@echo "This would generate documentation using Sphinx or similar"

# Benchmark and compatibility shortcuts
benchmark-quick:
	python tests/performance/demo_performance_suite.py --mode quick --providers all

benchmark-standard:
	python tests/performance/demo_performance_suite.py --mode standard --providers all

compatibility-test:
	python tests/compatibility/demo_compatibility_suite.py --providers all --format both

# Development workflow shortcuts
dev-setup: setup
	@echo "Creating reports directory..."
	mkdir -p reports
	@echo "Development setup complete! Run 'make test' to verify installation."

dev-test: format lint test-unit
	@echo "Development test suite completed successfully!"

pre-commit: format lint
	@echo "Pre-commit checks passed!"

# CI/CD simulation
ci-test:
	@echo "Simulating CI/CD pipeline..."
	$(MAKE) format
	$(MAKE) lint  
	$(MAKE) test-unit
	$(MAKE) test-security
	@echo "CI/CD simulation completed!"

# Release preparation
release-check: clean install-all ci-test coverage
	@echo "Release checks completed successfully!"
	@echo "Ready for release!"

# Integration with specific providers
test-openai:
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "OPENAI_API_KEY not set"; exit 1; fi
	pytest tests/ -k "openai" -v

test-anthropic:
	@if [ -z "$$ANTHROPIC_API_KEY" ]; then echo "ANTHROPIC_API_KEY not set"; exit 1; fi
	pytest tests/ -k "anthropic" -v

test-google:
	@if [ -z "$$GOOGLE_API_KEY" ]; then echo "GOOGLE_API_KEY not set"; exit 1; fi
	pytest tests/ -k "google" -v