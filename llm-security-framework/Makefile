.PHONY: help install install-dev format lint test clean build docs

help:
	@echo "Available commands:"
	@echo "  make install      Install the package in development mode"
	@echo "  make install-dev  Install with development dependencies"
	@echo "  make format       Format code with black and isort"
	@echo "  make lint         Run linting with flake8 and mypy"
	@echo "  make test         Run test suite"
	@echo "  make clean        Remove build artifacts and cache files"
	@echo "  make build        Build distribution packages"
	@echo "  make docs         Generate documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

format:
	black src/ tests/ --line-length 100
	isort src/ tests/ --profile black --line-length 100

lint:
	flake8 src/ tests/
	mypy src/

test:
	pytest tests/ -v --cov=llm_security --cov-report=term-missing

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

docs:
	cd docs && make html
