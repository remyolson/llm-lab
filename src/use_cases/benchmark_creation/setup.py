"""Setup configuration for Benchmark Creation Platform."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-benchmark-creation",
    version="0.1.0",
    author="LLM Lab Team",
    author_email="team@llm-lab.io",
    description="Comprehensive platform for creating and managing LLM benchmarks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/llm-lab/llm-lab",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "datasets>=2.14.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pydantic>=2.0.0",
        "jsonschema>=4.0.0",
        "click>=8.0.0",
        "rich>=13.0.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "gitpython>=3.1.0",
        "pyarrow>=12.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
        ],
        "nlp": [
            "nltk>=3.8.0",
            "spacy>=3.6.0",
            "transformers>=4.30.0",
        ],
        "generation": [
            "openai>=1.0.0",
            "anthropic>=0.7.0",
            "google-generativeai>=0.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "benchmark-builder=benchmark_builder.cli:cli",
            "bench-build=benchmark_builder.cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "benchmark_builder": [
            "templates/*.yaml",
            "templates/*.json",
            "config/*.yaml",
        ],
    },
)
