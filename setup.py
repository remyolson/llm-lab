"""
Setup configuration for LLM Lab package.

This file enables pip installation of the package, which is required
for Sphinx autodoc to work properly in the CI/CD pipeline.
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Documentation requirements
docs_requirements = []
docs_requirements_path = Path(__file__).parent / "docs" / "requirements.txt"
if docs_requirements_path.exists():
    with open(docs_requirements_path, "r", encoding="utf-8") as f:
        docs_requirements = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

setup(
    name="llm-lab",
    version="1.0.0",
    author="LLM Lab Team",
    author_email="team@llm-lab.dev",
    description="Comprehensive LLM benchmarking and evaluation framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/llm-lab/llm-lab",
    project_urls={
        "Documentation": "https://llm-lab.readthedocs.io",
        "Bug Tracker": "https://github.com/llm-lab/llm-lab/issues",
        "Source Code": "https://github.com/llm-lab/llm-lab",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
        "docs": docs_requirements,
        "all": requirements + docs_requirements,
    },
    entry_points={
        "console_scripts": [
            "llm-lab=src.scripts.run_benchmarks:main",
            "llm-benchmark=src.scripts.run_benchmarks:main",
            "llm-compare=src.scripts.compare_results:main",
            "llm-config=src.scripts.config_cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
    },
    zip_safe=False,
)
