"""Setup configuration for LLM Security Testing Framework."""

from setuptools import find_packages, setup

setup(
    name="llm-security-testing",
    version="1.0.0",
    description="Comprehensive security testing framework for Large Language Models",
    author="LLM Lab Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "rich>=10.0.0",
        "pydantic>=2.0.0",
        "requests>=2.25.0",
        "aiohttp>=3.8.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "pytest>=7.0.0",
        "pytest-asyncio>=0.18.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ],
        "reporting": [
            "reportlab>=3.6.0",  # For PDF generation
            "jinja2>=3.0.0",  # For HTML templates
            "matplotlib>=3.5.0",  # For charts
            "seaborn>=0.11.0",  # For advanced visualizations
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-security=cli:cli",
            "llm-scan=cli:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
