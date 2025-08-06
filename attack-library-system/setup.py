"""Setup script for Attack Library System."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="attack-library-system",
    version="0.1.0",
    author="LLM Lab Team",
    author_email="team@llmlab.com",
    description="Comprehensive attack library and prompt generation system for LLM security testing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/llm-lab/attack-library-system",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Security",
        "Topic :: Software Development :: Testing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "click>=8.1.0",
        "pyyaml>=6.0.1",
        "jinja2>=3.1.2",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "requests>=2.31.0",
        "aiohttp>=3.9.0",
        "rich>=13.0.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
        "jsonschema>=4.17.0",
        "faker>=19.0.0",
        "nltk>=3.8.0",
        "textstat>=0.7.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "ml": [
            "transformers>=4.33.0",
            "torch>=2.0.0",
            "scikit-learn>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "attack-library=attack_library.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "attack_library": [
            "data/attacks/*.json",
            "data/schemas/*.json",
            "templates/*.j2",
        ],
    },
)
