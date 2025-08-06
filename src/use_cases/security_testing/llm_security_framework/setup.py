from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-security-framework",
    version="0.1.0",
    author="LLM Lab Team",
    author_email="team@llm-lab.io",
    description="A comprehensive framework for testing and evaluating LLM security",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/llm-lab/llm-lab",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Testing",
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
        "requests>=2.31.0",
        "aiohttp>=3.9.0",
        "pydantic>=2.0.0",
        "click>=8.1.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
        "rich>=13.0.0",
        "httpx>=0.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.4.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "autodoc>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-security=llm_security.cli.main:main",
            "llm-scan=llm_security.cli.scanner:scan",
            "llm-redteam=llm_security.cli.redteam:redteam",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
