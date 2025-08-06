from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-interpretability",
    version="0.1.0",
    author="LLM Lab Team",
    author_email="team@llm-lab.io",
    description="Comprehensive interpretability toolkit for Large Language Models",
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
        "Topic :: Scientific/Engineering :: Visualization",
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
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.6.0",
        "plotly>=5.14.0",
        "dash>=2.10.0",
        "dash-bootstrap-components>=1.4.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "captum>=0.6.0",  # PyTorch interpretability library
        "bertviz>=1.4.0",  # Attention visualization
        "seaborn>=0.12.0",
        "jupyter>=1.0.0",
        "ipywidgets>=8.0.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "click>=8.1.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
            "isort>=5.12.0",
        ],
        "notebooks": [
            "jupyterlab>=4.0.0",
            "notebook>=7.0.0",
            "ipykernel>=6.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "interpretability=interpretability.cli:cli",
            "interp-suite=interpretability.cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "interpretability": ["templates/*.html", "static/*.css", "static/*.js"],
    },
    zip_safe=False,
)
