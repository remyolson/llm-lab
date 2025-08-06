from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="model-documentation-system",
    version="0.1.0",
    author="LLM Lab Team",
    author_email="team@llm-lab.io",
    description="Automated documentation generation system for machine learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/llm-lab/model-documentation-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Documentation",
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
        "pydantic>=2.0.0",
        "jinja2>=3.1.0",
        "markdown>=3.4.0",
        "reportlab>=4.0.0",
        "GitPython>=3.1.0",
        "PyYAML>=6.0",
        "click>=8.1.0",
        "python-dotenv>=1.0.0",
        "rich>=13.0.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "tensorflow>=2.13.0",
        "onnx>=1.14.0",
        "transformers>=4.30.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.3.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "model-docs=model_docs.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "model_docs": ["templates/*.j2", "templates/*.yaml"],
    },
    zip_safe=False,
)
