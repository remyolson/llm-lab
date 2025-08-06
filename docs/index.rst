.. LLM Lab documentation master file

Welcome to LLM Lab Documentation
=================================

LLM Lab is a comprehensive framework for benchmarking, evaluating, and comparing Large Language Models (LLMs) across multiple providers. It provides tools for systematic testing, performance monitoring, and fine-tuning of language models.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart
   getting_started/configuration

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/providers
   user_guide/benchmarking
   user_guide/evaluation
   user_guide/monitoring
   user_guide/fine_tuning

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/providers
   api/evaluation
   api/benchmarks
   api/monitoring
   api/fine_tuning
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/basic_usage
   examples/custom_benchmarks
   examples/multi_model_comparison
   examples/fine_tuning_workflow
   examples/monitoring_setup

.. toctree::
   :maxdepth: 1
   :caption: Advanced Topics

   advanced/custom_providers
   advanced/custom_metrics
   advanced/distributed_training
   advanced/deployment

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/contributing
   development/testing
   development/architecture
   development/changelog

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Quick Links
===========

* **Installation**: :doc:`getting_started/installation`
* **Quick Start**: :doc:`getting_started/quickstart`
* **API Documentation**: :doc:`api/providers`
* **Examples**: :doc:`examples/basic_usage`
* **Contributing**: :doc:`development/contributing`

Features
========

Core Capabilities
-----------------

* **Multi-Provider Support**: Seamless integration with OpenAI, Anthropic, Google, and local models
* **Comprehensive Benchmarking**: Standard benchmarks (ARC, GSM8K, HellaSWAG, MMLU, TruthfulQA)
* **Custom Evaluation**: Create and run custom evaluation metrics
* **Fine-Tuning Studio**: Complete fine-tuning workflow with monitoring
* **Performance Monitoring**: Real-time monitoring and alerting
* **Cost Tracking**: Detailed cost analysis and optimization

Code Examples
=============

Basic Provider Usage
--------------------

.. code-block:: python

   >>> from src.providers import OpenAIProvider
   >>>
   >>> # Initialize provider
   >>> provider = OpenAIProvider(api_key="your-api-key")
   >>>
   >>> # Generate text
   >>> response = provider.generate(
   ...     prompt="Explain quantum computing",
   ...     max_tokens=100,
   ...     temperature=0.7
   ... )
   >>> print(response['content'])
   'Quantum computing uses quantum mechanical phenomena...'

Multi-Model Comparison
----------------------

.. code-block:: python

   >>> from src.benchmarks import BenchmarkRunner
   >>> from src.providers import OpenAIProvider, AnthropicProvider
   >>>
   >>> # Set up providers
   >>> providers = [
   ...     OpenAIProvider(model="gpt-4"),
   ...     AnthropicProvider(model="claude-3-opus")
   ... ]
   >>>
   >>> # Run benchmark
   >>> runner = BenchmarkRunner(providers)
   >>> results = runner.run_benchmark("truthfulness")
   >>>
   >>> # Analyze results
   >>> for provider, score in results.items():
   ...     print(f"{provider}: {score:.2%}")
   'OpenAI: 87.50%'
   'Anthropic: 89.25%'

Support
=======

* **GitHub Issues**: `Report bugs or request features <https://github.com/llm-lab/issues>`_
* **Documentation**: `Read the full documentation <https://llm-lab.readthedocs.io>`_
* **Community**: Join our Discord server for discussions

License
=======

LLM Lab is released under the MIT License. See the LICENSE file for details.
