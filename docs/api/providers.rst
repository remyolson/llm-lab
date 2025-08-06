Providers API
=============

This module provides interfaces for interacting with various LLM providers.

Base Provider
-------------

.. automodule:: src.providers.base
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

OpenAI Provider
---------------

.. automodule:: src.providers.openai
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   >>> from src.providers import OpenAIProvider
   >>>
   >>> # Initialize with API key
   >>> provider = OpenAIProvider(
   ...     api_key="sk-...",
   ...     model="gpt-4",
   ...     temperature=0.7
   ... )
   >>>
   >>> # Simple generation
   >>> response = provider.generate("What is machine learning?")
   >>> print(response['content'])
   'Machine learning is a subset of artificial intelligence...'
   >>>
   >>> # Batch generation
   >>> prompts = ["Define AI", "Define ML", "Define DL"]
   >>> responses = provider.batch_generate(prompts)
   >>> for r in responses:
   ...     print(r['content'][:50] + "...")
   'Artificial Intelligence (AI) refers to...'
   'Machine Learning (ML) is a method of...'
   'Deep Learning (DL) is a specialized subset...'

Anthropic Provider
------------------

.. automodule:: src.providers.anthropic
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   >>> from src.providers import AnthropicProvider
   >>>
   >>> # Initialize provider
   >>> provider = AnthropicProvider(
   ...     api_key="sk-ant-...",
   ...     model="claude-3-opus-20240229"
   ... )
   >>>
   >>> # Generate with system message
   >>> response = provider.generate(
   ...     prompt="Analyze this code: print('hello')",
   ...     system_message="You are a code reviewer"
   ... )
   >>> print(response['content'])
   'This is a simple Python print statement...'

Google Provider
---------------

.. automodule:: src.providers.google
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   >>> from src.providers import GoogleProvider
   >>>
   >>> # Initialize provider
   >>> provider = GoogleProvider(
   ...     api_key="...",
   ...     model="gemini-1.5-pro"
   ... )
   >>>
   >>> # Generate with safety settings
   >>> response = provider.generate(
   ...     prompt="Explain neural networks",
   ...     safety_settings={"harassment": "BLOCK_NONE"}
   ... )
   >>> print(response['content'])
   'Neural networks are computational models...'

Local Provider
--------------

.. automodule:: src.providers.local.unified_provider
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   >>> from src.providers.local import UnifiedLocalProvider
   >>>
   >>> # Initialize with Ollama backend
   >>> provider = UnifiedLocalProvider(
   ...     backend="ollama",
   ...     model_name="llama2:7b"
   ... )
   >>>
   >>> # Generate response
   >>> response = provider.generate("What is Python?")
   >>> print(response['content'])
   'Python is a high-level programming language...'

Provider Registry
-----------------

.. automodule:: src.providers.registry
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   >>> from src.providers import ProviderRegistry
   >>>
   >>> # Get available providers
   >>> registry = ProviderRegistry()
   >>> providers = registry.list_providers()
   >>> print(providers)
   ['openai', 'anthropic', 'google', 'local']
   >>>
   >>> # Get provider class
   >>> OpenAIClass = registry.get_provider('openai')
   >>> provider = OpenAIClass(api_key="...")

Provider Exceptions
-------------------

.. automodule:: src.providers.exceptions
   :members:
   :undoc-members:
   :show-inheritance:
