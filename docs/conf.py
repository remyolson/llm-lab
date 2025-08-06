"""
Configuration file for the Sphinx documentation builder.

This file contains configuration for building comprehensive documentation
with support for autodoc, Google/NumPy docstrings, interactive examples,
and cross-references.

For the full list of built-in configuration values, see:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "LLM Lab"
copyright = f"{datetime.now().year}, LLM Lab Contributors"
author = "LLM Lab Team"
release = "1.0.0"
version = "1.0"

# -- General configuration ---------------------------------------------------

# Extensions to enable
extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",  # Automatic documentation from docstrings
    "sphinx.ext.napoleon",  # Support for Google/NumPy style docstrings
    "sphinx.ext.intersphinx",  # Cross-references to other projects
    "sphinx.ext.viewcode",  # Add links to source code
    "sphinx.ext.doctest",  # Test code snippets in documentation
    "sphinx.ext.duration",  # Show durations of slow builds
    "sphinx.ext.coverage",  # Check documentation coverage
    "sphinx.ext.githubpages",  # GitHub Pages support
    "sphinx.ext.todo",  # Support for TODO items
    "sphinx.ext.mathjax",  # Math rendering support
    # Optional extensions (install if needed)
    # 'jupyter_sphinx',             # Interactive Jupyter examples
    # 'sphinx_copybutton',          # Copy button for code blocks
    # 'sphinx_tabs',                # Tabbed content
    # 'sphinx_design',              # Cards, grids, and other components
    # 'myst_parser',                # Markdown support
]

# Add any paths that contain templates here
templates_path = ["_templates"]

# List of patterns to exclude from documentation
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/__pycache__",
    "**/test_*.py",
    "**/tests/**",
    "**/conftest.py",
]

# The suffix(es) of source filenames
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document
master_doc = "index"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages
html_theme = "sphinx_rtd_theme"  # ReadTheDocs theme (install: pip install sphinx-rtd-theme)
# Alternative themes: 'alabaster', 'classic', 'sphinxdoc', 'nature', 'pyramid'

# Theme options
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "style_nav_header_background": "#2980B9",
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Add any paths that contain custom static files (such as style sheets)
html_static_path = ["_static"]

# Custom sidebar templates
html_sidebars = {
    "**": [
        "globaltoc.html",
        "relations.html",
        "sourcelink.html",
        "searchbox.html",
    ]
}

# Output file base name for HTML help builder
htmlhelp_basename = "llmlab_doc"

# -- Options for autodoc -----------------------------------------------------

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
    "inherited-members": False,
    "private-members": False,
}

# Mock imports for external dependencies during doc build
autodoc_mock_imports = [
    "torch",
    "transformers",
    "numpy",
    "pandas",
    "matplotlib",
    "plotly",
    "gradio",
    "fastapi",
    "pydantic",
    "sqlalchemy",
    "optuna",
    "ray",
    "jupyter",
]

# -- Options for Napoleon (Google/NumPy docstrings) -------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_attr_annotations = True

# -- Options for intersphinx -------------------------------------------------

# Cross-reference configuration for external documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en/", None),
    "pydantic": ("https://docs.pydantic.dev/", None),
    "fastapi": ("https://fastapi.tiangolo.com/", None),
}

# -- Options for doctest -----------------------------------------------------

# Doctest configuration
doctest_global_setup = """
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
"""

doctest_test_doctest_blocks = "default"

# -- Options for todo extension ----------------------------------------------

# Show TODO items in output
todo_include_todos = True

# -- Options for coverage extension ------------------------------------------

# Coverage checking
coverage_ignore_modules = []
coverage_ignore_functions = []
coverage_ignore_classes = []
coverage_ignore_pyobjects = []
coverage_write_headline = True
coverage_skip_undoc_in_source = True

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files
latex_documents = [
    (master_doc, "llmlab.tex", "LLM Lab Documentation", "LLM Lab Team", "manual"),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page
man_pages = [(master_doc, "llmlab", "LLM Lab Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files
texinfo_documents = [
    (
        master_doc,
        "llmlab",
        "LLM Lab Documentation",
        author,
        "llmlab",
        "Comprehensive LLM benchmarking and evaluation framework.",
        "Miscellaneous",
    ),
]

# -- Extension configuration -------------------------------------------------

# Jupyter Sphinx configuration (if enabled)
# jupyter_execute_notebooks = "off"  # Don't execute notebooks during build
# jupyter_sphinx_thebelab_config = {
#     'requestKernel': True,
#     'binderOptions': {
#         'repo': "your-github-repo/llm-lab",
#     },
# }


def setup(app):
    """Custom Sphinx setup."""
    app.add_css_file("custom.css")  # Custom CSS if needed
    app.add_js_file("custom.js")  # Custom JavaScript if needed
