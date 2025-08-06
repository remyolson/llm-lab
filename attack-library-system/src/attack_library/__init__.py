"""Attack Library System - Comprehensive attack prompt library for LLM security testing."""

from .core.library import AttackLibrary
from .core.models import Attack, AttackCategory, AttackMetadata, AttackSeverity
from .core.search import SearchEngine, SearchFilter

try:
    from .generators.base import AttackGenerator
    from .generators.prompt_generator import PromptGenerator, PromptMutator, PromptTemplate
    from .generators.variants import VariantGenerator
except ImportError:
    # Generators are optional for basic functionality
    AttackGenerator = None
    VariantGenerator = None
    PromptGenerator = None
    PromptTemplate = None
    PromptMutator = None

__version__ = "0.1.0"

__all__ = [
    # Core models
    "Attack",
    "AttackMetadata",
    "AttackCategory",
    "AttackSeverity",
    # Main library
    "AttackLibrary",
    # Search functionality
    "SearchEngine",
    "SearchFilter",
    # Generation
    "AttackGenerator",
    "VariantGenerator",
    "PromptGenerator",
    "PromptTemplate",
    "PromptMutator",
]
