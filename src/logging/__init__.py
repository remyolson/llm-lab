"""
Logging module for LLM Lab

This module provides comprehensive logging functionality for benchmark results.
"""

from .results_logger import CSVResultLogger, ResultRecord

__all__ = ['CSVResultLogger', 'ResultRecord']