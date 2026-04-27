"""Multilingual GSM-Symbolic: A Python package for generating synthetic multilingual math problems from symbolic templates.

The package provides:
- `load_data()`: Load symbolic templates for a given language
- `load_gsm()`: Load concrete problems for a given language
- `load_replacements()`: Load language-specific named values used in templates
- `available_languages()`: List available languages and their sample counts
- `AnnotatedQuestion`: Core class for symbolic templates with generation methods
- `Question`: Dataclass for a single generated problem
- `GSMProblem`: Dataclass for a concrete problem from the GSM dataset
"""

from multilingual_gsm_symbolic.gsm_parser import AnnotatedQuestion, Question
from multilingual_gsm_symbolic.load_data import (
    GSMProblem,
    available_languages,
    load_data,
    load_gsm,
    load_replacements,
)

__all__ = [
    "AnnotatedQuestion",
    "Question",
    "GSMProblem",
    "available_languages",
    "load_data",
    "load_gsm",
    "load_replacements",
]
