from importlib.metadata import version

from multilingual_gsm_symbolic.gsm_parser import AnnotatedQuestion, Question

__version__: str = version("multilingual-gsm-symbolic")
from multilingual_gsm_symbolic.load_data import (
    GSMProblem,
    available_languages,
    load_data,
    load_gsm,
    load_replacements,
)

__all__ = [
    "__version__",
    "AnnotatedQuestion",
    "Question",
    "GSMProblem",
    "available_languages",
    "load_data",
    "load_gsm",
    "load_replacements",
]
