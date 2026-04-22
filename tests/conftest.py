from pathlib import Path

from multilingual_gsm_symbolic.gsm_parser import AnnotatedQuestion
from multilingual_gsm_symbolic.load_data import _DATA_ROOT


def get_template_files() -> list[Path]:
    template_files = []
    for lang_dir in sorted(_DATA_ROOT.iterdir()):
        symbolic_dir = lang_dir / "symbolic"
        if symbolic_dir.exists():
            for template_file in sorted(symbolic_dir.glob("**/*.json")):
                template_files.append(template_file)
    return template_files


def get_unconstrained_template_files(n: int = 5) -> list[Path]:
    result = []
    for path in get_template_files():
        if not AnnotatedQuestion.from_json(path).constrained_variables:
            result.append(path)
        if len(result) >= n:
            break
    return result


def get_lightly_constrained_template_files(n: int = 3) -> list[Path]:
    """Templates with exactly 2 constrained variables — exercises the constrained
    path without hitting the combinatorial explosion of heavily constrained ones."""
    result = []
    for path in get_template_files():
        if len(AnnotatedQuestion.from_json(path).constrained_variables) == 2:
            result.append(path)
        if len(result) >= n:
            break
    return result
