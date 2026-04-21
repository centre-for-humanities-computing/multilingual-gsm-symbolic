from multilingual_gsm_symbolic.gsm_parser import AnnotatedQuestion
from multilingual_gsm_symbolic.load_data import _DATA_ROOT


def get_template_files():
    template_dirs = [
        (_DATA_ROOT / "dan" / "symbolic", "dan"),
        (_DATA_ROOT / "eng" / "symbolic", "eng"),
    ]
    template_files = []
    for template_dir, language in template_dirs:
        if template_dir.exists():
            for template_file in sorted(template_dir.glob("**/*.json")):
                template_files.append((template_file, language))
    return template_files


def get_unconstrained_template_files(n: int = 5):
    result = []
    for path, lang in get_template_files():
        if not AnnotatedQuestion.from_json(path).constrained_variables:
            result.append((path, lang))
        if len(result) >= n:
            break
    return result


def get_lightly_constrained_template_files(n: int = 3):
    """Templates with exactly 2 constrained variables — exercises the constrained
    path without hitting the combinatorial explosion of heavily constrained ones."""
    result = []
    for path, lang in get_template_files():
        if len(AnnotatedQuestion.from_json(path).constrained_variables) == 2:
            result.append((path, lang))
        if len(result) >= n:
            break
    return result
