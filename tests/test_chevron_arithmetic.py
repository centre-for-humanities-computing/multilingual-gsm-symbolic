"""Validate that every <<lhs=rhs>> marker in a template's answer field is arithmetically correct.

Each marker encodes a computation step: the expression on the left of '=' must evaluate
to the number on the right.  We parse the stored answer field (which equals the rendered
output for the default variable assignments) and check every such marker.

Locale-specific decimal commas (e.g. '1,5') are normalised to dots before evaluation.
After the thousands-separator cleanup, no <<...>> region should contain a period used as
a thousands separator, so replacing ',' with '.' is unambiguous.

Only purely arithmetic expressions (digits, +, -, *, /, parentheses, dots) are checked;
markers that contain variables or other text are skipped.
"""

import math
import re

import pytest
from conftest import get_template_files

from multilingual_gsm_symbolic.gsm_parser import AnnotatedQuestion

_RE_CHEVRON = re.compile(r"<<([^>]+)>>")
_RE_PURE_ARITHMETIC = re.compile(r"^[\d+\-*/().\s]+$")


def _check_answer(answer: str, template_name: str) -> list[str]:
    errors = []
    for inner in _RE_CHEVRON.findall(answer):
        eq_idx = inner.rfind("=")
        if eq_idx == -1:
            continue
        lhs_raw, rhs_raw = inner[:eq_idx], inner[eq_idx + 1:]

        # Normalise locale decimal commas to Python dots.
        lhs = lhs_raw.replace(",", ".")
        rhs = rhs_raw.replace(",", ".")

        if not _RE_PURE_ARITHMETIC.match(lhs) or not _RE_PURE_ARITHMETIC.match(rhs):
            continue  # skip expressions with variables or non-arithmetic content

        try:
            computed = eval(lhs)  # noqa: S307
            expected = float(rhs)
        except Exception:
            continue

        if not math.isclose(computed, expected, rel_tol=1e-6, abs_tol=1e-9):
            errors.append(
                f"  <<{lhs_raw}={rhs_raw}>>: computed {computed}, expected {expected}"
            )
    return errors


@pytest.mark.parametrize("template_file", get_template_files())
def test_chevron_arithmetic(template_file):
    aq = AnnotatedQuestion.from_json(template_file)
    errors = _check_answer(aq.answer, template_file.name)
    assert not errors, (
        f"{template_file.name} has incorrect <<lhs=rhs>> computations:\n"
        + "\n".join(errors)
    )
