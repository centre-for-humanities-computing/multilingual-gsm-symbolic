"""Validate that every <<lhs=rhs>> marker in a template's answer field is arithmetically correct.

Each marker encodes a computation step: the expression on the left of '=' must evaluate
to the number on the right. We delegate the actual check to the shared validation helper
so generated templates and pytest use the same rules.
"""

import pytest
from conftest import get_template_files

from multilingual_gsm_symbolic.templates import AnnotatedQuestion
from multilingual_gsm_symbolic.validation import validate_chevron_arithmetic


@pytest.mark.parametrize("template_file", get_template_files())
def test_chevron_arithmetic(template_file):
    aq = AnnotatedQuestion.from_toml(template_file)
    validate_chevron_arithmetic(aq, source=template_file)


def test_validate_chevron_arithmetic_rejects_incorrect_marker():
    template = AnnotatedQuestion(
        question="Q",
        answer="Broken step <<2+2=5>>",
        id_orig=1,
        id_shuffled=1,
        question_annotated="Q\n#init:\n- $x = range(1, 2)\n#conditions:\n- True\n#answer: x",
        answer_annotated="{x}",
    )

    with pytest.raises(AssertionError, match=r"incorrect <<lhs=rhs>> computations"):
        validate_chevron_arithmetic(template, source="broken.toml")
