from fractions import Fraction

import pytest
from test_chevron_arithmetic import _check_answer

from multilingual_gsm_symbolic._helpers import EVAL_CONTEXT_HELPERS, eval_node, parse_expr, parse_value
from multilingual_gsm_symbolic.load_data import load_data, load_replacements


@pytest.mark.parametrize(
    "template_id, updates, expected",
    [
        (40, {}, 16),
        (
            40,
            {
                "n1": 200,
                "n2": 200,
                "n12": 850,
                "p1": 20,
                "p2": 17,
                "n3": 4,
                "p3": 2,
                "total": 100,
                "discount": Fraction(1, 2),
            },
            10,
        ),
        (
            40,
            {
                "n1": 200,
                "n2": 200,
                "n12": 800,
                "p1": 20,
                "p2": 17,
                "n3": 4,
                "p3": 2,
                "total": 100,
                "discount": Fraction(1, 4),
            },
            25,
        ),
        (45, {"length": 28.749999999999996, "space": 25, "owned": 17, "cost": 40}, 3920),
        (51, {"p1": 26.5, "p2": 12.3, "p3": 14.9}, 687),
        (76, {"n": 5, "w1": 41, "w2": 30, "w3": 21, "price": 0.5}, 67),
    ],
)
def test_danish_saturation_answers(template_id, updates, expected):
    template = next(t for t in load_data("dan") if t.id_shuffled == template_id)
    assignments = {
        k: parse_value(v) for k, v in template._get_full_default_assignments(load_replacements("dan")).items()
    } | updates
    answer = template.format_answer(assignments)
    formula = template.question_annotated.split("#answer:", 1)[1].strip()
    assert eval_node(parse_expr(formula), EVAL_CONTEXT_HELPERS | assignments) == expected
    assert int(answer.split("####")[-1]) == expected
    assert not _check_answer(answer, str(template_id))
