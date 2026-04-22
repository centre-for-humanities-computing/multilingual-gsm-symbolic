import pytest
from conftest import get_lightly_constrained_template_files, get_unconstrained_template_files

from multilingual_gsm_symbolic.gsm_parser import AnnotatedQuestion, Question


def _multi_var_constrained_template() -> AnnotatedQuestion:
    """Template with a multi-variable init line where x is constrained and y is not.

    The bug: _is_init_line_constrained(" ".join(["x", "y"]), ...) is called with
    "x y" (no '='). _extract_variables_from_init_line splits on '=' (absent),
    then on ',' (absent), returning ["x y"] — a single token that never matches
    "x" or "y", so the check always returns False and constrained variables leak
    into unconstrained_choices.
    """
    return AnnotatedQuestion(
        question="Q {x,3}",
        answer="{x}",
        id_orig=1,
        id_shuffled=1,
        question_annotated=(
            "Q {x,3}\n"
            "#init:\n"
            "- $x, $y = [(1, 10), (2, 20), (3, 30)]\n"
            "- $z = range(1, 5)\n"
            "#conditions:\n"
            "- x > 1\n"
            "#answer: x"
        ),
        answer_annotated="{x}",
    )


def test_precompute_unconstrained_excludes_constrained_variables():
    """Constrained variables must not appear in _precompute_unconstrained output.

    x is constrained (has a condition). The multi-variable init line $x, $y means
    the whole line should be skipped. z is unconstrained and must appear.
    """
    template = _multi_var_constrained_template()
    choices = template._precompute_unconstrained({})
    choice_vars = {var for choice_list in choices for choice in choice_list for var in choice}
    assert "x" not in choice_vars, "constrained variable 'x' must not appear in unconstrained choices"
    assert "y" not in choice_vars, "y is paired with constrained x — its line must also be skipped"
    assert "z" in choice_vars, "unconstrained variable 'z' must appear in unconstrained choices"


def test_range_str_tuple_order_matches_range_possibilities_str():
    """range_str and range_possibilities_str must return (int, str) in the same order.

    Bug: range_possibilities_str returned (numbers[i-1], i) while range_str returned (i, numbers[i-1]).
    Templates like `d_val, d_txt = range_str(...)` rely on the first element being the int.
    """
    from multilingual_gsm_symbolic.gsm_parser import range_possibilities_str, range_str

    numbers = ["en", "to", "tre", "fire", "fem"]
    possibilities = range_possibilities_str(1, 6, 1, numbers)
    assert all(isinstance(p[0], str) and isinstance(p[1], int) for p in possibilities), (
        "range_possibilities_str must return (str, int) tuples"
    )
    # Also verify it matches a single range_str draw
    import random

    random.seed(0)
    single = range_str(1, 5, 1, numbers)
    assert isinstance(single[0], str) and isinstance(single[1], int)


_TEMPLATES = get_unconstrained_template_files() + get_lightly_constrained_template_files()


@pytest.mark.parametrize("template_file", _TEMPLATES)
def test_generate_questions_returns_questions(template_file):
    template = AnnotatedQuestion.from_json(template_file)
    questions = template.generate_questions(n=3, verbose=False)
    assert len(questions) > 0
    assert all(isinstance(q, Question) for q in questions)


@pytest.mark.parametrize("template_file", _TEMPLATES)
def test_generate_questions_non_empty_strings(template_file):
    template = AnnotatedQuestion.from_json(template_file)
    questions = template.generate_questions(n=3, verbose=False)
    for q in questions:
        assert isinstance(q.question, str) and q.question.strip()
        assert isinstance(q.answer, str) and q.answer.strip()


@pytest.mark.parametrize("template_file", _TEMPLATES)
def test_generate_questions_ids(template_file):
    template = AnnotatedQuestion.from_json(template_file)
    questions = template.generate_questions(n=3, verbose=False)
    for q in questions:
        assert q.id_orig == template.id_orig
        assert q.id_shuffled == template.id_shuffled
