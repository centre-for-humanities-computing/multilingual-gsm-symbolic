import pytest
from conftest import get_template_files

from multilingual_gsm_symbolic._helpers import arange_possibilities
from multilingual_gsm_symbolic.load_data import load_replacements
from multilingual_gsm_symbolic.templates import AnnotatedQuestion
from multilingual_gsm_symbolic.validation import validate_formatting_matches_original


@pytest.mark.parametrize("template_file", get_template_files())
def test_template_formatting_matches_original(template_file):
    annotated_question = AnnotatedQuestion.from_toml(template_file)
    replacements = load_replacements(annotated_question.language)
    validate_formatting_matches_original(annotated_question, replacements, source=template_file)


def make_template(answer_annotated: str) -> AnnotatedQuestion:
    return AnnotatedQuestion(
        question="Q",
        answer="A",
        id_orig=1,
        id_shuffled=1,
        question_annotated="Q\n#init:\n- $x = range(1, 5)\n#conditions:\n- True\n#answer: x",
        answer_annotated=answer_annotated,
    )


def test_format_answer_simple_expression():
    t = make_template("Result is {x+1}.")
    assert t.format_answer({"x": 3}) == "Result is 4."


def test_format_answer_multiple_expressions():
    t = make_template("{a} + {b} = {a+b}")
    assert t.format_answer({"a": 2, "b": 3}) == "2 + 3 = 5"


def test_format_answer_integer_float_coercion():
    t = make_template("Answer: {x/2}")
    assert t.format_answer({"x": 4}) == "Answer: 2"


def test_format_answer_expr_asts_cached():
    t = make_template("Value is {x*2}.")
    _ = t.format_answer({"x": 3})
    assert "x*2" in t._answer_expr_asts


def test_format_answer_repeated_expression():
    t = make_template("{x} and {x} again")
    assert t.format_answer({"x": 5}) == "5 and 5 again"


def test_format_answer_ignores_placeholder_syntax():
    t = make_template("Keep {name,Pat} literal and compute {x+1}.")
    assert t.format_answer({"x": 3}) == "Keep {name,Pat} literal and compute 4."
    assert "x+1" in t._answer_expr_asts
    assert "name,Pat" not in t._answer_expr_asts


def test_arange_no_floating_point_noise():
    """arange_possibilities must not produce strings like '1.7999999999999998'.

    np.linspace(0.2, 4.8, 47) can produce values that numpy float64 renders as
    '1.7999999999999998' instead of '1.8'. These corrupt the question text.
    """

    values = arange_possibilities(0.2, 4.8, 0.1)
    noisy = [v for v in values if len(v.split(".")[-1]) > 10 if "." in v]
    assert not noisy, f"Floating-point noise in arange values: {noisy[:5]}"


def test_format_question_arange_variable_no_noise():
    """A template using arange must not produce '1.7999...' in the rendered question."""
    t = AnnotatedQuestion(
        question="Price is 1.8",
        answer="1.8",
        id_orig=1,
        id_shuffled=1,
        question_annotated=(
            "Price is {unit_price,1.8}\n"
            "#init:\n"
            "- $unit_price = arange(0.2, 4.8, 0.1)\n"
            "#conditions:\n"
            "- True\n"
            "#answer: unit_price"
        ),
        answer_annotated="{unit_price}",
    )
    questions = t.generate_questions(n=50, verbose=False)
    for q in questions:
        assert "1.7999" not in q.question, f"Floating-point noise in question: {q.question}"
        assert "1.7999" not in q.answer, f"Floating-point noise in answer: {q.answer}"


def test_validate_formatting_ignores_whitespace_only_differences():
    template = AnnotatedQuestion(
        question="James writes a 3-page letter twice a week.  How many pages?",
        answer="First line\nSecond line",
        id_orig=1,
        id_shuffled=1,
        question_annotated=(
            "James writes a {pages,3}-page letter twice a week. How many pages?\n"
            "#init:\n"
            "- $pages = range(1, 5)\n"
            "#conditions:\n"
            "- True\n"
            "#answer: pages"
        ),
        answer_annotated="First    line\nSecond line",
    )
    validate_formatting_matches_original(template, replacements={})


def test_validate_formatting_includes_diff_for_mismatch():
    template = AnnotatedQuestion(
        question="Expected question.",
        answer="Expected answer.",
        id_orig=1,
        id_shuffled=1,
        question_annotated=(
            "Rendered {thing,question}.\n"
            "#init:\n"
            "- thing = sample([\"question\"])\n"
            "#conditions:\n"
            "- True\n"
            "#answer: 1"
        ),
        answer_annotated="Expected answer.",
    )

    with pytest.raises(AssertionError) as exc_info:
        validate_formatting_matches_original(template, replacements={})

    message = str(exc_info.value)
    assert "Formatted question doesn't match original" in message
    assert "--- original_question" in message
    assert "+++ formatted_question" in message
    assert "-Expected question." in message
    assert "+Rendered question." in message
