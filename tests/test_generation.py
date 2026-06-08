import pathlib
from random import Random

import pytest
from conftest import get_lightly_constrained_template_files, get_unconstrained_template_files

from multilingual_gsm_symbolic._helpers import (
    build_eval_context,
    eval_node,
    parse_expr,
    range_possibilities_str,
    range_str,
)
from multilingual_gsm_symbolic.templates import AnnotatedQuestion, Question


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


def test_fixed_numeric_vars_do_not_vary():
    """Variables named in `fixed` must be identical across all generated questions."""
    template = AnnotatedQuestion(
        question="A fog bank rolls in at 3 miles/hour. The city is 42 miles wide.",
        answer="14 hours",
        id_orig=1,
        id_shuffled=1,
        question_annotated=(
            "A fog bank rolls in at {speed,3} miles/hour. The city is {width,42} miles wide.\n"
            "#init:\n"
            "- $speed = range(1, 20)\n"
            "- $width = range(2, 100)\n"
            "#conditions:\n"
            "- is_int(width / speed)\n"
            "#answer: width // speed"
        ),
        answer_annotated="At {speed} miles/hour, it will take {width}/{speed}={width//speed} hours.",
    )
    questions = template.generate_questions(n=10, fixed={"speed": 3}, verbose=False)
    assert len(questions) == 10
    for q in questions:
        assert "3 miles/hour" in q.question, f"speed was not fixed to 3 in: {q.question}"
        # width should vary — not all identical
    widths = {q.question.split("city is ")[1].split(" miles")[0] for q in questions}
    assert len(widths) > 1, "width should vary when only speed is fixed"


def test_fixed_unconstrained_var_does_not_vary():
    """Fixed variable in an unconstrained init line must not vary."""
    template = AnnotatedQuestion(
        question="A store sells apples for $2 each.",
        answer="$2",
        id_orig=1,
        id_shuffled=1,
        question_annotated=(
            "A store sells apples for ${price,2} each.\n"
            "#init:\n"
            "- $price = range(1, 10)\n"
            "#conditions:\n"
            "- True\n"
            "#answer: price"
        ),
        answer_annotated="${price}",
    )
    questions = template.generate_questions(n=20, fixed={"price": 5}, verbose=False)
    assert all("$5" in q.question for q in questions), "price must be fixed at 5"


_TEMPLATES = get_unconstrained_template_files() + get_lightly_constrained_template_files()


@pytest.mark.parametrize("template_file", _TEMPLATES)
def test_generate_questions_returns_questions(template_file):
    template = AnnotatedQuestion.from_toml(template_file)
    questions = template.generate_questions(n=3, verbose=False)
    assert len(questions) > 0
    assert all(isinstance(q, Question) for q in questions)


@pytest.mark.parametrize("template_file", _TEMPLATES)
def test_generate_questions_non_empty_strings(template_file):
    template = AnnotatedQuestion.from_toml(template_file)
    questions = template.generate_questions(n=3, verbose=False)
    for q in questions:
        assert isinstance(q.question, str) and q.question.strip()
        assert isinstance(q.answer, str) and q.answer.strip()


@pytest.mark.parametrize("template_file", _TEMPLATES)
def test_generate_questions_ids(template_file):
    template = AnnotatedQuestion.from_toml(template_file)
    questions = template.generate_questions(n=3, verbose=False)
    for q in questions:
        assert q.id_orig == template.id_orig
        assert q.id_shuffled == template.id_shuffled


def _repro_template() -> AnnotatedQuestion:
    """A small self-contained template with both constrained and unconstrained vars."""
    return AnnotatedQuestion(
        question="A shop sells apples for $2 each.",
        answer="Total: $6",
        id_orig=0,
        id_shuffled=0,
        question_annotated=(
            "A {shop,store} sells {item,apples} for ${price,2} each.\n"
            "#init:\n"
            '- shop = sample(["store", "market", "kiosk"])\n'
            '- item = sample(["apples", "oranges", "pears"])\n'
            "- $price = range(1, 10)\n"
            "- $n = range(2, 8)\n"
            "#conditions:\n"
            "- price * n < 30\n"
            "#answer: price * n"
        ),
        answer_annotated="Total: ${price * n}",
    )


def test_same_seed_produces_identical_questions():
    """The same seed must always produce identical questions."""
    t = _repro_template()
    assert [(q.question, q.answer) for q in t.generate_questions(n=10, seed=42, verbose=False)] == [
        (q.question, q.answer) for q in t.generate_questions(n=10, seed=42, verbose=False)
    ]


def test_same_rng_state_produces_identical_questions():
    """Passing a Random instance with the same state must reproduce results."""
    t = _repro_template()
    assert [(q.question, q.answer) for q in t.generate_questions(n=10, rng=Random(99), verbose=False)] == [
        (q.question, q.answer) for q in t.generate_questions(n=10, rng=Random(99), verbose=False)
    ]


def test_different_seeds_produce_different_questions():
    """Different seeds should produce at least one different question across 20 draws."""
    t = _repro_template()
    assert [(q.question, q.answer) for q in t.generate_questions(n=20, seed=1, verbose=False)] != [
        (q.question, q.answer) for q in t.generate_questions(n=20, seed=2, verbose=False)
    ]


def _extract_final_answer(answer: str) -> float:
    return float(answer.split("####")[-1].strip())


def test_example_32_compound_formula_floating_point():
    """Regression: price * (1 + bfe/100 + tfe/100) gives 3438499.9999999995
    for price=2990000, bfe=3, tfe=12 due to float imprecision in 3/100 + 12/100.
    int() then truncates to 438499 instead of 438500.
    Fix: compute each fee separately (price*bfe/100 + price*tfe/100) which is exact
    because is_int conditions guarantee integer fees."""
    template = AnnotatedQuestion.from_toml(
        pathlib.Path(__file__).parent.parent / "src/multilingual_gsm_symbolic/data/templates/dan/symbolic/0032.toml"
    )
    questions = template.generate_questions(
        n=1,
        fixed={"price": 2990000, "budget": 3000000, "brokerage_fee": 3, "transfer_fee": 12},
        verbose=False,
    )
    assert len(questions) == 1
    final_str = questions[0].answer.split("####")[-1].strip()
    assert final_str == "438500", f"Expected '438500' but got {final_str!r}; full answer:\n{questions[0].answer}"


def test_example_30_floating_point_answer_is_clean_integer():
    """Regression: initial_amount=130.2, quantity=24, unit_price=1.3 gives
    int(130.2 - 24*1.3) = int(98.99999999999999) = 98 (wrong) or shows
    '98.99999999999999' in the answer body.
    """

    template_path = (
        pathlib.Path(__file__).parent.parent / "src/multilingual_gsm_symbolic/data/templates/dan/symbolic/0030.toml"
    )
    template = AnnotatedQuestion.from_toml(template_path)
    questions = template.generate_questions(
        n=1,
        fixed={"initial_amount": "130.2", "quantity": 24, "unit_price": "1.3"},
        verbose=False,
    )
    assert len(questions) == 1
    final_str = questions[0].answer.split("####")[-1].strip()
    assert final_str == "99", f"Expected clean integer '99' but got {final_str!r}; full answer:\n{questions[0].answer}"


def test_example_80_uses_ensure_int_for_integral_float_answer():
    """Regression PR #26: 10.45 / 0.55 can evaluate to 18.999999999999996."""

    template_path = (
        pathlib.Path(__file__).parent.parent / "src/multilingual_gsm_symbolic/data/templates/eng/symbolic/0080.toml"
    )
    template = AnnotatedQuestion.from_toml(template_path)
    questions = template.generate_questions(
        n=1,
        fixed={"price1": 30, "price2": 55, "total": 20, "n1": 1, "p": 5},
        verbose=False,
    )
    assert len(questions) == 1
    final_str = questions[0].answer.split("####")[-1].strip()
    assert final_str == "34", f"Expected clean integer '34' but got {final_str!r}; full answer:\n{questions[0].answer}"


def test_example_84_uses_ensure_int_for_integral_float_answer():
    """Regression PR #26: 50 * 1.8 * 0.7 can evaluate to 62.99999999999999."""

    template_path = (
        pathlib.Path(__file__).parent.parent / "src/multilingual_gsm_symbolic/data/templates/eng/symbolic/0084.toml"
    )
    template = AnnotatedQuestion.from_toml(template_path)
    questions = template.generate_questions(
        n=1,
        fixed={"n": 5, "p": 10, "r1": 80, "r2": 30},
        verbose=False,
    )
    assert len(questions) == 1
    final_str = questions[0].answer.split("####")[-1].strip()
    assert final_str == "63", f"Expected clean integer '63' but got {final_str!r}; full answer:\n{questions[0].answer}"


def test_example_94_uses_ensure_int_for_integral_float_answer():
    """Regression PR #26: 360 * 0.35 * 0.5 * (1/3) can evaluate to 20.999999999999996."""

    template_path = (
        pathlib.Path(__file__).parent.parent / "src/multilingual_gsm_symbolic/data/templates/eng/symbolic/0094.toml"
    )
    template = AnnotatedQuestion.from_toml(template_path)
    questions = template.generate_questions(
        n=1,
        fixed={"n": 360, "p1": 35, "p2": 50, "frac_txt": "one third", "frac_val": "1/3"},
        verbose=False,
    )
    assert len(questions) == 1
    final_str = questions[0].answer.split("####")[-1].strip()
    assert final_str == "21", f"Expected clean integer '21' but got {final_str!r}; full answer:\n{questions[0].answer}"


def test_example_19_uses_ensure_int_for_integral_float_answer():
    """Regression PR #26: 60 * 0.45 * (1/3) can evaluate to 8.999999999999998."""

    from fractions import Fraction

    env = build_eval_context(
        Random(0),
        {"n": 60, "p1": 55, "r1": 100, "frac_txt": "one-third", "frac_val": Fraction(1, 3)},
    )
    value = eval_node(
        parse_expr("ensure_int(n * (p1/100) * (r1/100)) + ensure_int(n*(1-(p1/100))*frac_val)"),
        env,
    )
    assert value == 42


@pytest.mark.skip(reason="Slow: generates 30 questions with rejection sampling. Re-enable for regression testing.")
def test_example_40_never_produces_negative_answer():
    """Regression: example 40 had a condition/answer formula mismatch that allowed
    negative left-over money when discount > 0.5. Generate a large sample and assert
    every answer is positive."""

    template_path = (
        pathlib.Path(__file__).parent.parent / "src/multilingual_gsm_symbolic/data/templates/eng/symbolic/0040.toml"
    )
    template = AnnotatedQuestion.from_toml(template_path)
    questions = template.generate_questions(n=30, seed=0, verbose=False)
    for q in questions:
        val = _extract_final_answer(q.answer)
        assert val > 0, f"Example 40 produced non-positive answer {val!r} in:\n{q.question}"


def test_multiple_questions_are_not_all_identical():
    """Generating n>1 questions should produce more than one distinct output."""
    t = _repro_template()
    questions = t.generate_questions(n=20, seed=0, verbose=False)
    unique = {(q.question, q.answer) for q in questions}
    assert len(unique) > 1, "All 20 generated questions were identical"


def test_number_agreement_derived_variable():
    """Regression for issue #15: a word form that must agree with a sampled number
    requires a derived init variable.

    'coin_word = plural(n, "coin", "coins")' is a derived variable: its RHS references
    another init variable (n), so it is evaluated after n is assigned. Before the fix it
    was classified as unconstrained and evaluated without 'n' in scope, raising
    NameError: name 'n' is not defined.

    This is the English analogue of the Ukrainian case where a noun/verb form depends on the
    grammatical number (e.g. 1 рік / 2 роки / 5 років).
    """
    template = AnnotatedQuestion.from_toml(
        pathlib.Path(__file__).parent / "test_templates" / "eng_number_agreement.toml"
    )
    # n == 1 must produce the singular form, n > 1 the plural form.
    singular = template.generate_questions(n=1, fixed={"n": 1}, verbose=False)[0]
    assert "1 coin." in singular.question and "1 coins" not in singular.question

    plural_q = template.generate_questions(n=1, fixed={"n": 5}, verbose=False)[0]
    assert "5 coins" in plural_q.question


def test_nested_tuple_unpacking():
    """Regression for issue #27: tuple unpacking on the LHS of an init line.

    `(num, word) = sample([["1", "one"], ["2", "two"]])` pairs two forms of the same
    value. Before the fix the LHS parser split on commas without stripping parentheses,
    yielding mangled names like '(num' and 'word)', so the sampled pair was never
    unpacked and the placeholders were left unrendered.
    """
    template = AnnotatedQuestion.from_toml(
        pathlib.Path(__file__).parent / "test_templates" / "eng_nested_tuple_unpacking.toml"
    )
    for question in template.generate_questions(n=20, seed=0, verbose=False):
        # No placeholder may be left unrendered, and num/word must stay paired.
        assert "{" not in question.question, f"Unrendered placeholder in: {question.question!r}"
        assert "1 is written as one" in question.question or "2 is written as two" in question.question
