import math
import re
from pathlib import Path
from random import Random

import pytest

from multilingual_gsm_symbolic._helpers import EVAL_CONTEXT_HELPERS, eval_node, parse_value
from multilingual_gsm_symbolic.load_data import load_replacements
from multilingual_gsm_symbolic.templates import AnnotatedQuestion


def _check_answer(answer, template_id):
    errors = []
    for lhs, rhs in re.findall(r"<<([^<>]+)=([^<>]+)>>", answer):
        assert math.isclose(eval(lhs), float(rhs), abs_tol=1e-6), (template_id, lhs, rhs)
    return errors


ROOT = Path(__file__).parents[1] / "src/multilingual_gsm_symbolic/data/templates/eng/symbolic"
IDS = [
    "0010",
    "0019",
    "0039",
    "0040",
    "0045",
    "0049",
    "0051",
    "0065",
    "0067",
    "0076",
    "0080",
    "0084",
    "0091",
    "0094",
    "0098",
]


@pytest.mark.parametrize("template_id", IDS)
def test_english_answer_arithmetic(template_id):
    template = AnnotatedQuestion.from_toml(ROOT / f"{template_id}.toml")
    replacements = load_replacements("eng")
    defaults = template._get_full_default_assignments(replacements)
    assignments = [defaults]
    rng = Random(42)
    for _ in range(50000):
        values = {}
        for line in template.init:
            values.update(template._evaluate_unconstrained_init_line(line, replacements, rng))
        env = EVAL_CONTEXT_HELPERS | {k: parse_value(v) for k, v in values.items()}
        if all(eval_node(condition, env) for condition in template._condition_asts):
            assignments.append(values)
            if len(assignments) == 31:
                break
    assert len(assignments) == 31
    for values in assignments:
        answer = template.format_answer(values)
        assert not _check_answer(answer, template_id)
        env = EVAL_CONTEXT_HELPERS | {k: parse_value(v) for k, v in values.items()}
        expression = template.question_annotated.split("#answer:")[1].strip()
        assert float(answer.split("####")[-1]) == pytest.approx(eval(expression, {}, env))


@pytest.mark.parametrize("pounds,expected", [(4, 3.5), (5, 0.5)])
def test_repeating_discount(pounds, expected):
    template = AnnotatedQuestion.from_toml(ROOT / "0040.toml")
    values = template.get_default_assignments() | {"n12": pounds}
    assert float(template.format_answer(values).split("####")[-1]) == expected


def test_half_dollar_rounds_up_and_first_price_can_round_down():
    template = AnnotatedQuestion.from_toml(ROOT / "0051.toml")
    values = template.get_default_assignments() | {"p1": 1.26, "p2": 2.5, "p3": 4.5}
    answer = template.format_answer(values)
    assert "marigolds down" in answer
    assert float(answer.split("####")[-1]) == 12 + 9 * 3 + 17 * 5


def test_remaining_fish_each_have_the_stated_weight():
    template = AnnotatedQuestion.from_toml(ROOT / "0076.toml")
    values = template.get_default_assignments() | {"n": 4, "w1": 71, "w2": 31, "w3": 24}
    assert "each remaining" in template.format_question(values)
    assert not _check_answer(template.format_answer(values), "0076")
    assert float(template.format_answer(values).split("####")[-1]) == 75


def test_purchase_comparison_is_explicit():
    template = AnnotatedQuestion.from_toml(ROOT / "0091.toml")
    assert "cannolis he bought at the store" in template.question


def test_quiz_rejects_odd_equal_split():
    template = AnnotatedQuestion.from_toml(ROOT / "0019.toml")
    values = template.get_default_assignments() | {"n": 30, "p1": 10, "r1": 100, "frac_val": "1/3"}
    env = EVAL_CONTEXT_HELPERS | {k: parse_value(v) for k, v in values.items()}
    assert not all(eval_node(condition, env) for condition in template._condition_asts)
    with pytest.raises(IndexError, match="empty sequence"):
        template.generate_questions(n=1, fixed={"n": 30, "p1": 10, "r1": 100}, seed=42, verbose=False)


def test_generator_accepts_integral_quiz_split():
    template = AnnotatedQuestion.from_toml(ROOT / "0019.toml")
    fixed = {"n": 60, "p1": 40, "r1": 75, "frac_val": "1/2"}
    questions = template.generate_questions(n=3, fixed=fixed, seed=42, verbose=False)
    assert len(questions) == 3
    for question in questions:
        assert "60-item quiz, 40%" in question.question
        assert float(question.answer.split("####")[-1]) == 36
        assert not _check_answer(question.answer, "0019")


@pytest.mark.parametrize("pounds,expected", [(4, 3.5), (5, 0.5)])
def test_generator_repeats_discount_and_prices_leftovers(pounds, expected):
    template = AnnotatedQuestion.from_toml(ROOT / "0040.toml")
    fixed = {
        "total": 15,
        "n1": 1,
        "n2": 1,
        "n12": pounds,
        "n3": 4,
        "p1": 3,
        "p2": "1.5",
        "p3": "0.25",
        "discount": "1/2",
    }
    questions = template.generate_questions(n=3, fixed=fixed, seed=42, verbose=False)
    assert len(questions) == 3
    for question in questions:
        assert f"She scooped up {pounds} pounds" in question.question
        assert "complete group of 1 full-price pounds and 1 discounted pounds" in question.question
        assert float(question.answer.split("####")[-1]) == expected
        assert not _check_answer(question.answer, "0040")


def test_generator_preserves_each_remaining_fish_weight():
    template = AnnotatedQuestion.from_toml(ROOT / "0076.toml")
    fixed = {"n": 4, "w1": 71, "w2": 31, "w3": 24, "price": "0.5"}
    questions = template.generate_questions(n=3, fixed=fixed, seed=42, verbose=False)
    assert len(questions) == 3
    for question in questions:
        assert "each remaining" in question.question
        assert float(question.answer.split("####")[-1]) == 75
        assert not _check_answer(question.answer, "0076")
