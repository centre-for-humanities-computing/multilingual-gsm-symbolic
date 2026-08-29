import difflib
import math
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

from multilingual_gsm_symbolic._helpers import (
    EVAL_CONTEXT_HELPERS,
    try_parse_float,
    try_parse_fraction,
)
from multilingual_gsm_symbolic.templates import AnnotatedQuestion

_RE_CHEVRON = re.compile(r"<<([^>]+)>>")
_RE_PURE_ARITHMETIC = re.compile(r"^[\d+\-*/().\s]+$")
_RE_WHITESPACE = re.compile(r"\s+")
_DEVANAGARI_TO_ASCII = str.maketrans("०१२३४५६७८९", "0123456789")


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace so validation ignores spacing-only differences."""
    return _RE_WHITESPACE.sub(" ", text).strip()


def _format_text_diff(expected: str, rendered: str, *, label: str) -> str:
    diff_lines = list(
        difflib.unified_diff(
            expected.splitlines(),
            rendered.splitlines(),
            fromfile=f"original_{label}",
            tofile=f"formatted_{label}",
            lineterm="",
        )
    )
    if not diff_lines:
        return f"No line-level diff available for {label}."
    return "\n".join(diff_lines)


def validate_minimum_numeric_combinations(
    annotated_question: AnnotatedQuestion,
    replacements: dict[str, Any],
    minimum: int = 100,
    source: str | Path | None = None,
) -> None:
    source_name = Path(source).name if source is not None else f"template {annotated_question.id_shuffled}"
    combinations = annotated_question.get_combinations(replacements=replacements, only_numeric=True, limit=minimum)
    if len(combinations) < minimum:
        raise AssertionError(
            f"Template has only {len(combinations)} numeric combinations, fewer than {minimum}, for {source_name}"
        )


def validate_maximum_numeric_combinations(
    annotated_question: AnnotatedQuestion,
    replacements: dict[str, Any],
    maximum: int = 100_000,
    source: str | Path | None = None,
) -> None:
    source_name = Path(source).name if source is not None else f"template {annotated_question.id_shuffled}"
    combinations = annotated_question.get_combinations(replacements=replacements, only_numeric=True, limit=maximum + 1)
    if len(combinations) > maximum:
        raise AssertionError(
            f"Template has more than {maximum} numeric combinations for {source_name}"
        )


def validate_formatting_matches_original(
    annotated_question: AnnotatedQuestion,
    replacements: dict[str, Any],
    source: str | Path | None = None,
    fidelity: str = "surface",
) -> None:
    """Check that rendering the template at its defaults reproduces the stored text.

    fidelity="surface": both question and answer must match the stored originals.
    fidelity="answer": the question must match; a mismatching answer is accepted by
    rebasing the stored answer onto the rendered template output ("og answer based
    on the template"). This mutates ``annotated_question.answer`` in place.
    """
    if fidelity not in {"surface", "answer"}:
        raise ValueError(f"Unknown fidelity mode: {fidelity}")
    source_name = Path(source).name if source is not None else f"template {annotated_question.id_shuffled}"
    default_assignments = annotated_question._get_full_default_assignments(replacements)

    formatted_question = annotated_question.format_question(default_assignments)
    formatted_answer = annotated_question.format_answer(default_assignments)

    if _normalize_whitespace(formatted_question) != _normalize_whitespace(annotated_question.question):
        raise AssertionError(
            f"Formatted question doesn't match original for {source_name}\n"
            f"{_format_text_diff(annotated_question.question, formatted_question, label='question')}"
        )
    if _normalize_whitespace(formatted_answer) != _normalize_whitespace(annotated_question.answer):
        if fidelity == "surface":
            raise AssertionError(
                f"Formatted answer doesn't match original for {source_name}\n"
                f"{_format_text_diff(annotated_question.answer, formatted_answer, label='answer')}"
            )
        annotated_question.answer = formatted_answer


def validate_default_assignments(
    annotated_question: AnnotatedQuestion,
    replacements: dict[str, Any],
    source: str | Path | None = None,
) -> None:
    source_name = Path(source).name if source is not None else f"template {annotated_question.id_shuffled}"
    default_assignments = annotated_question._get_full_default_assignments(replacements)
    constrained_lines = annotated_question.constrained_lines
    conditions = annotated_question.conditions

    if not constrained_lines:
        return

    all_possible_assignments = annotated_question._get_all_possible_assignments(constrained_lines, replacements)

    for var_name, possible_assignments in all_possible_assignments.items():
        if var_name not in default_assignments:
            continue
        possible_values_for_var = [assignment[var_name] for assignment in possible_assignments]
        possible_values_for_var = [
            str(value).translate(_DEVANAGARI_TO_ASCII)
            if not isinstance(value, (int, float, Fraction))
            else value
            for value in possible_values_for_var
        ]
        default_value = default_assignments[var_name]

        if isinstance(default_value, tuple):
            default_value = tuple(int(c) if str(c).isnumeric() else str(c) for c in default_value)
            if default_value not in possible_values_for_var and list(default_value) not in possible_values_for_var:
                raise AssertionError(
                    f"Example assignment {var_name}={default_value} not found in "
                    f"{possible_values_for_var} for {source_name}"
                )
        else:
            val_as_float = try_parse_float(str(default_value))
            val_as_fraction = try_parse_fraction(str(default_value))
            val_as_int = (
                int(default_value)
                if str(default_value).isnumeric() or isinstance(default_value, float) and default_value.is_integer()
                else default_value
            )

            if not (
                val_as_float in possible_values_for_var
                or str(val_as_float) in possible_values_for_var
                or val_as_fraction in possible_values_for_var
                or str(val_as_fraction) in possible_values_for_var
                or val_as_int in possible_values_for_var
            ):
                raise AssertionError(
                    f"Example assignment {var_name}={default_value} not found in "
                    f"{possible_values_for_var} for {source_name}"
                )

    if not conditions or all(cond.strip() == "True" for cond in conditions):
        return

    example_combination = {}
    for var_name in all_possible_assignments.keys():
        if var_name in default_assignments:
            default_value = default_assignments[var_name]
            if isinstance(default_value, tuple):
                numeric_val = None
                for component in default_value:
                    try:
                        numeric_val = float(component) if "." in str(component) else int(component)
                        break
                    except (ValueError, TypeError):
                        continue
                example_combination[var_name] = (
                    var_name,
                    numeric_val if numeric_val is not None else default_value[0],
                )
            else:
                example_combination[var_name] = (var_name, default_value)

    for cond in conditions:
        if cond.strip() == "True":
            continue

        temp_combination = example_combination | {
            k: v[1] for k, v in example_combination.items() if isinstance(v, tuple)
        }
        try:
            condition_result = eval(
                cond.translate(_DEVANAGARI_TO_ASCII),
                {"__builtins__": {}},
                EVAL_CONTEXT_HELPERS | temp_combination,
            )
            assert condition_result, f"Example assignments {default_assignments} failed condition '{cond}' for {source_name}"
        except Exception:
            pass


def _chevron_arithmetic_errors(answer: str) -> list[str]:
    errors = []
    for inner in _RE_CHEVRON.findall(answer):
        eq_idx = inner.rfind("=")
        if eq_idx == -1:
            continue
        lhs_raw, rhs_raw = inner[:eq_idx], inner[eq_idx + 1 :]

        # Normalise locale decimal commas and Devanagari digits for Python.
        lhs = lhs_raw.replace(",", ".").translate(_DEVANAGARI_TO_ASCII)
        rhs = rhs_raw.replace(",", ".").translate(_DEVANAGARI_TO_ASCII)

        if not _RE_PURE_ARITHMETIC.match(lhs) or not _RE_PURE_ARITHMETIC.match(rhs):
            continue

        try:
            computed = eval(lhs, {"__builtins__": {}}, {})  # noqa: S307
            expected = float(rhs)
        except Exception:
            continue

        if not math.isclose(computed, expected, rel_tol=1e-6, abs_tol=1e-9):
            errors.append(f"  <<{lhs_raw}={rhs_raw}>>: computed {computed}, expected {expected}")
    return errors


def validate_chevron_arithmetic(
    annotated_question: AnnotatedQuestion,
    source: str | Path | None = None,
) -> None:
    source_name = Path(source).name if source is not None else f"template {annotated_question.id_shuffled}"
    errors = _chevron_arithmetic_errors(annotated_question.answer)
    if errors:
        raise AssertionError(
            f"{source_name} has incorrect <<lhs=rhs>> computations:\n" + "\n".join(errors)
        )


def validate_template_against_pytest_checks(
    annotated_question: AnnotatedQuestion,
    replacements: dict[str, Any],
    source: str | Path | None = None,
    fidelity: str = "surface",
) -> None:
    """Run the per-template checks that pytest applies to active templates."""
    validate_default_assignments(annotated_question, replacements, source=source)
    validate_minimum_numeric_combinations(annotated_question, replacements, source=source)
    validate_maximum_numeric_combinations(annotated_question, replacements, source=source)
    validate_chevron_arithmetic(annotated_question, source=source)
    validate_formatting_matches_original(annotated_question, replacements, source=source, fidelity=fidelity)
