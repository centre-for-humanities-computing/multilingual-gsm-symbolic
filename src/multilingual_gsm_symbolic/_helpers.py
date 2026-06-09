"""Pure helper functions for template parsing, evaluation, sampling, and number formatting.

Extracted from ``templates``; these have no dependency on the ``AnnotatedQuestion`` /
``Question`` dataclasses and are safe to import anywhere.
"""

import ast
import decimal
import itertools
import operator as _operator
import re
from fractions import Fraction
from random import Random
from typing import Any

import numpy as np

_BINOPS: dict[type, Any] = {
    ast.Add: _operator.add,
    ast.Sub: _operator.sub,
    ast.Mult: _operator.mul,
    ast.Div: _operator.truediv,
    ast.FloorDiv: _operator.floordiv,
    ast.Mod: _operator.mod,
    ast.Pow: _operator.pow,
}
_CMPOPS: dict[type, Any] = {
    ast.Eq: _operator.eq,
    ast.NotEq: _operator.ne,
    ast.Lt: _operator.lt,
    ast.LtE: _operator.le,
    ast.Gt: _operator.gt,
    ast.GtE: _operator.ge,
}


def eval_node(node: ast.expr, env: dict[str, Any]) -> Any:
    """Evaluate a restricted Python AST node against an environment dict.

    Supports the subset of Python used in template init lines, conditions,
    and answer expressions: constants, name lookups, lists/tuples, arithmetic,
    comparisons, boolean operators, and function calls.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise NameError(f"name '{node.id}' is not defined")
        return env[node.id]
    if isinstance(node, ast.List):
        return [eval_node(e, env) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(eval_node(e, env) for e in node.elts)
    if isinstance(node, ast.BinOp):
        op_fn = _BINOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        return op_fn(eval_node(node.left, env), eval_node(node.right, env))
    if isinstance(node, ast.UnaryOp):
        val = eval_node(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return +val
        if isinstance(node.op, ast.Not):
            return not val
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    if isinstance(node, ast.Call):
        func = eval_node(node.func, env)
        args = [eval_node(a, env) for a in node.args]
        kwargs = {kw.arg: eval_node(kw.value, env) for kw in node.keywords if kw.arg is not None}
        return func(*args, **kwargs)
    if isinstance(node, ast.Compare):
        left = eval_node(node.left, env)
        for op, comparator in zip(node.ops, node.comparators):
            op_fn = _CMPOPS.get(type(op))
            if op_fn is None:
                raise ValueError(f"Unsupported comparison: {type(op).__name__}")
            right = eval_node(comparator, env)
            if not op_fn(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(eval_node(v, env) for v in node.values)
        if isinstance(node.op, ast.Or):
            return any(eval_node(v, env) for v in node.values)
        raise ValueError(f"Unsupported bool operator: {type(node.op).__name__}")
    if isinstance(node, ast.IfExp):
        return eval_node(node.body if eval_node(node.test, env) else node.orelse, env)
    raise ValueError(f"Unsupported AST node type: {type(node).__name__}")


def parse_expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def is_int(value: Any) -> bool:
    return (
        isinstance(value, int)
        or (isinstance(value, float) and (value.is_integer() or abs(value - round(value)) < 1e-6))
        or (isinstance(value, Fraction) and value.denominator == 1)
    )


def divides(a: int | float, b: int | float) -> bool:
    if b == 0:
        return False
    return a % b == 0


def _make_sample(rng: Random):
    def sample(items: list, n: int = 1) -> Any:
        if n == 1:
            return rng.choice(items)
        return rng.sample(items, n)

    return sample


def _make_range_sample(rng: Random):
    def range_sample(start: int, end: int, step: int = 1) -> int:
        if start > end:
            raise ValueError(f"Start ({start}) must be less than or equal to end ({end}).")
        return rng.choice(list(range(start, end + 1, step)))

    return range_sample


def _make_range_str(rng: Random):
    def range_str(start: int, end: int, step: int, numbers: list) -> tuple:
        if start > end:
            return ()
        candidates = [(numbers[i - 1], i) for i in range(start, end + 1, step) if 0 < i <= len(numbers)]
        return rng.choice(candidates)

    return range_str


def _make_sample_sequential(rng: Random):
    def sample_sequential(items: list, n: int) -> list:
        start_idx = rng.randint(0, len(items) - 1)
        return [items[(start_idx + i) % len(items)] for i in range(n)]

    return sample_sequential


def _make_arange_sample(rng: Random):
    def arange_sample(start: float, end: float, step: float = 1) -> str:
        if start > end:
            return ""
        values = np.linspace(start, end, round((end - start) / step) + 1)
        precision = _step_precision(step)
        return str(round(float(rng.choice(values)), precision))

    return arange_sample


def ensure_int(value: Any) -> int:
    """Convert a value to int if it's a float representing an integer, else return an error"""
    if is_int(value):
        return round(value)
    raise ValueError(f"Value {value} cannot be converted to int.")


def build_eval_context(rng: Random, replacements: dict[str, Any]) -> dict[str, Any]:
    """Build an eval context with rng-bound sampling functions."""
    return {
        "is_int": is_int,
        "divides": divides,
        "int": int,
        "float": float,
        "round": round,
        "str": str,
        "len": len,
        "sample": _make_sample(rng),
        "sample_sequential": _make_sample_sequential(rng),
        "list": list,
        "range": _make_range_sample(rng),
        "range_list": range_possibilities,
        "range_str": _make_range_str(rng),
        "arange": _make_arange_sample(rng),
        "Fraction": frac_format,
        "plural": plural,
        "ensure_int": ensure_int,
        **replacements,
    }


# Keep module-level stubs so templates evaluated outside generate_questions still work.
def sample(items: list, n: int = 1) -> Any:
    return _make_sample(Random())(items, n)


def range_sample(start: int, end: int, step: int = 1) -> int:
    return _make_range_sample(Random())(start, end, step)


def range_str(start: int, end: int, step: int, numbers: list) -> tuple:
    return _make_range_str(Random())(start, end, step, numbers)


def sample_sequential(items: list, n: int) -> list:
    return _make_sample_sequential(Random())(items, n)


def _step_precision(step: float) -> int:
    exponent = decimal.Decimal(str(step)).as_tuple().exponent
    return max(0, -exponent) if isinstance(exponent, int) else 0


def arange_sample(start: float, end: float, step: float = 1) -> str:
    if start > end:
        return ""
    values = np.linspace(start, end, round((end - start) / step) + 1)
    precision = _step_precision(step)
    # standalone arange_sample (used outside generate_questions)
    rng = Random()
    values = np.linspace(start, end, round((end - start) / step) + 1)
    precision = _step_precision(step)
    return str(round(float(rng.choice(values)), precision))


def frac_format(value: Any) -> str:
    if isinstance(value, float):
        frac = Fraction(value).limit_denominator()
        return f"{frac.numerator}/{frac.denominator}" if frac.denominator != 1 else str(frac.numerator)
    return str(value)


def is_variable_mentioned(variable_name: str, text_list: list[str]) -> bool:
    pattern = re.compile(rf"\b{re.escape(variable_name)}\b", re.I)
    return any(pattern.search(text) for text in text_list)


def range_possibilities(start: int, end: int, step: int = 1) -> list[int]:
    if start > end:
        return []
    return list(range(start, end, step))


def range_possibilities_str(start: int, end: int, step: int, numbers: list) -> list[tuple]:
    return [(numbers[i - 1], i) for i in range_possibilities(start, end, step)]


def arange_possibilities(start: float, end: float, step: float = 1) -> list[str]:
    if start > end:
        return []
    values = np.linspace(start, end, round((end - start) / step) + 1)
    precision = _step_precision(step)
    return [str(round(float(v), precision)) for v in values]


def sample_possibilities(items: list, n: int = 1) -> list:
    return list(itertools.combinations(items, n)) if n > 1 else items


def sample_sequential_possibilities(items: list, n: int) -> list[list]:
    return [[items[(i + j) % len(items)] for j in range(n)] for i in range(len(items))]


def parse_lhs_variables(variable_part: str) -> list[str]:
    """Parse the variable names from the left-hand side of an init line.

    Handles plain (``a, b``), ``$``-prefixed (``$a, $b``), and tuple-unpacking
    (``(a, a_reg), (b, b_reg)``) forms, returning a flat list of clean names.

    Args:
        variable_part: The text to the left of ``=`` on an init line.

    Returns:
        The variable names, with parentheses, ``$`` markers, and surrounding
        whitespace removed.
    """
    cleaned = variable_part.replace("(", "").replace(")", "").replace("$", "")
    return [name.strip() for name in cleaned.split(",") if name.strip()]


def align_values_to_variables(variables: list[str], values: Any) -> list:
    """Align a sampled value sequence with a flat list of unpacking variables.

    Tuple-unpacking init lines like ``(a, a_reg), (b, b_reg) = [[1, 2], [3, 4]]``
    parse to four flat variables but only two nested value pairs. When the counts
    disagree, flatten one level so the values zip element-wise with the variables.
    A non-sequence value (or a string) is wrapped/kept intact rather than split.

    Args:
        variables: The flat list of variable names from the init-line LHS.
        values: The evaluated right-hand side.

    Returns:
        A flat list of values; ``len`` equal to ``variables`` when the nesting matches.
    """
    if not isinstance(values, (list, tuple)):
        return [values]
    if len(values) == len(variables):
        return list(values)
    flat: list = []
    for value in values:
        if isinstance(value, (list, tuple)):
            flat.extend(value)
        else:
            flat.append(value)
    return flat


def plural(n: float, *forms: str) -> str:
    """Select the grammatical form that agrees with the number ``n``.

    Pass the forms positionally; the number of forms selects the rule:

    - 2 forms — singular / plural, e.g. English: ``plural(n, "coin", "coins")``
      returns the first form when ``|n| == 1`` and the second otherwise.
    - 3 forms — one / few / many, e.g. East Slavic: ``plural(n, "рік", "роки", "років")``
      applies the Russian/Ukrainian rule based on ``n % 10`` and ``n % 100``.

    Args:
        n: The number the form must agree with.
        forms: Either 2 (singular, plural) or 3 (one, few, many) word forms.

    Returns:
        The form agreeing with ``n``.
    """
    if len(forms) == 2:
        one, other = forms
        return one if abs(n) == 1 else other
    if len(forms) == 3:
        one, few, many = forms
        m = abs(int(n))
        if m % 10 == 1 and m % 100 != 11:
            return one
        if m % 10 in (2, 3, 4) and m % 100 not in (12, 13, 14):
            return few
        return many
    raise ValueError(f"plural() expects 2 (singular/plural) or 3 (one/few/many) forms, got {len(forms)}")


# Non-random helpers shared by both eval contexts and combination enumeration.
_BASE_HELPERS: dict[str, Any] = {
    "is_int": is_int,
    "divides": divides,
    "int": int,
    "float": float,
    "round": round,
    "str": str,
    "len": len,
    "list": list,
    "Fraction": frac_format,
    "plural": plural,
    "ensure_int": ensure_int,
}


# Legacy alias used by condition evaluation and answer formatting (no sampling needed there).
EVAL_CONTEXT_HELPERS: dict[str, Any] = _BASE_HELPERS

COMBINATION_HELPERS: dict[str, Any] = {
    "range": range_possibilities,
    "range_list": range_possibilities,
    "range_str": range_possibilities_str,
    "sample_sequential": sample_sequential_possibilities,
    "arange": arange_possibilities,
    "sample": sample_possibilities,
    "list": list,
    # deterministic helpers shared with EVAL_CONTEXT_HELPERS
    "int": int,
    "float": float,
    "str": str,
    "len": len,
    "round": round,
    "is_int": is_int,
    "divides": divides,
    "Fraction": frac_format,
    "plural": plural,
}

# Pre-compiled regex patterns used in hot paths
RE_TEMPLATE_VAR = re.compile(r"\{(\w+),\s*([^}]+)\}")
RE_CURLY_EXPR = re.compile(r"\{([^}]+)\}")
_RE_NUMBER_FORMAT = re.compile(r"\b\d+(?:\.\d+)\b|\b\d{5,}\b")
_RE_SENTENCE_CAP = re.compile(r"([.!?]+\s*)([a-z])")


def parse_value(val: Any) -> int | float | Fraction | str:
    if isinstance(val, float) and val.is_integer():
        return int(val)
    if isinstance(val, str) and val.isnumeric():
        # str.isnumeric() returns True for CJK numerals (e.g. '四'), but int() cannot
        # parse them. Fall back to returning the string as-is in that case.
        try:
            return int(val)
        except ValueError:
            return val
    return try_parse_fraction(try_parse_float(val))


def try_parse_float(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return float(value)
    except ValueError:
        return value


def try_parse_fraction(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if value.count("/") == 1:
        num, denom = value.split("/")
        if num.lstrip("-").isdigit() and denom.lstrip("-").isdigit():
            return Fraction(int(num), int(denom))
    return value


def capitalize_sentences(text: str) -> str:
    text = text[0].upper() + text[1:] if text else text
    return _RE_SENTENCE_CAP.sub(lambda m: m.group(1) + m.group(2).upper(), text)


# Languages that use comma as decimal separator and period as thousands separator
_COMMA_DECIMAL_LANGUAGES = {"dan", "nob", "nno", "swe", "deu", "fin", "isl", "nld", "fra"}


def format_numbers_by_language(text: str, language: str) -> str:
    comma_decimal = language in _COMMA_DECIMAL_LANGUAGES

    def format_number(match: re.Match) -> str:
        number_str = match.group(0)
        if "." in number_str:
            integer_part, decimal_part = number_str.split(".")
            number = int(integer_part)
            formatted_int = f"{number:,}" if number >= 10000 else str(number)
            if comma_decimal:
                return formatted_int.replace(",", ".") + "," + decimal_part
            return formatted_int + "." + decimal_part
        else:
            number = int(number_str)
            if number >= 10000:
                formatted = f"{number:,}"
                return formatted.replace(",", ".") if comma_decimal else formatted
            return number_str

    def format_decimal_only(match: re.Match) -> str:
        """Inside <<...>>: convert decimal separator only, no thousands sep on integers."""
        number_str = match.group(0)
        if "." in number_str and comma_decimal:
            integer_part, decimal_part = number_str.split(".")
            return integer_part + "," + decimal_part
        return number_str

    # Apply full formatting outside <<...>> markers and the #### answer line.
    # Inside <<...>>: decimal separator only (no thousands sep on integers).
    # On #### line: no formatting (keep plain integer for scoring).
    _RE_SKIP = re.compile(r"(<<[^>]*>>|####\s*-?\d[\d.,]*)")
    parts = _RE_SKIP.split(text)
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            result.append(_RE_NUMBER_FORMAT.sub(format_number, part))
        else:
            # <<...>> or #### line: decimal separator only, no thousands sep on integers
            result.append(_RE_NUMBER_FORMAT.sub(format_decimal_only, part))
    return "".join(result)
