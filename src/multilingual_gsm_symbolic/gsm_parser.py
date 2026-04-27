import ast
import decimal
import itertools
import json
import logging
import math
import operator as _operator
import random
import re
import warnings
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import cached_property
from pathlib import Path
from typing import Any, Self

import numpy as np

logger = logging.getLogger(__name__)

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


def _eval_node(node: ast.expr, env: dict[str, Any]) -> Any:
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
        return [_eval_node(e, env) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(e, env) for e in node.elts)
    if isinstance(node, ast.BinOp):
        op_fn = _BINOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        return op_fn(_eval_node(node.left, env), _eval_node(node.right, env))
    if isinstance(node, ast.UnaryOp):
        val = _eval_node(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return +val
        if isinstance(node.op, ast.Not):
            return not val
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    if isinstance(node, ast.Call):
        func = _eval_node(node.func, env)
        args = [_eval_node(a, env) for a in node.args]
        kwargs = {kw.arg: _eval_node(kw.value, env) for kw in node.keywords if kw.arg is not None}
        return func(*args, **kwargs)
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, env)
        for op, comparator in zip(node.ops, node.comparators):
            op_fn = _CMPOPS.get(type(op))
            if op_fn is None:
                raise ValueError(f"Unsupported comparison: {type(op).__name__}")
            right = _eval_node(comparator, env)
            if not op_fn(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_eval_node(v, env) for v in node.values)
        if isinstance(node.op, ast.Or):
            return any(_eval_node(v, env) for v in node.values)
        raise ValueError(f"Unsupported bool operator: {type(node.op).__name__}")
    if isinstance(node, ast.IfExp):
        return _eval_node(node.body if _eval_node(node.test, env) else node.orelse, env)
    raise ValueError(f"Unsupported AST node type: {type(node).__name__}")


def _parse_expr(source: str) -> ast.expr:
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


def sample(items: list, n: int = 1) -> Any:
    if n == 1:
        return random.choice(items)
    return random.sample(items, n)


def range_sample(start: int, end: int, step: int = 1) -> int:
    if start > end:
        raise ValueError(f"Start ({start}) must be less than or equal to end ({end}).")
    return random.choice(list(range(start, end + 1, step)))


def range_str(start: int, end: int, step: int, numbers: list) -> tuple:
    if start > end:
        return ()
    candidates = [(numbers[i - 1], i) for i in range(start, end + 1, step) if 0 < i <= len(numbers)]
    return random.choice(candidates)


def sample_sequential(items: list, n: int) -> list:
    start_idx = random.randint(0, len(items) - 1)
    return [items[(start_idx + i) % len(items)] for i in range(n)]


def _step_precision(step: float) -> int:
    exponent = decimal.Decimal(str(step)).as_tuple().exponent
    return max(0, -exponent) if isinstance(exponent, int) else 0


def arange_sample(start: float, end: float, step: float = 1) -> str:
    if start > end:
        return ""
    values = np.linspace(start, end, round((end - start) / step) + 1)
    precision = _step_precision(step)
    return str(round(float(random.choice(values)), precision))


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


def strip_elements(lst: list[str]) -> list[str]:
    return [s.strip() for s in lst]


EVAL_CONTEXT_HELPERS: dict[str, Any] = {
    "is_int": is_int,
    "divides": divides,
    "int": int,
    "float": float,
    "round": round,
    "str": str,
    "len": len,
    "sample": sample,
    "sample_sequential": sample_sequential,
    "list": list,
    "range": range_sample,
    "range_list": range_possibilities,
    "range_str": range_str,
    "arange": arange_sample,
    "Fraction": frac_format,
}

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
}

# Pre-compiled regex patterns used in hot paths
_RE_TEMPLATE_VAR = re.compile(r"\{(\w+),\s*([^}]+)\}")
_RE_CURLY_EXPR = re.compile(r"\{([^}]+)\}")
_RE_NUMBER_FORMAT = re.compile(r"\b\d+(?:\.\d+)\b|\b\d{5,}\b")
_RE_SENTENCE_CAP = re.compile(r"([.!?]+\s*)([a-z])")


def parse_value(val: Any) -> int | float | Fraction | str:
    if (isinstance(val, str) and val.isnumeric()) or (isinstance(val, float) and val.is_integer()):
        return int(val)
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


@dataclass
class Question:
    """Dataclass holding a single generated problem.

    Attributes:
        question: The rendered question text.
        answer: The rendered answer text.
        id_orig: Index of the original template.
        id_shuffled: Index within the shuffled sample.
    """

    question: str
    answer: str
    id_orig: int
    id_shuffled: int

    def to_json(self, filepath: Path) -> None:
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False)


@dataclass
class AnnotatedQuestion:
    question: str
    answer: str
    id_orig: int
    id_shuffled: int
    question_annotated: str
    answer_annotated: str
    language: str = "eng"
    creation: str = ""

    @classmethod
    def from_json(cls, filepath: Path) -> Self:
        """Load an AnnotatedQuestion from a JSON template file.

        Args:
            filepath: Path to the JSON template file.

        Returns:
            The loaded AnnotatedQuestion instance.
        """
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @cached_property
    def question_template(self) -> str:
        return self.question_annotated.splitlines()[0].strip()

    @cached_property
    def variables(self) -> list[str]:
        all_vars: set[str] = set()
        for line in self.init:
            if "=" in line:
                all_vars.update(self._extract_variables_from_init_line(line))
        return list(all_vars)

    @cached_property
    def init(self) -> list[str]:
        init_block = (
            self.question_annotated.split("#init:")[1]
            .split("#answer:")[0]
            .split("#conditions:")[0]
            .strip()
            .splitlines()
        )
        return [line.strip("- ") for line in init_block]

    @cached_property
    def conditions(self) -> list[str]:
        if "#conditions:" not in self.question_annotated:
            return []
        condition_block = self.question_annotated.split("#conditions:")[1].split("#answer:")[0].strip().splitlines()
        return [line.strip("- ") for line in condition_block if line.strip()]

    @cached_property
    def constrained_variables(self) -> list[str]:
        if not self.conditions:
            return []
        return [v for v in self.variables if is_variable_mentioned(v, self.conditions)]

    @cached_property
    def unconstrained_lines(self) -> list[str]:
        return [line for line in self.init if not self._is_init_line_constrained(line, self.constrained_variables)]

    @cached_property
    def constrained_lines(self) -> list[str]:
        return [line for line in self.init if self._is_init_line_constrained(line, self.constrained_variables)]

    @cached_property
    def _answer_expr_asts(self) -> dict[str, ast.expr]:
        exprs = _RE_CURLY_EXPR.findall(self.answer_annotated)
        return {expr.strip(): _parse_expr(expr.strip()) for expr in set(exprs)}

    @cached_property
    def _init_line_asts(self) -> list[tuple[list[str], ast.expr]]:
        result = []
        for line in self.init:
            if "=" not in line:
                continue
            variable_part, definition_part = line.split("=", 1)
            variables = strip_elements(variable_part.strip("$").split(","))
            result.append((variables, _parse_expr(definition_part.strip())))
        return result

    @cached_property
    def _condition_asts(self) -> list[ast.expr]:
        return [_parse_expr(cond.strip()) for cond in self.conditions if cond.strip() != "True"]

    def get_default_assignments(self) -> dict[str, Any]:
        """Extract the default variable values from the question template placeholders.

        Returns:
            Mapping of variable name → default value.
        """
        """Return the default variable values as written in the question template placeholders."""
        assignment_tuples = _RE_TEMPLATE_VAR.findall(self.question_template)
        return {var: parse_value(val) for var, val in assignment_tuples}

    def _get_full_default_assignments(self, replacements: dict[str, Any]) -> dict[str, Any]:
        """Return defaults for all variables, deriving paired variables not in the question template."""
        assignments = self.get_default_assignments()

        for var in self.variables:
            if var in assignments:
                continue
            logger.debug(f"Variable {var} not in question template; deriving from init in question {self.id_shuffled}.")
            assignment_line = next(
                (line for line in self.init if var in self._extract_variables_from_init_line(line)),
                None,
            )
            if not assignment_line:
                raise ValueError(f"Variable {var} not found in any assignment line in question {self.id_shuffled}.")
            line_vars = self._extract_variables_from_init_line(assignment_line)
            definition_part = assignment_line.split("=", 1)[1].strip()
            other_var = next((v for v in line_vars if v != var), None)
            if not (other_var and other_var in assignments):
                raise ValueError(
                    f"Variable {var} not found in assignments, and no other variable to derive from "
                    f"in question {self.id_shuffled}."
                )
            other_value = assignments[other_var]
            potential_values = _eval_node(_parse_expr(definition_part), COMBINATION_HELPERS | replacements)
            for val in potential_values:
                if isinstance(val, (tuple, list)) and len(val) == 2:
                    if val[0] == other_value or str(val[0]) == str(other_value):
                        assignments[var] = val[1]
                        break
                    if val[1] == other_value or str(val[1]) == str(other_value):
                        assignments[var] = val[0]
                        break
            if var not in assignments:
                raise ValueError(
                    f"Could not derive value for {var} (other_var={other_var}, value={other_value}) "
                    f"from line '{assignment_line}' in question {self.id_shuffled}."
                )

        return assignments

    def _extract_variables_from_init_line(self, line: str) -> list[str]:
        return strip_elements(line.split("=")[0].strip("- ").strip("$").split(","))

    def _is_init_line_constrained(self, line: str, constrained_variables: list[str]) -> bool:
        return any(v in self._extract_variables_from_init_line(line) for v in constrained_variables)

    def _evaluate_constrained_init_lines(
        self, init_lines: list[str], replacements: dict[str, Any], fixed: dict[str, Any] | None = None
    ) -> list[dict]:
        possible_assignments = self._get_all_possible_assignments(init_lines, replacements, fixed)
        all_combinations = self._get_all_combinations(possible_assignments)
        return self._filter_invalid_combinations(all_combinations)

    def _get_all_possible_assignments(
        self, init_lines: list[str], replacements: dict[str, Any], fixed: dict[str, Any] | None = None
    ) -> dict[str, list[dict]]:
        possible_assignments: dict[str, list[dict]] = {}
        env = COMBINATION_HELPERS | replacements
        for line in init_lines:
            variable_part, definition_part = line.split("=", 1)
            variables = strip_elements(variable_part.strip("$").split(","))
            possible_values = _eval_node(_parse_expr(definition_part.strip()), env)

            key = ", ".join(variables)
            if len(variables) == 1:
                var = variables[0]
                candidates = [{var: val} for val in possible_values]
                if fixed and var in fixed:
                    candidates = [c for c in candidates if c[var] == fixed[var]] or candidates
                possible_assignments[key] = candidates
            elif isinstance(possible_values, list):
                candidates = [dict(zip(variables, pos_val)) for pos_val in possible_values]
                if fixed:
                    filtered = [c for c in candidates if all(c.get(k) == fixed[k] for k in fixed if k in c)]
                    candidates = filtered or candidates
                possible_assignments[key] = candidates
            elif isinstance(possible_values, tuple) and len(possible_values) == len(variables):
                possible_assignments[key] = [dict(zip(variables, possible_values))]
            else:
                logger.warning(f"Incompatible variables {variables} and values {possible_values} for line '{line}'.")

        return possible_assignments

    def _get_all_combinations(self, possibilities: dict[str, list[dict]]) -> list[dict]:
        num_combinations = math.prod(len(v) for v in possibilities.values())
        logger.info(f"Number of combinations: {num_combinations}")
        if num_combinations > 10_000_000:
            raise ValueError(
                f"Too many combinations ({num_combinations}) for question {self.id_shuffled}. "
                "Please reduce the number of variables or their possible values."
            )
        return [
            {k: parse_value(v) for d in combo for k, v in d.items()}
            for combo in itertools.product(*possibilities.values())
        ]

    def _filter_invalid_combinations(self, combinations: list[dict]) -> list[dict]:
        condition_asts = self._condition_asts
        valid = [
            combo
            for combo in combinations
            if all(_eval_node(cond, EVAL_CONTEXT_HELPERS | combo) for cond in condition_asts)
        ]
        logger.debug(f"Number of valid combinations: {len(valid)}")
        return valid

    def format_question(self, assignments: dict[str, Any]) -> str:
        """Render the question text for a given variable assignment.

        Args:
            assignments: Variable name → value mapping.

        Returns:
            The rendered question string.
        """

        def replace_placeholder(match: re.Match) -> str:
            variable_name = match.group(1)
            return str(assignments[variable_name]) if variable_name in assignments else match.group(0)

        processed_text = _RE_TEMPLATE_VAR.sub(replace_placeholder, self.question_template)
        processed_text = format_numbers_by_language(processed_text, self.language)
        return capitalize_sentences(processed_text)

    def format_answer(self, assignments: dict[str, Any]) -> str:
        """Render the answer text for a given variable assignment.

        Args:
            assignments: Variable name → value mapping.

        Returns:
            The rendered answer string.
        """
        eval_env = EVAL_CONTEXT_HELPERS | {k: parse_value(v) for k, v in assignments.items()}

        def eval_curly_expr(match: re.Match) -> str:
            expr_str = match.group(1).strip()
            logger.debug(f"Evaluating expression: {expr_str}")
            value = _eval_node(self._answer_expr_asts[expr_str], eval_env)
            logger.debug(f"Evaluated value: {value}")
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            return str(value)

        processed_text = _RE_CURLY_EXPR.sub(eval_curly_expr, self.answer_annotated)
        processed_text = format_numbers_by_language(processed_text, self.language)
        return capitalize_sentences(processed_text)

    def _precompute_unconstrained(
        self, replacements: dict[str, Any], fixed: dict[str, Any] | None = None
    ) -> list[list[dict]]:
        """Pre-enumerate all possible assignments for each unconstrained init line."""
        constrained = set(self.constrained_variables)
        env = COMBINATION_HELPERS | replacements
        choices_per_line: list[list[dict]] = []
        for variables, ast_node in self._init_line_asts:
            if any(v in constrained for v in variables):
                continue
            possible_values = _eval_node(ast_node, env)
            if len(variables) == 1:
                var = variables[0]
                if not isinstance(possible_values, list):
                    possible_values = [possible_values]
                choices: list[dict] = [{var: v} for v in possible_values]
                if fixed and var in fixed:
                    choices = [c for c in choices if c[var] == fixed[var]] or choices
            else:
                choices = [dict(zip(variables, vals)) for vals in possible_values]
                if fixed:
                    filtered = [c for c in choices if all(c.get(k) == fixed[k] for k in fixed if k in c)]
                    choices = filtered or choices
            choices_per_line.append(choices)
        return choices_per_line

    def _project_assignment(self, assignment: dict[str, Any], only_numeric: bool = True) -> dict[str, Any]:
        projected: dict[str, Any] = {}
        for variable, value in assignment.items():
            value = parse_value(value)
            if only_numeric and (not isinstance(value, (int, float, Fraction)) or isinstance(value, bool)):
                continue
            projected[variable] = value
        return projected

    def get_combinations(
        self,
        replacements: dict[str, Any] | None = None,
        only_numeric: bool = True,
        fixed: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Enumerate unique valid assignments for this template.

        Args:
            replacements: Replacement values used by template init expressions. If
                omitted, replacements for the template language are loaded
                automatically.
            only_numeric: If True, keep only numeric variables in each returned
                assignment.
            fixed: Optional mapping of variable names to values that should be
                held constant while enumerating combinations.
            limit: Optional maximum number of unique combinations to return.

        Returns:
            A list of unique valid assignments for the template.
        """
        if replacements is None:
            from multilingual_gsm_symbolic.load_data import load_replacements

            replacements = load_replacements(self.language)

        valid_combinations = (
            self._evaluate_constrained_init_lines(self.constrained_lines, replacements, fixed)
            if self.constrained_lines
            else [{}]
        )
        unconstrained_choices = self._precompute_unconstrained(replacements, fixed)

        combinations: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()

        for constrained_assignment in valid_combinations:
            unconstrained_products = itertools.product(*unconstrained_choices) if unconstrained_choices else [()]
            for unconstrained_assignment in unconstrained_products:
                assignment = dict(constrained_assignment)
                for partial_assignment in unconstrained_assignment:
                    assignment.update(partial_assignment)

                projected = self._project_assignment(assignment, only_numeric=only_numeric)
                key = tuple(sorted((variable, repr(value)) for variable, value in projected.items()))
                if key in seen:
                    continue
                seen.add(key)
                combinations.append(projected)

                if limit is not None and len(combinations) >= limit:
                    return combinations

        return combinations

    def _generate_question(
        self,
        replacements: dict[str, Any],
        valid_combinations: list[dict] | None = None,
        unconstrained_choices: list[list[dict]] | None = None,
    ) -> Question:
        if unconstrained_choices is not None:
            unconstrained_assignments = [random.choice(choices) for choices in unconstrained_choices]
        else:
            unconstrained_assignments = [
                self._evaluate_unconstrained_init_line(line, replacements) for line in self.unconstrained_lines
            ]
        logger.debug(f"Unconstrained assignments: {unconstrained_assignments}")
        if self.constrained_lines:
            if valid_combinations is None:
                valid_combinations = self._evaluate_constrained_init_lines(self.constrained_lines, replacements)
            constrained_assignments = random.choice(valid_combinations)
        else:
            constrained_assignments = {}
        logger.debug(f"Constrained assignments: {constrained_assignments}")
        collected_assignments = constrained_assignments | {
            k: v for d in unconstrained_assignments for k, v in d.items()
        }
        logger.debug(f"All assignments: {collected_assignments}")
        formatted_question = self.format_question(collected_assignments)
        logger.info(f"Formatted question: {formatted_question}")
        formatted_answer = self.format_answer(collected_assignments)
        logger.info(f"Formatted answer: {formatted_answer}")
        return Question(formatted_question, formatted_answer, self.id_orig, self.id_shuffled)

    def _evaluate_unconstrained_init_line(self, init_line: str, replacements: dict[str, Any]) -> dict[str, Any]:
        variable_part, definition_part = init_line.split("=", 1)
        variables = strip_elements(variable_part.strip("$").split(","))
        values = _eval_node(_parse_expr(definition_part.strip()), EVAL_CONTEXT_HELPERS | replacements)
        if not isinstance(values, (list, tuple)):
            values = [values]
        if len(values) != len(variables):
            logger.warning(f"Incompatible variables {variables} and values {values} in template {self.id_shuffled}.")
            return {}
        return dict(zip(variables, values))

    def generate_questions(
        self,
        n: int,
        replacements: dict[str, Any] | None = None,
        seed: int | None = None,
        verbose: bool = True,
        fixed: dict[str, Any] | None = None,
    ) -> list[Question]:
        """Generate concrete Question instances from the template.

        Args:
            n: Number of questions to generate.
            replacements: Replacement values; loaded automatically if omitted.
            seed: Random seed for reproducibility.
            verbose: Show warnings for slow generation.
            fixed: Variables to hold constant; only remaining variables are sampled.

        Returns:
            The generated questions.
        """
        if replacements is None:
            from multilingual_gsm_symbolic.load_data import load_replacements

            replacements = load_replacements(self.language)
        if seed is not None:
            random.seed(seed)
        if verbose and self.constrained_variables:
            warnings.warn(
                f"Template {self.id_shuffled} has constrained variables {self.constrained_variables}. "
                "Generation may be slow for large n. Set verbose=False to suppress this warning.",
                stacklevel=2,
            )
        valid_combinations = (
            self._evaluate_constrained_init_lines(self.constrained_lines, replacements, fixed)
            if self.constrained_lines
            else None
        )
        unconstrained_choices = self._precompute_unconstrained(replacements, fixed)
        return [self._generate_question(replacements, valid_combinations, unconstrained_choices) for _ in range(n)]
