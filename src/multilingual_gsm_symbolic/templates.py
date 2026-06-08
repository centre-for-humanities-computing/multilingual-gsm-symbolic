import ast
import itertools
import json
import logging
import math
import re
import tomllib
import warnings
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import cached_property
from pathlib import Path
from random import Random
from typing import Any, Self

from multilingual_gsm_symbolic._helpers import (
    COMBINATION_HELPERS,
    EVAL_CONTEXT_HELPERS,
    RE_CURLY_EXPR,
    RE_TEMPLATE_VAR,
    build_eval_context,
    capitalize_sentences,
    eval_node,
    format_numbers_by_language,
    is_variable_mentioned,
    parse_expr,
    parse_value,
    strip_elements,
)

logger = logging.getLogger(__name__)


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

    def __post_init__(self) -> None:
        constrained_derived = [v for v in self.derived_variables if is_variable_mentioned(v, self.conditions)]
        if constrained_derived:
            raise ValueError(
                f"Template {self.id_shuffled}: derived variable(s) {constrained_derived} are referenced in "
                "#conditions. Derived variables are computed after constraint sampling and cannot be constrained. "
                "Inline the expression in the condition instead (e.g. write 'a + b < 10' rather than 'total < 10')."
            )

    @classmethod
    def from_json(cls, filepath: Path) -> Self:
        """Load an AnnotatedQuestion from a JSON template file.

        .. note::
            TOML is the preferred format. Use :meth:`from_toml` for new templates.

        Args:
            filepath: Path to the JSON template file.

        Returns:
            The loaded AnnotatedQuestion instance.
        """
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_toml(cls, filepath: str | Path) -> Self:
        """Load an AnnotatedQuestion from a TOML template file.

        The TOML file must use the same top-level keys as the JSON format.
        Long text fields (``question_annotated``, ``answer_annotated``,
        ``question``, ``answer``) are typically stored as TOML multiline
        strings, which automatically strips the leading newline.

        Args:
            filepath: Path to the TOML template file.

        Returns:
            The loaded AnnotatedQuestion instance.
        """
        with Path(filepath).open("rb") as f:
            data = tomllib.load(f)
        # TOML multiline basic strings strip the first newline, but may keep a
        # trailing newline before the closing \"\"\". Strip both ends to match
        # the JSON values which have no surrounding whitespace.
        for key in ("question", "answer", "question_annotated", "answer_annotated"):
            if key in data and isinstance(data[key], str):
                data[key] = data[key].strip("\n")
        data.pop("ignore", None)
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
    def derived_lines(self) -> list[str]:
        """Init lines whose right-hand side references another init variable.

        These cannot be sampled independently: their value is determined by other
        variables (e.g. a noun form agreeing with a number, or `total = a + b`), so
        they are evaluated last, after all other variables have been assigned.
        """
        init_vars = set(self.variables)
        return [line for line in self.init if "=" in line and (self._init_line_rhs_names(line) & init_vars)]

    @cached_property
    def _derived_line_asts(self) -> list[tuple[list[str], ast.expr]]:
        return [
            (self._extract_variables_from_init_line(line), parse_expr(line.split("=", 1)[1].strip()))
            for line in self.derived_lines
        ]

    @cached_property
    def derived_variables(self) -> list[str]:
        return [var for variables, _ in self._derived_line_asts for var in variables]

    @cached_property
    def unconstrained_lines(self) -> list[str]:
        derived = set(self.derived_lines)
        return [
            line
            for line in self.init
            if line not in derived and not self._is_init_line_constrained(line, self.constrained_variables)
        ]

    @cached_property
    def constrained_lines(self) -> list[str]:
        derived = set(self.derived_lines)
        return [
            line
            for line in self.init
            if line not in derived and self._is_init_line_constrained(line, self.constrained_variables)
        ]

    @cached_property
    def _answer_expr_asts(self) -> dict[str, ast.expr]:
        exprs = RE_CURLY_EXPR.findall(self.answer_annotated)
        return {expr.strip(): parse_expr(expr.strip()) for expr in set(exprs)}

    @cached_property
    def _init_line_asts(self) -> list[tuple[list[str], ast.expr]]:
        result = []
        for line in self.init:
            if "=" not in line:
                continue
            variable_part, definition_part = line.split("=", 1)
            variables = strip_elements(variable_part.strip("$").split(","))
            result.append((variables, parse_expr(definition_part.strip())))
        return result

    @cached_property
    def _condition_asts(self) -> list[ast.expr]:
        return [parse_expr(cond.strip()) for cond in self.conditions if cond.strip() != "True"]

    def get_default_assignments(self) -> dict[str, Any]:
        """Extract the default variable values from the question template placeholders.

        Returns:
            Mapping of variable name → default value.
        """
        """Return the default variable values as written in the question template placeholders."""
        assignment_tuples = RE_TEMPLATE_VAR.findall(self.question_template)
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
            potential_values = eval_node(parse_expr(definition_part), COMBINATION_HELPERS | replacements)
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

    def _init_line_rhs_names(self, line: str) -> set[str]:
        """Return the names referenced on the right-hand side of an init line.

        Includes both variable references and function names; callers intersect with
        the set of init variables to find genuine cross-variable dependencies.
        """
        definition_part = line.split("=", 1)[1].strip()
        return {node.id for node in ast.walk(parse_expr(definition_part)) if isinstance(node, ast.Name)}

    def _is_init_line_constrained(self, line: str, constrained_variables: list[str]) -> bool:
        return any(v in self._extract_variables_from_init_line(line) for v in constrained_variables)

    def _evaluate_constrained_init_lines(
        self, init_lines: list[str], replacements: dict[str, Any], fixed: dict[str, Any] | None = None
    ) -> list[dict]:
        possible_assignments = self._get_all_possible_assignments(init_lines, replacements, fixed)
        return self._filter_invalid_combinations_streaming(possible_assignments)

    def _get_all_possible_assignments(
        self, init_lines: list[str], replacements: dict[str, Any], fixed: dict[str, Any] | None = None
    ) -> dict[str, list[dict]]:
        possible_assignments: dict[str, list[dict]] = {}
        env = COMBINATION_HELPERS | replacements
        for line in init_lines:
            variable_part, definition_part = line.split("=", 1)
            variables = strip_elements(variable_part.strip("$").split(","))
            possible_values = eval_node(parse_expr(definition_part.strip()), env)

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
        """Materialise all combinations (used when full list is needed, e.g. for generation)."""
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

    def _filter_invalid_combinations_streaming(
        self, possibilities: dict[str, list[dict]], limit: int | None = None
    ) -> list[dict]:
        """Stream combinations from itertools.product and stop as soon as limit is reached."""
        num_combinations = math.prod(len(v) for v in possibilities.values())
        logger.info(f"Number of combinations: {num_combinations}")
        if num_combinations > 10_000_000:
            raise ValueError(
                f"Too many combinations ({num_combinations}) for question {self.id_shuffled}. "
                "Please reduce the number of variables or their possible values."
            )
        condition_asts = self._condition_asts
        valid = []
        for combo in itertools.product(*possibilities.values()):
            assignment = {k: parse_value(v) for d in combo for k, v in d.items()}
            if all(eval_node(cond, EVAL_CONTEXT_HELPERS | assignment) for cond in condition_asts):
                valid.append(assignment)
                if limit is not None and len(valid) >= limit:
                    break
        logger.debug(f"Number of valid combinations: {len(valid)}")
        return valid

    def _filter_invalid_combinations(self, combinations: list[dict], limit: int | None = None) -> list[dict]:
        condition_asts = self._condition_asts
        valid = []
        for combo in combinations:
            if all(eval_node(cond, EVAL_CONTEXT_HELPERS | combo) for cond in condition_asts):
                valid.append(combo)
                if limit is not None and len(valid) >= limit:
                    break
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

        processed_text = RE_TEMPLATE_VAR.sub(replace_placeholder, self.question_template)
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
            value = eval_node(self._answer_expr_asts[expr_str], eval_env)
            logger.debug(f"Evaluated value: {value}")
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            return str(value)

        processed_text = RE_CURLY_EXPR.sub(eval_curly_expr, self.answer_annotated)
        processed_text = format_numbers_by_language(processed_text, self.language)
        return capitalize_sentences(processed_text)

    def _precompute_unconstrained(
        self, replacements: dict[str, Any], fixed: dict[str, Any] | None = None
    ) -> list[list[dict]]:
        """Pre-enumerate all possible assignments for each unconstrained init line."""
        constrained = set(self.constrained_variables)
        derived = set(self.derived_variables)
        env = COMBINATION_HELPERS | replacements
        choices_per_line: list[list[dict]] = []
        for variables, ast_node in self._init_line_asts:
            if any(v in constrained or v in derived for v in variables):
                continue
            possible_values = eval_node(ast_node, env)
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
            # When only_numeric=True, non-numeric unconstrained variables (names, strings)
            # are stripped during projection, so all unconstrained products yield the same
            # key. We only need to iterate unconstrained products when they carry numeric vars.
            if only_numeric and unconstrained_choices:
                # Check whether any unconstrained line contributes numeric values
                sample_combo = {k: v for d in [c[0] for c in unconstrained_choices] for k, v in d.items()}
                sample_proj = self._project_assignment(dict(constrained_assignment) | sample_combo, only_numeric=True)
                constrained_proj = self._project_assignment(constrained_assignment, only_numeric=True)
                unconstrained_adds_numeric = set(sample_proj) != set(constrained_proj)
            else:
                unconstrained_adds_numeric = True

            if unconstrained_adds_numeric:
                unconstrained_products = itertools.product(*unconstrained_choices) if unconstrained_choices else [()]
            else:
                unconstrained_products = [()]  # type: ignore[assignment]

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
        rng: Random,
        valid_combinations: list[dict] | None = None,
        unconstrained_choices: list[list[dict]] | None = None,
    ) -> Question:
        if unconstrained_choices is not None:
            unconstrained_assignments = [rng.choice(choices) for choices in unconstrained_choices]
        else:
            unconstrained_assignments = [
                self._evaluate_unconstrained_init_line(line, replacements, rng) for line in self.unconstrained_lines
            ]
        logger.debug(f"Unconstrained assignments: {unconstrained_assignments}")
        if self.constrained_lines:
            if valid_combinations is None:
                valid_combinations = self._evaluate_constrained_init_lines(self.constrained_lines, replacements)
            constrained_assignments = rng.choice(valid_combinations)
        else:
            constrained_assignments = {}
        logger.debug(f"Constrained assignments: {constrained_assignments}")
        collected_assignments = constrained_assignments | {
            k: v for d in unconstrained_assignments for k, v in d.items()
        }
        if self.derived_lines:
            collected_assignments |= self._evaluate_derived_lines(collected_assignments, replacements, rng)
        logger.debug(f"All assignments: {collected_assignments}")
        formatted_question = self.format_question(collected_assignments)
        logger.info(f"Formatted question: {formatted_question}")
        formatted_answer = self.format_answer(collected_assignments)
        logger.info(f"Formatted answer: {formatted_answer}")
        return Question(formatted_question, formatted_answer, self.id_orig, self.id_shuffled)

    def _assign_from_ast(self, variables: list[str], ast_node: ast.expr, env: dict[str, Any]) -> dict[str, Any]:
        """Evaluate one init-line expression against ``env`` and zip results to variables."""
        values = eval_node(ast_node, env)
        if not isinstance(values, (list, tuple)):
            values = [values]
        if len(values) != len(variables):
            logger.warning(f"Incompatible variables {variables} and values {values} in template {self.id_shuffled}.")
            return {}
        return dict(zip(variables, values))

    def _evaluate_unconstrained_init_line(
        self, init_line: str, replacements: dict[str, Any], rng: Random | None = None
    ) -> dict[str, Any]:
        variables = self._extract_variables_from_init_line(init_line)
        ast_node = parse_expr(init_line.split("=", 1)[1].strip())
        return self._assign_from_ast(variables, ast_node, build_eval_context(rng or Random(), replacements))

    def _evaluate_derived_lines(
        self, assignments: dict[str, Any], replacements: dict[str, Any], rng: Random
    ) -> dict[str, Any]:
        """Evaluate derived init lines against already-assigned variables.

        Lines are evaluated in declaration order; each result is fed back into the
        environment so a later derived line may depend on an earlier one. The rng-bound
        eval context is used so derived lines may themselves call ``sample`` (e.g. to
        choose among synonyms that agree with a sampled number).
        """
        env = build_eval_context(rng, replacements) | {k: parse_value(v) for k, v in assignments.items()}
        derived: dict[str, Any] = {}
        for variables, ast_node in self._derived_line_asts:
            for var, value in self._assign_from_ast(variables, ast_node, env).items():
                derived[var] = value
                env[var] = parse_value(value)
        return derived

    def generate_questions(
        self,
        n: int,
        replacements: dict[str, Any] | None = None,
        seed: int | None = None,
        rng: Random | None = None,
        verbose: bool = True,
        fixed: dict[str, Any] | None = None,
    ) -> list[Question]:
        """Generate concrete Question instances from the template.

        Args:
            n: Number of questions to generate.
            replacements: Replacement values; loaded automatically if omitted.
            seed: Random seed for reproducibility. Ignored if ``rng`` is supplied.
            rng: A :class:`random.Random` instance to use for all sampling.
                Allows full control over the RNG state (e.g. for reproducible
                multi-template experiments). Takes precedence over ``seed``.
            verbose: Show warnings for slow generation.
            fixed: Variables to hold constant; only remaining variables are sampled.

        Returns:
            The generated questions.
        """
        if replacements is None:
            from multilingual_gsm_symbolic.load_data import load_replacements

            replacements = load_replacements(self.language)
        if rng is not None:
            _rng = rng
        elif seed is not None:
            _rng = Random(seed)
        else:
            _rng = Random()
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
        return [
            self._generate_question(replacements, _rng, valid_combinations, unconstrained_choices) for _ in range(n)
        ]
