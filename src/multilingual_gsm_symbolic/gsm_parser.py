from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import itertools
import random
import logging
from functools import reduce
from typing import Self
import numpy as np
from fractions import Fraction

logger = logging.getLogger(__name__)


def is_int(value):
    return (
        isinstance(value, int)
        or (
            isinstance(value, float)
            and (value.is_integer() or (value - round(value)) < 1e-6)
        )
        or (isinstance(value, Fraction) and value.denominator == 1)
    )


def divides(a, b):
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise TypeError("Both arguments must be int or float.")
    if b == 0:
        return False
    return a % b == 0


def sample(items, n=1):
    """Sample n items from the list"""
    if n == 1:
        return random.choice(items)
    return random.sample(items, n)


def range_sample(start, end, step=1):
    """Sample an item from a range statement."""
    if start > end:
        raise ValueError(f"Start ({start}) must be less than or equal to end ({end}).")
    if step <= 0:
        raise ValueError(f"Step ({step}) must be a positive integer.")
    possible_numbers = [i for i in range(start, end + 1, step)]
    if not possible_numbers:
        raise ValueError(f"No valid numbers in range({start}, {end}, {step}).")
    return random.choice(possible_numbers)


def range_str(start, end, step, numbers):
    """Return a string representation of a range statement."""
    if start > end:
        return ""
    number_range = range(start, end + 1, step)
    possible_numbers = [
        (i, numbers[i - 1]) for i in number_range if i > 0 and i <= len(numbers)
    ]
    return random.choice(possible_numbers)


def sample_sequential(items, n):
    """Sample n sequential items from the list"""
    start_idx = random.randint(0, len(items) - 1)
    return [items[(start_idx + i) % len(items)] for i in range(n)]


def arange_sample(start, end, step=1):
    """Sample an item from a numpy arange"""
    if start > end:
        return []
    return str(random.choice(np.linspace(start, end, round((end - start) / step) + 1)))


def frac_format(value):
    """Format a value as a fraction if it is a float, otherwise return as is."""
    if isinstance(value, float):
        # Convert float to Fraction
        frac = Fraction(value).limit_denominator()
        return (
            f"{frac.numerator}/{frac.denominator}"
            if frac.denominator != 1
            else str(frac.numerator)
        )
    return str(value)


def is_variable_mentioned(variable_name, text_list):
    """
    Check if a variable is mentioned in any text from a list.
    """
    # Create a regular expression that matches the variable name
    # surrounded by word boundaries to ensure it's a standalone reference
    # \b is a word boundary that matches positions where a word character
    # is not followed or preceded by another word character
    variable_pattern = re.compile(r"\b%s\b" % re.escape(variable_name), re.I)

    for text in text_list:
        if variable_pattern.search(text):
            return True

    return False


def range_possibilities(start, end, step=1):
    """Return possibilities for given range statement."""
    if start > end:
        return []
    return list(range(start, end, step))


def range_possibilities_str(start, end, step, numbers):
    """Return possibilities for given range statement."""
    possible_numbers = range_possibilities(start, end, step)
    return [(numbers[i - 1], i) for i in possible_numbers]


def arange_possibilities(start, end, step=1):
    """Return possibilities for given numpy arange statement."""
    if start > end:
        return []
    return list(map(str, np.linspace(start, end, round((end - start) / step) + 1)))


def sample_possibilities(items, n=1):
    """Return possibilities for given sample statement."""
    return list(itertools.combinations(items, n)) if n > 1 else items


def strip_elements(lst):
    """Strip whitespace from each element in a list"""
    return [elem.strip() for elem in lst]


EVAL_CONTEXT_HELPERS = {
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

COMBINATION_HELPERS = {
    "range": range_possibilities,
    "range_str": range_possibilities_str,
    "arange": arange_possibilities,
    "sample": sample_possibilities,
    "list": list,
}


# Convert value to int, float, or fraction if possible
def parse_value(val):
    if (
        isinstance(val, str)
        and val.isnumeric()
        or isinstance(val, float)
        and val.is_integer()
    ):
        return int(val)
    return try_parse_fraction(try_parse_float(val))


def try_parse_float(value):
    """Try to parse a string as float, return string if it fails."""
    if not isinstance(value, str):
        return value
    try:
        return float(value)
    except ValueError:
        return value


def try_parse_fraction(value):
    """Try to parse a string as a fraction, return string if it fails."""
    if not isinstance(value, str):
        return value
    if "/" in value:
        try:
            num, denom = value.split("/")
            return Fraction(int(num), int(denom))
        except ValueError:
            return value
    return value


def capitalize_sentences(text):
    """Capitalize the first letter of each sentence using regex."""
    import re

    # Capitalize first letter of text
    text = text[0].upper() + text[1:] if text else text

    # Capitalize letters after sentence-ending punctuation
    text = re.sub(
        r"([.!?]+\s*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text
    )

    return text


def format_numbers_by_language(text, language):
    import re

    def format_number(match):
        number_str = match.group(0)

        if "." in number_str:
            integer_part, decimal_part = number_str.split(".")
            number = int(integer_part)
            formatted_int = f"{number:,}" if number >= 10000 else str(number)

            if language == "dan":
                return formatted_int.replace(",", ".") + "," + decimal_part
            else:
                return formatted_int + "." + decimal_part
        else:
            number = int(number_str)
            if number >= 10000:
                formatted = f"{number:,}"
                return formatted.replace(",", ".") if language == "dan" else formatted
            else:
                return number_str

    return re.sub(r"\b\d+(?:\.\d+)\b|\b\d{5,}\b", format_number, text)


@dataclass
class Question:
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

    @classmethod
    def from_json(cls, filepath: Path) -> Self:
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @property
    def question_template(self) -> str:
        """extract question template from question_annotated"""
        return self.question_annotated.splitlines()[0].strip()

    @property
    def variables(self) -> list[str]:
        """extract variable names from question_annotated"""

        variables_per_line = [
            self._extract_variables_from_init_line(line)
            for line in self.init
            if "=" in line
        ]
        variable_sets = [set(v) for v in variables_per_line]
        return list(reduce(set.union, variable_sets))

    @property
    def init(self) -> list[str]:
        """extract variable definitions from question_annotated"""
        init_block = (
            self.question_annotated.split("#init:")[1]
            .split("#answer:")[0]
            .split("#conditions:")[0]
            .strip()
            .splitlines()
        )
        return [line.strip("- ") for line in init_block]

    @property
    def conditions(self) -> list[str]:
        """extract conditions from question_annotated"""
        if "#conditions:" not in self.question_annotated:
            return []

        try:
            condition_block = (
                self.question_annotated.split("#conditions:")[1]
                .split("#answer:")[0]
                .strip()
                .splitlines()
            )
            return [line.strip("- ") for line in condition_block if line.strip()]
        except IndexError:
            return []

    @property
    def constrained_variables(self) -> list[str]:
        """extract variable names from conditions"""
        if not self.conditions:
            return []
        return [v for v in self.variables if is_variable_mentioned(v, self.conditions)]

    @property
    def unconstrained_lines(self) -> list[str]:
        """extract unconstrained lines from question_annotated"""
        return [
            line
            for line in self.init
            if not self._is_init_line_constrained(line, self.constrained_variables)
        ]

    @property
    def constrained_lines(self) -> list[str]:
        """extract constrained lines from question_annotated"""
        return [
            line
            for line in self.init
            if self._is_init_line_constrained(line, self.constrained_variables)
        ]

    def get_default_assignments(self, replacements: dict) -> dict:
        """extract example assignments from question_annotated"""
        assignment_tuples = re.findall(r"\{(\w+),\s*([^}]+)\}", self.question_template)

        assignments = {var: parse_value(val) for var, val in assignment_tuples}

        # Ensure that all variables in the answer are also in the question template
        for var in self.variables:
            if var not in assignments:
                logger.debug(
                    f"Variable {var} found in answer but not in question template. Attempting to derive value from other variables. In question {self.id_shuffled}."
                )
                assignment_line = next(
                    (
                        line
                        for line in self.init
                        if var in self._extract_variables_from_init_line(line)
                    ),
                    None,
                )
                if not assignment_line:
                    raise ValueError(
                        f"Variable {var} not found in any assignment line in question {self.id_shuffled}. Please check the question template."
                    )
                vars = self._extract_variables_from_init_line(assignment_line)
                definition_part = self._extract_definition_part_from_init_line(
                    assignment_line
                )
                other_var = next((v for v in vars if v != var), None)
                if other_var and other_var in assignments:
                    other_value = assignments[other_var]
                    potential_values = eval(
                        definition_part,
                        {"__builtins__": {}},
                        COMBINATION_HELPERS | replacements,
                    )
                    for val in potential_values:
                        if isinstance(val, (tuple, list)) and len(val) == 2:
                            if (
                                val[0] == other_value
                                or val[1] == other_value
                                or str(val[0]) == str(other_value)
                                or str(val[1]) == str(other_value)
                            ):
                                assignments[var] = (
                                    val[1]
                                    if val[0] == other_value
                                    or str(val[0]) == str(other_value)
                                    else val[0]
                                )
                                break
                    if assignments.get(var) is None:
                        raise ValueError(
                            f"Could not derive value for variable {var} with value {other_value} from other variables in assignment line: {assignment_line} with potential values {potential_values}. Please check the question template."
                        )
                else:
                    raise ValueError(
                        f"Variable {var} not found in assignments, and no other variable found to derive value from. Please check the question template for question {self.id_shuffled}."
                    )

        return assignments

    def _extract_variables_from_init_line(self, line: str) -> list[str]:
        """extract variable names from a line"""
        variables = line.split("=")[0].strip("- ").strip("$").split(",")
        return [v.strip() for v in variables]

    def _extract_definition_part_from_init_line(self, line: str) -> str:
        """extract the assignment statement from a line"""
        if "=" in line:
            return line.split("=", 1)[1].strip()
        return ""

    def _is_init_line_constrained(
        self, line: str, constrained_variables: list[str]
    ) -> bool:
        """check if a line is constrained"""
        return any(
            v in self._extract_variables_from_init_line(line)
            for v in constrained_variables
        )

    def _evaluate_unconstrained_init_line(self, init_line, replacements):
        """Evaluate a single unconstrained init line and return the assignments."""
        #  If the line is unconstrained, we evaluate it directly since no other variables depend on it.
        logger.debug(f"Evaluating unconstrained init line: {init_line}")
        assignments = {}
        variable_part, definition_part = init_line.split("=", 1)
        variables = strip_elements(variable_part.strip("$").split(","))
        definition_part = definition_part.strip()

        try:
            values = eval(
                definition_part,
                {"__builtins__": {}},
                EVAL_CONTEXT_HELPERS | replacements,
            )

            values = [values] if not isinstance(values, (list, tuple)) else values
            logger.debug(
                f"Variables: {variables}, Definition part: {definition_part}, Evaluated values: {values}"
            )
        except Exception as e:
            logger.error(
                f"Error evaluating assignment for {variable_part}: {definition_part} -> {e}"
            )
            raise e
        if (isinstance(values, list) or isinstance(values, tuple)) and len(
            values
        ) == len(variables):
            for var, val in zip(variables, values):
                assignments[var] = val
        else:
            logger.warning(
                f"Warning: {variables} and {values} are incompatible for line {init_line}."
            )

        return assignments

    def _evaluate_constrained_init_lines(self, init_lines, conditions, replacements):
        """Returns a list of valid combinations of values for the constrained init lines."""

        possible_assignments = self._get_all_possible_assignments(
            init_lines, replacements
        )
        all_combinations = self._get_all_combinations(possible_assignments)
        return self._filter_invalid_combinations(all_combinations, conditions)

    def _get_all_possible_assignments(self, init_lines, replacements):
        possible_assignments = {}
        for line in init_lines:
            variable_part, definition_part = line.split("=", 1)
            variables = strip_elements(variable_part.strip("$").split(","))
            definition_part = definition_part.strip()
            logger.debug(f"Variables: {variables}, Definition part: {definition_part}")
            if len(variables) == 1:
                variable_name = variables[0].strip()
                try:
                    possible_values = eval(
                        definition_part,
                        {"__builtins__": {}},
                        COMBINATION_HELPERS | replacements,
                    )

                    possible_assignments[variable_name] = [
                        {variable_name: val} for val in possible_values
                    ]
                except Exception as e:
                    logger.error(
                        f"Error evaluating line '{line}': {e} for file {self.id_shuffled}"
                    )
                    raise e
            else:
                # If there are multiple variables, we need to handle them as a collected assignment
                try:
                    possible_values = eval(
                        definition_part,
                        {"__builtins__": {}},
                        COMBINATION_HELPERS | replacements,
                    )
                    logger.debug(
                        f"Variables: {variables}, Definition part: {definition_part}, Possible values: {possible_values}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error evaluating assignment for {variable_part}: {definition_part} -> {e}"
                    )
                    raise e

                num_vars = len(variables)
                num_vals = len(
                    possible_values[0]
                    if isinstance(possible_values, list)
                    else possible_values
                )
                if num_vars == num_vals and isinstance(possible_values, list):
                    assignment = ", ".join(variables)
                    # We need to save it as a collected assignment in order to avoid splitting them up when generating combinations later
                    possible_assignments[assignment] = [
                        {var: val for var, val in zip(variables, pos_val)}
                        for pos_val in possible_values
                    ]
                elif num_vars == num_vals and isinstance(possible_values, tuple):
                    # If the possible values are a single tuple, we can directly map them to the variables
                    possible_assignments[", ".join(variables)] = [
                        {var: val for var, val in zip(variables, possible_values)}
                    ]
                else:
                    logger.warning(
                        f"Warning: {variables} and {possible_values} are incompatible for line {line}."
                    )

        return possible_assignments

    def _get_all_combinations(self, possibilities):
        num_combinations = reduce(lambda x, y: x * len(y), possibilities.values(), 1)
        print(f"Number of combinations: {num_combinations}")
        if num_combinations > 10000000:
            raise ValueError(
                f"Too many combinations ({num_combinations}) for question {self.id_shuffled}. Please reduce the number of variables or their possible values."
            )
        all_combinations = list(itertools.product(*possibilities.values()))
        unpacked_combinations = [
            reduce(lambda x, y: x | y, combination) for combination in all_combinations
        ]
        combination_dicts = [
            {k: parse_value(v) for k, v in combination.items()}
            for combination in unpacked_combinations
        ]
        return combination_dicts

    def _filter_invalid_combinations(self, combinations, conditions):
        valid_combinations = []
        # Iterate through each combination and check against every condition
        for combination in combinations:
            is_valid = True
            for cond in conditions:
                if not eval(
                    cond, {"__builtins__": {}}, EVAL_CONTEXT_HELPERS | combination
                ):
                    is_valid = False
                    break

            if is_valid:
                valid_combinations.append(combination)

        logger.debug(f"Number of valid combinations: {len(valid_combinations)}")
        return valid_combinations

    def format_question(self, assignments, language: str = "eng"):
        def replace_placeholder(match):
            variable_name = match.group(1)
            if variable_name in assignments:
                value = assignments[variable_name]
                return str(value)

            return match.group(0)

        processed_text = re.sub(
            r"\{(\w+),\s*([^}]+)\}", replace_placeholder, self.question_template
        )
        processed_text = format_numbers_by_language(processed_text, language)
        return capitalize_sentences(processed_text)

    def format_answer(self, assignments, language: str = "eng"):
        def eval_curly_expr(match):
            expr_str = match.group(1)  # Expression inside curly braces

            logger.debug(f"Evaluating expression: {expr_str}")

            eval_env = EVAL_CONTEXT_HELPERS | assignments
            # Parse the occasional integer...
            eval_env = eval_env | {
                k: int(v)
                for k, v in eval_env.items()
                if isinstance(v, str)
                and v.isnumeric()
                or isinstance(v, float)
                and v.is_integer()
            }
            # Parse the occational float...
            eval_env = eval_env | {k: try_parse_float(v) for k, v in eval_env.items()}
            # Parse the occational fraction...
            eval_env = eval_env | {
                k: try_parse_fraction(v) for k, v in eval_env.items()
            }
            try:
                value = eval(expr_str, {"__builtins__": {}}, eval_env)
                logger.debug(f"Evaluated value: {value}")
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                return str(value)
            except NameError as e:
                raise NameError(
                    str(e)
                    + f"\nNameError evaluating expression '{expr_str}' with environment {eval_env} for answer {self.answer_annotated} with assignments{assignments} in file {self.id_shuffled}"
                )
            except Exception as e:
                logger.error(
                    f"Error evaluating expression '{expr_str}': {e} with environment {eval_env} for answer {self.answer_annotated} with assignments{assignments} in file {self.id_shuffled}"
                )
                raise e

        processed_text = self.answer_annotated
        processed_text = re.sub(
            r"\{([^}]+)\}", lambda m: eval_curly_expr(m), processed_text
        )
        processed_text = format_numbers_by_language(processed_text, language)
        return capitalize_sentences(processed_text)

    def _generate_question(self, language, replacements: dict[str, list]) -> Question:
        unconstrained_assignments = [
            self._evaluate_unconstrained_init_line(line, replacements)
            for line in self.unconstrained_lines
        ]
        logger.debug(f"Unconstrained assignments: {unconstrained_assignments}")
        if len(self.constrained_lines) > 0:
            constrained_assignments = random.choice(
                self._evaluate_constrained_init_lines(
                    self.constrained_lines, self.conditions, replacements
                )
            )
        else:
            constrained_assignments = {}
        logger.debug(f"Constrained assignments: {constrained_assignments}")
        collected_assignments = constrained_assignments | reduce(
            lambda x, y: x | y, unconstrained_assignments
        )
        logger.debug(f"All assignments: {collected_assignments}")
        formatted_question = self.format_question(collected_assignments, language)
        logger.info(f"Formatted question: {formatted_question}")
        formatted_answer = self.format_answer(collected_assignments, language)
        logger.info(f"Formatted answer: {formatted_answer}")

        return Question(
            formatted_question, formatted_answer, self.id_orig, self.id_shuffled
        )

    def generate_questions(
        self, n, language: str, replacements: dict[str, list]
    ) -> list[Question]:
        questions = []
        for i in range(n):
            try:
                question = self._generate_question(language, replacements)
                questions.append(question)
            except Exception as e:
                logger.error(f"Error generating question {i + 1}: {e}")
                continue
        return questions
