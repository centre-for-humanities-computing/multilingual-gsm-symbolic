import pytest
from multilingual_gsm_symbolic.gsm_parser import (
    EVAL_CONTEXT_HELPERS,
    AnnotatedQuestion,
    try_parse_float,
    try_parse_fraction,
)
from multilingual_gsm_symbolic.load_data import _DATA_ROOT, load_replacements


class TestGetAllPossibleAssignments:
    """Test class for testing the _get_all_possible_assignments method."""

    def test_range_expression(self):
        """Test with range expressions."""
        annotated_question = AnnotatedQuestion(
            question="Test question",
            answer="Test answer",
            id_orig=1,
            id_shuffled=1,
            question_annotated="Test template\n#init:\n- $x = range(1, 6)\n#conditions:\n- True\n#answer:\nAnswer is {x}",
            answer_annotated="Answer is {x}",
        )

        init_lines = ["$x = range(1, 6)"]
        replacements = {}
        result = annotated_question._get_all_possible_assignments(
            init_lines, replacements
        )

        # Expected: x should have values 1, 2, 3, 4, 5
        expected = {"x": [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}, {"x": 5}]}
        assert result == expected

    def test_range_with_step(self):
        """Test range with step parameter."""
        annotated_question = AnnotatedQuestion(
            question="Test question",
            answer="Test answer",
            id_orig=1,
            id_shuffled=1,
            question_annotated="Test template\n#init:\n- $x = range(1, 10, 2)\n#conditions:\n- True\n#answer:\nAnswer is {x}",
            answer_annotated="Answer is {x}",
        )

        init_lines = ["$x = range(1, 10, 2)"]
        replacements = {}
        result = annotated_question._get_all_possible_assignments(
            init_lines, replacements
        )

        # Expected: x should have values 1, 3, 5, 7, 9 with step 2
        expected = {"x": [{"x": 1}, {"x": 3}, {"x": 5}, {"x": 7}, {"x": 9}]}
        assert result == expected

    def test_sample_possibility(self):
        """Test with sample possibility."""
        annotated_question = AnnotatedQuestion(
            question="Test question",
            answer="Test answer",
            id_orig=1,
            id_shuffled=1,
            question_annotated="Test template\n#init:\n- $x = sample([10, 20, 30])\n#conditions:\n- True\n#answer:\nAnswer is {x}",
            answer_annotated="Answer is {x}",
        )

        init_lines = ["$x = [10, 20, 30]"]
        replacements = {}
        result = annotated_question._get_all_possible_assignments(
            init_lines, replacements
        )

        # Expected: x should have possible values 10, 20, 30
        expected = {"x": [{"x": 10}, {"x": 20}, {"x": 30}]}
        assert result == expected

    def test_empty_range(self):
        """Test with empty range."""
        annotated_question = AnnotatedQuestion(
            question="Test question",
            answer="Test answer",
            id_orig=1,
            id_shuffled=1,
            question_annotated="Test template\n#init:\n- $x = range(5, 3)\n#conditions:\n- True\n#answer:\nAnswer is {x}",
            answer_annotated="Answer is {x}",
        )

        init_lines = ["$x = range(5, 3)"]
        replacements = {}
        result = annotated_question._get_all_possible_assignments(
            init_lines, replacements
        )

        # Expected: empty list for x since the range is invalid
        assert result == {"x": []}

    def test_with_replacements(self):
        """Test with replacement values."""
        annotated_question = AnnotatedQuestion(
            question="Test question",
            answer="Test answer",
            id_orig=1,
            id_shuffled=1,
            question_annotated="Test template\n#init:\n- $x = range(start, end)\n#conditions:\n- True\n#answer:\nAnswer is {x}",
            answer_annotated="Answer is {x}",
        )

        init_lines = ["$x = range(start, end)"]
        replacements = {"start": 2, "end": 6}
        result = annotated_question._get_all_possible_assignments(
            init_lines, replacements
        )

        # Expected: x should have values 2, 3, 4, 5
        expected = {"x": [{"x": 2}, {"x": 3}, {"x": 4}, {"x": 5}]}
        assert result == expected


def get_template_files():
    template_dirs = [
        (_DATA_ROOT / "dan" / "symbolic", "dan"),
        (_DATA_ROOT / "eng" / "symbolic", "eng"),
    ]

    template_files = []
    for template_dir, language in template_dirs:
        if template_dir.exists():
            for template_file in template_dir.glob("**/*.json"):
                template_files.append((template_file, language))

    return template_files


@pytest.mark.parametrize("template_file,language", get_template_files())
def test_template_formatting_matches_original(template_file, language):
    annotated_question = AnnotatedQuestion.from_json(template_file)
    replacements = load_replacements(language)
    default_assignments = annotated_question.get_default_assignments(replacements)

    formatted_question = annotated_question.format_question(
        default_assignments, language=language
    )
    formatted_answer = annotated_question.format_answer(
        default_assignments, language=language
    )

    assert formatted_question == annotated_question.question, (
        f"Formatted question doesn't match original for {template_file.name}"
    )

    assert formatted_answer == annotated_question.answer, (
        f"Formatted answer doesn't match original for {template_file.name}"
    )


@pytest.mark.parametrize("template_file,language", get_template_files())
def test_default_assignments_are_valid(template_file, language):
    annotated_question = AnnotatedQuestion.from_json(template_file)
    replacements = load_replacements(language)
    default_assignments = annotated_question.get_default_assignments(replacements)
    constrained_lines = annotated_question.constrained_lines
    conditions = annotated_question.conditions

    if not constrained_lines:
        return

    replacements = load_replacements(language)
    all_possible_assignments = annotated_question._get_all_possible_assignments(
        constrained_lines, replacements
    )

    # Check example values are in possible assignments
    for var_name, possible_assignments in all_possible_assignments.items():
        if var_name not in default_assignments:
            continue
        possible_values_for_var = [
            assignment[var_name] for assignment in possible_assignments
        ]

        default_value = default_assignments[var_name]

        if isinstance(default_value, tuple):
            default_value = tuple(
                int(c) if str(c).isnumeric() else str(c) for c in default_value
            )
            assert (
                default_value in possible_values_for_var
                or list(default_value) in possible_values_for_var
            ), (
                f"Example assignment {var_name}={default_value} not found in {possible_values_for_var} for {template_file.name}"
            )
        else:
            val_as_float = try_parse_float(str(default_value))
            val_as_fraction = try_parse_fraction(str(default_value))
            val_as_int = (
                int(default_value)
                if str(default_value).isnumeric()
                or isinstance(default_value, float)
                and default_value.is_integer()
                else default_value
            )

            assert (
                val_as_float in possible_values_for_var
                or str(val_as_float) in possible_values_for_var
                or val_as_fraction in possible_values_for_var
                or str(val_as_fraction) in possible_values_for_var
                or val_as_int in possible_values_for_var
            ), (
                f"Example assignment {var_name}={default_value} not found in {possible_values_for_var} for {template_file.name}"
            )

    # Check conditions are satisfied
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
                        numeric_val = (
                            float(component)
                            if "." in str(component)
                            else int(component)
                        )
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
                cond, {"__builtins__": {}}, EVAL_CONTEXT_HELPERS | temp_combination
            )
            assert condition_result, (
                f"Example assignments {default_assignments} failed condition '{cond}' for {template_file.name}"
            )
        except Exception:
            pass  # Some conditions reference variables not in example_assignments
