import pytest
from conftest import get_template_files

from multilingual_gsm_symbolic.load_data import load_replacements
from multilingual_gsm_symbolic.templates import AnnotatedQuestion
from multilingual_gsm_symbolic.validation import validate_default_assignments


class TestGetAllPossibleAssignments:
    def test_range_expression(self):
        annotated_question = AnnotatedQuestion(
            question="Test question",
            answer="Test answer",
            id_orig=1,
            id_shuffled=1,
            question_annotated="Test template\n#init:\n- $x = range(1, 6)\n#conditions:\n- True\n#answer:\nAnswer is {x}",  # noqa: E501
            answer_annotated="Answer is {x}",
        )

        result = annotated_question._get_all_possible_assignments(["$x = range(1, 6)"], {})
        assert result == {"x": [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}, {"x": 5}]}

    def test_range_with_step(self):
        annotated_question = AnnotatedQuestion(
            question="Test question",
            answer="Test answer",
            id_orig=1,
            id_shuffled=1,
            question_annotated="Test template\n#init:\n- $x = range(1, 10, 2)\n#conditions:\n- True\n#answer:\nAnswer is {x}",  # noqa: E501
            answer_annotated="Answer is {x}",
        )

        result = annotated_question._get_all_possible_assignments(["$x = range(1, 10, 2)"], {})
        assert result == {"x": [{"x": 1}, {"x": 3}, {"x": 5}, {"x": 7}, {"x": 9}]}

    def test_sample_possibility(self):
        annotated_question = AnnotatedQuestion(
            question="Test question",
            answer="Test answer",
            id_orig=1,
            id_shuffled=1,
            question_annotated="Test template\n#init:\n- $x = sample([10, 20, 30])\n#conditions:\n- True\n#answer:\nAnswer is {x}",  # noqa: E501
            answer_annotated="Answer is {x}",
        )

        result = annotated_question._get_all_possible_assignments(["$x = [10, 20, 30]"], {})
        assert result == {"x": [{"x": 10}, {"x": 20}, {"x": 30}]}

    def test_empty_range(self):
        annotated_question = AnnotatedQuestion(
            question="Test question",
            answer="Test answer",
            id_orig=1,
            id_shuffled=1,
            question_annotated="Test template\n#init:\n- $x = range(5, 3)\n#conditions:\n- True\n#answer:\nAnswer is {x}",  # noqa: E501
            answer_annotated="Answer is {x}",
        )

        result = annotated_question._get_all_possible_assignments(["$x = range(5, 3)"], {})
        assert result == {"x": []}

    def test_with_replacements(self):
        annotated_question = AnnotatedQuestion(
            question="Test question",
            answer="Test answer",
            id_orig=1,
            id_shuffled=1,
            question_annotated="Test template\n#init:\n- $x = range(start, end)\n#conditions:\n- True\n#answer:\nAnswer is {x}",  # noqa: E501
            answer_annotated="Answer is {x}",
        )

        result = annotated_question._get_all_possible_assignments(["$x = range(start, end)"], {"start": 2, "end": 6})
        assert result == {"x": [{"x": 2}, {"x": 3}, {"x": 4}, {"x": 5}]}


@pytest.mark.parametrize("template_file", get_template_files())
def test_default_assignments_are_valid(template_file):
    annotated_question = AnnotatedQuestion.from_toml(template_file)
    replacements = load_replacements(annotated_question.language)
    validate_default_assignments(annotated_question, replacements, source=template_file)
