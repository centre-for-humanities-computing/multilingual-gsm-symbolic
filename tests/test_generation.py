import pytest
from conftest import get_lightly_constrained_template_files, get_unconstrained_template_files

from multilingual_gsm_symbolic.gsm_parser import AnnotatedQuestion, Question
from multilingual_gsm_symbolic.load_data import load_replacements

_TEMPLATES = get_unconstrained_template_files() + get_lightly_constrained_template_files()


@pytest.mark.parametrize("template_file,language", _TEMPLATES)
def test_generate_questions_returns_questions(template_file, language):
    template = AnnotatedQuestion.from_json(template_file)
    replacements = load_replacements(language)
    questions = template.generate_questions(n=3, language=language, replacements=replacements, verbose=False)
    assert len(questions) > 0
    assert all(isinstance(q, Question) for q in questions)


@pytest.mark.parametrize("template_file,language", _TEMPLATES)
def test_generate_questions_non_empty_strings(template_file, language):
    template = AnnotatedQuestion.from_json(template_file)
    replacements = load_replacements(language)
    questions = template.generate_questions(n=3, language=language, replacements=replacements, verbose=False)
    for q in questions:
        assert isinstance(q.question, str) and q.question.strip()
        assert isinstance(q.answer, str) and q.answer.strip()


@pytest.mark.parametrize("template_file,language", _TEMPLATES)
def test_generate_questions_ids(template_file, language):
    template = AnnotatedQuestion.from_json(template_file)
    replacements = load_replacements(language)
    questions = template.generate_questions(n=3, language=language, replacements=replacements, verbose=False)
    for q in questions:
        assert q.id_orig == template.id_orig
        assert q.id_shuffled == template.id_shuffled
