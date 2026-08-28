from pathlib import Path

from multilingual_gsm_symbolic.templates import AnnotatedQuestion


def _pan_template_files() -> list[Path]:
    root = Path(__file__).resolve().parents[1] / "src" / "multilingual_gsm_symbolic" / "data" / "templates" / "pan"
    return sorted((root / "symbolic").glob("*.toml"))


def test_pan_templates_load_and_generate():
    files = _pan_template_files()
    assert len(files) >= 25

    for path in files:
        template = AnnotatedQuestion.from_toml(path)
        questions = template.generate_questions(n=1, seed=0, verbose=False)
        assert len(questions) == 1, path

        question = questions[0]
        assert isinstance(question.question, str) and question.question.strip(), path
        assert isinstance(question.answer, str) and question.answer.strip(), path
