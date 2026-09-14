import json
from pathlib import Path

import pytest

from multilingual_gsm_symbolic.load_data import _DATA_ROOT
from multilingual_gsm_symbolic.templates import AnnotatedQuestion
from scripts.update_readme_table import (
    END_MARKER,
    START_MARKER,
    collect_language_validation,
    render_language_tables,
    render_latex_table,
    update_readme,
)


def _template_data(**metadata: str) -> dict:
    return {
        "question": "Question",
        "answer": "Answer",
        "id_orig": 0,
        "id_shuffled": 0,
        "question_annotated": "Question\n#init:\n- $x = [1]\n#answer: x",
        "answer_annotated": "Answer",
        **metadata,
    }


def test_loader_reads_current_schema_and_rejects_obsolete_fields(tmp_path: Path) -> None:
    path = tmp_path / "template.json"
    data = _template_data(
        **{
            "source-language": "eng",
            "initial_translation_model": "example/model",
            "computationally-validated": "test suite passes",
            "human-validated": "by a native speaker",
            "error-analysis": "reviewed with Claude Opus",
        }
    )
    path.write_text(json.dumps(data), encoding="utf-8")
    template = AnnotatedQuestion.from_json(path)
    assert template.source_language == "eng"
    assert template.initial_translation_model == "example/model"
    assert template.computationally_validated == "test suite passes"
    assert template.human_validated == "by a native speaker"
    assert template.error_analysis == "reviewed with Claude Opus"
    for key in ("model", "creation"):
        path.write_text(json.dumps({**data, key: "obsolete"}), encoding="utf-8")
        with pytest.raises(TypeError, match=key):
            AnnotatedQuestion.from_json(path)


def test_repository_metadata() -> None:
    languages = collect_language_validation(_DATA_ROOT)
    assert languages


def test_tables_only_report_whole_language_validation(tmp_path: Path) -> None:
    for language in ("eng", "eng_metric", "spa", "dan"):
        symbolic = tmp_path / language / "symbolic"
        symbolic.mkdir(parents=True)
        for index in range(2):
            text = f'language = "{language}"\ncomputationally-validated = "test suite passes"\n'
            if language not in {"eng", "eng_metric"}:
                text += 'source-language = "eng"\ninitial_translation_model = "example/model"\n'
            if language == "dan" or (language == "spa" and index == 0):
                text += 'human-validated = "by a native speaker"\nerror-analysis = "inspected"\n'
            (symbolic / f"{index:04}.toml").write_text(text, encoding="utf-8")
    # Ignored templates do not affect a language's completion.
    (tmp_path / "dan/symbolic/0002.toml").write_text("ignore = true\n", encoding="utf-8")
    languages = collect_language_validation(tmp_path)
    rendered = render_language_tables(languages)
    overview, full = rendered.split("<details>")
    assert "`dan`" in overview and "`spa`" not in overview
    assert "| `spa` | example/model | test suite passes |  |  |" in full
    assert "Partial" not in rendered and "Source language" not in rendered
    latex = render_latex_table(languages)
    assert r"eng\_metric & Yes &  & " in latex
    assert "spa & Yes &  & " in latex
    # Missing translation provenance is an error, not a blank/unknown table entry.
    (tmp_path / "spa/symbolic/0000.toml").write_text('language = "spa"\n', encoding="utf-8")
    with pytest.raises(KeyError, match="initial_translation_model"):
        collect_language_validation(tmp_path)


def test_update_readme_preserves_surrounding_content(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"before\n{START_MARKER}\nold\n{END_MARKER}\nafter\n", encoding="utf-8")
    update_readme(readme, "new")
    assert readme.read_text(encoding="utf-8") == f"before\n{START_MARKER}\nnew\n{END_MARKER}\nafter\n"
