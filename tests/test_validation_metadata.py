from dataclasses import asdict
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


def test_repository_metadata() -> None:
    languages = collect_language_validation(_DATA_ROOT)
    assert next(lang.human for lang in languages if lang.language == "dan") == "by three native speakers"
    template = AnnotatedQuestion.from_toml(_DATA_ROOT / "afr/symbolic/0000.toml")
    assert template.human_validated == template.error_analysis == "none"
    assert "computationally_validated" not in asdict(template)


def test_tables_distinguish_none_in_progress_and_complete(tmp_path: Path) -> None:
    for language in ("eng", "eng_metric", "spa", "dan", "deu", "fra"):
        symbolic = tmp_path / language / "symbolic"
        symbolic.mkdir(parents=True)
        for index in range(2):
            original = language in {"eng", "eng_metric"}
            source, model = ("none", "none") if original else ("eng", "example/model")
            human = "none"
            if language == "dan" or (language == "spa" and index == 0):
                human = "by three native speakers"
            if language == "fra":
                human = "in progress"
            text = (
                f'language = "{language}"\nsource-language = "{source}"\n'
                f'initial_translation_model = "{model}"\nhuman-validated = "{human}"\n'
                'error-analysis = "none"\n'
            )
            (symbolic / f"{index:04}.toml").write_text(text, encoding="utf-8")
    (tmp_path / "dan/symbolic/0002.toml").write_text("ignore = true\n", encoding="utf-8")
    languages = collect_language_validation(tmp_path)
    rendered = render_language_tables(languages)
    assert "| `spa` | ✓ | In progress |  |" in rendered
    assert "| `fra` | ✓ | In progress |  |" in rendered
    assert "| `dan` | ✓ | by three native speakers |  |" in rendered
    assert "`eng`" in rendered
    assert "`deu`" not in rendered and "eng_metric" not in rendered
    assert "<details>" not in rendered and "Source language" not in rendered
    latex = render_latex_table(languages)
    assert latex.count("Language &") == 1
    assert "three native speakers" in latex and "In progress" in latex
    assert "Src" not in latex and "deu &" not in latex
    assert r"\begin{tabular}{llccl}" in latex
    (tmp_path / "spa/symbolic/0000.toml").write_text('language = "spa"\n', encoding="utf-8")
    with pytest.raises(KeyError, match="initial_translation_model"):
        collect_language_validation(tmp_path)


def test_update_readme_preserves_surrounding_content(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"before\n{START_MARKER}\nold\n{END_MARKER}\nafter\n", encoding="utf-8")
    update_readme(readme, "new")
    assert readme.read_text(encoding="utf-8") == f"before\n{START_MARKER}\nnew\n{END_MARKER}\nafter\n"
