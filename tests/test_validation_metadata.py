import json
import tomllib
from pathlib import Path

import pytest

from multilingual_gsm_symbolic.load_data import _DATA_ROOT
from multilingual_gsm_symbolic.templates import AnnotatedQuestion
from scripts.update_readme_table import (
    END_MARKER,
    START_MARKER,
    collect_language_validation,
    render_language_tables,
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


def test_json_loader_accepts_documented_hyphenated_validation_keys(tmp_path: Path) -> None:
    path = tmp_path / "template.json"
    path.write_text(
        json.dumps(
            _template_data(
                **{
                    "source-language": "eng",
                    "model": "example/model",
                    "computationally-validated": "test suite passes",
                    "human-validated": "by a native speaker",
                    "error-analysis": "performed using anthropic/claude-opus-4-8, with errors manually inspected",
                }
            )
        ),
        encoding="utf-8",
    )

    template = AnnotatedQuestion.from_json(path)

    assert template.source_language == "eng"
    assert template.model == "example/model"
    assert template.computationally_validated == "test suite passes"
    assert template.human_validated == "by a native speaker"
    assert template.error_analysis == "performed using anthropic/claude-opus-4-8, with errors manually inspected"


def test_loader_rejects_duplicate_serialized_and_python_validation_keys(tmp_path: Path) -> None:
    path = tmp_path / "template.json"
    path.write_text(
        json.dumps(
            _template_data(
                **{
                    "computationally-validated": "test suite passes",
                    "computationally_validated": "other value",
                }
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="both"):
        AnnotatedQuestion.from_json(path)


def test_repository_templates_have_complete_validation_metadata() -> None:
    records: dict[str, list[tuple[Path, dict]]] = {}
    for language_dir in sorted(path for path in _DATA_ROOT.iterdir() if path.is_dir()):
        symbolic_dir = language_dir / "symbolic"
        if symbolic_dir.is_dir():
            records[language_dir.name] = [
                (path, tomllib.load(path.open("rb"))) for path in sorted(symbolic_dir.glob("*.toml"))
            ]

    human_validated_languages = {
        language
        for language, templates in records.items()
        if any(template.get("human-validated") for _, template in templates)
    }
    errors = []
    for language, templates in records.items():
        for path, template in templates:
            creation = template.get("creation", "")
            ignored_machine_translation = template.get("ignore") and (
                creation == "machine-translated" or set(template) == {"ignore"}
            )
            if ignored_machine_translation:
                continue

            required = {"creation", "language", "computationally-validated"}
            if creation == "machine-translated" or creation.startswith("derived from GSM-Symbolic"):
                required.update({"source-language", "model"})
            if language in human_validated_languages:
                required.update({"human-validated", "error-analysis"})

            missing = sorted(field for field in required if not template.get(field))
            if missing:
                errors.append(f"{path.relative_to(_DATA_ROOT)}: missing {', '.join(missing)}")

    assert not errors, "Incomplete template validation metadata:\n" + "\n".join(errors)


def test_readme_table_separates_complete_and_partial_validation(tmp_path: Path) -> None:
    root = tmp_path / "templates"
    for language, validation_values in {"eng": [True, True], "spa": [True, False]}.items():
        symbolic = root / language / "symbolic"
        symbolic.mkdir(parents=True)
        for index, validated in enumerate(validation_values):
            metadata = (
                'source-language = "eng"\n'
                'model = "example/model"\n'
                'human-validated = "by a native speaker"\n'
                'error-analysis = "performed using anthropic/claude-opus-4-8, with errors manually inspected"\n'
            )
            if validated:
                metadata += 'computationally-validated = "test suite passes"\n'
            (symbolic / f"{index:04}.toml").write_text(metadata, encoding="utf-8")
    original = root / "original" / "symbolic"
    original.mkdir(parents=True)
    (original / "0000.toml").write_text(
        'creation = "derived from GSM-Symbolic (Apple) templates"\ncomputationally-validated = "test suite passes"\n',
        encoding="utf-8",
    )

    rendered = render_language_tables(collect_language_validation(root))

    assert (
        "| `eng` | eng | example/model | test suite passes | by a native speaker | "
        "performed using anthropic/claude-opus-4-8, with errors manually inspected |" in rendered
    )
    assert "| `original` | original-derived | not applicable | test suite passes | — | — |" in rendered
    assert "Languages with incomplete computational validation" in rendered
    assert (
        "| `spa` | eng | example/model | Partial (1/2 templates): test suite passes | by a native speaker | "
        "performed using anthropic/claude-opus-4-8, with errors manually inspected |" in rendered
    )


def test_update_readme_replaces_only_marker_contents(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"before\n{START_MARKER}\nold\n{END_MARKER}\nafter\n", encoding="utf-8")

    update_readme(readme, "new")

    assert readme.read_text(encoding="utf-8") == f"before\n{START_MARKER}\nnew\n{END_MARKER}\nafter\n"
