#!/usr/bin/env python
"""Generate the README language validation table from template metadata."""

import argparse
import tomllib
from dataclasses import dataclass
from pathlib import Path

START_MARKER = "<!-- LANGUAGE TABLE START -->"
END_MARKER = "<!-- LANGUAGE TABLE END -->"


@dataclass(frozen=True)
class LanguageValidation:
    language: str
    source_language: str
    model: str
    human: str
    error: str


def _completed_value(records: list[dict], key: str) -> str:
    values = [record.get(key) for record in records]
    return "; ".join(sorted(set(values))) if all(values) else ""


def collect_language_validation(templates_root: Path) -> list[LanguageValidation]:
    """Summarize each validation only when it covers every active template."""
    languages = []
    for lang_dir in sorted(path for path in templates_root.iterdir() if path.is_dir()):
        if (lang_dir / "ignore").exists():
            continue
        records = []
        for path in sorted((lang_dir / "symbolic").glob("*.toml")):
            with path.open("rb") as file:
                record = tomllib.load(file)
            if not record.get("ignore"):
                records.append(record)
        if not records:
            continue
        for record in records:
            assert not {"creation", "model", "computationally-validated"}.intersection(record), (
                f"{lang_dir.name}: obsolete metadata"
            )
            assert record["language"] == lang_dir.name, f"{lang_dir.name}: incorrect language"
            if lang_dir.name not in {"eng", "eng_metric"}:
                assert record.get("initial_translation_model"), f"{lang_dir.name}: missing translation model"
                assert record.get("source-language"), f"{lang_dir.name}: missing source language"
        human_values = [record.get("human-validated", "") for record in records]
        human = _completed_value(records, "human-validated")
        if any("in progress" in value.lower() for value in human_values) or (not human and any(human_values)):
            human = "In progress"
        languages.append(
            LanguageValidation(
                language=lang_dir.name,
                source_language=_completed_value(records, "source-language"),
                model=_completed_value(records, "initial_translation_model"),
                human=human,
                error=_completed_value(records, "error-analysis"),
            )
        )
    return languages


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _included_languages(languages: list[LanguageValidation]) -> list[LanguageValidation]:
    return [lang for lang in languages if lang.language != "eng_metric" and (lang.human or lang.language == "eng")]


def render_language_tables(languages: list[LanguageValidation]) -> str:
    lines = [
        "| Language | Computationally validated | Human validated | Error analysis |",
        "| --- | --- | --- | --- |",
    ]
    for lang in _included_languages(languages):
        error = "✓" if lang.error else ""
        lines.append(f"| `{lang.language}` | ✓ | {_markdown_cell(lang.human)} | {error} |")
    return "\n".join(lines)


def update_readme(readme: Path, table_content: str) -> None:
    text = readme.read_text(encoding="utf-8")
    assert text.count(START_MARKER) == text.count(END_MARKER) == 1, "README table markers must occur once"
    before, remainder = text.split(START_MARKER)
    _, after = remainder.split(END_MARKER)
    readme.write_text(f"{before}{START_MARKER}\n{table_content}\n{END_MARKER}{after}", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--templates-root", type=Path, default=Path("src/multilingual_gsm_symbolic/data/templates"))
    args = parser.parse_args()
    languages = collect_language_validation(args.templates_root)
    update_readme(args.readme, render_language_tables(languages))
    print(f"Updated {args.readme} for {len(languages)} languages")


if __name__ == "__main__":
    main()
