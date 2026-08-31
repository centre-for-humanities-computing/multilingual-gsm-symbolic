#!/usr/bin/env python
"""Update README language validation tables from structured template metadata."""

import argparse
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

START_MARKER = "<!-- LANGUAGE TABLE START -->"
END_MARKER = "<!-- LANGUAGE TABLE END -->"
VALIDATION_KEYS = {
    "computational": "computationally-validated",
    "human": "human-validated",
    "error": "error-analysis",
}


@dataclass(frozen=True)
class LanguageValidation:
    language: str
    template_count: int
    source_language: str
    model: str
    computational: str
    human: str
    error: str
    fully_computationally_validated: bool


def _display_value(values: list[str | None]) -> str:
    present = [value for value in values if value]
    if not present:
        return "—"
    unique = sorted(set(present))
    if len(present) == len(values) and len(unique) == 1:
        return unique[0]
    if len(unique) == 1:
        return f"Partial ({len(present)}/{len(values)} templates): {unique[0]}"
    counts = Counter(present)
    summary = ", ".join(f"{value} ({count}/{len(values)})" for value, count in sorted(counts.items()))
    return f"Mixed: {summary}"


def collect_language_validation(templates_root: Path) -> list[LanguageValidation]:
    """Collect validation metadata from active templates, grouped by language."""
    languages = []
    for lang_dir in sorted(path for path in templates_root.iterdir() if path.is_dir()):
        symbolic_dir = lang_dir / "symbolic"
        if not symbolic_dir.exists() or (lang_dir / "ignore").exists():
            continue

        records = []
        for template_path in sorted(symbolic_dir.glob("*.toml")):
            with template_path.open("rb") as file:
                record = tomllib.load(file)
            if not record.get("ignore"):
                records.append(record)
        if not records:
            continue

        computational = [record.get(VALIDATION_KEYS["computational"]) for record in records]
        human = [record.get(VALIDATION_KEYS["human"]) for record in records]
        error = [record.get(VALIDATION_KEYS["error"]) for record in records]
        original_derived = [record.get("creation", "").startswith("derived from GSM-Symbolic") for record in records]
        source_languages = [
            record.get("source-language") or ("original-derived" if original else None)
            for record, original in zip(records, original_derived)
        ]
        models = [
            record.get("model") or ("not applicable" if original else None)
            for record, original in zip(records, original_derived)
        ]
        languages.append(
            LanguageValidation(
                language=lang_dir.name,
                template_count=len(records),
                source_language=_display_value(source_languages),
                model=_display_value(models),
                computational=_display_value(computational),
                human=_display_value(human),
                error=_display_value(error),
                fully_computationally_validated=all(computational),
            )
        )
    return languages


def _table(languages: list[LanguageValidation]) -> str:
    lines = [
        "| Language | Source language | Model | Computationally validated | Human validated | Error analysis |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| `{lang.language}` | {_markdown_cell(lang.source_language)} | {_markdown_cell(lang.model)} | "
        f"{_markdown_cell(lang.computational)} | {_markdown_cell(lang.human)} | {_markdown_cell(lang.error)} |"
        for lang in languages
    )
    return "\n".join(lines)


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def render_language_tables(languages: list[LanguageValidation]) -> str:
    """Render complete and incomplete language validation tables as Markdown."""
    validated = [lang for lang in languages if lang.fully_computationally_validated]
    incomplete = [lang for lang in languages if not lang.fully_computationally_validated]
    sections = ["The following languages are computationally validated:", "", _table(validated)]
    if incomplete:
        sections.extend(
            [
                "",
                "<details>",
                "<summary>Languages with incomplete computational validation</summary>",
                "",
                _table(incomplete),
                "",
                "</details>",
            ]
        )
    return "\n".join(sections)


def update_readme(readme: Path, table_content: str) -> None:
    """Replace the content between the README language-table markers."""
    text = readme.read_text(encoding="utf-8")
    if text.count(START_MARKER) != 1 or text.count(END_MARKER) != 1:
        raise ValueError("README must contain exactly one start marker and one end marker")
    before, remainder = text.split(START_MARKER, 1)
    _, after = remainder.split(END_MARKER, 1)
    readme.write_text(
        f"{before}{START_MARKER}\n{table_content}\n{END_MARKER}{after}",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument(
        "--templates-root",
        type=Path,
        default=Path("src/multilingual_gsm_symbolic/data/templates"),
    )
    args = parser.parse_args()
    languages = collect_language_validation(args.templates_root)
    update_readme(args.readme, render_language_tables(languages))
    print(f"Updated {args.readme} with validation metadata for {len(languages)} languages")


if __name__ == "__main__":
    main()
