#!/usr/bin/env python
"""Generate README and LaTeX language validation tables from template metadata."""

import argparse
import tomllib
from dataclasses import dataclass
from math import ceil
from pathlib import Path

START_MARKER = "<!-- LANGUAGE TABLE START -->"
END_MARKER = "<!-- LANGUAGE TABLE END -->"


@dataclass(frozen=True)
class LanguageValidation:
    language: str
    source_language: str
    model: str
    computational: str
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
            assert "creation" not in record and "model" not in record, f"{lang_dir.name}: obsolete metadata"
            assert record["language"] == lang_dir.name, f"{lang_dir.name}: incorrect language"
            if lang_dir.name not in {"eng", "eng_metric"}:
                assert record["initial_translation_model"], f"{lang_dir.name}: missing translation model"
                assert record["source-language"], f"{lang_dir.name}: missing source language"
        languages.append(
            LanguageValidation(
                language=lang_dir.name,
                source_language=_completed_value(records, "source-language"),
                model=_completed_value(records, "initial_translation_model"),
                computational=_completed_value(records, "computationally-validated"),
                human=_completed_value(records, "human-validated"),
                error=_completed_value(records, "error-analysis"),
            )
        )
    return languages


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _format_source_language(source: str) -> str:
    if not source:
        return ""
    langs = [s.strip() for s in source.split(";")]
    return ", ".join(f"`{l}`" for l in langs)


def _table(languages: list[LanguageValidation], *, overview: bool = False) -> str:
    headings = ["Language", "Computationally validated", "Human validated", "Error analysis"]
    if not overview:
        headings = [
            "Language",
            "Source language",
            "Initial translation model",
            "Computationally validated",
            "Human validated",
            "Error analysis",
        ]
    lines = ["| " + " | ".join(headings) + " |", "| " + " | ".join(["---"] * len(headings)) + " |"]
    for lang in languages:
        comp = "✓" if lang.computational else ""
        human = lang.human
        error = "✓" if lang.error else ""
        if overview:
            values = [comp, _markdown_cell(human), error]
        else:
            values = [
                _format_source_language(lang.source_language),
                _markdown_cell(lang.model),
                comp,
                _markdown_cell(human),
                error,
            ]
        lines.append("| " + " | ".join([f"`{lang.language}`", *values]) + " |")
    return "\n".join(lines)


def render_language_tables(languages: list[LanguageValidation]) -> str:
    filtered_languages = [lang for lang in languages if lang.language != "eng_metric"]
    validated = [lang for lang in filtered_languages if lang.human or lang.language == "eng"]
    return "\n".join(
        [
            "The following languages are validated:",
            "",
            _table(validated, overview=True),
            "",
            "<details>",
            "<summary>Full details</summary>",
            "",
            _table(filtered_languages, overview=False),
            "",
            "</details>",
        ]
    )


def _latex_cell(value: str) -> str:
    escapes = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\n": " ",
    }
    return "".join(escapes.get(char, char) for char in value)


def render_latex_table(languages: list[LanguageValidation]) -> str:
    """A compact 2-column multipage paper table; requires longtable and amssymb."""
    filtered = [l for l in languages if l.language != "eng_metric"]
    mid = ceil(len(filtered) / 2)
    left = filtered[:mid]
    right = filtered[mid:]

    def format_half(lang: LanguageValidation | None) -> list[str]:
        if not lang:
            return ["", "", "", "", "", ""]
        src = _latex_cell(lang.source_language.replace(";", ", "))
        model = _latex_cell(lang.model)
        comp = r"$\checkmark$" if lang.computational else ""
        human = _latex_cell(lang.human)
        error = _latex_cell(lang.error)
        return [_latex_cell(lang.language), src, model, comp, human, error]

    lines = [
        "% Generated by src/scripts/update_readme_table.py; do not edit.",
        r"\begin{longtable}{llp{2.3cm}cp{1.6cm}p{1.8cm}@{\quad}|@{\quad}llp{2.3cm}cp{1.6cm}p{1.8cm}}",
        r"\caption{Language validation metadata across all languages.}\label{tab:language-validation}\\",
        r"\hline",
        r"Lang & Src & Model & Comp & Human & Error & Lang & Src & Model & Comp & Human & Error \\",
        r"\hline",
        r"\endfirsthead",
        r"\hline",
        r"Lang & Src & Model & Comp & Human & Error & Lang & Src & Model & Comp & Human & Error \\",
        r"\hline",
        r"\endhead",
        r"\hline",
        r"\endfoot",
        r"\hline",
        r"\endlastfoot",
    ]
    for i in range(mid):
        l_entry = format_half(left[i])
        r_entry = format_half(right[i] if i < len(right) else None)
        lines.append(" & ".join(l_entry + r_entry) + r" \\")
    lines.append(r"\end{longtable}")
    return "\n".join(lines) + "\n"


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
    parser.add_argument("--latex", type=Path, default=Path("docs/language_validation.tex"))
    args = parser.parse_args()
    languages = collect_language_validation(args.templates_root)
    update_readme(args.readme, render_language_tables(languages))
    args.latex.parent.mkdir(parents=True, exist_ok=True)
    args.latex.write_text(render_latex_table(languages), encoding="utf-8", newline="\n")
    print(f"Updated {args.readme} and {args.latex} for {len(languages)} languages")


if __name__ == "__main__":
    main()
