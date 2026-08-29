# /// script
# dependencies = ["pandas", "pyarrow"]
# ///
"""Write a compact all-model LaTeX table of evaluation results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from plot_config import LANGUAGE_LABELS, language_order, model_sort_key

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS = REPO_ROOT / "paper" / "artifacts" / "transfer_tables" / "analysis.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "paper" / "artifacts" / "tables" / "original_vs_synthetic_accuracy.tex"


@dataclass(frozen=True)
class TableRow:
    model: str
    language: str
    original_accuracy: float
    original_variance: float
    synthetic_accuracy: float
    synthetic_variance: float


def collect_model_rows(
    problems: pd.DataFrame,
    model: str,
    languages: list[str] | None,
) -> list[TableRow]:
    """Collect split-level accuracy and sample variance by language for one model."""
    matching_models = [name for name in problems["model"].unique() if name.lower() == model.lower()]
    if not matching_models:
        available = ", ".join(sorted(problems["model"].unique()))
        raise ValueError(f"Model {model!r} was not found. Available models: {available}")

    resolved_model = matching_models[0]
    model_rows = problems[problems["model"] == resolved_model]
    available_languages = set(model_rows["language"].unique())
    selected_languages = language_order(available_languages, languages)
    missing = [language for language in languages or [] if language not in available_languages]
    if missing:
        raise ValueError(f"No {resolved_model} results for language(s): {', '.join(missing)}")

    rows: list[TableRow] = []
    for language in selected_languages:
        language_rows = model_rows[model_rows["language"] == language]
        original = language_rows[language_rows["split"] == "original"]
        synthetic = language_rows[language_rows["split"] == "synthetic"]
        if original.empty or synthetic.empty:
            raise ValueError(f"Both original and synthetic results are required for {language}.")
        rows.append(
            TableRow(
                model=resolved_model,
                language=language,
                original_accuracy=float(original["correct"].mean()),
                original_variance=float(original["correct"].var(ddof=1)),
                synthetic_accuracy=float(synthetic["correct"].mean()),
                synthetic_variance=float(synthetic["correct"].var(ddof=1)),
            )
        )
    return rows


def collect_rows(
    problems: pd.DataFrame,
    models: list[str] | None,
    languages: list[str] | None,
) -> list[TableRow]:
    """Collect every selected model-language combination."""
    available_models = sorted(problems["model"].dropna().unique(), key=model_sort_key)
    if models:
        lookup = {model.lower(): model for model in available_models}
        missing = [model for model in models if model.lower() not in lookup]
        if missing:
            available = ", ".join(available_models)
            raise ValueError(f"Model(s) not found: {', '.join(missing)}. Available models: {available}")
        selected_models = list(dict.fromkeys(lookup[model.lower()] for model in models))
    else:
        selected_models = available_models

    return [row for model in selected_models for row in collect_model_rows(problems, model, languages)]


def latex_escape(value: str) -> str:
    """Escape plain text for use in LaTeX."""
    replacements = {
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
    }
    return "".join(replacements.get(character, character) for character in value)


def render_table(
    rows: list[TableRow],
    caption: str | None = None,
    label: str = "tab:original-vs-synthetic-accuracy",
) -> str:
    """Render a compact booktabs/longtable-compatible LaTeX table."""
    if not rows:
        raise ValueError("At least one model is required.")

    models = list(dict.fromkeys(row.model for row in rows))
    scope = models[0] if len(models) == 1 else f"all {len(models)} models"
    caption_text = caption or (
        f"Original and synthetic exact-answer accuracy by model and language for {scope}. "
        "Variance is the sample variance of the scored examples within each split."
    )
    header = (
        r"Model & Language & \shortstack{Original\\accuracy} & \shortstack{Original\\variance} & "
        r"\shortstack{Synthetic\\accuracy} & \shortstack{Synthetic\\variance} \\"
    )
    lines = [
        r"% Requires \usepackage{longtable}; all other commands use packages already in the paper.",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\begin{longtable}{p{0.27\linewidth}lrrrr}",
        f"\\caption{{{latex_escape(caption_text)}}}\\label{{{label}}}" + r" \\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{6}{c}{\tablename\ \thetable{} -- continued from previous page} \\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{6}{r}{Continued on next page} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    previous_model: str | None = None
    for row in rows:
        if previous_model is not None and row.model != previous_model:
            lines.append(r"\addlinespace[2pt]")
        model = latex_escape(row.model) if row.model != previous_model else ""
        lines.append(
            f"{model} & {latex_escape(LANGUAGE_LABELS.get(row.language, row.language))} & "
            f"{row.original_accuracy * 100:.1f}\\% & "
            f"{row.original_variance:.4f} & {row.synthetic_accuracy * 100:.1f}\\% & "
            f"{row.synthetic_variance:.4f} \\\\"
        )
        previous_model = row.model
    lines.extend([r"\end{longtable}", r"\endgroup", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument(
        "--model",
        nargs="+",
        help="Optional model names; defaults to every available model.",
    )
    parser.add_argument("--languages", nargs="+", help="Optional language codes; defaults to every available language.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--caption", help="Optional replacement table caption.")
    parser.add_argument("--label", default="tab:original-vs-synthetic-accuracy")
    args = parser.parse_args()

    problems = pd.read_parquet(
        args.analysis,
        columns=["model", "language", "split", "source_id", "correct"],
    )
    problems = problems[problems["language"] != "uncorrected_isl"]
    rows = collect_rows(problems, args.model, args.languages)
    output = render_table(rows, args.caption, args.label)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8", newline="\n")
    model_count = len({row.model for row in rows})
    print(f"Saved {args.output} ({model_count} models, {len(rows)} model-language rows)")


if __name__ == "__main__":
    main()
