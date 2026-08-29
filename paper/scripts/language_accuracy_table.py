# /// script
# dependencies = ["pandas", "pyarrow"]
# ///
"""Write an all-language LaTeX table of original and synthetic results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from plot_config import LANGUAGE_LABELS, language_order

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS = REPO_ROOT / "paper" / "artifacts" / "transfer_tables" / "analysis.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "paper" / "artifacts" / "tables" / "original_vs_synthetic_accuracy.tex"
DEFAULT_MODEL = "Qwen2.5-7B-Instruct"


@dataclass(frozen=True)
class TableRow:
    language: str
    original_accuracy: float
    original_variance: float
    synthetic_accuracy: float
    synthetic_variance: float


def collect_rows(
    problems: pd.DataFrame,
    model: str,
    languages: list[str] | None,
) -> tuple[str, list[TableRow]]:
    """Collect split-level accuracy and sample variance for every language."""
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
                language=language,
                original_accuracy=float(original["correct"].mean()),
                original_variance=float(original["correct"].var(ddof=1)),
                synthetic_accuracy=float(synthetic["correct"].mean()),
                synthetic_variance=float(synthetic["correct"].var(ddof=1)),
            )
        )
    return resolved_model, rows


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
    model: str,
    rows: list[TableRow],
    caption: str | None = None,
    label: str = "tab:original-vs-synthetic-accuracy",
) -> str:
    """Render a booktabs-compatible LaTeX table fragment."""
    if not rows:
        raise ValueError("At least one table row is required.")

    caption_text = caption or (
        f"Original and synthetic exact-answer accuracy by language for {model}. "
        "Variance is the sample variance of the scored examples within each split."
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{latex_escape(caption_text)}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Language & \shortstack{Original\\accuracy} & \shortstack{Original\\variance} & "
        r"\shortstack{Synthetic\\accuracy} & \shortstack{Synthetic\\variance} \\",
        r"\midrule",
    ]

    for row in rows:
        language = latex_escape(LANGUAGE_LABELS.get(row.language, row.language))
        lines.append(
            f"{language} & {row.original_accuracy * 100:.1f}\\% & {row.original_variance:.4f} & "
            f"{row.synthetic_accuracy * 100:.1f}\\% & {row.synthetic_variance:.4f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip",
            r"\begin{minipage}{\linewidth}",
            r"\footnotesize",
            r"\textit{Note:} Accuracy is exact-answer accuracy. Variance is computed over scored examples.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
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
    model, rows = collect_rows(problems, args.model, args.languages)
    output = render_table(model, rows, args.caption, args.label)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8", newline="\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
