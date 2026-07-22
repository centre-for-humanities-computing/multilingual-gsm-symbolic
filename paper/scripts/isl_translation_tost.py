#!/usr/bin/env python3
# /// script
# dependencies = ["pandas", "pyarrow", "scipy"]
# ///
"""Paired TOST for machine-translated vs human-verified Icelandic templates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_1samp


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "paper/artifacts/transfer_tables/analysis.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "paper/artifacts/equivalence/isl_translation_tost.txt"


def model_scores(
    path: Path,
    split: str,
    left_language: str,
    right_language: str,
    left_label: str,
    right_label: str,
) -> pd.DataFrame:
    data = pd.read_parquet(path, columns=["language", "split", "model", "correct"])
    data = data.loc[
        (data["split"] == split) & data["language"].isin([left_language, right_language])
    ]
    scores = data.groupby(["model", "language"])["correct"].mean().unstack()
    scores = scores.rename(
        columns={left_language: left_label, right_language: right_label}
    ).dropna(subset=[left_label, right_label])
    if len(scores) < 2:
        raise ValueError("At least two models with both Icelandic template variants are required")
    return scores.sort_index()


def tost_report(
    scores: pd.DataFrame,
    margin_pp: float,
    split: str,
    title: str,
    left_label: str,
    right_label: str,
) -> str:
    margin = margin_pp / 100
    differences = scores[right_label] - scores[left_label]

    # TOST rejects both non-equivalence null hypotheses: mean difference at or
    # below -margin, and mean difference at or above +margin.
    lower = ttest_1samp(differences, -margin, alternative="greater")
    upper = ttest_1samp(differences, margin, alternative="less")
    tost_p = max(float(lower.pvalue), float(upper.pvalue))
    equivalent = tost_p < 0.05

    table = scores.assign(difference=differences).to_string(
        float_format=lambda value: f"{value:.6f}"
    )
    table = "\n".join(line.rstrip() for line in table.splitlines())
    lines = [
        f"Paired TOST equivalence test: {title}",
        f"Split: {split}",
        f"Models paired: {len(scores)}",
        f"Equivalence margin: +/-{margin_pp:g} percentage points",
        f"Difference: {right_label} minus {left_label} accuracy",
        f"{left_label} mean: {scores[left_label].mean():.6f}",
        f"{right_label} mean: {scores[right_label].mean():.6f}",
        f"Mean difference: {differences.mean():.6f} ({differences.mean() * 100:+.3f} pp)",
        f"Lower-bound one-sided t: {lower.statistic:.6f}; p = {lower.pvalue:.6g}",
        f"Upper-bound one-sided t: {upper.statistic:.6f}; p = {upper.pvalue:.6g}",
        f"TOST p-value (maximum one-sided p): {tost_p:.6g}",
        f"Equivalent at alpha=0.05: {'yes' if equivalent else 'no'}",
        "",
        "Per-model accuracies",
        table,
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="synthetic")
    parser.add_argument("--margin-pp", type=float, default=0.5)
    args = parser.parse_args()
    if args.margin_pp <= 0:
        parser.error("--margin-pp must be positive")

    isl_scores = model_scores(
        args.input,
        args.split,
        "isl",
        "uncorrected_isl",
        "human_verified",
        "machine_translated",
    )
    english_scores = model_scores(
        args.input,
        args.split,
        "eng",
        "eng_metric",
        "eng",
        "eng_metric",
    )
    report = "\n\n".join(
        [
            tost_report(
                isl_scores,
                args.margin_pp,
                args.split,
                "Icelandic machine-translated vs human-verified templates",
                "human_verified",
                "machine_translated",
            ),
            tost_report(
                english_scores,
                args.margin_pp,
                args.split,
                "eng vs eng_metric templates",
                "eng",
                "eng_metric",
            ),
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
