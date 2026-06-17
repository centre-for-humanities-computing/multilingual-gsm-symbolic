#!/usr/bin/env python3
import csv
from pathlib import Path

from statsmodels.stats.proportion import proportions_ztest

ANALYSIS_CSV = Path(__file__).resolve().parents[2] / "paper/artifacts/transfer_tables/analysis.csv"
CorrectCount = tuple[int, int]


def correct_count(rows: list[dict[str, object]], split: str) -> CorrectCount:
    values = [row["correct"] for row in rows if row["split"] == split]
    return sum(str(value).strip().lower() in {"1", "true", "correct", "yes"} for value in values), len(values)


def ztest() -> None:
    with ANALYSIS_CSV.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    original, synthetic = correct_count(rows, "original"), correct_count(rows, "synthetic")
    z_statistic, p_value = proportions_ztest([synthetic[0], original[0]], [synthetic[1], original[1]])

    original_rate, synthetic_rate = original[0] / original[1], synthetic[0] / synthetic[1]
    print(
        f"""Two-sample z-test for proportions: synthetic vs original templates
        original:  {original[0]}/{original[1]} = {original_rate:.6f}
        synthetic: {synthetic[0]}/{synthetic[1]} = {synthetic_rate:.6f}
        difference (synthetic - original): {synthetic_rate - original_rate:.6f}
        z statistic: {z_statistic:.6f}
        two-sided p-value: {p_value:.6g}"""
    )


if __name__ == "__main__":
    ztest()
