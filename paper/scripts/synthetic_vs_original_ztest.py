#!/usr/bin/env python3
# /// script
# dependencies = ["pandas", "pyarrow", "statsmodels"]
# ///
from pathlib import Path

import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

ANALYSIS_PARQUET = Path(__file__).resolve().parents[2] / "paper/artifacts/transfer_tables/analysis.parquet"
CorrectCount = tuple[int, int]


def correct_count(analysis: pd.DataFrame, split: str) -> CorrectCount:
    values = analysis.loc[analysis["split"] == split, "correct"]
    return int(values.sum()), len(values)


def ztest() -> None:
    analysis = pd.read_parquet(ANALYSIS_PARQUET, columns=["split", "correct"])
    original = correct_count(analysis, "original")
    synthetic = correct_count(analysis, "synthetic")
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
