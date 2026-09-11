# /// script
# dependencies = ["inspect-ai", "matplotlib", "numpy", "pandas", "pyarrow", "scipy"]
# ///
"""Regenerate only the two full appendix comparisons from canonical samples."""

import pandas as pd

from visualizegrid import (
    DEFAULT_ANALYSIS,
    DEFAULT_OUT_DIR,
    collect_correction_comparison_rows,
    plot_correction_comparison,
)


def main():
    samples = pd.read_parquet(DEFAULT_ANALYSIS).rename(columns={"id": "sample_id"})
    samples["model_raw"] = samples["model"]
    for old, new, language, path, labels in [
        ("eng", "eng_metric", "eng", "eng_vs_eng_metric_full.png", ("English", "English metric")),
        ("uncorrected_isl", "isl", "isl", "correction_comparison/isl.png", ("Machine translated", "Verified")),
    ]:
        before = samples[samples.language == old].copy()
        after = samples[samples.language == new].copy()
        before["language"] = after["language"] = language
        rows = collect_correction_comparison_rows(before, after, language, 2000, 0)
        plot_correction_comparison(rows, language, DEFAULT_OUT_DIR / path, legend_labels=labels)
        full_page_path = (DEFAULT_OUT_DIR / path).with_stem((DEFAULT_OUT_DIR / path).stem + "_fullpage")
        plot_correction_comparison(rows, language, full_page_path, legend_labels=labels, full_page=True)
        print(f"{path}: {len(rows)} models", flush=True)


if __name__ == "__main__":
    main()
