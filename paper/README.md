# Paper Content

This directory contains evaluation and visualization tooling used for the paper.
It intentionally lives only on the `paper-content` branch.

## Layout

- `eval.yaml`: Inspect task definitions used by `scripts/ucloudeval`.
- `scripts/`: evaluation and analysis entry points.
- `artifacts/figures/`: topic-organized publication figures:
  - `overview/`: introductory and example figures.
  - `accuracy/`: accuracy heatmaps, original vs. synthetic, split degradation, and family scaling plots.
  - `distributions/`: overall distribution, selected ridgeline overview, and `by_model/<family>/` ridgelines.
  - `transfer/`: English-normalized transfer, transfer robustness, reasoning deltas, absolute transfer gaps, compute budget curves (`compute_budget/absolute/` and `compute_budget/relative/`), and `language_features/`.
  - `ablations/`: unit ablations (`english_units/`) and Icelandic human-verification comparisons (`icelandic_validation/`).
  - `number_coverage/`: number-coverage heatmaps and correlation figures.
- `artifacts/analysis/`: supporting CSVs, JSON summaries, and measurements matched by topic:
  - `accuracy/`: model evaluation run summaries (`run_summary.csv`).
  - `transfer/`: compute budget data tables and transfer feature measurements.
  - `number_coverage/`: coverage summary and sample records.
  - `subset_size/`: subset-size stability draws and summaries.
  - `equivalence/`: TOST equivalence test outputs.
- `artifacts/tables/`: paper-ready LaTeX tables.
- `artifacts/transfer_tables/`: canonical evaluation parquets (`analysis.parquet`).

The scripts read evaluation logs from `hf_dataset/logs` by default. That directory
is retained locally and is not copied into this branch.

Run scripts from the repository root, for example:

```bash
uv run paper/scripts/visualizegrid.py
uv run paper/scripts/numbercoverage.py
uv run paper/scripts/language_accuracy_table.py
uv run paper/scripts/transferfeatures.py
uv run paper/scripts/ucloudeval --help
```

## Figure language exclusions

Norwegian (`nob`, `nno`, `nor`) is not human validated and is excluded before
figure aggregation by `plot_config.figure_rows`. Raw evaluation data is retained.
Regenerate plots with `visualizegrid.py`, `visualize_results.py`, `ridgeline.py`,
`qwen_compute_budget.py`, and `numbercoverage.py`.
Use `transferfeatures.py --cached-features` to redraw feature plots from saved
feature measurements without downloading tokenizers.
