# Paper Content

This directory contains evaluation and visualization tooling used for the paper.
It intentionally lives only on the `paper-content` branch.

## Layout

- `eval.yaml`: Inspect task definitions used by `scripts/ucloudeval`.
- `scripts/`: evaluation and analysis entry points.
- `artifacts/figures/`: general plots, model-grid outputs, and transfer-feature outputs.
- `artifacts/tables/`: paper-ready LaTeX tables.
- `artifacts/prompt_number_coverage/`: number-coverage plots and analysis tables.

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
