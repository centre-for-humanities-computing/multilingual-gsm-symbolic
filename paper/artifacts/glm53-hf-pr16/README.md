# GLM-5.3 evaluation results

All 30 GLM-5.3 Inspect `.eval` files on the `paper-content` branch are in [`logs/`](logs/).
These 30 files are the complete, authoritative evaluation logs used for `summary.csv` and `table.tex`.

`manifest.json`, `summary.csv`, and `errors.json` describe the completed GLM-5.3 run, and `eval.stdout` contains its console output.
Run `src/scripts/verify_results.py` from the repository to check the 30 complete logs against the manifest and summary.

Evaluation run completed 2026-09-19 across all 15 languages in the multilingual GSM-Symbolic benchmark.
The 30 Inspect `.eval` files contain all 31,500 attempted samples.
All 31,500 samples (1,500 original across 15 languages @ 100 samples; 30,000 synthetic across 15 languages @ 2,000 samples) are fully scored with zero API errors, parse failures, or truncations.
The model evaluated is `zai-org/GLM-5.3` served through the SDU Chat Completions endpoint (`reasoning_effort: max`, `temperature: 1.0`).
The evaluation dataset is pinned to commit `d34a0ffcb2851179ccac891807fa3a29ccd896f6`.

