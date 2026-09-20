# GLM-5.3 saturation analysis using the paper's HF evaluations

Uses the published original and synthetic splits and native Inspect evaluation
definitions from [HF PR #16](https://huggingface.co/datasets/danish-foundation-models/multilingual-gsm-symbolic/discussions/16),
`refs/pr/16`, pinned to commit `d34a0ffcb2851179ccac891807fa3a29ccd896f6`.
No problems are generated locally. Localized prompts and the built-in `math`
scorer come directly from that commit's `eval.yaml`.

The 15 completed human-validated languages in the paper table are Arabic,
Danish, German, English, Estonian, French, Hindi, Icelandic, Italian, Japanese,
Marathi, Dutch, Russian, Ukrainian and Chinese. The runner checks 100 unique
source IDs in each original split, 20 synthetic variants per source, and
matching original/synthetic source IDs: **30 tasks, 31,500 model requests**.
It excludes Norwegian, Urdu, Swedish and English metric from this cohort.

Defaults: `zai-org/GLM-5.3` on SDU's OpenAI-compatible Chat Completions endpoint,
one completion per problem, temperature 1.0, reasoning effort `max` (SDU's default),
no output token cap and four concurrent requests. Inspect's generic
`openai-api` provider preserves the temperature setting for this non-OpenAI model.

From the repository root in PowerShell:

```powershell
# Download and validate the published splits; no inference
./src/scripts/run_glm53_saturation.ps1 --dry-run
# One published sample per task; separate smoke output
./src/scripts/run_glm53_saturation.ps1 --smoke
# Full evaluation; repeat the same command to resume
./src/scripts/run_glm53_saturation.ps1
```

The launcher uses `SDU_API_KEY` if set, otherwise the Windows-user-encrypted key
at `~/.codex/secrets/sdu-glm53.dpapi`. No credentials are stored in the repository.
The replacement key passed model discovery and real Inspect inference on
2026-09-16. The existing Python environment has the required Inspect, Hugging
Face and math-scoring dependencies. Use Python UTF-8 mode on Windows (the
launcher already does) so Inspect reads the multilingual YAML correctly.

Results are saved under ignored `logs/glm53-hf-pr16/`: standard Inspect `.eval`
files, pinned settings in `manifest.json`, `summary.csv`, and `errors.json`.
Accuracy is reported only for complete splits. Original intervals are Wilson
95%; synthetic intervals use the 100 per-template accuracies with a t interval,
so the 20 related variants are not treated as independent templates. Intervals
are conditional on this fixed benchmark and decoding setup. Review parsing
failures and truncation before drawing ceiling conclusions.

Use a fresh `--output` directory when changing settings. `--revision` accepts
an immutable 40-character HF commit, so future branch changes do not silently
alter a resumed run. No paper text, PR messages, or uploads are automated.

## Credit estimate

User-supplied UCloud GLM-5.3 rates: 5.815 credits per million input tokens,
17.362 per million output tokens, 1.163 per million cached input tokens.
Ignoring caching and retries, 31,500 requests averaging 250 input and 500
output tokens cost about **319 credits**; at 1,000 output tokens, **593 credits**.
These are planning scenarios, not a guarantee. The user accepted this estimate;
no further cost-sampling run is needed.

```powershell
.venv/Scripts/python.exe -X utf8 -m pytest tests/test_glm53_saturation.py -n 0 -q
```
