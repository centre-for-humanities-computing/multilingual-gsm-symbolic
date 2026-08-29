#!/usr/bin/env bash

set -euo pipefail

# Full Inspect logs are multi-gigabyte archives. Loading dozens concurrently can
# fill the job cgroup with file-backed page cache even when process RSS is low.
# Use one third of this 48-core host by default, retaining RAM headroom for the
# parent aggregation process and the kernel page cache.
artifact_workers="${ARTIFACT_WORKERS:-16}"

log_dir="hf_dataset/logs_pr16_merged"
historical_log_dir="hf_dataset/logs_unvalidated_revisions"
revision="d34a0ffcb2851179ccac891807fa3a29ccd896f6"
languages=(dan deu est jpn urd)

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN must be set in the process environment." >&2
    exit 2
fi

models=(
    Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct
    Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct
    Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct
    Qwen/Qwen2.5-72B-Instruct Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B
    Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-32B
    Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B
    Qwen/Qwen3.5-27B allenai/OLMo-2-0425-1B-Instruct
    allenai/OLMo-2-1124-7B-Instruct allenai/OLMo-2-1124-13B-Instruct
    allenai/OLMo-2-0325-32B-Instruct allenai/Olmo-3-7B-Think
    allenai/Olmo-3-7B-Instruct allenai/Olmo-3-32B-Think
    allenai/Olmo-3.1-32B-Instruct ibm-granite/granite-3.2-2b-instruct
    ibm-granite/granite-3.2-8b-instruct google/gemma-3-1b-it
    google/gemma-3-4b-it google/gemma-3-12b-it google/gemma-3-27b-it
    swiss-ai/Apertus-8B-Instruct-2509 swiss-ai/Apertus-70B-Instruct-2509
    utter-project/EuroLLM-1.7B-Instruct utter-project/EuroLLM-9B-Instruct-2512
    utter-project/EuroLLM-22B-Instruct-2512
)
gemma4_models=(google/gemma-4-12B-it google/gemma-4-31B-it)

eval_status=0
uv run paper/scripts/ucloudeval \
    --model "${models[@]}" \
    --language "${languages[@]}" \
    --revision "$revision" \
    --log-dir "$log_dir" || eval_status=$?

gemma4_status=0
uv run paper/scripts/ucloudeval \
    --model "${gemma4_models[@]}" \
    --language "${languages[@]}" \
    --revision "$revision" \
    --log-dir "$log_dir" || gemma4_status=$?

historical_status=0
uv run paper/scripts/ucloudeval \
    --model "${models[@]}" \
    --language isl \
    --split all \
    --historical-revisions \
    --log-dir "$historical_log_dir" || historical_status=$?

historical_gemma4_status=0
uv run paper/scripts/ucloudeval \
    --model "${gemma4_models[@]}" \
    --language isl \
    --split all \
    --historical-revisions \
    --log-dir "$historical_log_dir" || historical_gemma4_status=$?

if (( eval_status != 0 || gemma4_status != 0 || historical_status != 0 || historical_gemma4_status != 0 )); then
    echo "Evaluation failed; artifact regeneration not started: latest=$eval_status latest_gemma4=$gemma4_status historical_isl=$historical_status historical_isl_gemma4=$historical_gemma4_status" >&2
    exit 1
fi

uv run --with scipy paper/scripts/collect_transfer_tables.py \
    --log-dir "$log_dir" hf_dataset/logs_unvalidated_revisions \
    --workers "$artifact_workers"
uv run paper/scripts/visualizegrid.py --workers "$artifact_workers"
uv run paper/scripts/visualize_results.py
uv run paper/scripts/numbercoverage.py --workers "$artifact_workers"
uv run paper/scripts/language_accuracy_table.py
uv run paper/scripts/qwen_compute_budget.py --workers "$artifact_workers"
uv run paper/scripts/transferfeatures.py
uv run paper/scripts/isl_translation_tost.py
uv run paper/scripts/migrate_cache_unlimited.py

echo "Latest-language evaluation and artifact regeneration finished successfully."
