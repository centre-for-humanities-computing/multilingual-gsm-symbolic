# /// script
# dependencies = ["inspect-ai", "pandas", "pyarrow"]
# ///
"""Build transfer-analysis tables from Inspect eval logs.

Example:
    uv run paper/scripts/collect_transfer_tables.py --workers 8
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from eval_log_utils import (
    discover_logs,
    infer_model_info,
    map_log_loader,
    model_name,
    parse_task,
    sample_score,
    select_logs,
)
from inspect_ai.log import read_eval_log
from plot_config import language_order, ordered_models

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO_ROOT / "hf_dataset" / "logs"
DEFAULT_UNVALIDATED_LOG_DIR = REPO_ROOT / "hf_dataset" / "logs_unvalidated_revisions"
DEFAULT_LOG_DIRS = [DEFAULT_LOG_DIR, DEFAULT_UNVALIDATED_LOG_DIR]
DEFAULT_OUT_DIR = REPO_ROOT / "paper" / "artifacts" / "transfer_tables"
DEFAULT_LANGUAGE_FEATURES = (
    REPO_ROOT / "paper" / "artifacts" / "figures" / "transfer_features" / "language_features.csv"
)
DEFAULT_FERTILITY = REPO_ROOT / "paper" / "artifacts" / "figures" / "transfer_features" / "tokenizer_fertility.csv"


def scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_log_rows(path: Path, scorer: str | None) -> tuple[str, pd.DataFrame, str | None]:
    try:
        log = read_eval_log(str(path))
    except Exception as exc:
        return path.name, pd.DataFrame(), f"Skipping unreadable log {path.name}: {exc}"

    parsed_task = parse_task(log.eval.task)
    if parsed_task is None:
        return path.name, pd.DataFrame(), f"Skipping unrecognized task {log.eval.task!r}"

    split, task_language = parsed_task
    info = infer_model_info(log.eval.model)
    model = model_name(log.eval.model, log.eval.model_args)
    rows: list[dict[str, Any]] = []

    is_uncorrected_isl = (
        ("logs_unvalidated_revisions" in path.parts or "0850f21ea319" in str(path) or "0850f21ea319" in log.eval.task)
        and task_language == "isl"
    )

    for sample in log.samples or []:
        score = sample_score(sample, scorer)
        if score is None:
            continue
        metadata = sample.metadata or {}
        correct = score >= 0.5
        language = "uncorrected_isl" if is_uncorrected_isl else metadata.get("language", task_language)
        sample_id = scalar(sample.id)
        rows.append(
            {
                "id": sample_id,
                "source_id": scalar(metadata.get("source_id")),
                "language": language,
                "correct": bool(correct),
                "model": model,
                "model_raw": log.eval.model,
                "family": info.family,
                "params_b": info.params_b,
                "vocab_size": info.vocab_size,
                "split": split,
                "target": scalar(getattr(sample, "target", None)),
                "prompt_chars": len(str(getattr(sample, "input", "") or "")),
                "completion_chars": len(str(getattr(sample, "output", "") or "")),
                "total_time": scalar(getattr(sample, "total_time", None)),
                "scorer": scorer or "auto",
                "eval_id": log.eval.eval_id,
                "task": log.eval.task,
            }
        )

    del log  # free the large decompressed log before returning
    log_label_lang = "uncorrected_isl" if is_uncorrected_isl else task_language
    return f"{model} / {log_label_lang} / {split}", pd.DataFrame(rows), None


def load_observations(selected: list[tuple[Path, Any]], scorer: str | None, workers: int) -> pd.DataFrame:
    paths = [path for path, _header in selected]
    if not paths:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    _CONCAT_CHUNK = 8  # flush accumulated frames every N logs to cap memory
    for label, frame, warning in map_log_loader(load_log_rows, paths, scorer, workers):
        if warning:
            print(warning)
        else:
            print(f"Loaded {label}: {len(frame)} scored samples")
        if not frame.empty:
            frames.append(frame)
        if len(frames) >= _CONCAT_CHUNK:
            frames = [pd.concat(frames, ignore_index=True)]

    if not frames:
        return pd.DataFrame()

    observations = pd.concat(frames, ignore_index=True)
    model_categories = ordered_models(observations["model"].dropna().unique())
    language_categories = language_order(observations["language"].dropna().unique())
    observations["model"] = pd.Categorical(observations["model"], categories=model_categories, ordered=True)
    observations["language"] = pd.Categorical(observations["language"], categories=language_categories, ordered=True)
    observations = observations.sort_values(["model", "language", "split", "id"]).reset_index(drop=True)
    observations["model"] = observations["model"].astype(str)
    observations["language"] = observations["language"].astype(str)
    return observations


def build_analysis_tables(
    observations: pd.DataFrame,
    language_features: pd.DataFrame,
    fertility: pd.DataFrame,
) -> pd.DataFrame:
    working = observations.drop(
        columns=[
            "observation_id",
            "started_at",
            "completed_at",
            "training_language",
            "pretrain_tokens_t",
            "question_type",
            "correct_label",
            "epoch",
            "log_file",
        ],
        errors="ignore",
    ).copy()
    main = working.drop(columns=["model_raw"], errors="ignore")

    languages = language_features.copy()
    if not languages.empty:
        if "isl" in languages["language"].values and "uncorrected_isl" not in languages["language"].values:
            isl_row = languages.loc[languages["language"] == "isl"].copy()
            isl_row["language"] = "uncorrected_isl"
            languages = pd.concat([languages, isl_row], ignore_index=True)
        language_counts = main.groupby("language", as_index=False).agg(
            n_observations=("correct", "size"), mean_accuracy=("correct", "mean")
        )
        languages = languages.merge(language_counts, on="language", how="outer")

    analysis_languages = languages.drop(
        columns=[
            "language_name",
            "typological_feature_set",
            "common_crawl_language",
            "common_crawl_crawl",
            "resource_source_path",
            "n_observations",
            "mean_accuracy",
        ],
        errors="ignore",
    ).rename(
        columns={
            "typological_distance_from_english": "typological_distance",
            "common_crawl_pages": "commoncrawl_page_count",
        }
    )
    analysis_fertility = fertility.drop(
        columns=[
            "family",
            "family_model_language",
            "n_matched_questions",
            "token_count",
            "non_whitespace_character_count",
            "english_token_count",
            "english_non_whitespace_character_count",
            "tokenizer_repo",
            "normalized_fertility",
        ],
        errors="ignore",
    ).rename(columns={"fertility_tokens_per_character": "tokenizer_fertility"})

    if not analysis_fertility.empty and "isl" in analysis_fertility["language"].values:
        if "uncorrected_isl" not in analysis_fertility["language"].values:
            isl_fert = analysis_fertility.loc[analysis_fertility["language"] == "isl"].copy()
            isl_fert["language"] = "uncorrected_isl"
            analysis_fertility = pd.concat([analysis_fertility, isl_fert], ignore_index=True)

    analysis = working.copy()
    if not analysis_languages.empty:
        analysis = analysis.merge(analysis_languages, on="language", how="left", suffixes=("", "_language"))
    if not fertility.empty:
        analysis = analysis.merge(
            analysis_fertility,
            on=["model", "model_raw", "language"],
            how="left",
            suffixes=("", "_model_language"),
        )
    analysis = analysis.drop(columns=["model_raw"], errors="ignore")

    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", nargs="+", type=Path, default=DEFAULT_LOG_DIRS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--language-features", type=Path, default=DEFAULT_LANGUAGE_FEATURES)
    parser.add_argument("--fertility", type=Path, default=DEFAULT_FERTILITY)
    parser.add_argument("--scorer", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--include-incomplete", action="store_true")
    args = parser.parse_args()

    paths = discover_logs(args.log_dir)
    selected = select_logs(paths, args.include_incomplete, workers=args.workers)
    print(f"Discovered {len(paths)} logs; selected {len(selected)}.")

    observations = load_observations(selected, args.scorer, args.workers)
    if observations.empty:
        raise SystemExit("No scored samples found in selected logs.")

    analysis = build_analysis_tables(
        observations,
        load_csv(args.language_features),
        load_csv(args.fertility),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "analysis.parquet"
    analysis.to_parquet(output, index=False)
    print(f"Saved analysis: {output}")


if __name__ == "__main__":
    main()
