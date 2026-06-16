# /// script
# dependencies = ["inspect-ai", "pandas"]
# ///
"""Build transfer-analysis tables from Inspect eval logs.

Example:
    uv run paper/scripts/collect_transfer_tables.py --workers 8
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from inspect_ai.log import read_eval_log

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from visualizegrid import (  # noqa: E402
    discover_logs,
    infer_model_info,
    model_name,
    parse_task,
    sample_score,
    select_logs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO_ROOT / "hf_dataset" / "logs"
DEFAULT_OUT_DIR = REPO_ROOT / "paper" / "artifacts" / "transfer_tables"
DEFAULT_LANGUAGE_FEATURES = REPO_ROOT / "paper" / "artifacts" / "figures" / "transfer_features" / "language_features.csv"
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
    model = model_name(log.eval.model)
    rows: list[dict[str, Any]] = []

    for sample in log.samples or []:
        score = sample_score(sample, scorer)
        if score is None:
            continue
        metadata = sample.metadata or {}
        correct = score >= 0.5
        language = metadata.get("language", task_language)
        sample_id = scalar(sample.id)
        rows.append(
            {
                "id": sample_id,
                "sample_id": sample_id,
                "source_id": scalar(metadata.get("source_id")),
                "language": language,
                "correct": bool(correct),
                "correct_label": "correct" if correct else "incorrect",
                "score": float(score),
                "model": model,
                "model_raw": log.eval.model,
                "family": info.family,
                "params_b": info.params_b,
                "vocab_size": info.vocab_size,
                "training_language": info.training_language,
                "pretrain_tokens_t": info.pretrain_tokens_t,
                "split": split,
                "question_type": split,
                "target": scalar(getattr(sample, "target", None)),
                "prompt_chars": len(str(getattr(sample, "input", "") or "")),
                "completion_chars": len(str(getattr(sample, "output", "") or "")),
                "started_at": scalar(getattr(sample, "started_at", None)),
                "completed_at": scalar(getattr(sample, "completed_at", None)),
                "total_time": scalar(getattr(sample, "total_time", None)),
                "epoch": scalar(getattr(sample, "epoch", None)),
                "scorer": scorer or "auto",
                "eval_id": log.eval.eval_id,
                "task": log.eval.task,
                "log_file": path.name,
            }
        )

    return f"{model} / {task_language} / {split}", pd.DataFrame(rows), None


def load_observations(selected: list[tuple[Path, Any]], scorer: str | None, workers: int) -> pd.DataFrame:
    paths = [path for path, _header in selected]
    if not paths:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    if workers <= 1:
        results = [load_log_rows(path, scorer) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as pool:
            futures = [pool.submit(load_log_rows, path, scorer) for path in paths]
            results = [future.result() for future in as_completed(futures)]

    for label, frame, warning in results:
        if warning:
            print(warning)
        else:
            print(f"Loaded {label}: {len(frame)} scored samples")
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    observations = pd.concat(frames, ignore_index=True)
    observations["observation_id"] = (
        observations["model"].astype(str)
        + "|"
        + observations["split"].astype(str)
        + "|"
        + observations["language"].astype(str)
        + "|"
        + observations["id"].astype(str)
    )
    return observations.sort_values(["model", "language", "split", "id"]).reset_index(drop=True)


def build_analysis_tables(
    observations: pd.DataFrame,
    language_features: pd.DataFrame,
    fertility: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observations = observations.copy()
    if "question_type" not in observations.columns and "split" in observations.columns:
        observations["question_type"] = observations["split"]

    core = ["id", "language", "correct", "model"]
    rest = [column for column in observations.columns if column not in core]
    main = observations[core + rest].copy()
    main["correct"] = main["correct"].map(lambda value: bool(value)).astype(object)

    model_columns = [
        "model",
        "model_raw",
        "family",
        "params_b",
        "vocab_size",
        "training_language",
        "pretrain_tokens_t",
    ]
    models = main[[column for column in model_columns if column in main.columns]].drop_duplicates().reset_index(drop=True)

    model_languages = fertility.copy()
    if not fertility.empty:
        if "normalized_fertility" in model_languages.columns:
            model_languages["language_specific_normalized_fertility"] = model_languages["normalized_fertility"]
        if "fertility_tokens_per_character" in model_languages.columns:
            model_languages["language_specific_fertility_tokens_per_character"] = model_languages[
                "fertility_tokens_per_character"
            ]

    languages = language_features.copy()
    if not languages.empty:
        language_counts = (
            main.groupby("language", as_index=False)
            .agg(n_observations=("correct", "size"), mean_accuracy=("score", "mean"))
        )
        languages = languages.merge(language_counts, on="language", how="outer")

    analysis = main.copy()
    if not languages.empty:
        analysis = analysis.merge(languages, on="language", how="left", suffixes=("", "_language"))
    if not fertility.empty:
        analysis = analysis.merge(
            model_languages,
            on=["model", "model_raw", "language"],
            how="left",
            suffixes=("", "_model_language"),
        )

    return main, models, languages, model_languages, analysis


def write_tables(
    observations: pd.DataFrame,
    language_features: pd.DataFrame,
    fertility: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    main, models, languages, model_languages, analysis = build_analysis_tables(
        observations,
        language_features,
        fertility,
    )
    outputs = {
        "main": out_dir / "main.csv",
        "models": out_dir / "models.csv",
        "languages": out_dir / "languages.csv",
        "model_languages": out_dir / "model_language_features.csv",
        "analysis": out_dir / "analysis.csv",
    }
    main.to_csv(outputs["main"], index=False)
    models.to_csv(outputs["models"], index=False)
    languages.to_csv(outputs["languages"], index=False)
    model_languages.to_csv(outputs["model_languages"], index=False)
    analysis.to_csv(outputs["analysis"], index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", nargs="+", type=Path, default=[DEFAULT_LOG_DIR])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--language-features", type=Path, default=DEFAULT_LANGUAGE_FEATURES)
    parser.add_argument("--fertility", type=Path, default=DEFAULT_FERTILITY)
    parser.add_argument("--scorer", default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--include-incomplete", action="store_true")
    args = parser.parse_args()

    paths = discover_logs(args.log_dir)
    selected = select_logs(paths, args.include_incomplete, workers=args.workers)
    print(f"Discovered {len(paths)} logs; selected {len(selected)}.")

    observations = load_observations(selected, args.scorer, args.workers)
    if observations.empty:
        raise SystemExit("No scored samples found in selected logs.")

    outputs = write_tables(
        observations,
        load_csv(args.language_features),
        load_csv(args.fertility),
        args.out_dir,
    )
    for name, path in outputs.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
