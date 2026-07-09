# /// script
# dependencies = ["inspect-ai", "pandas"]
# ///
"""Build transfer-analysis tables from Inspect eval logs.

Example:
    uv run paper/scripts/collect_transfer_tables.py --workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from eval_log_utils import (
    discover_logs,
    infer_model_info,
    model_name,
    parse_task,
    sample_score,
    select_logs,
)
from inspect_ai.log import read_eval_log
from plot_config import language_order, ordered_models

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO_ROOT / "hf_dataset" / "logs"
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

    return f"{model} / {task_language} / {split}", pd.DataFrame(rows), None


def load_observations(selected: list[tuple[Path, Any]], scorer: str | None, workers: int) -> pd.DataFrame:
    paths = [path for path, _header in selected]
    if not paths:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    if workers <= 1:
        results = [load_log_rows(path, scorer) for path in paths]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(paths))) as pool:
            results = list(pool.map(load_log_rows, paths, [scorer] * len(paths)))

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
    analysis = observations.copy()
    if not language_features.empty:
        analysis = analysis.merge(language_features, on="language", how="left", suffixes=("", "_language"))
    if not fertility.empty:
        analysis = analysis.merge(
            fertility,
            on=["model", "model_raw", "language"],
            how="left",
            suffixes=("", "_model_language"),
        )
    analysis = analysis.drop(columns=["model_raw"], errors="ignore")
    return analysis


def write_tables(
    observations: pd.DataFrame,
    language_features: pd.DataFrame,
    fertility: pd.DataFrame,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis = build_analysis_tables(observations, language_features, fertility)
    out_path = out_dir / "analysis.csv"
    analysis.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", nargs="+", type=Path, default=[DEFAULT_LOG_DIR])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--language-features", type=Path, default=DEFAULT_LANGUAGE_FEATURES)
    parser.add_argument("--fertility", type=Path, default=DEFAULT_FERTILITY)
    parser.add_argument("--scorer", default=None)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--include-incomplete", action="store_true")
    args = parser.parse_args()

    paths = discover_logs(args.log_dir)
    selected = select_logs(paths, args.include_incomplete, workers=args.workers)
    print(f"Discovered {len(paths)} logs; selected {len(selected)}.")

    observations = load_observations(selected, args.scorer, args.workers)
    if observations.empty:
        raise SystemExit("No scored samples found in selected logs.")

    out_path = write_tables(
        observations,
        load_csv(args.language_features),
        load_csv(args.fertility),
        args.out_dir,
    )
    print(f"Saved analysis: {out_path}")


if __name__ == "__main__":
    main()
