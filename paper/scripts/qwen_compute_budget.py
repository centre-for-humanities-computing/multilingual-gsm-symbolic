# /// script
# dependencies = ["inspect-ai", "matplotlib", "numpy", "pandas"]
# ///
"""Plot Qwen reasoning trade-offs under a per-sample generation budget.

The script reads Inspect ``.eval`` logs directly and writes:

* ``qwen_compute_budget_transfer.csv``: paired Qwen reasoning variants with
  transfer gaps and mean generation seconds per sample.
* ``qwen_compute_budget_transfer.png``: relative transfer gap vs generation
  seconds per sample.

Only Qwen-family models with both reasoning-on and reasoning-off variants for
the same raw model are included.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eval_log_utils import discover_logs, infer_model_info, model_name, parse_task, sample_score, select_logs
from inspect_ai.log import read_eval_log
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO_ROOT / "hf_dataset" / "logs"
DEFAULT_OUT_DIR = REPO_ROOT / "paper" / "artifacts" / "figures" / "model_grid"
QWEN_FAMILIES = {"Qwen2.5", "Qwen3", "Qwen3.5", "Qwen"}
REASONING_COLORS = {"off": "#2563EB", "on": "#DC2626"}
REASONING_LABELS = {"off": "reasoning off", "on": "reasoning on"}

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.dpi": 160,
        "font.family": "sans-serif",
    }
)


def reasoning_variant_name(model: str) -> tuple[str, str | None]:
    suffixes = {
        " (reasoning on)": "on",
        " (reasoning off)": "off",
    }
    for suffix, mode in suffixes.items():
        if model.endswith(suffix):
            return model[: -len(suffix)], mode
    return model, None


def _load_one_log(path: Path, scorer: str | None) -> tuple[str, pd.DataFrame | None, str | None]:
    try:
        log = read_eval_log(str(path))
    except Exception as exc:
        return path.name, None, f"Skipping unreadable log {path.name}: {exc}"

    parsed_task = parse_task(log.eval.task)
    if parsed_task is None:
        return path.name, None, f"Skipping unrecognized task {log.eval.task!r}"

    split, task_language = parsed_task
    label = f"{model_name(log.eval.model, log.eval.model_args)} / {task_language} / {split}"
    if split != "synthetic" or not log.samples:
        return label, pd.DataFrame(), None

    info = infer_model_info(log.eval.model)
    if info.family not in QWEN_FAMILIES or info.params_b is None:
        return label, pd.DataFrame(), None

    rows: list[dict[str, Any]] = []
    for sample in log.samples:
        correct = sample_score(sample, scorer)
        generation_seconds = getattr(sample.output, "time", None)
        if correct is None or generation_seconds is None:
            continue

        metadata = sample.metadata or {}
        rows.append(
            {
                "model_raw": log.eval.model,
                "model": model_name(log.eval.model, log.eval.model_args),
                "family": info.family,
                "params_b": info.params_b,
                "language": metadata.get("language", task_language),
                "correct": correct,
                "generation_seconds": generation_seconds,
            }
        )

    if not rows:
        return label, pd.DataFrame(), None

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["model_raw", "model", "family", "params_b", "language"], dropna=False)
        .agg(
            accuracy=("correct", "mean"),
            n_problems=("correct", "size"),
            avg_generation_seconds=("generation_seconds", "mean"),
        )
        .reset_index()
    )
    summary["split"] = "synthetic"
    return label, summary, None


def load_qwen_summary(selected: list[tuple[Path, Any]], scorer: str | None, workers: int) -> pd.DataFrame:
    paths = [path for path, _header in selected]
    if not paths:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    if workers <= 1:
        for index, path in enumerate(paths, start=1):
            label, frame, warning = _load_one_log(path, scorer)
            if warning:
                print(warning)
            elif frame is not None and not frame.empty:
                print(f"[{index}/{len(paths)}] {label}")
            if frame is not None and not frame.empty:
                frames.append(frame)
    else:
        max_workers = min(workers, len(paths))
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_load_one_log, path, scorer) for path in paths]
            for index, future in enumerate(as_completed(futures), start=1):
                label, frame, warning = future.result()
                if warning:
                    print(warning)
                elif frame is not None and not frame.empty:
                    print(f"[{index}/{len(paths)}] {label}")
                if frame is not None and not frame.empty:
                    frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby(["model_raw", "model", "family", "params_b", "language", "split"], dropna=False)
        .agg(
            accuracy=("accuracy", "mean"),
            n_problems=("n_problems", "sum"),
            avg_generation_seconds=("avg_generation_seconds", "mean"),
        )
        .reset_index()
    )


def qwen_compute_budget_table(summary: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model_raw",
        "model",
        "family",
        "params_b",
        "language",
        "split",
        "accuracy",
        "avg_generation_seconds",
    }
    if summary.empty or not required.issubset(summary.columns):
        return pd.DataFrame()

    rows = summary[
        (summary["split"] == "synthetic")
        & (summary["family"].isin(QWEN_FAMILIES))
        & summary["params_b"].notna()
        & summary["avg_generation_seconds"].notna()
    ].copy()
    if rows.empty:
        return pd.DataFrame()

    parsed = rows["model"].map(reasoning_variant_name)
    rows["base_model_candidate"] = [item[0] for item in parsed]
    rows["reasoning"] = [item[1] for item in parsed]

    off_name_by_raw = rows[rows["reasoning"] == "off"].groupby("model_raw")["base_model_candidate"].first()
    has_off_variant = rows["model_raw"].isin(set(off_name_by_raw.index))
    rows["canonical_base_model"] = rows["model_raw"].map(off_name_by_raw).fillna(rows["base_model_candidate"])
    rows.loc[
        rows["reasoning"].isna() & has_off_variant & (rows["model"] == rows["canonical_base_model"]),
        "reasoning",
    ] = "on"

    paired_raws = rows.groupby("model_raw")["reasoning"].agg(lambda values: {"on", "off"}.issubset(set(values)))
    rows = rows[rows["model_raw"].isin(set(paired_raws[paired_raws].index))]
    rows = rows[rows["reasoning"].isin(["on", "off"])]
    if rows.empty:
        return pd.DataFrame()

    accuracy = rows.pivot_table(
        index=["model_raw", "model", "family", "params_b", "reasoning"],
        columns="language",
        values="accuracy",
    )
    if "eng" not in accuracy.columns:
        return pd.DataFrame()

    non_english = [language for language in accuracy.columns if language not in {"eng", "eng_metric"}]
    if not non_english:
        return pd.DataFrame()

    timing = (
        rows.groupby(["model_raw", "model", "family", "params_b", "reasoning"], dropna=False)[
            "avg_generation_seconds"
        ]
        .mean()
        .rename("avg_generation_seconds")
    )
    table = accuracy.join(timing).reset_index()
    table["english_accuracy"] = table["eng"]
    table["non_english_accuracy"] = table[non_english].mean(axis=1, skipna=True)
    table["absolute_transfer_gap"] = table["english_accuracy"] - table["non_english_accuracy"]
    table["relative_transfer_gap"] = table["absolute_transfer_gap"] / table["english_accuracy"].replace(0, np.nan)
    table["n_transfer_languages"] = table[non_english].notna().sum(axis=1)
    table = table[
        [
            "model_raw",
            "model",
            "family",
            "params_b",
            "reasoning",
            "avg_generation_seconds",
            "english_accuracy",
            "non_english_accuracy",
            "absolute_transfer_gap",
            "relative_transfer_gap",
            "n_transfer_languages",
        ]
    ]
    table = table.dropna(subset=["relative_transfer_gap", "avg_generation_seconds"])
    return table[table["n_transfer_languages"] > 0].sort_values(
        ["avg_generation_seconds", "relative_transfer_gap", "params_b", "model"]
    )


def plot_qwen_compute_budget_transfer(summary: pd.DataFrame, out: Path, csv_out: Path) -> bool:
    table = qwen_compute_budget_table(summary)
    if table.empty:
        return False

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_out, index=False)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    marker_by_family = {"Qwen3": "o", "Qwen3.5": "s", "Qwen": "^", "Qwen2.5": "D"}

    for (reasoning, family), group in table.groupby(["reasoning", "family"], sort=False):
        group = group.sort_values("avg_generation_seconds")
        ax.scatter(
            group["avg_generation_seconds"],
            group["relative_transfer_gap"],
            s=38 + group["params_b"].clip(upper=72) * 1.6,
            marker=marker_by_family.get(family, "o"),
            color=REASONING_COLORS[reasoning],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9,
            label=REASONING_LABELS[reasoning],
            zorder=3,
        )

    for row in table.itertuples(index=False):
        label = row.model.replace("Qwen", "Q")
        ax.annotate(
            label,
            (row.avg_generation_seconds, row.relative_transfer_gap),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="#374151",
        )

    for (reasoning, family), group in table.groupby(["reasoning", "family"], sort=False):
        frontier = group.sort_values("avg_generation_seconds").copy()
        frontier["best_gap_so_far"] = frontier["relative_transfer_gap"].cummin()
        frontier = frontier[frontier["best_gap_so_far"].diff().fillna(-1) < 0]
        if len(frontier) > 1:
            ax.step(
                frontier["avg_generation_seconds"],
                frontier["best_gap_so_far"],
                where="post",
                color=REASONING_COLORS[reasoning],
                linewidth=1.5,
                linestyle="--",
                alpha=0.75,
                label=f"best {REASONING_LABELS[reasoning]} gap within seconds/sample budget",
                zorder=2,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Mean generation seconds per sample (log scale)")
    ax.set_ylabel("Relative transfer gap: (English - non-English mean) / English")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="both", color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title("Qwen reasoning trade-off under a per-sample generation budget")
    ax.text(
        0.01,
        0.02,
        "Lower-left is better: smaller transfer gap at lower generation time.",
        transform=ax.transAxes,
        fontsize=8,
        color="#4B5563",
    )
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=color, label=REASONING_LABELS[reasoning])
        for reasoning, color in REASONING_COLORS.items()
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            marker=marker_by_family.get(family, "o"),
            linestyle="none",
            markerfacecolor="#6B7280",
            markeredgecolor="white",
            color="none",
            label=family,
        )
        for family in sorted(table["family"].unique())
    )
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        type=Path,
        nargs="+",
        default=[DEFAULT_LOG_DIR],
        help="One or more directories searched recursively for .eval logs.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--scorer", help="Inspect scorer name to use; defaults to math, pattern, then first score.")
    parser.add_argument("--include-incomplete", action="store_true", help="Include readable samples from failed logs.")
    parser.add_argument("--workers", type=int, default=32, help="Workers used for log selection and full log loading.")
    args = parser.parse_args()

    paths = discover_logs(args.log_dir)
    selected = select_logs(paths, args.include_incomplete, workers=args.workers)
    print(f"Discovered {len(paths)} logs; selected {len(selected)} after status filtering and deduplication.")

    summary = load_qwen_summary(selected, args.scorer, args.workers)
    if summary.empty:
        raise SystemExit("No scored Qwen synthetic samples with generation timings found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png_out = args.out_dir / "qwen_compute_budget_transfer.png"
    csv_out = args.out_dir / "qwen_compute_budget_transfer.csv"
    if not plot_qwen_compute_budget_transfer(summary, png_out, csv_out):
        raise SystemExit("No paired Qwen reasoning-on/off models found.")

    print(f"Saved {png_out}")
    print(f"Saved {csv_out}")


if __name__ == "__main__":
    main()
