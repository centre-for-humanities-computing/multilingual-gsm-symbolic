# /// script
# dependencies = ["inspect-ai", "matplotlib", "numpy", "pandas", "scipy"]
# ///
"""Plot language-transfer trade-offs under an approximate inference budget.

The script reads Inspect ``.eval`` logs directly and writes:

* one ``qwen_compute_budget_transfer.png`` per model family with paired
  reasoning-on/off variants.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eval_log_utils import (
    classify_reasoning_variants,
    discover_logs,
    infer_model_info,
    map_log_loader,
    model_name,
    parse_task,
    sample_score,
    select_logs,
)
from inspect_ai.log import read_eval_log
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from plot_config import PLOT_STYLE, path_slug
from scipy.stats import bootstrap

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO_ROOT / "hf_dataset" / "logs"
DEFAULT_OUT_DIR = REPO_ROOT / "paper" / "artifacts" / "figures" / "model_grid"
REASONING_LABELS = {"off": "reasoning off", "on": "reasoning on"}
FAMILY_COLORS = ["#2563EB", "#DC2626", "#059669", "#7C3AED", "#D97706", "#0891B2"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260722

plt.rcParams.update(PLOT_STYLE)


def _bootstrap_gap_cis(outcomes: list[dict[str, float]], rng: np.random.Generator) -> tuple[float, float]:
    shared_ids = sorted(set.intersection(*(set(language) for language in outcomes)))
    if not shared_ids:
        return 0.0, 0.0
    samples = tuple(np.array([language[sample_id] for sample_id in shared_ids]) for language in outcomes)

    def gap_statistics(english: np.ndarray, *others: np.ndarray, axis: int = -1) -> np.ndarray:
        english_accuracy = english.mean(axis=axis)
        gap = english_accuracy - np.mean([sample.mean(axis=axis) for sample in others], axis=0)
        relative_gap = np.divide(
            gap,
            english_accuracy,
            out=np.full_like(gap, np.nan),
            where=english_accuracy > 0,
        )
        return np.stack((gap, relative_gap))

    result = bootstrap(
        samples,
        gap_statistics,
        paired=True,
        vectorized=True,
        n_resamples=BOOTSTRAP_REPLICATES,
        batch=1_000,
        method="percentile",
        rng=rng,
    )
    estimates = gap_statistics(*samples)
    half_widths = np.maximum(estimates - result.confidence_interval.low, result.confidence_interval.high - estimates)
    return float(half_widths[0]), float(half_widths[1])






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
    if info.params_b is None:
        return label, pd.DataFrame(), None

    rows: list[dict[str, Any]] = []
    for sample in log.samples:
        correct = sample_score(sample, scorer)
        usage = getattr(sample.output, "usage", None)
        total_tokens = getattr(usage, "total_tokens", None)
        if correct is None or total_tokens is None:
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
                "sample_id": str(sample.id),
                "total_tokens": total_tokens,
            }
        )

    if not rows:
        return label, pd.DataFrame(), None

    frame = pd.DataFrame(rows)
    keys = ["model_raw", "model", "family", "params_b", "language"]
    grouped = frame.groupby(keys, dropna=False)
    summary = (
        grouped
        .agg(
            accuracy=("correct", "mean"),
            n_problems=("correct", "size"),
            avg_total_tokens=("total_tokens", "mean"),
        )
        .reset_index()
    )
    sample_correct = grouped[["sample_id", "correct"]].apply(
        lambda group: dict(zip(group["sample_id"], group["correct"], strict=True))
    )
    summary = summary.merge(sample_correct.rename("sample_correct").reset_index(), on=keys)
    summary["split"] = "synthetic"
    return label, summary, None


def load_qwen_summary(selected: list[tuple[Path, Any]], scorer: str | None, workers: int) -> pd.DataFrame:
    paths = [path for path, _header in selected]
    if not paths:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for index, (label, frame, warning) in enumerate(map_log_loader(_load_one_log, paths, scorer, workers), start=1):
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
            avg_total_tokens=("avg_total_tokens", "mean"),
            sample_correct=(
                "sample_correct",
                lambda mappings: {sample_id: value for mapping in mappings for sample_id, value in mapping.items()},
            ),
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
        "avg_total_tokens",
    }
    if summary.empty or not required.issubset(summary.columns):
        return pd.DataFrame()

    rows = summary[
        (summary["split"] == "synthetic") & summary["params_b"].notna() & summary["avg_total_tokens"].notna()
    ].copy()
    if rows.empty:
        return pd.DataFrame()

    rows = classify_reasoning_variants(rows)
    paired_raws = rows.groupby("model_raw")["reasoning"].agg(lambda values: {"on", "off"}.issubset(set(values)))
    rows = rows[rows["model_raw"].isin(set(paired_raws[paired_raws].index))]
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

    if "sample_correct" in rows.columns:
        paired_outcomes = rows.pivot_table(
            index=["model_raw", "model", "family", "params_b", "reasoning"],
            columns="language",
            values="sample_correct",
            aggfunc="first",
        )
    else:
        paired_outcomes = pd.DataFrame(index=accuracy.index, columns=accuracy.columns, dtype=object)

    token_usage = (
        rows.groupby(["model_raw", "model", "family", "params_b", "reasoning"], dropna=False)[
            "avg_total_tokens"
        ]
        .mean()
        .rename("avg_total_tokens")
    )
    table = accuracy.join(token_usage).reset_index()
    table["english_accuracy"] = table["eng"]
    table["non_english_accuracy"] = table[non_english].mean(axis=1, skipna=True)
    table["absolute_transfer_gap"] = table["english_accuracy"] - table["non_english_accuracy"]
    table["relative_transfer_gap"] = table["absolute_transfer_gap"] / table["english_accuracy"].replace(0, np.nan)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    absolute_ci: list[float] = []
    relative_ci: list[float] = []
    for index, row in table.set_index(["model_raw", "model", "family", "params_b", "reasoning"]).iterrows():
        outcomes = paired_outcomes.loc[index]
        languages = [language for language in non_english if isinstance(outcomes.get(language), dict)]
        if not languages or not isinstance(outcomes.get("eng"), dict):
            absolute_ci.append(0.0)
            relative_ci.append(0.0)
            continue

        absolute_half_width, relative_half_width = _bootstrap_gap_cis(
            [outcomes["eng"], *(outcomes[language] for language in languages)],
            rng,
        )
        absolute_ci.append(absolute_half_width)
        relative_ci.append(relative_half_width)
    table["transfer_gap_ci95"] = absolute_ci
    table["relative_transfer_gap_ci95"] = relative_ci
    # Dense-transformer inference is approximately two floating-point operations
    # per parameter per processed token. This intentionally reports an estimate.
    table["inference_flops"] = 2 * table["params_b"] * 1e9 * table["avg_total_tokens"]
    table["n_transfer_languages"] = table[non_english].notna().sum(axis=1)
    table = table[
        [
            "model_raw",
            "model",
            "family",
            "params_b",
            "reasoning",
            "avg_total_tokens",
            "inference_flops",
            "english_accuracy",
            "non_english_accuracy",
            "absolute_transfer_gap",
            "transfer_gap_ci95",
            "relative_transfer_gap",
            "relative_transfer_gap_ci95",
            "n_transfer_languages",
        ]
    ]
    table = table.dropna(subset=["absolute_transfer_gap", "inference_flops"])
    return table[table["n_transfer_languages"] > 0].sort_values(
        ["inference_flops", "absolute_transfer_gap", "params_b", "model"]
    )


def pareto_frontier(table: pd.DataFrame, gap_column: str = "absolute_transfer_gap") -> pd.DataFrame:
    frontier = table.sort_values(["inference_flops", gap_column]).copy()
    frontier["best_gap_so_far"] = frontier[gap_column].cummin()
    return frontier[frontier["best_gap_so_far"].diff().fillna(-1) < 0]


def reasoning_budget_summary_sentence(table: pd.DataFrame) -> str | None:
    if table.empty:
        return None

    paired = table.pivot_table(
        index="model_raw",
        columns="reasoning",
        values=["relative_transfer_gap", "inference_flops"],
        aggfunc="mean",
    )
    if not {"on", "off"}.issubset(paired["relative_transfer_gap"].columns) or not {
        "on",
        "off",
    }.issubset(paired["inference_flops"].columns):
        return None

    gap_off = paired["relative_transfer_gap"]["off"]
    flops_off = paired["inference_flops"]["off"]
    changes = pd.DataFrame(
        {
            "gap_reduction": (gap_off - paired["relative_transfer_gap"]["on"]) / gap_off,
            "compute_increase": (paired["inference_flops"]["on"] - flops_off) / flops_off,
        }
    ).replace([np.inf, -np.inf], np.nan)
    changes = changes.dropna()
    if changes.empty:
        return None

    return (
        "Enabling reasoning reduces transfer gaps by "
        f"{changes['gap_reduction'].mean() * 100:.1f}% on average while increasing estimated inference FLOPs per question by "
        f"{changes['compute_increase'].mean() * 100:.1f}%."
    )


def save_reasoning_budget_summary(table: pd.DataFrame, out_dir: Path) -> Path | None:
    sentence = reasoning_budget_summary_sentence(table)
    if not sentence:
        return None

    out = out_dir / "qwen_compute_budget_transfer" / "reasoning_budget_summary.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"{sentence}\n", encoding="utf-8")
    return out


def _plot_compute_budget_table(table: pd.DataFrame, out: Path, *, relative: bool = False) -> bool:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    gap_column = "relative_transfer_gap" if relative else "absolute_transfer_gap"
    ci_column = "relative_transfer_gap_ci95" if relative else "transfer_gap_ci95"
    families = sorted(table["family"].unique())
    marker_by_family = {family: MARKERS[index % len(MARKERS)] for index, family in enumerate(families)}
    color_by_family = {family: FAMILY_COLORS[index % len(FAMILY_COLORS)] for index, family in enumerate(families)}

    def mode_color(family: str, reasoning: str) -> str:
        color = color_by_family[family]
        if reasoning != "off":
            return color
        rgb = np.asarray(to_rgb(color))
        return to_hex(rgb + (1 - rgb) * 0.5)

    for (reasoning, family), group in table.groupby(["reasoning", "family"], sort=False):
        group = group.sort_values("inference_flops")
        color = mode_color(family, reasoning)
        linestyle = ":" if reasoning == "off" else "-"
        ax.plot(
            group["inference_flops"],
            group[gap_column],
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            zorder=1,
        )
        ax.errorbar(
            group["inference_flops"],
            group[gap_column],
            yerr=group[ci_column],
            fmt="none",
            ecolor=color,
            elinewidth=1,
            capsize=2,
            alpha=0.55,
            zorder=2,
        )
        ax.scatter(
            group["inference_flops"],
            group[gap_column],
            s=38 + group["params_b"].clip(upper=72) * 1.6,
            marker=marker_by_family.get(family, "o"),
            color=color,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9,
            label="_nolegend_",
            zorder=3,
        )

    for row in table.itertuples(index=False):
        label = f"{row.params_b:g}B"
        ax.annotate(
            label,
            (row.inference_flops, getattr(row, gap_column)),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="#374151",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Estimated inference FLOPs per sample (log scale)")
    if relative:
        ax.set_ylabel("Relative gap: (English − non-English mean) / English")
    else:
        ax.set_ylabel("English accuracy − non-English mean accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="both", color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title("Reasoning on vs. off under a fixed compute budget")
    ax.text(
        0.01,
        0.02,
        "Lower-left is better. Bars are 95% bootstrap CIs over questions.",
        transform=ax.transAxes,
        fontsize=8,
        color="#4B5563",
    )
    reasoning_order = [key for key in ("standard", "off", "on") if key in set(table["reasoning"])]
    handles = [
        Line2D([0], [0], color="#374151", linestyle=":" if reasoning == "off" else "-", linewidth=1.8,
               label=REASONING_LABELS[reasoning])
        for reasoning in reasoning_order
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            marker=marker_by_family.get(family, "o"),
            linestyle="-",
            markerfacecolor=color_by_family[family],
            markeredgecolor="white",
            color=color_by_family[family],
            label=family,
        )
        for family in families
    )
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_qwen_compute_budget_transfer(summary: pd.DataFrame, out: Path) -> bool:
    table = qwen_compute_budget_table(summary)
    if table.empty:
        return False

    return _plot_compute_budget_table(table, out)


def plot_qwen_compute_budget_relative_transfer(summary: pd.DataFrame, out: Path) -> bool:
    table = qwen_compute_budget_table(summary)
    if table.empty:
        return False

    return _plot_compute_budget_table(table, out, relative=True)


def plot_qwen_compute_budget_family_transfers(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    table = qwen_compute_budget_table(summary)
    if table.empty:
        return []

    outputs: list[Path] = []
    root = out_dir / "qwen_compute_budget_transfer"
    for family, family_table in table.groupby("family", sort=True):
        family_dir = root / path_slug(family)
        family_out = family_dir / "qwen_compute_budget_transfer.png"
        family_dir.mkdir(parents=True, exist_ok=True)
        _plot_compute_budget_table(family_table, family_out)
        outputs.append(family_out)

    return outputs


def plot_qwen_compute_budget_relative_family_transfers(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    table = qwen_compute_budget_table(summary)
    if table.empty:
        return []

    outputs: list[Path] = []
    root = out_dir / "qwen_compute_budget_transfer_relative"
    for family, family_table in table.groupby("family", sort=True):
        family_dir = root / path_slug(family)
        family_out = family_dir / "qwen_compute_budget_transfer_relative.png"
        family_dir.mkdir(parents=True, exist_ok=True)
        _plot_compute_budget_table(family_table, family_out, relative=True)
        outputs.append(family_out)

    return outputs


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
        raise SystemExit("No scored synthetic samples with generation timings found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined_out = args.out_dir / "qwen_compute_budget_transfer" / "qwen_compute_budget_transfer.png"
    combined_out.parent.mkdir(parents=True, exist_ok=True)
    if not plot_qwen_compute_budget_transfer(summary, combined_out):
        raise SystemExit("No combined model transfer rows found.")
    print(f"Saved {combined_out}")

    png_outputs = plot_qwen_compute_budget_family_transfers(summary, args.out_dir)
    if not png_outputs:
        raise SystemExit("No model transfer rows found.")

    for png_out in png_outputs:
        print(f"Saved {png_out}")

    relative_combined_out = (
        args.out_dir / "qwen_compute_budget_transfer_relative" / "qwen_compute_budget_transfer_relative.png"
    )
    relative_combined_out.parent.mkdir(parents=True, exist_ok=True)
    if not plot_qwen_compute_budget_relative_transfer(summary, relative_combined_out):
        raise SystemExit("No combined relative model transfer rows found.")
    print(f"Saved {relative_combined_out}")

    relative_outputs = plot_qwen_compute_budget_relative_family_transfers(summary, args.out_dir)
    for png_out in relative_outputs:
        print(f"Saved {png_out}")

    table = qwen_compute_budget_table(summary)
    summary_out = save_reasoning_budget_summary(table, args.out_dir)
    if summary_out:
        print(f"Saved {summary_out}")
        print(summary_out.read_text(encoding="utf-8").strip())



if __name__ == "__main__":
    main()
