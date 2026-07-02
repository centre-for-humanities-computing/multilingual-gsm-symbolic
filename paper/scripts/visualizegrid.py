# /// script
# dependencies = ["inspect-ai", "matplotlib", "numpy", "pandas", "scipy"]
# ///
"""Visualize multilingual GSM evaluation logs across models and splits.

The script reads Inspect ``.eval`` logs and writes:

* ``run_summary.csv``: tidy accuracy table for downstream analysis.
* ``accuracy_heatmaps.png``: accuracy on original and synthetic benchmark splits.
* ``original_vs_synthetic.png``: paired split performance for each model/language.
* ``family_scaling.png``: within-family accuracy as a function of parameter count.
* ``english_normalized_transfer.png``: language accuracy relative to English.
* ``eng_vs_eng_metric.png``: paired English and English-metric accuracy by model.
* ``transfer_robustness.png``: transfer penalty and cross-language dispersion by size.
* ``split_degradation_heatmaps.png``: absolute and relative original-to-synthetic drop.
* ``reasoning_delta_heatmap.png``: English-vs-non-English synthetic gap by reasoning mode.
* ``correction_comparison/*.png``: uncorrected vs corrected synthetic distributions when corrected logs exist.

Only successful logs are included by default. Repeated/resumed logs with the same
evaluation id are deduplicated, preferring a successful and then newer log.

Example:
    uv run paper/scripts/visualizegrid.py --workers 8
"""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from plot_config import (
    FAMILY_COLORS,
    FAMILY_ORDER,
    HUMAN_VERIFIED_LANGUAGES,
    LANGUAGE_LABELS,
    LANGUAGE_SPEAKERS,
    SPLIT_LABELS,
    language_order,
    model_sort_key,
    ordered_families,
    reasoning_sort_bucket,
)
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO_ROOT / "hf_dataset" / "logs"
DEFAULT_CORRECTED_LOG_DIR = REPO_ROOT / "hf_dataset" / "logs_unvalidated_revisions"
DEFAULT_OUT_DIR = REPO_ROOT / "paper" / "artifacts" / "figures" / "model_grid"
CORRECTION_COMPARISON_WIDTH = 13.5


FAMILY_MARKERS = {
    "Qwen2.5": "o",
    "Qwen3": "v",
    "Qwen": ">",
    "DeepSeek-R1-Distill-Qwen": "h",
    "Llama 3": "s",
    "DeepSeek-R1-Distill-Llama": "p",
    "Gemma 3": "^",
    "OLMo 2": "D",
    "OLMo 3": "d",
    "Granite": "X",
    "EuroLLM": "<",
    "Apertus": "P",
    "OpenAI": "X",
}
EXCLUDED_SPLIT_PAIR = ("OLMo-2-1124-7B-Instruct", "dan")
SPLIT_PAIR_LABELS = {
    ("Qwen2.5-1.5B-Instruct", "nob"): "Qwen2.5 1.5B (Norwegian)",
    ("Llama-3.2-3B-Instruct", "dan"): "Llama 3.2 3B (Danish)",
}

PROBLEM_KEYS = [
    "model_raw",
    "model",
    "family",
    "params_b",
    "vocab_size",
    "language",
    "split",
    "sample_id",
    "source_id",
]

EXCLUDED_SUMMARY_MODELS = {
    "Qwen3-0.6B (reasoning on)",
}

UNCORRECTED_COLOR = "#D95F50"
UNCORRECTED_FILL = "#F0A095"
CORRECTED_COLOR = "#5CA950"
CORRECTED_FILL = "#BCE6A8"
ORIGINAL_COLOR = "#111827"

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.dpi": 160,
        "font.family": "sans-serif",
    }
)


@dataclass(frozen=True)
class CorrectionComparisonRow:
    model: str
    original_accuracy: float
    uncorrected_sets: np.ndarray
    corrected_sets: np.ndarray


def _load_one_log(path: Path, scorer: str | None) -> tuple[str, pd.DataFrame | None, str | None]:
    """Load and score one full Inspect log.

    Returns a partially aggregated dataframe to reduce data copied back from
    worker processes.
    """
    try:
        log = read_eval_log(str(path))
    except Exception as exc:
        return path.name, None, f"Skipping unreadable log {path.name}: {exc}"

    parsed_task = parse_task(log.eval.task)
    if parsed_task is None:
        return path.name, None, f"Skipping unrecognized task {log.eval.task!r}"

    split, task_language = parsed_task
    label = f"{model_name(log.eval.model, log.eval.model_args)} / {task_language} / {split}"

    if not log.samples:
        return label, pd.DataFrame(), None

    info = infer_model_info(log.eval.model)
    rows: list[dict[str, Any]] = []

    for sample in log.samples:
        correct = sample_score(sample, scorer)
        if correct is None:
            continue

        metadata = sample.metadata or {}
        rows.append(
            {
                "model_raw": log.eval.model,
                "model": model_name(log.eval.model, log.eval.model_args),
                "family": info.family,
                "params_b": info.params_b,
                "vocab_size": info.vocab_size,
                "language": metadata.get("language", task_language),
                "split": split,
                "sample_id": sample.id,
                "source_id": metadata.get("source_id"),
                "correct": correct,
            }
        )

    if not rows:
        return label, pd.DataFrame(), None

    samples = pd.DataFrame(rows)

    grouped = samples.groupby(PROBLEM_KEYS, dropna=False, as_index=False)["correct"].agg(
        correct_sum="sum", correct_count="size"
    )

    return label, grouped, None


def load_samples(
    selected: list[tuple[Path, Any]],
    scorer: str | None,
    workers: int,
) -> pd.DataFrame:
    """Load selected logs, optionally in parallel."""
    frames: list[pd.DataFrame] = []
    paths = [path for path, _header in selected]

    if not paths:
        return pd.DataFrame()

    if workers <= 1:
        for index, path in enumerate(paths, start=1):
            label, frame, warning = _load_one_log(path, scorer)
            if warning:
                print(warning)
            else:
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
                else:
                    print(f"[{index}/{len(paths)}] {label}")
                if frame is not None and not frame.empty:
                    frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    samples = combined.groupby(PROBLEM_KEYS, dropna=False, as_index=False).agg(
        correct_sum=("correct_sum", "sum"), correct_count=("correct_count", "sum")
    )
    samples["correct"] = samples["correct_sum"] / samples["correct_count"]

    return samples.drop(columns=["correct_sum", "correct_count"])


def summarize(samples: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "model_raw",
        "model",
        "family",
        "params_b",
        "vocab_size",
        "language",
        "split",
    ]
    return (
        samples.groupby(group_cols, dropna=False)["correct"]
        .agg(accuracy="mean", n_problems="size", stderr="sem")
        .reset_index()
    )


def filter_summary_models(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[~summary["model"].isin(EXCLUDED_SUMMARY_MODELS)].copy()


def model_order(summary: pd.DataFrame) -> list[str]:
    return sorted(summary["model"].dropna().unique(), key=model_sort_key)


def sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    ordered = summary.copy()
    ordered["reasoning_order"] = ordered["model"].map(reasoning_sort_bucket)
    ordered["family_order"] = ordered["family"].map(FAMILY_ORDER).fillna(99)
    ordered["size_order"] = ordered["params_b"].fillna(math.inf)
    return ordered.sort_values(["family_order", "reasoning_order", "size_order", "model", "language", "split"]).drop(
        columns=["reasoning_order", "family_order", "size_order"]
    )


def path_slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def _synthetic_models(samples: pd.DataFrame, language: str) -> set[str]:
    rows = samples[(samples["language"] == language) & (samples["split"] == "synthetic")]
    return set(rows["model"].unique())


def paired_correction_languages(
    uncorrected: pd.DataFrame,
    corrected: pd.DataFrame,
    requested: list[str] | None,
) -> list[str]:
    required = {"split", "language"}
    if (
        uncorrected.empty
        or corrected.empty
        or not required.issubset(uncorrected.columns)
        or not required.issubset(corrected.columns)
    ):
        return []

    old = set(uncorrected.loc[uncorrected["split"] == "synthetic", "language"].unique())
    new = set(corrected.loc[corrected["split"] == "synthetic", "language"].unique())
    return language_order(old & new, requested)


def _sample_synthetic_sets(
    synthetic: pd.DataFrame,
    n_sets: int,
    rng: np.random.Generator,
) -> np.ndarray:
    variants = [
        group["correct"].to_numpy(dtype=float)
        for _source_id, group in synthetic.groupby("source_id", sort=True, dropna=False)
    ]
    if not variants:
        raise ValueError("Synthetic data contains no source templates.")

    set_totals = np.zeros(n_sets, dtype=float)
    for values in variants:
        set_totals += rng.choice(values, size=n_sets, replace=True)
    return set_totals / len(variants)


def _normal_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    mean = float(values.mean())
    std = max(float(values.std(ddof=1)), 0.005)
    lower = max(0.0, mean - 4 * std)
    upper = min(1.0, mean + 4 * std)
    x = np.linspace(lower, upper, 500)
    return x, norm.pdf(x, loc=mean, scale=std), mean


def _original_accuracy(corrected: pd.DataFrame, uncorrected: pd.DataFrame) -> float:
    for rows in (corrected, uncorrected):
        original = rows[rows["split"] == "original"]
        if not original.empty:
            return float(original["correct"].mean())
    return float("nan")


def collect_correction_comparison_rows(
    uncorrected: pd.DataFrame,
    corrected: pd.DataFrame,
    language: str,
    n_sets: int,
    seed: int,
) -> list[CorrectionComparisonRow]:
    models = sorted(
        _synthetic_models(uncorrected, language) & _synthetic_models(corrected, language),
        key=model_sort_key,
    )
    rows: list[CorrectionComparisonRow] = []

    for index, model in enumerate(models):
        old_rows = uncorrected[(uncorrected["language"] == language) & (uncorrected["model"] == model)]
        new_rows = corrected[(corrected["language"] == language) & (corrected["model"] == model)]
        rows.append(
            CorrectionComparisonRow(
                model=model,
                original_accuracy=_original_accuracy(new_rows, old_rows),
                uncorrected_sets=_sample_synthetic_sets(
                    old_rows[old_rows["split"] == "synthetic"],
                    n_sets,
                    np.random.default_rng(seed + index),
                ),
                corrected_sets=_sample_synthetic_sets(
                    new_rows[new_rows["split"] == "synthetic"],
                    n_sets,
                    np.random.default_rng(seed + index),
                ),
            )
        )

    return rows


def english_metric_pairs(summary: pd.DataFrame, split: str = "synthetic") -> pd.DataFrame:
    paired = summary[summary["split"] == split].pivot_table(
        index=["model", "family", "params_b"],
        columns="language",
        values="accuracy",
    )
    if not {"eng", "eng_metric"}.issubset(paired.columns):
        return pd.DataFrame(columns=["model", "family", "params_b", "eng", "eng_metric", "metric_minus_eng"])

    result = paired[["eng", "eng_metric"]].dropna().reset_index()
    result["metric_minus_eng"] = (result["eng_metric"] - result["eng"]).round(10)
    result = sort_summary(result.assign(language="eng", split=split)).drop(columns=["language", "split"])
    average = pd.DataFrame(
        [
            {
                "model": "Average",
                "family": "Average",
                "params_b": np.nan,
                "eng": result["eng"].mean(),
                "eng_metric": result["eng_metric"].mean(),
                "metric_minus_eng": result["metric_minus_eng"].mean(),
            }
        ]
    )
    return pd.concat([average, result], ignore_index=True)


def format_speaker_count(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:g}M"
    if count >= 1_000:
        return f"{count / 1_000:g}K"
    return str(count)


def heatmap_language_label(language: str) -> str:
    name = LANGUAGE_LABELS.get(language, language)
    speakers = LANGUAGE_SPEAKERS.get(language)
    return f"{name}\n({format_speaker_count(speakers)} speakers)" if speakers is not None else name


def annotated_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    signed: bool = False,
) -> Any:
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    families = [infer_model_info(model).family for model in matrix.index]
    for row in range(1, len(families)):
        if families[row] != families[row - 1]:
            ax.axhline(
                row - 0.5,
                color="white",
                linewidth=1.5,
                alpha=0.9,
            )

    ax.set_xticks(
        range(len(matrix.columns)),
        [heatmap_language_label(x) for x in matrix.columns],
        rotation=40,
        ha="right",
    )

    for col, language in enumerate(matrix.columns):
        if language not in HUMAN_VERIFIED_LANGUAGES:
            continue
        ax.annotate(
            "*",
            xy=(col, 0),
            xycoords=ax.get_xaxis_transform(),
            xytext=(1, -4),
            textcoords="offset points",
            color="#FFC107",
            fontweight="bold",
            rotation=40,
            ha="left",
            va="top",
            annotation_clip=False,
        )

    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title(title)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isnan(value):
                continue
            label = f"{value:+.1%}" if signed else f"{value:.1%}"
            ax.text(col, row, label, ha="center", va="center", fontsize=7)

    return image


def plot_heatmaps(summary: pd.DataFrame, out: Path) -> None:
    order = model_order(summary)
    languages = language_order(summary["language"].unique())

    available_splits = [split for split in ("original", "synthetic") if split in set(summary["split"])]

    matrices: dict[str, pd.DataFrame] = {}
    for split in available_splits:
        split_rows = summary[summary["split"] == split]
        matrices[split] = split_rows.pivot_table(
            index="model",
            columns="language",
            values="accuracy",
        ).reindex(index=order, columns=languages)

    panels = [(SPLIT_LABELS[split], matrices[split]) for split in available_splits]

    height = max(4.0, 0.42 * len(order) + 1.5)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(max(6, 2.5 * len(languages) * len(panels)), height),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    images = [
        annotated_heatmap(ax, matrix, title, "viridis", 0, 1, False)
        for ax, (title, matrix) in zip(axes, panels, strict=True)
    ]

    axes[0].set_ylabel("Evaluated instruction-tuned model")
    fig.suptitle("Exact-answer accuracy by model, problem language, and benchmark split")
    fig.text(0.465, 0.02, "*", color="#FFC107", fontweight="bold", ha="right", fontsize=8)
    fig.text(0.467, 0.02, "Human verified", ha="left", fontsize=8)
    fig.subplots_adjust(left=0.2, right=0.92, bottom=0.28, top=0.84, wspace=0.18)

    colorbar = fig.colorbar(
        images[0],
        ax=axes,
        shrink=0.8,
        label="Problems answered correctly",
    )
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_split_pairs(summary: pd.DataFrame, out: Path) -> bool:
    if not {"original", "synthetic"}.issubset(summary["split"].unique()):
        return False

    paired = summary.pivot_table(
        index=["model", "family", "params_b", "language"],
        columns="split",
        values="accuracy",
    ).dropna(subset=["original", "synthetic"])

    if paired.empty:
        return False

    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    sized = paired.reset_index()
    sized = sized[~((sized["model"] == EXCLUDED_SPLIT_PAIR[0]) & (sized["language"] == EXCLUDED_SPLIT_PAIR[1]))]
    finite_sizes = sized["params_b"].dropna()
    norm = Normalize(
        vmin=float(finite_sizes.min()) if not finite_sizes.empty else 0,
        vmax=float(finite_sizes.max()) if not finite_sizes.empty else 1,
    )
    cmap = plt.get_cmap("viridis")

    for family in ordered_families(sized["family"]):
        family_rows = sized[sized["family"] == family]
        marker = FAMILY_MARKERS.get(family, "v")
        known_size = family_rows.dropna(subset=["params_b"])
        if not known_size.empty:
            ax.scatter(
                known_size["original"],
                known_size["synthetic"],
                c=known_size["params_b"],
                cmap=cmap,
                norm=norm,
                marker=marker,
                s=58,
                alpha=0.9,
                edgecolors="white",
                linewidths=0.5,
            )

        unknown_size = family_rows[family_rows["params_b"].isna()]
        if not unknown_size.empty:
            ax.scatter(
                unknown_size["original"],
                unknown_size["synthetic"],
                color="#888888",
                marker=marker,
                s=58,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.5,
            )

    for row in sized.itertuples():
        label = SPLIT_PAIR_LABELS.get((row.model, row.language))
        if label:
            ax.annotate(label, (row.original, row.synthetic), xytext=(5, 0), textcoords="offset points", fontsize=7)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    colorbar = fig.colorbar(mappable, ax=ax, shrink=0.82)
    colorbar.set_label("Model parameters (billions)")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=FAMILY_MARKERS.get(family, "v"),
            linestyle="none",
            markerfacecolor="#777777",
            markeredgecolor="white",
            markersize=7,
        )
        for family in ordered_families(sized["family"])
    ]
    if legend_handles:
        fig.legend(
            legend_handles,
            ordered_families(sized["family"]),
            loc="lower center",
            ncol=min(6, len(legend_handles)),
            frameon=False,
            title="Model family",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, alpha=0.6)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Accuracy on original benchmark questions",
        ylabel="Accuracy on synthetic numerical variants",
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return True


def plot_english_normalized_transfer(summary: pd.DataFrame, out: Path) -> bool:
    """Plot each language's accuracy difference from the same model's English score."""
    order = model_order(summary)
    splits = [split for split in ("original", "synthetic") if split in set(summary["split"])]
    languages = language_order(language for language in summary["language"].unique() if language != "eng")

    panels: list[tuple[str, pd.DataFrame]] = []
    for split in splits:
        matrix = (
            summary[summary["split"] == split]
            .pivot_table(index="model", columns="language", values="accuracy")
            .reindex(index=order)
        )
        if "eng" not in matrix or not languages:
            continue
        transfer = matrix.reindex(columns=languages).sub(matrix["eng"], axis=0)
        if transfer.notna().any().any():
            panels.append((SPLIT_LABELS[split], transfer))

    if not panels:
        return False

    max_gap = max(np.nanmax(np.abs(matrix.to_numpy())) for _title, matrix in panels if matrix.notna().any().any())
    max_gap = max(max_gap, 0.05)
    height = max(4.0, 0.42 * len(order) + 1.5)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(max(7, 2.8 * len(languages) * len(panels)), height),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    images = [
        annotated_heatmap(
            ax,
            matrix,
            title,
            "RdBu",
            -max_gap,
            max_gap,
            signed=True,
        )
        for ax, (title, matrix) in zip(axes, panels, strict=True)
    ]

    axes[0].set_ylabel("Evaluated instruction-tuned model")
    fig.suptitle("Target-language accuracy minus English accuracy for the same model")
    fig.subplots_adjust(left=0.2, right=0.89, bottom=0.27, top=0.84, wspace=0.18)
    colorbar_axis = fig.add_axes([0.91, 0.25, 0.012, 0.55])
    colorbar = fig.colorbar(
        images[0],
        cax=colorbar_axis,
        label="Accuracy difference: target language - English",
    )
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_eng_metric_comparison(summary: pd.DataFrame, out: Path, split: str = "synthetic") -> bool:
    paired = english_metric_pairs(summary, split)
    if paired.empty:
        return False

    paired = paired.reset_index(drop=True)
    y = np.arange(len(paired))
    height = max(4.0, 0.34 * len(paired) + 1.6)
    fig, ax = plt.subplots(figsize=(9.5, height))

    for index, row in paired.iterrows():
        color = FAMILY_COLORS.get(row["family"], "#4B5563")
        ax.plot(
            [row["eng"], row["eng_metric"]],
            [index, index],
            color=color,
            linewidth=3.0,
            solid_capstyle="round",
            alpha=0.85,
        )
        ax.scatter(row["eng"], index, color="white", edgecolor=color, linewidth=1.8, s=44, zorder=3)
        ax.scatter(row["eng_metric"], index, color=color, edgecolor=color, linewidth=1.2, s=44, zorder=3)

        label_x = max(row["eng"], row["eng_metric"]) + 0.012
        ax.text(
            min(label_x, 0.985),
            index,
            f"{row['metric_minus_eng']:+.1%}",
            va="center",
            ha="left" if label_x < 0.985 else "right",
            fontsize=8,
            color="#374151",
        )

    ax.set_yticks(y, paired["model"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Exact-answer accuracy")
    ax.set_ylabel("Evaluated instruction-tuned model")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"English metric vs English accuracy ({SPLIT_LABELS.get(split, split)})")
    ax.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#4B5563", label="English"
            ),
            Line2D([0], [0], marker="o", color="#4B5563", markerfacecolor="#4B5563", label="English metric"),
        ],
        loc="lower right",
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def transfer_robustness_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize the English transfer penalty across non-English languages."""
    indexed = summary.pivot_table(
        index=["model", "family", "params_b", "split"],
        columns="language",
        values="accuracy",
    )
    non_english = [column for column in indexed.columns if column != "eng"]
    if "eng" not in indexed or not non_english:
        return pd.DataFrame()

    penalties = indexed[non_english].rsub(indexed["eng"], axis=0)
    result = indexed.reset_index()[["model", "family", "params_b", "split"]].copy()
    result["mean_transfer_penalty"] = penalties.mean(axis=1, skipna=True).to_numpy()
    result["language_dispersion"] = indexed[non_english].std(axis=1, ddof=0, skipna=True).to_numpy()
    result["n_transfer_languages"] = penalties.notna().sum(axis=1).to_numpy()
    return result[result["n_transfer_languages"] > 0]


def plot_transfer_robustness(summary: pd.DataFrame, out: Path) -> bool:
    robust = transfer_robustness_summary(summary).dropna(subset=["params_b"])
    if robust.empty:
        return False

    splits = [split for split in ("original", "synthetic") if split in set(robust["split"])]
    metrics = [
        (
            "mean_transfer_penalty",
            "Mean accuracy gap:\nEnglish - non-English languages",
        ),
        (
            "language_dispersion",
            "Standard deviation of accuracy\nacross non-English languages",
        ),
    ]
    fig, axes = plt.subplots(
        len(splits),
        len(metrics),
        figsize=(11, 4 * len(splits)),
        squeeze=False,
    )

    for row_index, split in enumerate(splits):
        split_rows = robust[robust["split"] == split]
        for col_index, (metric, label) in enumerate(metrics):
            ax = axes[row_index, col_index]
            for family in ordered_families(split_rows["family"]):
                family_rows = split_rows[split_rows["family"] == family]
                family_rows = family_rows.sort_values("params_b")
                ax.plot(
                    family_rows["params_b"],
                    family_rows[metric],
                    marker="o",
                    linewidth=1.7,
                    color=FAMILY_COLORS.get(family, "#666666"),
                    label=family,
                )
                for row in family_rows.itertuples():
                    ax.annotate(
                        row.model,
                        (row.params_b, getattr(row, metric)),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=6.5,
                    )

            ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
            ax.set_xscale("log")
            ax.grid(alpha=0.2)
            ax.set_xlabel("Model parameters (billions; logarithmic scale)")
            ax.set_ylabel(label)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_title(SPLIT_LABELS[split])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle("English-to-non-English accuracy gaps and variability by model parameter count")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_split_degradation(summary: pd.DataFrame, out: Path) -> bool:
    if not {"original", "synthetic"}.issubset(summary["split"].unique()):
        return False

    order = model_order(summary)
    languages = language_order(summary["language"].unique())
    paired = summary.pivot_table(
        index="model",
        columns=["split", "language"],
        values="accuracy",
    )
    original = paired["original"].reindex(index=order, columns=languages)
    synthetic = paired["synthetic"].reindex(index=order, columns=languages)
    absolute = original - synthetic
    relative = absolute.div(original.where(original != 0))

    if not absolute.notna().any().any():
        return False

    absolute_limit = max(float(np.nanmax(np.abs(absolute.to_numpy()))), 0.05)
    finite_relative = relative.to_numpy()[np.isfinite(relative.to_numpy())]
    relative_limit = max(float(np.max(np.abs(finite_relative))) if finite_relative.size else 0, 0.05)

    height = max(4.0, 0.42 * len(order) + 1.5)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(12, 5 * len(languages)), height),
        sharey=True,
    )
    absolute_image = annotated_heatmap(
        axes[0],
        absolute,
        "Absolute accuracy drop: original - synthetic",
        "RdBu_r",
        -absolute_limit,
        absolute_limit,
        signed=True,
    )
    relative_image = annotated_heatmap(
        axes[1],
        relative,
        "Relative accuracy drop: (original - synthetic) / original",
        "RdBu_r",
        -relative_limit,
        relative_limit,
        signed=True,
    )
    axes[0].set_ylabel("Evaluated instruction-tuned model")
    fig.suptitle("Accuracy decrease from original benchmark questions to synthetic numerical variants")
    absolute_colorbar = fig.colorbar(
        absolute_image,
        ax=axes[0],
        shrink=0.8,
        label="Absolute accuracy difference: original - synthetic",
    )
    absolute_colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    relative_colorbar = fig.colorbar(
        relative_image,
        ax=axes[1],
        shrink=0.8,
        label="Accuracy drop divided by original accuracy",
    )
    relative_colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.text(
        0.5,
        0.02,
        "Positive values mean accuracy is lower on synthetic variants; negative values mean it is higher.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.2, right=0.92, bottom=0.3, top=0.84, wspace=0.18)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def reasoning_variant_name(model: str) -> tuple[str, str | None]:
    suffixes = {
        " (reasoning on)": "on",
        " (reasoning off)": "off",
    }
    for suffix, mode in suffixes.items():
        if model.endswith(suffix):
            return model[: -len(suffix)], mode
    return model, None


def plot_reasoning_delta(summary: pd.DataFrame, out: Path) -> bool:
    rows = summary[summary["split"] == "synthetic"].copy()
    parsed = rows["model"].map(reasoning_variant_name)
    rows["base_model_candidate"] = [item[0] for item in parsed]
    rows["reasoning"] = [item[1] for item in parsed]

    off_name_by_raw = rows[rows["reasoning"] == "off"].groupby("model_raw")["base_model_candidate"].first()
    rows["canonical_base_model"] = rows["model_raw"].map(off_name_by_raw).fillna(rows["base_model_candidate"])

    rows.loc[rows["reasoning"].isin(["on", "off"]), "base_model"] = rows["canonical_base_model"]
    rows.loc[
        rows["reasoning"].isna() & (rows["model"] == rows["canonical_base_model"]),
        "reasoning",
    ] = "on"
    rows.loc[rows["base_model"].isna(), "base_model"] = rows["canonical_base_model"]

    rows = rows[rows["reasoning"].isin(["on", "off"])]
    if rows.empty:
        return False

    averaged = (
        rows.groupby(["base_model", "reasoning", "family", "params_b", "language"], dropna=False)["accuracy"]
        .mean()
        .reset_index()
    )
    by_language = averaged.set_index(["base_model", "reasoning", "family", "params_b", "language"])[
        "accuracy"
    ].unstack("language")
    if "eng" not in by_language.columns:
        return False

    non_english = [language for language in by_language.columns if language != "eng"]
    if not non_english:
        return False

    by_language["gap"] = by_language["eng"] - by_language[non_english].mean(axis=1)
    by_language["relative_gap"] = by_language["gap"] / by_language["eng"].replace(0, np.nan)
    gaps = by_language[["gap", "relative_gap"]].dropna().reset_index()

    gaps = gaps[np.isfinite(gaps["params_b"])]
    if gaps.empty:
        return False

    line_styles = {"on": "-", "off": ":"}
    mode_labels = {"on": "reasoning on", "off": "reasoning off"}
    families = ordered_families(gaps["family"])
    fig, ax = plt.subplots(figsize=(max(8.5, 1.0 * len(families) + 6.0), 5.2))
    for family in families:
        family_rows = gaps[gaps["family"] == family]
        for mode in ["on", "off"]:
            line = family_rows[family_rows["reasoning"] == mode].sort_values("params_b")
            if line.empty:
                continue
            ax.plot(
                line["params_b"],
                line["relative_gap"],
                color=FAMILY_COLORS.get(family, "#666666"),
                linestyle=line_styles[mode],
                marker=FAMILY_MARKERS.get(family, "o"),
                linewidth=1.8,
                markersize=6,
                label=f"{family} {mode_labels[mode]}",
            )

    ax.axhline(0, color="#111827", linewidth=0.8, alpha=0.7)
    ax.set(xlabel="Model size (B parameters)", ylabel="Relative transfer gap")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    family_handles = [
        Line2D([0], [0], color=FAMILY_COLORS.get(family, "#666666"), marker=FAMILY_MARKERS.get(family, "o"), label=family)
        for family in families
    ]
    mode_handles = [
        Line2D([0], [0], color="#111827", linestyle=line_styles[mode], label=mode_labels[mode]) for mode in ["on", "off"]
    ]
    first_legend = ax.legend(handles=family_handles, title="Model family", frameon=False, loc="upper left")
    ax.add_artist(first_legend)
    ax.legend(handles=mode_handles, title="Variant", frameon=False, loc="upper right")
    fig.suptitle("Relative transfer gap by model size and reasoning mode")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_correction_comparison(rows: list[CorrectionComparisonRow], language: str, out: Path) -> None:
    fig, axes = plt.subplots(
        len(rows),
        1,
        figsize=(CORRECTION_COMPARISON_WIDTH, 1.45 * len(rows) + 1.35),
        sharex=True,
        squeeze=False,
    )

    for ax, row in zip(axes[:, 0], rows, strict=True):
        old_counts, _, _ = ax.hist(
            row.uncorrected_sets,
            bins=18,
            density=True,
            color=UNCORRECTED_FILL,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.38,
            zorder=1,
        )
        new_counts, _, _ = ax.hist(
            row.corrected_sets,
            bins=18,
            density=True,
            color=CORRECTED_FILL,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.38,
            zorder=1,
        )
        old_x, old_density, old_mean = _normal_curve(row.uncorrected_sets)
        new_x, new_density, new_mean = _normal_curve(row.corrected_sets)
        peak = max(
            float(old_counts.max()),
            float(new_counts.max()),
            float(old_density.max()),
            float(new_density.max()),
        )

        ax.fill_between(old_x, 0, old_density, color=UNCORRECTED_FILL, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(old_x, old_density, color=UNCORRECTED_COLOR, linewidth=1.8, zorder=3)
        ax.axvline(old_mean, color=UNCORRECTED_COLOR, linewidth=1.2, zorder=4)

        ax.fill_between(new_x, 0, new_density, color=CORRECTED_FILL, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(new_x, new_density, color=CORRECTED_COLOR, linewidth=1.8, zorder=3)
        ax.axvline(new_mean, color=CORRECTED_COLOR, linewidth=1.2, zorder=4)

        ax.set_ylabel(row.model, rotation=0, ha="right", va="center", labelpad=58, fontsize=11)
        ax.set_yticks([])
        ax.set_ylim(0, peak * 1.2)
        ax.grid(axis="x", color="#D8DEE8", linewidth=0.7, alpha=0.6)

    axes[-1, 0].set_xlim(0, 1)
    axes[-1, 0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[-1, 0].set_xlabel("Exact-answer accuracy", fontsize=12, labelpad=8)

    legend = [
        Line2D([0], [0], color=UNCORRECTED_COLOR, lw=2, label="Uncorrected"),
        Line2D([0], [0], color=CORRECTED_COLOR, lw=2, label="Corrected"),
    ]
    axes[0, 0].legend(handles=legend, loc="upper right", frameon=False, ncol=3, bbox_to_anchor=(1, 1.65))
    fig.suptitle(
        f"{LANGUAGE_LABELS.get(language, language)} correction comparison",
        x=0.08,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.06, 0.02, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_family_scaling(summary: pd.DataFrame, out: Path) -> bool:
    scaled = summary.dropna(subset=["params_b"])
    families = ordered_families(scaled["family"])

    if not families:
        return False

    splits = [split for split in ("original", "synthetic") if split in set(scaled["split"])]

    fig, axes = plt.subplots(
        len(splits),
        len(families),
        figsize=(5 * len(families), 4 * len(splits)),
        sharey=True,
        squeeze=False,
    )

    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    languages = language_order(scaled["language"].unique(), english_first=True)
    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    for row_index, split in enumerate(splits):
        for col_index, family in enumerate(families):
            ax = axes[row_index, col_index]
            panel = scaled[(scaled["split"] == split) & (scaled["family"] == family)]

            for language, marker in zip(languages, markers, strict=False):
                line = panel[panel["language"] == language].sort_values("params_b")
                if line.empty:
                    continue

                ax.errorbar(
                    line["params_b"],
                    line["accuracy"],
                    yerr=line["stderr"].fillna(0),
                    marker=marker,
                    linewidth=1.5,
                    capsize=2,
                    label=LANGUAGE_LABELS.get(language, language),
                )

            ax.set_xscale("log")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.2)
            ax.set_title(f"{family} on {SPLIT_LABELS[split].lower()}")
            ax.set_xlabel("Model parameters (billions; logarithmic scale)")

            if col_index == 0:
                ax.set_ylabel("Problems answered correctly")
                ax.yaxis.set_major_formatter(PercentFormatter(1))

            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels, strict=True):
                if label not in legend_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(5, len(legend_labels)),
            frameon=False,
        )

    fig.suptitle("Exact-answer accuracy by model size within each model family and problem language")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )
    parser.add_argument(
        "--corrected-log-dir",
        type=Path,
        default=DEFAULT_CORRECTED_LOG_DIR,
        help="Corrected-log directory for correction comparison figures.",
    )
    parser.add_argument(
        "--correction-samples",
        type=int,
        default=2_000,
        help="Synthetic benchmark sets to sample for correction comparison figures.",
    )
    parser.add_argument("--correction-seed", type=int, default=0)
    parser.add_argument(
        "--scorer",
        help="Inspect scorer name to use; defaults to math, pattern, then first numeric score.",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include readable samples from non-success logs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Workers used for log selection and full log loading. Use 1 to disable parallelism.",
    )

    args = parser.parse_args()

    if args.correction_samples < 2:
        parser.error("--correction-samples must be at least 2")

    paths = discover_logs(args.log_dir)
    selected = select_logs(paths, args.include_incomplete, workers=args.workers)

    print(f"Discovered {len(paths)} logs; selected {len(selected)} after status filtering and deduplication.")
    print(f"Loading selected logs with {args.workers} worker(s).")

    samples = load_samples(selected, args.scorer, workers=args.workers)

    if samples.empty:
        raise SystemExit("No scored samples found in the selected logs.")

    summary = filter_summary_models(summarize(samples))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.out_dir / "run_summary.csv"
    sort_summary(summary).to_csv(summary_path, index=False)

    plot_heatmaps(summary, args.out_dir / "accuracy_heatmaps.png")
    made_pairs = plot_split_pairs(summary, args.out_dir / "original_vs_synthetic.png")
    made_scaling = plot_family_scaling(summary, args.out_dir / "family_scaling.png")
    made_transfer = plot_english_normalized_transfer(
        summary,
        args.out_dir / "english_normalized_transfer.png",
    )
    made_metric = plot_eng_metric_comparison(
        summary,
        args.out_dir / "eng_vs_eng_metric.png",
    )
    made_robustness = plot_transfer_robustness(
        summary,
        args.out_dir / "transfer_robustness.png",
    )
    made_degradation = plot_split_degradation(
        summary,
        args.out_dir / "split_degradation_heatmaps.png",
    )
    made_reasoning = plot_reasoning_delta(
        summary,
        args.out_dir / "reasoning_delta_heatmap.png",
    )
    correction_outputs: list[Path] = []
    if args.corrected_log_dir:
        corrected_paths = discover_logs([args.corrected_log_dir])
        corrected_selected = select_logs(corrected_paths, args.include_incomplete, workers=args.workers)
        print(
            f"Discovered {len(corrected_paths)} corrected logs; "
            f"selected {len(corrected_selected)} after status filtering and deduplication."
        )
        corrected = load_samples(corrected_selected, args.scorer, workers=args.workers)
        if corrected.empty:
            print(f"Skipped correction comparison: no scored corrected samples found in {args.corrected_log_dir}.")
        else:
            languages = paired_correction_languages(samples, corrected, requested=None)
            for language in languages:
                rows = collect_correction_comparison_rows(
                    samples,
                    corrected,
                    language,
                    args.correction_samples,
                    args.correction_seed,
                )
                if not rows:
                    print(f"Skipping {language}: no paired corrected models.")
                    continue
                out = args.out_dir / "correction_comparison" / f"{path_slug(language)}.png"
                plot_correction_comparison(rows, language, out)
                correction_outputs.append(out)

    print(f"Saved {summary_path}")
    print(f"Saved {args.out_dir / 'accuracy_heatmaps.png'}")

    if made_pairs:
        print(f"Saved {args.out_dir / 'original_vs_synthetic.png'}")
    else:
        print("Skipped original_vs_synthetic.png: no model/language has both splits.")

    if made_scaling:
        print(f"Saved {args.out_dir / 'family_scaling.png'}")
    else:
        print("Skipped family_scaling.png: no recognized model parameter counts.")

    if made_transfer:
        print(f"Saved {args.out_dir / 'english_normalized_transfer.png'}")
    else:
        print("Skipped english_normalized_transfer.png: paired English/non-English results are required.")

    if made_metric:
        print(f"Saved {args.out_dir / 'eng_vs_eng_metric.png'}")
    else:
        print("Skipped eng_vs_eng_metric.png: paired English and English-metric results are required.")

    if made_robustness:
        print(f"Saved {args.out_dir / 'transfer_robustness.png'}")
    else:
        print("Skipped transfer_robustness.png: paired transfer results with model sizes are required.")

    if made_degradation:
        print(f"Saved {args.out_dir / 'split_degradation_heatmaps.png'}")
    else:
        print("Skipped split_degradation_heatmaps.png: paired original/synthetic results are required.")

    if made_reasoning:
        print(f"Saved {args.out_dir / 'reasoning_delta_heatmap.png'}")
    else:
        print("Skipped reasoning_delta_heatmap.png: paired synthetic English/non-English reasoning results are required.")

    for out in correction_outputs:
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
