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
* ``eng_vs_eng_metric_selected.png``: selected-model English/English-metric distributions.
* ``eng_vs_eng_metric_full.png``: all paired-model English/English-metric distributions.
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
from dataclasses import dataclass
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
    normal_curve,
    parse_task,
    sample_score,
    sample_synthetic_sets,
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
    PLOT_STYLE,
    SPLIT_LABELS,
    heatmap_language_label,
    language_order,
    model_sort_key,
    ordered_families,
    path_slug,
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
    ("Apertus-8B-Instruct-2509", "zho"): "Apertus 8B (Chinese)",
    ("granite-3.2-2b-instruct (reasoning on)", "zho"): "Granite 2B (Chinese)",
    ("gemma-3-4b-it", "isl"): "Gemma 3 4B (Icelandic)",
    ("OLMo-2-0425-1B-Instruct", "eng"): "OLMo 2 1B (English)",
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
SELECTED_MODEL_LABELS = {
    "Apertus-8B-Instruct-2509": "Apertus 8B",
    "EuroLLM-9B-Instruct-2512": "EuroLLM 9B",
    "OLMo-2-0425-1B-Instruct": "OLMo 2 1B",
    "OLMo-2-0325-32B-Instruct": "OLMo 2 32B",
    "gemma-3-12b-it": "Gemma 3 12B",
    "gemma-3-27b-it": "Gemma 3 27B",
    "granite-3.2-2b-instruct (reasoning off)": "Granite 3.2 2B\n(reasoning off)",
    "granite-3.2-8b-instruct (reasoning on)": "Granite 3.2 8B\n(reasoning)",
}

plt.rcParams.update(PLOT_STYLE)


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

    del log  # free the large decompressed log before returning
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

    _CONCAT_CHUNK = 8  # flush accumulated frames every N logs to cap memory

    for index, (label, frame, warning) in enumerate(map_log_loader(_load_one_log, paths, scorer, workers), start=1):
        if warning:
            print(warning)
        else:
            print(f"[{index}/{len(paths)}] {label}")
        if frame is not None and not frame.empty:
            frames.append(frame)
        if len(frames) >= _CONCAT_CHUNK:
            frames = [pd.concat(frames, ignore_index=True)]

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
                uncorrected_sets=sample_synthetic_sets(
                    old_rows[old_rows["split"] == "synthetic"],
                    n_sets,
                    np.random.default_rng(seed + index),
                )[0],
                corrected_sets=sample_synthetic_sets(
                    new_rows[new_rows["split"] == "synthetic"],
                    n_sets,
                    np.random.default_rng(seed + index),
                )[0],
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
            ax.annotate(
                label,
                (row.original, row.synthetic),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1},
            )

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
    """Plot each language's accuracy as a share of the same model's English score."""
    order = model_order(summary)
    splits = [split for split in ["synthetic"] if split in set(summary["split"])]
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
        transfer = matrix.reindex(columns=languages).div(matrix["eng"].replace(0, np.nan), axis=0)
        if transfer.notna().any().any():
            panels.append((SPLIT_LABELS[split], transfer))

    if not panels:
        return False

    finite_values = np.concatenate(
        [matrix.to_numpy()[np.isfinite(matrix.to_numpy())] for _title, matrix in panels]
    )
    value_min = min(0.0, float(finite_values.min()))
    value_max = max(1.0, float(finite_values.max()))
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
            "YlGnBu",
            value_min,
            value_max,
            signed=False,
        )
        for ax, (title, matrix) in zip(axes, panels, strict=True)
    ]

    axes[0].set_ylabel("Evaluated instruction-tuned model")
    fig.suptitle("Percentage of English performance recovered by target language")
    fig.subplots_adjust(left=0.2, right=0.89, bottom=0.27, top=0.84, wspace=0.18)
    colorbar_axis = fig.add_axes([0.91, 0.25, 0.012, 0.55])
    colorbar = fig.colorbar(
        images[0],
        cax=colorbar_axis,
        label="Percentage of English performance recovered",
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


def plot_reasoning_delta(summary: pd.DataFrame, out: Path) -> bool:
    rows = summary[summary["split"] == "synthetic"].copy()
    rows = classify_reasoning_variants(rows)
    rows.loc[rows["reasoning"].isin(["on", "off"]), "base_model"] = rows["canonical_base_model"]
    rows.loc[rows["base_model"].isna(), "base_model"] = rows["canonical_base_model"]
    if rows.empty:
        return False

    averaged = (
        rows.groupby(["base_model", "reasoning", "family", "params_b", "language"], dropna=False)
        .agg(accuracy=("accuracy", "mean"), stderr=("stderr", "mean"))
        .reset_index()
    )
    by_model = averaged.set_index(["base_model", "reasoning", "family", "params_b", "language"])
    by_language = by_model["accuracy"].unstack("language")
    stderr_by_language = by_model["stderr"].unstack("language").fillna(0)
    if "eng" not in by_language.columns:
        return False

    non_english = [language for language in by_language.columns if language not in {"eng", "eng_metric"}]
    if not non_english:
        return False

    english_accuracy = by_language["eng"].replace(0, np.nan)
    non_english_accuracy = by_language[non_english].mean(axis=1)
    non_english_stderr = np.sqrt(stderr_by_language[non_english].pow(2).sum(axis=1)) / by_language[non_english].count(
        axis=1
    )
    performance_recovered_stderr = np.sqrt(
        (non_english_accuracy / english_accuracy.pow(2)).pow(2) * stderr_by_language["eng"].pow(2)
        + (non_english_stderr / english_accuracy).pow(2)
    )
    by_language["performance_recovered"] = non_english_accuracy / english_accuracy
    by_language["performance_recovered_ci95"] = (
        performance_recovered_stderr * norm.ppf(0.975)
    ).fillna(0)
    gaps = by_language[["performance_recovered", "performance_recovered_ci95"]].dropna().reset_index()

    gaps = gaps[np.isfinite(gaps["params_b"])]
    gaps = gaps[gaps.groupby("base_model")["reasoning"].transform("nunique") == 2]
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
            ax.errorbar(
                line["params_b"],
                line["performance_recovered"],
                yerr=line["performance_recovered_ci95"],
                color=FAMILY_COLORS.get(family, "#666666"),
                linestyle=line_styles[mode],
                marker=FAMILY_MARKERS.get(family, "o"),
                linewidth=1.8,
                markersize=6,
                capsize=2,
                label=f"{family} {mode_labels[mode]}",
            )

    ax.axhline(1, color="#111827", linewidth=0.8, alpha=0.7)
    ax.set(xlabel="Model size (B parameters)", ylabel="Percentage of English performance recovered")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    family_handles = [
        Line2D(
            [0], [0], color=FAMILY_COLORS.get(family, "#666666"), marker=FAMILY_MARKERS.get(family, "o"), label=family
        )
        for family in families
    ]
    mode_handles = [
        Line2D([0], [0], color="#111827", linestyle=line_styles[mode], label=mode_labels[mode])
        for mode in ["on", "off"]
    ]
    first_legend = ax.legend(handles=family_handles, title="Model family", frameon=False, loc="upper left")
    ax.add_artist(first_legend)
    ax.legend(handles=mode_handles, title="Variant", frameon=False, loc="upper right")
    fig.suptitle("Percentage of English performance recovered by model size and reasoning mode")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_correction_comparison(
    rows: list[CorrectionComparisonRow],
    language: str,
    out: Path,
    legend_labels: tuple[str, str] = ("Uncorrected", "Corrected"),
    title: str | None = None,
) -> None:
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
        old_x, old_density, old_mean, _ = normal_curve(row.uncorrected_sets)
        new_x, new_density, new_mean, _ = normal_curve(row.corrected_sets)
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
        Line2D([0], [0], color=UNCORRECTED_COLOR, lw=2, label=legend_labels[0]),
        Line2D([0], [0], color=CORRECTED_COLOR, lw=2, label=legend_labels[1]),
    ]
    axes[0, 0].legend(handles=legend, loc="upper right", frameon=False, ncol=3, bbox_to_anchor=(1, 1.65))
    fig.suptitle(
        title or f"{LANGUAGE_LABELS.get(language, language)} correction comparison",
        x=0.08,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.06, 0.02, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_correction_comparison_selected(
    rows: list[CorrectionComparisonRow],
    language: str,
    out: Path,
    target_models: list[str] | None = None,
    legend_labels: tuple[str, str] = ("Unvalidated", "Validated"),
) -> None:
    if not target_models:
        target_models = [
            "gemma-3-12b-it",
            "Apertus-8B-Instruct-2509",
            "granite-3.2-2b-instruct (reasoning off)",
        ]

    selected_rows = [r for r in rows if r.model in target_models]
    if len(selected_rows) < 3:
        step = max(1, len(rows) // 3)
        selected_rows = [rows[0], rows[min(step, len(rows) - 1)], rows[min(2 * step, len(rows) - 1)]]

    # Sized for a single-column paper figure.  The larger type remains legible
    # after LaTeX scales the image to the column width.
    fig, ax = plt.subplots(figsize=(7.6, 3.2))

    # Distinct colour per model; unvalidated = solid, validated = dashed
    MODEL_COLORS = [
        "#1B365D",  # deep navy
        "#C0392B",  # crimson
        "#1A6B3C",  # forest green
        "#7B3FA0",  # purple
        "#D4680F",  # burnt orange (spare)
    ]

    all_peaks = []
    model_labels: list[tuple[float, float, str, str]] = []

    for i, row in enumerate(selected_rows):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        old_counts, _, _ = ax.hist(
            row.uncorrected_sets,
            bins=18,
            density=True,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.18,
            zorder=1,
        )
        new_counts, _, _ = ax.hist(
            row.corrected_sets,
            bins=18,
            density=True,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.30,
            zorder=1,
        )

        old_x, old_density, old_mean, _ = normal_curve(row.uncorrected_sets)
        new_x, new_density, new_mean, _ = normal_curve(row.corrected_sets)

        peak = max(
            float(old_counts.max()) if len(old_counts) > 0 else 0.0,
            float(new_counts.max()) if len(new_counts) > 0 else 0.0,
            float(old_density.max()),
            float(new_density.max()),
        )
        all_peaks.append(peak)

        # Unvalidated — solid line
        ax.plot(
            old_x,
            old_density,
            color=color,
            linestyle="-",
            linewidth=1.8,
            zorder=3,
        )
        ax.vlines(
            x=old_mean,
            ymin=0,
            ymax=float(old_density.max()),
            color=color,
            linestyle="-",
            linewidth=1.2,
            alpha=0.85,
            zorder=4,
        )

        # Validated — dashed line
        ax.plot(
            new_x,
            new_density,
            color=color,
            linestyle="--",
            linewidth=1.8,
            zorder=3,
        )
        ax.vlines(
            x=new_mean,
            ymin=0,
            ymax=float(new_density.max()),
            color=color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
            zorder=4,
        )

        peak_x = (old_mean + new_mean) / 2
        peak_y = max(float(old_density.max()), float(new_density.max()))
        model_labels.append((peak_x, peak_y, SELECTED_MODEL_LABELS.get(row.model, row.model), color))

    ax.set_xlim(0, 1)
    max_peak = max(all_peaks) if all_peaks else 15.0
    ax.set_ylim(bottom=0, top=max_peak * 1.06)

    # Alternate labels between two modest levels and keep edge labels inside
    # the axes.  This avoids collisions after the figure is narrowed.
    for tier, (peak_x, peak_y, model, color) in enumerate(sorted(model_labels)):
        label_y = peak_y + max_peak * (0.02 + 0.06 * (tier % 2))
        label_x = peak_x
        if "\n" in model:
            if peak_x < 0.2:
                label_x = 0.02
                horizontal_alignment = "left"
            else:
                horizontal_alignment = "center"
        elif len(model) > 30 or peak_x < 0.35:
            label_x = max(0.02, peak_x - 0.22)
            horizontal_alignment = "left"
        elif peak_x < 0.12:
            horizontal_alignment = "left"
        elif peak_x > 0.88:
            horizontal_alignment = "right"
        else:
            horizontal_alignment = "center"
        ax.text(
            label_x,
            label_y,
            model,
            ha=horizontal_alignment,
            va="bottom",
            fontsize=15.5,
            fontweight="bold",
            color=color,
            zorder=5,
        )

    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.tick_params(axis="x", labelsize=21)
    ax.set_xlabel("Exact-answer accuracy", fontsize=25, labelpad=6)
    ax.set_ylabel("Density", fontsize=25, labelpad=6)
    ax.set_yticks([])
    ax.grid(False)

    legend_elements = [
        Line2D([0], [0], color="#555555", lw=2.8, linestyle="-", label=legend_labels[0]),
        Line2D([0], [0], color="#555555", lw=2.8, linestyle="--", label=legend_labels[1]),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        frameon=False,
        fontsize=18,
        ncol=2,
    )
    fig.tight_layout(pad=0.18)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.03, facecolor="white")
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
        default=8,
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
    eng_metric_selected = False
    eng_metric_full = False
    english = samples[samples["language"] == "eng"].copy()
    english_metric = samples[samples["language"] == "eng_metric"].copy()
    if not english.empty and not english_metric.empty:
        english_metric["language"] = "eng"
        metric_rows = collect_correction_comparison_rows(
            english,
            english_metric,
            "eng",
            args.correction_samples,
            args.correction_seed,
        )
        if metric_rows:
            metric_full_out = args.out_dir / "eng_vs_eng_metric_full.png"
            plot_correction_comparison(
                metric_rows,
                "eng",
                metric_full_out,
                legend_labels=("English", "English metric"),
                title="English vs English metric comparison",
            )
            eng_metric_full = True
            metric_out = args.out_dir / "eng_vs_eng_metric_selected.png"
            plot_correction_comparison_selected(
                metric_rows,
                "eng",
                metric_out,
                target_models=[
                    "gemma-3-12b-it",
                    "Apertus-8B-Instruct-2509",
                    "granite-3.2-2b-instruct (reasoning off)",
                ],
                legend_labels=("English", "English metric"),
            )
            eng_metric_selected = True
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

                out_selected = args.out_dir / "correction_comparison" / f"{path_slug(language)}_selected.png"
                plot_correction_comparison_selected(rows, language, out_selected)
                correction_outputs.append(out_selected)

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
    if eng_metric_selected:
        print(f"Saved {args.out_dir / 'eng_vs_eng_metric_selected.png'}")
    if eng_metric_full:
        print(f"Saved {args.out_dir / 'eng_vs_eng_metric_full.png'}")

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
        print(
            "Skipped reasoning_delta_heatmap.png: paired synthetic English/non-English reasoning results with model sizes are required."
        )

    for out in correction_outputs:
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
