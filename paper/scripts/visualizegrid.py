# /// script
# dependencies = ["inspect-ai", "matplotlib", "numpy", "pandas"]
# ///
"""Visualize multilingual GSM evaluation logs across models and splits.

The script reads Inspect ``.eval`` logs and writes:

* ``run_summary.csv``: tidy accuracy table for downstream analysis.
* ``accuracy_heatmaps.png``: accuracy on original and synthetic benchmark splits.
* ``original_vs_synthetic.png``: paired split performance for each model/language.
* ``family_scaling.png``: within-family accuracy as a function of parameter count.
* ``english_normalized_transfer.png``: language accuracy relative to English.
* ``transfer_robustness.png``: transfer penalty and cross-language dispersion by size.
* ``split_degradation_heatmaps.png``: absolute and relative original-to-synthetic drop.

Only successful logs are included by default. Repeated/resumed logs with the same
evaluation id are deduplicated, preferring a successful and then newer log.

Example:
    uv run paper/scripts/visualizegrid.py --workers 8
"""

from __future__ import annotations

import argparse
import math
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO_ROOT / "hf_dataset" / "logs"
DEFAULT_OUT_DIR = REPO_ROOT / "paper" / "artifacts" / "figures" / "model_grid"


@dataclass(frozen=True)
class ModelInfo:
    family: str
    params_b: float | None
    vocab_size: int | None = None
    training_language: str = ""
    pretrain_tokens_t: float | None = None


MODEL_CATALOG = {
    "qwen2.5-0.5b-instruct": ModelInfo("Qwen2.5", 0.5, 151_936, "Chinese-English; multilingual", 18),
    "qwen2.5-1.5b-instruct": ModelInfo("Qwen2.5", 1.5, 151_936, "Chinese-English; multilingual", 18),
    "qwen2.5-3b-instruct": ModelInfo("Qwen2.5", 3, 151_936, "Chinese-English; multilingual", 18),
    "qwen2.5-7b-instruct": ModelInfo("Qwen2.5", 7, 151_936, "Chinese-English; multilingual", 18),
    "qwen2.5-14b-instruct": ModelInfo("Qwen2.5", 14, 151_936, "Chinese-English; multilingual", 18),
    "qwen2.5-32b-instruct": ModelInfo("Qwen2.5", 32, 151_936, "Chinese-English; multilingual", 18),
    "llama-3.2-1b-instruct": ModelInfo("Llama 3", 1, 128_256, "English-centric", 9),
    "llama-3.2-3b-instruct": ModelInfo("Llama 3", 3, 128_256, "English-centric", 9),
    "llama-3.1-8b-instruct": ModelInfo("Llama 3", 8, 128_256, "English-centric", 15),
    "llama-3.2-8b-instruct": ModelInfo("Llama 3", 8, 128_256, "English-centric", 15),
    "olmo-2-0425-1b-instruct": ModelInfo("OLMo 2", 1),
    "olmo-2-1124-7b-instruct": ModelInfo("OLMo 2", 7),
    "olmo-2-1124-13b-instruct": ModelInfo("OLMo 2", 13),
    "olmo-2-0325-32b-instruct": ModelInfo("OLMo 2", 32),
    "gemma-3-1b-it": ModelInfo("Gemma 3", 1, 262_144, "English-oriented", 2),
    "gemma-3-4b-it": ModelInfo("Gemma 3", 4, 262_144, "Multilingual; 140+ languages", 4),
    "gemma-3-12b-it": ModelInfo("Gemma 3", 12, 262_144, "Multilingual; 140+ languages", 12),
    "gemma-3-27b-it": ModelInfo("Gemma 3", 27, 262_144, "Multilingual; 140+ languages", 14),
    "apertus-8b-instruct-2509": ModelInfo("Apertus", 8),
}

LANGUAGE_LABELS = {
    "dan": "Danish",
    "deu": "German",
    "eng": "English",
    "fra": "French",
    "isl": "Icelandic",
    "ita": "Italian",
    "nob": "Norwegian Bokmal",
    "por": "Portuguese",
    "rus": "Russian",
    "spa": "Spanish",
    "ukr": "Ukrainian",
    "zho": "Chinese",
}

LANGUAGE_SPEAKERS = {
    "zho": 940_000_000,
    "eng": 380_000_000,
    "rus": 150_000_000,
    "deu": 100_000_000,
    "dan": 6_000_000,
    "nob": 5_000_000,
    "isl": 370_000,
}

HUMAN_VERIFIED_LANGUAGES = {"eng", "dan", "rus", "zho"}

FAMILY_COLORS = {
    "Qwen2.5": "#7B2CBF",
    "Llama 3": "#E76F51",
    "OLMo 2": "#D62828",
    "Gemma 3": "#2A9D8F",
    "Apertus": "#6A994E",
    "OpenAI": "#457B9D",
}

FAMILY_ORDER = {
    "Qwen2.5": 0,
    "Llama 3": 1,
    "Gemma 3": 2,
    "OLMo 2": 3,
    "Apertus": 4,
    "OpenAI": 5,
}

FAMILY_MARKERS = {
    "Qwen2.5": "o",
    "Llama 3": "s",
    "Gemma 3": "^",
    "OLMo 2": "D",
    "Apertus": "P",
    "OpenAI": "X",
}
OUTLIER_LABEL_COUNT = 5
EXCLUDED_SPLIT_PAIR = ("OLMo-2-1124-7B-Instruct", "dan")
HARD_CODED_SPLIT_PAIR_LABELS = {
    ("Qwen2.5-1.5B-Instruct", "nob"): "Qwen2.5 1.5B (Norwegian)",
    ("Llama-3.2-3B-Instruct", "dan"): "Llama 3.2 3B (Danish)",
}

SPLIT_LABELS = {
    "original": "Original benchmark questions",
    "synthetic": "Synthetic numerical variants",
}

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.dpi": 160,
        "font.family": "sans-serif",
    }
)


def model_name(raw_model: str) -> str:
    """Drop provider/organization prefixes while preserving the model id."""
    return raw_model.rstrip("/").split("/")[-1]


def infer_model_info(raw_model: str) -> ModelInfo:
    name = model_name(raw_model)
    catalog_info = MODEL_CATALOG.get(name.lower())
    if catalog_info:
        return catalog_info

    lower = name.lower()
    if "qwen" in lower:
        family = "Qwen2.5" if "2.5" in lower else "Qwen"
    elif "llama" in lower:
        family = "Llama 3" if re.search(r"llama[-_. ]?3", lower) else "Llama"
    elif "gemma" in lower:
        family = "Gemma 3" if re.search(r"gemma[-_. ]?3", lower) else "Gemma"
    elif "olmo" in lower:
        family = "OLMo 2" if re.search(r"olmo[-_. ]?2", lower) else "OLMo"
    elif "apertus" in lower:
        family = "Apertus"
    elif raw_model.lower().startswith("openai/") or lower.startswith(("gpt-", "o1", "o3", "o4")):
        family = "OpenAI"
    else:
        family = name.split("-", 1)[0].split("_", 1)[0].split(".", 1)[0]

    size_match = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*b(?:\b|[-_])", lower)
    params_b = float(size_match.group(1)) if size_match else None
    return ModelInfo(family, params_b)


def parse_task(task: str) -> tuple[str, str] | None:
    matches = re.findall(r"(original|synthetic)[_-]([a-z]{3})", task.lower())
    return matches[-1] if matches else None


def score_to_float(score: Any) -> float | None:
    value = getattr(score, "value", score)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in {"C", "CORRECT", "TRUE"}:
            return 1.0
        if normalized in {"I", "INCORRECT", "FALSE"}:
            return 0.0
        try:
            return float(normalized)
        except ValueError:
            return None
    return None


def sample_score(sample: Any, scorer: str | None) -> float | None:
    scores = sample.scores or {}
    if scorer:
        return score_to_float(scores.get(scorer)) if scorer in scores else None

    for preferred in ("math", "pattern"):
        if preferred in scores:
            return score_to_float(scores[preferred])

    for score in scores.values():
        value = score_to_float(score)
        if value is not None:
            return value

    return None


def discover_logs(log_dirs: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for log_dir in log_dirs:
        paths.update(path for path in log_dir.rglob("*.eval") if path.stat().st_size >= 1_000)
    return sorted(paths)


def _read_log_header(path: Path) -> tuple[Path, Any | None, str | None]:
    try:
        return path, read_eval_log(str(path), header_only=True), None
    except Exception as exc:
        return path, None, str(exc)


def select_logs(paths: list[Path], include_incomplete: bool, workers: int) -> list[tuple[Path, Any]]:
    """Deduplicate log snapshots, preferring successful and newer entries."""
    selected: dict[tuple[str, str], tuple[Path, Any]] = {}

    if not paths:
        return []

    if workers <= 1:
        results = [_read_log_header(path) for path in paths]
    else:
        max_workers = min(workers, len(paths))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_read_log_header, path) for path in paths]
            results = [future.result() for future in as_completed(futures)]

    for path, log, error in results:
        if error:
            print(f"Skipping unreadable log {path.name}: {error}")
            continue

        status = str(log.status)
        if not include_incomplete and status != "success":
            continue

        key = (log.eval.eval_id, log.eval.task)
        rank = (status == "success", path.stat().st_mtime_ns)

        previous = selected.get(key)
        if previous is None:
            selected[key] = (path, log)
            continue

        previous_rank = (
            str(previous[1].status) == "success",
            previous[0].stat().st_mtime_ns,
        )
        if rank > previous_rank:
            selected[key] = (path, log)

    return sorted(selected.values(), key=lambda item: item[0].name)


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
    label = f"{model_name(log.eval.model)} / {task_language} / {split}"

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
                "model": model_name(log.eval.model),
                "family": info.family,
                "params_b": info.params_b,
                "vocab_size": info.vocab_size,
                "training_language": info.training_language,
                "pretrain_tokens_t": info.pretrain_tokens_t,
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

    problem_keys = [
        "model_raw",
        "model",
        "family",
        "params_b",
        "vocab_size",
        "training_language",
        "pretrain_tokens_t",
        "language",
        "split",
        "sample_id",
        "source_id",
    ]

    grouped = samples.groupby(problem_keys, dropna=False, as_index=False)["correct"].agg(
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

    problem_keys = [
        "model_raw",
        "model",
        "family",
        "params_b",
        "vocab_size",
        "training_language",
        "pretrain_tokens_t",
        "language",
        "split",
        "sample_id",
        "source_id",
    ]

    samples = combined.groupby(problem_keys, dropna=False, as_index=False).agg(
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
        "training_language",
        "pretrain_tokens_t",
        "language",
        "split",
    ]
    return (
        samples.groupby(group_cols, dropna=False)["correct"]
        .agg(accuracy="mean", n_problems="size", stderr="sem")
        .reset_index()
    )


def model_order(summary: pd.DataFrame) -> list[str]:
    models = summary[["model", "family", "params_b"]].drop_duplicates().copy()
    models["family_order"] = models["family"].map(FAMILY_ORDER).fillna(99)
    models["size_order"] = models["params_b"].fillna(math.inf)
    return models.sort_values(["family_order", "size_order", "model"])["model"].tolist()


def sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    ordered = summary.copy()
    ordered["family_order"] = ordered["family"].map(FAMILY_ORDER).fillna(99)
    ordered["size_order"] = ordered["params_b"].fillna(math.inf)
    return ordered.sort_values(["family_order", "size_order", "model", "language", "split"]).drop(
        columns=["family_order", "size_order"]
    )


def ordered_families(families: pd.Series | set[str] | list[str]) -> list[str]:
    unique_families = set(families)
    known = [family for family in FAMILY_ORDER if family in unique_families]
    extra = sorted(unique_families - set(FAMILY_ORDER))
    return known + extra


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
    languages = sorted(
        summary["language"].unique(),
        key=lambda language: (-LANGUAGE_SPEAKERS.get(language, -1), language),
    )

    available_splits = [split for split in ("original", "synthetic") if split in set(summary["split"])]

    matrices: dict[str, pd.DataFrame] = {}
    for split in available_splits:
        split_rows = summary[summary["split"] == split]
        matrices[split] = split_rows.pivot_table(
            index="model",
            columns="language",
            values="accuracy",
        ).reindex(index=order, columns=languages)

    panels: list[tuple[str, pd.DataFrame, str, float, float, bool]] = [
        (SPLIT_LABELS[split], matrices[split], "viridis", 0, 1, False) for split in available_splits
    ]

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
        annotated_heatmap(ax, matrix, title, cmap, vmin, vmax, signed)
        for ax, (title, matrix, cmap, vmin, vmax, signed) in zip(axes, panels, strict=True)
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
    sized = sized[
        ~((sized["model"] == EXCLUDED_SPLIT_PAIR[0]) & (sized["language"] == EXCLUDED_SPLIT_PAIR[1]))
    ]
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

    outliers = (
        sized.dropna(subset=["params_b"])
        .assign(gap=lambda rows: (rows["synthetic"] - rows["original"]).abs())
        .sort_values("gap", ascending=False)
        .drop_duplicates("model")
        .head(OUTLIER_LABEL_COUNT)
    )
    annotations = []
    for row in sized.itertuples():
        label = HARD_CODED_SPLIT_PAIR_LABELS.get((row.model, row.language))
        if label:
            annotations.append(
                ax.annotate(label, (row.original, row.synthetic), xytext=(5, 0), textcoords="offset points", fontsize=7)
            )

    for row in outliers.itertuples():
        family = row.family if row.family != "Other" else row.model.split("-")[0]
        label = f"{family} {row.params_b:g}B ({LANGUAGE_LABELS.get(row.language, row.language)})"
        annotations.append(
            ax.annotate(label, (row.original, row.synthetic), xytext=(5, 0), textcoords="offset points", fontsize=7)
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
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_bounds = []
    for annotation in annotations:
        if any(annotation.get_window_extent(renderer).overlaps(bounds) for bounds in label_bounds):
            annotation.remove()
        else:
            label_bounds.append(annotation.get_window_extent(renderer))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return True


def plot_english_normalized_transfer(summary: pd.DataFrame, out: Path) -> bool:
    """Plot each language's accuracy difference from the same model's English score."""
    order = model_order(summary)
    splits = [split for split in ("original", "synthetic") if split in set(summary["split"])]
    languages = sorted(
        (language for language in summary["language"].unique() if language != "eng"),
        key=lambda language: (-LANGUAGE_SPEAKERS.get(language, -1), language),
    )

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
    languages = sorted(
        summary["language"].unique(),
        key=lambda language: (-LANGUAGE_SPEAKERS.get(language, -1), language),
    )
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
    languages = sorted(scaled["language"].unique(), key=lambda x: (x != "eng", x))
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
        help="Processes used for full log loading. Use 1 to disable multiprocessing.",
    )
    parser.add_argument(
        "--header-workers",
        type=int,
        default=32,
        help="Threads used for the header-only deduplication pass.",
    )

    args = parser.parse_args()

    paths = discover_logs(args.log_dir)
    selected = select_logs(paths, args.include_incomplete, workers=args.header_workers)

    print(f"Discovered {len(paths)} logs; selected {len(selected)} after status filtering and deduplication.")
    print(f"Loading selected logs with {args.workers} worker(s).")

    samples = load_samples(selected, args.scorer, workers=args.workers)

    if samples.empty:
        raise SystemExit("No scored samples found in the selected logs.")

    summary = summarize(samples)

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
    made_robustness = plot_transfer_robustness(
        summary,
        args.out_dir / "transfer_robustness.png",
    )
    made_degradation = plot_split_degradation(
        summary,
        args.out_dir / "split_degradation_heatmaps.png",
    )

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

    if made_robustness:
        print(f"Saved {args.out_dir / 'transfer_robustness.png'}")
    else:
        print("Skipped transfer_robustness.png: paired transfer results with model sizes are required.")

    if made_degradation:
        print(f"Saved {args.out_dir / 'split_degradation_heatmaps.png'}")
    else:
        print("Skipped split_degradation_heatmaps.png: paired original/synthetic results are required.")


if __name__ == "__main__":
    main()
