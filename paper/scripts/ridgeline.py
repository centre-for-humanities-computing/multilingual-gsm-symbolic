# /// script
# dependencies = ["inspect-ai", "matplotlib", "numpy", "pandas", "scipy"]
# ///
"""Plot original accuracy against sampled synthetic-set distributions by language.

The synthetic distribution follows the benchmark sampling scheme used in
``visualize_results.py``: each sampled set chooses one numerical variant from
every source template, then averages correctness across the selected problems.
"""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from plot_config import (
    HUMAN_VERIFIED_LANGUAGES,
    LANGUAGE_LABELS,
    LANGUAGE_SPEAKERS,
    language_order,
    model_family,
    model_name,
    model_sort_key,
)
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO_ROOT / "hf_dataset" / "logs"
DEFAULT_OUT_DIR = REPO_ROOT / "paper" / "artifacts" / "language_ridgeline"

SYNTHETIC_COLOR = "#173B75"
SYNTHETIC_FILL = "#AFC0DD"
ORIGINAL_COLOR = "#C61B3C"
GRID_COLOR = "#D8DEE8"
TEXT_COLOR = "#162033"

plt.rcParams.update(
    {
        "axes.spines.left": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.dpi": 160,
        "font.family": "sans-serif",
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": "#4A5568",
        "ytick.color": TEXT_COLOR,
    }
)


@dataclass(frozen=True)
class PlotStats:
    language: str
    original_accuracy: float
    synthetic_mean: float
    synthetic_std: float
    shift_pp: float
    n_models: int
    n_templates: int
    n_sampled_sets: int
def path_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9.]+", "-", value.lower()).strip("-")
    return slug or "unknown"


def format_speaker_count(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:g}M"
    if count >= 1_000:
        return f"{count / 1_000:g}K"
    return str(count)


def language_title(language: str) -> str:
    label = LANGUAGE_LABELS.get(language, language)
    speakers = LANGUAGE_SPEAKERS.get(language)
    speaker_label = f" ({format_speaker_count(speakers)} native speakers)" if speakers else ""
    verified = " ★" if language in HUMAN_VERIFIED_LANGUAGES else ""
    return f"{label}{speaker_label}{verified}"


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


def _read_log_header(path: Path) -> tuple[Path, Any | None, str | None]:
    try:
        return path, read_eval_log(str(path), header_only=True), None
    except Exception as exc:
        return path, None, str(exc)


def discover_selected_logs(
    log_dir: Path,
    requested_models: list[str] | None,
    workers: int,
) -> list[Path]:
    """Select all successful logs and deduplicate snapshots in parallel."""
    paths = sorted(path for path in log_dir.rglob("*.eval") if path.stat().st_size >= 1_000)
    if not paths:
        return []

    requested = {model.lower() for model in requested_models} if requested_models else None
    selected: dict[tuple[str, str], tuple[Path, Any]] = {}

    if workers <= 1:
        results = [_read_log_header(path) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as pool:
            results = list(pool.map(_read_log_header, paths))

    for path, header, error in results:
        if error:
            print(f"Skipping unreadable log {path.name}: {error}")
            continue
        if str(header.status) != "success" or parse_task(header.eval.task) is None:
            continue
        if requested and model_name(header.eval.model).lower() not in requested:
            continue

        key = (header.eval.eval_id, header.eval.task)
        previous = selected.get(key)
        if previous is None or path.stat().st_mtime_ns > previous[0].stat().st_mtime_ns:
            selected[key] = (path, header)

    return sorted(path for path, _header in selected.values())


def _load_one_log(
    path: Path,
    scorer: str | None,
) -> tuple[str, pd.DataFrame | None, str | None]:
    """Read and partially aggregate one log in a worker process."""
    try:
        log = read_eval_log(str(path))
    except Exception as exc:
        return path.name, None, f"Skipping unreadable log {path.name}: {exc}"

    parsed = parse_task(log.eval.task)
    if parsed is None:
        return path.name, None, f"Skipping unrecognized task {log.eval.task!r}"
    if not log.samples:
        return path.name, pd.DataFrame(), None

    split, task_language = parsed
    model = model_name(log.eval.model)
    label = f"{model} / {task_language} / {split}"
    rows: list[dict[str, Any]] = []

    for sample in log.samples:
        correct = sample_score(sample, scorer)
        if correct is None:
            continue
        metadata = sample.metadata or {}
        rows.append(
            {
                "model": model,
                "language": metadata.get("language", task_language),
                "split": split,
                "sample_id": sample.id,
                "source_id": metadata.get("source_id", sample.id),
                "correct": correct,
            }
        )

    if not rows:
        return label, pd.DataFrame(), None

    samples = pd.DataFrame(rows)
    grouped = samples.groupby(
        ["model", "language", "split", "sample_id", "source_id"],
        dropna=False,
        as_index=False,
    )["correct"].agg(correct_sum="sum", correct_count="size")
    return label, grouped, None


def load_problem_scores(
    paths: list[Path],
    scorer: str | None,
    workers: int,
) -> pd.DataFrame:
    """Load logs in parallel and combine repeated runs by model/problem."""
    frames: list[pd.DataFrame] = []

    if workers <= 1:
        results = [_load_one_log(path, scorer) for path in paths]
        for index, (label, frame, warning) in enumerate(results, start=1):
            if warning:
                print(warning)
            else:
                print(f"[{index}/{len(paths)}] {label}")
            if frame is not None and not frame.empty:
                frames.append(frame)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(paths))) as pool:
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
    keys = ["model", "language", "split", "sample_id", "source_id"]
    problems = combined.groupby(keys, dropna=False, as_index=False).agg(
        correct_sum=("correct_sum", "sum"), correct_count=("correct_count", "sum")
    )
    problems["correct"] = problems["correct_sum"] / problems["correct_count"]
    return problems.drop(columns=["correct_sum", "correct_count"])


def sample_synthetic_sets(
    synthetic: pd.DataFrame,
    n_sets: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Sample one variant per source template and return set-level accuracies."""
    variants = [
        group["correct"].to_numpy(dtype=float)
        for _source_id, group in synthetic.groupby("source_id", sort=True, dropna=False)
    ]
    if not variants:
        raise ValueError("Synthetic data contains no source templates.")

    set_totals = np.zeros(n_sets, dtype=float)
    for values in variants:
        set_totals += rng.choice(values, size=n_sets, replace=True)
    return set_totals / len(variants), len(variants)


def normal_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit and return a normal probability density for sampled accuracies."""
    mean = float(values.mean())
    std = max(float(values.std(ddof=1)), 0.005)
    lower = max(0.0, mean - 4 * std)
    upper = min(1.0, mean + 4 * std)
    x = np.linspace(lower, upper, 500)
    return x, norm.pdf(x, loc=mean, scale=std), mean, std


def collect_plot_data(
    problems: pd.DataFrame,
    languages: list[str],
    n_sets: int,
    seed: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], list[PlotStats]]:
    distributions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    stats: list[PlotStats] = []
    rng = np.random.default_rng(seed)

    for language in languages:
        rows = problems[problems["language"] == language]
        split_models = {split: set(split_rows["model"].unique()) for split, split_rows in rows.groupby("split")}
        paired_models = sorted(split_models.get("original", set()) & split_models.get("synthetic", set()))
        if not paired_models:
            print(f"Skipping {language}: both original and synthetic results are required.")
            continue

        original_by_model: list[float] = []
        synthetic_by_model: list[np.ndarray] = []
        n_templates = 0
        for model in paired_models:
            model_rows = rows[rows["model"] == model]
            original = model_rows[model_rows["split"] == "original"]
            synthetic = model_rows[model_rows["split"] == "synthetic"]
            original_by_model.append(float(original["correct"].mean()))
            model_set_means, model_templates = sample_synthetic_sets(
                synthetic,
                n_sets,
                rng,
            )
            synthetic_by_model.append(model_set_means)
            n_templates += model_templates

        set_means = np.mean(np.vstack(synthetic_by_model), axis=0)
        x, density, synthetic_mean, synthetic_std = normal_curve(set_means)
        original_accuracy = float(np.mean(original_by_model))
        distributions[language] = (set_means, x, density)
        stats.append(
            PlotStats(
                language=language,
                original_accuracy=original_accuracy,
                synthetic_mean=synthetic_mean,
                synthetic_std=synthetic_std,
                shift_pp=(synthetic_mean - original_accuracy) * 100,
                n_models=len(paired_models),
                n_templates=n_templates,
                n_sampled_sets=n_sets,
            )
        )

    return distributions, stats


def plot_distributions(
    distributions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    stats: list[PlotStats],
    scope_label: str,
    out_png: Path,
) -> None:
    if not stats:
        raise ValueError("No languages have paired original and synthetic results.")

    fig, axes = plt.subplots(
        len(stats),
        1,
        figsize=(10.5, 2.05 * len(stats) + 1.7),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    for ax, row in zip(axes, stats, strict=True):
        set_means, x, density = distributions[row.language]
        peak_density = float(density.max())
        histogram_bins = min(18, max(8, int(np.sqrt(len(set_means)) / 2)))

        ax.hist(
            set_means,
            bins=histogram_bins,
            density=True,
            color=SYNTHETIC_FILL,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.72,
            zorder=1,
        )
        ax.fill_between(
            x,
            0,
            density,
            color=SYNTHETIC_FILL,
            alpha=0.2,
            zorder=2,
        )
        ax.plot(x, density, color=SYNTHETIC_COLOR, linewidth=2.2, zorder=3)
        ax.scatter(
            row.synthetic_mean,
            peak_density,
            color=SYNTHETIC_COLOR,
            s=34,
            zorder=5,
        )

        ax.axvline(
            row.original_accuracy,
            color=ORIGINAL_COLOR,
            linewidth=2.4,
            zorder=4,
        )

        arrow_y = peak_density * 1.08
        gap = abs(row.synthetic_mean - row.original_accuracy)
        if gap < 0.025:
            cap_height = peak_density * 0.035
            ax.hlines(
                arrow_y,
                row.synthetic_mean,
                row.original_accuracy,
                color="black",
                linewidth=1.2,
                zorder=6,
            )
            ax.vlines(
                [row.synthetic_mean, row.original_accuracy],
                arrow_y - cap_height,
                arrow_y + cap_height,
                color="black",
                linewidth=1.2,
                zorder=6,
            )
        else:
            ax.annotate(
                "",
                xy=(row.synthetic_mean, arrow_y),
                xytext=(row.original_accuracy, arrow_y),
                arrowprops={
                    "arrowstyle": "<->",
                    "color": "black",
                    "linewidth": 1.2,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
                zorder=6,
            )
        midpoint = (row.original_accuracy + row.synthetic_mean) / 2
        ax.text(
            midpoint,
            arrow_y + peak_density * 0.045,
            f"performance gap ({row.shift_pp:+.1f} pp)",
            color="black",
            fontsize=10,
            ha="center",
            va="bottom",
            zorder=7,
        )

        ax.set_ylabel(
            language_title(row.language),
            rotation=0,
            ha="right",
            va="center",
            labelpad=70,
            fontsize=13,
            fontweight="semibold",
        )
        ax.set_ylim(0, peak_density * 1.32)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.65)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=9)

    axes[-1].set_xlim(0, 1)
    axes[-1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[-1].tick_params(axis="x", labelsize=10)
    axes[-1].set_xlabel("Exact-answer accuracy", fontsize=13, labelpad=10)
    fig.supylabel("Density", fontsize=13, x=0.03)
    fig.text(
        0.08,
        0.985,
        f"{scope_label} | one random variant per template in each sampled set",
        fontsize=11,
        color="#596579",
        ha="left",
        va="top",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=ORIGINAL_COLOR,
            marker="|",
            markersize=15,
            markeredgewidth=2.4,
            linewidth=0,
            label="Original benchmark accuracy",
        ),
        Line2D(
            [0],
            [0],
            color=SYNTHETIC_COLOR,
            linewidth=2.2,
            label="Sampled synthetic sets",
        ),
    ]
    axes[0].legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1, 1.42),
        frameon=False,
        ncol=3,
        fontsize=10,
        handlelength=1.8,
        columnspacing=1.7,
    )

    fig.tight_layout(rect=(0.045, 0.03, 1, 0.955))
    fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary(stats: list[PlotStats], out_csv: Path) -> None:
    pd.DataFrame(
        [
            {
                "language": row.language,
                "language_label": LANGUAGE_LABELS.get(row.language, row.language),
                "original_accuracy": row.original_accuracy,
                "synthetic_sampled_mean": row.synthetic_mean,
                "synthetic_sampled_std": row.synthetic_std,
                "synthetic_minus_original_pp": row.shift_pp,
                "paired_models": row.n_models,
                "source_templates": row.n_templates,
                "sampled_sets": row.n_sampled_sets,
            }
            for row in stats
        ]
    ).to_csv(out_csv, index=False)


def main() -> None:
    cpu_count = os.cpu_count() or 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory searched recursively for Inspect .eval logs.",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        help="Optional model identifiers without provider prefixes; defaults to all models.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        help="Optional three-letter language codes in display order; defaults to all discovered languages.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2_000,
        help="Number of synthetic benchmark sets to sample.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scorer",
        help="Inspect scorer name; defaults to math, pattern, then the first numeric score.",
    )
    parser.add_argument(
        "--output-name",
        help="Optional filename prefix; model identifiers are always appended.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(12, cpu_count),
        help="Workers used for header scans and full log loading; use 1 to disable parallelism.",
    )
    args = parser.parse_args()

    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    paths = discover_selected_logs(args.log_dir, args.model, args.workers)
    if not paths:
        requested = f" for models {args.model!r}" if args.model else ""
        raise SystemExit(f"No successful logs found{requested} in {args.log_dir}.")

    print(f"Selected {len(paths)} successful logs after deduplication.")
    print(f"Loading logs with {args.workers} worker process(es).")
    problems = load_problem_scores(paths, args.scorer, args.workers)
    if problems.empty:
        raise SystemExit("No scored samples found in the selected logs.")

    models = sorted(problems["model"].unique(), key=model_sort_key)
    all_languages = language_order(set(problems["language"].unique()), args.languages)
    print(f"Models ({len(models)}): {', '.join(models)}")
    print(f"Languages ({len(all_languages)}): {', '.join(all_languages)}")

    output_root = DEFAULT_OUT_DIR
    for model_index, model in enumerate(models):
        model_problems = problems[problems["model"] == model]
        model_languages = language_order(
            set(model_problems["language"].unique()),
            args.languages,
        )
        curves, stats = collect_plot_data(
            model_problems,
            model_languages,
            args.samples,
            args.seed + model_index,
        )
        if not stats:
            print(f"Skipping {model}: no languages have paired original and synthetic results.")
            continue

        family = model_family(model)
        out_dir = output_root / path_slug(family)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_slug = path_slug(model)
        base_name = f"{path_slug(args.output_name)}-{model_slug}" if args.output_name else model_slug
        out_png = out_dir / f"{base_name}.png"
        out_csv = out_dir / f"{base_name}.csv"

        plot_distributions(curves, stats, model, out_png)
        write_summary(stats, out_csv)

        print(f"Saved {out_png}")
        print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
