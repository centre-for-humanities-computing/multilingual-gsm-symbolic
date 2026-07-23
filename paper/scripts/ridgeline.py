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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eval_log_utils import (
    discover_logs,
    map_log_loader,
    normal_curve,
    parse_task,
    sample_score,
    sample_synthetic_sets,
    select_logs,
)
from inspect_ai.log import read_eval_log
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from plot_config import (
    HUMAN_VERIFIED_LANGUAGES,
    LANGUAGE_LABELS,
    LANGUAGE_SPEAKERS,
    format_speaker_count,
    language_order,
    model_family,
    model_name,
    model_sort_key,
    path_slug,
)

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


def language_title(language: str) -> str:
    label = LANGUAGE_LABELS.get(language, language)
    speakers = LANGUAGE_SPEAKERS.get(language)
    speaker_label = f"\n({format_speaker_count(speakers)} native)" if speakers else ""
    verified = " ★" if language in HUMAN_VERIFIED_LANGUAGES else ""
    return f"{label}{verified}{speaker_label}"


def discover_selected_logs(
    log_dir: Path,
    requested_models: list[str] | None,
    workers: int,
) -> list[Path]:
    """Select all successful logs with a recognized task and deduplicate snapshots."""
    paths = discover_logs([log_dir])
    if not paths:
        return []

    requested = {model.lower() for model in requested_models} if requested_models else None
    selected_pairs = select_logs(paths, include_incomplete=False, workers=workers)

    result: list[Path] = []
    for path, header in selected_pairs:
        if parse_task(header.eval.task) is None:
            continue
        if requested and model_name(header.eval.model).lower() not in requested:
            continue
        result.append(path)
    return sorted(result)


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

    del log  # free the large decompressed log before returning
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
    import gc
    frames: list[pd.DataFrame] = []
    _CONCAT_CHUNK = 4  # flush accumulated frames frequently to cap memory footprint

    for index, (label, frame, warning) in enumerate(map_log_loader(_load_one_log, paths, scorer, workers), start=1):
        if warning:
            print(warning)
        else:
            print(f"[{index}/{len(paths)}] {label}")
        if frame is not None and not frame.empty:
            frames.append(frame)
        if len(frames) >= _CONCAT_CHUNK:
            concatenated = pd.concat(frames, ignore_index=True)
            keys = ["model", "language", "split", "sample_id", "source_id"]
            grouped = concatenated.groupby(keys, dropna=False, as_index=False).agg(
                correct_sum=("correct_sum", "sum"), correct_count=("correct_count", "sum")
            )
            frames = [grouped]
            gc.collect()

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    keys = ["model", "language", "split", "sample_id", "source_id"]
    problems = combined.groupby(keys, dropna=False, as_index=False).agg(
        correct_sum=("correct_sum", "sum"), correct_count=("correct_count", "sum")
    )
    del combined
    gc.collect()

    problems["correct"] = problems["correct_sum"] / problems["correct_count"]
    problems.drop(columns=["correct_sum", "correct_count"], inplace=True)
    gc.collect()
    return problems


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

    n_rows = len(stats)
    if n_rows <= 4:
        fig_height = 2.2 * n_rows + 1.0
        left_margin = 0.30
        right_margin = 0.97
        top_margin = 0.85
        bottom_margin = 0.12
        hspace = 0.45
        legend_y = 1.25
        supylabel_x = 0.015
        title_y = 0.985
    else:
        fig_height = 1.8 * n_rows + 1.6
        left_margin = 0.28
        right_margin = 0.97
        top_margin = 0.93
        bottom_margin = 0.07
        hspace = 0.35
        legend_y = 1.45
        supylabel_x = 0.015
        title_y = 0.988

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(8.5, fig_height),
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
            alpha=0.22,
            zorder=2,
        )
        ax.plot(x, density, color=SYNTHETIC_COLOR, linewidth=2.4, zorder=3)
        ax.scatter(
            row.synthetic_mean,
            peak_density,
            color=SYNTHETIC_COLOR,
            s=48,
            zorder=5,
        )

        ax.axvline(
            row.original_accuracy,
            color=ORIGINAL_COLOR,
            linewidth=2.6,
            zorder=4,
        )

        arrow_y = peak_density * 1.08
        gap = abs(row.synthetic_mean - row.original_accuracy)
        if gap < 0.025:
            cap_height = peak_density * 0.04
            ax.hlines(
                arrow_y,
                row.synthetic_mean,
                row.original_accuracy,
                color="black",
                linewidth=1.4,
                zorder=6,
            )
            ax.vlines(
                [row.synthetic_mean, row.original_accuracy],
                arrow_y - cap_height,
                arrow_y + cap_height,
                color="black",
                linewidth=1.4,
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
                    "linewidth": 1.4,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
                zorder=6,
            )
        midpoint = (row.original_accuracy + row.synthetic_mean) / 2
        ax.text(
            midpoint,
            arrow_y + peak_density * 0.04,
            f"gap {row.shift_pp:+.1f} pp",
            color="black",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=7,
        )

        ax.set_ylabel(
            language_title(row.language),
            rotation=0,
            ha="right",
            va="center",
            labelpad=15,
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylim(0, peak_density * 1.40)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.65)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=11)

    # Compute model-specific accuracy bounds (zoomed xlim per model)
    all_vals: list[float] = []
    for row in stats:
        set_means, _, _ = distributions[row.language]
        all_vals.extend(set_means)
        all_vals.append(row.original_accuracy)

    if all_vals:
        data_min = float(np.min(all_vals))
        data_max = float(np.max(all_vals))
        pad = max(0.04, (data_max - data_min) * 0.08)
        xmin = max(0.0, float(np.floor((data_min - pad) * 20) / 20))
        xmax = min(1.0, float(np.ceil((data_max + pad) * 20) / 20))
        if xmax - xmin < 0.20:
            mid = (xmin + xmax) / 2
            xmin = max(0.0, float(np.floor((mid - 0.10) * 20) / 20))
            xmax = min(1.0, float(np.ceil((mid + 0.10) * 20) / 20))
    else:
        xmin, xmax = 0.0, 1.0

    axes[-1].set_xlim(xmin, xmax)
    axes[-1].xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[-1].tick_params(axis="x", labelsize=14, pad=5)
    axes[-1].set_xlabel("Exact-answer accuracy", fontsize=21, fontweight="bold", labelpad=10)
    fig.supylabel("Density", fontsize=21, fontweight="bold", x=supylabel_x)
    fig.text(
        left_margin,
        title_y,
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
            markeredgewidth=2.8,
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
        bbox_to_anchor=(1.0, legend_y),
        frameon=False,
        ncol=2,
        fontsize=13,
        handlelength=1.4,
        columnspacing=1.3,
    )

    fig.subplots_adjust(
        left=left_margin,
        right=right_margin,
        top=top_margin,
        bottom=bottom_margin,
        hspace=hspace,
    )
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
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


def plot_headline_figure(
    problems: pd.DataFrame,
    headline_model: str = "Qwen2.5-7B-Instruct",
    headline_languages: list[str] | None = None,
    n_sets: int = 2_000,
    seed: int = 42,
    out_root: Path = DEFAULT_OUT_DIR,
) -> None:
    if headline_languages is None:
        headline_languages = ["eng", "zho", "isl"]

    model_problems = problems[problems["model"] == headline_model]
    if model_problems.empty:
        available_models = sorted(problems["model"].unique())
        if not available_models:
            return
        headline_model = available_models[0]
        model_problems = problems[problems["model"] == headline_model]

    avail_langs = [l for l in headline_languages if l in model_problems["language"].unique()]
    if len(avail_langs) < 3:
        all_avail = language_order(set(model_problems["language"].unique()))
        avail_langs = all_avail[:3]

    curves, stats = collect_plot_data(model_problems, avail_langs, n_sets, seed)
    if not stats:
        return

    out_png = out_root / "ridgeline_selected.png"
    out_root.mkdir(parents=True, exist_ok=True)
    plot_distributions(curves, stats, headline_model, out_png)
    print(f"Saved headline figure {out_png}")

    fig_out = REPO_ROOT / "paper" / "artifacts" / "figures" / "ridgeline_selected.png"
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(out_png, fig_out)
    print(f"Saved headline figure {fig_out}")


def main() -> None:
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
        "--headline-model",
        default="Qwen2.5-7B-Instruct",
        help="Model to use for the compact 3-curve headline figure (ridgeline_selected.png).",
    )
    parser.add_argument(
        "--headline-languages",
        nargs="+",
        default=["eng", "zho", "isl"],
        help="Language codes for the compact 3-curve headline figure.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Workers used for header scans and log loading; defaults to 4 for memory efficiency.",
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
    plot_headline_figure(
        problems,
        headline_model=args.headline_model,
        headline_languages=args.headline_languages,
        n_sets=args.samples,
        seed=args.seed,
        out_root=output_root,
    )
    import gc
    gc.collect()

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
            del model_problems
            gc.collect()
            continue

        family = model_family(model)
        out_dir = output_root / path_slug(family)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_slug = path_slug(model)
        base_name = f"{path_slug(args.output_name)}-{model_slug}" if args.output_name else model_slug
        out_png = out_dir / f"{base_name}.png"

        plot_distributions(curves, stats, model, out_png)
        print(f"Saved {out_png}")

        del curves, stats, model_problems
        gc.collect()


if __name__ == "__main__":
    main()
