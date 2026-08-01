# /// script
# dependencies = ["matplotlib", "pandas", "scipy", "inspect-ai"]
# ///
"""Generate evaluation visualisations for multilingual-gsm-symbolic.

Produces figures under ``paper/artifacts/figures`` by default:
  1. distribution.png  — 20 set-level accuracy dots + KDE, with memorisation gap arrow
  2. by_steps.png      — synthetic accuracy by reasoning-step count with shaded CI

Usage:
    uv run paper/scripts/visualize_results.py
"""

import argparse
import multiprocessing
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eval_log_utils import sample_score
from inspect_ai.log import read_eval_log, read_eval_log_sample_summaries
from plot_config import LANGUAGE_COLORS, LANGUAGE_LABELS, language_order
from scipy.stats import gaussian_kde

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO_ROOT / "hf_dataset" / "logs"
DEFAULT_OUT_DIR = REPO_ROOT / "paper" / "artifacts" / "figures"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)


# ── data loading ──────────────────────────────────────────────────────────────


def _steps(answer: str) -> int:
    return len(re.findall(r"<<", answer))


def _load_log(path: Path) -> tuple[str, list[dict] | None, str | None]:
    try:
        log = read_eval_log(str(path), header_only=True)
        samples = read_eval_log_sample_summaries(str(path))
    except Exception as exc:
        return path.name, None, str(exc)
    if not samples:
        return path.name, None, None
    rows = []
    for s in samples:
        score = sample_score(s, None)
        if score is None:
            continue
        rows.append(
            {
                "sample_id": s.id,
                "epoch": s.epoch,
                "source_id": s.metadata.get("source_id"),
                "language": s.metadata.get("language", "?"),
                "correct": score,
                "steps": _steps(s.metadata.get("answer", "")),
            }
        )
    return log.eval.task.split("/")[-1], rows or None, None


def load_logs(log_dir: Path, workers: int) -> dict[str, pd.DataFrame]:
    paths = [path for path in sorted(log_dir.glob("*.eval")) if path.stat().st_size >= 10_000]
    if not paths:
        return {}
    dfs = {}
    with ProcessPoolExecutor(
        max_workers=min(workers, len(paths)),
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        for split, rows, error in pool.map(_load_log, paths):
            if error:
                print(f"Skipping unreadable log {split}: {error}")
                continue
            if rows is None:
                continue
            dfs[split] = pd.DataFrame(rows)
    return dfs


def per_problem(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse epochs → one row per (sample_id, source_id)."""
    return df.groupby(["sample_id", "source_id", "language", "steps"])["correct"].mean().reset_index()


def build_tables(splits: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Return {lang: {synthetic: df, original: df}} keyed by language."""
    tables: dict = defaultdict(dict)
    for split, df in splits.items():
        lang = df["language"].iloc[0]
        kind = "synthetic" if split.startswith("synthetic") else "original"
        tables[lang][kind] = per_problem(df)
    return dict(tables)


# ── Figure 1: distribution with memorisation gap ─────────────────────────────


def plot_distribution(tables: dict, out: Path) -> None:
    langs = language_order(l for l in tables if "synthetic" in tables[l])
    fig, axes = plt.subplots(1, len(langs), figsize=(5.5 * len(langs), 4.5), sharey=False)
    if len(langs) == 1:
        axes = [axes]

    rng = np.random.default_rng(0)
    n_sets = 500

    for ax, lang in zip(axes, langs):
        color = LANGUAGE_COLORS.get(lang, "steelblue")
        syn = tables[lang]["synthetic"].copy()

        # Build lookup: source_id → array of per-problem accuracies
        by_template = {sid: grp["correct"].values for sid, grp in syn.groupby("source_id")}
        templates = sorted(by_template)

        # Sample n_sets sets: each picks one variant per template uniformly at random
        set_means = np.array([np.mean([rng.choice(by_template[t]) for t in templates]) for _ in range(n_sets)])

        # Histogram of set means (behind KDE)
        ax.hist(set_means, bins=20, color=color, alpha=0.25, edgecolor="none", density=True, zorder=1)

        # KDE line (no fill). A constant bootstrap distribution represents a
        # point mass and has no invertible covariance for gaussian_kde.
        if np.ptp(set_means) <= np.finfo(float).eps:
            peak_x = float(set_means[0])
            peak_y = 1.0
            ax.axvline(peak_x, color=color, linewidth=2, zorder=3, label="Synthetic")
        else:
            kde = gaussian_kde(set_means, bw_method=0.3)
            x = np.linspace(set_means.min() - 0.02, set_means.max() + 0.02, 400)
            y = kde(x)
            ax.plot(x, y, color=color, linewidth=2, zorder=3, label="Synthetic")
            peak_x = x[np.argmax(y)]
            peak_y = y.max()

        # Single dot at the distribution peak
        ax.scatter(peak_x, peak_y, color=color, s=70, zorder=4)

        # Original accuracy line + performance degradation arrow at the dot height
        if "original" in tables[lang]:
            orig_acc = tables[lang]["original"]["correct"].mean()
            ax.axvline(
                orig_acc,
                color="black",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                zorder=2,
                label=f"Original ({orig_acc:.1%})",
            )
            arrow_y = peak_y
            ax.annotate(
                "",
                xy=(orig_acc, arrow_y),
                xytext=(peak_x, arrow_y),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
                zorder=5,
            )
            degradation = peak_x - orig_acc
            mid_x = (orig_acc + peak_x) / 2
            ax.text(
                mid_x,
                arrow_y + peak_y * 0.04,
                f"performance degradation ({degradation:+.1%})",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xlabel("Mean accuracy")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 2: accuracy by steps (synthetic only, shaded CI) ──────────────────


def plot_by_steps(tables: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax2 = ax.twinx()

    # Use the first available synthetic split for the sample-count histogram
    # (step counts are the same across languages as templates are shared)
    count_plotted = False
    all_stats = []

    rng = np.random.default_rng(42)

    for lang in language_order(tables):
        splits = tables[lang]
        if "synthetic" not in splits:
            continue
        color = LANGUAGE_COLORS.get(lang, "grey")
        df = splits["synthetic"]
        grouped = df.groupby("steps")["correct"]
        stats = grouped.agg(["mean", "count"]).reset_index()
        stats = stats[stats["count"] >= 5]
        all_stats.append(stats)

        # Bootstrapped 95% CI per step count
        ci_lo, ci_hi = [], []
        for _, row in stats.iterrows():
            vals = df[df["steps"] == row["steps"]]["correct"].values
            boot_means = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(1000)])
            ci_lo.append(np.percentile(boot_means, 2.5))
            ci_hi.append(np.percentile(boot_means, 97.5))

        ax.plot(
            stats["steps"],
            stats["mean"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            label=f"{LANGUAGE_LABELS.get(lang, lang)} (synthetic)",
            zorder=3,
        )
        ax.fill_between(
            stats["steps"],
            ci_lo,
            ci_hi,
            alpha=0.2,
            color=color,
            zorder=2,
            linewidth=0,
        )

        if not count_plotted:
            ax2.bar(
                stats["steps"], stats["count"], color="grey", alpha=0.2, width=0.6, zorder=1, label="Number of problems"
            )
            count_plotted = True

    ax.set_xlabel("Number of reasoning steps (count of ⟨⟨ ⟩⟩ in answer)")
    ax.set_ylabel("Mean accuracy (95% CI)")
    ax.set_ylim(0, 1.05)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax2.set_ylabel("Number of problems", color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")
    ax2.spines["right"].set_color("grey")
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)

    lines, labels = ax.get_legend_handles_labels()
    bars, blabels = ax2.get_legend_handles_labels()
    ax.legend(lines + bars, labels + blabels, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 3: overlapping language distributions with language gap arrow ──────


def plot_language_gap(tables: dict, out: Path) -> None:
    langs = [l for l in tables if "synthetic" in tables[l]]
    fig, ax = plt.subplots(figsize=(6, 4.5))

    rng = np.random.default_rng(0)
    n_sets = 500
    peaks: dict[str, tuple[float, float]] = {}  # lang → (peak_x, peak_y)

    for lang in langs:
        color = LANGUAGE_COLORS.get(lang, "steelblue")
        syn = tables[lang]["synthetic"].copy()
        by_template = {sid: grp["correct"].values for sid, grp in syn.groupby("source_id")}
        templates = sorted(by_template)
        set_means = np.array([np.mean([rng.choice(by_template[t]) for t in templates]) for _ in range(n_sets)])

        ax.hist(set_means, bins=20, color=color, alpha=0.2, edgecolor="none", density=True, zorder=1)
        label = f"{LANGUAGE_LABELS.get(lang, lang)} (synthetic)"
        if np.ptp(set_means) <= np.finfo(float).eps:
            peak_x = float(set_means[0])
            peak_y = 1.0
            ax.axvline(peak_x, color=color, linewidth=2, zorder=3, label=label)
        else:
            kde = gaussian_kde(set_means, bw_method=0.3)
            x = np.linspace(set_means.min() - 0.02, set_means.max() + 0.02, 400)
            y = kde(x)
            ax.plot(x, y, color=color, linewidth=2, zorder=3, label=label)
            peak_x = x[np.argmax(y)]
            peak_y = y.max()
        ax.scatter(peak_x, peak_y, color=color, s=70, zorder=4)
        peaks[lang] = (peak_x, peak_y)

    # Arrow between highest and lowest peak_x language, raised above distributions
    if len(peaks) >= 2:
        sorted_langs = sorted(peaks, key=lambda l: peaks[l][0])  # noqa
        lo_lang, hi_lang = sorted_langs[0], sorted_langs[-1]
        lo_x, lo_py = peaks[lo_lang]
        hi_x, hi_py = peaks[hi_lang]
        max_peak_y = max(p[1] for p in peaks.values())
        arrow_y = max_peak_y * 1.15

        ax.annotate(
            "",
            xy=(hi_x, arrow_y),
            xytext=(lo_x, arrow_y),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
            zorder=5,
        )
        gap = hi_x - lo_x
        mid_x = (lo_x + hi_x) / 2
        ax.text(mid_x, arrow_y + max_peak_y * 0.05, f"language gap ({gap:+.1%})", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Mean accuracy")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 4: speakers (log) vs synthetic accuracy ───────────────────────────

# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    parser.add_argument("--workers", default=32, type=int)
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True)
    splits = load_logs(args.log_dir, args.workers)
    if not splits:
        raise SystemExit("No completed eval logs found.")

    print("Loaded splits:", list(splits.keys()))
    tables = build_tables(splits)
    print("Languages:", list(tables.keys()))

    plot_distribution(tables, args.out_dir / "distribution.png")
    plot_by_steps(tables, args.out_dir / "by_steps.png")
    plot_language_gap(tables, args.out_dir / "language_gap.png")


if __name__ == "__main__":
    main()
