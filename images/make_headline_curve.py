# /// script
# dependencies = ["pandas", "pyarrow", "numpy", "matplotlib"]
# ///
"""Build the distribution panel of the headline figure.

Each curve is the accuracy of 4,000 resampled benchmark sets: one variant drawn at
random per template, 100 templates per set. The red marker is the accuracy the
original (static) GSM8K items would have reported for the same language.

The analysis table is not part of this package; it is produced by
``paper/scripts/collect_transfer_tables.py`` on the ``analysis`` branch.

Usage:
    uv run images/make_headline_curve.py path/to/analysis.parquet
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

MODEL = "Qwen2.5-7B-Instruct"
LANGS = {"eng": ("English", "#104281"), "dan": ("Danish", "#2a78d6"), "mar": ("Marathi", "#86b6ef")}
# languages excluded from the paper's analysis set: metric variants, the pre-correction
# Icelandic subset, and Urdu (evaluated on only 26 of the 48 models)
EXCLUDED = ["uncorrected_isl", "eng_metric", "urd"]
BLUE, RED, INK, MUTED = "#2a78d6", "#e34948", "#0b0b0b", "#52514e"
GRID = np.linspace(0, 100, 600)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.edgecolor": "#b9b8b3", "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def load(parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    df = df[(df.model == MODEL) & ~df.language.isin(EXCLUDED)]
    # a few languages were evaluated more than once; keep the first run of each
    return df[df.eval_id.isin(df.groupby(["language", "split"]).eval_id.first())]


def resample(df: pd.DataFrame, lang: str, n: int = 4000, seed: int = 0) -> np.ndarray:
    """Accuracy of `n` benchmark sets, each one random variant per template."""
    s = df[(df.language == lang) & (df.split == "synthetic")]
    piv = s.pivot_table(index="source_id", columns=s.groupby("source_id").cumcount(),
                        values="correct", aggfunc="first")
    m = piv.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    templates, variants = m.shape
    return m[np.arange(templates)[None, :], rng.integers(0, variants, (n, templates))].mean(1) * 100


def kde(x: np.ndarray, bw: float = 1.2) -> np.ndarray:
    d = (GRID[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * d ** 2).sum(1) / (len(x) * bw * np.sqrt(2 * np.pi))


def main(parquet: Path, out: Path) -> None:
    df = load(parquet)
    fig, ax = plt.subplots(figsize=(1.95, 1.95))

    means, statics = {}, {}
    for lang, (label, colour) in LANGS.items():
        drawn = resample(df, lang)
        static = df[(df.language == lang) & (df.split == "original")].correct.mean() * 100
        means[lang], statics[lang] = drawn.mean(), static
        y = kde(drawn)
        ax.fill_between(GRID, y, color=colour, alpha=.20, lw=0)
        ax.plot(GRID, y, color=colour, lw=1.6)
        if lang == "eng":
            ax.plot([static, static], [0, .272], color=RED, lw=1.4)
        lift = 8 if lang == "eng" else 0
        ax.annotate(label, (drawn.mean(), y.max()), color=colour, fontsize=8.5, fontweight="bold",
                    ha="center", xytext=(0, 12 + lift), textcoords="offset points")
        ax.annotate(f"{drawn.mean():.0f}%", (drawn.mean(), y.max()), color=colour, fontsize=8,
                    ha="center", xytext=(0, 3 + lift), textcoords="offset points")

    # ── synthetic gap: how far the original items sit above the templates they came
    #    from. Marked on English only — one label is enough to name the effect.
    lo, hi = sorted((means["eng"], statics["eng"]))
    ax.plot([lo, hi], [.285, .285], color=RED, lw=1)
    for x in (lo, hi):
        ax.plot([x, x], [.279, .291], color=RED, lw=1)
    ax.annotate("synthetic gap", (hi, .297), color=RED, fontsize=8,
                ha="right", va="bottom")

    # ── language gap: the same problems, in a different language
    lo, hi = means["mar"], means["eng"]
    ax.annotate("", (lo, .05), (hi, .05),
                arrowprops=dict(arrowstyle="<|-|>", color=MUTED, lw=1,
                                shrinkA=0, shrinkB=0, mutation_scale=7))
    ax.text((lo + hi) / 2, .065, "language gap", color=MUTED, fontsize=8,
            ha="center", va="bottom")

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, .33)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["0", "50", "100%"], fontsize=9)
    ax.tick_params(length=3, pad=3)
    ax.set_xlabel("Benchmark accuracy", fontsize=9.5, color=INK, labelpad=4)
    ax.text(.5, -.30, MODEL, transform=ax.transAxes, ha="center", va="top",
            fontsize=7.5, color=MUTED)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Written: {out}")


if len(sys.argv) != 2:
    sys.exit(__doc__)
main(Path(sys.argv[1]), Path(__file__).parent / "headline_curve_overlay.png")
