# /// script
# dependencies = ["matplotlib", "numpy", "pandas", "pyarrow", "scipy"]
# ///
"""Plot the absolute transfer gap with reasoning enabled vs. disabled as a line plot with no title."""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = (
    REPO_ROOT
    / "paper"
    / "artifacts"
    / "figures"
    / "model_grid"
    / "qwen_compute_budget_transfer"
    / "figure_11_data.csv"
)
DEFAULT_OUT = (
    REPO_ROOT
    / "paper"
    / "artifacts"
    / "figures"
    / "model_grid"
    / "absolute_transfer_gap.png"
)

PLOT_STYLE = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "figure.dpi": 300,
    "font.family": "sans-serif",
}


def plot_absolute_gap_line(
    df: pd.DataFrame,
    out_path: Path,
    family: str = "Qwen3",
    use_log_params: bool = False,
) -> Path:
    plt.rcParams.update(PLOT_STYLE)

    qwen = df[df["family"] == family].copy()
    if qwen.empty:
        raise ValueError(f"No rows found for family '{family}'")

    # Sort models by parameter count
    qwen = qwen.sort_values(["params_b", "reasoning"])
    models = (
        qwen.drop_duplicates(subset=["model_raw"])
        .sort_values("params_b")["model_raw"]
        .tolist()
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    # Styling aligned with paper and original figure
    styles = {
        "off": {
            "label": "reasoning off",
            "color": "#4B5563",
            "linestyle": "--",
            "marker": "s",
        },
        "on": {
            "label": "reasoning on",
            "color": "#2563EB",
            "linestyle": "-",
            "marker": "o",
        },
    }

    x_positions = {model: idx for idx, model in enumerate(models)}

    for reasoning in ["off", "on"]:
        subset = qwen[qwen["reasoning"] == reasoning].copy()
        subset = subset.sort_values("params_b")
        if subset.empty:
            continue

        cfg = styles[reasoning]
        if use_log_params:
            x_vals = subset["params_b"].to_numpy()
        else:
            x_vals = np.array([x_positions[m] for m in subset["model_raw"]])

        y_vals = subset["absolute_transfer_gap"].to_numpy()
        y_err = (
            subset["transfer_gap_ci95"].to_numpy()
            if "transfer_gap_ci95" in subset.columns
            else None
        )

        ax.plot(
            x_vals,
            y_vals,
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=2.0,
            marker=cfg["marker"],
            markersize=6.5,
            markerfacecolor=cfg["color"],
            markeredgecolor="white",
            markeredgewidth=1.2,
            label=cfg["label"],
            zorder=3,
        )

        if y_err is not None:
            ax.errorbar(
                x_vals,
                y_vals,
                yerr=y_err,
                fmt="none",
                ecolor=cfg["color"],
                elinewidth=1.2,
                capsize=3,
                capthick=1.2,
                alpha=0.75,
                zorder=2,
            )

    # Configure axes
    if use_log_params:
        ax.set_xscale("log")
        ax.set_xlabel("Model parameters (B; log scale)", fontsize=10, labelpad=8)
        param_vals = sorted(qwen["params_b"].unique())
        ax.set_xticks(param_vals)
        ax.set_xticklabels([f"{p:g}B" for p in param_vals], fontsize=9)
    else:
        ax.set_xlabel("Model", fontsize=10, labelpad=8)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)

    ax.set_ylabel(
        "English accuracy - non-English accuracy",
        fontsize=9.5,
        labelpad=8,
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8, linestyle="-", zorder=1)

    ax.legend(frameon=False, fontsize=9.5, loc="upper right")

    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    # Also save a PDF version for vector graphics in papers
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--family", default="Qwen3")
    parser.add_argument(
        "--log-params",
        action="store_true",
        help="Use model parameter size (log scale) on x-axis instead of categorical models.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    saved = plot_absolute_gap_line(
        df, args.out, family=args.family, use_log_params=args.log_params
    )
    print(f"Saved: {saved}")
    print(f"Saved: {saved.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
