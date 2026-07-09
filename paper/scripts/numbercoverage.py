# /// script
# dependencies = ["inspect-ai", "matplotlib", "numpy", "pandas", "scipy"]
# ///
"""Measure whether every number in each prompt appears in the model response.

Numbers are matched as complete numeric tokens and normalized by value, so
10, 10.0, and 10,000-style formatting compare consistently. Repeated prompt
values are treated as one distinct number.

Example:
    uv run paper/scripts/numbercoverage.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from inspect_ai.log import read_eval_log
from matplotlib.ticker import PercentFormatter
from plot_config import (
    HUMAN_VERIFIED_LANGUAGES,
    LANGUAGE_LABELS,
    LANGUAGE_ORDER,
    LANGUAGE_SPEAKERS,
    format_speaker_count,
    heatmap_language_label,
    language_order,
    model_name,
    model_sort_key,
)
from eval_log_utils import select_logs

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO_ROOT / "hf_dataset" / "logs"
DEFAULT_OUT_DIR = REPO_ROOT / "paper" / "artifacts" / "prompt_number_coverage"
NUMBER_RE = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w|\.\d)")
plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.dpi": 160,
        "font.family": "sans-serif",
    }
)


def normalize_number(token: str) -> Decimal | None:
    """Return a canonical numeric value for a matched token."""
    try:
        value = Decimal(token.replace(",", ""))
    except InvalidOperation:
        return None
    return value.normalize() if value else Decimal(0)


def extract_numbers(text: str) -> set[Decimal]:
    """Extract distinct, normalized numeric tokens from text."""
    numbers = {normalize_number(match.group(0)) for match in NUMBER_RE.finditer(text)}
    numbers.discard(None)
    return numbers


def display_number(value: Decimal) -> str:
    """Format a normalized Decimal without scientific notation."""
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def score_to_bool(sample: Any) -> bool | None:
    """Read the preferred final-answer score from an Inspect sample."""
    scores = sample.scores or {}
    for preferred in ("math", "pattern"):
        if preferred not in scores:
            continue
        value = getattr(scores[preferred], "value", scores[preferred])
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized in {"C", "CORRECT", "TRUE", "1"}:
                return True
            if normalized in {"I", "INCORRECT", "FALSE", "0"}:
                return False
    return None


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    covered = sum(row["all_prompt_numbers_present"] for row in rows)
    return {
        "samples": len(rows),
        "samples_with_all_prompt_numbers_present": covered,
        "samples_missing_prompt_numbers": len(rows) - covered,
        "all_prompt_numbers_present_rate": safe_rate(covered, len(rows)),
    }


def analyze_log(path: Path, max_samples: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    log = read_eval_log(str(path))
    samples = (log.samples or [])[:max_samples]
    sample_rows: list[dict[str, Any]] = []

    for sample in samples:
        prompt = sample.input if isinstance(sample.input, str) else str(sample.input)
        response = sample.output.completion
        prompt_numbers = extract_numbers(prompt)
        response_numbers = extract_numbers(response)
        missing_numbers = prompt_numbers - response_numbers

        sample_rows.append(
            {
                "log": path.name,
                "model": log.eval.model,
                "task": log.eval.task,
                "sample_id": sample.id,
                "epoch": sample.epoch,
                "source_id": (sample.metadata or {}).get("source_id"),
                "language": (sample.metadata or {}).get("language"),
                "final_correct": score_to_bool(sample),
                "prompt_number_count": len(prompt_numbers),
                "prompt_numbers": " | ".join(display_number(value) for value in sorted(prompt_numbers)),
                "missing_prompt_number_count": len(missing_numbers),
                "missing_prompt_numbers": " | ".join(display_number(value) for value in sorted(missing_numbers)),
                "all_prompt_numbers_present": not missing_numbers,
            }
        )

    final_scored = [row for row in sample_rows if row["final_correct"] is not None]
    final_correct_rows = [row for row in final_scored if row["final_correct"]]
    final_incorrect_rows = [row for row in final_scored if not row["final_correct"]]
    correct_summary = summarize_group(final_correct_rows)
    incorrect_summary = summarize_group(final_incorrect_rows)
    correct_rate = correct_summary["all_prompt_numbers_present_rate"]
    incorrect_rate = incorrect_summary["all_prompt_numbers_present_rate"]

    summary = {
        "log": path.name,
        "status": str(log.status),
        "model": log.eval.model,
        "task": log.eval.task,
        "samples": len(samples),
        "final_scored_samples": len(final_scored),
        "final_accuracy": safe_rate(
            sum(row["final_correct"] for row in final_scored),
            len(final_scored),
        ),
        "samples_with_all_prompt_numbers_present": sum(row["all_prompt_numbers_present"] for row in sample_rows),
        "all_prompt_numbers_present_rate": safe_rate(
            sum(row["all_prompt_numbers_present"] for row in sample_rows),
            len(sample_rows),
        ),
        "number_coverage_breakdown": {
            "final_correct": correct_summary,
            "final_incorrect": incorrect_summary,
        },
        "correct_minus_incorrect_percentage_points": (
            (correct_rate - incorrect_rate) * 100 if correct_rate is not None and incorrect_rate is not None else None
        ),
        "matcher": "distinct normalized complete numeric tokens",
        "note": (
            "A sample is covered when every distinct numeric value in the original prompt "
            "appears somewhere in the full model completion."
        ),
    }
    return summary, sample_rows


def discover_logs(inputs: list[Path]) -> list[Path]:
    logs: set[Path] = set()
    for path in inputs:
        if path.is_dir():
            logs.update(candidate for candidate in path.rglob("*.eval") if candidate.stat().st_size >= 1_000)
        elif path.suffix == ".eval" and path.stat().st_size >= 1_000:
            logs.add(path)
    return sorted(logs)


def resolve_worker_count(requested: int | None, log_count: int) -> int:
    if requested is not None and requested < 1:
        raise ValueError("--workers must be at least 1.")
    default = 128
    return min(requested or default, log_count)


def analyze_logs(
    logs: list[Path],
    max_samples: int | None,
    workers: int,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    if workers == 1:
        results = []
        for path in logs:
            print(f"Analyzing {path}")
            results.append(analyze_log(path, max_samples))
        return results

    ordered_results: list[tuple[dict[str, Any], list[dict[str, Any]]] | None] = [None] * len(logs)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyze_log, path, max_samples): (index, path) for index, path in enumerate(logs)}
        for future in as_completed(futures):
            index, path = futures[future]
            ordered_results[index] = future.result()
            print(f"Analyzed {path}")

    return [result for result in ordered_results if result is not None]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number_coverage_grid(
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Aggregate sample-level number coverage into model-by-language cells."""
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        model = str(row.get("model") or "").strip()
        language = str(row.get("language") or "").strip()
        if not model or language not in LANGUAGE_ORDER:
            continue
        cell = counts[(model, language)]
        cell[0] += int(bool(row["all_prompt_numbers_present"]))
        cell[1] += 1

    models = sorted({model for model, _ in counts}, key=model_sort_key)
    languages = language_order(language for _, language in counts)
    rates = np.full((len(models), len(languages)), np.nan)
    samples = np.zeros((len(models), len(languages)), dtype=int)
    for model_index, model in enumerate(models):
        for language_index, language in enumerate(languages):
            covered, total = counts.get((model, language), (0, 0))
            if total:
                rates[model_index, language_index] = covered / total
                samples[model_index, language_index] = total
    return models, languages, rates, samples


def plot_number_coverage_heatmap(rows: list[dict[str, Any]], path: Path) -> bool:
    """Plot the sample-level coverage results by evaluated model and language."""
    models, languages, rates, samples = number_coverage_grid(rows)
    if not models or not languages:
        return False

    display_names = [model_name(model) for model in models]
    if len(set(display_names)) != len(display_names):
        display_names = models

    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("#E0E0E0")
    fig, ax = plt.subplots(figsize=(max(7, 1.35 * len(languages) + 3.5), max(4, 0.6 * len(models) + 1.8)))
    image = ax.imshow(rates, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(
        range(len(languages)),
        [heatmap_language_label(language) for language in languages],
        rotation=40,
        ha="right",
    )
    for column_index, language in enumerate(languages):
        if language not in HUMAN_VERIFIED_LANGUAGES:
            continue
        ax.annotate(
            "*",
            xy=(column_index, 0),
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
    ax.set_yticks(range(len(models)), display_names)
    ax.set_xlabel("Problem language (* = Human verified)")
    ax.set_ylabel("Evaluated instruction-tuned model")

    for row_index in range(len(models)):
        for column_index in range(len(languages)):
            rate = rates[row_index, column_index]
            if np.isnan(rate):
                continue
            text_color = "white" if rate < 0.45 else "black"
            ax.text(
                column_index,
                row_index,
                f"{rate:.1%}\n(n={samples[row_index, column_index]:,})",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )

    colorbar = fig.colorbar(image, ax=ax, label="Samples containing every prompt number")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.subplots_adjust(left=0.22, right=0.88, bottom=0.31, top=0.96)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def coverage_accuracy_points(summaries: list[dict[str, Any]]) -> list[tuple[str, str, str, float, float]]:
    """Return plot points as (model, split, language, coverage, accuracy)."""
    points = []
    for summary in summaries:
        coverage = summary.get("all_prompt_numbers_present_rate")
        accuracy = summary.get("final_accuracy")
        matches = re.findall(r"(original|synthetic)[_-]([a-z]{3}(?:_[a-z]+)?)", str(summary.get("task", "")).lower())
        if coverage is None or accuracy is None or not matches:
            continue
        split, language = matches[-1]
        if split != "synthetic":
            continue
        points.append((model_name(str(summary.get("model", ""))), split, language, float(coverage), float(accuracy)))
    return points


def plot_coverage_accuracy_correlation(summaries: list[dict[str, Any]], path: Path) -> bool:
    """Plot prompt-number coverage against final-answer accuracy."""
    points = coverage_accuracy_points(summaries)
    if not points:
        return False

    x = np.array([point[3] for point in points])
    y = np.array([point[4] for point in points])
    corr = np.corrcoef(x, y)[0, 1] if len(points) > 1 and np.std(x) and np.std(y) else np.nan

    fig, ax = plt.subplots(figsize=(7, 5))
    split_colors = {"original": "#2F80ED", "synthetic": "#F2994A"}
    for split in ("original", "synthetic"):
        split_points = [point for point in points if point[1] == split]
        if not split_points:
            continue
        ax.scatter(
            [point[3] for point in split_points],
            [point[4] for point in split_points],
            s=45,
            alpha=0.75,
            color=split_colors[split],
            edgecolor="white",
            linewidth=0.5,
            label=split.title(),
        )

    if len(points) > 1 and len(set(x)) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 100)
        ax.plot(line_x, slope * line_x + intercept, color="#222222", linewidth=1.5, label="Linear fit")

    label = "r = n/a" if np.isnan(corr) else f"r = {corr:.2f}"
    ax.text(0.03, 0.95, label, transform=ax.transAxes, ha="left", va="top", fontsize=11)
    ax.set_xlabel("Prompt-number coverage")
    ax.set_ylabel("Final-answer accuracy")
    ax.set_title("Synthetic number coverage vs. accuracy")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlim(max(0, x.min() - 0.03), min(1, x.max() + 0.03))
    ax.set_ylim(max(0, y.min() - 0.05), min(1, y.max() + 0.05))
    ax.grid(axis="both", color="#E6E6E6", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def run_self_test() -> None:
    cases = [
        ("Use 10, 10.0, and 1,000.", "10 and 1000", True),
        ("Use 10 and 2.", "The result is 210.", False),
        ("Temperatures are -5 and +3.", "-5 + 3 = -2", True),
        ("There are 4 groups of 4.", "4 groups", True),
        ("A value is 1.5.", "15", False),
    ]
    for prompt, response, expected in cases:
        actual = extract_numbers(prompt) <= extract_numbers(response)
        if actual != expected:
            raise AssertionError(f"Failed: prompt={prompt!r}, response={response!r}, expected={expected}, got={actual}")
    models, languages, rates, samples = number_coverage_grid(
        [
            {"model": "provider/model-1B", "language": "eng", "all_prompt_numbers_present": True},
            {"model": "provider/model-1B", "language": "eng", "all_prompt_numbers_present": False},
            {"model": "provider/model-1B", "language": "dan", "all_prompt_numbers_present": True},
        ]
    )
    if models != ["provider/model-1B"] or languages != ["eng", "dan"]:
        raise AssertionError("Failed to build the expected model-language coverage grid.")
    if rates.tolist() != [[0.5, 1.0]] or samples.tolist() != [[2, 1]]:
        raise AssertionError("Failed to aggregate model-language coverage rates.")
    points = coverage_accuracy_points(
        [
            {
                "model": "provider/model-1B",
                "task": "hf/repo/synthetic_eng/synthetic_eng",
                "final_accuracy": 0.75,
                "all_prompt_numbers_present_rate": 0.95,
            },
            {
                "model": "provider/model-1B",
                "task": "hf/repo/original_dan/original_dan",
                "final_accuracy": 0.5,
                "all_prompt_numbers_present_rate": 0.9,
            },
        ]
    )
    if points != [("model-1B", "synthetic", "eng", 0.95, 0.75)]:
        raise AssertionError("Failed to extract coverage/accuracy correlation points.")
    print(f"Self-test passed ({len(cases)} cases).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=f"Inspect .eval logs or directories containing logs. Defaults to {DEFAULT_LOG_DIR}.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to analyze per log. Defaults to all samples.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Parallel log processes. Defaults to min(4, CPU count); use 1 for serial execution.",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        if not args.inputs:
            return

    inputs = args.inputs or [DEFAULT_LOG_DIR]
    logs = discover_logs(inputs)
    if not logs:
        raise SystemExit("No non-empty .eval logs found.")

    try:
        workers = resolve_worker_count(args.workers, len(logs))
    except ValueError as error:
        parser.error(str(error))
    selected = select_logs(logs, include_incomplete=False, workers=workers)
    logs = [path for path, _header in selected]
    print(f"Analyzing {len(logs)} successful deduplicated log(s) with {workers} worker(s).")

    summaries: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for summary, samples in analyze_logs(logs, args.max_samples, workers):
        summaries.append(summary)
        sample_rows.extend(samples)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(args.out_dir / "samples.csv", sample_rows)
    heatmap_path = args.out_dir / "number_coverage_heatmap.png"
    wrote_heatmap = plot_number_coverage_heatmap(sample_rows, heatmap_path)
    correlation_path = args.out_dir / "coverage_accuracy_correlation.png"
    wrote_correlation = plot_coverage_accuracy_correlation(summaries, correlation_path)

    for summary in summaries:
        breakdown = summary["number_coverage_breakdown"]
        print(
            f"{summary['model']} | {summary['task']}: "
            f"all={summary['all_prompt_numbers_present_rate']!s}, "
            f"right={breakdown['final_correct']['all_prompt_numbers_present_rate']!s}, "
            f"wrong={breakdown['final_incorrect']['all_prompt_numbers_present_rate']!s}"
        )
    if wrote_heatmap:
        print(f"Wrote {heatmap_path}")
    if wrote_correlation:
        print(f"Wrote {correlation_path}")
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
