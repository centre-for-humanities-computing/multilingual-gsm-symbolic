#!/usr/bin/env python3
"""Score MultiZebra model outputs per puzzle, reproducing EuroEval's cell-wise metric.

Reproduces alexandrainst/zebra_puzzles `compute_cell_score`: for each object, the
set of the model's attributes is intersected with the set of the solution's
attributes (order within a row irrelevant, whitespace stripped, CASE-SENSITIVE),
objects matched by sorted key; cell_score = correct cells / (n_objects x n_attrs);
a puzzle is "solved" iff cell_score == 1. We additionally strip ```json ... ```
fences before parsing, which recovers predictions the raw json.loads misses.

Validated against EuroEval's official aggregate (mean |diff| ~ 1.2 points).

Writes one row per (model, language, puzzle): the cell counts (for a binomial
outcome), the solved flag, and the model/language covariates.

    python score_zebra.py   ->  zebra_puzzle_scores.csv
"""

from __future__ import annotations

import ast
import csv
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# log10 Common Crawl pages (CC-MAIN-2026-21), same source as the GSM features.
RESOURCE = {"da": 7.0571, "de": 8.1113, "en": 8.9465, "is": 5.8904, "nl": 7.5967}
LANG3 = {"da": "dan", "de": "deu", "en": "eng", "is": "isl", "nl": "nld"}
# base model, optionally with a "-no-thinking" (reasoning-off) suffix.
MODEL_RE = re.compile(r"(Qwen3\.5-\d+B|EuroLLM-[\d.]+B-Instruct(?:-2512)?)(-no-thinking)?")


def parse_model(filename: str) -> tuple[str, str, str]:
    """Return (config, base_model, reasoning) from an output filename.

    Qwen3.5 models run both ways: the bare name is reasoning-on, "-no-thinking" is
    reasoning-off. EuroLLM has no reasoning variant, so it is labelled off with no
    pair. Config names mirror the GSM analysis, "<base> (reasoning on/off)".
    """
    m = MODEL_RE.search(filename.replace("--", "/"))
    base = m.group(1)
    is_off = m.group(2) is not None
    if base.startswith("Qwen"):
        reasoning = "off" if is_off else "on"
        config = f"{base} (reasoning {reasoning})"
    else:
        reasoning, config = "off", base
    return config, base, reasoning


def parse_grid(x):
    """Parse a solution/prediction into a dict, stripping markdown fences."""
    if isinstance(x, dict):
        return x
    s = str(x).strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    for fn in (json.loads, ast.literal_eval):
        try:
            out = fn(s)
            return out if isinstance(out, dict) else None
        except Exception:
            pass
    return None


def cell_counts(target: dict, predicted) -> tuple[int, int]:
    """Return (n_correct_cells, n_total_cells) per EuroEval's compute_cell_score."""
    solution = {k: {str(a).strip() for a in v} for k, v in target.items()}
    if isinstance(predicted, dict):
        output = {k: {str(a).strip() for a in v} for k, v in predicted.items() if isinstance(v, list)}
    else:
        output = {}

    n_objects = len(solution)
    n_attributes = len(next(iter(target.values())))
    n_total = n_objects * n_attributes

    # objects matched by sorted key, position-zipped (as in compute_cell_score)
    sol_sorted = dict(sorted(solution.items()))
    out_sorted = dict(sorted(output.items()))
    n_correct = 0
    for out_attrs, sol_attrs in zip(out_sorted.values(), sol_sorted.values()):
        n_correct += len(out_attrs & sol_attrs)
    return n_correct, n_total


def model_params() -> dict[str, float]:
    """Parameter counts (in billions) keyed by base model (reasoning suffix stripped)."""
    params = {}
    with open(os.path.join(HERE, "zebra.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            name = r["model_info"]["name"].split("/")[-1]
            base = re.sub(r"[#-]no-thinking$", "", name)
            params[base] = int(r["model_info"]["additional_details"]["num_model_parameters"]) / 1e9
    return params


def main() -> None:
    params = model_params()
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "zebra", "*.json"))):
        fname = os.path.basename(path)
        config, base_model, reasoning = parse_model(fname)
        lang2 = re.search(r"easy-([a-z]{2})-model", fname).group(1)
        with open(path) as f:
            data = json.load(f)
        for v in data.values():
            target = parse_grid(v["target_text"])
            if not isinstance(target, dict):
                continue
            # An empty prediction is a truncation: reasoning models that exhaust the
            # 8k generation budget mid-thought emit nothing. Flagged (not dropped) so
            # the analysis can treat them as wrong or exclude them as a sensitivity.
            truncated = int(str(v["predicted_label"]).strip() == "")
            n_correct, n_total = cell_counts(target, parse_grid(v["predicted_label"]))
            rows.append(
                {
                    "model": config,
                    "base_model": base_model,
                    "reasoning": reasoning,
                    "family": "Qwen3.5" if base_model.startswith("Qwen") else "EuroLLM",
                    "language": LANG3[lang2],
                    "puzzle": v["index"],
                    "truncated": truncated,
                    "n_correct_cells": n_correct,
                    "n_cells": n_total,
                    "solved": int(n_correct == n_total),
                    "params_b": round(params[base_model], 3),
                    "log10_resource": RESOURCE[lang2],
                }
            )

    out_path = os.path.join(HERE, "zebra_puzzle_scores.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_models = len({r["model"] for r in rows})
    n_langs = len({r["language"] for r in rows})
    print(f"wrote {out_path}: {len(rows)} rows ({n_models} models x {n_langs} languages)")


if __name__ == "__main__":
    main()
