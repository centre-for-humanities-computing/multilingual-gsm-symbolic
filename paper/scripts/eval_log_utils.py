from __future__ import annotations

import math
import multiprocessing
import re
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log
from plot_config import model_family, model_name, model_size_b, reasoning_mode, reasoning_variant_name
from scipy.stats import norm


@dataclass(frozen=True)
class ModelInfo:
    family: str
    params_b: float | None
    vocab_size: int | None = None


MODEL_CATALOG = {
    "qwen2.5-0.5b-instruct": ModelInfo("Qwen2.5", 0.5, 151_936),
    "qwen2.5-1.5b-instruct": ModelInfo("Qwen2.5", 1.5, 151_936),
    "qwen2.5-3b-instruct": ModelInfo("Qwen2.5", 3, 151_936),
    "qwen2.5-7b-instruct": ModelInfo("Qwen2.5", 7, 151_936),
    "qwen2.5-14b-instruct": ModelInfo("Qwen2.5", 14, 151_936),
    "qwen2.5-32b-instruct": ModelInfo("Qwen2.5", 32, 151_936),
    "qwen2.5-72b-instruct": ModelInfo("Qwen2.5", 72, 152_064),
    "olmo-2-0425-1b-instruct": ModelInfo("OLMo 2", 1, 100_352),
    "olmo-2-1124-7b-instruct": ModelInfo("OLMo 2", 7, 100_352),
    "olmo-2-1124-13b-instruct": ModelInfo("OLMo 2", 13, 100_352),
    "olmo-2-0325-32b-instruct": ModelInfo("OLMo 2", 32, 100_352),
    "olmo-3-7b-think": ModelInfo("OLMo 3", 7, 100_278),
    "olmo-3-7b-think (reasoning on)": ModelInfo("OLMo 3", 7, 100_278),
    "olmo-3-7b-instruct": ModelInfo("OLMo 3", 7, 100_278),
    "olmo-3-32b-think": ModelInfo("OLMo 3", 32, 100_278),
    "olmo-3-32b-think (reasoning on)": ModelInfo("OLMo 3", 32, 100_278),
    "olmo-3.1-32b-instruct": ModelInfo("OLMo 3", 32, 100_278),
    "granite-3.2-2b-instruct": ModelInfo("Granite", 2, 49_155),
    "granite-3.2-8b-instruct": ModelInfo("Granite", 8, 49_155),
    "qwen3-0.6b": ModelInfo("Qwen3", 0.6, 151_936),
    "qwen3-1.7b": ModelInfo("Qwen3", 1.7, 151_936),
    "qwen3-4b": ModelInfo("Qwen3", 4, 151_936),
    "qwen3-8b": ModelInfo("Qwen3", 8, 151_936),
    "qwen3-14b": ModelInfo("Qwen3", 14, 151_936),
    "qwen3-32b": ModelInfo("Qwen3", 32, 151_936),
    "qwen3.5-0.8b": ModelInfo("Qwen3.5", 0.8, 248_320),
    "qwen3.5-4b": ModelInfo("Qwen3.5", 4, 248_320),
    "qwen3.5-9b": ModelInfo("Qwen3.5", 9, 248_320),
    "qwen3.5-27b": ModelInfo("Qwen3.5", 27, 248_320),
    "gemma-3-1b-it": ModelInfo("Gemma 3", 1, 262_144),
    "gemma-3-4b-it": ModelInfo("Gemma 3", 4, 262_144),
    "gemma-3-12b-it": ModelInfo("Gemma 3", 12, 262_144),
    "gemma-3-27b-it": ModelInfo("Gemma 3", 27, 262_144),
    "apertus-8b-instruct-2509": ModelInfo("Apertus", 8, 131_072),
    "apertus-70b-instruct-2509": ModelInfo("Apertus", 70, 131_072),
    "eurollm-1.7b-instruct": ModelInfo("EuroLLM", 1.7, 128_000),
    "eurollm-9b-instruct-2512": ModelInfo("EuroLLM", 9, 128_000),
    "eurollm-22b-instruct-2512": ModelInfo("EuroLLM", 22, 128_000),
    "phi-4": ModelInfo("Phi 4", 14, 100_352),
    "phi-4-mini-reasoning": ModelInfo("Phi 4", 3.8, 200_064),
}


def infer_model_info(raw_model: str) -> ModelInfo:
    name = model_name(raw_model)
    catalog_name = re.sub(r" \(reasoning (?:on|off)\)$", "", name, flags=re.IGNORECASE)
    catalog_info = MODEL_CATALOG.get(catalog_name.lower())
    if catalog_info:
        return catalog_info

    family = model_family(raw_model)
    size = model_size_b(raw_model)
    params_b = None if math.isinf(size) else size
    return ModelInfo(family, params_b)


def parse_task(task: str) -> tuple[str, str] | None:
    matches = re.findall(r"(original|synthetic)[_-]([a-z]{3}(?:_[a-z]+)?)", task.lower())
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


def discover_logs(inputs: list[Path]) -> list[Path]:
    logs: set[Path] = set()
    for path in inputs:
        if path.is_dir():
            logs.update(candidate for candidate in path.rglob("*.eval") if candidate.stat().st_size >= 1_000)
        elif path.suffix == ".eval" and path.stat().st_size >= 1_000:
            logs.add(path)
    return sorted(logs)



def _read_log_header(path: Path) -> tuple[Path, Any | None, str | None]:
    try:
        return path, read_eval_log(str(path), header_only=True), None
    except Exception as exc:
        return path, None, str(exc)


def select_logs(paths: list[Path], include_incomplete: bool, workers: int) -> list[tuple[Path, Any]]:
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

        key = (log.eval.eval_id, log.eval.task, reasoning_mode(log.eval.model_args))
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


def map_log_loader(
    loader: Callable[[Path, str | None], tuple[str, pd.DataFrame | None, str | None]],
    paths: list[Path],
    scorer: str | None,
    workers: int,
) -> Iterator[tuple[str, pd.DataFrame | None, str | None]]:
    if workers <= 1:
        yield from (loader(path, scorer) for path in paths)
        return

    max_workers = min(workers, len(paths))
    paths_iter = iter(paths)
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(loader, path, scorer) for path in islice(paths_iter, max_workers)}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                try:
                    path = next(paths_iter)
                except StopIteration:
                    continue
                pending.add(pool.submit(loader, path, scorer))


def classify_reasoning_variants(rows: pd.DataFrame) -> pd.DataFrame:
    parsed = rows["model"].map(reasoning_variant_name)
    rows["base_model_candidate"] = [item[0] for item in parsed]
    rows["reasoning"] = [item[1] for item in parsed]
    off_name_by_raw = rows[rows["reasoning"] == "off"].groupby("model_raw")["base_model_candidate"].first()
    rows["canonical_base_model"] = rows["model_raw"].map(off_name_by_raw).fillna(rows["base_model_candidate"])
    has_off_variant = rows["model_raw"].isin(off_name_by_raw.index)
    rows.loc[
        rows["reasoning"].isna() & has_off_variant & (rows["model"] == rows["canonical_base_model"]),
        "reasoning",
    ] = "on"
    return rows[rows["reasoning"].isin(["on", "off"])]


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
