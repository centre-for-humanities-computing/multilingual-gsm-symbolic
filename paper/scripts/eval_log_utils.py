from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log
from plot_config import model_family, model_name, model_size_b, reasoning_mode


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
    "llama-3.2-1b-instruct": ModelInfo("Llama 3", 1, 128_256),
    "llama-3.2-3b-instruct": ModelInfo("Llama 3", 3, 128_256),
    "llama-3.1-8b-instruct": ModelInfo("Llama 3", 8, 128_256),
    "llama-3.2-8b-instruct": ModelInfo("Llama 3", 8, 128_256),
    "olmo-2-0425-1b-instruct": ModelInfo("OLMo 2", 1),
    "olmo-2-1124-7b-instruct": ModelInfo("OLMo 2", 7),
    "olmo-2-1124-13b-instruct": ModelInfo("OLMo 2", 13),
    "olmo-2-0325-32b-instruct": ModelInfo("OLMo 2", 32),
    "gemma-3-1b-it": ModelInfo("Gemma 3", 1, 262_144),
    "gemma-3-4b-it": ModelInfo("Gemma 3", 4, 262_144),
    "gemma-3-12b-it": ModelInfo("Gemma 3", 12, 262_144),
    "gemma-3-27b-it": ModelInfo("Gemma 3", 27, 262_144),
    "apertus-8b-instruct-2509": ModelInfo("Apertus", 8),
}


def infer_model_info(raw_model: str) -> ModelInfo:
    name = model_name(raw_model)
    catalog_info = MODEL_CATALOG.get(name.lower())
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
