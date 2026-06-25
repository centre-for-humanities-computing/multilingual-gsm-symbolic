from __future__ import annotations

import math
import re
from collections.abc import Iterable


LANGUAGE_LABELS = {
    "zho": "Chinese",
    "spa": "Spanish",
    "eng": "English",
    "eng_metric": "English metric",
    "por": "Portuguese",
    "rus": "Russian",
    "deu": "German",
    "fra": "French",
    "ita": "Italian",
    "ukr": "Ukrainian",
    "dan": "Danish",
    "nob": "Norwegian Bokmal",
    "isl": "Icelandic",
}

LANGUAGE_SPEAKERS = {
    "zho": 940_000_000,
    "spa": 486_000_000,
    "eng": 380_000_000,
    "por": 236_000_000,
    "rus": 150_000_000,
    "deu": 100_000_000,
    "fra": 80_000_000,
    "ita": 67_000_000,
    "ukr": 33_000_000,
    "dan": 6_000_000,
    "nob": 5_000_000,
    "isl": 370_000,
}

HUMAN_VERIFIED_LANGUAGES = {"eng", "dan", "rus", "zho"}

LANGUAGE_COLORS = {
    "zho": "#D62828",
    "spa": "#F77F00",
    "eng": "#012169",
    "eng_metric": "#4B5563",
    "por": "#2A9D8F",
    "rus": "#4361EE",
    "deu": "#FFCE00",
    "fra": "#457B9D",
    "ita": "#2D6A4F",
    "ukr": "#3A86FF",
    "dan": "#C60C30",
    "nob": "#002868",
    "isl": "#003897",
}

FAMILY_ORDER = {
    "Qwen2.5": 0,
    "Qwen3": 1,
    "Qwen": 2,
    "DeepSeek-R1-Distill-Qwen": 3,
    "Llama 3": 4,
    "DeepSeek-R1-Distill-Llama": 5,
    "Gemma 3": 6,
    "OLMo 2": 7,
    "EuroLLM": 8,
    "Apertus": 9,
    "BLOOMZ": 10,
    "Pythia": 11,
    "OpenAI": 12,
}

LANGUAGE_ORDER = {
    "zho": 0,
    "spa": 1,
    "eng": 2,
    "eng_metric": 3,
    "por": 4,
    "rus": 5,
    "deu": 6,
    "fra": 7,
    "ita": 8,
    "ukr": 9,
    "dan": 10,
    "nob": 11,
    "isl": 12,
}


def model_name(raw_model: str) -> str:
    return raw_model.rstrip("/").split("/")[-1]


def model_family(raw_model: str) -> str:
    name = model_name(raw_model)
    lower = name.lower()
    raw_lower = raw_model.lower()
    if "deepseek-r1-distill-qwen" in lower:
        return "DeepSeek-R1-Distill-Qwen"
    if "deepseek-r1-distill-llama" in lower:
        return "DeepSeek-R1-Distill-Llama"
    if "qwen2.5" in lower:
        return "Qwen2.5"
    if "qwen3" in lower:
        return "Qwen3"
    if "qwen" in lower:
        return "Qwen"
    if "olmo-2" in lower:
        return "OLMo 2"
    if "eurollm" in lower:
        return "EuroLLM"
    if "gemma-3" in lower:
        return "Gemma 3"
    if re.search(r"llama[-_. ]?3", lower):
        return "Llama 3"
    if "apertus" in lower:
        return "Apertus"
    if "bloomz" in lower:
        return "BLOOMZ"
    if "pythia" in lower:
        return "Pythia"
    if raw_lower.startswith("openai/") or lower.startswith(("gpt-", "o1", "o3", "o4")):
        return "OpenAI"
    return name.split("-", 1)[0].split("_", 1)[0].split(".", 1)[0]


def model_size_b(raw_model: str) -> float:
    name = model_name(raw_model).lower()
    match = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*b(?:\b|[-_])", name)
    if match:
        return float(match.group(1))

    compact_b = re.search(r"(?<![\d.])(\d+)b(\d+)(?:\b|[-_])", name)
    if compact_b:
        whole, decimal = compact_b.groups()
        return float(f"{whole}.{decimal}")

    compact_m = re.search(r"(?<![\d.])(\d+)m(?:\b|[-_])", name)
    if compact_m:
        return float(compact_m.group(1)) / 1_000

    return math.inf


def model_sort_key(raw_model: str) -> tuple[int, float, str]:
    family = model_family(raw_model)
    return FAMILY_ORDER.get(family, 99), model_size_b(raw_model), model_name(raw_model).lower()


def ordered_models(models: Iterable[str]) -> list[str]:
    return sorted(models, key=model_sort_key)


def ordered_families(families: Iterable[str]) -> list[str]:
    unique_families = set(families)
    known = [family for family in FAMILY_ORDER if family in unique_families]
    extra = sorted(unique_families - set(FAMILY_ORDER))
    return known + extra


def language_order(languages: Iterable[str], requested: list[str] | None = None, *, english_first: bool = False) -> list[str]:
    if requested:
        return [language for language in requested if language in set(languages) or not set(languages)]

    def key(language: str) -> tuple[int, int, str]:
        if english_first and language == "eng":
            return -1, 0, language
        known = 0 if language in LANGUAGE_ORDER else 1
        return known, LANGUAGE_ORDER.get(language, 999), language

    return sorted(set(languages), key=key)
