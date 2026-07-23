"""Shared prompt-number coverage metrics."""

from __future__ import annotations

import re
import json
from functools import lru_cache
from decimal import Decimal, InvalidOperation
from pathlib import Path

NUMBER_RE = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w|\.\d)")
FRACTION_RE = re.compile(r"(?<![\w.])([-+]?\d+)\s*/\s*(\d+)(?![\w.])")
CHEVRON_RE = re.compile(r"<<(.+?)>>", re.DOTALL)
REPLACEMENTS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "multilingual_gsm_symbolic" / "data" / "templates"
)


def normalize_number(token: str) -> Decimal | None:
    """Return a canonical numeric value for a matched token."""
    try:
        value = Decimal(token.replace(",", ""))
    except InvalidOperation:
        return None
    return value.normalize() if value else Decimal(0)


def fraction_decimal(value: str) -> Decimal | None:
    numerator, separator, denominator = value.partition("/")
    if not separator:
        return normalize_number(value)
    numerator_value = normalize_number(numerator)
    denominator_value = normalize_number(denominator)
    if numerator_value is None or not denominator_value:
        return None
    return (numerator_value / denominator_value).normalize()


@lru_cache(maxsize=None)
def word_number_matcher(language: str) -> tuple[re.Pattern[str] | None, dict[str, Decimal]]:
    """Build the localized number aliases from the dataset replacement table."""
    language = "isl" if language == "uncorrected_isl" else language
    path = REPLACEMENTS_DIR / language / "replacements.json"
    if not path.exists():
        return None, {}
    replacements = json.loads(path.read_text(encoding="utf-8"))
    aliases: dict[str, Decimal] = {}
    for value, word in enumerate(replacements.get("numbers", []), start=1):
        aliases[str(word).casefold()] = Decimal(value)
    for word, value in replacements.get("fraction_alnum", []):
        normalized = fraction_decimal(str(value))
        if normalized is not None:
            aliases[str(word).casefold()] = normalized
    for word, value in replacements.get("multi_times", []):
        normalized = normalize_number(str(value))
        if normalized is not None:
            aliases[str(word).casefold()] = normalized
    if not aliases:
        return None, {}
    alternatives = "|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True))
    if language in {"zho", "jpn"}:
        pattern = re.compile(f"(?:{alternatives})", re.IGNORECASE)
    else:
        pattern = re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)
    return pattern, aliases


def extract_numbers(text: str, language: str = "eng") -> set[Decimal]:
    """Extract equivalent digit, numeric-fraction, and localized word-number values."""
    text_without_fractions = FRACTION_RE.sub(lambda match: " " * len(match.group(0)), text)
    numbers = {normalize_number(match.group(0)) for match in NUMBER_RE.finditer(text_without_fractions)}
    for match in FRACTION_RE.finditer(text):
        normalized = fraction_decimal(match.group(0).replace(" ", ""))
        if normalized is not None:
            numbers.add(normalized)
    pattern, aliases = word_number_matcher(language)
    if pattern is not None:
        numbers.update(
            value
            for match in pattern.finditer(text)
            if (value := aliases.get(match.group(0).casefold())) is not None
        )
    numbers.discard(None)
    return numbers


def extract_chevron_side_numbers(text: str, language: str = "eng") -> tuple[set[Decimal], set[Decimal]]:
    """Extract distinct numbers from each side of ``<<lhs=rhs>>`` markers."""
    lhs_numbers: set[Decimal] = set()
    rhs_numbers: set[Decimal] = set()
    for marker in CHEVRON_RE.findall(text):
        lhs, separator, rhs = marker.rpartition("=")
        if not separator:
            continue
        lhs_numbers.update(extract_numbers(lhs, language))
        rhs_numbers.update(extract_numbers(rhs, language))
    return lhs_numbers, rhs_numbers


def display_number(value: Decimal) -> str:
    """Format a normalized Decimal without scientific notation."""
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def number_coverage_counts(
    prompt: object,
    response: object,
    target: object = "",
    language: str = "eng",
) -> dict[str, int | bool]:
    """Return the requested sample-level prompt-number coverage fields."""
    prompt_numbers = extract_numbers(str(prompt or ""), language)
    response_numbers = extract_numbers(str(response or ""), language)
    retrieved_count = len(prompt_numbers & response_numbers)
    lhs_numbers, rhs_numbers = extract_chevron_side_numbers(str(target or ""), language)
    return {
        "all_prompt_numbers_present": retrieved_count == len(prompt_numbers),
        "prompt_number_count": len(prompt_numbers),
        "retrieved_prompt_number_count": retrieved_count,
        "lhs_count": len(lhs_numbers),
        "lhs_retrieved": len(prompt_numbers & lhs_numbers),
        "rhs_count": len(rhs_numbers),
        "rhs_retrieved": len(prompt_numbers & rhs_numbers),
    }
