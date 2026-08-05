"""Shared prompt-number coverage metrics."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from unicodedata import normalize

NUMBER_RE = re.compile(r"(?<![\d.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\d|\.\d)")
FRACTION_RE = re.compile(r"(?<![\d.])([-+]?\d+)\s*/\s*(\d+)(?![\d.])")
CJK_FRACTION_RE = re.compile(r"(?<![\d.])([-+]?\d+)\s*分[の之]\s*(\d+)(?![\d.])")
DOT_GROUP_RE = re.compile(r"(?<![\d.])[-+]?\d{1,3}(?:\.\d{3})+(?!\d|\.\d)")
DECIMAL_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")
CHEVRON_RE = re.compile(r"<<(.+?)>>", re.DOTALL)
COMMA_DECIMAL_LANGUAGES = {"dan", "deu", "fao", "fra", "isl", "ita", "nld", "nob", "por", "uncorrected_isl"}


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


def extract_numbers(text: str, language: str = "eng") -> set[Decimal]:
    """Extract equivalent digit and numeric-fraction values."""
    text = normalize("NFKC", text)
    if language in COMMA_DECIMAL_LANGUAGES:
        text = DOT_GROUP_RE.sub(lambda match: match.group(0).replace(".", ""), text)
        text = DECIMAL_COMMA_RE.sub(".", text)
    text = CJK_FRACTION_RE.sub(lambda match: f"{match.group(2)}/{match.group(1)}", text)
    text_without_fractions = FRACTION_RE.sub(lambda match: " " * len(match.group(0)), text)
    numbers = {normalize_number(match.group(0)) for match in NUMBER_RE.finditer(text_without_fractions)}
    for match in FRACTION_RE.finditer(text):
        normalized = fraction_decimal(match.group(0).replace(" ", ""))
        if normalized is not None:
            numbers.add(normalized)
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
        "all_prompt_numbers_present": bool(prompt_numbers) and retrieved_count == len(prompt_numbers),
        "prompt_number_count": len(prompt_numbers),
        "retrieved_prompt_number_count": retrieved_count,
        "lhs_count": len(lhs_numbers),
        "lhs_retrieved": len(response_numbers & lhs_numbers),
        "rhs_count": len(rhs_numbers),
        "rhs_retrieved": len(response_numbers & rhs_numbers),
    }
