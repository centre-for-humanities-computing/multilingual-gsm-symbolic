"""Shared prompt-number coverage metrics."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

NUMBER_RE = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w|\.\d)")


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


def number_coverage_counts(prompt: object, response: object) -> dict[str, int | bool]:
    """Return the requested sample-level prompt-number coverage fields."""
    prompt_numbers = extract_numbers(str(prompt or ""))
    response_numbers = extract_numbers(str(response or ""))
    retrieved_count = len(prompt_numbers & response_numbers)
    return {
        "all_prompt_numbers_present": retrieved_count == len(prompt_numbers),
        "prompt_number_count": len(prompt_numbers),
        "retrieved_prompt_number_count": retrieved_count,
    }
