"""Unit tests for helpers."""

import pytest

from multilingual_gsm_symbolic._helpers import plural


@pytest.mark.parametrize(
    "n,expected",
    [(1, "coin"), (2, "coins"), (5, "coins"), (0, "coins")],
)
def test_plural_two_forms_english(n, expected):
    assert plural(n, "coin", "coins") == expected


@pytest.mark.parametrize(
    "n,expected",
    [
        (1, "рік"),
        (2, "роки"),
        (4, "роки"),
        (5, "років"),
        (11, "років"),
        (12, "років"),
        (21, "рік"),
        (22, "роки"),
        (25, "років"),
        (100, "років"),
    ],
)
def test_plural_three_forms_east_slavic(n, expected):
    assert plural(n, "рік", "роки", "років") == expected
