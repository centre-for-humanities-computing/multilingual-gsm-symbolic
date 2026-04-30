"""Test that TOML and JSON template fixtures produce identical AnnotatedQuestion objects."""

from pathlib import Path

import pytest

from multilingual_gsm_symbolic.gsm_parser import AnnotatedQuestion

_FIXTURES = Path(__file__).parent / "test_templates"

# Every (json, toml) pair present in tests/fixtures/
_PAIRS = sorted(
    [(_FIXTURES / p.name.replace(".toml", ".json"), p) for p in _FIXTURES.glob("*.toml")],
    key=lambda t: t[0].name,
)


@pytest.mark.parametrize("json_path,toml_path", _PAIRS, ids=[t.stem for _, t in _PAIRS])
def test_toml_matches_json(json_path: Path, toml_path: Path) -> None:
    """AnnotatedQuestion loaded from TOML must equal the one loaded from JSON."""
    assert AnnotatedQuestion.from_toml(toml_path) == AnnotatedQuestion.from_json(json_path)
