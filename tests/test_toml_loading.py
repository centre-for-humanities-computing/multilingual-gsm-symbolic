"""Test that TOML and JSON template fixtures produce identical AnnotatedQuestion objects."""

from pathlib import Path

import pytest

from multilingual_gsm_symbolic.templates import AnnotatedQuestion

pairs_path = Path(__file__).parent / "test_templates" / "test_json_toml_loading"

json_toml_pairs: list[tuple[Path, Path]] = sorted(
    [(pairs_path / p.name.replace(".toml", ".json"), p) for p in pairs_path.glob("*.toml")],
    key=lambda t: t[0].name,
)


@pytest.mark.parametrize("json_path,toml_path", json_toml_pairs, ids=[t.stem for _, t in json_toml_pairs])
def test_toml_matches_json(json_path: Path, toml_path: Path) -> None:
    """AnnotatedQuestion loaded from TOML must equal the one loaded from JSON."""
    assert AnnotatedQuestion.from_toml(toml_path) == AnnotatedQuestion.from_json(json_path)
