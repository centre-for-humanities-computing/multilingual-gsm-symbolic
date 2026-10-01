import json
from pathlib import Path

import pytest
from conftest import get_unconstrained_template_files

from multilingual_gsm_symbolic import available_languages, load_instruction
from scripts.generate_dataset import generate_template, output_path, stale_and_orphaned


@pytest.mark.parametrize("language", sorted(available_languages()))
def test_instruction(language: str) -> None:
    instruction = load_instruction(language)
    assert "\n" not in instruction and "\\boxed{}" in instruction


def test_dataset_is_up_to_date() -> None:
    stale, orphaned = stale_and_orphaned(list(available_languages()))
    assert not stale and not orphaned, (
        f"{len(stale)} missing or outdated and {len(orphaned)} orphaned files in dataset/, "
        "run `uv run src/scripts/generate_dataset.py`"
    )


@pytest.mark.parametrize("template_path", get_unconstrained_template_files(n=3), ids=lambda path: path.stem)
def test_dataset_is_reproducible(template_path: Path) -> None:
    committed = json.loads(output_path(template_path).read_text(encoding="utf-8"))
    assert generate_template(template_path) == committed
