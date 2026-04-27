import json
from pathlib import Path

import pytest

from multilingual_gsm_symbolic.gsm_parser import AnnotatedQuestion
from multilingual_gsm_symbolic.load_data import (
    _DATA_ROOT,
    _active_template_files,
    available_languages,
    load_replacements,
)


def test_get_combinations_deduplicates_numeric_solutions():
    template = AnnotatedQuestion(
        question="Q",
        answer="A",
        id_orig=1,
        id_shuffled=1,
        question_annotated=(
            "{name,Ada} has {x,1} apples.\n"
            "#init:\n"
            '- name = sample(["Ada", "Grace"])\n'
            "- $x = range(1, 4)\n"
            "#conditions:\n"
            "- True\n"
            "#answer: x"
        ),
        answer_annotated="{x}",
    )

    combinations = template.get_combinations(replacements={}, only_numeric=True)

    assert combinations == [{"x": 1}, {"x": 2}, {"x": 3}]


_CACHE_DIR = Path(__file__).with_name("combinations_cache")


def _cache_path(language: str) -> Path:
    return _CACHE_DIR / f"{language}.json"


def _load_language_cache(language: str) -> dict[str, int]:
    path = _cache_path(language)
    if not path.exists():
        return {}
    entries = json.loads(path.read_text(encoding="utf-8"))
    return {entry["question_annotated"]: entry["count"] for entry in entries}


def _write_language_cache(
    language: str, cache: dict[str, int], templates: list[tuple[Path, AnnotatedQuestion]]
) -> None:
    _CACHE_DIR.mkdir(exist_ok=True)
    template_ids = {template.question_annotated: template.id_shuffled for _, template in templates}
    entries = [
        {"id": template_ids[question_annotated], "question_annotated": question_annotated, "count": cache[question_annotated]}
        for question_annotated in sorted(cache, key=lambda question_annotated: (template_ids[question_annotated], question_annotated))
    ]
    _cache_path(language).write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sync_language_cache(language: str) -> tuple[list[tuple[Path, AnnotatedQuestion]], dict[str, int]]:
    templates = [
        (path, AnnotatedQuestion.from_json(path)) for path in sorted(_active_template_files(_DATA_ROOT / language))
    ]
    original_cache = _load_language_cache(language)
    valid_questions = {template.question_annotated for _, template in templates}

    cache = {
        question_annotated: count
        for question_annotated, count in original_cache.items()
        if question_annotated in valid_questions
    }

    if cache != original_cache:
        _write_language_cache(language, cache, templates)

    replacements = load_replacements(language)
    for _, template in templates:
        if template.question_annotated not in cache:
            combinations = template.get_combinations(replacements=replacements, only_numeric=True, limit=100)
            cache[template.question_annotated] = len(combinations)
            _write_language_cache(language, cache, templates)

    return templates, cache


@pytest.mark.parametrize("language", sorted(available_languages()))
def test_each_template_has_at_least_100_solutions(language: str):
    templates, cache = _sync_language_cache(language)

    failures = []
    for path, template in templates:
        count = cache[template.question_annotated]
        if count < 100:
            failures.append((path, count, template.question_annotated.splitlines()[0]))
    if failures:
        details = "\n".join(
            f"- {path}\n  numeric combinations: {count}\n  template: {first_line}"
            for path, count, first_line in failures
        )
        pytest.fail(f"Templates with fewer than 100 numeric combinations in {language}:\n{details}")
