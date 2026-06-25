"""Inspect AI tasks for evaluating the Italian GSM-Symbolic templates.

These tasks intentionally load only ``ita/symbolic`` so templates parked under
``ita/exclude`` and the language-level ``ita/ignore`` marker are not considered.
They mirror the prompt and pattern-scorer shape used in ``eval.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from random import Random

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import pattern
from inspect_ai.solver import generate, prompt_template

from multilingual_gsm_symbolic import AnnotatedQuestion, Question, load_data, load_replacements


PROJECT_ROOT = Path(__file__).parent
ITA_SYMBOLIC_DIR = PROJECT_ROOT / "src" / "multilingual_gsm_symbolic" / "data" / "templates" / "ita" / "symbolic"

ITALIAN_PROMPT_TEMPLATE = (
    "Risolvi il seguente problema di matematica passo dopo passo. "
    "Concludi la risposta con una riga che contenga solo la risposta numerica "
    "preceduta da '####', per esempio: '#### 42'.\n\n{prompt}"
)

ANSWER_PATTERN = r"####\s*(.+)"


def _target_from_answer(answer: str) -> str:
    if "####" not in answer:
        raise ValueError(f"Answer does not contain a final '####' target: {answer!r}")
    return answer.split("####")[-1].strip()


def _load_italian_templates() -> list[AnnotatedQuestion]:
    return load_data(directory=ITA_SYMBOLIC_DIR)


def _sample_from_question(question: Question, *, split: str, variant: int) -> Sample:
    return Sample(
        id=f"ita_{split}_{question.id_shuffled:04d}_{variant:02d}",
        input=question.question,
        target=_target_from_answer(question.answer),
        metadata={
            "answer": question.answer,
            "language": "ita",
            "source_id": question.id_shuffled,
            "split": split,
            "variant": variant,
        },
    )


def _sample_from_template(template: AnnotatedQuestion, *, split: str) -> Sample:
    return Sample(
        id=f"ita_{split}_{template.id_shuffled:04d}",
        input=template.question,
        target=_target_from_answer(template.answer),
        metadata={
            "answer": template.answer,
            "language": "ita",
            "source_id": template.id_shuffled,
            "split": split,
            "variant": 0,
        },
    )


def _generate_limited_questions(template: AnnotatedQuestion, *, n: int, seed: int) -> list[Question]:
    """Generate questions without enumerating every valid constrained assignment."""
    rng = Random(seed)
    replacements = load_replacements(template.language)

    valid_combinations = (
        template._filter_invalid_combinations_streaming(  # noqa: SLF001
            template._get_all_possible_assignments(template.constrained_lines, replacements),  # noqa: SLF001
            limit=n,
        )
        if template.constrained_lines
        else [{}]
    )
    if not valid_combinations:
        raise ValueError(f"Template {template.id_shuffled} has no valid constrained assignments.")

    unconstrained_choices = template._precompute_unconstrained(replacements)  # noqa: SLF001
    questions = []
    for variant in range(n):
        unconstrained_assignments = [rng.choice(choices) for choices in unconstrained_choices]
        assignments = dict(valid_combinations[variant % len(valid_combinations)])
        assignments.update({k: v for d in unconstrained_assignments for k, v in d.items()})
        if template.derived_lines:
            assignments.update(template._evaluate_derived_lines(assignments, replacements, rng))  # noqa: SLF001

        questions.append(
            Question(
                question=template.format_question(assignments),
                answer=template.format_answer(assignments),
                id_orig=template.id_orig,
                id_shuffled=template.id_shuffled,
            )
        )
    return questions


def _dataset(split: str, variants_per_template: int = 1, seed: int = 0) -> MemoryDataset:
    templates = _load_italian_templates()
    if split == "original":
        samples = [_sample_from_template(template, split=split) for template in templates]
    elif split == "synthetic":
        samples = []
        for template in templates:
            questions = _generate_limited_questions(
                template,
                n=variants_per_template,
                seed=seed + template.id_shuffled,
            )
            samples.extend(
                _sample_from_question(question, split=split, variant=variant)
                for variant, question in enumerate(questions)
            )
    else:
        raise ValueError(f"Unknown split: {split!r}")
    return MemoryDataset(samples=samples, name=f"multilingual-gsm-symbolic-ita-{split}")


def _task(split: str, variants_per_template: int = 1, seed: int = 0) -> Task:
    return Task(
        dataset=_dataset(split=split, variants_per_template=variants_per_template, seed=seed),
        solver=[
            prompt_template(template=ITALIAN_PROMPT_TEMPLATE),
            generate(),
        ],
        scorer=pattern(pattern=ANSWER_PATTERN),
        metadata={
            "config": "ita",
            "split": split,
            "template_dir": str(ITA_SYMBOLIC_DIR),
            "evaluation_framework": "inspect-ai",
        },
    )


@task
def original_ita() -> Task:
    """Evaluate the 51 localized Italian template defaults."""
    return _task(split="original")


@task
def synthetic_ita(variants_per_template: int = 1, seed: int = 0) -> Task:
    """Evaluate generated variants from the 51 Italian symbolic templates."""
    return _task(split="synthetic", variants_per_template=variants_per_template, seed=seed)
