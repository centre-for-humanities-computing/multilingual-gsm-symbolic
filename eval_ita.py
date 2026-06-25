"""Inspect AI tasks for evaluating the Italian GSM-Symbolic templates.

These tasks intentionally load only ``ita/symbolic`` so templates parked under
``ita/symbolic_todo`` and the language-level ``ita/ignore`` marker are not considered.
They mirror the prompt and pattern-scorer shape used in ``eval.yaml``.
# TODO: adapt once ita/symbolic_todo templates are ready for evaluation.
"""

from __future__ import annotations

from pathlib import Path
from random import Random

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model._model_output import ModelOutput, ModelUsage
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


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _install_openai_compatible_usage_patch() -> None:
    """Handle SGLang/OpenAI-compatible responses that report null usage fields.

    Inspect 0.3.241 assumes usage.prompt_tokens is always an int. Some local
    OpenAI-compatible servers return a usage object with null token counts,
    which otherwise crashes the eval before scoring the model answer.
    """

    def safe_model_output_from_openai(completion, choices):  # type: ignore[no-untyped-def]
        usage = None
        if completion.usage:
            prompt_tokens = _optional_int(getattr(completion.usage, "prompt_tokens", None)) or 0
            completion_tokens = _optional_int(getattr(completion.usage, "completion_tokens", None)) or 0
            total_tokens = _optional_int(getattr(completion.usage, "total_tokens", None))
            prompt_details = getattr(completion.usage, "prompt_tokens_details", None)
            cached_tokens = (
                _optional_int(getattr(prompt_details, "cached_tokens", None)) or 0 if prompt_details is not None else 0
            )
            completion_details = getattr(completion.usage, "completion_tokens_details", None)
            reasoning_tokens = (
                _optional_int(getattr(completion_details, "reasoning_tokens", None))
                if completion_details is not None
                else None
            )
            usage = ModelUsage(
                input_tokens=max(prompt_tokens - cached_tokens, 0),
                output_tokens=completion_tokens,
                input_tokens_cache_read=cached_tokens if prompt_details is not None else None,
                reasoning_tokens=reasoning_tokens,
                total_tokens=total_tokens if total_tokens is not None else prompt_tokens + completion_tokens,
            )

        return ModelOutput(model=completion.model, choices=choices, usage=usage)

    import inspect_ai.model._openai as openai_model
    import inspect_ai.model._providers.openai_compatible as openai_compatible
    import inspect_ai.model._providers.openai_completions as openai_completions

    openai_model.model_output_from_openai = safe_model_output_from_openai
    openai_compatible.model_output_from_openai = safe_model_output_from_openai
    openai_completions.model_output_from_openai = safe_model_output_from_openai


_install_openai_compatible_usage_patch()


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
