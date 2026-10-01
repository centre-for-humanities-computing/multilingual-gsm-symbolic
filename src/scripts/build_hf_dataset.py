# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "multilingual-gsm-symbolic",
#   "datasets>=3.0.0",
#   "huggingface-hub>=1.0.0",
# ]
#
# [tool.uv.sources]
# multilingual-gsm-symbolic = { path = "../..", editable = true }
# ///
"""Build the Hugging Face dataset from ``dataset/`` and push it to the Hub.

- ``build`` writes the parquet files and ``eval.yaml`` to ``--out``. It fails if ``dataset/`` is not up to date
  with the templates (run ``src/scripts/generate_dataset.py`` first).
- ``push`` does the same, then downloads the dataset card (``README.md``) from the Hub, updates the parts between
  its markers (the metadata, the version and the language overview), uploads the folder and tags the commit
  ``v<version>``. Everything else in the card is edited directly on the Hub.

Validated languages get the splits ``test_original`` and ``test_synthetic``; all other languages get
``test_original_unvalidated`` and ``test_synthetic_unvalidated``. The English train splits are not built here and
are left untouched on the Hub.

Usage:
    uv run src/scripts/build_hf_dataset.py build --out build/hf_dataset
    uv run src/scripts/build_hf_dataset.py push --out build/hf_dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import multilingual_gsm_symbolic  # noqa: E402
from multilingual_gsm_symbolic import available_languages  # noqa: E402
from multilingual_gsm_symbolic.load_data import _DATA_ROOT, _active_template_files  # noqa: E402
from scripts.generate_dataset import DATASET_DIR, output_path, stale_and_orphaned  # noqa: E402
from scripts.update_readme_table import (  # noqa: E402
    LanguageValidation,
    collect_language_validation,
    render_language_tables,
)

REPO_ID = "danish-foundation-models/multilingual-gsm-symbolic"
GITHUB_URL = "https://github.com/centre-for-humanities-computing/multilingual-gsm-symbolic"
# Splits that exist on the Hub but are not built by this script
EXTRA_SPLITS = {"eng": ("train_original", "train_synthetic")}
# Files on the Hub that are replaced on every push (the English train splits are kept) or no longer used
DELETE_PATTERNS = ["data/*/test_*.parquet", "data/generation_log.json", "generate_hf_dataset.py"]
# The parts of the dataset card that are updated on every push
CARD_MARKERS = {
    "metadata": ("# GENERATED METADATA START (updated on every release)", "# GENERATED METADATA END"),
    "version": ("<!-- VERSION START (updated on every release) -->", "<!-- VERSION END -->"),
    "languages": ("<!-- LANGUAGES START (updated on every release) -->", "<!-- LANGUAGES END -->"),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Language:
    code: str
    validated: bool
    instruction: str
    instruction_validated: bool

    def split(self, kind: str) -> str:
        """Name of the ``original`` or ``synthetic`` test split."""
        return f"test_{kind}" if self.validated else f"test_{kind}_unvalidated"

    def task_id(self, kind: str) -> str:
        return f"{kind}_{self.code}" if self.validated else f"{kind}_unvalidated_{self.code}"

    @property
    def splits(self) -> list[str]:
        return [self.split("original"), self.split("synthetic"), *EXTRA_SPLITS.get(self.code, ())]


def is_validated(language: LanguageValidation) -> bool:
    """A language is validated once every template has been checked by a native speaker."""
    return bool(language.human) and language.human != "In progress"


def collect_languages(validation: list[LanguageValidation]) -> list[Language]:
    validated = {lang.language for lang in validation if is_validated(lang)}
    languages = []
    for code in available_languages():
        with (_DATA_ROOT / code / "instruction.toml").open("rb") as file:
            instruction = tomllib.load(file)
        instruction_validated = bool(instruction.get("human-validated"))
        languages.append(Language(code, code in validated, instruction["instruction"], instruction_validated))
    return languages


def load_rows(language: str, dataset_dir: Path = DATASET_DIR) -> tuple[list[dict], list[dict]]:
    """The original and synthetic rows of a language, in template order."""
    original, synthetic = [], []
    for template_path in _active_template_files(_DATA_ROOT / language):
        record = json.loads(output_path(template_path, dataset_dir).read_text(encoding="utf-8"))
        meta = {"language": record["language"], "source_id": record["source_id"]}
        original.append({**record["original"], **meta})
        synthetic.extend({**question, **meta} for question in record["synthetic"])
    return original, synthetic


def build(languages: list[Language], out: Path, dataset_dir: Path = DATASET_DIR) -> int:
    """Write the parquet files and eval.yaml; returns the number of rows."""
    from datasets import Dataset

    n_rows = 0
    for lang in languages:
        lang_dir = out / "data" / lang.code
        lang_dir.mkdir(parents=True, exist_ok=True)
        for kind, rows in zip(("original", "synthetic"), load_rows(lang.code, dataset_dir), strict=True):
            Dataset.from_list(rows).to_parquet(lang_dir / f"{lang.split(kind)}-00000-of-00001.parquet")
            n_rows += len(rows)
    (out / "eval.yaml").write_text(render_eval_yaml(languages), encoding="utf-8")
    return n_rows


def _yaml_str(value: str) -> str:
    # JSON strings are valid YAML double-quoted scalars
    return json.dumps(value, ensure_ascii=False)


def render_eval_yaml(languages: list[Language]) -> str:
    lines = [
        "# Generated from the instructions in the GitHub repository on every release; changes here are overwritten.",
        "name: Multilingual GSM-Symbolic",
        "description: >",
        "  Multilingual symbolic math benchmark derived from GSM8K. Evaluating on both the",
        "  original concrete problems and symbolically-generated variants lets you measure",
        "  how much a model relies on memorisation vs. genuine arithmetic reasoning.",
        "evaluation_framework: inspect-ai",
        "",
        "tasks:",
    ]
    for lang in languages:
        # inspect formats the template with the question as {prompt}, so literal braces are escaped
        template = lang.instruction.replace("{", "{{").replace("}", "}}") + "\n\n{prompt}"
        for kind in ("original", "synthetic"):
            lines += [
                f"  - id: {lang.task_id(kind)}",
                f"    config: {lang.code}",
                f"    split: {lang.split(kind)}",
                "    field_spec:",
                "      input: question",
                "      target: target",
                "      metadata: [answer, language, source_id]",
                "    solvers:",
                "      - name: prompt_template",
                "        args:",
                f"          template: {_yaml_str(template)}",
                "      - name: generate",
                "    scorers:",
                "      - name: math",
                "",
            ]
    return "\n".join(lines)


def _size_category(n_rows: int) -> str:
    # only counts the built splits; the English train splits do not change the category
    for upper, category in ((10_000, "1K<n<10K"), (100_000, "10K<n<100K"), (1_000_000, "100K<n<1M")):
        if n_rows < upper:
            return category
    return "1M<n<10M"


def render_metadata(languages: list[Language], n_rows: int) -> str:
    lines = ["language:", *(f"- {code}" for code in sorted({lang.code.split("_")[0] for lang in languages}))]
    lines += ["size_categories:", f"- {_size_category(n_rows)}", "configs:"]
    for lang in languages:
        lines += [f"- config_name: {lang.code}", "  data_files:"]
        for split in lang.splits:
            lines += [f"  - split: {split}", f"    path: data/{lang.code}/{split}-*.parquet"]
    return "\n".join(lines)


def render_version(version: str, languages: list[Language]) -> str:
    n_validated = sum(lang.validated for lang in languages)
    return "\n".join(
        [
            f"[![GitHub](https://img.shields.io/badge/GitHub-multilingual--gsm--symbolic-blue?logo=github)]({GITHUB_URL})",
            f"[![Version](https://img.shields.io/badge/version-v{version}-green)]"
            f"(https://huggingface.co/datasets/{REPO_ID}/tree/v{version})",
            "",
            "| | |",
            "|---|---|",
            f"| **Version** | [`v{version}`](https://huggingface.co/datasets/{REPO_ID}/tree/v{version}) |",
            f"| **Released** | {datetime.now(UTC):%Y-%m-%d} |",
            f"| **Generated from** | [`multilingual-gsm-symbolic` v{version}]({GITHUB_URL}/releases/tag/v{version}) |",
            f'| **Load this version** | `load_dataset("{REPO_ID}", name="eng", revision="v{version}")` |',
            f"| **Languages** | {len(languages)} ({n_validated} validated, {len(languages) - n_validated} unvalidated) |",
        ]
    )


def render_languages(languages: list[Language], validation: list[LanguageValidation]) -> str:
    validated = [lang for lang in languages if lang.validated]
    unvalidated = [lang for lang in languages if not lang.validated]
    validated_codes = {lang.code for lang in validated}
    table = render_language_tables([v for v in validation if v.language in validated_codes], exclude=frozenset())
    instructions = ", ".join(f"`{lang.code}`" for lang in languages if lang.instruction_validated) or "none"
    return "\n".join(
        [
            f"The dataset contains {len(validated)} validated languages, where every template has been "
            "computationally validated and checked by native speakers:",
            "",
            table,
            "",
            f"The remaining {len(unvalidated)} languages are machine translated and only computationally validated. "
            "Their splits are suffixed with `_unvalidated`: "
            + ", ".join(f"`{lang.code}`" for lang in unvalidated)
            + ".",
            "",
            f"The evaluation instructions (see `eval.yaml`) have been checked by native speakers for: {instructions}.",
        ]
    )


def update_card(card: str, sections: dict[str, str]) -> str:
    """Replace the content between each pair of markers, keeping everything else."""
    for name, content in sections.items():
        start, end = CARD_MARKERS[name]
        if card.count(start) != 1 or card.count(end) != 1:
            raise ValueError(f"The dataset card must contain the {name} markers exactly once: {start} ... {end}")
        before, rest = card.split(start)
        _, after = rest.split(end)
        card = f"{before}{start}\n{content}\n{end}{after}"
    return card


def card_sections(
    languages: list[Language], validation: list[LanguageValidation], n_rows: int, version: str
) -> dict[str, str]:
    return {
        "metadata": render_metadata(languages, n_rows),
        "version": render_version(version, languages),
        "languages": render_languages(languages, validation),
    }


def push(out: Path, sections: dict[str, str], version: str) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    card = Path(hf_hub_download(REPO_ID, "README.md", repo_type="dataset")).read_text(encoding="utf-8")
    (out / "README.md").write_text(update_card(card, sections), encoding="utf-8")

    api = HfApi()
    commit = api.upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=out,
        commit_message=f"Release v{version}",
        commit_description=f"Generated from {GITHUB_URL}/releases/tag/v{version}",
        # files uploaded in the same commit are not deleted
        delete_patterns=DELETE_PATTERNS,
    )
    api.create_tag(REPO_ID, repo_type="dataset", tag=f"v{version}", revision=commit.oid, exist_ok=True)
    log.info(f"Pushed {commit.commit_url} and tagged it v{version}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "push"])
    parser.add_argument("--out", type=Path, required=True, help="Output folder")
    args = parser.parse_args()

    validation = collect_language_validation(_DATA_ROOT)
    languages = collect_languages(validation)
    stale, orphaned = stale_and_orphaned([lang.code for lang in languages])
    if stale or orphaned:
        sys.exit(
            f"dataset/ is not up to date ({len(stale)} missing or outdated, {len(orphaned)} orphaned files). "
            "Run `uv run src/scripts/generate_dataset.py` first."
        )

    n_rows = build(languages, args.out)
    log.info(f"Built {n_rows} rows for {len(languages)} languages in {args.out}")
    if args.command == "push":
        version = multilingual_gsm_symbolic.__version__
        if "dev" in version or "+" in version:
            sys.exit(f"Only a released version can be pushed, got {version}. Check out a release tag.")
        push(args.out, card_sections(languages, validation, n_rows, version), version)


if __name__ == "__main__":
    main()
