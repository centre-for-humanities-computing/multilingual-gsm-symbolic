# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "multilingual-gsm-symbolic>=0.3.1",
#   "datasets>=3.0.0",
#   "huggingface_hub>=0.24.0",
# ]
# ///
"""Upload multilingual-gsm-symbolic dataset to HuggingFace Hub.

Produces two splits:
  - original:   the 100 concrete GSM problems per language (200 rows total)
  - synthetic:  20 generated variants per template per language

Checkpoints each completed template to .cache/ so the script can be safely
interrupted and restarted without recomputing already-finished templates.

Usage:
    uv run src/scripts/upload_to_huggingface.py
"""

import json
import logging
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict

from multilingual_gsm_symbolic import available_languages, load_data, load_gsm, load_replacements

HF_REPO_ID = "danish-foundation-models/multilingual-gsm-symbolic"
N_SYNTHETIC_PER_TEMPLATE = 20
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

LOG_FILE = Path(__file__).with_suffix(".py.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE),
    ],
)
logging.getLogger("multilingual_gsm_symbolic").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def _checkpoint_path(lang: str) -> Path:
    return CACHE_DIR / f"synthetic_{lang}.jsonl"


def _load_checkpoint(lang: str) -> tuple[list[dict], set[int]]:
    path = _checkpoint_path(lang)
    if not path.exists():
        return [], set()
    raw = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [
        {
            "question": r["question"],
            "answer": r["answer"],
            "target": r.get("target", _extract_target(r["answer"])),
            "language": r["language"],
            "source_id": r.get("source_id", r.get("id_orig")),
        }
        for r in raw
    ]
    done = {r.get("id_shuffled", r.get("source_id")) for r in raw}
    return rows, done


def _append_checkpoint(lang: str, rows: list[dict]) -> None:
    with _checkpoint_path(lang).open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_target(answer: str) -> str:
    return answer.split("####")[-1].strip()


def build_original_rows() -> list[dict]:
    rows = []
    for lang in available_languages():
        for problem in load_gsm(lang):
            rows.append(
                {
                    "question": problem.question,
                    "answer": problem.answer,
                    "target": _extract_target(problem.answer),
                    "language": lang,
                    "source_id": problem.id_orig,
                }
            )
    return rows


def build_synthetic_rows() -> list[dict]:
    all_rows = []
    languages = list(available_languages())
    for lang in languages:
        existing_rows, done = _load_checkpoint(lang)
        if existing_rows:
            log.info(f"[{lang}] Resuming — {len(done)} templates already done ({len(existing_rows)} rows loaded)")
        all_rows.extend(existing_rows)

        replacements = load_replacements(lang)
        templates = load_data(lang)
        remaining = [t for t in templates if t.id_shuffled not in done]
        log.info(f"[{lang}] {len(remaining)}/{len(templates)} templates to generate")

        for i, template in enumerate(remaining, 1):
            questions = template.generate_questions(
                n=N_SYNTHETIC_PER_TEMPLATE,
                replacements=replacements,
            )
            rows = [
                {
                    "question": q.question,
                    "answer": q.answer,
                    "target": _extract_target(q.answer),
                    "language": lang,
                    "source_id": q.id_orig,
                }
                for q in questions
            ]
            _append_checkpoint(lang, rows)
            all_rows.extend(rows)
            log.info(
                f"[{lang}] {i}/{len(remaining)} templates done (id_shuffled={template.id_shuffled}, {len(questions)} questions)"
            )

    return all_rows


def main() -> None:
    log.info("Starting upload script")

    log.info("Building original split...")
    original_rows = build_original_rows()
    log.info(f"Original split: {len(original_rows)} rows")

    log.info(f"Building synthetic split (n={N_SYNTHETIC_PER_TEMPLATE} per template)...")
    synthetic_rows = build_synthetic_rows()
    log.info(f"Synthetic split: {len(synthetic_rows)} rows")

    def filter_lang(rows: list[dict], lang: str) -> list[dict]:
        return [r for r in rows if r["language"] == lang]

    dataset = DatasetDict(
        {
            "original": Dataset.from_list(original_rows),
            "synthetic": Dataset.from_list(synthetic_rows),
            "original_eng": Dataset.from_list(filter_lang(original_rows, "eng")),
            "original_dan": Dataset.from_list(filter_lang(original_rows, "dan")),
            "synthetic_eng": Dataset.from_list(filter_lang(synthetic_rows, "eng")),
            "synthetic_dan": Dataset.from_list(filter_lang(synthetic_rows, "dan")),
        }
    )
    log.info(str(dataset))

    log.info(f"Pushing to {HF_REPO_ID} ...")
    dataset.push_to_hub(HF_REPO_ID, private=False)
    log.info("Done.")


if __name__ == "__main__":
    main()
