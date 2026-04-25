# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "datasets>=3.0.0",
#   "huggingface_hub>=0.24.0",
# ]
# ///
"""Upload multilingual-gsm-symbolic dataset to HuggingFace Hub.

Each language is pushed as a named config (subset) with two splits:
  - original:   the 100 concrete GSM problems for that language
  - synthetic:  20 generated variants per template for that language

Checkpoints each completed template to .cache/ so the script can be safely
interrupted and restarted without recomputing already-finished templates.

Usage:
    uv run src/scripts/hf_scripts/upload_to_huggingface.py
"""

import json
import logging
import sys
from pathlib import Path

# Use local source so template fixes (ignore flags, etc.) are picked up immediately
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

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
    done = {r["id_shuffled"] for r in raw if "id_shuffled" in r}
    return rows, done


def _append_checkpoint(lang: str, rows: list[dict], id_shuffled: int) -> None:
    with _checkpoint_path(lang).open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({**row, "id_shuffled": id_shuffled}, ensure_ascii=False) + "\n")


def _extract_target(answer: str) -> str:
    return answer.split("####")[-1].strip()


def build_original_rows(lang: str) -> list[dict]:
    return [
        {
            "question": problem.question,
            "answer": problem.answer,
            "target": _extract_target(problem.answer),
            "language": lang,
            "source_id": problem.id_orig,
        }
        for problem in load_gsm(lang)
    ]


def build_synthetic_rows(lang: str) -> list[dict]:
    existing_rows, done = _load_checkpoint(lang)
    if existing_rows:
        log.info(f"[{lang}] Resuming — {len(done)} templates already done ({len(existing_rows)} rows loaded)")

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
        _append_checkpoint(lang, rows, template.id_shuffled)
        existing_rows.extend(rows)
        log.info(f"[{lang}] {i}/{len(remaining)} templates done (id_shuffled={template.id_shuffled}, {len(questions)} questions)")

    return existing_rows


def main() -> None:
    log.info("Starting upload script")
    languages = list(available_languages())

    for lang in languages:
        log.info(f"[{lang}] Building splits...")
        original_rows = build_original_rows(lang)
        synthetic_rows = build_synthetic_rows(lang)

        dataset = DatasetDict({
            "original": Dataset.from_list(original_rows),
            "synthetic": Dataset.from_list(synthetic_rows),
        })
        log.info(f"[{lang}] {dataset}")
        log.info(f"[{lang}] Pushing config '{lang}' to {HF_REPO_ID} ...")
        dataset.push_to_hub(HF_REPO_ID, config_name=lang, private=False)
        log.info(f"[{lang}] Done.")

    log.info("All languages uploaded.")


if __name__ == "__main__":
    main()
