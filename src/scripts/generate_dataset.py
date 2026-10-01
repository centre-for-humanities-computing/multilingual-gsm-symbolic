"""Generate the dataset from the templates into human-readable files in ``dataset/``.

Each active template ``templates/<lang>/symbolic/<id>.toml`` gets a file ``dataset/<lang>/<id>.json`` with the
original question and its synthetic variants. A file is only regenerated when it is missing or its template has
changed (based on a hash of the template file). Changes elsewhere, e.g. to ``replacements.json`` or the generation
code, are not detected: delete the affected files (or all of ``dataset/``) and run this again to regenerate them.

Usage:
    uv run src/scripts/generate_dataset.py                  # all languages
    uv run src/scripts/generate_dataset.py --lang dan eng   # only these languages
"""

import argparse
import concurrent.futures
import hashlib
import json
import logging
import sys
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multilingual_gsm_symbolic import available_languages, load_replacements  # noqa: E402
from multilingual_gsm_symbolic.load_data import _DATA_ROOT, _active_template_files  # noqa: E402
from multilingual_gsm_symbolic.templates import AnnotatedQuestion  # noqa: E402

DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"
SEED = 42
N_SYNTHETIC = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("multilingual_gsm_symbolic").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def template_hash(template_path: Path) -> str:
    return hashlib.sha256(template_path.read_bytes()).hexdigest()


def output_path(template_path: Path, dataset_dir: Path = DATASET_DIR) -> Path:
    language = template_path.parents[1].name
    return dataset_dir / language / f"{template_path.stem}.json"


def _question(question: str, answer: str) -> dict:
    return {"question": question, "answer": answer, "target": answer.split("####")[-1].strip()}


def generate_template(template_path: Path, n: int = N_SYNTHETIC, seed: int = SEED) -> dict:
    """The original question and ``n`` synthetic variants of a template, seeded per template."""
    template = AnnotatedQuestion.from_toml(template_path)
    rng = Random(f"{seed}_{template.id_orig}")
    replacements = load_replacements(template_path.parents[1].name)
    questions = template.generate_questions(n=n, replacements=replacements, rng=rng, verbose=False)
    return {
        "template": template_path.relative_to(_DATA_ROOT.parents[3]).as_posix(),
        "template_hash": template_hash(template_path),
        "language": template.language,
        "source_id": template.id_orig,
        "seed": seed,
        "original": _question(template.question, template.answer),
        "synthetic": [_question(q.question, q.answer) for q in questions],
    }


def _write(template_path: Path, dataset_dir: Path) -> Path:
    path = output_path(template_path, dataset_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(generate_template(template_path), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def is_up_to_date(template_path: Path, dataset_dir: Path = DATASET_DIR) -> bool:
    path = output_path(template_path, dataset_dir)
    if not path.exists():
        return False
    return json.loads(path.read_text(encoding="utf-8"))["template_hash"] == template_hash(template_path)


def stale_and_orphaned(languages: list[str], dataset_dir: Path = DATASET_DIR) -> tuple[list[Path], list[Path]]:
    """The templates whose files are missing or outdated, and the files whose template is gone or ignored."""
    stale, orphaned = [], []
    for language in languages:
        templates = _active_template_files(_DATA_ROOT / language)
        stale += [path for path in templates if not is_up_to_date(path, dataset_dir)]
        expected = {output_path(path, dataset_dir) for path in templates}
        orphaned += [path for path in sorted((dataset_dir / language).glob("*.json")) if path not in expected]
    return stale, orphaned


def generate(languages: list[str], dataset_dir: Path = DATASET_DIR, workers: int | None = None) -> list[Path]:
    """Generate the missing and outdated files; returns the templates that failed."""
    stale, orphaned = stale_and_orphaned(languages, dataset_dir)
    for path in orphaned:
        log.info(f"Removing {path.relative_to(dataset_dir)} (its template no longer exists or is ignored)")
        path.unlink()
    log.info(f"{len(stale)} templates to generate")
    failed = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_write, path, dataset_dir): path for path in stale}
        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            template_path = futures[future]
            try:
                log.info(f"[{i}/{len(stale)}] {future.result().relative_to(dataset_dir)}")
            except Exception as error:
                log.error(f"[{i}/{len(stale)}] {template_path} failed: {error!r}")
                failed.append(template_path)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", nargs="+", help="Languages to generate (default: all)")
    parser.add_argument("--workers", type=int, help="Worker processes (default: number of CPUs)")
    args = parser.parse_args()
    languages = list(available_languages())
    unknown = set(args.lang or ()) - set(languages)
    if unknown:
        parser.error(f"unknown language(s): {', '.join(sorted(unknown))}")
    failed = generate(args.lang or languages, workers=args.workers)
    if failed:
        sys.exit(f"{len(failed)} templates failed:\n" + "\n".join(str(path) for path in failed))


if __name__ == "__main__":
    main()
