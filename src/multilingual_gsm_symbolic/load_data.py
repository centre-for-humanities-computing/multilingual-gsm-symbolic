import json
import logging
from pathlib import Path

from pydantic import BaseModel

from multilingual_gsm_symbolic.gsm_parser import AnnotatedQuestion

_DATA_ROOT = Path(__file__).parent / "data" / "templates"

logger = logging.getLogger(__name__)


def _active_template_files(lang_dir: Path) -> list[Path]:
    files = []
    for f in (lang_dir / "symbolic").glob("*.json"):
        with f.open(encoding="utf-8") as fp:
            data = json.load(fp)
        if not data.get("ignore"):
            files.append(f)
    return files


def available_languages() -> dict[str, dict]:
    return {
        lang.name: {"number of samples": len(_active_template_files(lang))}
        for lang in sorted(_DATA_ROOT.iterdir())
        if lang.is_dir() and (lang / "symbolic").exists() and not (lang / "ignore").exists()
    }


def load_replacements(language: str = "eng") -> dict:
    replacement_path = _DATA_ROOT / language / "replacements.json"
    with replacement_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_data(language: str = "eng", directory: str | Path | None = None) -> list[AnnotatedQuestion]:
    if directory is not None:
        lang_dir = Path(directory).parent if Path(directory).name == "symbolic" else Path(directory)
        template_files = _active_template_files(lang_dir) if (lang_dir / "symbolic").exists() else list(Path(directory).glob("*.json"))
    else:
        template_files = _active_template_files(_DATA_ROOT / language)
    return [AnnotatedQuestion.from_json(f) for f in template_files]


class GSMProblem(BaseModel):
    question: str
    answer: str
    id_orig: int
    filepath: Path


def _parse_json_file(filepath: str | Path) -> GSMProblem:
    filepath = Path(filepath)
    with open(filepath, encoding="utf-8") as file:
        content = json.load(file)

    return GSMProblem(
        question=content["question"],
        answer=content["answer"],
        id_orig=content["id_orig"],
        filepath=filepath,
    )


def load_gsm(language: str = "eng", directory: str | Path | None = None) -> list[GSMProblem]:
    if directory is not None:
        template_files = list(Path(directory).glob("*.json"))
    else:
        langs = available_languages()
        if language not in langs:
            available = ", ".join(f"'{k}'" for k in langs)
            raise ValueError(f"Unknown language '{language}'. Available languages: {available}.")
        template_files = _active_template_files(_DATA_ROOT / language)
    return [_parse_json_file(f) for f in template_files]
