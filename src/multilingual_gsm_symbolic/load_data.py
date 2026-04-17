import json
import logging
from pathlib import Path

from pydantic import BaseModel

from multilingual_gsm_symbolic.gsm_parser import AnnotatedQuestion

_DATA_ROOT = Path(__file__).parent / "data" / "templates"

logger = logging.getLogger(__name__)


def available_languages() -> dict[str, dict]:
    return {
        lang.name: {"number of samples": len(list((lang / "symbolic").glob("*.json")))}
        for lang in sorted(_DATA_ROOT.iterdir())
        if lang.is_dir() and (lang / "symbolic").exists()
    }


def load_replacements(language: str = "eng") -> dict:
    replacement_path = _DATA_ROOT / language / "replacements.json"
    with replacement_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_data(language: str = "eng", directory: str | Path | None = None) -> list[AnnotatedQuestion]:
    if directory is not None:
        template_path = Path(directory)
    else:
        template_path = _DATA_ROOT / language / "symbolic"
    template_files = list(template_path.glob("*.json"))
    return [AnnotatedQuestion.from_json(template_file) for template_file in template_files]


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
    dir_path = Path(directory) if directory is not None else _DATA_ROOT / language / "symbolic"
    return [_parse_json_file(f) for f in dir_path.glob("*.json")]
