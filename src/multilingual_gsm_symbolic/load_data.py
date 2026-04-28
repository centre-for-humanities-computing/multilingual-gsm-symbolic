import json
import logging
import tomllib
from pathlib import Path

from pydantic import BaseModel

from multilingual_gsm_symbolic.gsm_parser import AnnotatedQuestion

_DATA_ROOT = Path(__file__).parent / "data" / "templates"

logger = logging.getLogger(__name__)


def _active_template_files(lang_dir: Path) -> list[Path]:
    files = []
    for f in sorted((lang_dir / "symbolic").glob("*.toml")):
        with f.open("rb") as fp:
            data = tomllib.load(fp)
        if not data.get("ignore"):
            files.append(f)
    return files


def available_languages() -> dict[str, dict]:
    """Return the available languages and their sample counts.

    Returns:
        Mapping of language code to metadata (e.g., {'dan': {'number of samples': 100}}).
    """
    return {
        lang.name: {"number of samples": len(_active_template_files(lang))}
        for lang in sorted(_DATA_ROOT.iterdir())
        if lang.is_dir() and (lang / "symbolic").exists() and not (lang / "ignore").exists()
    }


def load_replacements(language: str = "eng") -> dict:
    """Load language-specific named values used inside templates.

    Args:
        language: Language code, e.g. "eng" (default).

    Returns:
        Mapping of replacement name → value list.
    """
    replacement_path = _DATA_ROOT / language / "replacements.json"
    with replacement_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_data(language: str = "eng", directory: str | Path | None = None) -> list[AnnotatedQuestion]:
    """Load symbolic templates.

    Args:
        language: Language code, e.g. "eng" (default).
        directory: Override the bundled data; load templates from this path instead.

    Returns:
        The loaded templates as AnnotatedQuestion objects.
    """
    if directory is not None:
        lang_dir = Path(directory).parent if Path(directory).name == "symbolic" else Path(directory)
        template_files = (
            _active_template_files(lang_dir)
            if (lang_dir / "symbolic").exists()
            else list(Path(directory).glob("*.toml"))
        )
    else:
        template_files = _active_template_files(_DATA_ROOT / language)
    return [AnnotatedQuestion.from_toml(f) for f in template_files]


class GSMProblem(BaseModel):
    """Dataclass for a concrete problem from the GSM dataset.

    Attributes:
        question: The concrete question text.
        answer: The concrete answer text.
        id_orig: Index of the original template.
        filepath: Path to the source JSON file.
    """

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
    """Load the bundled concrete problems for a given language.

    Args:
        language: Language code, e.g. "eng" (default).
        directory: Override the bundled data directory.

    Returns:
        The loaded concrete problems as GSMProblem objects.
    """
    if directory is not None:
        template_files = list(Path(directory).glob("*.toml"))
    else:
        langs = available_languages()
        if language not in langs:
            available = ", ".join(f"'{k}'" for k in langs)
            raise ValueError(f"Unknown language '{language}'. Available languages: {available}.")
        template_files = _active_template_files(_DATA_ROOT / language)
    return [_parse_json_file(f) for f in template_files]
