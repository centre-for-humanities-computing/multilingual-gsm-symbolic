from multilingual_gsm_symbolic.templates import AnnotatedQuestion
from src.multilingual_gsm_symbolic.load_data import Path, _DATA_ROOT, tomllib

def _active_template_files(lang_dir: Path, filename) -> list[Path]:
    files = []
    for f in sorted((lang_dir / "symbolic").glob("*.toml")):
        if f.name == filename:
            with f.open("rb") as fp:
                data = tomllib.load(fp)
            if not data.get("ignore"):
                files.append(f)
    return files


def load_data(language: str = "eng", directory: str | Path | None = None, filename=None):
    """Load symbolic templates.

    Args:
        language: Language code, e.g. "eng" (default).
        directory: Override the bundled data; load templates from this path instead.

    Returns:
        The loaded templates as AnnotatedQuestion objects.
    """

    template_files = _active_template_files(_DATA_ROOT / language, filename)
    return [AnnotatedQuestion.from_toml(f) for f in template_files]

for filename in ['0043.toml']:
# for i in range(14):
#     filename = "0000" + str(i)
#     filename = filename[-4:] + ".toml"
    templates = load_data("ukr", filename =filename)
    print(filename)

    # Test generating a question
    try:
        questions = templates[0].generate_questions(n=1)
        print(f'Success! Generated {len(questions)} question(s)')
        if questions:
            q = questions[0]
            print()
            print('Generated question:')
            print(q.question)
            print()
            print('Generated answer:')
            print(q.answer)
    except Exception as e:
        print(f'Error: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()