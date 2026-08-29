import sys
from pathlib import Path

from multilingual_gsm_symbolic.load_data import load_replacements
from multilingual_gsm_symbolic.templates import AnnotatedQuestion
from multilingual_gsm_symbolic.validation import validate_template_against_pytest_checks

repl = load_replacements("eng")
bad = []
files = sorted(Path("src/multilingual_gsm_symbolic/data/templates/eng/train").glob("*.toml"))
for i, p in enumerate(files):
    try:
        t = AnnotatedQuestion.from_toml(p)
        if t.creation.startswith("machine-generated"):
            validate_template_against_pytest_checks(t, repl, source=p.name)
            t.generate_questions(n=5, replacements=repl, verbose=False)
    except Exception as e:
        bad.append((p.name, str(e)[:200]))
    if (i + 1) % 25 == 0:
        print(f"audit progress {i + 1}/{len(files)}", flush=True)
print(f"AUDIT DONE: checked {len(files)} files, bad: {len(bad)}")
for b in bad:
    print("BAD", b[0], b[1])
