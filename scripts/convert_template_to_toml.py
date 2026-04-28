"""Convert a single JSON template file to TOML, keeping the same keys.

Usage:
    python scripts/convert_template_to_toml.py <path-to-template.json>

The TOML file is written next to the JSON file with the same name but .toml
extension. The original JSON file is left untouched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Fields that benefit from TOML multiline strings (they contain \n)
_MULTILINE = {"question_annotated", "answer_annotated", "question", "answer"}

# Fields whose values are plain ints (no quotes needed in TOML)
_INT_FIELDS = {"id_orig", "id_shuffled"}

# Fields whose values are booleans
_BOOL_FIELDS = {"ignore"}


def to_toml_value(key: str, value: object) -> str:
    if key in _INT_FIELDS:
        return str(int(value))  # type: ignore[arg-type]
    if key in _BOOL_FIELDS:
        return "true" if value else "false"
    if key in _MULTILINE and isinstance(value, str) and "\n" in value:
        # TOML basic multiline string – escape only backslashes and the
        # closing triple-quote sequence (which never appears in our data).
        escaped = value.replace("\\", "\\\\")
        return f'"""\n{escaped}\n"""'
    # Everything else: plain TOML basic string
    assert isinstance(value, str)
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def convert(src: Path) -> Path:
    data = json.loads(src.read_text(encoding="utf-8"))
    lines: list[str] = []
    for key, value in data.items():
        lines.append(f"{key} = {to_toml_value(key, value)}")
        lines.append("")          # blank line between fields for readability
    dst = src.with_suffix(".toml")
    dst.write_text("\n".join(lines), encoding="utf-8")
    return dst


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    src = Path(sys.argv[1])
    if not src.exists():
        print(f"File not found: {src}")
        sys.exit(1)
    dst = convert(src)
    print(f"Written: {dst}")


if __name__ == "__main__":
    main()
