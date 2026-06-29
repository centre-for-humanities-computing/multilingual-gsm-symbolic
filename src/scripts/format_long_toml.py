# /// script
# dependencies = []
# ///
"""Reformat template TOML files into the readable "long" style used by the Danish set.

`tomli_w` serialises every string on a single line with escaped ``\\n``. The hand-written
Danish templates instead use multi-line basic strings (``\"\"\" ... \"\"\"``) with real
newlines, and a blank line between every top-level key. This script rewrites the target
language's templates into that same style **without changing any parsed values** (it
re-parses the result and asserts equality).

Usage:
    uv run src/scripts/format_long_toml.py --lang fao
    uv run src/scripts/format_long_toml.py --lang fao --subfolder symbolic
"""

import argparse
import tomllib
from pathlib import Path

_DATA_ROOT = Path("src/multilingual_gsm_symbolic/data/templates")


def _dump_single(value: str) -> str:
    """Serialise a string as a single-line TOML basic string."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    return f'"{escaped}"'


def _dump_multiline(value: str) -> str:
    """Serialise a string as a multi-line TOML basic string.

    A newline immediately following the opening delimiter is trimmed by TOML readers,
    so we prepend one for readability; the value is reproduced exactly on parse.
    Backslashes are escaped (basic strings interpret them); literal double quotes are
    fine as long as they don't form the ``\"\"\"`` delimiter.
    """
    escaped = value.replace("\\", "\\\\")
    return f'"""\n{escaped}"""'


def _dump_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return _dump_multiline(value) if "\n" in value else _dump_single(value)
    raise TypeError(f"Unsupported value type for long-format dump: {type(value)!r}")


def format_long(data: dict) -> str:
    """Render a flat template dict to the long-style TOML text (blank line between keys)."""
    blocks = [f"{key} = {_dump_value(value)}" for key, value in data.items()]
    return "\n\n".join(blocks) + "\n"


def reformat_file(path: Path) -> None:
    with path.open("rb") as f:
        original = tomllib.load(f)

    text = format_long(original)

    # Safety: the reformatted text must parse back to identical values.
    roundtrip = tomllib.loads(text)
    if roundtrip != original:
        diffs = [k for k in original if original.get(k) != roundtrip.get(k)]
        raise ValueError(f"{path.name}: round-trip mismatch on keys {diffs}")

    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reformat template TOML into long multi-line style.")
    parser.add_argument("--lang", required=True, help="Language code, e.g. fao")
    parser.add_argument("--subfolder", default="symbolic", help="Template subfolder (default: symbolic)")
    args = parser.parse_args()

    target = _DATA_ROOT / args.lang / args.subfolder
    if not target.exists():
        raise SystemExit(f"Directory not found: {target}")

    files = sorted(target.glob("*.toml"))
    for path in files:
        reformat_file(path)
        print(f"formatted {path}")

    print(f"\nReformatted {len(files)} files in {target}")


if __name__ == "__main__":
    main()
