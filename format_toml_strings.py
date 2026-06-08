#!/usr/bin/env python3

import argparse
from pathlib import Path
import tomllib
import tomlkit


def convert_value(value):
    """Recursively convert multiline strings to TOML multiline strings."""
    if isinstance(value, str):
        if "\n" in value:
            return tomlkit.string(value, multiline=True)
        return value

    if isinstance(value, dict):
        table = tomlkit.table()
        for k, v in value.items():
            table[k] = convert_value(v)
        return table

    if isinstance(value, list):
        return [convert_value(v) for v in value]

    return value


def rewrite_toml(path: Path):
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)

        doc = tomlkit.document()

        for k, v in data.items():
            doc[k] = convert_value(v)

        with open(path, "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(doc))

        print(f"✓ {path}")

    except Exception as e:
        print(f"✗ {path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite TOML files using multiline strings for readability."
    )
    parser.add_argument(
        "directory",
        help="Directory containing TOML files",
    )

    args = parser.parse_args()

    root = Path(args.directory)

    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    files = sorted(root.rglob("*.toml"))

    print(f"Found {len(files)} TOML files")

    for path in files:
        rewrite_toml(path)


if __name__ == "__main__":
    main()