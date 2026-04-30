#!/usr/bin/env python
# /// script
# dependencies = ["tomli-w"]
# ///
"""Convert Russian translated template files from JSON to TOML.

By default this converts:
    src/multilingual_gsm_symbolic/data/templates/rus/symbolic/*.json
into adjacent .toml files.

Usage:
    uv run src/scripts/convert_rus_json_to_toml.py
    uv run src/scripts/convert_rus_json_to_toml.py --delete-json
"""

import argparse
import json
from pathlib import Path

import tomli_w


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Russian template JSON files to TOML.")
    parser.add_argument(
        "--dir",
        default="src/multilingual_gsm_symbolic/data/templates/rus/symbolic",
        help="Directory containing .json template files",
    )
    parser.add_argument(
        "--delete-json",
        action="store_true",
        help="Delete the source .json file after successful conversion",
    )
    args = parser.parse_args()

    directory = Path(args.dir)
    if not directory.exists():
        raise SystemExit(f"Directory not found: {directory}")

    json_files = sorted(directory.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {directory}")
        return

    converted = 0
    for json_file in json_files:
        with json_file.open(encoding="utf-8") as f:
            data = json.load(f)

        toml_file = json_file.with_suffix(".toml")
        with toml_file.open("wb") as f:
            f.write(tomli_w.dumps(data).encode("utf-8"))

        if args.delete_json:
            json_file.unlink()

        converted += 1
        print(f"Converted {json_file.name} -> {toml_file.name}")

    print(f"Done. Converted {converted} file(s).")


if __name__ == "__main__":
    main()
