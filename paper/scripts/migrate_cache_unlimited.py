#!/usr/bin/env python3
"""Rewrite existing Inspect generation-cache entries to never expire."""

from __future__ import annotations

import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


CACHE_ROOT = Path(__file__).resolve().parents[2] / "cache" / "generate"


def migrate_tree(root: Path) -> tuple[int, int, int, int]:
    scanned = migrated = skipped = failed = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        scanned += 1
        try:
            with path.open("rb") as source:
                expiry, output = pickle.load(source)
            if expiry is None:
                skipped += 1
                continue

            descriptor, temporary_name = tempfile.mkstemp(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
            )
            try:
                with os.fdopen(descriptor, "wb") as destination:
                    pickle.dump((None, output), destination)
                    destination.flush()
                    os.fsync(destination.fileno())
                os.chmod(temporary_name, path.stat().st_mode)
                os.replace(temporary_name, path)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
            migrated += 1
        except Exception as error:
            failed += 1
            print(f"WARNING: could not migrate {path}: {error}", flush=True)

    return scanned, migrated, skipped, failed


def main() -> int:
    roots = [
        model
        for provider in CACHE_ROOT.iterdir()
        if provider.is_dir()
        for organization in provider.iterdir()
        if organization.is_dir()
        for model in organization.iterdir()
        if model.is_dir()
    ]
    totals = [0, 0, 0, 0]
    with ProcessPoolExecutor(max_workers=min(64, len(roots) or 1)) as executor:
        futures = {executor.submit(migrate_tree, root): root for root in roots}
        for future in as_completed(futures):
            root = futures[future]
            try:
                result = future.result()
            except Exception as error:
                totals[3] += 1
                print(f"ERROR: worker failed for {root}: {error}", flush=True)
                continue
            totals = [left + right for left, right in zip(totals, result, strict=True)]
            print(
                f"completed={root}: scanned={totals[0]} migrated={totals[1]} "
                f"already_unlimited={totals[2]} failed={totals[3]}",
                flush=True,
            )

    scanned, migrated, skipped, failed = totals
    print(
        f"finished: scanned={scanned} migrated={migrated} "
        f"already_unlimited={skipped} failed={failed}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
