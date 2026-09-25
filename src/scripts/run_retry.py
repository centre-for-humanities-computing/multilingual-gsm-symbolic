import csv
import getpass
import os
import runpy
import sys
import time
from pathlib import Path

script = r"C:\g53-handoff\src\scripts\retry_glm53_saturation.py"
results_dir = Path(__file__).resolve().parents[2] / "paper" / "artifacts" / "glm53-hf-pr16"
os.environ["SDU_API_KEY"] = getpass.getpass("SDU API key: ")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
args = [
    script,
    "--eval-dir", str(results_dir / "logs"),
    "--output-dir", str(results_dir),
    "--max-connections", "4",
    "--max-tasks", "4",
    "--timeout", "180",
]
summary = results_dir / "summary.csv"
last_pending = None
stalled_passes = 0
pass_number = 0
try:
    while True:
        pass_number += 1
        print(f"Starting retry pass {pass_number}", flush=True)
        sys.argv = args[:]
        runpy.run_path(script, run_name="__main__")
        with summary.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        pending = sum(int(row["expected"]) - int(row["scored"]) for row in rows)
        print(f"Retry pass {pass_number} finished; {pending} samples remain unscored.", flush=True)
        if pending == 0:
            break
        stalled_passes = stalled_passes + 1 if last_pending is not None and pending >= last_pending else 0
        if stalled_passes >= 3:
            raise RuntimeError("Three consecutive retry passes made no scoring progress")
        last_pending = pending
        delay = min(600, 60 * 2**stalled_passes)
        print(f"Waiting {delay} seconds before the next retry pass.", flush=True)
        time.sleep(delay)
finally:
    os.environ.pop("SDU_API_KEY", None)
